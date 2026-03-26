import logging
import pickle
import shutil
import time
from copy import deepcopy
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from sbi.neural_nets import posterior_nn

from data_loaders import H5Dataset
import os
from temporal_encoders import build_temporal_encoder
from utils import (
    assert_equal_train_steps as _assert_equal_train_steps,
    cleanup_ddp,
    count_model_parameters,
    get_device,
    git_commit as _git_commit,
    gradient_norm as _gradient_norm,
    is_main_process,
    load_checkpoint,
    parameter_norm as _parameter_norm,
    resolve_data_dir,
    resolve_embedding_name as _resolve_embedding_name,
    save_checkpoint,
    set_training_seed,
    setup_ddp,
    split_trainable_parameters as _split_trainable_parameters,
    trim_ddp_train_chunks as _trim_ddp_train_chunks,
)

log = logging.getLogger(__name__)


class _LossForward(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wrapped = model

    def forward(self, theta, x):
        return self.wrapped.loss(theta, x)


def _train_one_epoch(
    loss_forward,
    density_estimator,
    train_loader,
    optimizer,
    scheduler,
    device,
    grad_clip,
    ddp,
):
    density_estimator.train()
    train_loss_sum = 0
    train_samples = 0
    train_steps = 0
    clip_steps = 0
    for theta_batch, x_batch in train_loader:
        train_steps += 1
        theta_batch = theta_batch.to(device, non_blocking=True)
        x_batch = x_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        train_losses = loss_forward(theta_batch, x_batch)
        train_loss = train_losses.mean()
        train_loss_sum += train_losses.sum().item()
        train_samples += theta_batch.size(0)
        train_loss.backward()
        if grad_clip is not None and grad_clip > 0:
            grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                density_estimator.parameters(),
                max_norm=grad_clip,
            )
            if torch.isfinite(grad_norm_before_clip) and grad_norm_before_clip.item() > grad_clip:
                clip_steps += 1
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    if ddp:
        t = torch.tensor([train_loss_sum, float(train_samples)], device=device)
        dist.all_reduce(t)
        train_samples_total = t[1].item()
        train_loss_avg = (t[0] / t[1]).item() if t[1] > 0 else 0
    else:
        train_samples_total = float(train_samples)
        train_loss_avg = train_loss_sum / train_samples if train_samples > 0 else 0

    clip_fraction = clip_steps / train_steps if train_steps > 0 else 0.0
    return train_loss_avg, train_samples_total, clip_fraction


def _validate_one_epoch(density_estimator, val_loader, device, ddp):
    density_estimator.eval()
    val_loss_sum = 0
    val_samples = 0
    with torch.no_grad():
        for theta_batch, x_batch in val_loader:
            val_losses = density_estimator.loss(
                theta_batch.to(device), x_batch.to(device)
            )
            val_loss_sum += val_losses.sum().item()
            val_samples += theta_batch.size(0)

    if ddp:
        t = torch.tensor([val_loss_sum, float(val_samples)], device=device)
        dist.all_reduce(t)
        val_loss = (t[0] / t[1]).item() if t[1] > 0 else 0
    else:
        val_loss = val_loss_sum / val_samples if val_samples > 0 else 0

    return val_loss


@hydra.main(version_base=None, config_path="../conf/training", config_name="density_estimator")
def train(cfg: DictConfig) -> float:
    """Train the neural posterior estimator."""
    ddp = setup_ddp()
    device = get_device()
    training_seed = int(getattr(cfg.training, "seed", 42))
    set_training_seed(training_seed)

    # Resolve data paths relative to the launch directory first, then the repo root.
    orig_cwd = hydra.utils.get_original_cwd()
    data_dir = resolve_data_dir(cfg.data.data_dir, orig_cwd, logger=log)

    # Load HDF5 data
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    h5_path = h5_files[0]
    chunk_size = cfg.data.chunk_size
    chunk_step = cfg.data.chunk_step
    batch_size = cfg.training.batch_size

    log.info(f"HDF5 data: {h5_path}")
    log.info(f"  chunk_size={chunk_size}, chunk_step={chunk_step}")
    log.info(f"  Samples loaded per iteration: {chunk_size * chunk_step}")

    data_config = sorted(data_dir.glob("config.yaml"))
    if data_config:
        log.info(f"  Found data config: {data_config[0]}")
        hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
        (hydra_output_dir / ".hydra/data_conf.yaml").write_text(data_config[0].read_text())
    else:
        log.warning(f"  No config.yaml found in {data_dir}, proceeding without data config")

    # Get dataset info and init samples
    temp_dataset = H5Dataset(h5_path)
    num_examples = len(temp_dataset)
    theta_init, x_init = temp_dataset[:100]
    max_len = x_init.shape[1]
    probe_x = x_init[: min(8, x_init.shape[0])].clone()
    del temp_dataset

    # Build embedding network (CNN / Transformer)
    encoder = build_temporal_encoder(
        cfg_embedding=cfg.embedding,
        transition_dim=2,
        max_len=max_len
    )

    # Build normalizing flow
    neural_posterior = posterior_nn(
        model=cfg.flow.model,
        z_score_x='none',
        hidden_features=cfg.flow.hidden_features,
        num_transforms=cfg.flow.num_transforms,
        embedding_net=encoder,
        kwargs={
            "num_blocks": cfg.flow.num_blocks,
            "dropout_probability": cfg.flow.dropout_probability,
            "use_batch_norm": cfg.flow.use_batch_norm,
        }
    )

    # Split by CHUNKS (not random indices) for fast contiguous I/O
    num_chunks = (num_examples + chunk_size - 1) // chunk_size
    num_train_chunks = int((1 - cfg.training.validation_fraction) * num_chunks)
    
    # Shuffle chunk assignment, then split (fixed seed for consistent split across DDP ranks)
    g = torch.Generator().manual_seed(42)
    chunk_perm = torch.randperm(num_chunks, generator=g).tolist()
    train_chunk_indices = chunk_perm[:num_train_chunks]
    val_chunk_indices = chunk_perm[num_train_chunks:]

    # Partition chunks across DDP ranks
    if ddp:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        (
            train_chunk_indices,
            dropped_partial_train_chunks,
            dropped_remainder_train_chunks,
        ) = _trim_ddp_train_chunks(
            train_chunk_indices,
            chunk_size=chunk_size,
            num_examples=num_examples,
            world_size=world_size,
        )
        if is_main_process() and (dropped_partial_train_chunks or dropped_remainder_train_chunks):
            log.info(
                "Trimmed train chunks for even DDP sharding: dropped %d partial chunk(s) and %d extra full chunk(s).",
                dropped_partial_train_chunks,
                dropped_remainder_train_chunks,
            )
        train_chunk_indices = train_chunk_indices[rank::world_size]
        val_chunk_indices = val_chunk_indices[rank::world_size]

    log.info(f"  Split: {len(train_chunk_indices)} train chunks, {len(val_chunk_indices)} val chunks (this rank)")

    # Create train/val datasets with chunk-based splitting
    train_dataset = H5Dataset(
        h5_path,
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_step=chunk_step,
        shuffle=True,
        chunk_indices=train_chunk_indices,
        drop_last=True,
    )
    val_dataset = H5Dataset(
        h5_path,
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_step=chunk_step,
        shuffle=False,
        chunk_indices=val_chunk_indices,
    )

    # Wrap in DataLoader with workers for prefetching
    num_workers = cfg.data.num_workers
    prefetch_factor = cfg.data.prefetch_factor

    log.info(f"  Workers: {num_workers}, prefetch_factor: {prefetch_factor}")
    train_loader_generator = torch.Generator().manual_seed(training_seed)
    val_loader_generator = torch.Generator().manual_seed(training_seed + 1)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=None,  # H5Dataset yields batches internally
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        generator=train_loader_generator,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        generator=val_loader_generator,
    )

    # Calculate steps per epoch for scheduler from dataset batching logic
    steps_per_epoch = train_dataset.num_batches(num_workers=max(1, num_workers))
    if steps_per_epoch <= 0:
        raise ValueError("train_loader has zero steps per epoch; cannot train or schedule LR")
    if ddp:
        ddp_steps = _assert_equal_train_steps(steps_per_epoch, device)
        if is_main_process():
            log.info(f"  DDP train steps per rank: {ddp_steps}")

    density_estimator = neural_posterior(theta_init, x_init)
    density_estimator.to(device)

    num_trainable_parameters = count_model_parameters(density_estimator)
    encoder_params, flow_params = _split_trainable_parameters(density_estimator, encoder)

    log.info(f"Training seed: {training_seed}")
    log.info(f"Total trainable parameters: {num_trainable_parameters:,}")

    loss_forward = _LossForward(density_estimator)
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        loss_forward = DDP(loss_forward, device_ids=[local_rank])
    if cfg.training.compile:
        loss_forward = torch.compile(loss_forward)

    optimizer = torch.optim.AdamW(
        density_estimator.parameters(),
        weight_decay=cfg.optimizer.weight_decay,
        lr=cfg.optimizer.lr
    )

    scheduler_cfg = cfg.scheduler if "scheduler" in cfg and cfg.scheduler is not None else None
    if scheduler_cfg is not None:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg.scheduler.max_lr,
            epochs=cfg.training.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.scheduler.pct_start,
            final_div_factor=cfg.scheduler.final_div_factor,
        )
        log.info("Scheduler enabled from config: OneCycleLR")
    else:
        scheduler = None
        log.info(f"Scheduler not configured; using constant learning rate: {cfg.optimizer.lr}")

    best_val_loss = float("Inf")
    best_model_state_dict = None
    best_epoch = 0
    start_epoch = 0

    # Get Hydra output directory
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)

    if is_main_process():
        log.info("=" * 60)
        log.info(f"git commit : {_git_commit(Path(orig_cwd))}")
        log.info(f"run dir    : {hydra_output_dir.resolve()}")
        log.info(f"data dir   : {data_dir.resolve()}")
        log.info(f"config     : {HydraConfig.get().job.config_name}")
        log.info(f"overrides  : {list(HydraConfig.get().overrides.task)}")
        log.info("=" * 60)
        data_cfg = data_dir / "config.yaml"
        if data_cfg.exists():
            shutil.copy2(data_cfg, hydra_output_dir / "data_config.yaml")
            log.info(f"Copied dataset config -> {hydra_output_dir / 'data_config.yaml'}")
        else:
            log.warning(f"No config.yaml found in {data_dir}; skipping data config copy")

    # Resume from checkpoint if configured
    resume_path = cfg.training.resume_from if "resume_from" in cfg.training else None
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = Path(orig_cwd) / resume_path
        if resume_path.exists():
            start_epoch, best_val_loss, best_model_state_dict, best_epoch = load_checkpoint(
                resume_path, density_estimator, optimizer, scheduler, device, logger=log
            )
        else:
            log.warning(f"Checkpoint not found at {resume_path}, starting from scratch")

    # Time-based checkpointing
    ckpt_interval_min = cfg.training.checkpoint_interval_minutes if "checkpoint_interval_minutes" in cfg.training else 0
    last_ckpt_time = time.time()
    grad_clip = cfg.training.grad_clip if "grad_clip" in cfg.training else None

    # TensorBoard logging (in Hydra output directory) — only on rank 0
    if is_main_process():
        writer = SummaryWriter(log_dir=hydra_output_dir / "tensorboard", flush_secs=30)
        log.info(f"TensorBoard logs: {hydra_output_dir / 'tensorboard'}")
        if ckpt_interval_min > 0:
            log.info(f"Checkpointing every {ckpt_interval_min} min")

    for epoch in range(start_epoch + 1, cfg.training.num_epochs + 1):
        epoch_start = time.time()

        train_loss_avg, train_samples_total, clip_fraction = _train_one_epoch(
            loss_forward, density_estimator, train_loader, optimizer, scheduler,
            device, grad_clip, ddp,
        )
        val_loss = _validate_one_epoch(density_estimator, val_loader, device, ddp)

        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = deepcopy(density_estimator.state_dict())
            best_epoch = epoch

            if is_main_process():
                with open(hydra_output_dir / f"last_{cfg.artifacts.model_filename}", "wb") as f:
                    pickle.dump(density_estimator, f)

        # Logging (rank 0 only)
        if is_main_process():
            total_norm = _gradient_norm(density_estimator.parameters())
            encoder_grad_norm = _gradient_norm(encoder_params)
            flow_grad_norm = _gradient_norm(flow_params)
            encoder_param_norm = _parameter_norm(encoder_params)
            flow_param_norm = _parameter_norm(flow_params)
            throughput = train_samples_total / max(epoch_time, 1e-12)
            with torch.no_grad():
                probe_embeddings = encoder(probe_x.to(device, non_blocking=True)).detach().float()
                if probe_embeddings.ndim == 1:
                    probe_embeddings = probe_embeddings.unsqueeze(-1)
                probe_mean = probe_embeddings.mean().item()
                probe_std = probe_embeddings.std(unbiased=False).item()
                probe_l2 = probe_embeddings.norm(dim=-1).mean().item()

            writer.add_scalar("Norm/Grad_Norm", total_norm, epoch)
            writer.add_scalar("Norm/Grad_Norm_Encoder", encoder_grad_norm, epoch)
            writer.add_scalar("Norm/Grad_Norm_Flow", flow_grad_norm, epoch)
            writer.add_scalar("Norm/Param_Norm_Encoder", encoder_param_norm, epoch)
            writer.add_scalar("Norm/Param_Norm_Flow", flow_param_norm, epoch)
            writer.add_scalar("Loss/train", train_loss_avg, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/gap", train_loss_avg - val_loss, epoch)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Epoch_time_seconds", epoch_time, epoch)
            writer.add_scalar("Optimization/clip_fraction", clip_fraction, epoch)
            writer.add_scalar("Embedding/probe_mean", probe_mean, epoch)
            writer.add_scalar("Embedding/probe_std", probe_std, epoch)
            writer.add_scalar("Embedding/probe_l2", probe_l2, epoch)
            writer.add_scalar("Throughput/samples_per_sec", throughput, epoch)

            remaining_epochs = cfg.training.num_epochs - epoch
            estimated_remaining_time = epoch_time * remaining_epochs
            log.info(f"Epoch {epoch}: train={train_loss_avg:.4f}, val={val_loss:.4f}, ETA: {estimated_remaining_time/60:.1f}min")
            writer.flush()

        # Time-based checkpoint (rank 0 only)
        if is_main_process() and ckpt_interval_min > 0:
            if (time.time() - last_ckpt_time) / 60.0 >= ckpt_interval_min:
                save_checkpoint(
                    hydra_output_dir / cfg.artifacts.checkpoint_filename,
                    epoch, density_estimator, optimizer, scheduler,
                    best_val_loss, best_model_state_dict, best_epoch,
                    logger=log,
                )
                last_ckpt_time = time.time()

    # Save and log only on rank 0
    if is_main_process():
        # Log hyperparameters
        hparams = {
            "optimizer_lr": cfg.optimizer.lr,
            "optimizer_weight_decay": cfg.optimizer.weight_decay,
            "max_lr": cfg.scheduler.max_lr if scheduler_cfg is not None else cfg.optimizer.lr,
            "batch_size": cfg.training.batch_size,
            "num_epochs": cfg.training.num_epochs,
            "training_seed": training_seed,
            "embedding_name": _resolve_embedding_name(cfg.embedding),
            "embedding_cfg": OmegaConf.to_yaml(cfg.embedding),
            "data_chunk_step": cfg.data.chunk_step,
            "shuffle_buffer_samples": chunk_size * chunk_step,
            "num_parameters": num_trainable_parameters,
            "hidden_features": cfg.flow.hidden_features,
            "model": cfg.flow.model,
            "num_transforms": cfg.flow.num_transforms,
            "num_blocks": cfg.flow.num_blocks,
            "dropout_probability": cfg.flow.dropout_probability,
            "use_batch_norm": cfg.flow.use_batch_norm,
        }
        writer.add_hparams(
            hparam_dict=hparams, 
            metric_dict={"hparam/best_val_loss": best_val_loss},
            run_name="."
        )
        writer.close()

        # Save best model (in Hydra output directory)
        density_estimator.load_state_dict(best_model_state_dict)
        output_path = hydra_output_dir / cfg.artifacts.model_filename
        with open(output_path, "wb") as f:
            pickle.dump(density_estimator, f)

        # Clean up intermediate files after successful completion
        for tmp_name in [f"last_{cfg.artifacts.model_filename}", cfg.artifacts.checkpoint_filename]:
            tmp_path = hydra_output_dir / tmp_name
            if tmp_path.exists():
                tmp_path.unlink()

        log.info(f"\nBest val loss: {best_val_loss:.4f}")
        log.info(f"Saved model to {output_path}")

    cleanup_ddp()
    return best_val_loss


if __name__ == "__main__":
    train()
