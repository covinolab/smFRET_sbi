import logging
import os
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
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as sbi_utils
from sbi.neural_nets import classifier_nn

from data_loaders import H5Dataset
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

def _resolve_h5_data_source(cfg_data: DictConfig, orig_cwd) -> tuple[Path, Path]:
    if hasattr(cfg_data, "data_dir"):
        data_dir = resolve_data_dir(cfg_data.data_dir, orig_cwd, logger=log)
        h5_files = sorted(data_dir.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {data_dir}")
        return data_dir, h5_files[0]

    if hasattr(cfg_data, "h5_path"):
        h5_path = resolve_data_dir(cfg_data.h5_path, orig_cwd, logger=log)
        if h5_path.is_dir():
            h5_files = sorted(h5_path.glob("*.h5"))
            if not h5_files:
                raise FileNotFoundError(f"No .h5 files found in {h5_path}")
            return h5_path, h5_files[0]
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        return h5_path.parent, h5_path

    raise KeyError("Expected either data.data_dir or data.h5_path in the training config.")


def _artifact_setting(cfg: DictConfig, name: str, default: str) -> str:
    artifacts = getattr(cfg, "artifacts", None)
    value = getattr(artifacts, name, None) if artifacts is not None else None
    return str(value) if value is not None else default


def classifier_logits(net, theta, x, num_atoms):
    batch_size = theta.shape[0]
    if batch_size < 2:
        raise ValueError("classifier loss requires batch_size >= 2")
    if num_atoms < 2:
        raise ValueError("classifier.num_atoms must be >= 2")
    if num_atoms > batch_size:
        raise ValueError(
            f"classifier.num_atoms ({num_atoms}) must be <= batch size ({batch_size}). "
            "Increase training.batch_size or lower classifier.num_atoms."
        )

    probs = torch.full(
        (batch_size, batch_size),
        1.0 / (batch_size - 1),
        device=theta.device,
    )
    probs.fill_diagonal_(0.0)
    choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
    contrasting_theta = theta[choices]
    atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
        batch_size * num_atoms, -1
    )

    x_embedded = net.embedding_net_x(x)
    repeated_x_embedded = sbi_utils.repeat_rows(x_embedded, num_atoms)
    theta_embedded = net.embedding_net_theta(atomic_theta)
    input_to_classifier = torch.cat([theta_embedded, repeated_x_embedded], dim=1)
    return net.net(input_to_classifier)


def classifier_losses(net, theta, x, num_atoms):
    logits = classifier_logits(net, theta, x, num_atoms)
    logits = logits.reshape(theta.shape[0], num_atoms)
    log_probs = logits[:, 0] - torch.logsumexp(logits, dim=-1)
    return -log_probs


class _LossForward(torch.nn.Module):
    def __init__(self, model, num_atoms):
        super().__init__()
        self.wrapped = model
        self.num_atoms = int(num_atoms)

    def forward(self, theta, x):
        return classifier_losses(self.wrapped, theta, x, self.num_atoms)


def _train_one_epoch(
    loss_forward,
    ratio_estimator,
    train_loader,
    optimizer,
    scheduler,
    device,
    grad_clip,
    ddp,
):
    ratio_estimator.train()
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
                ratio_estimator.parameters(),
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


def _validate_one_epoch(ratio_estimator, val_loader, num_atoms, device, ddp):
    ratio_estimator.eval()
    val_loss_sum = 0
    val_samples = 0
    skipped_val_batches = 0
    with torch.no_grad():
        for theta_batch, x_batch in val_loader:
            if theta_batch.size(0) < num_atoms:
                skipped_val_batches += 1
                continue
            theta_batch = theta_batch.to(device, non_blocking=True)
            x_batch = x_batch.to(device, non_blocking=True)
            val_losses = classifier_losses(ratio_estimator, theta_batch, x_batch, num_atoms)
            val_loss_sum += val_losses.sum().item()
            val_samples += theta_batch.size(0)

    if ddp:
        t = torch.tensor([val_loss_sum, float(val_samples), float(skipped_val_batches)], device=device)
        dist.all_reduce(t)
        val_loss = (t[0] / t[1]).item() if t[1] > 0 else 0
        skipped_val_batches_total = int(t[2].item())
    else:
        val_loss = val_loss_sum / val_samples if val_samples > 0 else 0
        skipped_val_batches_total = skipped_val_batches

    return val_loss, skipped_val_batches_total


@hydra.main(version_base=None, config_path="../conf/training", config_name="ratio_estimator")
def train(cfg: DictConfig) -> float:
    """Train the neural ratio estimator."""
    ddp = setup_ddp()
    device = get_device()
    training_seed = int(getattr(cfg.training, "seed", 42))
    set_training_seed(training_seed)

    orig_cwd = hydra.utils.get_original_cwd()
    data_dir, h5_path = _resolve_h5_data_source(cfg.data, orig_cwd)

    chunk_size = int(getattr(cfg.data, "chunk_size", 512))
    chunk_step = int(getattr(cfg.data, "chunk_step", 1))
    batch_size = int(cfg.training.batch_size)
    num_atoms = int(cfg.classifier.num_atoms)
    if batch_size < num_atoms:
        raise ValueError(
            f"training.batch_size ({batch_size}) must be >= classifier.num_atoms ({num_atoms}) "
            "for contrastive sampling without replacement."
        )

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

    temp_dataset = H5Dataset(h5_path)
    num_examples = len(temp_dataset)
    theta_init, x_init = temp_dataset[:100]
    max_len = x_init.shape[1]
    probe_x = x_init[: min(8, x_init.shape[0])].clone()
    del temp_dataset

    encoder = build_temporal_encoder(
        cfg_embedding=cfg.embedding,
        transition_dim=2,
        max_len=max_len,
    )

    neural_ratio = classifier_nn(
        model="resnet",
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=cfg.classifier.hidden_features,
        embedding_net_theta=nn.Identity(),
        embedding_net_x=encoder,
        **{'dropout_probability': cfg.classifier.dropout_probability, 'num_blocks': cfg.classifier.num_blocks, 'use_batch_norm': cfg.classifier.use_batch_norm},
    )

    num_chunks = (num_examples + chunk_size - 1) // chunk_size
    num_train_chunks = int((1 - cfg.training.validation_fraction) * num_chunks)

    g = torch.Generator().manual_seed(42)
    chunk_perm = torch.randperm(num_chunks, generator=g).tolist()
    train_chunk_indices = chunk_perm[:num_train_chunks]
    val_chunk_indices = chunk_perm[num_train_chunks:]

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

    num_workers = int(getattr(cfg.data, "num_workers", 0))
    prefetch_factor = int(getattr(cfg.data, "prefetch_factor", 2))

    log.info(f"  Workers: {num_workers}, prefetch_factor: {prefetch_factor}")
    train_loader_generator = torch.Generator().manual_seed(training_seed)
    val_loader_generator = torch.Generator().manual_seed(training_seed + 1)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=None,
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

    steps_per_epoch = train_dataset.num_batches(num_workers=max(1, num_workers))
    if steps_per_epoch <= 0:
        raise ValueError("train_loader has zero steps per epoch; cannot train or schedule LR")
    if ddp:
        ddp_steps = _assert_equal_train_steps(steps_per_epoch, device)
        if is_main_process():
            log.info(f"  DDP train steps per rank: {ddp_steps}")

    ratio_estimator = neural_ratio(theta_init, x_init)
    ratio_estimator.to(device)

    num_trainable_parameters = count_model_parameters(ratio_estimator)
    encoder_params, classifier_params = _split_trainable_parameters(ratio_estimator, encoder)

    log.info(f"Training seed: {training_seed}")
    log.info(f"Total trainable parameters: {num_trainable_parameters:,}")

    loss_forward = _LossForward(ratio_estimator, num_atoms)
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        loss_forward = DDP(loss_forward, device_ids=[local_rank])

    if cfg.training.compile:
        loss_forward = torch.compile(loss_forward)

    optimizer = torch.optim.AdamW(
        ratio_estimator.parameters(),
        weight_decay=cfg.optimizer.weight_decay,
        lr=cfg.optimizer.lr,
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

    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_filename = _artifact_setting(cfg, "model_filename", "ratio_estimator.pkl")
    checkpoint_filename = _artifact_setting(cfg, "checkpoint_filename", "checkpoint.pt")
    data_config_copy_filename = _artifact_setting(cfg, "data_config_copy_filename", "data_config.yaml")

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
            shutil.copy2(data_cfg, hydra_output_dir / data_config_copy_filename)
            log.info(f"Copied dataset config -> {hydra_output_dir / data_config_copy_filename}")
        else:
            log.warning(f"No config.yaml found in {data_dir}; skipping data config copy")

    resume_path = cfg.training.resume_from if "resume_from" in cfg.training else None
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = Path(orig_cwd) / resume_path
        if resume_path.exists():
            start_epoch, best_val_loss, best_model_state_dict, best_epoch = load_checkpoint(
                resume_path, ratio_estimator, optimizer, scheduler, device, logger=log
            )
        else:
            log.warning(f"Checkpoint not found at {resume_path}, starting from scratch")

    ckpt_interval_min = cfg.training.checkpoint_interval_minutes if "checkpoint_interval_minutes" in cfg.training else 0
    last_ckpt_time = time.time()
    grad_clip = cfg.training.grad_clip if "grad_clip" in cfg.training else None

    if is_main_process():
        writer = SummaryWriter(log_dir=hydra_output_dir / "tensorboard", flush_secs=30)
        log.info(f"TensorBoard logs: {hydra_output_dir / 'tensorboard'}")
        if ckpt_interval_min > 0:
            log.info(f"Checkpointing every {ckpt_interval_min} min")

    for epoch in range(start_epoch + 1, cfg.training.num_epochs + 1):
        epoch_start = time.time()

        train_loss_avg, train_samples_total, clip_fraction = _train_one_epoch(
            loss_forward, ratio_estimator, train_loader, optimizer, scheduler,
            device, grad_clip, ddp,
        )
        val_loss, skipped_val_batches_total = _validate_one_epoch(
            ratio_estimator, val_loader, num_atoms, device, ddp,
        )

        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = deepcopy(ratio_estimator.state_dict())
            best_epoch = epoch

            if is_main_process():
                with open(hydra_output_dir / f"last_{model_filename}", "wb") as f:
                    pickle.dump(ratio_estimator, f)

        if is_main_process():
            total_norm = _gradient_norm(ratio_estimator.parameters())
            encoder_grad_norm = _gradient_norm(encoder_params)
            classifier_grad_norm = _gradient_norm(classifier_params)
            encoder_param_norm = _parameter_norm(encoder_params)
            classifier_param_norm = _parameter_norm(classifier_params)
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
            writer.add_scalar("Norm/Grad_Norm_Classifier", classifier_grad_norm, epoch)
            writer.add_scalar("Norm/Param_Norm_Encoder", encoder_param_norm, epoch)
            writer.add_scalar("Norm/Param_Norm_Classifier", classifier_param_norm, epoch)
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

            if skipped_val_batches_total > 0:
                writer.add_scalar("Validation/skipped_small_batches", skipped_val_batches_total, epoch)

            remaining_epochs = cfg.training.num_epochs - epoch
            estimated_remaining_time = epoch_time * remaining_epochs
            log.info(f"Epoch {epoch}: train={train_loss_avg:.4f}, val={val_loss:.4f}, ETA: {estimated_remaining_time/60:.1f}min")
            if skipped_val_batches_total > 0:
                log.warning(
                    "Skipped %d validation batch(es) with batch_size < classifier.num_atoms=%d",
                    skipped_val_batches_total,
                    num_atoms,
                )
            writer.flush()

        if is_main_process() and ckpt_interval_min > 0:
            if (time.time() - last_ckpt_time) / 60.0 >= ckpt_interval_min:
                save_checkpoint(
                    hydra_output_dir / checkpoint_filename,
                    epoch,
                    ratio_estimator,
                    optimizer,
                    scheduler,
                    best_val_loss,
                    best_model_state_dict,
                    best_epoch,
                    logger=log,
                )
                last_ckpt_time = time.time()

    if is_main_process():
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
            "hidden_features": cfg.classifier.hidden_features,
            "num_atoms": cfg.classifier.num_atoms,
            "best_epoch": best_epoch,
            "dropout_probability": cfg.classifier.dropout_probability,
            "num_blocks": cfg.classifier.num_blocks,
            "use_batch_norm": cfg.classifier.use_batch_norm,
        }
        writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={"hparam/best_val_loss": best_val_loss},
            run_name=".",
        )
        writer.close()

        ratio_estimator.load_state_dict(best_model_state_dict)
        output_path = hydra_output_dir / model_filename
        with open(output_path, "wb") as f:
            pickle.dump(ratio_estimator, f)

        for tmp_name in [f"last_{model_filename}", checkpoint_filename]:
            tmp_path = hydra_output_dir / tmp_name
            if tmp_path.exists():
                tmp_path.unlink()

        log.info(f"\nBest val loss: {best_val_loss:.4f}")
        log.info(f"Saved model to {output_path}")

    cleanup_ddp()
    return best_val_loss


if __name__ == "__main__":
    train()
