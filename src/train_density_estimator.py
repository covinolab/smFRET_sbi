import pickle
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
import time
import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as utils
from sbi.neural_nets import posterior_nn

from temporal_encoders import build_temporal_encoder
from data_loaders import H5Dataset

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

log = logging.getLogger(__name__)


def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_checkpoint(path, epoch, model, optimizer, scheduler,
                    best_val_loss, best_model_state_dict):
    """Save a resumable training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'best_model_state_dict': best_model_state_dict,
    }, path)
    log.info(f"Checkpoint saved: epoch {epoch}, best_val={best_val_loss:.4f}, path={path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load checkpoint and return (start_epoch, best_val_loss, best_model_state_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    log.info(f"Resumed from checkpoint: epoch {ckpt['epoch']}, best_val={ckpt['best_val_loss']:.4f}, path={path}")
    return ckpt['epoch'], ckpt['best_val_loss'], ckpt['best_model_state_dict']


def setup_ddp():
    """Initialize DDP if launched via torchrun. Returns True if DDP is active."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_device():
    if dist.is_initialized():
        return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="../conf/training", config_name="density_estimator")
def train(cfg: DictConfig) -> float:
    """Train the neural posterior estimator."""
    ddp = setup_ddp()
    device = get_device()

    # Load data (use absolute path from original working directory)
    orig_cwd = hydra.utils.get_original_cwd()
    data_dir = Path(orig_cwd) / cfg.data.data_dir

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

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=None,  # H5Dataset yields batches internally
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    # Calculate steps per epoch for scheduler
    steps_per_epoch = len(train_loader) // batch_size

    density_estimator = neural_posterior(theta_init, x_init)
    density_estimator.to(device)
    
    log.info(f"Total trainable parameters: {count_model_parameters(density_estimator):,}")

    # Wrap in DDP: route loss() through forward() for gradient synchronization
    if ddp:
        class _LossForward(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.wrapped = model
            def forward(self, theta, x):
                return self.wrapped.loss(theta, x)

        local_rank = int(os.environ["LOCAL_RANK"])
        ddp_loss = DDP(_LossForward(density_estimator), device_ids=[local_rank])
    else:
        ddp_loss = None

    # compile model (for faster training)
    if cfg.training.compile:
        if ddp_loss is not None:
            ddp_loss = torch.compile(ddp_loss)
        else:
            density_estimator = torch.compile(density_estimator)

    optimizer = torch.optim.AdamW(
        density_estimator.parameters(),
        weight_decay=cfg.optimizer.weight_decay,
        lr=cfg.optimizer.lr
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.scheduler.max_lr,
        epochs=cfg.training.num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=cfg.scheduler.pct_start,
        final_div_factor=cfg.scheduler.final_div_factor,
    )

    best_val_loss = float("Inf")
    best_model_state_dict = None
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
    resume_path = cfg.training.resume_from
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_absolute():
            resume_path = Path(orig_cwd) / resume_path
        if resume_path.exists():
            start_epoch, best_val_loss, best_model_state_dict = load_checkpoint(
                resume_path, density_estimator, optimizer, scheduler, device
            )
        else:
            log.warning(f"Checkpoint not found at {resume_path}, starting from scratch")

    # Time-based checkpointing
    ckpt_interval_min = cfg.training.checkpoint_interval_minutes
    last_ckpt_time = time.time()

    # TensorBoard logging (in Hydra output directory) — only on rank 0
    if is_main_process():
        writer = SummaryWriter(log_dir=hydra_output_dir / "tensorboard", flush_secs=30)
        log.info(f"TensorBoard logs: {hydra_output_dir / 'tensorboard'}")
        if ckpt_interval_min > 0:
            log.info(f"Checkpointing every {ckpt_interval_min} min")

    for epoch in range(start_epoch + 1, cfg.training.num_epochs + 1):
        epoch_start = time.time()
        
        # Training
        density_estimator.train()
        train_loss_sum = 0
        train_samples = 0
        for theta_batch, x_batch in train_loader:
            theta_batch = theta_batch.to(device, non_blocking=True)
            x_batch = x_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            if ddp_loss is not None:
                train_losses = ddp_loss(theta_batch, x_batch)
            else:
                train_losses = density_estimator.loss(
                    theta_batch, x_batch
                )

            train_loss = train_losses.mean()
            train_loss_sum += train_losses.sum().item()
            train_samples += theta_batch.size(0)
            train_loss.backward()
            if cfg.training.grad_clip is not None and cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(density_estimator.parameters(), max_norm=cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()

        # Aggregate training loss across DDP ranks
        if ddp:
            t = torch.tensor([train_loss_sum, float(train_samples)], device=device)
            dist.all_reduce(t)
            train_loss_avg = (t[0] / t[1]).item() if t[1] > 0 else 0
        else:
            train_loss_avg = train_loss_sum / train_samples if train_samples > 0 else 0

        # Validation
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

        # Aggregate validation loss across DDP ranks
        if ddp:
            t = torch.tensor([val_loss_sum, float(val_samples)], device=device)
            dist.all_reduce(t)
            val_loss = (t[0] / t[1]).item() if t[1] > 0 else 0
        else:
            val_loss = val_loss_sum / val_samples if val_samples > 0 else 0
        epoch_time = time.time() - epoch_start

        # Logging (rank 0 only)
        if is_main_process():
            grad_norms = [p.grad.detach().norm(2) for p in density_estimator.parameters() if p.grad is not None]
            total_norm = torch.norm(torch.stack(grad_norms)).cpu().item() if grad_norms else 0.0
            writer.add_scalar("Norm/Grad_Norm", total_norm, epoch)
            writer.add_scalar("Loss/train", train_loss_avg, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/gap", train_loss_avg - val_loss, epoch)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Epoch_time_seconds", epoch_time, epoch)

            # Estimate remaining training time
            elapsed_time = time.time() - epoch_start
            remaining_epochs = cfg.training.num_epochs - epoch
            estimated_remaining_time = elapsed_time * remaining_epochs
            log.info(f"Epoch {epoch}: train={train_loss_avg:.4f}, val={val_loss:.4f}, ETA: {estimated_remaining_time/60:.1f}min")
            writer.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = deepcopy(density_estimator.state_dict())

            if is_main_process():
                with open(hydra_output_dir / f"last_{cfg.artifacts.model_filename}", "wb") as f:
                    pickle.dump(density_estimator, f)

        # Time-based checkpoint (rank 0 only)
        if is_main_process() and ckpt_interval_min > 0:
            if (time.time() - last_ckpt_time) / 60.0 >= ckpt_interval_min:
                save_checkpoint(
                    hydra_output_dir / cfg.artifacts.checkpoint_filename,
                    epoch, density_estimator, optimizer, scheduler,
                    best_val_loss, best_model_state_dict,
                )
                last_ckpt_time = time.time()

    # Save and log only on rank 0
    if is_main_process():
        # Log hyperparameters
        hparams = {
            "optimizer_lr": cfg.optimizer.lr,
            "optimizer_weight_decay": cfg.optimizer.weight_decay,
            "max_lr": cfg.scheduler.max_lr,
            "batch_size": cfg.training.batch_size,
            "num_epochs": cfg.training.num_epochs,
            "embedding_name": str(getattr(cfg.embedding, "name", "cnn")),
            "embedding_cfg": OmegaConf.to_yaml(cfg.embedding),
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