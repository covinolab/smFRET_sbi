import pickle
from copy import deepcopy
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sbi import utils as utils
from sbi.neural_nets import classifier_nn

from temporal_encoders import TemporalCNN
from data_loaders import H5Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
log = logging.getLogger(__name__)

def classifier_logits(net, theta, x, num_atoms):
    batch_size = theta.shape[0]
    
    # Build contrasting theta indices without repeating x
    probs = torch.ones(batch_size, batch_size, device=theta.device) * (1 - torch.eye(batch_size, device=theta.device)) / (batch_size - 1)
    choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
    contrasting_theta = theta[choices]
    atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
        batch_size * num_atoms, -1
    )
    
    # Embed x once, then repeat the (low-dim) embedding instead of the raw input
    x_embedded = net.embedding_net_x(x)  # (batch_size, embed_dim)
    repeated_x_embedded = utils.repeat_rows(x_embedded, num_atoms)
    # Embed theta (usually Identity, but be safe)
    theta_embedded = net.embedding_net_theta(atomic_theta)
    # Concatenate and pass through the classifier head directly
    input_to_classifier = torch.cat([theta_embedded, repeated_x_embedded], dim=1)
    return net.net(input_to_classifier)


@hydra.main(version_base=None, config_path="conf/training", config_name="ratio_estimator")
def train(cfg: DictConfig) -> float:
    """Train the ratio estimator (classifier-based)."""
    
    # Build embedding network
    cnn = TemporalCNN(
        transition_dim=2,
        dim=cfg.embedding.dim,
        dim_mults=tuple(cfg.embedding.dim_mults),
        attention=cfg.embedding.attention,
        padding_mode='reflect',
        kernel_size=cfg.embedding.kernel_size,
        output_dim=cfg.embedding.output_dim
    )

    # Build ratio estimator (classifier)
    neural_ratio = classifier_nn(
        model=cfg.classifier.model,
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=cfg.classifier.hidden_features,
        embedding_net_theta=nn.Identity(),
        embedding_net_x=cnn,
    )


    # Load data (use absolute path from original working directory)
    orig_cwd = hydra.utils.get_original_cwd()
    
    # Check if using lazy HDF5 loading (for large datasets)
    use_lazy_loading = getattr(cfg.data, 'h5_path', None) is not None
    
    if use_lazy_loading:
        # Efficient HDF5 streaming with H5Dataset
        h5_path = Path(orig_cwd) / cfg.data.h5_path
        chunk_size = getattr(cfg.data, 'chunk_size', 1000)
        chunk_step = getattr(cfg.data, 'chunk_step', 10)
        batch_size = cfg.training.batch_size
        
        log.info(f"Using H5Dataset from: {h5_path}")
        log.info(f"  chunk_size={chunk_size}, chunk_step={chunk_step}")
        log.info(f"  Samples loaded per iteration: {chunk_size * chunk_step}")
        
        # Get dataset info and init samples
        temp_dataset = H5Dataset(h5_path)
        num_examples = len(temp_dataset)
        theta_init, x_init = temp_dataset[:100]
        del temp_dataset
        
        # Split by CHUNKS (not random indices) for fast contiguous I/O
        num_chunks = (num_examples + chunk_size - 1) // chunk_size
        num_train_chunks = int((1 - cfg.training.validation_fraction) * num_chunks)
        
        # Shuffle chunk assignment, then split
        chunk_perm = torch.randperm(num_chunks).tolist()
        train_chunk_indices = chunk_perm[:num_train_chunks]
        val_chunk_indices = chunk_perm[num_train_chunks:]
        
        log.info(f"  Split: {num_train_chunks} train chunks, {len(val_chunk_indices)} val chunks")
        
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
            drop_last=True,
        )
        
        # Wrap in DataLoader with workers for prefetching
        num_workers = getattr(cfg.data, 'num_workers', 8)
        prefetch_factor = getattr(cfg.data, 'prefetch_factor', 4)
        
        log.info(f"  Workers: {num_workers}, prefetch_factor: {prefetch_factor}")
        
        # IterableDataset yields batches, so set batch_size=None
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
        steps_per_epoch = len(train_dataset) // batch_size
        
    else:
        # Traditional loading - load entire dataset into RAM
        theta = torch.load(Path(orig_cwd) / cfg.data.theta_path, weights_only=True)
        x = torch.load(Path(orig_cwd) / cfg.data.x_path, weights_only=True)
        
        num_examples = theta.size(0)
        theta_init = theta[:100, :]
        x_init = x[:100, :]
        
        dataset = data.TensorDataset(theta, x)
        
        # Train/val split
        num_training_examples = int((1 - cfg.training.validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        permuted_indices = torch.randperm(num_examples)
        train_indices = permuted_indices[:num_training_examples].tolist()
        val_indices = permuted_indices[num_training_examples:].tolist()
        
        train_loader = data.DataLoader(
            dataset,
            batch_size=min(cfg.training.batch_size, num_training_examples),
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(cfg.training.batch_size, num_validation_examples),
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices),
        )
        steps_per_epoch = len(train_loader)

    ratio_estimator = neural_ratio(theta_init, x_init)
    ratio_estimator.to(device)

    optimizer = Adam(
        ratio_estimator.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # OneCycleLR scheduler (steps_per_epoch already calculated above)
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

    # Get Hydra output directory
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)

    # TensorBoard logging (in Hydra output directory)
    writer = SummaryWriter(log_dir=hydra_output_dir / "tensorboard")
    log.info(f"TensorBoard logs: {hydra_output_dir / 'tensorboard'}")

    for epoch in range(1, cfg.training.num_epochs + 1):
        epoch_start = time.time()
        
        # Training
        ratio_estimator.train()
        train_loss_sum = 0
        train_samples_seen = 0
        for theta_batch, x_batch in train_loader:
            optimizer.zero_grad()
            batch_size = theta_batch.shape[0]
            logits = classifier_logits(
                ratio_estimator, theta_batch.to(device), x_batch.to(device), cfg.classifier.num_atoms
            )
            logits = logits.reshape(batch_size, cfg.classifier.num_atoms)
            log_probs = logits[:, 0] - torch.logsumexp(logits, dim=-1)
            train_loss = -torch.mean(log_probs)
            # Sum individual losses (not mean) for correct averaging
            train_loss_sum += -log_probs.sum().item()
            train_samples_seen += batch_size
            train_loss.backward()
            if cfg.training.grad_clip is not None and cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ratio_estimator.parameters(), max_norm=cfg.training.grad_clip)
            optimizer.step()
            scheduler.step()

        train_loss_avg = train_loss_sum / train_samples_seen

        # Validation
        ratio_estimator.eval()
        val_loss_sum = 0
        val_samples_seen = 0
        with torch.no_grad():
            for theta_batch, x_batch in val_loader:
                batch_size = theta_batch.shape[0]
                logits = classifier_logits(
                    ratio_estimator, theta_batch.to(device), x_batch.to(device), cfg.classifier.num_atoms
                )
                logits = logits.reshape(batch_size, cfg.classifier.num_atoms)
                log_probs = logits[:, 0] - torch.logsumexp(logits, dim=-1)
                val_loss_sum += -log_probs.sum().item()
                val_samples_seen += batch_size

        val_loss = val_loss_sum / val_samples_seen
        epoch_time = time.time() - epoch_start

        # Logging
        total_norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in ratio_estimator.parameters() if p.grad is not None])).cpu().item()
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = deepcopy(ratio_estimator.state_dict())

    # Log hyperparameters
    hparams = {
        "max_lr": cfg.scheduler.max_lr,
        "batch_size": cfg.training.batch_size,
        "num_epochs": cfg.training.num_epochs,
        "dim": cfg.embedding.dim,
        "dim_mults": str(list(cfg.embedding.dim_mults)),
        "attention": cfg.embedding.attention,
        "kernel_size": cfg.embedding.kernel_size,
        "output_dim": cfg.embedding.output_dim,
        "hidden_features": cfg.classifier.hidden_features,
        "model": cfg.classifier.model,
        "num_atoms": cfg.classifier.num_atoms,
    }
    writer.add_hparams(
        hparam_dict=hparams, 
        metric_dict={"hparam/best_val_loss": best_val_loss},
        run_name="."
    )
    writer.close()

    # Save best model (in Hydra output directory)
    ratio_estimator.load_state_dict(best_model_state_dict)
    output_path = hydra_output_dir / "ratio_estimator.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(ratio_estimator, f)

    log.info(f"\nBest val loss: {best_val_loss:.4f}")
    log.info(f"Saved model to {output_path}")
    
    return best_val_loss


if __name__ == "__main__":
    train()