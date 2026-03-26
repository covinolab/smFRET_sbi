import logging
import os
import random
import subprocess
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from gsl_spline import spline


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    best_model_state_dict,
    best_epoch,
    logger: Optional[logging.Logger] = None,
):
    """Save a resumable training checkpoint."""
    scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state_dict,
            "best_val_loss": best_val_loss,
            "best_model_state_dict": best_model_state_dict,
            "best_epoch": best_epoch,
        },
        path,
    )
    if logger is not None:
        logger.info(
            f"Checkpoint saved: epoch {epoch}, best_val={best_val_loss:.4f}, path={path}"
        )


def load_checkpoint(path, model, optimizer, scheduler, device, logger: Optional[logging.Logger] = None):
    """Load checkpoint and return (start_epoch, best_val_loss, best_model_state_dict, best_epoch)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    ckpt_scheduler_state = ckpt.get("scheduler_state_dict", None)
    if scheduler is not None and ckpt_scheduler_state is not None:
        scheduler.load_state_dict(ckpt_scheduler_state)
    elif scheduler is not None and ckpt_scheduler_state is None:
        if logger is not None:
            logger.warning("Checkpoint has no scheduler state, but scheduler is enabled in config.")
    elif scheduler is None and ckpt_scheduler_state is not None:
        if logger is not None:
            logger.info(
                "Checkpoint contains scheduler state, but scheduler is disabled in config; ignoring scheduler state."
            )
    if logger is not None:
        logger.info(
            f"Resumed from checkpoint: epoch {ckpt['epoch']}, best_val={ckpt['best_val_loss']:.4f}, path={path}"
        )
    return (
        ckpt["epoch"],
        ckpt["best_val_loss"],
        ckpt["best_model_state_dict"],
        ckpt.get("best_epoch", ckpt["epoch"]),
    )


def setup_ddp() -> bool:
    """Initialize DDP if launched via torchrun. Returns True if DDP is active."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_device() -> torch.device:
    if dist.is_initialized():
        return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def count_model_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_embedding_name(cfg_embedding) -> str:
    name = getattr(cfg_embedding, "name", None)
    if name is not None:
        return str(name)
    if hasattr(cfg_embedding, "kernel_sizes") or hasattr(cfg_embedding, "depth"):
        return "transformer"
    return "cnn"


def l2_norm(tensors) -> float:
    norms = [tensor.detach().float().norm(2) for tensor in tensors if tensor is not None]
    if not norms:
        return 0.0
    return torch.norm(torch.stack(norms), p=2).cpu().item()


def parameter_norm(parameters) -> float:
    return l2_norm([param for param in parameters if param.requires_grad])


def gradient_norm(parameters) -> float:
    return l2_norm(
        [param.grad for param in parameters if param.requires_grad and param.grad is not None]
    )


def split_trainable_parameters(model, encoder):
    encoder_param_ids = {id(param) for param in encoder.parameters() if param.requires_grad}
    encoder_params = []
    other_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in encoder_param_ids:
            encoder_params.append(param)
        else:
            other_params.append(param)
    return encoder_params, other_params


def resolve_data_dir(data_dir, orig_cwd, logger: Optional[logging.Logger] = None) -> Path:
    data_dir = Path(data_dir)
    if data_dir.is_absolute():
        return data_dir

    orig_candidate = Path(orig_cwd) / data_dir
    if orig_candidate.exists():
        return orig_candidate

    repo_candidate = Path(__file__).resolve().parents[1] / data_dir
    if repo_candidate.exists():
        if logger is not None:
            logger.info(f"Resolved relative data_dir via repo root: {repo_candidate}")
        return repo_candidate

    return orig_candidate


def chunk_num_samples(chunk_index: int, chunk_size: int, num_examples: int) -> int:
    start = chunk_index * chunk_size
    end = min(start + chunk_size, num_examples)
    return end - start


def trim_ddp_train_chunks(train_chunk_indices, chunk_size: int, num_examples: int, world_size: int):
    full_train_chunks = [
        chunk_index
        for chunk_index in train_chunk_indices
        if chunk_num_samples(chunk_index, chunk_size, num_examples) == chunk_size
    ]
    dropped_partial_chunks = len(train_chunk_indices) - len(full_train_chunks)
    usable_chunk_count = len(full_train_chunks) - (len(full_train_chunks) % world_size)
    dropped_remainder_chunks = len(full_train_chunks) - usable_chunk_count

    if usable_chunk_count <= 0:
        raise ValueError(
            "DDP training requires at least one full chunk per rank after trimming uneven chunks. "
            f"Got {len(train_chunk_indices)} candidate train chunks, {len(full_train_chunks)} full chunks, "
            f"world_size={world_size}."
        )

    return (
        full_train_chunks[:usable_chunk_count],
        dropped_partial_chunks,
        dropped_remainder_chunks,
    )


def assert_equal_train_steps(local_steps: int, device: torch.device):
    local_steps_tensor = torch.tensor([local_steps], device=device, dtype=torch.long)
    gathered_steps = [torch.zeros_like(local_steps_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_steps, local_steps_tensor)
    steps_per_rank = [int(steps.item()) for steps in gathered_steps]
    if len(set(steps_per_rank)) != 1:
        raise RuntimeError(
            "DDP ranks have different numbers of train steps per epoch, which would deadlock NCCL. "
            f"steps_per_rank={steps_per_rank}"
        )
    return steps_per_rank


def make_spline_knots(
    params,
    x_min=0.0,
    x_max=10.0,
    num_knots=10,
    max_G0=10.0,
    num_boundary_knots=3,
    boundary_condition="bound",
):
    knot_spacing = (x_max - x_min) / (num_knots - 1)
    x_knots = np.linspace(
        x_min - num_boundary_knots * knot_spacing,
        x_max + num_boundary_knots * knot_spacing,
        num_knots + 2 * num_boundary_knots,
    )
    y_knots = np.zeros(num_knots + 2 * num_boundary_knots)
    for i in range(1, num_boundary_knots + 1):
        if boundary_condition == "linear":
            y_knots[-num_boundary_knots + i] = i * (max_G0 / num_boundary_knots) + params[num_knots]
            y_knots[num_boundary_knots - i - 1] = i * (max_G0 / num_boundary_knots) + params[1]
        elif boundary_condition == "quadratic":
            y_knots[-num_boundary_knots + i - 1] = (i / num_boundary_knots) ** 2 * max_G0 + params[num_knots]
            y_knots[num_boundary_knots - i] = (i / num_boundary_knots) ** 2 * max_G0 + params[1]

    y_knots[num_boundary_knots:-num_boundary_knots] = params[1 : 1 + num_knots]
    return x_knots, y_knots


def eval_spline(
    params,
    x_eval,
    N_knots,
    x_min,
    x_max,
    boundary_condition,
    max_G0,
    spline_type="cubic",
    num_boundary_knots=4,
):
    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()

    x_knots, y_knots = make_spline_knots(
        params=params,
        x_min=x_min,
        x_max=x_max,
        num_knots=N_knots,
        max_G0=max_G0,
        boundary_condition=boundary_condition,
        num_boundary_knots=num_boundary_knots,
    )

    y = spline(x_knots, y_knots, x_eval, spline_type=spline_type)
    return x_eval, y


def eval_spline_from_config(cfg):
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
    return partial(
        eval_spline,
        N_knots=cfg.prior.num_knots,
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        boundary_condition=cfg.simulator.boundary_condition,
        max_G0=cfg.simulator.max_G0,
        spline_type=cfg.simulator.spline_type,
        num_boundary_knots=cfg.simulator.num_boundary_knots,
    )
