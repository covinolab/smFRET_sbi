"""Focused smoke test for PhotonDtSSM on real exp_9 data.

Runs N_BATCHES mini-batches through the full encoder + MAF-RQS flow pipeline
(same modules used by train_density_estimator.py) and prints loss every batch.
Goal: confirm forward/backward runs end-to-end and the loss decreases.
"""
from __future__ import annotations

import sys
import time

import torch
from sbi.neural_nets import posterior_nn

from data_loaders import H5Dataset
from temporal_encoders import build_set_embedding


DATA_H5 = "/home/covino-shared/data/dingeldein/fret_sbi/sim_data/exp_9/data.h5"
N_BATCHES = 30
BATCH_SIZE = 2
CHUNK_SIZE = 8


class _EmbeddingCfg:
    """Stand-in for an OmegaConf node so we skip Hydra loading."""
    name = "photon_dt_ssm"
    hidden_dim = 32
    num_blocks = 2
    kernel_size = 3              # ignored by photon_dt_ssm; used by photon_tcn
    output_dim = 128
    bidirectional = True
    alpha_init_range_ms = (1e-3, 1e2)
    chunk_size = 64

    def __init__(self, name="photon_dt_ssm"):
        self.name = name


def main(name="photon_dt_ssm"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"\n=== encoder: {name} ===")

    ds = H5Dataset(
        DATA_H5,
        batch_size=BATCH_SIZE,
        chunk_size=CHUNK_SIZE,
        chunk_step=1,
        shuffle=False,
        chunk_indices=list(range(8)),  # 8 chunks × 8 samples = 64 samples
        drop_last=False,
    )
    theta_init, x_init = ds[:8]
    transition_dim = 4
    print(f"theta_init shape: {tuple(theta_init.shape)}, x_init shape: {tuple(x_init.shape)}")

    encoder = build_set_embedding(
        cfg_embedding=_EmbeddingCfg(name=name),
        transition_dim=transition_dim,
        max_len=x_init.shape[2],
        post_agg_hidden=None,
    )
    neural_posterior = posterior_nn(
        model="maf_rqs",
        z_score_x="none",
        hidden_features=128,
        num_transforms=3,
        embedding_net=encoder,
        kwargs={"num_blocks": 2, "dropout_probability": 0.05, "use_batch_norm": False},
    )
    flow = neural_posterior(theta_init, x_init).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=5e-4, weight_decay=1e-2)
    n_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=0, pin_memory=True
    )

    losses = []
    t_start = time.time()
    for i, (theta, x) in enumerate(loader):
        if i >= N_BATCHES:
            break
        theta = theta.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        t0 = time.time()
        loss = flow.loss(theta, x).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000
        losses.append(loss.item())
        print(f"batch {i:3d}: loss={loss.item():+.4f}  ({dt_ms:.0f} ms)", flush=True)
        if not torch.isfinite(torch.tensor(loss.item())):
            print("ABORT  non-finite loss")
            return 1

    print("---")
    elapsed = time.time() - t_start
    print(f"total: {elapsed:.1f}s, mean batch: {elapsed/len(losses)*1000:.0f} ms")
    print(f"loss first/last/min: {losses[0]:+.4f} / {losses[-1]:+.4f} / {min(losses):+.4f}")
    initial_mean = sum(losses[:5]) / 5
    final_mean = sum(losses[-5:]) / 5
    print(f"first-5 mean: {initial_mean:+.4f}, last-5 mean: {final_mean:+.4f}")
    if final_mean < initial_mean - 0.1:
        print("PASS  loss decreased meaningfully")
        return 0
    print("WARN  loss did not decrease much (final - initial = "
          f"{final_mean - initial_mean:+.4f})")
    return 0


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "photon_dt_ssm"
    sys.exit(main(name))
