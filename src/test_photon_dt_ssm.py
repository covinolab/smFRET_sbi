"""Tests for the Δt-aware diagonal SSM photon encoder.

Run with:
    cd smFRET_sbi/src
    pytest test_photon_dt_ssm.py -v

or as a script:
    python test_photon_dt_ssm.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from temporal_encoders import (
    PhotonDtSSM,
    SetEmbedding,
    build_temporal_encoder,
    dt_aware_scan,
)


LOG_IPT_EPS = 1e-12  # must match data_loaders.PhotonEventDataset.LOG_IPT_EPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(B, N, fill_n=None, dt_scale=1.0, seed=0):
    """Build a synthetic (B, N, 4) photon tensor matching the dataloader layout.

    Per photon channels: (log(Δt + ε), G_onehot, R_onehot, mask).
    If `fill_n` is None, all N positions are real photons; otherwise the first
    `fill_n` are real and the rest are zero-padded (mask=0).
    """
    rng = np.random.default_rng(seed)
    if fill_n is None:
        fill_n = N
    # Inter-photon times: log-uniform in [1e-5, 1e-1] ms, scaled.
    dt = rng.uniform(1e-5, 1e-1, size=(B, fill_n)).astype(np.float32) * dt_scale
    chans = rng.integers(0, 2, size=(B, fill_n)).astype(np.uint8)
    g_oh = (chans == 0).astype(np.float32)
    r_oh = (chans == 1).astype(np.float32)
    log_dt = np.log(dt + np.float32(LOG_IPT_EPS))
    mask = np.ones((B, fill_n), dtype=np.float32)

    if fill_n < N:
        pad = N - fill_n
        log_dt = np.pad(log_dt, ((0, 0), (0, pad)))
        g_oh = np.pad(g_oh, ((0, 0), (0, pad)))
        r_oh = np.pad(r_oh, ((0, 0), (0, pad)))
        mask = np.pad(mask, ((0, 0), (0, pad)))
    x = np.stack([log_dt, g_oh, r_oh, mask], axis=-1)
    return torch.from_numpy(x).contiguous()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_shape():
    """Forward pass on (B=4, N=2048, 4) returns (4, 128)."""
    model = PhotonDtSSM(transition_dim=4, hidden_dim=32, num_blocks=2, output_dim=128)
    x = _make_input(B=4, N=2048)
    y = model(x)
    assert y.shape == (4, 128), f"expected (4, 128), got {y.shape}"
    assert torch.isfinite(y).all(), "output contains non-finite values"


def test_mask_invariance():
    """Appending masked-out padded photons must not change the output."""
    model = PhotonDtSSM(transition_dim=4, hidden_dim=16, num_blocks=2, output_dim=32)
    model.eval()
    x_short = _make_input(B=2, N=200, fill_n=200, seed=1)
    x_long = _make_input(B=2, N=512, fill_n=200, seed=1)  # same first 200 real photons + padding
    # First 200 positions must be identical between the two.
    assert torch.allclose(x_short[:, :200], x_long[:, :200])

    with torch.no_grad():
        y_short = model(x_short)
        y_long = model(x_long)
    assert torch.allclose(y_short, y_long, atol=1e-4, rtol=1e-4), \
        f"mask invariance violated: max abs diff = {(y_short - y_long).abs().max():.2e}"


def test_dt_sensitivity():
    """Scaling Δt by 2× must change the output (model is actually using Δt)."""
    model = PhotonDtSSM(transition_dim=4, hidden_dim=16, num_blocks=2, output_dim=32)
    model.eval()
    x1 = _make_input(B=2, N=300, dt_scale=1.0, seed=2)
    x2 = _make_input(B=2, N=300, dt_scale=2.0, seed=2)
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    diff = (y1 - y2).abs().max().item()
    assert diff > 1e-3, f"output is insensitive to Δt scaling (max diff {diff:.2e})"


def test_gradients_finite():
    """Backward gives finite gradients for α, B, C, blocks, and head."""
    model = PhotonDtSSM(transition_dim=4, hidden_dim=16, num_blocks=2, output_dim=32)
    x = _make_input(B=2, N=128, seed=3)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    none_grad = []
    bad_grad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            none_grad.append(name)
        elif not torch.isfinite(p.grad).all():
            bad_grad.append(name)
    assert not none_grad, f"params missing gradient: {none_grad}"
    assert not bad_grad, f"params with non-finite gradient: {bad_grad}"


def test_long_sequence_stability():
    """Scan stays finite for N=8192 with realistic Δt values."""
    model = PhotonDtSSM(transition_dim=4, hidden_dim=16, num_blocks=2, output_dim=32)
    model.eval()
    x = _make_input(B=1, N=8192, seed=4)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all(), "long-sequence output is non-finite"


def test_factory_builds_photon_dt_ssm():
    """build_temporal_encoder dispatches photon_dt_ssm correctly."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "name": "photon_dt_ssm",
        "hidden_dim": 32,
        "num_blocks": 2,
        "output_dim": 64,
        "bidirectional": True,
    })
    enc = build_temporal_encoder(cfg, transition_dim=4)
    assert isinstance(enc, PhotonDtSSM)
    x = _make_input(B=2, N=256)
    y = enc(x)
    assert y.shape == (2, 64)


def test_set_embedding_wraps_photon_dt_ssm():
    """SetEmbedding works around PhotonDtSSM with multi-trace input."""
    inner = PhotonDtSSM(transition_dim=4, hidden_dim=16, num_blocks=1, output_dim=24)
    wrapper = SetEmbedding(inner, post_agg_hidden=None, output_dim=24)
    x = _make_input(B=2 * 3, N=128).view(2, 3, 128, 4)
    y = wrapper(x)
    assert y.shape == (2, 24)


def test_scan_matches_naive_loop():
    """The chunked dt_aware_scan must equal a naive sequential reference."""
    B, N, H = 2, 257, 4  # N intentionally not a power of two
    torch.manual_seed(0)
    Bx = torch.randn(B, N, H)
    dt = torch.rand(B, N) * 1e-2 + 1e-5  # realistic Δt range in ms
    mask = torch.ones(B, N)
    mask[:, 200:] = 0.0  # last chunk padded
    alpha = torch.tensor([0.01, 0.1, 1.0, 10.0])  # spans μs → 100 ms timescales

    y_fast = dt_aware_scan(Bx, dt, mask, alpha)

    # Naive reference: pure sequential recurrence with mask gating.
    h = torch.zeros(B, H)
    out = torch.zeros(B, N, H)
    Bx_m = Bx * mask.unsqueeze(-1)
    dt_m = dt * mask
    for n in range(N):
        a = torch.exp(-alpha[None, :] * dt_m[:, n:n + 1])  # (B, H)
        h = a * h + Bx_m[:, n, :]
        out[:, n, :] = h

    max_diff = (y_fast - out).abs().max().item()
    assert max_diff < 1e-4, f"chunked scan diverges from naive loop: max diff {max_diff:.2e}"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
