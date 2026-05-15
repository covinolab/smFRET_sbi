"""Temporal embedding networks for SBI on per-photon event sequences.

Provides two encoders consumed via the `build_temporal_encoder` factory:
- `PhotonTCN`: dilated 1D-CNN over single-photon events (original baseline).
- `PhotonDtSSM`: Δt-aware diagonal linear SSM with chunked parallel scan
  (see plan at /home/dingeldein/.claude/plans/can-you-brainstorm-about-serene-scroll.md).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint


# Must match data_loaders.PhotonEventDataset.LOG_IPT_EPS — used to recover
# the raw Δt values from the `log(Δt + ε)` channel.
LOG_IPT_EPS = 1e-12


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Conv1dBlock(nn.Module):
    """Conv1d → channel-wise LayerNorm → GELU.

    Channel-wise LayerNorm (normalizes per timestep, across channels only) is
    used instead of GroupNorm/BatchNorm so the embedding is invariant to the
    length of zero-padding past the last real photon. With GroupNorm the
    mean/variance is computed across the spatial axis, mixing zeros from
    padded positions into the normalization stats, which breaks mask
    invariance.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # padding='same' so conv output length always matches input length,
        # for ANY kernel_size (PyTorch handles the asymmetric-padding case
        # internally when the kernel is even). The integer formula
        # `dilation * (kernel_size - 1) // 2` is off by 1 for even kernels.
        # bias=False so conv(zero) = zero; combined with end-of-block masking,
        # this preserves mask invariance (conv-zero-padding at the boundary
        # and feature-zeros in the masked region produce the same output).
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding='same',
            bias=False,
        )
        # elementwise_affine=False for the same reason: a learned bias inside
        # LayerNorm would also produce non-zero values at masked positions.
        self.norm = nn.LayerNorm(out_channels, elementwise_affine=False)
        self.act = nn.GELU()

    def forward(self, x, mask):
        # Mask immediately after the conv: a dilated conv reads from positions
        # outside the kernel's "real region," and even with bias=False those
        # outputs become non-zero anywhere the kernel touches a real photon.
        # Re-zeroing here ensures the next stage (norm, conv2 in the residual
        # block) doesn't see those leaked values, preserving mask invariance.
        x = self.conv(x) * mask    # (B, C, T)
        x = x.transpose(1, 2)      # (B, T, C)
        x = self.norm(x)
        x = x.transpose(1, 2)      # (B, C, T)
        return self.act(x)


class DilatedResidualBlock(nn.Module):
    """Two dilated convs + residual. Both convs share dilation."""

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = Conv1dBlock(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = Conv1dBlock(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x, mask):
        return x + self.conv2(self.conv1(x, mask), mask)


class MaskedAttentionPool(nn.Module):
    """Multi-head learned-query attention pool over the time axis.

    For each of `num_heads` heads, a linear scorer over channels produces a
    score per (sample, time, head). A softmax over time — with -inf at
    padded positions — gives normalised attention weights; each head's
    pooled vector is the weighted sum of features across time. Returns the
    H pooled vectors concatenated → (B, C*H).

    Why this and not full self-attention: at N≈16384 photons, O(N²) is
    catastrophic; this learned-query variant is O(N) — same complexity as
    the mean+std pool it augments.

    Mask discipline matches the rest of the trunk: scores at padded
    positions are set to -inf BEFORE the softmax. Zeroing the weights
    after softmax would leak softmax mass into padding (since the softmax
    normalises over all positions including masked ones), breaking mask
    invariance.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        # bias=False mirrors the rest of the trunk's discipline; not strictly
        # required here (the pool is the last op before the head) but cheap.
        self.score = nn.Linear(channels, num_heads, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h:    (B, C, T)
        # mask: (B, 1, T) or (B, T) — float in {0, 1}
        if mask.dim() == 3:
            mask_t = mask.squeeze(1)
        else:
            mask_t = mask
        scores = self.score(h.transpose(1, 2))                  # (B, T, H)
        scores = scores.masked_fill(mask_t.unsqueeze(-1) == 0, float('-inf'))
        # Guard against all-masked rows (would give NaN softmax). The loader
        # enforces lengths ≥ 1 so this is defensive only.
        all_masked = (mask_t.sum(dim=-1) == 0).unsqueeze(-1).unsqueeze(-1)
        weights = scores.softmax(dim=1)                         # (B, T, H)
        weights = torch.where(all_masked, torch.zeros_like(weights), weights)
        pooled = torch.einsum('bct,bth->bch', h, weights)       # (B, C, H)
        return pooled.reshape(pooled.size(0), -1)               # (B, C*H)


# ---------------------------------------------------------------------------
# PhotonTCN
# ---------------------------------------------------------------------------

class PhotonTCN(nn.Module):
    """Dilated 1D-CNN embedding on per-photon event sequences.

    Input
    -----
    x : (B, N_max, 4) float32
        Per photon channels: (log Δt, G_onehot, R_onehot, mask).
        The last channel is the validity mask (1.0 for real photons, 0.0 for padding).

    Output
    ------
    (B, output_dim) float32
    """

    def __init__(
        self,
        transition_dim: int = 4,
        hidden_dim: int = 64,
        num_blocks: int = 7,
        kernel_size: int = 3,
        output_dim: int = 128,
        num_attn_heads: int = 4,
    ):
        super().__init__()
        if transition_dim < 4:
            raise ValueError(
                f"PhotonTCN expects transition_dim >= 4 (3 features + 1 mask channel); "
                f"got transition_dim={transition_dim}"
            )
        self.transition_dim = transition_dim
        # Number of real feature channels (everything except the trailing mask).
        self.num_feature_channels = transition_dim - 1
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_attn_heads = num_attn_heads

        self.input_proj = nn.Conv1d(self.num_feature_channels, hidden_dim, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(hidden_dim, kernel_size, dilation=2 ** k)
            for k in range(num_blocks)
        ])
        self.attn_pool = MaskedAttentionPool(hidden_dim, num_heads=num_attn_heads)
        # Head input: mean (C) + std (C) + attn_pool (H*C) = (2+H)*C.
        # Final LayerNorm anchors the output magnitude — without it the encoder
        # can shrink its outputs toward zero (collapse) and the flow stops
        # conditioning on x. elementwise_affine=False so the learned γ can't
        # gradually shrink individual channels (which we saw as slow drift in
        # probe_l2). Boundary-leak concern doesn't apply: no convs downstream.
        self.head = nn.Sequential(
            nn.Linear((2 + num_attn_heads) * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_max, 4) - split features and mask.
        features = x[..., :self.num_feature_channels]    # (B, N_max, num_feat)
        mask = x[..., -1]                                 # (B, N_max), float in {0,1}

        # → channels-first for conv.
        h = features.transpose(1, 2)                      # (B, num_feat, N_max)
        m = mask.unsqueeze(1)                             # (B, 1, N_max)

        h = self.input_proj(h) * m                        # zero out padded positions
        for block in self.blocks:
            h = block(h, m) * m                           # mask threaded inside the block too

        # Masked mean + std pool over the event axis, plus a multi-head
        # learned-query attention pool. mean+std give cheap "average FRET /
        # average heterogeneity" baselines; attention lets the network learn
        # what specific events along the trace are diagnostic of θ (e.g.
        # focus on transition photons rather than dwell photons).
        denom = m.sum(dim=-1).clamp(min=1.0)              # (B, 1)
        mean = (h * m).sum(dim=-1) / denom                # (B, C)
        var = ((h * h) * m).sum(dim=-1) / denom - mean * mean
        std = var.clamp(min=1e-8).sqrt()                  # (B, C)
        attn = self.attn_pool(h, m)                       # (B, H*C)
        pooled = torch.cat([mean, std, attn], dim=-1)     # (B, (2+H)*C)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Δt-aware diagonal SSM (PhotonDtSSM)
# ---------------------------------------------------------------------------

def dt_aware_scan(Bx: torch.Tensor,
                  dt: torch.Tensor,
                  mask: torch.Tensor,
                  alpha: torch.Tensor,
                  chunk_size: int = 64) -> torch.Tensor:
    """Compute h_n = exp(-α ⊙ Δt_n) ⊙ h_{n-1} + Bx_n via chunked parallel scan.

    Bx    : (B, N, H) input contributions per photon.
    dt    : (B, N)    interphoton times (same units as 1/α; here ms).
    mask  : (B, N)    1.0 for real photons, 0.0 for padding.
    alpha : (H,)      positive decay rates, one per hidden dim.

    At masked positions we set a_n = 1 and Bx_n = 0, so padded photons are
    a no-op (state propagates unchanged).

    Implementation: split the sequence into chunks of size K. Inside each
    chunk run a sequential intra-chunk scan vectorised over (B, M, H); the
    cross-chunk carry is a second sequential scan over only M values. Total
    sequential GPU launches: K + M ≈ 2√N for K = √N, instead of N.
    Computation kept in fp32 for stable cumsums.
    """
    B, N, H = Bx.shape
    K = chunk_size
    pad = (K - N % K) % K
    if pad > 0:
        Bx = F.pad(Bx, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad), value=0.0)
        dt = F.pad(dt, (0, pad), value=0.0)
    NP = N + pad
    M = NP // K

    orig_dtype = Bx.dtype
    Bx32 = Bx.float()
    dt32 = dt.float()
    mask32 = mask.float()
    alpha32 = alpha.float()

    Bx_m = Bx32 * mask32.unsqueeze(-1)
    dt_m = dt32 * mask32                                   # padded → dt=0 → a=1
    log_a = -alpha32[None, None, :] * dt_m[:, :, None]     # (B, NP, H), ≤ 0

    Bx_r = Bx_m.view(B, M, K, H)
    log_a_r = log_a.view(B, M, K, H)

    # Intra-chunk: small sequential scan vectorised over (B, M, H).
    h = torch.zeros(B, M, H, device=Bx.device, dtype=torch.float32)
    h_steps = []
    for k in range(K):
        a_k = torch.exp(log_a_r[:, :, k, :])               # (B, M, H)
        h = a_k * h + Bx_r[:, :, k, :]
        h_steps.append(h)
    h_intra = torch.stack(h_steps, dim=2)                  # (B, M, K, H)
    h_intra_final = h_intra[:, :, -1, :]                   # (B, M, H)

    cum_log_a_chunk = torch.cumsum(log_a_r, dim=2)         # (B, M, K, H)
    chunk_decay = cum_log_a_chunk[:, :, -1, :]             # (B, M, H)

    # Cross-chunk: sequential scan over M chunks of the carry-state.
    h_carry = torch.zeros(B, H, device=Bx.device, dtype=torch.float32)
    prior_carries = []
    for m in range(M):
        prior_carries.append(h_carry)
        h_carry = torch.exp(chunk_decay[:, m, :]) * h_carry + h_intra_final[:, m, :]
    prior_carry = torch.stack(prior_carries, dim=1)        # (B, M, H)

    # Combine intra-chunk results with the chunk's carry-in.
    full = h_intra + torch.exp(cum_log_a_chunk) * prior_carry.unsqueeze(2)
    out = full.view(B, NP, H)
    if pad > 0:
        out = out[:, :N, :]
    return out.to(orig_dtype)


class DtSSM(nn.Module):
    """Single-direction Δt-aware diagonal linear SSM.

    Each hidden dimension is a continuous-time exponential filter on the
    photon stream:
        h_n = exp(-α_h Δt_n) · h_{n-1} + (B x_n)_h
    α is parameterised as softplus(raw_α) for positivity and initialised
    log-spaced over `alpha_init_range_ms` so the bank covers multiple
    physical timescales (μs → 100 ms) by construction.
    """

    def __init__(self, dim_in: int, hidden: int, dim_out: int,
                 alpha_init_range_ms=(1e-3, 1e2),
                 chunk_size: int = 64):
        super().__init__()
        self.hidden = hidden
        self.chunk_size = chunk_size

        log_lo = math.log(float(alpha_init_range_ms[0]))
        log_hi = math.log(float(alpha_init_range_ms[1]))
        alpha_init = torch.exp(torch.linspace(log_lo, log_hi, hidden)).clamp(min=1e-6)
        # softplus(raw) = α  ⇒  raw = log(exp(α) − 1) = α + log(1 − exp(−α)).
        # The second form avoids overflow of expm1(α) for large α (e.g. α=100
        # overflows fp32 since exp(100) > 3.4e38).
        raw_alpha_init = alpha_init + torch.log(-torch.expm1(-alpha_init))
        self.raw_alpha = nn.Parameter(raw_alpha_init)

        self.in_proj = nn.Linear(dim_in, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, dim_out, bias=False)

    def forward(self, x: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim_in)
        Bx = self.in_proj(x)                                   # (B, N, hidden)
        alpha = F.softplus(self.raw_alpha)                     # (hidden,)
        h = dt_aware_scan(Bx, dt, mask, alpha, chunk_size=self.chunk_size)
        return self.out_proj(h)


class BiDirDtSSM(nn.Module):
    """Bidirectional Δt-aware SSM. Forward and reverse passes have independent
    α, in_proj, out_proj. Reverse pass runs on time-flipped (x, dt, mask) and
    the output is flipped back before concatenation along the feature axis.
    """

    def __init__(self, dim_in: int, hidden: int, dim_out: int,
                 alpha_init_range_ms=(1e-3, 1e2),
                 chunk_size: int = 64):
        super().__init__()
        self.fwd = DtSSM(dim_in, hidden, dim_out, alpha_init_range_ms,
                         chunk_size)
        self.rev = DtSSM(dim_in, hidden, dim_out, alpha_init_range_ms,
                         chunk_size)

    def forward(self, x, dt, mask):
        y_f = self.fwd(x, dt, mask)
        x_r = x.flip(1)
        dt_r = dt.flip(1)
        mask_r = mask.flip(1)
        y_r = self.rev(x_r, dt_r, mask_r).flip(1)
        return torch.cat([y_f, y_r], dim=-1)


class BiDirDtSSMBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → BiDir SSM → GLU → Linear → residual.

    Mask is applied immediately after the SSM/MLP path so padded positions
    contribute zero to the residual stream (preserving mask invariance even
    though the SSM scan internally carries state forward through padding).
    """

    def __init__(self, channels: int,
                 alpha_init_range_ms=(1e-3, 1e2),
                 bidirectional: bool = True,
                 chunk_size: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        if bidirectional:
            self.ssm = BiDirDtSSM(channels, channels, channels,
                                  alpha_init_range_ms, chunk_size)
            mix_in = 2 * channels
        else:
            self.ssm = DtSSM(channels, channels, channels,
                             alpha_init_range_ms, chunk_size)
            mix_in = channels
        self.glu = nn.Sequential(
            nn.Linear(mix_in, 2 * channels, bias=False),
            nn.GLU(dim=-1),
        )
        self.proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x, dt, mask):
        m = mask.unsqueeze(-1)
        h = self.norm(x) * m
        h = self.ssm(h, dt, mask)
        h = self.glu(h)
        h = self.proj(h)
        h = h * m
        return x + h


class PhotonDtSSM(nn.Module):
    """Δt-aware diagonal SSM embedding on per-photon event sequences.

    Input
    -----
    x : (B, N_max, transition_dim) float32
        Channels: (log(Δt + ε), G_onehot, R_onehot, mask). transition_dim must
        be ≥ 4; only the first `transition_dim - 1` channels are used as
        features, the last is the validity mask.

    Output
    ------
    (B, output_dim) float32
    """

    def __init__(self,
                 transition_dim: int = 4,
                 hidden_dim: int = 64,
                 num_blocks: int = 2,
                 output_dim: int = 128,
                 bidirectional: bool = True,
                 alpha_init_range_ms=(1e-3, 1e2),
                 chunk_size: int = 64,
                 num_attn_heads: int = 4,
                 **_ignored):
        super().__init__()
        if transition_dim < 4:
            raise ValueError(
                f"PhotonDtSSM expects transition_dim >= 4; got {transition_dim}"
            )
        self.transition_dim = transition_dim
        self.num_feature_channels = transition_dim - 1
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_attn_heads = num_attn_heads

        self.input_proj = nn.Linear(self.num_feature_channels, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([
            BiDirDtSSMBlock(
                hidden_dim,
                alpha_init_range_ms=alpha_init_range_ms,
                bidirectional=bidirectional,
                chunk_size=chunk_size,
            )
            for _ in range(num_blocks)
        ])
        self.attn_pool = MaskedAttentionPool(hidden_dim, num_heads=num_attn_heads)
        # Same head shape as PhotonTCN: mean (C) + std (C) + attn (H*C).
        # Clean apples-to-apples swap on the downstream flow.
        self.head = nn.Sequential(
            nn.Linear((2 + num_attn_heads) * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x[..., :self.num_feature_channels]          # (B, N, num_feat)
        log_dt = x[..., 0]                                      # (B, N)
        mask = x[..., -1]                                       # (B, N)
        # Recover raw Δt from the log-Δt channel — the dataloader writes
        # log(Δt + LOG_IPT_EPS), so invert exactly. Padded positions in the
        # source tensor have log_dt = 0 → recovered Δt ≈ 1 - ε, but mask=0
        # gates them out inside the scan.
        dt = (torch.exp(log_dt) - LOG_IPT_EPS).clamp(min=0.0)

        m = mask.unsqueeze(-1)
        h = self.input_proj(features) * m
        for block in self.blocks:
            h = block(h, dt, mask) * m

        denom = m.sum(dim=1).clamp(min=1.0)                    # (B, 1)
        mean = (h * m).sum(dim=1) / denom                      # (B, hidden_dim)
        var = ((h * h) * m).sum(dim=1) / denom - mean * mean
        std = var.clamp(min=1e-8).sqrt()
        # MaskedAttentionPool expects channels-first; SSM trunk is (B, N, C).
        attn = self.attn_pool(h.transpose(1, 2), mask)         # (B, H*hidden_dim)
        pooled = torch.cat([mean, std, attn], dim=-1)          # (B, (2+H)*hidden_dim)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# FretSSM — photophysics-aware encoder (L3 + L4 + L5 + L6)
# ---------------------------------------------------------------------------
#
# Entirely additive: ignore this class and the matching factory branch and
# behaviour of TCN/SSM is unchanged. To remove FretSSM completely:
#   1. Delete the FretSSM class block below.
#   2. Delete the `fret_ssm` branch in build_temporal_encoder().
#   3. Delete the matching aux-loss hook in train_density_estimator.py
#      (clearly marked with "FretSSM aux hook").
#   4. Delete conf/training/embedding/fret_ssm.yaml.
#   5. Delete conf/training/exp_9_fret.yaml.

class FretSSM(nn.Module):
    """Photophysics-aware Δt-SSM encoder for smFRET per-photon events.

    Implements four photophysics priors on top of the basic Δt-SSM:

    - L3 (paired G/R SSM branches): Each hidden dim has one shared decay
      rate α_h, but separate state for G and R photons. Only G events
      update the G-state; only R events update the R-state. Outputs
      softplus-positive rate estimates λ_G[h], λ_R[h], and per-timescale
      FRET efficiency E[h] = λ_R[h]/(λ_G[h]+λ_R[h]).

    - L4 (brightness-invariant pooling): The pool input is *only*
      brightness-invariant features — per-timescale E, log(λ_R/λ_G),
      and rel-log-total (log total rate minus per-position mean across
      timescales). Absolute rates are NOT exposed to the pool, so the
      encoder representation is robust to absolute brightness changes
      between sim and experiment.

    - L5 (nuisance head): A small unsupervised head maps a 8-dim per-trace
      summary to a 5-dim conditioning vector ψ̂ which is concatenated to
      the main embedding for the flow. No true nuisance values are
      exposed anywhere — ψ̂ is learned purely from gradient signal back
      through the flow's loss.

    - L6 (Poisson auxiliary loss): During training, the encoder computes
      a self-supervised next-photon log-likelihood and stashes it on
      self.last_aux_loss. The training script reads this and adds
      aux_loss_weight × last_aux_loss to its NPE loss. Default weight
      is 0.0 (the encoder computes nothing); set > 0 in the config to
      enable. Inference (eval mode) skips this regardless.

    Output shape: (B, output_dim + 5)   — main embedding + ψ̂.

    Cost vs PhotonDtSSM: ~2× scan compute (G+R branches) at the same
    hidden_dim. Pool input is 3× wider. Default hidden_dim=32 keeps
    total compute comparable to a hidden_dim=64 single-branch SSM.
    """

    EPS_RATE = 1e-6        # numerical floor on rates / log denominator
    LOG_RATIO_CLAMP = 8.0  # clip log(λ_R/λ_G) to avoid blowup at extreme ratios
    NUM_NUISANCES = 5      # nominally (η_G, η_R, k_D, k_gb, k_rb); names only
    GLOBAL_STATS_DIM = 8   # number of per-trace summary scalars (see _global_stats)

    def __init__(self,
                 transition_dim: int = 4,
                 hidden_dim: int = 64,
                 output_dim: int = 128,
                 alpha_init_range_ms=(1e-3, 1e2),
                 chunk_size: int = 64,
                 num_attn_heads: int = 4,
                 aux_loss_weight: float = 0.0,
                 gradient_checkpoint: bool = False,
                 **_ignored):
        super().__init__()
        if transition_dim < 4:
            raise ValueError(
                f"FretSSM expects transition_dim >= 4 (log_dt, G_oh, R_oh, mask); "
                f"got transition_dim={transition_dim}"
            )
        self.transition_dim = transition_dim
        self.num_feature_channels = transition_dim - 1
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.chunk_size = chunk_size
        self.aux_loss_weight = float(aux_loss_weight)
        # When True, the L3 paired-branch core (in_proj_G/R → scans → out_proj_G/R)
        # is wrapped in torch.utils.checkpoint, freeing scan activations after
        # forward and recomputing during backward. ~30% slower per step, frees
        # the dominant ~2×(B·N·C·4B) of GPU memory. Aux loss correctness is
        # unaffected — both NPE and aux gradients flow through the recomputed
        # graph. Off by default.
        self.gradient_checkpoint = bool(gradient_checkpoint)

        # Input projection + pre-norm.
        self.input_proj = nn.Linear(self.num_feature_channels, hidden_dim, bias=False)
        self.pre_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # α-bank shared between G and R branches for matched-timescale ratios.
        log_lo = math.log(float(alpha_init_range_ms[0]))
        log_hi = math.log(float(alpha_init_range_ms[1]))
        alpha_init = torch.exp(torch.linspace(log_lo, log_hi, hidden_dim)).clamp(min=1e-6)
        # softplus(raw) = α  ⇒  raw = α + log(1 − exp(−α)). Same form as DtSSM.
        raw_alpha_init = alpha_init + torch.log(-torch.expm1(-alpha_init))
        self.raw_alpha = nn.Parameter(raw_alpha_init)

        # Separate in/out projections per branch.
        self.in_proj_G = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.in_proj_R = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj_G = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj_R = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Pool the L4-feature stream (3 × hidden_dim channels per position).
        self.attn_pool = MaskedAttentionPool(3 * hidden_dim, num_heads=num_attn_heads)
        pool_dim = (2 + num_attn_heads) * 3 * hidden_dim
        # Main head. Final LayerNorm pins embedding magnitude (same pattern as
        # PhotonTCN / PhotonDtSSM).
        self.head = nn.Sequential(
            nn.Linear(pool_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
        )

        # Nuisance head: small MLP from per-trace global summary.
        self.nuisance_head = nn.Sequential(
            nn.Linear(self.GLOBAL_STATS_DIM, 32),
            nn.GELU(),
            nn.Linear(32, self.NUM_NUISANCES),
        )

        # L6 stash. Training script reads this if aux_loss_weight > 0.
        self.last_aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, transition_dim). Channels: (log_dt, G_oh, R_oh, mask).
        features = x[..., :self.num_feature_channels]    # (B, N, 3)
        log_dt = x[..., 0]                                # (B, N)
        G_oh = x[..., 1]                                  # (B, N)
        R_oh = x[..., 2]                                  # (B, N)
        mask = x[..., -1]                                 # (B, N)
        dt = (torch.exp(log_dt) - LOG_IPT_EPS).clamp(min=0.0)

        # Project + pre-norm; zero out padded positions defensively.
        h_in = self.pre_norm(self.input_proj(features)) * mask.unsqueeze(-1)
        alpha = F.softplus(self.raw_alpha)                # (hidden_dim,)

        # L3 paired-branch core. Wrapped in gradient_checkpoint when enabled
        # (training only) to free the dominant scan activations.
        if self.gradient_checkpoint and self.training:
            lam_G, lam_R = _checkpoint(
                self._branch_forward,
                h_in, dt, G_oh, R_oh, mask, alpha,
                use_reentrant=False,
            )
        else:
            lam_G, lam_R = self._branch_forward(h_in, dt, G_oh, R_oh, mask, alpha)

        # L4: brightness-invariant feature stream for the pool.
        lam_total = lam_G + lam_R
        E = lam_R / lam_total                             # (B, N, C)
        log_ratio = (torch.log(lam_R) - torch.log(lam_G)).clamp(
            -self.LOG_RATIO_CLAMP, self.LOG_RATIO_CLAMP
        )
        log_total = torch.log(lam_total)
        rel_log_total = log_total - log_total.mean(dim=-1, keepdim=True)
        feats = torch.cat([E, log_ratio, rel_log_total], dim=-1)   # (B, N, 3C)
        m = mask.unsqueeze(-1)
        feats = feats * m

        # Mean + std + attention pool over time.
        denom = m.sum(dim=1).clamp(min=1.0)               # (B, 1)
        mean = (feats * m).sum(dim=1) / denom             # (B, 3C)
        var = ((feats * feats) * m).sum(dim=1) / denom - mean * mean
        std = var.clamp(min=1e-8).sqrt()
        attn = self.attn_pool(feats.transpose(1, 2), mask)        # (B, H*3C)
        pooled = torch.cat([mean, std, attn], dim=-1)             # (B, (2+H)*3C)
        emb_main = self.head(pooled)                              # (B, output_dim)

        # L5: nuisance head from per-trace global stats.
        stats = self._global_stats(lam_G, lam_R, log_dt, G_oh, mask, denom)
        psi = self.nuisance_head(stats)                            # (B, 5)

        # L6: Poisson next-photon log-likelihood, only when training & enabled.
        if self.training and self.aux_loss_weight > 0:
            self.last_aux_loss = self._poisson_aux_loss(
                lam_G, lam_R, dt, G_oh, R_oh, mask
            )
        else:
            self.last_aux_loss = None

        return torch.cat([emb_main, psi], dim=-1)         # (B, output_dim + 5)

    def _branch_forward(self, h_in, dt, G_oh, R_oh, mask, alpha):
        """L3 paired-branch core. Separated so it can be gradient-checkpointed.

        Computes (λ_G, λ_R) of shape (B, N, hidden_dim) each. The scan
        outputs h_G/h_R and the softplus(rate) outputs are the activations
        that dominate FretSSM's memory footprint — they are released after
        forward when this is invoked under torch.utils.checkpoint.
        """
        u_G = self.in_proj_G(h_in) * G_oh.unsqueeze(-1)
        h_G = dt_aware_scan(u_G, dt, mask, alpha, chunk_size=self.chunk_size)
        lam_G = F.softplus(self.out_proj_G(h_G)) + self.EPS_RATE

        u_R = self.in_proj_R(h_in) * R_oh.unsqueeze(-1)
        h_R = dt_aware_scan(u_R, dt, mask, alpha, chunk_size=self.chunk_size)
        lam_R = F.softplus(self.out_proj_R(h_R)) + self.EPS_RATE
        return lam_G, lam_R

    def _global_stats(self, lam_G, lam_R, log_dt, G_oh, mask, denom):
        """Per-trace 8-d summary fed to the nuisance head."""
        m = mask.unsqueeze(-1)
        # Mean over time, then mean over hidden timescales → single scalar per
        # trace, per stat.
        log_lam_G_mean = ((torch.log(lam_G) * m).sum(dim=1) / denom).mean(dim=-1, keepdim=True)
        log_lam_R_mean = ((torch.log(lam_R) * m).sum(dim=1) / denom).mean(dim=-1, keepdim=True)
        log_lam_tot_mean = ((torch.log(lam_G + lam_R) * m).sum(dim=1) / denom).mean(dim=-1, keepdim=True)
        log_n_real = torch.log(mask.sum(dim=1, keepdim=True) + 1.0)
        denom_2d = mask.sum(dim=1, keepdim=True).clamp(min=1)
        g_frac = (G_oh * mask).sum(dim=1, keepdim=True) / denom_2d
        log_dt_mean = (log_dt * mask).sum(dim=1, keepdim=True) / denom_2d
        E_per_event = lam_R / (lam_G + lam_R)
        E_mean = ((E_per_event * m).sum(dim=1) / denom).mean(dim=-1, keepdim=True)
        log_brightness_ratio = log_lam_G_mean - log_lam_R_mean
        return torch.cat([
            log_lam_G_mean, log_lam_R_mean, log_lam_tot_mean,
            log_n_real, g_frac, log_dt_mean, E_mean,
            log_brightness_ratio,
        ], dim=-1)

    def _poisson_aux_loss(self, lam_G, lam_R, dt, G_oh, R_oh, mask):
        """Next-photon Poisson NLL using rate estimates at the previous event.

        log p(next event = (channel c, waiting time Δt))
            = log λ_c(t_n) - (λ_G + λ_R)(t_n) · Δt_{n+1}

        Reduces over hidden timescales by averaging (consensus rate); reduces
        over events by masked mean. Returns scalar (mean over batch).
        """
        # Consensus rate across hidden timescales: simple mean.
        lam_G_t = lam_G.mean(dim=-1)        # (B, N)
        lam_R_t = lam_R.mean(dim=-1)
        lam_tot_t = lam_G_t + lam_R_t

        # Shift-by-one: rates at position n predict event n+1.
        lam_G_pred = lam_G_t[:, :-1]
        lam_R_pred = lam_R_t[:, :-1]
        lam_tot_pred = lam_tot_t[:, :-1]
        c_G_next = G_oh[:, 1:]
        c_R_next = R_oh[:, 1:]
        dt_next = dt[:, 1:]
        # Pair-mask: both positions must be real photons.
        pair_mask = mask[:, :-1] * mask[:, 1:]

        log_lik = (
            c_G_next * torch.log(lam_G_pred) + c_R_next * torch.log(lam_R_pred)
            - lam_tot_pred * dt_next
        ) * pair_mask
        n_pairs = pair_mask.sum(dim=1).clamp(min=1)
        nll = -log_lik.sum(dim=1) / n_pairs
        return nll.mean()


# ---------------------------------------------------------------------------
# Set-mode wrapper (multi-trace per parameter)
# ---------------------------------------------------------------------------

class SetEmbedding(nn.Module):
    """Permutation-invariant embedding for sets of per-trace event sequences.

    Applies a per-trace encoder independently, then mean-pools across the set.
    Input x is (B, N_traces, N_max, transition_dim).

    Why mean (not sum): set size is fixed in this pipeline, so the choice is
    just a scale factor. But with sum, the embedding magnitude grows linearly
    with set size (std ≈ 20× per-trace std for N=20), pushing the downstream
    flow's conditioning MLP into a saturated regime where ∂loss/∂embedding ≈ 0
    and the encoder receives no gradient. Mean keeps magnitudes O(1) regardless
    of set size.
    """

    def __init__(self, single_trace_encoder, post_agg_hidden=None, output_dim=None):
        super().__init__()
        self.encoder = single_trace_encoder
        if post_agg_hidden and output_dim:
            self.post_agg = nn.Sequential(
                nn.Linear(output_dim, post_agg_hidden),
                nn.GELU(),
                nn.Linear(post_agg_hidden, output_dim),
            )
        else:
            self.post_agg = nn.Identity()

    def forward(self, x):
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C)
        x = self.encoder(x)
        x = x.reshape(B, N, -1)
        x = x.mean(dim=1)
        return self.post_agg(x)


# ---------------------------------------------------------------------------
# Factories (kept thin — only one valid encoder name now)
# ---------------------------------------------------------------------------

def build_temporal_encoder(cfg_embedding, transition_dim, max_len=None):
    """Build a temporal encoder from a Hydra `embedding` config.

    Supported `embedding.name` values: `photon_tcn`, `photon_dt_ssm`,
    `fret_ssm`. `max_len` is accepted for backward-compat with old call
    sites but is unused — all encoders handle variable lengths via the
    mask channel.
    """
    encoder_name = str(getattr(cfg_embedding, "name", "photon_tcn")).lower()
    num_attn_heads = int(getattr(cfg_embedding, "num_attn_heads", 4))
    if encoder_name in {"photon_tcn", "phototcn"}:
        return PhotonTCN(
            transition_dim=transition_dim,
            hidden_dim=int(getattr(cfg_embedding, "hidden_dim", 64)),
            num_blocks=int(getattr(cfg_embedding, "num_blocks", 7)),
            kernel_size=int(getattr(cfg_embedding, "kernel_size", 3)),
            output_dim=int(getattr(cfg_embedding, "output_dim", 128)),
            num_attn_heads=num_attn_heads,
        )
    if encoder_name in {"photon_dt_ssm", "photondtssm"}:
        alpha_range = getattr(cfg_embedding, "alpha_init_range_ms", (1e-3, 1e2))
        # Hydra/OmegaConf returns a ListConfig; convert to a plain tuple.
        if not isinstance(alpha_range, tuple):
            alpha_range = tuple(float(v) for v in alpha_range)
        return PhotonDtSSM(
            transition_dim=transition_dim,
            hidden_dim=int(getattr(cfg_embedding, "hidden_dim", 64)),
            num_blocks=int(getattr(cfg_embedding, "num_blocks", 2)),
            output_dim=int(getattr(cfg_embedding, "output_dim", 128)),
            bidirectional=bool(getattr(cfg_embedding, "bidirectional", True)),
            alpha_init_range_ms=alpha_range,
            chunk_size=int(getattr(cfg_embedding, "chunk_size", 64)),
            num_attn_heads=num_attn_heads,
        )
    if encoder_name in {"fret_ssm", "fretssm"}:
        alpha_range = getattr(cfg_embedding, "alpha_init_range_ms", (1e-3, 1e2))
        if not isinstance(alpha_range, tuple):
            alpha_range = tuple(float(v) for v in alpha_range)
        return FretSSM(
            transition_dim=transition_dim,
            hidden_dim=int(getattr(cfg_embedding, "hidden_dim", 64)),
            output_dim=int(getattr(cfg_embedding, "output_dim", 128)),
            alpha_init_range_ms=alpha_range,
            chunk_size=int(getattr(cfg_embedding, "chunk_size", 64)),
            num_attn_heads=num_attn_heads,
            aux_loss_weight=float(getattr(cfg_embedding, "aux_loss_weight", 0.0)),
            gradient_checkpoint=bool(getattr(cfg_embedding, "gradient_checkpoint", False)),
        )
    raise ValueError(
        f"Unknown embedding.name='{encoder_name}'. "
        f"Supported: 'photon_tcn', 'photon_dt_ssm', 'fret_ssm'."
    )


def build_set_embedding(cfg_embedding, transition_dim, max_len,
                        post_agg_hidden=None):
    encoder = build_temporal_encoder(cfg_embedding, transition_dim, max_len)
    output_dim = int(getattr(cfg_embedding, "output_dim", 128))
    return SetEmbedding(encoder, post_agg_hidden, output_dim)
