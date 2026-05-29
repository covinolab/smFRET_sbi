"""SmFRET photon-stream encoder.

A summary network for SBI on photon-by-photon smFRET traces.

Architecture:

    photons (log_dt, channel, dt, mask)
        -> PhotonTokenizer
        -> LocalConvStack
        -> FastSSMBlock * n_ssm     -- chunked diagonal SSM, time-aware
            |-> NextPhotonHead      -- self-supervised next-photon NLL
        -> SetTransformerPool       -- permutation-invariant pool
        -> SummaryProjection        -> z (B, d_z)

The module file is intentionally self-contained: deleting this file plus the
one factory branch in `temporal_encoders.py` and the YAML config fully
removes the encoder.

Input adapter
-------------
The pipeline feeds the encoder a `(B, N, 4)` tensor where the 4 channels are
`(log(Δt + 1e-12) [ms], G_one_hot, R_one_hot, mask_float)`.
`SmFRETEncoderAdapter` splits that into the four named arguments
(`log_dt`, `channel`, `dt`, `mask`), standardizes `log_dt`, rescales `dt`
to O(1), and on the first forward pass auto-calibrates the standardization
buffers from the batch it sees. The next-photon NLL is computed against the
unstandardized log Δt (required for the change-of-variables Jacobian) and
exposed as `last_aux_loss` so the existing training-loop hook (which reads
`last_aux_loss` / `aux_loss_weight` from the inner encoder) adds it to the
NPE gradient automatically.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the project's diagonal real-α SSM (same one FretSSM / PhotonDtSSM
# use); its chunked-sequential `dt_aware_scan` fits the step-and-decay
# Poisson photon stream and runs fast on GPU.
from temporal_encoders import DtSSM


# ---------------------------------------------------------------------------
# 1. PhotonTokenizer
# ---------------------------------------------------------------------------

class PhotonTokenizer(nn.Module):
    def __init__(self, d_model: int = 128, n_channels: int = 2):
        super().__init__()
        self.n_channels = n_channels
        self.proj = nn.Linear(1 + n_channels, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, log_dt: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        ch = F.one_hot(channel, num_classes=self.n_channels).float()
        x = torch.cat([log_dt.unsqueeze(-1), ch], dim=-1)
        return self.norm(self.proj(x))


# ---------------------------------------------------------------------------
# 2. LocalConvStack
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size[0] - 1) * self.dilation[0]
        x = F.pad(x, (pad, 0))
        return super().forward(x)


class LocalConvStack(nn.Module):
    def __init__(self, d_model: int = 128, dilations=(1, 2, 4), kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalConv1d(d_model, d_model, kernel_size=kernel_size, dilation=d)
            for d in dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for conv, norm in zip(self.layers, self.norms):
            y = self.act(conv(x))
            x = norm((x + y).transpose(1, 2)).transpose(1, 2)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# 3. FastSSMBlock
# ---------------------------------------------------------------------------

class FastSSMBlock(nn.Module):
    """Transformer-style block around the project's diagonal real-α SSM.

    Recurrence: ``h_t = exp(-α ⊙ dt_t) · h_{t-1} + B·u_t`` per hidden dim.
    α is positive (softplus-parameterized) and log-spaced over
    ``alpha_init_range_ms`` at init so the bank covers μs→100 ms timescales.
    """

    def __init__(self, d_model: int = 128, d_state: int = 64, mlp_mult: int = 4,
                 chunk_size: int = 64,
                 alpha_init_range_ms=(1e-3, 1e2)):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = DtSSM(
            dim_in=d_model, hidden=d_state, dim_out=d_model,
            alpha_init_range_ms=alpha_init_range_ms,
            chunk_size=chunk_size,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # `mask` may be bool; `DtSSM` casts to float internally.
        x = x + self.ssm(self.norm1(x), dt, mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 4. NextPhotonHead
# ---------------------------------------------------------------------------

class NextPhotonHead(nn.Module):
    def __init__(self, d_model: int = 128, K: int = 8, n_channels: int = 2):
        super().__init__()
        self.K = K
        self.dt_proj = nn.Linear(d_model, 3 * K)
        self.ch_proj = nn.Linear(d_model, n_channels)

    def forward(self, h: torch.Tensor) -> dict:
        dt = self.dt_proj(h)
        log_w = F.log_softmax(dt[..., : self.K], dim=-1)
        mu = dt[..., self.K : 2 * self.K]
        log_s = dt[..., 2 * self.K :].clamp(-7.0, 4.0)
        ch_logits = self.ch_proj(h)
        return {"log_w": log_w, "mu": mu, "log_s": log_s, "ch_logits": ch_logits}


# ---------------------------------------------------------------------------
# 5. SetTransformerPool (ISAB + PMA, masked)
# ---------------------------------------------------------------------------

class MaskedMAB(nn.Module):
    def __init__(self, d_q: int, d_kv: int, d_out: int, num_heads: int = 4):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_out // num_heads
        self.q = nn.Linear(d_q, d_out)
        self.k = nn.Linear(d_kv, d_out)
        self.v = nn.Linear(d_kv, d_out)
        self.o = nn.Linear(d_out, d_out)
        self.ln1 = nn.LayerNorm(d_out)
        self.ln2 = nn.LayerNorm(d_out)
        self.ff = nn.Sequential(nn.Linear(d_out, d_out), nn.GELU(), nn.Linear(d_out, d_out))

    def forward(self, Q, KV, mask_kv: Optional[torch.Tensor] = None):
        B, n_q, _ = Q.shape
        n_kv = KV.shape[1]
        q = self.q(Q).view(B, n_q, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k(KV).view(B, n_kv, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v(KV).view(B, n_kv, self.num_heads, self.d_head).transpose(1, 2)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.d_head ** 0.5)
        if mask_kv is not None:
            scores = scores.masked_fill(~mask_kv[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        # Guard against rows with no valid keys (NaN from softmax over all -inf).
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, n_q, -1)
        h = self.ln1(Q + self.o(out)) if Q.shape[-1] == out.shape[-1] else self.ln1(self.o(out))
        return self.ln2(h + self.ff(h))


class ISAB(nn.Module):
    def __init__(self, d_model: int, m_inducing: int, num_heads: int = 4):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m_inducing, d_model) * 0.02)
        self.mab1 = MaskedMAB(d_model, d_model, d_model, num_heads)
        self.mab2 = MaskedMAB(d_model, d_model, d_model, num_heads)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B = X.shape[0]
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X, mask_kv=mask)
        return self.mab2(X, H, mask_kv=None)


class PMA(nn.Module):
    def __init__(self, d_model: int, k_seeds: int, num_heads: int = 4):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k_seeds, d_model) * 0.02)
        self.mab = MaskedMAB(d_model, d_model, d_model, num_heads)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B = X.shape[0]
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X, mask_kv=mask)


class SetTransformerPool(nn.Module):
    def __init__(self, d_model: int = 128, m_inducing: int = 16, k_seeds: int = 4,
                 num_heads: int = 4, n_isab: int = 2):
        super().__init__()
        self.isabs = nn.ModuleList([ISAB(d_model, m_inducing, num_heads) for _ in range(n_isab)])
        self.pma = PMA(d_model, k_seeds, num_heads)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for isab in self.isabs:
            h = isab(h, mask)
        return self.pma(h, mask)


# ---------------------------------------------------------------------------
# 6. SmFRETEncoder (top-level)
# ---------------------------------------------------------------------------

class SmFRETEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        n_ssm: int = 4,
        m_inducing: int = 16,
        k_seeds: int = 4,
        d_z: int = 64,
        K: int = 8,
        use_next_photon: bool = True,
        n_channels: int = 2,
        chunk_size: int = 64,
        alpha_init_range_ms=(1e-3, 1e2),
    ):
        super().__init__()
        self.tok = PhotonTokenizer(d_model, n_channels)
        self.conv = LocalConvStack(d_model)
        self.ssm_blocks = nn.ModuleList(
            [FastSSMBlock(d_model, d_state,
                          chunk_size=chunk_size,
                          alpha_init_range_ms=alpha_init_range_ms)
             for _ in range(n_ssm)]
        )
        self.next_ph = NextPhotonHead(d_model, K=K, n_channels=n_channels) if use_next_photon else None
        self.pool = SetTransformerPool(d_model, m_inducing, k_seeds)
        self.proj = nn.Sequential(
            nn.Linear(k_seeds * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_z),
        )

    def forward(self, log_dt: torch.Tensor, channel: torch.Tensor,
                dt: torch.Tensor, mask: torch.Tensor):
        h = self.tok(log_dt, channel)
        h = self.conv(h)
        for blk in self.ssm_blocks:
            h = blk(h, dt, mask)

        aux = {}
        if self.next_ph is not None:
            aux["next"] = self.next_ph(h)

        z_set = self.pool(h, mask)
        z = self.proj(z_set.flatten(1))
        return z, aux


# ---------------------------------------------------------------------------
# Auxiliary loss helpers
# ---------------------------------------------------------------------------

def lognorm_mix_log_prob(log_dt: torch.Tensor,
                         log_w: torch.Tensor,
                         mu: torch.Tensor,
                         log_s: torch.Tensor) -> torch.Tensor:
    y = log_dt.unsqueeze(-1)
    inv_s = (-log_s).exp()
    log_p_y = -log_s - 0.5 * math.log(2 * math.pi) - 0.5 * ((y - mu) * inv_s) ** 2
    log_p_x = log_p_y - y
    return torch.logsumexp(log_w + log_p_x, dim=-1)


def next_photon_loss(next_aux: dict,
                     log_dt: torch.Tensor,
                     channel: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    log_w = next_aux["log_w"][:, :-1]
    mu = next_aux["mu"][:, :-1]
    log_s = next_aux["log_s"][:, :-1]
    ch_lg = next_aux["ch_logits"][:, :-1]

    log_dt_next = log_dt[:, 1:]
    channel_next = channel[:, 1:]
    mask_next = mask[:, 1:] & mask[:, :-1]

    dt_lp = lognorm_mix_log_prob(log_dt_next, log_w, mu, log_s)
    ch_lp = -F.cross_entropy(ch_lg.transpose(1, 2), channel_next, reduction="none")

    nll = -(dt_lp + ch_lp)
    return (nll * mask_next).sum() / mask_next.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Adapter: matches the project's `forward(x: (B, N, 4)) -> (B, d_z)` contract
# ---------------------------------------------------------------------------

class SmFRETEncoderAdapter(nn.Module):
    """Wraps `SmFRETEncoder` for the SBI pipeline.

    Input `x` is `(B, N, 4)` with channels:
        x[..., 0] = log(Δt + 1e-12), Δt in ms (unstandardized)
        x[..., 1] = G one-hot (donor)
        x[..., 2] = R one-hot (acceptor)
        x[..., 3] = mask, 1.0 for real photons else 0.0

    Auto-calibrates `log_dt_mean`, `log_dt_std`, `dt_median` from the first
    batch seen (or via an explicit `calibrate(x)` call). Exposes
    `last_aux_loss` and `aux_loss_weight` so the training loop's aux-loss
    hook adds the next-photon NLL to the NPE gradient automatically.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        n_ssm: int = 4,
        m_inducing: int = 16,
        k_seeds: int = 4,
        d_z: int = 64,
        K: int = 8,
        use_next_photon: bool = True,
        aux_loss_weight: float = 0.5,
        forward_chunk_size: Optional[int] = None,
        chunk_size: int = 64,
        alpha_init_range_ms=(1e-3, 1e2),
    ):
        super().__init__()
        self.core = SmFRETEncoder(
            d_model=d_model,
            d_state=d_state,
            n_ssm=n_ssm,
            m_inducing=m_inducing,
            k_seeds=k_seeds,
            d_z=d_z,
            K=K,
            use_next_photon=use_next_photon,
            chunk_size=chunk_size,
            alpha_init_range_ms=alpha_init_range_ms,
        )
        self.aux_loss_weight = float(aux_loss_weight) if use_next_photon else 0.0
        self.last_aux_loss: Optional[torch.Tensor] = None
        # Micro-batch over the leading dim when set. Bounds peak activations
        # during the shape-probing forward (5000 reshaped samples) inside
        # `neural_posterior(theta_init, x_init)` — that runs on CPU before
        # .to(device), and the FastSSMBlock MLP intermediate at 4·d_model
        # can OOM the system without bucketing.
        self.forward_chunk_size = forward_chunk_size

        self.register_buffer("log_dt_mean", torch.zeros(()))
        self.register_buffer("log_dt_std", torch.ones(()))
        self.register_buffer("dt_median", torch.ones(()))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> None:
        """Set standardization buffers from a batch. Idempotent overwrite."""
        log_ipt = x[..., 0].detach()
        mask = x[..., 3].detach() > 0.5
        valid = log_ipt[mask]
        if valid.numel() < 2:
            # Degenerate batch — leave defaults but mark calibrated so we
            # don't keep retrying on every forward.
            self.calibrated.fill_(True)
            return
        mean = valid.mean()
        std = valid.std().clamp(min=1e-6)
        median = valid.exp().median().clamp(min=1e-12)
        self.log_dt_mean.copy_(mean)
        self.log_dt_std.copy_(std)
        self.dt_median.copy_(median)
        self.calibrated.fill_(True)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the core encoder on a single (B, N, 4) tensor.

        Stashes the next-photon NLL on `self.last_aux_loss` (or None).
        """
        log_ipt_raw = x[..., 0]
        # x[..., 1] is the G one-hot; column 2 is R one-hot. Channel index:
        # 0 = donor (G), 1 = acceptor (R).
        channel = x[..., 2].long()
        mask_bool = x[..., 3] > 0.5

        log_dt_std = (log_ipt_raw - self.log_dt_mean) / self.log_dt_std.clamp(min=1e-6)
        dt_o1 = (log_ipt_raw.exp() / self.dt_median.clamp(min=1e-12)).clamp(1e-3, 1e3)

        z, aux = self.core(log_dt_std, channel, dt_o1, mask_bool)

        if self.training and self.aux_loss_weight > 0 and "next" in aux:
            self.last_aux_loss = next_photon_loss(
                aux["next"], log_ipt_raw, channel, mask_bool
            )
        else:
            self.last_aux_loss = None

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calibrate once, using the full batch (better stats than from the
        # first micro-batch only) — must precede any chunked dispatch.
        if not self.calibrated:
            self.calibrate(x)

        B = x.shape[0]
        chunk = self.forward_chunk_size
        if chunk is None or chunk <= 0 or B <= chunk:
            return self._forward_single(x)

        # Micro-batch over the leading dim. The encoder has no cross-sample
        # mixing (the Set Transformer pool acts inside each sample), so
        # chunked output is numerically identical to the unchunked output.
        z_chunks = []
        # (loss_tensor, n_in_chunk) pairs; we weight by chunk size — close
        # to the exact NLL (which weights by valid-photon count) when chunks
        # have similar fill rates, and good enough as a per-step regularizer.
        loss_terms = []
        for i in range(0, B, chunk):
            xc = x[i : i + chunk]
            z = self._forward_single(xc)
            z_chunks.append(z)
            if self.last_aux_loss is not None:
                loss_terms.append((self.last_aux_loss, xc.shape[0]))

        z_out = torch.cat(z_chunks, dim=0)
        if loss_terms:
            total_n = float(sum(n for _, n in loss_terms))
            self.last_aux_loss = sum(L * (n / total_n) for L, n in loss_terms)
        else:
            self.last_aux_loss = None
        return z_out


# ---------------------------------------------------------------------------
# Factory used by `build_temporal_encoder()`
# ---------------------------------------------------------------------------

def build_smfret_photon_encoder(cfg_embedding, transition_dim: int) -> SmFRETEncoderAdapter:
    if transition_dim != 4:
        raise ValueError(
            f"smfret_photon encoder expects transition_dim=4 (the pipeline's "
            f"canonical channel count: log_dt, G_oh, R_oh, mask); got "
            f"{transition_dim}. Set-mode (num_traces_per_set > 1) is handled "
            f"by the SetEmbedding wrapper around this per-trace encoder and "
            f"still passes transition_dim=4 to the inner encoder."
        )
    raw_chunk = getattr(cfg_embedding, "forward_chunk_size", None)
    forward_chunk_size = int(raw_chunk) if raw_chunk not in (None, 0) else None
    alpha_range = getattr(cfg_embedding, "alpha_init_range_ms", (1e-3, 1e2))
    if not isinstance(alpha_range, tuple):
        alpha_range = tuple(float(v) for v in alpha_range)
    return SmFRETEncoderAdapter(
        d_model=int(getattr(cfg_embedding, "d_model", 128)),
        d_state=int(getattr(cfg_embedding, "d_state", 64)),
        n_ssm=int(getattr(cfg_embedding, "n_ssm", 4)),
        m_inducing=int(getattr(cfg_embedding, "m_inducing", 16)),
        k_seeds=int(getattr(cfg_embedding, "k_seeds", 4)),
        d_z=int(getattr(cfg_embedding, "output_dim", 64)),
        K=int(getattr(cfg_embedding, "K", 8)),
        use_next_photon=bool(getattr(cfg_embedding, "use_next_photon", True)),
        aux_loss_weight=float(getattr(cfg_embedding, "aux_loss_weight", 0.5)),
        forward_chunk_size=forward_chunk_size,
        chunk_size=int(getattr(cfg_embedding, "chunk_size", 64)),
        alpha_init_range_ms=alpha_range,
    )
