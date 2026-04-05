import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from math import sqrt
from einops.layers.torch import Rearrange


class LRU(nn.Module):
    """Single Linear Recurrent Unit with diagonal complex-valued state dynamics."""

    def __init__(self, input_dim, state_dim, r_min, r_max, phase_max, bidirectional):
        super().__init__()
        self.state_dim = state_dim
        self.bidirectional = bidirectional
        N = state_dim * 2 if bidirectional else state_dim

        # Eigenvalue magnitudes (log-parameterised for stability, see §3.2 Orvieto 2023)
        r_init = torch.rand(N) * (r_max ** 2 - r_min ** 2) + r_min ** 2
        self.log_nu    = nn.Parameter(torch.log(-0.5 * torch.log(r_init)))
        self.log_theta = nn.Parameter(torch.log(phase_max * torch.rand(N)))

        # Glorot-initialised complex projection matrices
        def _cparam(rows, cols):
            scale = 1.0 / sqrt(2 * cols)
            return nn.Parameter(torch.complex(
                torch.randn(rows, cols) * scale,
                torch.randn(rows, cols) * scale,
            ))
        self.B = _cparam(N, input_dim)
        self.C = _cparam(input_dim, N)
        self.D = nn.Parameter(torch.randn(input_dim))

        # Input normalisation factor γ = sqrt(1 - |λ|²)
        self.log_gamma = nn.Parameter(
            torch.log(torch.sqrt(1 - torch.exp(-torch.exp(self.log_nu)) ** 2))
        )

    @property
    def lambda_complex(self):
        return torch.exp(-torch.exp(self.log_nu) + 1j * torch.exp(self.log_theta))

    def forward(self, x):
        # x: [batch, seq, input_dim]
        gamma = torch.exp(self.log_gamma)
        B_norm = self.B * gamma.unsqueeze(-1)          # normalised input matrix
        u = x.to(dtype=B_norm.dtype) @ B_norm.T        # [batch, seq, N]

        if self.bidirectional:
            # Second half of state sees a time-reversed input stream
            u = torch.cat([
                u[..., :self.state_dim],
                torch.flip(u[..., self.state_dim:], dims=[1]),
            ], dim=-1)

        # Sequential recurrence  x_t = λ · x_{t-1} + u_t
        lam = self.lambda_complex
        x_t = torch.zeros(x.size(0), u.size(-1), dtype=u.dtype, device=x.device)
        states = []
        for u_t in u.unbind(dim=1):
            x_t = lam * x_t + u_t
            states.append(x_t)
        states = torch.stack(states, dim=1)            # [batch, seq, N]

        if self.bidirectional:
            # Restore chronological order for the backward direction
            states = torch.cat([
                states[..., :self.state_dim],
                torch.flip(states[..., self.state_dim:], dims=[1]),
            ], dim=-1)

        return (states @ self.C.mT).real + x * self.D


class LRUBlock(nn.Module):
    """LRU wrapped with optional normalisation, GELU, gating, dropout, and a skip connection."""

    def __init__(self, hidden_dim, state_dim, r_min, r_max, phase_max,
                 bidirectional, dropout, apply_input_normalization):
        super().__init__()
        self.norm    = nn.LayerNorm(hidden_dim) if apply_input_normalization else nn.Identity()
        self.lru     = LRU(hidden_dim, state_dim, r_min, r_max, phase_max, bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.gate    = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        y = self.lru(self.norm(x))
        y = F.gelu(y)
        y = self.dropout(y)
        y = y * torch.sigmoid(self.gate(y))   # state-mixing gate (Smith et al. 2023)
        y = self.dropout(y)
        return y + x                          # skip connection


class LRUEmbedding(nn.Module):
    """Picklable LRU-based embedding for sequential data (e.g. smFRET traces).

    Differences from sbi's LRUEmbedding:
      - ``aggregate_fcn`` is stored as a string, not a lambda → fully picklable.
      - Scan mode removed (uses unstable private PyTorch APIs).
    """

    def __init__(self, input_dim, output_dim, state_dim=20, hidden_dim=40,
                 num_blocks=2, r_min=0.0, r_max=1.0, phase_max=2 * math.pi,
                 bidirectional=True, dropout=0.0,
                 apply_input_normalization=False, aggregate_fcn="mean"):
        super().__init__()
        assert num_blocks >= 1
        assert aggregate_fcn in ("mean", "last_step"), \
            f"aggregate_fcn must be 'mean' or 'last_step', got '{aggregate_fcn}'"
        self.aggregate_fcn = aggregate_fcn   # stored as plain string — picklable

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.lru_blocks  = nn.Sequential(*[
            LRUBlock(hidden_dim, state_dim, r_min, r_max, phase_max,
                     bidirectional, dropout, apply_input_normalization)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq, input_dim]
        x = self.input_layer(x)
        x = self.lru_blocks(x)
        x = x[:, -1] if self.aggregate_fcn == "last_step" else x.mean(dim=1)
        return self.output_layer(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        Uses asymmetric padding to support both odd and even kernel sizes.
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8, padding_mode='reflect'):
        super().__init__()

        # Asymmetric padding for 'same' output size (works for both odd and even kernels)
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size // 2
        self.padding_mode = padding_mode

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=0),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return self.block(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)
    
class StatsPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # mean + std + min + max + 3 quantiles = 7
        self.proj = nn.Sequential(
            nn.Linear(dim * 7, dim),
            nn.Mish(),
        )

    def forward(self, x):
        # x: [batch, channels, time]
        stats = [
            x.mean(dim=-1),
            x.std(dim=-1),
            x.min(dim=-1).values,
            x.max(dim=-1).values,
            x.quantile(0.25, dim=-1),
            x.quantile(0.50, dim=-1),
            x.quantile(0.75, dim=-1),
        ]
        return self.proj(torch.cat(stats, dim=-1))

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, kernel_size=5, padding_mode='reflect'):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, padding_mode=padding_mode),
            Conv1dBlock(out_channels, out_channels, kernel_size, padding_mode=padding_mode),
        ])

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x):
        '''
            x : [ batch_size x inp_channels x horizon ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class TemporalCNN(nn.Module):
    def __init__(
        self,
        transition_dim,
        dim=32,
        dim_mults=(1, 2, 4),
        attention=True,
        padding_mode='reflect',
        kernel_size=3,
        downsample=0,
        output_dim=1
    ):
        super().__init__()

        self.downsample = nn.Conv1d(transition_dim, transition_dim, kernel_size=downsample, stride=downsample) if downsample > 0 else nn.Identity()
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {dims}')
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size = kernel_size, padding_mode=padding_mode),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size = kernel_size, padding_mode=padding_mode),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        print(f'[ models/temporal ] Mid dimension: {mid_dim}')
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, kernel_size = kernel_size, padding_mode=padding_mode)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, kernel_size = kernel_size, padding_mode=padding_mode)
        self.pool = StatsPool(mid_dim)
        self.proj_out = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        '''
            x : [ batch x horizon x transition ]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')
        x = self.downsample(x)

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x)
            x = resnet2(x)
            x = attn(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        x = self.pool(x)
        x = self.proj_out(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise RuntimeError(
                f"PositionalEncoding max_len={self.pe.size(1)} is smaller than sequence length={seq_len}. "
                "This should not happen when max_len is inferred from training data."
            )

        pe = self.pe
        if pe.device != x.device or pe.dtype != x.dtype:
            pe = pe.to(device=x.device, dtype=x.dtype)

        x = x + pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        transition_dim,
        dim=32,
        kernel_sizes=(4, 4),
        stride=2,
        num_heads=4,
        depth=4,
        mlp_dim=128,
        dropout=0.1,
        max_len=20000,
        output_dim=128,
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=transition_dim if i == 0 else dim,
                out_channels=dim,
                kernel_size=ks,
                stride=stride,
                padding=ks // 2,
                padding_mode='reflect',
            )
            for i, ks in enumerate(kernel_sizes)
        ])

        self.downsampled_len = max_len // (stride ** len(kernel_sizes)) + 2
        self.pos_embedding = PositionalEncoding(dim, dropout=dropout, max_len=self.downsampled_len)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        self.to_out = nn.Linear(dim, output_dim)

    def forward(self, x):
        # x: [batch, horizon, transition]
        x = einops.rearrange(x, 'b h t -> b t h')
        for conv in self.conv_layers:
            x = F.gelu(conv(x))
        x = einops.rearrange(x, 'b t h -> b h t')
        x = self.pos_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.to_out(x)
        return x


def _resolve_transformer_kernel_sizes(tr_cfg):
    if hasattr(tr_cfg, "kernel_sizes"):
        kernel_sizes = getattr(tr_cfg, "kernel_sizes")
        if isinstance(kernel_sizes, int):
            return (int(kernel_sizes), int(kernel_sizes))
        return tuple(int(kernel_size) for kernel_size in kernel_sizes)

    # Backward-compatible fallback for older configs.
    if hasattr(tr_cfg, "kernel_size"):
        kernel_size = int(getattr(tr_cfg, "kernel_size"))
        return (kernel_size, kernel_size)

    return (4, 4)


def build_temporal_encoder(cfg_embedding, transition_dim, max_len=None):
    encoder_name = str(getattr(cfg_embedding, "name", "cnn")).lower()

    if encoder_name in {"cnn", "temporalcnn", "temporal_cnn"}:
        cnn_cfg = getattr(cfg_embedding, "cnn", cfg_embedding)
        return TemporalCNN(
            transition_dim=transition_dim,
            dim=int(getattr(cnn_cfg, "dim", 32)),
            dim_mults=tuple(getattr(cnn_cfg, "dim_mults", [1, 2, 4])),
            attention=bool(getattr(cnn_cfg, "attention", True)),
            padding_mode=str(getattr(cnn_cfg, "padding_mode", "reflect")),
            kernel_size=int(getattr(cnn_cfg, "kernel_size", 3)),
            downsample=int(getattr(cnn_cfg, "downsample", 0)),
            output_dim=int(getattr(cnn_cfg, "output_dim", 128)),
        )

    if encoder_name in {"transformer", "temporaltransformer", "temporal_transformer"}:
        tr_cfg = getattr(cfg_embedding, "transformer", cfg_embedding)

        return TemporalTransformer(
            transition_dim=transition_dim,
            dim=int(getattr(tr_cfg, "dim", 32)),
            kernel_sizes=_resolve_transformer_kernel_sizes(tr_cfg),
            stride=int(getattr(tr_cfg, "stride", 2)),
            num_heads=int(getattr(tr_cfg, "num_heads", 4)),
            depth=int(getattr(tr_cfg, "depth", 4)),
            mlp_dim=int(getattr(tr_cfg, "mlp_dim", 128)),
            dropout=float(getattr(tr_cfg, "dropout", 0.1)),
            max_len=max_len,
            output_dim=int(getattr(tr_cfg, "output_dim", 128)),
        )
    
    if encoder_name in {"LRU", "lru"}:
        return LRUEmbedding(
            input_dim=transition_dim,
            output_dim=int(getattr(cfg_embedding, "output_dim", 128)),
            state_dim=int(getattr(cfg_embedding, "state_dim", 20)),
            hidden_dim=int(getattr(cfg_embedding, "hidden_dim", 40)),
            num_blocks=int(getattr(cfg_embedding, "num_blocks", 2)),
            r_min=float(getattr(cfg_embedding, "r_min", 0.0)),
            r_max=float(getattr(cfg_embedding, "r_max", 1.0)),
            phase_max=float(getattr(cfg_embedding, "phase_max", 2 * math.pi)),
            bidirectional=True,
            dropout=float(getattr(cfg_embedding, "dropout", 0.0)),
            apply_input_normalization=False,
            aggregate_fcn="mean",
        )
    

    raise ValueError(
        f"Unknown embedding.name='{encoder_name}'. Use one of: cnn, transformer."
    )
