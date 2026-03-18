import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from einops.layers.torch import Rearrange

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
        output_dim=1
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {dims}')

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
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
            output_dim=int(getattr(cnn_cfg, "output_dim", 128)),
        )

    if encoder_name in {"transformer", "temporaltransformer", "temporal_transformer"}:
        tr_cfg = getattr(cfg_embedding, "transformer", cfg_embedding)

        return TemporalTransformer(
            transition_dim=transition_dim,
            dim=int(getattr(tr_cfg, "dim", 32)),
            kernel_sizes=tuple(getattr(tr_cfg, "kernel_sizes", [4, 4])),
            stride=int(getattr(tr_cfg, "stride", 2)),
            num_heads=int(getattr(tr_cfg, "num_heads", 4)),
            depth=int(getattr(tr_cfg, "depth", 4)),
            mlp_dim=int(getattr(tr_cfg, "mlp_dim", 128)),
            dropout=float(getattr(tr_cfg, "dropout", 0.1)),
            max_len=max_len,
            output_dim=int(getattr(tr_cfg, "output_dim", 128)),
        )

    raise ValueError(
        f"Unknown embedding.name='{encoder_name}'. Use one of: cnn, transformer."
    )