import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
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
        return x