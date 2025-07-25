import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.utils.checkpoint import checkpoint
from toolkit.arch_util import trunc_normal_

# https://arxiv.org/pdf/2409.03516

# https://arxiv.org/pdf/2208.11247v3
# https://arxiv.org/pdf/2404.00722v5
# https://arxiv.org/pdf/2205.04437v3 maybe try out lam


# https://arxiv.org/pdf/2205.04437


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class LastChannelConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

    def forward(self, x):
        # x: B, H, W, C
        return self.conv(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ffn_scale=4, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * ffn_scale)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


# input/output B, C, H, W
class PatchMerging(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(scale)
        self.norm = nn.LayerNorm((scale**2)*dim)
        self.linear = nn.Linear((scale**2)*dim, dim, bias=False)

    def forward(self, x):
        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.pixel_unshuffle(x)

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.norm(x)
        x = self.linear(x)

        return x

class RelativePositionalEncoding(nn.Module):
    def __init__(self, q_size, k_size, num_heads=1):
        super().__init__()
        assert q_size <= k_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * k_size - 1) * (2 * k_size - 1), num_heads))

        coords = torch.stack(torch.meshgrid(torch.arange(k_size), torch.arange(k_size), indexing='ij'), dim=0)

        start = (k_size - q_size) // 2
        end = start + q_size
        coords_q = coords[:, start:end, start:end].reshape(2, -1)
        coords_k = coords.reshape(2, -1)

        relative_coords = coords_q[:, :, None] - coords_k[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() + k_size - 1
        relative_position_index = relative_coords[:, :, 0] * (2 * k_size - 1) + relative_coords[:, :, 1]
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        bias = self.relative_position_bias_table[self.relative_position_index]
        bias = bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        return bias

class WindowedAttention(nn.Module):
    def __init__(self, window_size, dim, num_heads=4):
        super().__init__()
        self.win = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

    def window_partition(self, x, win, keep_batch=False):
        # Input shape: B, H, W, C
        # Output shape: N, win^2, C or B, H/win * W/win, win^2, C
        B, H, W, C = x.shape
        x = x.view(B, H // win, win, W // win, win, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, (H // win) * (W // win), win**2, C) if keep_batch else x.view(B * (H // win) * (W // win), win**2, C)

    def reverse_window_partition(self, x, win, H, W, with_batch=False):
        # Input shape: N, win*win, C or B, H/win * W/win, win^2, C
        # Output shape: B, H, W, C
        if with_batch:
            #x = x.view(-1, win**2, x.shape[-1])
            x = x.reshape(-1, win ** 2, x.shape[-1])
        N, _, C = x.shape
        x = x.view(N, win, win, C).view(-1, H // win, W // win, win, win, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)

    def headify(self, *xs):
        # Input shape: N, Win, C
        # Output shape: N, n_heads, Win, C // n_heads
        outs = []
        for x in xs:
            N, win, C = x.shape
            x = x.view(N, win, self.num_heads, C // self.num_heads)
            outs.append(x.permute(0, 2, 1, 3).contiguous())
        return outs if len(outs) > 1 else outs[0]

    def deheadify(self, *xs):
        # Input shape: N, n_heads, Win, C // n_heads
        # Output shape: N, Win, C
        outs = []
        for x in xs:
            N, _, win, _ = x.shape
            outs.append(x.permute(0, 2, 1, 3).contiguous().view(N, win, -1))
        return outs if len(outs) > 1 else outs[0]

class WindowedContextCA(WindowedAttention):
    def __init__(self, window_size, dim, num_heads=4, attn_drop=0.0, proj_drop=0.):
        super().__init__(window_size, dim, num_heads)
        self.win = window_size
        self.num_heads = num_heads
        self.summary_size = 2
        self.kernal_size = 3
        self.win_s = window_size // self.summary_size

        self.kv_proj = LastChannelConv2D(in_channels=dim, out_channels=2 * dim, kernel_size=5, padding=2, groups=num_heads)
        self.q_proj = nn.Linear(dim, dim)

        self.pe = RelativePositionalEncoding(q_size=self.win, k_size=self.win, num_heads=num_heads)

        self.out_proj = nn.Linear(dim, dim)
        #self.out_conv = LastChannelConv2D(dim, dim, 3, 1, 1, groups=num_heads)
        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, x_wsa, x_original):
        # x_wsa: B, H, W, C
        # x_original: B, H, W, C

        # TODO make kernel for overlapping windows

        B, H, W, C = x_wsa.shape
        S = self.summary_size
        H_S, W_S = H // S, W // S

        # B, H, W, C -> B, H/S, W/S, C -> B * H/win * W/win, (win/S)^2, C
        q = self.window_partition(self.q_proj(x_wsa), self.win)

        # B, H, W, C -> B * H/win * W/win, win^2, C
        k, v = self.window_partition(self.kv_proj(x_original), self.win).chunk(2, dim=-1)

        # B * (H/win) * (W/win), (win/S)^2, C -> B * (H/win) * (W/win), n_heads, K^2 * (win/S)^2, C/n_heads
        q, k, v = self.headify(q, k, v)

        x_cross = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.pe())

        # B * (H/win) * (W/win), n_heads, K^2 * (win/S)^2, C/n_heads ->
        x_cross = self.deheadify(x_cross)

        # B, (H/win) * (W/win), (win/S)^2, C -> B, H/S, W/S, C -> B, H/S, 1, W/S, 1, C
        x_cross = self.reverse_window_partition(x_cross, self.win, H, W) #.view(B, H_S, 1, W_S, 1, C)

        # B, H/S, 1, W/S, 1, C -> B, H/S, S, W/S, S, C -> B, H, W, C
        # x_cross = x_cross.expand(-1, -1,  self.summary_size, -1, self.summary_size,  -1).reshape(B, H, W, C)

        return self.rezero * self.out_proj(x_cross)


class WindowedMSA(WindowedAttention):
    def __init__(self, window_size, dim, num_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__(window_size, dim, num_heads)
        self.win = window_size
        self.num_heads = num_heads
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, 3 * dim)
        self.pe = RelativePositionalEncoding(q_size=self.win, k_size=self.win, num_heads=num_heads)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.context_ca = WindowedContextCA(window_size, dim, num_heads=num_heads)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        # B, H, W, C -> N, win^2, C
        z = self.window_partition(x, self.win)
        qkv = self.qkv(z)
        q, k, v = self.headify(qkv).chunk(3, dim=-1)

        z = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.pe(), dropout_p=self.attn_drop)

        z = self.norm(self.reverse_window_partition(self.out_proj(self.deheadify(z)), self.win, H, W))

        z = z + self.context_ca(z, x)

        z = self.drop(z)
        return z


class ViTBlock(nn.Module):
    def __init__(self, window_size, dim, n_heads=4, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.wmsa = WindowedMSA(dim=dim, window_size=window_size, num_heads=n_heads)
        self.mlp = MLP(dim, ffn_scale=ffn_scale)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(drop_path)
        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.drop_path(self.wmsa(self.ln1(x))) + x
        x = self.drop_path(self.mlp(self.ln2(x))) + x
        return x

class ResidualBlock(nn.Module):
    def __init__(self, n_sub_blocks, levels, window_size, dim, level_dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.):
        super().__init__()
        self.layers = nn.Sequential(*[ViTBlock(
            window_size=window_size,
            dim=dim,
            n_heads=n_heads,
            ffn_scale=ffn_scale,
            drop_path=drop_path) for _ in range(n_sub_blocks)])

        self.out_layer = LastChannelConv2D(dim, dim, 3, 1, 1)

    def forward(self, x):
        return self.out_layer(self.layers(x)) + x


class MSFTransformer(nn.Module):
    def __init__(
            self,
            block_type,
            n_blocks,
            n_sub_blocks,
            levels,
            window_size,
            dim,
            level_dim,
            n_heads,
            n_heads_fuse,
            feature_dim,
            scale_factor,
            ffn_scale=4,
            drop_path=0.1):
        super().__init__()

        self.LHSA_levels = levels
        self.dim = dim
        self.feature_dim = feature_dim
        self.scale_factor = scale_factor
        self.window_size = window_size
        self.levels=levels

        self.feature_extractor = nn.Conv2d(3, dim, 3, 1, 1)
        if block_type != "default":
            self.layers = nn.Sequential(*[
                DenseResidualBlock(
                    n_sub_blocks=n_sub_blocks,
                    levels=levels,
                    dim=dim,
                    window_size=window_size,
                    n_heads=n_heads,
                    n_heads_fuse=n_heads_fuse,
                    ffn_scale=ffn_scale,
                    drop_path=drop_path) for _ in range(n_blocks)])
        else:
            self.layers = nn.Sequential(*[
                ResidualBlock(
                    n_sub_blocks=n_sub_blocks,
                    levels=levels,
                    dim=dim,
                    level_dim=level_dim,
                    window_size=window_size,
                    n_heads=n_heads,
                    n_heads_fuse=n_heads_fuse,
                    ffn_scale=ffn_scale,
                    drop_path=drop_path) for _ in range(n_blocks)])

        self.feature_transition = LastChannelConv2D(dim, dim, 3, 1, 1)

        img_reconstruction = [
            nn.Conv2d(dim, feature_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
        ]
        for i in range(int(np.log2(scale_factor))):
            img_reconstruction.extend([
                nn.Conv2d(feature_dim, feature_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ])
        img_reconstruction.extend([
            nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, 3, 3, 1, 1, )
        ])

        self.img_reconstruction = nn.Sequential(*img_reconstruction)

        self.apply(self.init_weights)

    def padding2(self, x):
        _, _, h, w = x.size()
        scaled_size = self.window_size

        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def padding(self, x):
        _, _, h, w = x.size()

        if self.levels <= 0:
            required_divisor = self.window_size
        else:
            required_divisor = self.window_size * (2 ** (self.levels - 1))
        mod_pad_h = (required_divisor - h % required_divisor) % required_divisor
        mod_pad_w = (required_divisor - w % required_divisor) % required_divisor

        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 1, 3, 16, 16 = x
    def forward(self, x):
        B, C, H, W = x.shape

        x = self.padding(x)

        x = self.feature_extractor(x)
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.feature_transition(self.layers(x)) + x
        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.img_reconstruction(x)

        # crop padding
        return x[:, :, :H*self.scale_factor, :W*self.scale_factor]


if __name__ == '__main__':
    import yaml
    import torch
    from fvcore.nn import parameter_count_table, FlopCountAnalysis

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model = MSFTransformer(
        block_type=config["model"]["block_type"],
        n_blocks=config["model"]["n_blocks"],
        n_sub_blocks=config["model"]["n_sub_blocks"],
        levels=config["model"]["levels"],
        window_size=config["model"]["window_size"],
        dim=config["model"]["dim"],
        level_dim=config["model"]["level_dim"],
        n_heads=config["model"]["n_heads"],
        n_heads_fuse=config["model"]["n_heads_fuse"],
        feature_dim=config["model"]["feature_dim"],
        scale_factor=config["model"]["scale_factor"]
    )

    print(parameter_count_table(model))

    tensor = torch.randn(1, 3, 64, 64)
    flop_count = FlopCountAnalysis(model, tensor)
    flops = flop_count.total()
    print(flops)

