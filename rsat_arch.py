import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.utils.checkpoint import checkpoint

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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

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

    def locally_enhanced_PE(self, x):
        # Input shape: N, Win^2, C
        # Output shape: N, Win^2, C
        N, win_sq, C = x.shape
        x = x.view(N, self.win, self.win, C).permute(0, 3, 1, 2).contiguous()
        return self.pe_encoder(x).permute(0, 2, 3, 1).contiguous().view(N, self.win * self.win, C)

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
        self.q_kernel_size = 4

        self.spatial_proj = LastChannelConv2D(dim, 1, 3, 1, 1)
        #self.channel_proj = PatchMerging(dim, self.q_kernel_size)
        self.channel_proj = LastChannelConv2D(dim, dim, self.q_kernel_size, self.q_kernel_size, groups=dim)

        self.pe_encoder = LastChannelConv2D(dim, dim, 3, 1, 1, groups=dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.q_proj = nn.Linear(dim, dim)

        self.out_conv = LastChannelConv2D(dim, dim, 3, 1, 1)

        self.sigmoid = nn.Sigmoid()
        self.rezero = nn.Parameter(torch.zeros(1))

    def overlapping_windows(self, x):
        # Input shape: B, H, W, C
        # Output shape: B, H/win, W/win, 9*win^2, C

        B, H, W, C = x.shape
        win = self.win
        kernel_size_unfold = 3 * win
        stride_unfold = win
        padding_unfold = win
        H_win = (H + 2 * padding_unfold - kernel_size_unfold) // stride_unfold + 1
        W_win = (W + 2 * padding_unfold - kernel_size_unfold) // stride_unfold + 1
        L = H_win * W_win

        x = x.permute(0, 3, 1, 2)
        x_padded = F.pad(x, (padding_unfold, padding_unfold, padding_unfold, padding_unfold))
        sB_pad, sC_pad, sH_pad, sW_pad = x_padded.stride()
        shape_as_strided = (B, C, H_win, W_win, kernel_size_unfold, kernel_size_unfold)

        strides_as_strided = (sB_pad, sC_pad, stride_unfold * sH_pad, stride_unfold * sW_pad, sH_pad, sW_pad)

        # B, C, H/win, W/win, K * win, K * win
        x_strided_patches = x_padded.as_strided(shape_as_strided, strides_as_strided)

        # B, H/win, W/win, K*win, K*win, C
        x = x_strided_patches.permute(0, 2, 3, 4, 5, 1)

        # B, H/win * W/win, K^2 * win^2, C
        x = x.reshape(B, L, kernel_size_unfold * kernel_size_unfold, C)

        return x.contiguous()

    def overlapping_windows_2(self, x):
        # Input shape: B, H, W, C
        # Output shape: B, H/win, W/win, 9*win^2, C

        B, H, W, C = x.shape
        H_win, W_win = H // self.win, W // self.win

        x = x.permute(0, 3, 1, 2)
        x = F.unfold(x, kernel_size=(3 * self.win, 3 * self.win), stride=self.win, padding=self.win)

        # B, C * K^2 * win^2, H/win * W/win -> B, C, K^2 * win^2, H/win * W/win
        x = x.view(B, C, 9 * self.win**2, H_win * W_win)

        # B, C, K^2 * win^2, H/win * W/win -> B, H/win * W/win, K^2 * win^2, C
        x = x.permute(0, 3, 2, 1).contiguous()
        return x

    def forward(self, x_wsa, x_original):
        # x_wsa: B, H, W, C
        # x_original: B * H/win * W/win, win^2, C

        # TODO make kernel for overlapping windows

        B, H, W, C = x_wsa.shape
        K = self.q_kernel_size
        H_K, W_K = H // K, W // K

        # Find spatial summaries by reducing channel size to 1
        # B, H, W, C -> B, H, W, 1 -> B, H/K * W/K, K**2, 1
        x_s = self.window_partition(self.spatial_proj(x_wsa), K, keep_batch=True)

        # Combine spatial regions to find channel summaries
        # B, H, W, C -> B, H/K, W/K, C -> B, H/win, W/win, (win/K)**2, C -> B, H/win * W/win, (win/K)**2, C
        x_c = self.window_partition(self.channel_proj(x_wsa), self.win//K, keep_batch=True).view(B, (H//self.win) * (W//self.win), (self.win//K)**2, C)

        q = self.q_proj(x_c)
        kv = self.kv_proj(x_original)
        kv[..., C:] = kv[..., C:] + self.pe_encoder(kv[..., C:])

        # Produce overlapping window
        # B, (H/win) * (W/win), K^2 * win^2, C
        k, v = self.overlapping_windows_2(kv).chunk(2, dim=-1)

        # Cross attention using the channel summary x_c and neighboring windows with input tensor
        # B, H/win * W/win, (win/K)**2, C -> B, H/K, W/K, C -> B, H/K * W/K, 1, C
        x_cross = nn.functional.scaled_dot_product_attention(q, k, v)
        x_cross = self.reverse_window_partition(x_cross, self.win//K, H_K, W_K, with_batch=True).view(B, H_K * W_K, 1, C)

        # Combine pixel and channel summaries for context modulation
        # B, H/K * W/K, K**2, C -> B, H, W, C
        x_cross_context = self.reverse_window_partition(x_s * x_cross, K, H, W, with_batch=True)

        x_cross_context = self.out_conv(x_cross_context)

        return x_wsa + self.rezero * x_cross_context


class WindowedMSA(WindowedAttention):
    def __init__(self, window_size, dim, num_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__(window_size, dim, num_heads)
        self.win = window_size
        self.num_heads = num_heads
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, 3 * dim)
        self.context_ca = WindowedContextCA(window_size, dim)

        self.pe_encoder = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        # B, H, W, C -> N, win^2, C
        z = self.window_partition(x, self.win)
        qkv = self.qkv(z)
        q, k, v = qkv.chunk(3, dim=-1)

        lepe = self.locally_enhanced_PE(v)
        q, k, v, lepe = self.headify(q, k, v, lepe)

        z = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop) + lepe

        # N, n_heads, W, C // n_heads -> B, H, W, C
        z = self.reverse_window_partition(self.deheadify(z), self.win, H, W)

        z = self.context_ca(z, x)

        z = self.out_proj(z)
        z = self.proj_drop(z)
        return z


class ViTBlock(nn.Module):
    def __init__(self, window_size, dim, levels, n_heads=4, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.wmsa = WindowedMSA(dim=dim, window_size=window_size, num_heads=n_heads)
        self.mlp = MLP(dim, ffn_scale=ffn_scale)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(drop_path)

    def forward(self, x):
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.drop_path(self.wmsa(self.ln1(x))) + x
        x = self.drop_path(self.mlp(self.ln2(x))) + x

        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class MSFBlock_3(nn.Module):
    def __init__(self, levels, window_size, dim, level_dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.fuse = ViTBlock(window_size, dim, levels, n_heads=n_heads_fuse, ffn_scale=ffn_scale, drop_path=drop_path)
        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.rezero * self.fuse(x)

class ResidualBlock(nn.Module):
    def __init__(self, n_sub_blocks, levels, window_size, dim, level_dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.):
        super().__init__()
        self.layers = nn.Sequential(*[MSFBlock_3(
            levels=levels,
            level_dim=level_dim,
            dim=dim,
            window_size=window_size,
            n_heads=n_heads,
            n_heads_fuse=n_heads_fuse,
            ffn_scale=ffn_scale,
            drop_path=drop_path) for _ in range(n_sub_blocks)])

        self.out_layer = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        return self.out_layer(self.layers(x)) + x

class DenseResidualBlock(nn.Module):
    def __init__(self, n_sub_blocks, levels, window_size, dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.n_sub_blocks = n_sub_blocks
        self.layers = nn.ModuleList([MSFBlock_3(
            levels=levels,
            dim=dim,
            window_size=window_size,
            n_heads=n_heads,
            n_heads_fuse=n_heads_fuse,
            ffn_scale=ffn_scale,
            drop_path=drop_path) for _ in range(n_sub_blocks)])

        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.2)) for _ in range(self.n_sub_blocks)])

        self.fuse = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim*(i+1), dim, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2)) for i in range(self.n_sub_blocks)
        ])

        self.out_layer = nn.Sequential(
            MSFBlock_3(
                levels=levels,
                dim=dim,
                window_size=window_size,
                n_heads=n_heads,
                n_heads_fuse=n_heads_fuse,
                ffn_scale=ffn_scale,
                drop_path=drop_path),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        residuals_list = [x]

        for i in range(self.n_sub_blocks):
            x = self.layers[i](x)
            residual = self.fuse[i](torch.cat(residuals_list, dim=1))
            x += self.alpha[i] * residual
            residuals_list.append(x)

        return self.out_layer(x)


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

        self.feature_transition = nn.Conv2d(dim, dim, 3, 1, 1)

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

        x = self.feature_transition(self.layers(x)) + x

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

