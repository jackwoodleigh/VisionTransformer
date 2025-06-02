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
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = self.pixel_unshuffle(x)

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.norm(x)
        x = self.linear(x)

        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

# input/output B, C, H, W
class PatchUnMerging(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.norm(x)
        x = self.linear(x)

        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.pixel_shuffle(x)

        return x

class WindowedMSA(nn.Module):
    def __init__(self, window_size, dim, levels, num_heads=4, attn_drop=0.0, proj_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.qkv = nn.Linear(dim, 3 * dim)

        self.cross_kv = nn.ModuleList([nn.Linear(dim, dim*2) for _ in range(levels)])
        self.cross_mlp = nn.ModuleList([MLP(dim) for _ in range(levels-1)])
        self.cross_conv = nn.ModuleList([nn.Conv2d(1, 1, 3, 1, 1) for _ in range(levels-1)])

        self.cross_unmerge = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim // (4 ** (levels - i)), 4**i, 1, 1),
            PatchUnMerging(dim=dim, scale=2 ** i)
        ) for i in range(1, levels)])

        self.pe_encoder = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def window_partition(self, x, win):
        B, H, W, C = x.shape
        x = x.view(B, H // win, win, W // win, win, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win * win,  C)
        return x

    def reverse_window_partition(self, x, win, H, W):
        # N, win*win, C
        N, _, C = x.shape
        x = x.view(N, win, win, C)
        x = x.view(-1, H // win, W // win, win, win, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)

    def locally_enhanced_PE(self, x):
        # N, n_heads, win*win, C/n_heads -> N, win*win, C
        x = self.deheadify(x)

        N, win_sq, C = x.shape
        win = self.window_size

        # N, win*win, C -> N, C, win, win
        x = x.view(N, win, win, C).permute(0, 3, 1, 2).contiguous()
        lepe = self.pe_encoder(x)
        lepe = lepe.permute(0, 2, 3, 1).contiguous().view(N, win*win, C)
        return self.headify(lepe)

    def headify(self, *xs):
        outs = []
        for x in xs:
            N, win, C = x.shape
            x = x.view(N, win, self.num_heads, C // self.num_heads)
            outs.append(x.permute(0, 2, 1, 3).contiguous())
        return outs if len(outs) > 1 else outs[0]

    def deheadify(self, *xs):
        outs = []
        for x in xs:
            # N, n_heads, Win, C // n_heads
            N, _, win, _ = x.shape
            outs.append(x.permute(0, 2, 1, 3).contiguous().view(N, win, -1))
        return outs if len(outs) > 1 else outs[0]

    def sub_window_pool(self, x, win):
        # N, Win, C
        N, Win, C = x.shape
        return x.view(N, Win // win, win, C).mean()

    def cross_attn_gates(self, x, q, context):
        # x: B, H, W, C
        # q: B, H, W, C

        sub_cross_win = 8*8
        for i in range(len(context)):
            # B, C, H, W -> N, Win, C
            B, H, W, C = x.shape
            x = self.cross_mlp[i](x)

            # B, H*W, 1
            x_s = x.mean(dim=2, keepdim=True)

            # B, H*W, C -> N2, CrossWin, C
            x = self.window_partition(x, 64)
            N, win, _ = x.shape

            # N2, H*W / cross_win, cross_win, C -> N2, H*W / cross_win, C
            x_c = x.view(N, win // sub_cross_win, sub_cross_win, C).mean(dim=2)

            k, v = self.cross_kv[i](x_c).chunk(2, dim=-1)
            k, v = self.headify(k, v)
            

            z = z_c * z_s
            z = self.cross_conv[i](z)





            z = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)

            # N, n_heads, Win, C/n_heads -> B, H, W, C
            z = self.deheadify(z)
            z = self.reverse_window_partition(z, self.window_size, H, W).permute(0, 3, 1, 2).contiguous()

            x += z

        return x

    def forward(self, x, context=None):
        B, H, W, C = x.shape

        # B, H, W, C -> N, Win, C -> N, n_heads, Win, C // n_heads
        z = self.window_partition(x, self.window_size)
        qkv = self.qkv(z)
        qkv = self.headify(qkv)
        q, k, v = qkv.chunk(3, dim=-1)

        lepe = self.locally_enhanced_PE(v)

        z = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop) + lepe

        # N, n_heads, W, C // n_heads -> B, H, W, C
        z = self.deheadify(z)
        z = self.reverse_window_partition(z, self.window_size, H, W)
        q = self.reverse_window_partition(q, self.window_size, H, W)

        z = self.cross_attn_gates(x, q, context)

        z = self.out_proj(z)
        z = self.proj_drop(z)
        return z


class ViTBlock(nn.Module):
    def __init__(self, window_size, dim, levels, n_heads=4, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.wmsa = WindowedMSA(dim=dim, window_size=window_size, levels=levels, num_heads=n_heads)
        self.mlp = MLP(dim, ffn_scale=ffn_scale)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(drop_path)

    def forward(self, x, context=None):
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.drop_path(self.wmsa(self.ln1(x), context)) + x
        x = self.drop_path(self.mlp(self.ln2(x))) + x

        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class MSFBlock_3(nn.Module):
    def __init__(self, levels, window_size, dim, level_dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels

        self.downsample = nn.ModuleList([nn.Conv2d(dim // (4 ** i), dim // (4 ** (i + 1)), 1, 1) for i in range(levels - 1)])

        self.merge = nn.ModuleList([
            nn.Sequential(
                PatchMerging(dim=dim, scale=2 ** i),
                nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
            ) for i in range(1, levels)
        ])

        self.fuse = ViTBlock(window_size, dim, levels, n_heads=n_heads_fuse, ffn_scale=ffn_scale, drop_path=drop_path)
        self.out_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        context_maps = []
        z = x
        for i in range(self.levels - 1):
            z=self.downsample[i](z)
            context_maps.append(self.merge[i](z))

        return self.rezero * self.out_conv(self.fuse(x, context_maps)) + x

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

