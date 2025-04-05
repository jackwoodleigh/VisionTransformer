import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# https://arxiv.org/pdf/2409.03516

# https://arxiv.org/pdf/2208.11247v3
# https://arxiv.org/pdf/2404.00722v5
# https://arxiv.org/pdf/2205.04437v3 maybe try out lam


# https://arxiv.org/pdf/2205.04437

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ffn_scale=4, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * ffn_scale)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class WindowedMSA(nn.Module):
    def __init__(self, window_size, dim, num_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.qkv = nn.Linear(dim, 3 * dim)
        self.pe_encoder = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def window_partition(self, x, H, W):
        B, N, C = x.shape
        H_sp = W_sp = self.window_size

        x = x.view(B, H, W, C)
        x = x.view(B, H // H_sp, H_sp, W // W_sp, W_sp, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H_sp * W_sp,  C)
        x = x.view(-1, H_sp * W_sp, self.num_heads, C // self.num_heads)
        return x.permute(0, 2, 1, 3).contiguous()

    def reverse_window_partition(self, x, H, W):
        C = x.shape[1] * x.shape[3]
        H_sp = W_sp = self.window_size

        x = x.permute(0, 2, 1, 3).contiguous().view(-1,  H_sp * W_sp, C)
        x = x.view(-1, H_sp, W_sp, C)
        x = x.view(-1, H // H_sp, W // W_sp, H_sp, W_sp, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C).view(-1, H*W, C)

    def locally_enhanced_PE(self, x, H, W):
        B, N, C = x.shape
        H_sp = W_sp = self.window_size

        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, H_sp, W_sp)

        lepe = self.pe_encoder(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, x, H, W):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # B, N, C -> B, n_heads, W_sp, C // n_heads
        q = self.window_partition(q, H, W)
        k = self.window_partition(k, H, W)
        v, lepe = self.locally_enhanced_PE(v, H, W)

        x = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop) + lepe

        # B, n_heads, W_sp, C // n_heads -> B, N, C
        x = self.reverse_window_partition(x, H, W)

        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, window_size, dim, n_heads=4, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.msa = WindowedMSA(dim=dim, window_size=window_size, num_heads=n_heads)
        self.mlp = MLP(dim, ffn_scale=ffn_scale)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(drop_path)

    # Input/Output B,C,H,W
    def forward(self, x):
        B, C, H, W = x.shape

        # B, C, H, W -> B, N, C
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()

        x = self.drop_path(self.msa(self.ln1(x), H, W)) + x
        x = self.drop_path(self.mlp(self.ln2(x))) + x

        # B, N, C -> B, C, H, W
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return x

class MSFBlock(nn.Module):
    def __init__(self, levels, window_size, dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels=levels
        self.level_layer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // (4 ** i), 3, 1, 1),
                nn.PixelUnshuffle(2 ** i),
                ViTBlock(window_size, dim, n_heads=n_heads, ffn_scale=ffn_scale, drop_path=drop_path),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.PixelShuffle(2 ** i)
            ) for i in range(1, levels)
        ])

        total_dim = sum([dim // (4 ** i) for i in range(levels)])
        self.fuse = nn.Sequential(
            ViTBlock(window_size, total_dim, n_heads=n_heads_fuse, ffn_scale=ffn_scale, drop_path=drop_path),
            nn.Conv2d(total_dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        maps = [x]
        for i in range(self.levels-1):
            maps.append(self.level_layer[i](x))
        return self.fuse(torch.cat(maps, dim=1)) + x

# use dense resuidals for this for next version

# combine layers before applying ?

class MSFBlock_2(nn.Module):
    def __init__(self, levels, window_size, dim, level_dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels
        # 1, 64, 64, 64 -> 1, 8, 64, 64 -> 1, 32, 32, 32 -> 1, 8, 64, 64
        self.level_layer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, level_dim // (4 ** i), 3, 1, 1),
                nn.PixelUnshuffle(2 ** i),
                ViTBlock(window_size, level_dim, n_heads=n_heads, ffn_scale=ffn_scale, drop_path=drop_path),
                nn.Conv2d(level_dim, level_dim, 3, 1, 1),
                nn.PixelShuffle(2 ** i)
            ) for i in range(1, levels)
        ])

        total_level_dim = sum([level_dim // (4 ** i) for i in range(1, levels)])
        self.post_level_fuse = nn.Conv2d(total_level_dim, dim//4, 1, 1)

        self.fuse = nn.Sequential(
            ViTBlock(window_size, dim + dim//4, n_heads=n_heads_fuse, ffn_scale=ffn_scale, drop_path=drop_path),
            nn.Conv2d(dim + dim//4, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        maps = []
        for i in range(self.levels-1):
            maps.append(self.level_layer[i](x))

        levels = self.post_level_fuse(torch.cat(maps, dim=1))
        return self.fuse(torch.cat([x, levels], dim=1)) + x


class DenseResidualBlock(nn.Module):
    def __init__(self, n_sub_blocks, levels, window_size, dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.n_sub_blocks = n_sub_blocks
        self.layers = nn.ModuleList([MSFBlock(
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
            MSFBlock(
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


class ResidualBlock(nn.Module):
    def __init__(self, n_sub_blocks, levels, window_size, dim, level_dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.):
        super().__init__()
        self.layers = nn.Sequential(*[MSFBlock_2(
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

class LMLTransformer(nn.Module):
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
        x = (x - 0.5) / 0.25
        B, C, H, W = x.shape

        x = self.padding(x)

        x = self.feature_extractor(x)

        x = self.feature_transition(self.layers(x)) + x

        x = self.img_reconstruction(x)

        x = x * 0.25 + 0.5
        x = torch.clamp(x, min=0, max=1.0)

        # crop padding
        return x[:, :, :H*self.scale_factor, :W*self.scale_factor]


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    print(torch.cuda.is_available())

    x = torch.randn(8, 3, 480, 270, requires_grad=True).to("cuda")  # 1920x1080 output
    model = LMLTransformer(n_blocks=12, levels=4, dim=84, window_size=8, scale_factor=4)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')

    model = model.to('cuda')
    #output = model(x)
    output = checkpoint(model, x, use_reentrant=True)
    print(output.shape)

#vit = VisionTransformer(4, 4, 3, 32, 32)
# t = Transformer(4)
#x = torch.randn(1, 3, 32, 32)
#print(vit(x).shape)
