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

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    # B, H, W, C - input and output
    def forward(self, x):
        x = self.ccm(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, window_size, dim, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.pe_encoder = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.out_proj = nn.Linear(dim, dim)

    # input: -1, window_size^2, C
    def locally_enhanced_pe(self, x, func):
        N, num_heads, W_s, head_dim = x.shape

        # N, num_heads, W_s, head_dim -> N, Win_s, C
        x = x.permute(0, 2, 1, 3).contiguous().view(N, W_s, -1)

        # converting into spatial form: N, window_size^2, C -> B, C, window_size, window_size
        x = x.permute(0, 2, 1).contiguous().view(N, -1, self.window_size, self.window_size)

        # find spatial encodings for each window
        lepe = func(x)

        # converting into back into image with: B, C, H, W -> N, W^2, C
        lepe = lepe.view(N, -1, self.window_size * self.window_size).permute(0, 2, 1).contiguous()

        return lepe

    # 1, 3, 16, 16 = x
    def forward(self, x):
        N, W_s, C = x.shape

        # q,k,v: N, window_size^2, C -> N, num_heads, W_s, head_dim
        qkv = self.qkv(x).reshape(N, W_s, 3*self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q, k, v = qkv.chunk(3, dim=1)

        # attention and apply positional encoding
        lepe = self.locally_enhanced_pe(v, self.pe_encoder)
        x = nn.functional.scaled_dot_product_attention(q, k, v)

        # N, num_heads, W_s, head_dim -> N, Win_s, C
        x = x.permute(0, 2, 1, 3).contiguous().view(N, W_s, C)
        x += lepe

        x = self.out_proj(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, window_size, dim, ffn_scale=1.0):
        super().__init__()
        self.window_size = window_size
        self.sa = SelfAttention(dim=dim, window_size=window_size)
        self.mlp = MLP(dim, int(dim * ffn_scale))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def window_partition(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return x.view(-1, self.window_size * self.window_size, C)

    def reverse_window_partition(self, x, h, w):
        _, _, c = x.shape
        # b = int(x.shape[0] / (w * h / self.window_size / self.window_size))
        x = x.view(-1, self.window_size, self.window_size, c)
        x = x.view(-1, h // self.window_size, w // self.window_size, self.window_size, self.window_size, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, h, w, c)
        return x.permute(0, 3, 1, 2).contiguous()

    # Input/Output B,C,H,W
    def forward(self, x):
        B, C, H, W = x.shape

        # B, C, H, W -> N, window_size^2, C
        x = self.window_partition(x)

        x = self.sa(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x

        # N, window_size^2, C -> B, C, H, W
        x = self.reverse_window_partition(x, H, W)
        return x


class LHSABlock(nn.Module):
    def __init__(self, levels, window_size, dim):
        super().__init__()
        self.levels = levels
        self.dim = dim

        self.vit = nn.ModuleList([*[ViTBlock(window_size, dim) for _ in range(levels + 1)]])
        #self.fuse = nn.Conv2d(dim * levels, dim, 1, 1, 0)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        self.activation = nn.GELU()

        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim * (2 ** 2), kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1)

            ) for _ in range(levels)])

        self.re_zero = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(self.levels)])

    def forward(self, x):
        B, C, H, W = x.size()

        z_prior = 0
        #maps = []
        for i in reversed(range(self.levels)):
            # downsample to level size
            if i > 0:
                size = (H // 2 ** i, W // 2 ** i)
                z = F.interpolate(x, size=size, mode='bicubic')
            else:
                z = x

            z = self.vit[i](z + self.re_zero[i] * z_prior)

            if i > 0:
                z_prior = self.upsample[self.levels - i - 1](z)
                #maps.append(F.interpolate(z_prior, size=(H, W), mode='bicubic'))

        #maps.append(z)

        # feature fusion
        #z = self.fuse(torch.cat(maps, dim=1)) + x
        z = self.aggr(z) + z
        return z


class DenseResidualBlock(nn.Module):
    def __init__(self, n_sub_blocks, levels, window_size, dim, ffn_scale=2.0):
        super().__init__()
        self.n_sub_blocks = n_sub_blocks
        self.layers = nn.Sequential(*[LHSABlock(levels=levels, dim=dim, window_size=window_size, ffn_scale=ffn_scale) for _ in range(n_sub_blocks)])
        self.out_layer = nn.Sequential(
            LHSABlock(levels=levels, dim=dim, window_size=window_size, ffn_scale=ffn_scale),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.re_zero = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(self.n_sub_blocks)])

    def forward(self, x):
        residuals = [x]
        for i in range(self.n_sub_blocks):
            x = self.re_zero[i] * self.layers[i](x)
            for r in residuals:
                x += r
            residuals.append(x)

        return self.out_layer(x) + residuals[0]


class LMLTransformer(nn.Module):
    def __init__(self, n_blocks, levels, window_size, dim, features, scale_factor, ffn_scale=2.0):
        super().__init__()
        self.depth = n_blocks
        self.LHSA_levels = levels
        self.dim = dim
        self.features = features
        self.scale_factor = scale_factor
        self.window_size = window_size

        self.feature_extractor = nn.Conv2d(3, dim, 3, 1, 1)

        #self.layers = nn.Sequential(*[Blocks(levels=levels, dim=dim, window_size=window_size, ffn_scale=ffn_scale) for _ in range(n_blocks)])
        self.layers = nn.Sequential(*[LHSABlock(levels=levels, dim=dim, window_size=window_size) for _ in range(n_blocks)])
        #self.layers = nn.Sequential(*[DenseResidualBlock(n_sub_blocks=3, levels=levels, dim=dim, window_size=window_size, ffn_scale=ffn_scale) for _ in range(2)])

        img_reconstruction = [
            nn.Conv2d(dim, features, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(features, features, 3, 1, 1),
        ]
        for i in range(int(np.log2(scale_factor))):
            img_reconstruction.extend([
                nn.Conv2d(features, features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ])
        img_reconstruction.extend([
            nn.Conv2d(features, features, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(features, 3, 3, 1, 1, )
        ])

        self.img_reconstruction = nn.Sequential(*img_reconstruction)

        self.apply(self.init_weights)

    def padding(self, x):
        _, _, h, w = x.size()
        scaled_size = self.window_size ** 2

        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
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

        x = self.layers(x) + x

        # crop padding
        x = x[:, :, :H, :W]

        x = self.img_reconstruction(x)

        x = x * 0.25 + 0.5

        return torch.clamp(x, min=0, max=1.0)


# TODO possible reorder channels so it conv upscales to x4 instead of running constant dim

# TODO potentially use adapative residual connections
# TODO maybe downscaling with cnn
# TODO Enhanced Multi-scale Fusion:

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
