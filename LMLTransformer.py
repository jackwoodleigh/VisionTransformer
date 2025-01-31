import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# https://arxiv.org/pdf/2409.03516

class ViTBlock(nn.Module):
    def __init__(self, depth, window_size, patch_size, channels, height, width, dim=128):
        super().__init__()
        self.depth = depth
        self.patch_size = patch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.num_patches = height*width // patch_size**2

        self.window_size = window_size
        self.dim = dim
        self.qkv = nn.Linear(dim, 3*dim)

        self.pe_encoder = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.out_proj = nn.Linear(dim, dim)

    # output: B, C, H, W
    # input: N, window_size^2, C
    def window_partition(self, x):
        B, C, H, W = x.shape
        # B, n_h_windows, window size, n_w_windows, window size, C -> B, n_h_windows, n_w_windows, window size, window size, C
        # then merge batch and number of windows
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        x = x.view(-1, self.window_size * self.window_size, C)      # probably not needed extra step
        return x

    # input: N, window_size^2, C
    # output: B, C, H, W
    def reverse_window_partition(self, x, h, w):
        # inverse of window create
        b = x.shape[0] // (w * h // self.window_size // self.window_size)
        x = x.view(b, self.window_size, self.window_size, -1)
        x = x.view(b, h // self.window_size, w//self.window_size, self.window_size, self.window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, h, w)

    # input: -1, window_size^2, C
    def locally_enhanced_pe(self, x, func):
        N, win_size_sq, C = x.shape

        # converting into spatial form: N, window_size^2, C -> B, C, window_size, window_size
        x = x.view(N, C, self.window_size, self.window_size)
        lepe = func(x)

        # converting into back into image with: B, C, H, W -> N, W^2, C
        lepe = lepe.view(N, self.window_size * self.window_size, C)

        return lepe

    # 1, 3, 16, 16 = x
    def forward(self, x):
        B, C, H, W = x.shape

        # convert to windows
        # B, C, H, W -> N, window_size^2, C
        y = self.window_partition(x)

        # q,k,v: N, window_size^2, C
        qkv = self.qkv(y)
        q, k, v = qkv.chunk(3, dim=-1)

        # attention
        y = nn.functional.scaled_dot_product_attention(q, k, v)

        # positional encoding
        y += self.locally_enhanced_pe(v, self.pe_encoder, H, W)

        y = self.out_proj(y)
        y = self.reverse_window_partition(y, H, W)

        return y


class HLMTBlock(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = levels

        self.vit = nn.ModuleList([])

        self.activation = nn.GELU()

        # passing in B, C, H, W
    def forward(self, x):
        B, C, H, W = x.size()

        # we chunk the features into levels
        x_chunked = x.chunk(self.levels, dim=1)

        # we are getting the hierarchical layers by via down sampling
        downsampled_maps = []
        for i in range(self.levels):
            if i > 0:
                patch_size = (H//2**i, W//2**i)

                # down samples to new size
                z_down = F.adaptive_max_pool2d(x_chunked[i], patch_size)
                downsampled_maps.append(z_down)

            else:
                downsampled_maps.append(x_chunked[i])

        out_maps = []
        for i in reversed(range(self.levels)):
            z = self.vit[i](downsampled_maps[i])

            # interpolating it the size of the layer above
            z_up = F.interpolate(z, size=(z.shape[2]*2, z.shape[3]*2), mode='nearest')

            # adding elementwise the up-sampled feature map for increased detail
            if i > 0:
                downsampled_maps[i-1] += z_up

            # interpolating image back to original H*W feature map size and returning
            z = F.interpolate(z_up, size=(H, W), mode='nearest')
            out_maps.append(z)

        # joins feature maps
        out_maps = self.aggr(torch.cat(out_maps, dim=1))

        # multiplicative residual connection for less dependency on original.
        out_maps = self.activation(out_maps) * x

        return out_maps


class LMLTransformer(nn.Module):
    def __init__(self, dim, depth, patch_size, channels, height, width, emb_dim=128):
        super().__init__()
        self.depth = depth
        self.patch_size = patch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.num_patches = height * width // patch_size ** 2

        self.patch_emb = nn.Linear(patch_size ** 2 * channels, emb_dim)
        self.class_tokens = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(emb_dim, 2),
                MLP(emb_dim)
            ]))

        self.mlp = MLP(emb_dim, out_features=3 * patch_size ** 2)

        self.feature_extractor = nn.Conv2d(3, dim, 3, 1, 1)

    # 1, 3, 16, 16 = x
    def forward(self, x):
        B, C, H, W = x.shape

        # B, emb_dim, H, W
        x = self.feature_extractor(x)

        return x


#vit = VisionTransformer(4, 4, 3, 32, 32)
# t = Transformer(4)
#x = torch.randn(1, 3, 32, 32)
#print(vit(x).shape)