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

class FConv(nn.Module):
    def __init__(self):
        super.__init__()

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

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


class ViTBlock(nn.Module):
    def __init__(self, window_size, dim):
        super().__init__()
        self.window_size = window_size
        self.dim = dim

        self.qkv = nn.Linear(dim, 3 * dim)
        self.pe_encoder = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.out_proj = nn.Linear(dim, dim)

    # output: B, H, W, C
    # input: N, window_size^2, C
    def window_partition(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C).view(-1, self.window_size * self.window_size, C)

    # input: N, window_size^2, C
    # output: B, H, W, C
    def reverse_window_partition(self, x, h, w):
        b = int(x.shape[0] / (w * h / self.window_size / self.window_size))
        x = x.view(b, h // self.window_size, w // self.window_size, self.window_size, self.window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

    # input: -1, window_size^2, C
    def locally_enhanced_pe(self, x, func):
        N, win_size_sq, C = x.shape

        # converting into spatial form: N, window_size^2, C -> B, C, window_size, window_size
        x = x.reshape(N, C, self.window_size, self.window_size)

        # find spatial encodings for each window
        lepe = func(x)

        # converting into back into image with: B, C, H, W -> N, W^2, C
        lepe = lepe.view(N, self.window_size * self.window_size, C)

        return lepe

    # 1, 3, 16, 16 = x
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        # convert to windows
        # B, H, W, C -> N, window_size^2, C
        x = self.window_partition(x)

        # q,k,v: N, window_size^2, C
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # attention and apply positional encoding
        lepe = self.locally_enhanced_pe(v, self.pe_encoder)
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        x += lepe

        x = self.out_proj(x)
        x = self.reverse_window_partition(x, H, W)

        return x.permute(0, 3, 1, 2)


class LHSABlock(nn.Module):
    def __init__(self, levels, window_size, dim):
        super().__init__()
        self.levels = levels
        self.dim = dim

        self.vit = nn.ModuleList([*[ViTBlock(window_size, dim // levels) for _ in range(levels)]])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        self.activation = nn.GELU()

        self.upsample_1st = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim // levels, (dim // levels) * (2 ** 2), kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(dim // levels, (dim // levels), 3, 1, 1)

            ) for _ in range(levels)])

        self.pixel_shuffle = nn.PixelShuffle(2)

        '''self.upsample_2nd = nn.ModuleList([])
        for i in range(levels)[1:-1]:
            scale = 2**i
            self.upsample_2nd.append(nn.Sequential(
                nn.Conv2d(dim // levels, (dim // levels) * (scale ** 2), kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(scale)
            ))'''

    def forward(self, x):
        B, C, H, W = x.size()

        # we chunk the features into levels
        x_chunked = x.chunk(self.levels, dim=1)

        # we are getting the hierarchical layers by via down sampling
        downsampled_maps = []
        for i in range(self.levels):
            if i > 0:
                # new size
                size = (H // 2 ** i, W // 2 ** i)

                # down samples to new size
                z_down = F.adaptive_max_pool2d(x_chunked[i], size)
                downsampled_maps.append(z_down)

            else:
                downsampled_maps.append(x_chunked[i])

        out_maps = []
        for i in reversed(range(self.levels)):
            z = self.vit[i](downsampled_maps[i])
            if i > 0:
                # interpolating it the size of the layer above
                #z = F.interpolate(z, size=(z.shape[2] * 2, z.shape[3] * 2), mode='nearest')
                #z = self.pixel_shuffle(self.upsample_1st[self.levels-1-i](z))
                z = self.upsample_1st[self.levels-1-i](z)

                # adding elementwise the up-sampled feature map for increased detail
                downsampled_maps[i - 1] = downsampled_maps[i - 1] + z

                # interpolating image back to original H*W feature map size and returning
                z = F.interpolate(z, size=(H, W), mode='nearest')
                '''if i > 1:
                    z = self.upsample_2nd[i-2](z)'''

            out_maps.append(z)

        # aggregate feature maps
        out_maps = self.aggr(torch.cat(out_maps, dim=1))

        # multiplicative residual connection for less dependency on original.
        out_maps = self.activation(out_maps) * x

        return out_maps


class LMLTBlock(nn.Module):
    def __init__(self, levels, window_size, dim, ffn_scale=2.0):
        super().__init__()
        self.LHSA = LHSABlock(levels=levels, dim=dim, window_size=window_size)
        self.CCM = CCM(dim, ffn_scale)
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)

    def forward(self, x):
        x = self.LHSA(self.ln1(x)) + x
        x = self.CCM(self.ln2(x)) + x
        return x


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

        self.layers = nn.Sequential(*[LMLTBlock(levels=levels, dim=dim, window_size=window_size, ffn_scale=ffn_scale) for _ in range(n_blocks)])

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

    def padding(self, x):
        _, _, h, w = x.size()
        scaled_size = self.window_size ** 2

        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    # 1, 3, 16, 16 = x
    def forward(self, x):
        B, C, H, W = x.shape

        x = self.padding(x)

        x = self.feature_extractor(x)

        x = self.layers(x) + x

        # crop padding
        x = x[:, :, :H, :W]

        x = self.img_reconstruction(x)

        return x


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
