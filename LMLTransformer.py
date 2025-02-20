import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# https://arxiv.org/pdf/2409.03516

# https://arxiv.org/pdf/2208.11247v3
# https://arxiv.org/pdf/2404.00722v5
# https://arxiv.org/pdf/2205.04437v3 maybe try out lam

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
        B, H, W, C = x.size()
        x = x.view(B, C, H, W)
        x = self.ccm(x)
        x = x.view(B, H, W, C)
        return x


class ViTBlock(nn.Module):
    def __init__(self, window_size, dim):
        super().__init__()
        self.window_size = window_size
        self.dim = dim

        self.qkv = nn.Linear(dim, 3 * dim)
        self.pe_encoder = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.out_proj = nn.Linear(dim, dim)

    # output: B, C, H, W
    # input: N, window_size^2, C
    def window_partition(self, x):
        B, C, H, W = x.shape
        # B, n_h_windows, window size, n_w_windows, window size, C -> B, n_h_windows, n_w_windows, window size, window size, C
        # then merge batch and number of windows
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)
        return x

    # input: N, window_size^2, C
    # output: B, C, H, W
    def reverse_window_partition(self, x, h, w):
        # inverse of window create
        b = x.shape[0] // ((w // self.window_size) * (h // self.window_size))
        x = x.view(b, self.window_size, self.window_size, -1)
        x = x.view(b, h // self.window_size, w // self.window_size, self.window_size, self.window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, h, w)

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

        # convert to windows
        # B, C, H, W -> N, window_size^2, C
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

        return x


class LHSABlock(nn.Module):
    def __init__(self, levels, window_size, dim):
        super().__init__()
        self.levels = levels
        self.dim = dim

        self.vit = nn.ModuleList([*[ViTBlock(window_size, dim // levels) for _ in range(levels)]])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        self.activation = nn.GELU()

    # passing in B, H, W, C
    # returning  B, H, W, C
    def forward(self, x):
        B, H, W, C = x.size()
        x = x.view(B, C, H, W)

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
                z_up = F.interpolate(z, size=(z.shape[2] * 2, z.shape[3] * 2), mode='bicubic')

                # adding elementwise the up-sampled feature map for increased detail
                downsampled_maps[i - 1] = downsampled_maps[i - 1] + z_up

                # interpolating image back to original H*W feature map size and returning
                z = F.interpolate(z_up, size=(H, W), mode='bicubic')

            out_maps.append(z)

        # aggregate feature maps
        out_maps = self.aggr(torch.cat(out_maps, dim=1))

        # multiplicative residual connection for less dependency on original.
        out_maps = self.activation(out_maps) * x

        # returns to B,C,H,W -> B, H, W, C
        out_maps = out_maps.view(B, H, W, C)

        return out_maps


class LMLTBlock(nn.Module):
    def __init__(self, levels, window_size, dim, ffn_scale=2.0):
        super().__init__()
        self.LHSA = LHSABlock(levels=levels, dim=dim, window_size=window_size)
        self.CCM = CCM(dim, ffn_scale)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.rezero_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.LHSA(self.ln1(x)) + x
        x = self.CCM(self.ln2(x)) + x
        return x


class LMLTransformer(nn.Module):
    def __init__(self, n_blocks, levels, window_size, dim, scale_factor, ffn_scale=2.0):
        super().__init__()
        self.depth = n_blocks
        self.LHSA_levels = levels
        self.dim = dim
        self.scale_factor = scale_factor
        self.window_size = window_size

        self.feature_extractor = nn.Conv2d(3, dim, 3, 1, 1)

        self.layers = nn.Sequential(
            *[LMLTBlock(levels=levels, dim=dim, window_size=window_size, ffn_scale=ffn_scale) for _ in range(n_blocks)])

        # convert from dim -> color scale * img scale factor and uses pixel shuffle to organize image
        self.img_reconstruction = nn.Sequential(
            nn.Conv2d(dim, 3 * scale_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(scale_factor)
        )

    def padding(self, x):
        _, _, h, w = x.size()
        scaled_size = self.window_size ** 2

        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    # 1, 3, 16, 16 = x
    def forward(self, x):
        _, _, orig_H, orig_W = x.shape

        x = self.padding(x)
        B, C, H, W = x.shape

        # B, dim, H, W,
        x = self.feature_extractor(x)

        x = x.view(B, H, W, self.dim)
        x = self.layers(x) + x
        x = x.view(B, self.dim, H, W)

        # crop padding
        x = x[:, :, :orig_H, :orig_W]

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
