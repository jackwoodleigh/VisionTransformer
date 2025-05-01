import torch
from torch import nn
import torch.nn.functional as F
from MSFT_Arch import ViTBlock


class LHSABlock_1(nn.Module):
    def __init__(self, levels, window_size, dim, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels
        self.dim = dim

        self.vit = nn.ModuleList(
            [*[ViTBlock(window_size, dim, ffn_scale=ffn_scale, drop_path=drop_path) for _ in range(levels)]])
        self.aggr = nn.Conv2d(dim * levels, dim * levels, 1, 1, 0)

        self.fuse = nn.Conv2d(dim * levels, dim, 1, 1, 0)
        self.vit_fuse = ViTBlock(window_size, dim * levels, ffn_scale=ffn_scale, drop_path=drop_path)

        self.activation = nn.GELU()

        self.upsample_1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim * (2 ** 2), kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1)

            ) for _ in range(levels - 1)
        ])

        self.re_zero = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(self.levels)])

    def forward(self, x):
        B, C, H, W = x.size()

        z_prior = 0
        maps = []
        for i in reversed(range(self.levels)):
            # downsample to level size
            if i > 0:
                size = (H // 2 ** i, W // 2 ** i)
                z = F.interpolate(x, size=size, mode='bilinear')
            else:
                z = x

            z = self.vit[i](z + self.re_zero[i] * z_prior)

            if i > 0:
                z_prior = self.upsample_1[-i + 1](z)
                maps.append(F.interpolate(z, size=(H, W), mode='bilinear'))

        maps.append(z)
        z = self.aggr(torch.cat(maps, dim=1))

        # feature fusion
        z = self.vit_fuse(z)
        z = self.fuse(z) + x
        return z


class LHSABlock_2(nn.Module):
    def __init__(self, levels, window_size, dim, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels
        self.dim = dim

        self.vit = nn.ModuleList([*[ViTBlock(window_size, dim * (2 ** i) ** 2, ffn_scale=ffn_scale, drop_path=drop_path)
                                    for i in range(levels)]])
        self.re_zero = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(self.levels)])

        self.aggr = nn.Conv2d(dim * levels, dim * levels, 1, 1, 0)
        self.vit_fuse = ViTBlock(window_size, dim * levels, ffn_scale=ffn_scale, drop_path=drop_path)
        self.fuse = nn.Conv2d(dim * levels, dim, 1, 1, 0)

    def forward(self, x):
        z_prior = 0
        maps = []
        for i in reversed(range(self.levels)):
            z = F.pixel_unshuffle(x, 2 ** i)

            z = self.vit[i](z + self.re_zero[i] * z_prior)

            maps.append(F.pixel_shuffle(z, 2 ** i))
            if i > 0:
                z_prior = F.pixel_shuffle(z, 2)

        z = self.aggr(torch.cat(maps, dim=1))

        # feature fusion
        z = self.vit_fuse(z)
        z = self.fuse(z) + x
        return z


class LHSABlock_3(nn.Module):
    def __init__(self, levels, window_size, dim, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels
        self.dim = dim

        self.vit = nn.ModuleList([*[
            ViTBlock(window_size, (dim // levels) * (2 ** i) ** 2, ffn_scale=ffn_scale, drop_path=drop_path) for i in
            range(levels)]])
        self.re_zero = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(self.levels)])

        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.vit_fuse = ViTBlock(window_size, dim, ffn_scale=ffn_scale, drop_path=drop_path)
        self.fuse = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        xc = torch.chunk(x, self.levels, dim=1)

        z_prior = 0
        maps = []
        for i in reversed(range(self.levels)):
            z = F.pixel_unshuffle(xc[i], 2 ** i)

            z = self.vit[i](z + self.re_zero[i] * z_prior)

            maps.insert(0, F.pixel_shuffle(z, 2 ** i))
            if i > 0:
                z_prior = F.pixel_shuffle(z, 2)

        z = self.aggr(torch.cat(maps, dim=1))

        # feature fusion
        z = self.vit_fuse(z)
        z = self.fuse(z) + x
        return z


class LHSABlock_4(nn.Module):
    def __init__(self, levels, window_size, dim, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels
        self.dim = dim

        self.level_dims = [dim // (4 ** i) for i in reversed(range(1, levels))]
        self.level_dims.append(dim)
        temp_dim = sum(self.level_dims)

        self.vit = nn.ModuleList(
            [*[ViTBlock(window_size, dim, ffn_scale=ffn_scale, drop_path=drop_path) for _ in range(levels)]])

        self.downsample = nn.ModuleList(
            [*[nn.Conv2d(dim, dim // (4 ** i), 3, 1, 1) for i in reversed(range(1, levels))]])

        self.merge = nn.ModuleList([nn.PixelUnshuffle(2 ** i) for i in reversed(range(levels))])
        self.unmerge = nn.ModuleList([nn.PixelShuffle(2 ** i) for i in reversed(range(levels))])

        self.aggr = nn.Conv2d(temp_dim, temp_dim, 1, 1, 0)

        self.vit_fuse = ViTBlock(window_size, temp_dim, ffn_scale=ffn_scale, drop_path=drop_path)
        self.fuse = nn.Conv2d(temp_dim, dim, 1, 1, 0)

    def forward(self, x):
        maps = []
        for i in range(self.levels):
            #z = x[:, :self.level_dims[i], :, :]
            if i < self.levels - 1:
                z = self.downsample[i](x)
            else:
                z = x
            z = self.merge[i](z)
            z = self.vit[i](z)
            z = self.unmerge[i](z)
            maps.append(z)

        z = self.aggr(torch.cat(maps, dim=1))
        # feature fusion
        z = self.vit_fuse(z)
        z = self.fuse(z) + x
        return z


class LHSABlock_5(nn.Module):
    def __init__(self, levels, window_size, dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels
        self.dim = dim

        self.total_dim = sum([dim // (4 ** i) for i in range(levels)])
        self.level_layer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // (4 ** i), 3, 1, 1),
                nn.PixelUnshuffle(2 ** i),
                ViTBlock(window_size, dim, n_heads=n_heads, ffn_scale=ffn_scale, drop_path=drop_path),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.PixelShuffle(2 ** i)
            ) for i in range(1, levels)
        ])

        self.fuse = nn.Sequential(
            ViTBlock(window_size, self.total_dim, n_heads=n_heads_fuse, ffn_scale=ffn_scale, drop_path=drop_path),
            nn.Conv2d(self.total_dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        maps = [x]
        for i in range(self.levels - 1):
            maps.append(self.level_layer[i](x))

        z = self.fuse(torch.cat(maps, dim=1)) + x
        return z


class MSFBlock_1(nn.Module):
    def __init__(self, levels, window_size, dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels
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
        for i in range(self.levels - 1):
            maps.append(self.level_layer[i](x))
        return self.fuse(torch.cat(maps, dim=1)) + x


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
        self.post_level_fuse = nn.Conv2d(total_level_dim, dim // 4, 1, 1)

        self.fuse = nn.Sequential(
            ViTBlock(window_size, dim + dim // 4, n_heads=n_heads_fuse, ffn_scale=ffn_scale, drop_path=drop_path),
            nn.Conv2d(dim + dim // 4, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        maps = []
        for i in range(self.levels - 1):
            maps.append(self.level_layer[i](x))

        levels = self.post_level_fuse(torch.cat(maps, dim=1))
        return self.fuse(torch.cat([x, levels], dim=1)) + x


# Sequential down sampling with smaller kernel sizes

class MSFBlock_3(nn.Module):
    def __init__(self, levels, window_size, dim, level_dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels
        level_dim = dim // 2

        self.init_ds = nn.Conv2d(dim, level_dim, 1, 1)
        self.level_ds = nn.ModuleList(
            [nn.Conv2d(level_dim // (4 ** i), level_dim // (4 ** (i + 1)), 1, 1) for i in range(levels - 1)])

        self.level_layer = nn.ModuleList([
            nn.Sequential(
                nn.PixelUnshuffle(2 ** i),
                ViTBlock(window_size, level_dim, n_heads=n_heads, ffn_scale=ffn_scale, drop_path=drop_path),
                nn.PixelShuffle(2 ** i)
            ) for i in range(1, levels)
        ])

        total_level_dim = sum([level_dim // (4 ** i) for i in range(1, levels)])
        self.post_level_fuse = nn.Conv2d(total_level_dim, total_level_dim, 3, 1, 1)
        self.fuse = nn.Sequential(
            ViTBlock(window_size, dim + total_level_dim, n_heads=n_heads_fuse, ffn_scale=ffn_scale,
                     drop_path=drop_path),
            nn.Conv2d(dim + total_level_dim, dim, 1, 1),
        )

        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        maps = []
        z = self.init_ds(x)
        for i in range(self.levels - 1):
            z = self.level_ds[i](z)
            maps.append(self.level_layer[i](z))

        levels = self.post_level_fuse(torch.cat(maps, dim=1))
        return self.rezero * self.fuse(torch.cat([x, levels], dim=1)) + x


# parallel down sampling minimal convolutional parameter reduction and new merging
class MSFBlock_4(nn.Module):
    def __init__(self, levels, window_size, dim, level_dim, n_heads, n_heads_fuse, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.levels = levels
        level_dim = dim // 2

        self.each_level_dims = [level_dim // 4 ** i for i in range(1, levels)]
        total_level_dim = sum(self.each_level_dims)
        self.ds = nn.Conv2d(dim, total_level_dim, 1, 1)

        '''self.level_layer = nn.ModuleList([
            nn.Sequential(
                nn.PixelUnshuffle(2 ** i),
                ViTBlock(window_size, level_dim, n_heads=n_heads, ffn_scale=ffn_scale, drop_path=drop_path),
                nn.PixelShuffle(2 ** i)
            ) for i in range(1, levels)
        ])'''

        self.level_layer = nn.ModuleList([
            nn.Sequential(
                PatchMerging(level_dim, 2 ** i),
                ViTBlock(window_size, level_dim, n_heads=n_heads, ffn_scale=ffn_scale, drop_path=drop_path),
                PatchUnMerging(level_dim, 2 ** i)
            ) for i in range(1, levels)
        ])

        self.post_level_fuse = nn.Conv2d(total_level_dim, total_level_dim, 3, 1, 1)
        self.fuse = nn.Sequential(
            ViTBlock(window_size, dim + total_level_dim, n_heads=n_heads_fuse, ffn_scale=ffn_scale, drop_path=drop_path),
            nn.Conv2d(dim + total_level_dim, dim, 1, 1),
        )

        self.rezero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = self.ds(x)
        z_maps = list(z.split(self.each_level_dims, dim=1))

        for i in range(self.levels - 1):
            z_maps[i] = self.level_layer[i](z_maps[i])

        level_context = self.post_level_fuse(torch.cat(z_maps, dim=1))
        return self.rezero * self.fuse(torch.cat([x, level_context], dim=1)) + x
