import math

import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.utils.checkpoint import checkpoint
from toolkit.arch_util import trunc_normal_
from torch_scatter import scatter_max, scatter_mean


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
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)

    def forward(self, x):
        # x: B, H, W, C
        return self.conv(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ffn_scale=4, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * ffn_scale)

        self.fc1_gate = nn.Linear(in_features, hidden_features)
        self.fc1_content = nn.Linear(in_features, hidden_features)

        self.act = nn.SiLU()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        gate = self.fc1_gate(x)
        content = self.fc1_content(x)

        x = self.act(gate) * content

        x = self.fc2(x)
        x = self.drop(x)
        return x


# input/output B, C, H, W
class PatchMerging(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(scale)
        self.norm = nn.LayerNorm((scale ** 2) * dim)
        self.linear = nn.Linear((scale ** 2) * dim, dim, bias=False)

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
    # modified version of RPE that is adapted for cross attention assuming data is spatially correlated
    # if the input sizes are different it is assumed to be in a convolutional matrix format with q at the center

    def __init__(self, q_size, k_size, num_heads=1, scale=1):
        super().__init__()
        # in this case assumes q is smaller than k
        assert q_size <= k_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * k_size - 1) * (2 * k_size - 1), num_heads))
        self.scale = scale
        coords = torch.stack(torch.meshgrid(torch.arange(k_size), torch.arange(k_size), indexing='ij'), dim=0)

        # creates centered indices for q if it is smaller
        start_q = (k_size - q_size) // 2
        end_q = start_q + q_size
        coords_q = coords[:, start_q:end_q, start_q:end_q].reshape(2, -1)
        coords_k = coords.reshape(2, -1)

        # standard relative cords (2, num_queries, num_keys) -> (num_queries, num_keys, 2) with non neg offset
        relative_coords = coords_q[:, :, None] - coords_k[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() + k_size - 1

        # 2d -> 1d cords
        relative_position_index = relative_coords[:, :, 0] * (2 * k_size - 1) + relative_coords[:, :, 1]
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        bias = self.scale * self.relative_position_bias_table[self.relative_position_index]
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
        return x.view(B, (H // win) * (W // win), win ** 2, C) if keep_batch else x.view(B * (H // win) * (W // win),
                                                                                         win ** 2, C)

    def reverse_window_partition(self, x, win, H, W, with_batch=False):
        # Input shape: N, win*win, C or B, H/win * W/win, win^2, C
        # Output shape: B, H, W, C
        if with_batch:
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

    def attention(self, q, k, v, C, attn_drop=0., pe=0):
        return F.dropout(((q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5 + pe).softmax(dim=-1), p=attn_drop, training=self.training) @ v

class SparseDynamicWindowAttention(WindowedAttention):
    def __init__(self, window_size, dim, num_heads):
        super().__init__(window_size, dim, num_heads)
        self.temperature = 1.0
        self.w = window_size
        self.num_win_selected = 0.25
        self.sparse_discriminator = nn.Sequential(
            nn.Conv2d(dim, 4, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(4, 4, self.w, stride=self.w),
            nn.SiLU(),
            nn.Conv2d(4, 4, 3, 1, 1)
        )

        win_h_grid, win_w_grid = torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        )
        self.register_buffer("win_grid", torch.stack([win_h_grid, win_w_grid], dim=0), persistent=False)

        self.in_proj = nn.Linear(dim, 3 * dim)
        self.pe = RelativePositionalEncoding(q_size=self.w, k_size=self.w, num_heads=num_heads)
        self.out_proj = nn.Linear(dim, dim)
        self.out_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        # self.out_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=num_heads)

    def candidate_indexes(self, x, scores):
        B, _, H_win, W_win = scores.shape

        # total number of windows selected
        k = int((H_win * W_win) * self.num_win_selected)
        # ensure that important gates are also gates likely to be selected
        scores_flat = scores.view(B, -1)

        # hard gumbel softmax
        if self.training:
            gumbel_scores = F.gumbel_softmax(scores_flat, tau=self.temperature, hard=False)
        else:
            gumbel_scores = F.softmax(scores_flat / self.temperature, dim=-1)

        gumbel_scores, topk = torch.topk(gumbel_scores, k, dim=1)

        # STE for gumbel probs
        y_hard = torch.ones_like(gumbel_scores)
        scores_delta = (y_hard - gumbel_scores).detach() + gumbel_scores

        b_idx = torch.arange(B, device=x.device).view(-1, 1).expand(-1, k).reshape(-1)
        h_idx = (topk // W_win).reshape(-1)
        w_idx = (topk % W_win).reshape(-1)

        return b_idx, h_idx, w_idx, scores_delta.view(-1)

    def cords(self, b_idx, h_idx, w_idx, offsets, H, W):
        # values are 0 to 1 so it is scaled to half of window size for the maximum range of movement
        h_offset = offsets[b_idx, 0, h_idx, w_idx] * self.w - (self.w / 2.)
        w_offset = offsets[b_idx, 1, h_idx, w_idx] * self.w - (self.w / 2.)

        h1 = torch.clamp(h_idx * self.w + h_offset.long(), min=0, max=H - self.w)
        w1 = torch.clamp(w_idx * self.w + w_offset.long(), min=0, max=W - self.w)
        h2 = h1 + self.w
        w2 = w1 + self.w

        return h1, w1, h2, w2

    def integrate(self, x, delta, b_idx, h1, w1):
        B, C, H, W = x.shape

        win_h_grid, win_w_grid = self.win_grid
        abs_h_coords = h1.view(-1, 1, 1) + win_h_grid
        abs_w_coords = w1.view(-1, 1, 1) + win_w_grid
        flat_indices = (abs_h_coords * W + abs_w_coords).flatten(1)
        batch_offsets = b_idx * C * H * W
        channel_offsets = torch.arange(C, device=x.device).view(1, C) * H * W
        total_offsets = (batch_offsets.view(-1, 1) + channel_offsets).unsqueeze(2)
        final_indices = (total_offsets + flat_indices.unsqueeze(1)).view(-1)
        delta_flat = delta.view(-1)
        output = torch.zeros(B * C * H * W, dtype=delta.dtype, device=delta.device)
        output.scatter_reduce_(0, final_indices, delta_flat, reduce='mean', include_self=False)
        return output.view(B, C, H, W)

    def forward(self, x):
        # Input: B, H, W, C

        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape

        ########################################
        #    Offsets and candidate selection   #
        ########################################
        # converts x into information on data for each window
        sd = self.sparse_discriminator(x)
        scores = sd[:, :1, :, :]
        gates = sd[:, 1:2, :, :].sigmoid()
        offsets = sd[:, 2:, :, :].sigmoid()

        candidate_b_idx, candidate_h_idx, candidate_w_idx, scores_delta = self.candidate_indexes(x, scores)

        if candidate_b_idx.numel() == 0:
            return x.permute(0, 2, 3, 1).contiguous()

        ########################################
        #       Straight through estimator     #
        ########################################

        candidate_gates = gates[candidate_b_idx, 0, candidate_h_idx, candidate_w_idx]
        hard_gates = (candidate_gates > 0.5).float()
        gates_delta = (hard_gates - candidate_gates).detach() + candidate_gates

        pass_mask = hard_gates.bool()
        b_idx = candidate_b_idx[pass_mask]
        h_idx = candidate_h_idx[pass_mask]
        w_idx = candidate_w_idx[pass_mask]
        gates_delta = gates_delta[pass_mask]
        scores_delta = scores_delta[pass_mask]

        if b_idx.numel() == 0:
            return x.permute(0, 2, 3, 1).contiguous()

        h1, w1, h2, w2 = self.cords(b_idx, h_idx, w_idx, offsets, H, W)
        cords = torch.stack([b_idx, w1, h1, w2, h2], dim=1).float()

        x_sparse_before = torchvision.ops.roi_align(x, cords, output_size=(self.w, self.w))

        ########################################
        #               Attention              #
        ########################################
        # N, 3*C, win, win -> N, win, win, 3*C -> N, win^2, 3*C -> N, n_heads, win^2, 3*C / n_heads
        q, k, v = self.in_proj(x_sparse_before.permute(0, 2, 3, 1).contiguous()).view(-1, self.w ** 2, 3 * C).chunk(3, dim=-1)
        q, k, v = self.headify(q, k, v)
        x_sparse_after = self.attention(q, k, v, C, pe=self.pe())

        # N, n_heads, win^2, C / n_heads -> N, win^2, C -> N, C, win, win
        x_sparse_after = self.out_proj(self.deheadify(x_sparse_after).view(-1, self.w, self.w, C)).permute(0, 3, 1, 2).contiguous()
        delta = x_sparse_after - x_sparse_before
        ########################################
        #            Integration              #
        ########################################
        gated_delta = delta * gates_delta.view(-1, 1, 1, 1) * scores_delta.view(-1, 1, 1, 1)

        x = x + self.integrate(x, gated_delta, b_idx, h1, w1)

        return self.out_conv(x).permute(0, 2, 3, 1).contiguous()


class WindowedMSA(WindowedAttention):
    def __init__(self, window_size, dim, num_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__(window_size, dim, num_heads)
        self.win = window_size
        self.num_heads = num_heads
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, 3 * dim)
        self.pe = RelativePositionalEncoding(q_size=self.win, k_size=self.win, num_heads=num_heads)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        z = self.window_partition(x, self.win)
        qkv = self.qkv(z)
        q, k, v = self.headify(qkv).chunk(3, dim=-1)
        z = self.attention(q, k, v, C, attn_drop=self.attn_drop, pe=self.pe())
        z = self.reverse_window_partition(self.out_proj(self.deheadify(z)), self.win, H, W)
        return z

class ViTBlock(nn.Module):
    def __init__(self, window_size, dim, n_heads=4, ffn_scale=2, drop_path=0.0):
        super().__init__()
        self.wmsa = WindowedMSA(dim=dim, window_size=window_size, num_heads=n_heads)
        self.sdwa = SparseDynamicWindowAttention(window_size, dim, num_heads=n_heads)
        self.mlp1 = MLP(dim, ffn_scale=ffn_scale)
        self.ln1 = nn.RMSNorm(dim)
        self.ln2 = nn.RMSNorm(dim)
        self.ln3 = nn.RMSNorm(dim)
        self.rezero1 = nn.Parameter(torch.zeros(1))
        self.rezero2 = nn.Parameter(torch.zeros(1))
        self.rezero3 = nn.Parameter(torch.zeros(1))
        self.drop_path = nn.Dropout(drop_path)

    def forward(self, x):
        x = x + self.rezero1 * self.drop_path(self.wmsa(self.ln1(x)))
        x = x + self.rezero3 * self.drop_path(self.sdwa(self.ln3(x)))
        x = x + self.rezero2 * self.drop_path(self.mlp1(self.ln2(x)))
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
            ffn_scale=2,
            drop_path=0.1,
            annealing_itr=250000):
        super().__init__()

        self.LHSA_levels = levels
        self.dim = dim
        self.feature_dim = feature_dim
        self.scale_factor = scale_factor
        self.window_size = window_size
        self.levels = levels
        self.annealing_itr = annealing_itr
        self.decay_rate = (0.1 / 1.0) ** (1 / annealing_itr)

        self.feature_extractor = nn.Conv2d(3, dim, 3, 1, 1)
        if block_type != "default":
            pass
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
            nn.Conv2d(feature_dim, feature_dim // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 4, 3, 3, 1, 1, )
        ])

        self.img_reconstruction = nn.Sequential(*img_reconstruction)
        self.register_buffer('current_iteration', torch.tensor(0, dtype=torch.long))

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

    def update_temperature(self):
        if self.current_iteration < self.annealing_itr:
            start_temp = 1.0
            end_temp = 0.1
            progress = self.current_iteration / self.annealing_itr
            new_temp = end_temp + 0.5 * (start_temp - end_temp) * (1 + math.cos(math.pi * progress))

            for module in self.modules():
                if isinstance(module, SparseDynamicWindowAttention):
                    module.temperature = new_temp

        self.current_iteration += 1
    # 1, 3, 16, 16 = x
    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            self.update_temperature()

        x = self.padding(x)

        x = self.feature_extractor(x)
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.feature_transition(self.layers(x)) + x
        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.img_reconstruction(x)

        # crop padding
        return x[:, :, :H * self.scale_factor, :W * self.scale_factor]


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
