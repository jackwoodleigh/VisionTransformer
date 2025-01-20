import torch
from torch import nn


class VisionTransformer(nn.Module):
    def __init__(self, patch_size, channels, height, width, emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.num_patches = height*width // patch_size**2

        self.patch_emb = nn.Linear(patch_size**2 * channels, emb_dim)
        self.class_tokens = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.transformer = Transformer(emb_dim, 2)

    # 1, 3, 16, 16 = x
    def forward(self, x):
        B, C, H, W = x.shape

        # Batch, num patches, patch size
        x = x.view(B, -1, C * self.patch_size**2)

        # Batch, num patches, emb_dim
        x = self.patch_emb(x)

        # repeating class token across the batches and cating
        class_tokens = self.class_tokens.expand(B, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        x = self.transformer(x)

        return x


class Transformer(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        
        self.in_proj = nn.Linear(self.emb_dim, 3*self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

        self.ln1 = nn.LayerNorm(3*self.emb_dim)
        self.ln2 = nn.LayerNorm(self.emb_dim)


    # Batch, Token, Dim
    def forward(self, x):
        B, T, D = x.shape

        # projecting to qkv and norm
        qkv = self.in_proj(x)
        qkv = self.ln1(qkv)

        # multihead reshape
        qkv = qkv.view(B, T, self.n_heads, 3*self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # scaled dot product, reshape, and residual
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.reshape(B, T, D) + 1

        # out projection, norm, and residual
        y = self.out_proj(y)
        y = self.ln2(y) + x

        return y


vit = VisionTransformer(4, 3, 32, 32)
#t = Transformer(4)
x = torch.randn(1, 3, 32, 32)
print(vit(x).shape)