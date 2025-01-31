import torch
from torch import nn



class VisionTransformer(nn.Module):
    def __init__(self, depth, patch_size, channels, height, width, emb_dim=128):
        super().__init__()
        self.depth = depth
        self.patch_size = patch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.num_patches = height*width // patch_size**2

        self.patch_emb = nn.Linear(patch_size**2 * channels, emb_dim)
        self.class_tokens = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(emb_dim, 2),
                MLP(emb_dim)
            ]))

        self.mlp = MLP(emb_dim, out_features=3 * patch_size**2)

    # 1, 3, 16, 16 = x
    def forward(self, x):
        B, C, H, W = x.shape

        # Batch, num patches, patch size
        x = x.view(B, -1, C * self.patch_size**2)

        # Batch, num patches, emb_dim
        x = self.patch_emb(x)

        # repeating class token across the batches and cating
        # Batch, num patches +1, emb_dim
        class_tokens = self.class_tokens.expand(B, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        # output: batch, token, dim
        for transformer, mlp in self.layers:
            x = transformer(x) + x
            x = mlp(x) + x

        # remove class head and decode the embedding
        x = self.mlp(x[:, 1:])
        x = x.view(B, C, H, W)

        return x


class Transformer(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        
        self.in_proj = nn.Linear(self.emb_dim, 3*self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

        self.ln1 = nn.LayerNorm(self.emb_dim)
        self.ln2 = nn.LayerNorm(self.emb_dim)

    # Batch, Token, Dim
    def forward(self, x):
        B, T, D = x.shape

        x = self.ln1(x)

        # projecting to qkv and norm
        qkv = self.in_proj(x)

        # multihead reshape
        qkv = qkv.view(B, T, self.n_heads, 3*self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # scaled dot product
        y = nn.functional.scaled_dot_product_attention(q, k, v)

        # out projection, residual, and norm
        y = y.reshape(B, T, D)
        y = self.out_proj(y) + x
        y = self.ln2(y)

        return y


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


vit = VisionTransformer(4, 4, 3, 32, 32)
#t = Transformer(4)
x = torch.randn(1, 3, 32, 32)
print(vit(x).shape)