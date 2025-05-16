import torch
from torch import nn


class SlidingContext(nn.Module):
    def __init__(self, dim, window_size, stride):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(3, 3, 2,)




import torch
import torch.nn.functional as F

# Assume images are tensors with shape (B, C, H, W)
B, C = 1, 3
img_main = torch.randn(B, C, 64, 64)
img_context = torch.randn(B, C, 32, 32)
t = torch.randn(B, 32, 32, 3)
l = nn.Linear(3, 3)

'''# 2. Extract all sliding 8x8 patches from context (stride 1)
context_patches = F.unfold(img_context, kernel_size=4, stride=4)  # (B, C*8*8, L)
#context_patches = context_patches.transpose(1, 2).reshape(B, -1, C, 8, 8)  # (B, L, C, 8, 8)'''

print(l(t).shape)