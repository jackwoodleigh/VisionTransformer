
import torch
from torch import nn

t = torch.tensor(range(64)).view(1, 1, 8, 8)
print(t.shape)
print(t)

print("====================================")

pixel = nn.PixelUnshuffle(2)
t2 = pixel(t)
print(t2.shape)

print(t2[0][0])

print("====================================")
x = t.permute(0, 2, 3, 1)
x0 = x[:, 0::2, 0::2, :]
x1 = x[:, 1::2, 0::2, :]
x2 = x[:, 0::2, 1::2, :]
x3 = x[:, 1::2, 1::2, :]
t3 = torch.cat([x0, x1, x2, x3], -1)
print(t3.shape)

print(t3.permute(0, 3, 1, 2)[0][0])