import cv2
import torch
import torchvision.transforms.functional as F
import numpy as np

def bicubic_downscale_HWC(hr, scale):
    return cv2.resize(hr, (hr.shape[1]//scale, hr.shape[0]//scale), interpolation=cv2.INTER_CUBIC)

# B, C, H, W
def paired_random_crop(hr, lr, size, scale=4):
    H, W = lr.shape[-2:]

    H_c, W_c = size[0]//scale, size[1]//scale

    y = np.random.randint(0, H - H_c + 1, (1,)).item()
    x = np.random.randint(0, W - W_c + 1, (1,)).item()

    lr_c = lr[..., y:y+H_c, x:x+W_c]
    hr_c = hr[..., scale*y:scale*(y+H_c), scale*x:scale*(x+W_c)]
    return hr_c, lr_c

# B, C, H, W
def paired_random_flip_horizontal(hr, lr):
    if np.random.rand() < 0.5:
        return hr[..., :, ::-1], lr[..., :, ::-1]
    return hr, lr

# B, C, H, W
def paired_random_flip_vertical(hr, lr):
    if np.random.rand() < 0.5:
        return hr[..., ::-1, :], lr[..., ::-1, :]
    return hr, lr

def paired_random_rotate(hr, lr):
    x = np.random.randint(0, 4)
    return np.rot90(hr, k=x, axes=(-2, -1)), np.rot90(lr, k=x, axes=(-2, -1))

def normalize(x, mean=0.5, std=4):
    return (x - mean) * std

def denormalize(x, mean=0.5, std=4):
    return torch.clamp((x / std) + mean, min=0, max=1.0)


def rotate_if_wide(img):
    if img.height > img.width:
        return img.rotate(-90, expand=True)
    return img

class PadImg:
    def __init__(self, height, width):
        self.final_width = width
        self.final_height = height

    def __call__(self, img):
        w, h = img.size
        pad_w = max(0, self.final_width - w)
        pad_h = max(0, self.final_height - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        img = F.pad(img, padding, padding_mode='reflect')
        return img


class CropDivisibleBy:
    def __init__(self, divisor=4):
        self.divisor = divisor

    def __call__(self, img):
        w, h = img.size
        new_h = (h // self.divisor) * self.divisor
        new_w = (w // self.divisor) * self.divisor
        if new_h == 0:
            new_h = self.divisor
        if new_w == 0:
            new_w = self.divisor
        cropped_img = F.center_crop(img, (new_h, new_w))
        return cropped_img

    def __repr__(self):
        return self.__class__.__name__ + f'(divisor={self.divisor})'


