import io
import os
from os import path as osp

import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lmdb

import kornia.color
import cv2

def save_images_comparison(hr_p, hr, filename="comparison.png", compare_bicubic=True):
    imgs = [hr]
    if compare_bicubic:
        lr = F.interpolate(hr, scale_factor=(1 / 4), mode='bicubic')
        hr_bicubic = F.interpolate(lr, scale_factor=4, mode='bicubic')
        imgs.append(hr_bicubic)

    imgs.append(hr_p)

    comparison = torch.cat(imgs, dim=-1)
    torchvision.utils.save_image(comparison, filename, nrow=1, padding=2, normalize=True)
    print("Saved Image Comparison.")


def save_images(tensor, filename="images.png"):
    vutils.save_image(tensor, filename, nrow=tensor.shape[0])
    print("Saved Images.")

def tensor_to_pil(tensor):
    to_pil = transforms.ToPILImage()
    return [to_pil(t) for t in tensor]

def create_image_grid(images, grid_size):
    img_width, img_height = images[0].size
    grid_width, grid_height = grid_size
    grid_img = Image.new('RGB', (grid_width * img_width, grid_height * img_height))
    for index, img in enumerate(images):
        x = (index % grid_width) * img_width
        y = (index // grid_width) * img_height
        grid_img.paste(img, (x, y))
    return grid_img


def calculate_psnr_pt_y_channel(img_rgb, img2_rgb, crop_border, test_y_channel=True):
    assert img_rgb.shape == img2_rgb.shape, (f'Image shapes are different: {img_rgb.shape}, {img2_rgb.shape}.')
    assert img_rgb.dim() == 4 and img_rgb.shape[1] == 3, f'Input should be Bx3xHxW, got {img_rgb.shape}'

    if crop_border != 0:
        img_rgb = img_rgb[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2_rgb = img2_rgb[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img_y = kornia.color.rgb_to_ycbcr(img_rgb)[:, 0:1, :, :]
        img2_y = kornia.color.rgb_to_ycbcr(img2_rgb)[:, 0:1, :, :]
        img = img_y
        img2 = img2_y
        max_val = 1.0
    else:
        img = img_rgb
        img2 = img2_rgb
        max_val = 1.0

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])

    psnr = 10. * torch.log10(max_val**2 / (mse + 1e-10))

    return torch.clamp(psnr, min=0, max=100.)

def calculate_ssim_pt_y_channel(img_rgb, img2_rgb, crop_border, test_y_channel=True):

    assert img_rgb.shape == img2_rgb.shape, (f'Image shapes are different: {img_rgb.shape}, {img2_rgb.shape}.')
    assert img_rgb.dim() == 4 and img_rgb.shape[1] == 3, f'Input should be Bx3xHxW, got {img_rgb.shape}'

    if crop_border != 0:
        img_rgb = img_rgb[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2_rgb = img2_rgb[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img_y = kornia.color.rgb_to_ycbcr(img_rgb)[:, 0:1, :, :]
        img2_y = kornia.color.rgb_to_ycbcr(img2_rgb)[:, 0:1, :, :]
        img = img_y
        img2 = img2_y
    else:
        img = img_rgb
        img2 = img2_rgb

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim_val = _ssim_pth(img * 255., img2 * 255.)
    return ssim_val

def _ssim_pth(img, img2):
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')
    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def format_image_files(data_root, flatten=False):
    img_list = list(scandir(data_root, recursive=True, full_path=True))
    sub_directories = set()
    supported_extensions = ['.png', '.jpeg', '.jpg']
    for idx, src in enumerate(img_list):
        extension = osp.splitext(src)[1]
        if extension not in supported_extensions:
            raise Exception(f"Found unsupported file extension at {src}")
        new_name = f"img_{idx:08d}{extension}"
        if osp.dirname(src) != data_root and osp.dirname(src) not in sub_directories:
            sub_directories.add(osp.dirname(src))

        if not flatten and data_root != osp.dirname(src):
            dst = osp.join(osp.dirname(src), new_name)
        else:
            dst = osp.join(data_root, new_name)
        os.rename(src, dst)

    if flatten:
        for dir in sub_directories:
            if not os.listdir(dir):
                os.rmdir(dir)
            else:
                raise Exception(f"Failed to delete {dir} when flattening. It is not empty.")





