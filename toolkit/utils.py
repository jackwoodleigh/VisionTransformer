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


class SuperResolutionDataset(Dataset):
    def __init__(self, root, scale_values, transform=None, use_lmdb=False, enlarge_factor=1, subset=0):
        self.root = root
        self.scale_values = scale_values
        self.transform = transform
        self.enlarge_factor = enlarge_factor
        self.use_lmdb = use_lmdb
        self.env = None

        if use_lmdb:
            self.n = len(get_keys(self.root))
            if subset != 0:
                assert self.n > subset
                self.n = subset

        else:
            self.dataset = datasets.ImageFolder(root=root)

            if subset != 0:
                subset_indices = list(range(subset))
                self.dataset = Subset(self.dataset, subset_indices)

            self.n = len(self.dataset)

    def open_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.root,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
            self.env = None

    def __len__(self):
        return self.n * self.enlarge_factor

    def __getitem__(self, idx):
        idx = idx % self.n

        if self.use_lmdb:
            self.open_env()
            key = f"{idx:08d}".encode('ascii')

            with self.env.begin() as txn:
                img_bytes = txn.get(key)
                if img_bytes is None:
                    raise KeyError(f"Key {key} not found in LMDB!")

            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            hr_img = Image.fromarray(hr_img)

            if self.transform is not None:
                hr_img = self.transform(hr_img)

            lr_img = F.interpolate(hr_img.unsqueeze(0), scale_factor=(1 / self.scale_values), mode='bicubic')

        else:
            hr_img, _ = self.dataset[idx]
            if self.transform is not None:
                hr_img = self.transform(hr_img)
            lr_img = F.interpolate(hr_img.unsqueeze(0), scale_factor=(1 / self.scale_values), mode='bicubic')

        return hr_img, lr_img.squeeze(0)


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def denormalize_image(img, mean, std):
    device = img.device
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)
    img_denorm = img * std + mean
    img_denorm = torch.clamp(img_denorm, 0.0, 1.0)
    return img_denorm

def save_images_comparison(hr_p, hr, filename="comparison.png", denorm=False, compare_bicubic=True):
    if denorm:
        hr_p = denormalize_image(hr_p, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        hr = denormalize_image(hr, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
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

def read_from_lmdb(lmdb_path, key):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        data = txn.get(key.encode('ascii'))
        if data is None:
            print("Key not found!")
            return None
        img_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    env.close()
    return img

def get_keys(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        total_images = int(txn.get(b'__len__').decode('utf-8'))
        keys = {i: f"{i:08d}" for i in range(total_images)}
    env.close()
    return keys