import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SuperResolutionDataset(Dataset):
    def __init__(self, root, scale_values, transform, subset=None):
        self.root = root
        self.scale_values = scale_values

        self.transform = transform

        self.dataset = datasets.ImageFolder(root=root)

        if subset is not None:
            subset_indices = list(range(subset))
            self.dataset = Subset(self.dataset, subset_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr_img, _ = self.dataset[idx]
        hr_img = self.transform(hr_img)
        lr_img = F.interpolate(hr_img.unsqueeze(0), scale_factor=(1 / self.scale_values), mode='bicubic')
        return hr_img, lr_img.squeeze(0)

def denormalize_image(img, mean, std):
    device = img.device
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)
    img_denorm = img * std + mean
    img_denorm = torch.clamp(img_denorm, 0.0, 1.0)
    return img_denorm

def save_images_comparison(hr_p, hr, filename="comparison.png", denorm=True):
    if denorm:
        hr_p = denormalize_image(hr_p, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        hr = denormalize_image(hr, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    comparison = torch.cat((hr, hr_p), dim=-1)
    vutils.save_image(comparison, filename, nrow=hr.shape[0])
    print("Saved Image Comparison.")

def save_images(tensor, filename="images.png", denorm=True):
    if denorm:
        tensor = denormalize_image(tensor, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    vutils.save_image(tensor, filename, nrow=tensor.shape[0])
    print("Saved Images.")

def tensor_to_pil(tensor):
    tensor = denormalize_image(tensor, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
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

def calculate_psnr(hr_p, hr, max_pixel_value=1.0, denorm=True):
    if denorm:
        hr_p = denormalize_image(hr_p, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        hr = denormalize_image(hr, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])

    mse = F.mse_loss(hr_p, hr, reduction="none")
    mse = mse.mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)
    psnr[mse == 0] = float('inf')
    return psnr

def calculate_ssim(hr_p, hr, denorm=False):
    if denorm:
        hr_p = denormalize_image(hr_p, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        hr = denormalize_image(hr, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(hr_p.device)
    return ssim(hr_p, hr)
