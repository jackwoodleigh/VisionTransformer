import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision.models as models

class SuperResolutionDataset(Dataset):
    def __init__(self, root, scale_values, base_transforms):
        self.root = root
        self.scale_values = scale_values

        self.blur_transforms = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))

        self.dataset = datasets.ImageFolder(root=root, transform=base_transforms)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        hr_img, _ = self.dataset[idx]
        lr_img = self.blur_transforms(hr_img)
        lr_img = F.interpolate(lr_img.unsqueeze(0), scale_factor=(1 / self.scale_values), mode='area')

        return hr_img, lr_img.squeeze(0)

class PerceptualLoss(nn.Module):
    def __init__(self, layer_index=16):
        super().__init__()
        vgg = models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1").features
        self.features = nn.Sequential(*list(vgg.children())[:layer_index])
        self.features.eval()

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, predicted, target):
        predicted_features = self.features(predicted)
        target_features = self.features(target)
        return F.mse_loss(predicted_features, target_features)

class FFTLoss(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss()):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, predicted, target):

        predicted_fft = torch.fft.fft2(predicted.float())
        target_fft = torch.fft.fft2(target)

        predicted_mag = torch.abs(predicted_fft)
        target_mag = torch.abs(target_fft)

        return self.loss_fn(predicted_mag, target_mag)

# https://www.sciencedirect.com/science/article/pii/S0031320323002108 patch loss