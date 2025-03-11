import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from utils import denormalize_image


class PerceptualLoss(nn.Module):
    def __init__(self, layer_index=12):
        super().__init__()
        vgg = models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1").features
        self.features = nn.Sequential(*list(vgg.children())[:layer_index]).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        for param in self.features.parameters():
            param.requires_grad = False

    def norm(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, predicted, target):
        predicted = denormalize_image(predicted, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        target = denormalize_image(target, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        predicted_features = self.features(self.norm(predicted))
        target_features = self.features(self.norm(target.detach()))
        return F.mse_loss(predicted_features, target_features)


class FFTLoss(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss()):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, predicted, target):
        predicted_fft = torch.fft.rfft2(predicted.float())
        target_fft = torch.fft.rfft2(target.float())

        predicted_fft = torch.stack([predicted_fft.real, predicted_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_fn(predicted_fft, target_fft)

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = torch.add(prediction, -target)
        diff_sq = diff * diff
        diff_sq_color = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq_color + self.epsilon)
        return torch.mean(error)

class Criterion(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = []
        if "L1" in losses:
            self.losses.append([nn.L1Loss(), losses["L1"]])
        if "L2" in losses:
            self.losses.append([nn.MSELoss(), losses["L2"]])
        if "CharbonnierLoss" in losses:
            self.losses.append([CharbonnierLoss(), losses["CharbonnierLoss"]])
        if "PerceptualLoss" in losses:
            self.losses.append([PerceptualLoss(), losses["PerceptualLoss"]])
        if "FFTLoss" in losses:
            self.losses.append([FFTLoss(), losses["FFTLoss"]])
    def forward(self, predicted, target):
        total_loss = 0
        for loss, loss_weight in self.losses:
            total_loss += loss_weight * loss(predicted, target)
        return total_loss






# https://www.sciencedirect.com/science/article/pii/S0031320323002108 patch loss

# https://github.com/jonbarron/robust_loss_pytorch/tree/master

