
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
        predicted_fft = torch.fft.rfft2(predicted.float(), norm='ortho')
        target_fft = torch.fft.rfft2(target.float(), norm='ortho')

        predicted_fft = torch.stack([predicted_fft.real, predicted_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_fn(predicted_fft, target_fft)

# https://www.sciencedirect.com/science/article/pii/S0031320323002108 patch loss