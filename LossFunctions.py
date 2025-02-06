
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

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