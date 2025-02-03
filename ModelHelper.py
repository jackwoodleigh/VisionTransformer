import torch
import wandb
from tqdm import tqdm
import torchvision.models as models
import torch
from torch import nn
from torch.nn.functional import mse_loss
import numpy as np

class PerceptualLoss(nn.Module):
    def __init__(self, layer_index=16):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:layer_index])
        self.features.eval()

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, predicted, target):
        predicted_features = self.features(predicted)
        target_features = self.features(target)
        return mse_loss(predicted_features, target_features)


class ModelHelper:
    def __init__(self, model, optimizer, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.perceptual_loss = PerceptualLoss().to(device)
        self.device = device

        # TODO add an EMA

    def get_parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, hr, lr, pl_scale):
        lr = lr.to(self.device)
        hr = hr.to(self.device)
        hr_p = self.model(lr)
        loss = mse_loss(hr_p, hr) + pl_scale * self.perceptual_loss(hr_p, hr)
        return loss

    def train_model(self, train_loader, test_loader, epochs, batches_per_epoch, pl_scale, log=False, save_path="save.pt"):
        self.model.train()
        i = 0
        self.optimizer.zero_grad()

        for e in range(epochs):
            epoch_training_loss = []
            epoch_validation_loss = []
            print(f"Epoch {e}...")

            # Training
            for hr, lr in tqdm(train_loader):
                loss = self.predict(hr, lr, pl_scale)
                loss /= batches_per_epoch
                loss.backward()
                epoch_training_loss.append(loss.item())
                i += 1

                # might be some carry over with this
                if i == batches_per_epoch:
                    i = 0
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Validation
            with torch.no_grad():
                for hr, lr in tqdm(test_loader):
                    loss = self.predict(hr, lr, pl_scale)
                    epoch_validation_loss.append(loss.item())


            print(f"Training Loss: {np.mean(epoch_training_loss)}, Validation Loss: {np.mean(epoch_validation_loss)}")
            # Epoch logging
            if log:
                pil_image = self.sample(8, torch.tensor([1], device=self.device))
                image = wandb.Image(pil_image, caption=f"class 2")
                wandb.log(
                    {"Training_Loss": epoch_training_loss, "Validation_Loss": epoch_validation_loss,
                     "Sample": image})
                torch.save(self.EMA_model.state_dict(), save_path)
                print("Model Saved.")

