import torch
import wandb
from tqdm import tqdm
import torchvision.models as models
import torch
from torch import nn
from torch.nn.functional import mse_loss
import numpy as np
from torch.utils.checkpoint import checkpoint

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
        # TODO add mixed precision
        # TODO add scheduler

    def get_parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, hr, lr, pl_scale):
        lr = lr.to(self.device)
        hr = hr.to(self.device)
        hr_p = checkpoint(self.model, lr)   # insane memory issue saver
        loss = mse_loss(hr_p, hr) + pl_scale * self.perceptual_loss(hr_p, hr)
        return loss

    def train_model(self, train_loader, test_loader, epochs, accumulation_steps, pl_scale, log=False, save_path="save.pt"):
        self.model.train()
        n = len(train_loader)
        self.optimizer.zero_grad()

        for e in range(epochs):
            epoch_training_loss = []
            epoch_validation_loss = []
            print(f"Epoch {e}...")

            # Training
            for i, (hr, lr) in enumerate(tqdm(train_loader)):
                loss = self.predict(hr, lr, pl_scale)
                loss /= accumulation_steps
                loss.backward()
                epoch_training_loss.append(loss.item())
                i += 1

                # mini-batch
                if i % accumulation_steps == 0 or (i + 1) == n:
                    torch.cuda.synchronize()
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

