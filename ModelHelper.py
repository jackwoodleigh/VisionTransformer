import os
import torch
import wandb
from tqdm.auto import tqdm
from torch.nn.functional import mse_loss
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from utils import PerceptualLoss, FFTLoss


class ModelHelper:
    def __init__(self, model, optimizer, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.perceptual_loss = PerceptualLoss().to(device)
        self.fft_loss = FFTLoss().to(device)
        self.scaler = GradScaler()

        # TODO add sampling function
        # TODO finish weights and bias
        # TODO add scheduler
        # TODO add patch loss
        # TODO add an EMA
        # TODO add diffusion refinement

    def get_parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, name, path):
        if not os.path.exists(path):
            os.makedirs(path)
        save = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        file_name = f'{name}.pth'
        torch.save(save, os.path.join(path, file_name))
        print(f"Saved model to: {os.path.join(path, file_name)}")

    def load_model(self, path, load_optimizer=True):
        save = torch.load(path)
        self.model.load_state_dict(save['model_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(save['optimizer_state_dict'])

        print(f"Loaded model from: {path}")

    def sample_model_PIL(self, image):
        pass

    def predict(self, hr, lr, pl_scale, fft_loss_scale):
        lr = lr.to(self.device).requires_grad_()
        hr = hr.to(self.device)

        with autocast():
            hr_p = checkpoint(self.model, lr, use_reentrant=False)   # insane memory issue saver
            loss = mse_loss(hr_p, hr) + (pl_scale * self.perceptual_loss(hr_p, hr)) + (fft_loss_scale * self.fft_loss(hr_p, hr))

        return loss, hr_p

    def train_model(self, train_loader, test_loader, epochs, accumulation_steps, pl_scale, fft_loss_scale, log=False, save_model_every_i_epoch=1, save_path=""):
        self.model.train()
        self.optimizer.zero_grad()
        n = len(train_loader)
        ema_loss_decay = 0.999
        ema_loss = None

        for e in range(epochs):
            epoch_training_losses = []
            epoch_validation_losses = []
            loss_accumulator = 0

            # Training
            pbar = tqdm(train_loader, desc=f"Epoch {e+1}/{epochs}", leave=True, dynamic_ncols=True)
            for i, (hr, lr) in enumerate(pbar):
                loss, _ = self.predict(hr, lr, pl_scale, fft_loss_scale)
                loss /= accumulation_steps
                loss_accumulator += loss.item()
                self.scaler.scale(loss).backward()

                # gradient accumulation
                if (i + 1) % accumulation_steps == 0 or (i + 1) == n:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if ema_loss is None:
                        ema_loss = loss_accumulator
                    else:
                        ema_loss = ema_loss_decay * ema_loss + (1 - ema_loss_decay) * loss_accumulator

                    epoch_training_losses.append(loss_accumulator)
                    pbar.set_postfix({
                        "Batch Loss": f"{loss_accumulator:.4f}",
                        "EMA Batch Loss": f"{ema_loss:.4f}",
                        "Epoch Avg Loss": f"{np.mean(epoch_training_losses):.4f}"
                    })
                    loss_accumulator = 0

            '''# Validation
            with torch.no_grad():
                for hr, lr in tqdm(test_loader):
                    loss, _ = self.predict(hr, lr, pl_scale)
                    epoch_validation_loss.append(loss.item())'''

            if save_path != "" and save_model_every_i_epoch != 0 and (e+1) % save_model_every_i_epoch == 0:
                self.save_model(f"model_save_epoch_{e}", save_path)

            # Epoch logging
            if log:
                pil_image = self.sample(8, torch.tensor([1], device=self.device))
                image = wandb.Image(pil_image, caption=f"class 2")
                wandb.log(
                    {"Training_Loss": epoch_training_losses, "Validation_Loss": epoch_validation_losses,
                     "Sample": image})
                torch.save(self.EMA_model.state_dict(), save_path)
                print("Model Saved.")

