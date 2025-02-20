import os
import random
import time

import torch
import wandb
from tqdm.auto import tqdm
from torch.nn.functional import mse_loss
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from LossFunctions import PerceptualLoss, FFTLoss
from torchvision.transforms import ToPILImage
from PIL import Image
from utils import save_images_comparison, save_images, tensor_to_pil, create_image_grid, denormalize_image, calculate_psnr, calculate_ssim
from EMA import ParameterEMA
import copy

class ModelHelper:
    def __init__(self, model, optimizer, ema_beta=0.995, device="cuda"):
        self.model = model.to(device)
        self.ema = ParameterEMA(beta=ema_beta)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        self.optimizer = optimizer
        self.device = device
        self.perceptual_loss = PerceptualLoss().to(device)
        self.fft_loss = FFTLoss().to(device)
        self.scaler = GradScaler()
        self.new_ema = True

    def get_parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, name, path):
        if not os.path.exists(path):
            os.makedirs(path)
        save = {
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        file_name = f'{name}.pth'
        torch.save(save, os.path.join(path, file_name))
        print(f"Saved model to: {os.path.join(path, file_name)}")

    def load_model(self, directory, file_name, load_optimizer=True):
        file_path = os.path.join(directory, file_name)
        save = torch.load(file_path)
        self.model.load_state_dict(save['model_state_dict'])
        self.ema_model.load_state_dict(save['ema_model_state_dict'])
        self.new_ema = False
        if load_optimizer:
            self.optimizer.load_state_dict(save['optimizer_state_dict'])

        print(f"Loaded model from: {file_path}")

    def sample_model(self, lr=None, hr=None,  random_sample=0, dataset=None, save_img=False, compare=False, use_ema_model=True):
        if random_sample != 0 and dataset is not None:
            r = random.randint(0, len(dataset)-random_sample)
            lr = torch.stack([dataset[i][1] for i in range(r, r + random_sample)]).to("cuda")
            hr = torch.stack([dataset[i][0] for i in range(r, r + random_sample)]).to("cuda")
        if lr is not None:
            with torch.no_grad():
                if use_ema_model:
                    self.ema_model.eval()
                    hr_p = self.ema_model(lr.to(self.device))
                else:
                    self.model.eval()
                    hr_p = self.model(lr.to(self.device))

            if save_img:
                if compare:
                    save_images_comparison(hr, hr_p)
                else:
                    save_images(hr_p)
            else:

                return hr_p, hr if hr is not None else None

    def predict(self, hr, lr, pl_scale, fft_loss_scale, use_ema_model=False):
        lr = lr.to(self.device).requires_grad_()
        hr = hr.to(self.device)

        with autocast(device_type="cuda"):
            if use_ema_model:
                hr_p = checkpoint(self.ema_model, lr, use_reentrant=False)
            else:
                hr_p = checkpoint(self.model, lr, use_reentrant=False)   # insane memory issue saver

            loss = mse_loss(hr_p, hr) + (pl_scale * self.perceptual_loss(hr_p, hr)) + (fft_loss_scale * self.fft_loss(hr_p, hr))

        return loss, hr_p
    
    def train_model(self, train_loader, test_loader, epochs, accumulation_steps, pl_scale, fft_loss_scale, ema_start_epoch=3, log=False, save_model_every_i_epoch=1, save_path="", dataset=None):
        self.ema_model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        n = len(train_loader)
        #effective_ema_loss_decay = 0.999 ** accumulation_steps
        ema_loss_decay = 0.995
        ema_loss = None

        torch.cuda.synchronize()
        time.sleep(2)
        train_itr = iter(train_loader)
        for itr in range(10000):
            epoch_training_losses = []
            epoch_validation_losses = []
            loss_accumulator = 0
            self.model.train()

            # Training
            #pbar = tqdm(train_loader, desc=f"Training - Epoch {e+1}/{epochs}", leave=True, dynamic_ncols=True)
            for i, (hr, lr) in enumerate(pbar):
                hr, lr =
                loss, _ = self.predict(hr, lr, pl_scale, fft_loss_scale)
                loss /= accumulation_steps
                loss_accumulator += loss.item()
                self.scaler.scale(loss).backward()

                # gradient accumulation
                if (i + 1) % accumulation_steps == 0 or (i + 1) == n:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.ema.step_ema(ema_model=self.ema_model, model=self.model, start_step=len(train_loader)*ema_start_epoch // accumulation_steps, new_ema=self.new_ema)
                    epoch_training_losses.append(loss_accumulator)

                    if e+1 < ema_start_epoch:
                        ema_loss = np.mean(epoch_training_losses)
                    else:
                        ema_loss = ema_loss_decay * ema_loss + (1 - ema_loss_decay) * loss_accumulator

                    pbar.set_postfix({
                        "Batch Loss": f"{loss_accumulator:.5f}",
                        "Epoch Avg Loss": f"{np.mean(epoch_training_losses):.5f}",
                        "EMA Batch Loss": f"{ema_loss:.5f}",
                        "Learning_Rate": self.optimizer.param_groups[0]['lr']
                    })
                    loss_accumulator = 0

            #scheduler.step()

            ssim = []
            psnr = []
            # Validation
            with torch.no_grad():
                pbar = tqdm(test_loader, desc=f"Validating - Epoch {e + 1}/{epochs}", leave=True, dynamic_ncols=True)
                for i, (hr, lr) in enumerate(pbar):
                    loss, hr_p = self.predict(hr, lr, pl_scale, fft_loss_scale, use_ema_model=e+1 > ema_start_epoch or not self.new_ema)
                    epoch_validation_losses.append(loss.item())
                    hr = hr.to(self.device)
                    ssim.append(calculate_ssim(hr_p.float(), hr.float()).item())
                    psnr.append(calculate_psnr(hr_p.float(), hr.float()).mean().item())

            torch.cuda.synchronize()
            time.sleep(1)
            # Saving Model
            if save_path != "" and save_model_every_i_epoch != 0 and (e+1) % save_model_every_i_epoch == 0:
                self.save_model(f"model_save_epoch_{e}", save_path)

            # Epoch logging
            if log:
                log = {
                    "Training_Avg_Loss": np.mean(epoch_training_losses),
                    "Training_EMA_Loss": ema_loss,
                    "Validation_Avg_Loss": np.mean(epoch_validation_losses)
                }
                if dataset is not None:
                    hr_p, hr = self.sample_model(random_sample=6, dataset=dataset, use_ema_model=e+1 > ema_start_epoch)
                    pil_images = tensor_to_pil(hr_p)
                    grid_image = create_image_grid(pil_images, grid_size=(3, 2))
                    image = wandb.Image(grid_image, caption="Upscaled Images Grid")
                    log["Image"] = image
                    log["SSIM"] = np.mean(ssim)
                    log["PSNR_"] = np.mean(psnr)

                wandb.log(log)

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("###################################################################")
            time.sleep(2)

