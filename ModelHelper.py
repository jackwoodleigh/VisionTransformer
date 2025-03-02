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
#from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import ToPILImage
from PIL import Image
from utils import save_images_comparison2, save_images, tensor_to_pil, create_image_grid, infinite_dataloader, calculate_psnr, calculate_ssim
from EMA import ParameterEMA
import copy


class ModelHelper:
    def __init__(self, model, optimizer, criterion, multi_gpu=False, ema_beta=0.995, device="cuda", rank=0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.multi_gpu = multi_gpu
        self.rank = rank
        self.device = f"cuda:{rank}"

        self.ema = ParameterEMA(beta=ema_beta)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False).to(self.device)
        self.scaler = GradScaler()
        self.new_ema = True

    def get_parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, path, name):
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

    def model_call(self, model, lr, use_checkpoint=True):
        if use_checkpoint:
            return checkpoint(model, lr, use_reentrant=False)
        else:
            return model(lr)

    def sample_model(self, lr=None, hr=None,  random_sample=0, dataset=None, save_img=False, save_compare=False, use_ema_model=True):
        if random_sample != 0 and dataset is not None:
            r = random.randint(0, len(dataset)-random_sample)
            stack = [dataset[i] for i in range(r, r + random_sample)]
            hr_list, lr_list = zip(*stack)
            lr = torch.stack(lr_list).to(self.device)
            hr = torch.stack(hr_list).to(self.device)

        if lr is not None:
            with torch.no_grad():
                if use_ema_model:
                    self.ema_model.eval()
                    hr_p = self.ema_model(lr.to(self.device))
                else:
                    self.model.eval()
                    hr_p = self.model(lr.to(self.device))

            if save_img:
                if save_compare:
                    save_images_comparison2(hr, hr_p)
                else:
                    save_images(hr_p)
            else:
                return hr_p, hr if hr is not None else None

    def predict(self, hr, lr, use_ema_model=False):
        with autocast(device_type="cuda"):
            if use_ema_model:
                hr_p = self.model_call(self.ema_model, lr.to(self.device))
            else:
                hr_p = self.model_call(self.model, lr.to(self.device))

            loss = self.criterion(hr_p.to(self.device), hr.to(self.device))

        return loss, hr_p

    def validation_loop(self, test_loader, e, epochs, ema_start_epoch):
        epoch_validation_losses = []
        ssim = []
        psnr = []
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Validating - Epoch {e + 1}/{epochs}", leave=True, dynamic_ncols=True)
            for i, (hr, lr) in enumerate(pbar):
                loss, hr_p = self.predict(hr, lr, use_ema_model=e + 1 > ema_start_epoch or not self.new_ema)
                epoch_validation_losses.append(loss.item())
                ssim.append(calculate_ssim(hr_p.float(), hr.float()).item())
                psnr.append(calculate_psnr(hr_p.float(), hr.float()).mean().item())

        return epoch_validation_losses, ssim, psnr

    def training_loop(self, train_itr, scheduler, itrs_per_epoch, e, total_epochs, accumulation_steps, ema_loss_decay, ema_start_epoch):
        epoch_training_losses = []
        loss_accumulator = 0
        self.model.train()
        pbar = tqdm(range(itrs_per_epoch), disable=(self.rank != 0), desc=f"Training - Epoch {e + 1}/{total_epochs}", leave=True, dynamic_ncols=True)
        for i in pbar:
            # gradient accumulation
            for a in range(accumulation_steps):
                hr, lr = next(train_itr)
                loss, _ = self.predict(hr, lr)
                loss /= accumulation_steps
                loss_accumulator += loss.item()
                self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.ema.step_ema(ema_model=self.ema_model, model=self.model, new_ema=self.new_ema)
            epoch_training_losses.append(loss_accumulator)
            scheduler.step()

            if e + 1 < ema_start_epoch:
                ema_loss = np.mean(epoch_training_losses)
            else:
                ema_loss = ema_loss_decay * ema_loss + (1 - ema_loss_decay) * loss_accumulator

            pbar.set_postfix({
                "Batch Loss": f"{loss_accumulator:.4f}",
                "Epoch Avg Loss": f"{np.mean(epoch_training_losses):.4f}",
                "EMA Batch Loss": f"{ema_loss:.4f}",
                "Learning_Rate": self.optimizer.param_groups[0]['lr']
            })
            loss_accumulator = 0

        return epoch_training_losses

    def train_model(self, train_loader, test_loader, config, train_dataset=None, test_dataset=None, rank=0):
        iterations = config["training"]["iterations"]
        total_epochs = config["training"]["epochs"]
        accumulation_steps = config["training"]["accumulation_steps"]
        ema_start_epoch = config["training"]["ema_start_epoch"]
        save_model_every_i_epoch = config["tools"]["save_model_every_i_epoch"]
        save_path = config["tools"]["save_path"]
        save_name = config["tools"]["save_name"]
        log = config["tools"]["log"]

        self.ema_model.eval()
        self.optimizer.zero_grad(set_to_none=True)

        itrs_per_epoch = (iterations // total_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=iterations, eta_min=1e-5)

        ema_loss_decay = 0.9999
        ema_loss = None
        self.ema.start_step = len(train_loader) * ema_start_epoch // accumulation_steps

        train_itr = infinite_dataloader(train_loader)

        torch.cuda.synchronize()
        time.sleep(2)
        for e in range(total_epochs):
            if self.multi_gpu:
                train_loader.sampler.set_epoch(e)

            # Training
            epoch_training_losses = self.training_loop(train_itr, scheduler, itrs_per_epoch, e, total_epochs, accumulation_steps, ema_loss_decay, ema_start_epoch)

            if self.rank == 0:

                # Validation
                epoch_validation_losses, ssim, psnr = self.validation_loop(test_loader, e, total_epochs, ema_start_epoch)

                torch.cuda.synchronize()
                time.sleep(1)

                # Saving Model
                if save_path != "" and save_model_every_i_epoch != 0 and (e+1) % save_model_every_i_epoch == 0:
                    self.save_model(save_path, f"model_save_{save_name}")

                # Epoch logging
                if log:
                    log = {
                        "Training_Avg_Loss": np.mean(epoch_training_losses),
                        "Training_EMA_Loss": ema_loss,
                        "Validation_Avg_Loss": np.mean(epoch_validation_losses)
                    }
                    if train_dataset is not None:
                        hr_p, hr = self.sample_model(random_sample=6, dataset=test_dataset, use_ema_model=e+1 > ema_start_epoch)
                        pil_images = tensor_to_pil(hr_p)
                        grid_image = create_image_grid(pil_images, grid_size=(3, 2))
                        image = wandb.Image(grid_image, caption="Upscaled Images Grid")
                        log["Image"] = image
                        log["SSIM"] = np.mean(ssim)
                        log["PSNR_"] = np.mean(psnr)

                    wandb.log(log)

            torch.cuda.synchronize()
            print("###################################################################")
            time.sleep(2)

