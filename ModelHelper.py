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
from utils import save_images_comparison2, save_images, tensor_to_pil, create_image_grid, infinite_dataloader, calculate_psnr_pt, calculate_ssim_pt
from EMA import ParameterEMA
import copy


class ModelHelper:
    def __init__(self, model, optimizer, scheduler, criterion, multi_gpu=False, ema_beta=0.995, device="cuda", rank=0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.multi_gpu = multi_gpu
        self.rank = rank
        if device == "cuda":
            self.device = f"cuda:{rank}"
        else:
            self.device = device

        self.ema = ParameterEMA(beta=ema_beta)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False).to(self.device)
        self.scaler = GradScaler()
        self.new_ema = True

    def get_parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, directory, file_name, config=None):
        path = os.path.join(directory, "ModelSaves")
        if not os.path.exists(path):
            os.makedirs(path)

        save = {
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if config is not None:
            save["config"] = config

        file_name = f'{file_name}.pth'
        path = os.path.join(path, file_name)
        torch.save(save, path)
        print(f"Saved model to: {path}")

    def load_model(self, directory, file_name, load_optimizer=True):
        path = os.path.join(directory, "ModelSaves", file_name)
        save = torch.load(path)

        self.model.load_state_dict(save['model_state_dict'])
        self.ema_model.load_state_dict(save['ema_model_state_dict'])
        self.new_ema = False
        if load_optimizer:
            self.optimizer.load_state_dict(save['optimizer_state_dict'])

        print(f"Loaded model from: {path}")

    def sync(self, sec=0, clear_cache=False):
        if self.multi_gpu:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        if clear_cache:
            torch.cuda.empty_cache()
        if sec != 0:
            time.sleep(sec)

    def sample_model(self, random_sample, dataset, save_img=False, save_compare=False, use_ema_model=True):

        '''r = [23, 50, 78]
        stack = [dataset[i] for i in r]'''
        r = random.randint(0, len(dataset) - random_sample)
        stack = [dataset[i] for i in range(r, r + random_sample)]
        hr_list, lr_list = zip(*stack)
        lr = torch.stack(lr_list).to(self.device)
        hr = torch.stack(hr_list).to(self.device)

        with torch.no_grad():
            if use_ema_model:
                self.ema_model.eval()
                model = self.ema_model
            else:
                self.model.eval()
                model = self.model

            hr_p = []
            for i in range(random_sample):
                hr_p.append(self.model_call(model, lr[i].unsqueeze(0)).squeeze(0))
            hr_p = torch.stack(hr_p, dim=0)

        if save_img:
            if save_compare:
                save_images_comparison2(hr_p, hr)
            else:
                save_images(hr_p)
        else:
            return hr_p, hr if hr is not None else None

    def model_call(self, lr, use_ema_model=False, use_checkpoint=True, no_grad=False):
        if use_ema_model:
            model = self.ema_model
        else:
            model = self.model
        if no_grad:
            with torch.no_grad():
                if use_checkpoint:
                    return checkpoint(model, lr, use_reentrant=False)
                else:
                    return model(lr)
        else:
            if use_checkpoint:
                return checkpoint(model, lr, use_reentrant=False)
            else:
                return model(lr)

    def predict(self, hr, lr, use_ema_model=False):
        with autocast(device_type="cuda"):
            hr_p = self.model_call(lr.to(self.device), use_ema_model=use_ema_model)
            loss = self.criterion(hr_p.to(self.device), hr.to(self.device))
        return loss, hr_p

    def update_params(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.step_ema(ema_model=self.ema_model, model=self.model, new_ema=self.new_ema)

    def training_loop(self, train_itr, itrs_per_epoch, e, total_epochs, accumulation_steps, ema_loss, ema_loss_decay, ema_start_epoch):
        self.model.train()
        epoch_training_losses = []
        loss_accumulator = 0
        pbar = tqdm(range(itrs_per_epoch), disable=(self.rank != 0), desc=f"Training - Epoch {e + 1}/{total_epochs}", leave=True, dynamic_ncols=True)
        for _ in pbar:
            # gradient accumulation
            for _ in range(accumulation_steps):
                hr, lr = next(train_itr)
                loss, _ = self.predict(hr, lr)
                loss /= accumulation_steps
                loss_accumulator += loss.item()
                self.scaler.scale(loss).backward()

            self.update_params()
            epoch_training_losses.append(loss_accumulator)
            self.scheduler.step()

            if e + 1 < ema_start_epoch:
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

        return epoch_training_losses, ema_loss

    def validation_loop(self, test_loader, e, epochs, ema_start_epoch):
        epoch_validation_losses = []
        ssim = []
        psnr = []
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            pbar = tqdm(test_loader, disable=(self.rank != 0), desc=f"Validating - Epoch {e + 1}/{epochs}", leave=True, dynamic_ncols=True)
            for i, (hr, lr) in enumerate(pbar):
                loss, hr_p = self.predict(hr, lr, use_ema_model=e + 1 > ema_start_epoch or not self.new_ema)
                epoch_validation_losses.append(loss.item())
                ssim.append(calculate_ssim_pt(hr_p.float(), hr.to(self.device).float(), crop_border=5).mean().item())
                psnr.append(calculate_psnr_pt(hr_p.float(), hr.to(self.device).float(), crop_border=5).mean().item())

        return epoch_validation_losses, ssim, psnr

    def train_model(self, train_loader, test_loader, config, train_dataset=None, test_dataset=None):
        iterations = config["training"]["iterations"]
        total_epochs = config["training"]["epochs"]
        accumulation_steps = config["training"]["accumulation_steps"]
        ema_start_epoch = config["training"]["ema_start_epoch"]
        save_model_every_i_epoch = config["tools"]["save_model_every_i_epoch"]
        path = config["tools"]["logging_path"]
        model_save_name = config["tools"]["model_save_name"]
        log = config["tools"]["wandb_log"]

        self.ema_model.eval()
        self.optimizer.zero_grad(set_to_none=True)

        ema_loss_decay = 0.9999
        ema_loss = None
        self.ema.start_step = len(train_loader) * ema_start_epoch // accumulation_steps

        itrs_per_epoch = (iterations // total_epochs)
        train_itr = infinite_dataloader(train_loader)
        self.sync(sec=2)

        for e in range(total_epochs):
            if self.multi_gpu:
                train_loader.sampler.set_epoch(e)

            # Training Loop
            epoch_training_losses, ema_loss = self.training_loop(train_itr, itrs_per_epoch, e, total_epochs, accumulation_steps, ema_loss, ema_loss_decay, ema_start_epoch)

            self.sync(sec=2)

            # Validation Loop
            epoch_validation_losses, ssim, psnr = self.validation_loop(test_loader, e, total_epochs, ema_start_epoch)

            self.sync(sec=2, clear_cache=True)

            if self.rank == 0:

                # Saving Model
                if path != "" and save_model_every_i_epoch != 0 and (e+1) % save_model_every_i_epoch == 0:
                    self.save_model(path, f"model_save_{model_save_name}", config)

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

            self.sync(clear_cache=True)
            if self.rank == 0:
                print("###################################################################")
            time.sleep(2)

