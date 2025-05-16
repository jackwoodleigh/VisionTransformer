import os
import random
import time

import torch
import wandb
from tqdm.auto import tqdm
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
#from torch.cuda.amp import autocast, GradScaler
from toolkit.utils import save_images_comparison, save_images, tensor_to_pil, create_image_grid, calculate_psnr_pt_y_channel, calculate_ssim_pt_y_channel
from toolkit.param_ema import ParameterEMA
import copy
from toolkit.datasets import infinite_dataloader
from .transforms import denormalize
import torch.nn.functional as F

class ModelHelper:
    def __init__(self, model, optimizer, criterion, scheduler=None, multi_gpu=False, ema_beta=0.995, device="cuda", rank=0):
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

        self.ema_loss_decay = 0.999

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

    def load_model(self, src, load_optimizer=True):
        save = torch.load(src)

        self.model.load_state_dict(save['model_state_dict'])
        self.ema_model.load_state_dict(save['ema_model_state_dict'])
        self.new_ema = False
        if load_optimizer:
            self.optimizer.load_state_dict(save['optimizer_state_dict'])

        print(f"Loaded model from: {src}")

    def sync(self, sec=0, clear_cache=False):
        if self.multi_gpu:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        if clear_cache:
            torch.cuda.empty_cache()
        if sec != 0:
            time.sleep(sec)

    def sample_model(self, random_sample, dataset, save_img=False, save_compare=False, use_ema_model=True):
        r = random.randint(0, len(dataset) - random_sample)
        stack = [dataset[i] for i in range(r, r + random_sample)]
        hr_list, lr_list = zip(*stack)
        lr = torch.stack(lr_list).to(self.device)
        hr = torch.stack(hr_list).to(self.device)

        hr_p = []
        for i in range(random_sample):
            hr_p.append(self.model_call(lr[i].unsqueeze(0), use_ema_model=use_ema_model, no_grad=True).squeeze(0))
        hr_p = torch.stack(hr_p, dim=0)

        if save_img:
            if save_compare:
                save_images_comparison(hr_p, hr)
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

    def predict(self, hr, lr, use_ema_model=False, use_checkpoint=True, no_grad=False):
        with autocast(device_type="cuda"):
            hr_p = self.model_call(lr, use_ema_model=use_ema_model, use_checkpoint=use_checkpoint, no_grad=no_grad)
            loss = self.criterion(hr_p, hr)
        return loss, hr_p

    def update_params(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.step_ema(ema_model=self.ema_model, model=self.model, new_ema=self.new_ema)

    def train_loop(self, itr, n, accumulation_steps, desc, ema_loss=None):
        self.model.train()
        losses = []
        pbar = tqdm(range(n), disable=(self.rank != 0), desc=desc, leave=True, dynamic_ncols=True)
        for _ in pbar:
            loss = 0
            for _ in range(accumulation_steps):
                hr, lr = next(itr)
                hr, lr = hr.to(self.device), lr.to(self.device)
                loss_b, _ = self.predict(hr, lr)
                loss_b /= accumulation_steps
                loss += loss_b.item()
                self.scaler.scale(loss_b).backward()

            self.update_params()
            losses.append(loss)

            if self.scheduler is not None:
                self.scheduler.step()

            pbar_postfix_data = {
                "Batch Loss": f"{loss/2:.5f}",
                "Avg Loss": f"{np.mean(losses)/2:.5f}",
                "Learning_Rate": self.optimizer.param_groups[0]['lr']
            }

            if ema_loss is not None:
                ema_loss = self.ema_loss_decay * ema_loss + (1 - self.ema_loss_decay) * loss
                pbar_postfix_data["EMA Loss"] = f"{ema_loss:.5f}"

            pbar.set_postfix(pbar_postfix_data)

        return losses, ema_loss

    def validate_loop(self, itr, n, desc, use_ema_model=False, use_metrics=True):
        self.model.eval()
        losses = []
        metrics = {'ssim': [], 'psnr': []}
        pbar = tqdm(range(n), disable=(self.rank != 0), desc=desc, leave=True, dynamic_ncols=True)
        for _ in pbar:
            hr, lr = next(itr)
            hr, lr = hr.to(self.device), lr.to(self.device)
            loss, hr_p = self.predict(hr, lr, no_grad=True, use_ema_model=use_ema_model)
            losses.append(loss.item())

            pbar_postfix_data = {
                "Batch Loss": f"{loss:.5f}",
                "Avg Loss": f"{np.mean(losses):.5f}",
            }

            if use_metrics:
                metrics['ssim'].append(calculate_ssim_pt_y_channel(denormalize(hr_p), denormalize(hr.to(self.device)), crop_border=3).mean().item())
                metrics['psnr'].append(calculate_psnr_pt_y_channel(denormalize(hr_p), denormalize(hr.to(self.device)), crop_border=3).mean().item())
                pbar_postfix_data['SSIM'] = metrics['ssim'][-1]
                pbar_postfix_data['PSNR'] = metrics['psnr'][-1]

            pbar.set_postfix(pbar_postfix_data)

        return losses, metrics

    def train_model(self, train_loader, test_loader, config, train_dataset=None, test_dataset=None):
        iterations = config["training"]["iterations"]
        total_epochs = config["training"]["epochs"]
        accumulation_steps = config["training"]["accumulation_steps"]
        ema_start_epoch = config["training"]["ema_start_epoch"]
        save_model_every_i_epoch = config["logging"]["save_model_every_i_epoch"]
        path = config["logging"]["path"]
        model_save_name = config["logging"]["model_save_name"]
        log = config["logging"]["wandb_log"]

        self.ema_model.eval()
        self.ema.start_step = len(train_loader) * ema_start_epoch // accumulation_steps

        self.optimizer.zero_grad(set_to_none=True)

        ema_loss = None

        itrs_per_epoch = (iterations // total_epochs)
        train_itr = infinite_dataloader(train_loader)
        test_itr = infinite_dataloader(test_loader)

        self.sync(sec=2)
        for e in range(total_epochs):
            if self.multi_gpu:
                train_loader.sampler.set_epoch(e)

            # Training Loop
            training_losses, ema_loss = self.train_loop(train_itr, itrs_per_epoch, accumulation_steps, f"Training - Epoch {e + 1}/{total_epochs}", ema_loss)

            self.sync(sec=2, clear_cache=True)

            # Validation Loop
            validation_losses, metrics = self.validate_loop(test_itr, len(test_loader), f"Validating - Epoch {e + 1}/{total_epochs}", use_ema_model=False)#(e >= ema_start_epoch)

            self.sync(sec=2, clear_cache=True)

            if self.rank == 0:

                # Saving Model
                if path != "" and save_model_every_i_epoch != 0 and (e+1) % save_model_every_i_epoch == 0:
                    self.save_model(path, f"model_save_{model_save_name}", config)

                # Epoch logging
                if log:
                    log = {
                        "Training_Avg_Loss": np.mean(training_losses)/2,
                        "Validation_Avg_Loss": np.mean(validation_losses)/2,
                        'Training_EMA_Loss': ema_loss/2 if ema_loss is not None else np.mean(training_losses)/2
                    }
                    if train_dataset is not None:
                        hr_p, hr = self.sample_model(random_sample=6, dataset=test_dataset, use_ema_model=False) #(e >= ema_start_epoch)
                        pil_images = tensor_to_pil(denormalize(hr_p))
                        grid_image = create_image_grid(pil_images, grid_size=(3, 2))
                        image = wandb.Image(grid_image, caption="Upscaled Images Grid")
                        log["Image"] = image
                        log["SSIM"] = np.mean(metrics['ssim'])
                        log["PSNR_"] = np.mean(metrics['psnr'])

                    wandb.log(log)

            if e == 2:
                ema_loss = np.mean(training_losses)

            self.sync(clear_cache=True, sec=2)
            if self.rank == 0:
                print("###################################################################")

