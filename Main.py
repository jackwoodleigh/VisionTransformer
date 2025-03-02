import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from LMLTransformer import LMLTransformer
from ModelHelper import ModelHelper
from utils import SuperResolutionDataset
import yaml
import warnings
from utils import calculate_psnr, calculate_ssim
import torchvision.transforms.functional as F
from LossFunctions import Criterion
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


class PadImg:
    def __init__(self, size):
        self.final_width = size
        self.final_height = size
    def __call__(self, img):
        w, h = img.size
        pad_w = max(0, self.final_width - w)
        pad_h = max(0, self.final_height - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        img = F.pad(img, padding, padding_mode='reflect')
        return img

def rotate_if_wide(img):
    if img.height > img.width:
        return img.rotate(-90, expand=True)
    return img

def load_dataset(config, rank=0):
    training_pad_transform = PadImg(config["data"]["training_image_size"])
    validation_pad_transform = PadImg(config["data"]["validation_image_size"])

    if config["data"]["transform_data"]:
        transform = transforms.Compose([
            training_pad_transform,
            transforms.RandomCrop(config["data"]["training_image_size"]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=[-90, 90]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])
    else:
        transform = transforms.Compose([
            training_pad_transform,
            transforms.RandomCrop(config["data"]["training_image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])

    if config["data"]["data_subset"] != 0:
        sub = config["data"]["data_subset"]
        if rank == 0:
            print(f"Using a subset of {sub} images.")
        train_dataset = SuperResolutionDataset(root='./data/train', scale_values=config["model"]["scale_factor"], transform=transform, enlarge_factor=config["data"]["training_dataset_enlarge_scale"], subset=config["data"]["data_subset"])

    else:
        train_dataset = SuperResolutionDataset(root='./data/train', scale_values=config["model"]["scale_factor"], transform=transform, enlarge_factor=config["data"]["training_dataset_enlarge_scale"])

    test_dataset = SuperResolutionDataset(root='./data/test', scale_values=config["model"]["scale_factor"], transform=transforms.Compose([validation_pad_transform, transforms.RandomCrop(config["data"]["validation_image_size"]), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])]))

    return train_dataset, test_dataset


def initialize(config, rank=0, world_size=0):
    warnings.filterwarnings("ignore", message=".*compiled with flash attention.*")
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = False

    multi_gpu = False
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if rank == 0:
            print("Total GPUs in usage: ", torch.cuda.device_count())

        multi_gpu = True
        #os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        #os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:29500", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    model = LMLTransformer(
        n_blocks=config["model"]["n_blocks"],
        levels=config["model"]["levels"], window_size=config["model"]["n_blocks"],
        dim=config["model"]["dim"],
        features=config["model"]["features"],
        scale_factor=config["model"]["scale_factor"]
    )

    model = model.to(f"cuda:{rank}")
    if multi_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.99))
    criterion = Criterion(config["training"]["criterion"])

    helper = ModelHelper(model, optimizer, criterion, ema_beta=config["training"]["model_ema"], multi_gpu=multi_gpu)

    # Loading Save
    if config["tools"]["load_model_save_name"] != "":
        helper.load_model(config["tools"]["model_save_directory"], config["tools"]["load_model_save_name"], load_optimizer=config["tools"]["load_optimizer"])

    size = helper.get_parameter_count()
    if rank == 0:
        print(f"Model Size: {size}")

    # Loading Data
    train_dataset, test_dataset = load_dataset(config)

    sampler = None
    if multi_gpu:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=(sampler is None), num_workers=12, pin_memory=True, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=12, pin_memory=True)

    return model, helper, (train_dataset, test_dataset, sampler, train_loader, test_loader)

def training(rank, config, world_size=0):
    model, helper, (train_dataset, test_dataset, sampler, train_loader, test_loader) = initialize(config, rank, world_size)
    if rank == 0:
        print("Running Training...")
    helper.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    if world_size != 0:
        dist.destroy_process_group()

def sample_images(config, count):
    model, helper, (train_dataset, test_dataset, train_loader, test_loader) = initialize(config)
    print("Running Image Sampling...")
    helper.sample_model(random_sample=count, dataset=train_dataset, save_img=True)

def test(config):
    model, helper, (dataset, train_loader, test_loader) = initialize(config)
    #helper.sample_model(random_sample=3, dataset=dataset, save_img=True, save_compare=True, use_ema_model=False)
    ssim = []
    psnr = []
    # Validation
    with torch.no_grad():
        #pbar = tqdm(test_loader, desc=f"Validating - Epoch {e + 1}/{epochs}", leave=True, dynamic_ncols=True)
        for i, (hr, lr) in tqdm(enumerate(train_loader)):
            loss, hr_p = helper.predict(hr, lr, use_ema_model=True)

            hr = hr.to("cuda")
            ssim.append(calculate_ssim(hr_p.float(), hr.float()).item())
            psnr.append(calculate_psnr(hr_p.float(), hr.float()).mean().item())
    print(np.mean(ssim))
    print(np.mean(psnr))


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Start W&B run
    if config["tools"]["log"]:
        import wandb
        wandb.login()
        wandb.init(
            project="SuperResolution",
            config=config
        )
        run_name = wandb.run.name
        config["tools"]["run_name"] = run_name

    # Training
    if config["tools"]["multi_gpu_enable"] and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        mp.spawn(training, args=(config, world_size,), nprocs=world_size, join=True)
    else:
        training(config=config, rank=0, world_size=0)

    #training(config)

    #sample_images(config, 1)
