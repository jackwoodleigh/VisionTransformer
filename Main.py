import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

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

# TODO compare conv upscaling and interpolation
# TODO clean up the LMDB and data prep - possible multi threat extract better
# TODO clean up main - look at transforms and maybe do with tensors
# TODO figure issue with performance on multiple gpu
# TODO Add data prep to config


class PadImg:
    def __init__(self, height, width):
        self.final_width = width
        self.final_height = height
    def __call__(self, img):
        w, h = img.size
        pad_w = max(0, self.final_width - w)
        pad_h = max(0, self.final_height - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        img = F.pad(img, padding, padding_mode='reflect')
        return img
def random_rotate(image):
    angle = random.choice([0, 90, 180, 270])
    return F.rotate(image, angle)

def rotate_if_wide(img):
    if img.height > img.width:
        return img.rotate(-90, expand=True)
    return img


class CropDivisibleBy:
    def __init__(self, divisor=4):
        self.divisor = divisor
    def __call__(self, img):
        w, h = img.size
        new_h = (h // self.divisor) * self.divisor
        new_w = (w // self.divisor) * self.divisor
        if new_h == 0:
            new_h = self.divisor
        if new_w == 0:
            new_w = self.divisor
        cropped_img = F.center_crop(img, (new_h, new_w))
        return cropped_img

    def __repr__(self):
        return self.__class__.__name__ + f'(divisor={self.divisor})'

def load_data(config, rank=0, multi_gpu=False):

    if config["data"]["data_subset"] != 0:
        sub = config["data"]["data_subset"]
        if rank == 0:
            print(f"Using a subset of {sub} images.")

    path = os.path.join(config["data"]["data_path"], config["data"]["training_data_name"])

    # Validating sub-image set
    if config["tools"]["use_sub_images"]:
        if rank == 0:
            if os.path.exists(path + "_sub"):
                print("Using Sub-Images.")
            else:
                raise Exception("Sub-images directory not found. Run DatasetPreparation.py")
        path += "_sub"

    # Validating LMDB
    if config["tools"]["use_lmdb"]:
        if rank == 0:
            if os.path.exists(path + "_lmdb"):
                print(f"LMDB Enabled.")
            else:
                raise Exception("LMDB directory not found. Run DatasetPreparation.py")
        path += "_lmdb"

    torch.cuda.synchronize()

    transform = transforms.Compose([
        PadImg(config["data"]["training_image_size"],config["data"]["training_image_size"]),
        transforms.RandomCrop(config["data"]["training_image_size"]),
        transforms.RandomHorizontalFlip(p=0.5),
        random_rotate,
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    train_dataset = SuperResolutionDataset(
        root=path,
        scale_values=config["model"]["scale_factor"],
        transform=transform,
        use_lmdb=config["tools"]["use_lmdb"],
        enlarge_factor=config["data"]["training_dataset_enlarge_scale"],
        subset=config["data"]["data_subset"]
    )

    test_dataset = SuperResolutionDataset(
        root=os.path.join(config["data"]["data_path"], config["data"]["testing_data_name"]),
        scale_values=config["model"]["scale_factor"],
        transform=transforms.Compose([
            rotate_if_wide,
            CropDivisibleBy(4),
            #PadImg(512, 1024),
            #transforms.RandomCrop((512, 1024)),
            #transforms.RandomCrop(config["data"]["validation_image_size"]),
            transforms.ToTensor()])
        )

    sampler = None
    if multi_gpu:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size_per_gpu"],
        shuffle=(sampler is None),
        num_workers=config["data"]["num_dataloader_workers"],
        pin_memory=True,
        sampler=sampler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data"]["num_dataloader_workers"],
        pin_memory=True
    )

    return train_dataset, test_dataset, train_loader, test_loader, sampler


def initialize(config, rank=0, world_size=0):
    warnings.filterwarnings("ignore", message=".*compiled with flash attention.*")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    multi_gpu = False
    if config["tools"]["multi_gpu_enable"]:
        multi_gpu = True
        if rank == 0:
            print("Multi-GPU Support Enabled.")
            print("Total GPUs in usage: ", torch.cuda.device_count())
        #os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        #os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend='gloo', init_method="tcp://127.0.0.1:29500?use_libuv=0", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    from LMLTransformer_mod import LMLTransformer
    model = LMLTransformer(
        block_type=config["model"]["block_type"],
        n_blocks=config["model"]["n_blocks"],
        n_sub_blocks=config["model"]["n_sub_blocks"],
        levels=config["model"]["levels"],
        window_size=config["model"]["window_size"],
        dim=config["model"]["dim"],
        level_dim=config["model"]["level_dim"],
        n_heads=config["model"]["n_heads"],
        n_heads_fuse=config["model"]["n_heads_fuse"],
        feature_dim=config["model"]["feature_dim"],
        scale_factor=config["model"]["scale_factor"]
    )

    '''from network_swinir import SwinIR
    model = SwinIR(
        img_size=64,
        embed_dim=180,
        window_size=8,
        upsampler='nearest+conv',
        upscale=4,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        resi_connection="1conv",
        img_range=4.0
    )'''

    # Multi-GPU model
    model = model.to(f"cuda:{rank}")
    if multi_gpu:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.99))
    criterion = Criterion(config["training"]["criterion"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["iterations"], eta_min=1e-5)
    #milestones = [300000, 500000, 650000, 700000, 750000]
    '''milestones = [40000, 190000, 240000, 290000]
    # [250000, 400000, 450000, 475000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)'''

    helper = ModelHelper(model, optimizer, scheduler, criterion, ema_beta=config["training"]["model_ema"], multi_gpu=multi_gpu, rank=rank)

    size = helper.get_parameter_count()
    config["model_size"] = size
    if rank == 0:
        print(f"Model Size: {size}")

    # Loading Model Save
    if config["tools"]["load_model_save_name"] != "":
        helper.load_model(
            config["tools"]["logging_path"],
            config["tools"]["load_model_save_name"],
            load_optimizer=config["tools"]["load_optimizer"]
        )

    # Loading Datasetz
    train_dataset, test_dataset, train_loader, test_loader, sampler = load_data(config, rank, multi_gpu)

    return model, helper, (train_dataset, test_dataset, sampler, train_loader, test_loader)

def training(rank, config, world_size=0):
    model, helper, (train_dataset, test_dataset, sampler, train_loader, test_loader) = initialize(config, rank, world_size)
    if rank == 0:
        # Start W&B run
        if config["tools"]["wandb_log"]:
            import wandb
            wandb.login()
            wandb.init(
                project="SuperResolution",
                config=config
            )
            run_name = wandb.run.name
            config["tools"]["model_save_name"] = run_name
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

def test(rank, config, world_size=0):
    model, helper, (train_dataset, test_dataset, sampler, train_loader, test_loader) = initialize(config, rank, world_size)
    #from LMDB import read_from_lmdb, get_keys

    '''epoch_validation_losses, ssim, psnr = helper.validation_loop(test_loader, 10, 10, 20)
    print(np.mean(ssim), np.mean(psnr))'''

    test_dataset = SuperResolutionDataset(
        root=os.path.join(config["data"]["data_path"], config["data"]["testing_data_name"]),
        scale_values=config["model"]["scale_factor"],
        transform=transforms.Compose([
            rotate_if_wide,
            CropDivisibleBy(4),
            PadImg(512, 1024),
            transforms.RandomCrop((512, 1024)),
            transforms.ToTensor()])
    )
    helper.sample_model(random_sample=3, dataset=test_dataset, save_img=True, save_compare=True, use_ema_model=True)



if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    test(config=config, rank=0, world_size=0)

    '''# Training
    if config["tools"]["multi_gpu_enable"]:
        world_size = torch.cuda.device_count()
        mp.spawn(training, args=(config, world_size,), nprocs=world_size, join=True)
    else:
        training(config=config, rank=0, world_size=0)'''



    #training(config)

    #sample_images(config, 1)
