import os
import random

import numpy as np
import torch
from torchvision import transforms

from toolkit.model_helper2 import ModelHelper
from .datasets import SuperResolutionDataset
import warnings
from toolkit.loss_functions import Criterion
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from msft_arch6 import MSFTransformer
from .prefetcher import PrefetchDataLoader
from .transforms import PadImg


def init_data(config, rank=0):

    # Subsets
    if config["dataset"]["training"]["data_subset_size"] != 0:
        sub = config["data"]["data_subset_size"]
        if rank == 0:
            print(f"Using a training subset of {sub} images.")

    if config["dataset"]["validation"]["data_subset_size"] != 0:
        sub = config["data"]["data_subset_size"]
        if rank == 0:
            print(f"Using a validation subset of {sub} images.")

    path = os.path.join(config["dataset"]["data_root"], config["dataset"]["training"]["file_name"])

    # Validating existence of training sub-images
    if config["dataset"]["training"]["use_sub_images"]:
        if rank == 0:
            if os.path.exists(path + "_sub"):
                print("Using Sub-Images.")
            else:
                raise Exception("Sub-images directory not found. Run data_prep.py")
        path += "_sub"

    # Validating existence of LMDB for training
    if config["tools"]["file_backend_type"] == "LMDB":
        if rank == 0:
            if os.path.exists(path + "_lmdb"):
                print(f"LMDB Enabled.")
            else:
                raise Exception("LMDB directory not found. Run data_prep.py")
        path += "_lmdb"

    torch.cuda.synchronize()

    train_dataset = SuperResolutionDataset(
        root=path,
        file_backend_type=config["tools"]["file_backend_type"],
        scale_factor=config["model"]["scale_factor"],
        paired_imgs=config["dataset"]["training"]["paired_images"],
        crop_size=config["dataset"]["training"]["crop_size"],
        random_rot=config["dataset"]["training"]["random_rot"],
        random_flip=config["dataset"]["training"]["random_flip"],
        enlarge_factor=config["dataset"]["training"]["dataset_enlarge_scale"],
        subset=config["dataset"]["training"]["data_subset_size"]
    )

    test_dataset = SuperResolutionDataset(
        root=os.path.join(config["dataset"]["data_root"], config["dataset"]["validation"]["file_name"]),
        scale_factor=config["model"]["scale_factor"],
        crop_size=config["dataset"]["validation"]["crop_size"],
        subset=config["dataset"]["validation"]["data_subset_size"]
    )

    sampler = None
    if config["tools"]["multi_gpu_enable"]:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size_per_gpu"],
        shuffle=(sampler is None),
        num_workers=config["dataset"]["num_dataloader_workers"],
        pin_memory=True,
        sampler=sampler
    )

    # TODO only on rank=0?
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["dataset"]["num_dataloader_workers"],
        pin_memory=True
    )

    return train_dataset, test_dataset, train_loader, test_loader, sampler


def init(config, rank=0, world_size=0):
    warnings.filterwarnings("ignore", message=".*compiled with flash attention.*")
    torch.backends.cudnn.benchmark = config["tools"]["benchmark"]
    torch.backends.cudnn.deterministic = config["tools"]["deterministic"]
    #torch.autograd.set_detect_anomaly(True)

    if config["tools"]["multi_gpu_enable"]:
        if rank == 0:
            print("Multi-GPU Support Enabled.")
            print("Total GPUs in usage: ", torch.cuda.device_count())
        #os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        #os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend='gloo', init_method="tcp://127.0.0.1:29500?use_libuv=0", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    model = MSFTransformer(
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

    # Multi-GPU model
    model = model.to(f"cuda:{rank}")
    if config["tools"]["multi_gpu_enable"]:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.99))
    criterion = Criterion(config["training"]["criterion"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["iterations"], eta_min=5e-6)

    helper = ModelHelper(model, optimizer, criterion, scheduler, ema_beta=config["training"]["ema_start_epoch"], multi_gpu=config["tools"]["multi_gpu_enable"], rank=rank)

    size = helper.get_parameter_count()
    config["model_size"] = size
    if rank == 0:
        print(f"Model Size: {size}")

    # Loading Model Save
    if config["tools"]["pretrained_weights_path"] != "":
        helper.load_model(
            config["tools"]["pretrained_weights_path"],
            load_optimizer=config["tools"]["load_optimizer"]
        )

    # Loading Dataset
    train_dataset, test_dataset, train_loader, test_loader, sampler = init_data(config, rank)

    return model, helper, (train_dataset, test_dataset, sampler, train_loader, test_loader)
