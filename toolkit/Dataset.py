

import io
import os
from os import path as osp

import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lmdb

import kornia.color
import cv2
from LMDB import get_keys


class SuperResolutionDataset(Dataset):
    def __init__(self, root, scale_values, transform=None, use_lmdb=False, enlarge_factor=1, subset=0):
        self.root = root
        self.scale_values = scale_values
        self.transform = transform
        self.enlarge_factor = enlarge_factor
        self.use_lmdb = use_lmdb
        self.env = None

        if use_lmdb:
            self.n = len(get_keys(self.root))
            if subset != 0:
                assert self.n > subset
                self.n = subset

        else:
            self.dataset = datasets.ImageFolder(root=root)

            if subset != 0:
                subset_indices = list(range(subset))
                self.dataset = Subset(self.dataset, subset_indices)

            self.n = len(self.dataset)

    def open_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.root,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
            self.env = None

    def __len__(self):
        return self.n * self.enlarge_factor

    def __getitem__(self, idx):
        idx = idx % self.n

        if self.use_lmdb:
            self.open_env()
            key = f"{idx:08d}".encode('ascii')

            with self.env.begin() as txn:
                img_bytes = txn.get(key)
                if img_bytes is None:
                    raise KeyError(f"Key {key} not found in LMDB!")

            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            hr_img = Image.fromarray(hr_img)

            if self.transform is not None:
                hr_img = self.transform(hr_img)

            lr_img = F.interpolate(hr_img.unsqueeze(0), scale_factor=(1 / self.scale_values), mode='bicubic')

        else:
            hr_img, _ = self.dataset[idx]
            if self.transform is not None:
                hr_img = self.transform(hr_img)
            lr_img = F.interpolate(hr_img.unsqueeze(0), scale_factor=(1 / self.scale_values), mode='bicubic')

        return hr_img, lr_img.squeeze(0)


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch
