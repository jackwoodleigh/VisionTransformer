import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class SuperResolutionDataset(Dataset):
    def __init__(self, root, hr_transforms, lr_transforms):
        self.root = root
        self.hr_transform = hr_transforms
        self.lr_transform = lr_transforms

        self.image_files = [
            os.path.join(root, file)
            for file in os.listdir(root)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]