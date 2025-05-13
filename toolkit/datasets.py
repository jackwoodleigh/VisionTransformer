from .file_backend import FileBackend
from .utils import scandir
from .transforms import paired_random_crop, paired_random_flip_horizontal, paired_random_flip_vertical, paired_random_rotate, normalize, bicubic_downscale_HWC
import os
import torch
from torch.utils.data import Dataset, Subset
import numpy as np

class SuperResolutionDataset(Dataset):
    def __init__(self, root, file_backend_type="Disk", scale_factor=4, paired_imgs=False, crop_size=None, random_rot=False, random_flip=False, enlarge_factor=1, subset=0):
        self.root = root
        self.file_backend_type = file_backend_type
        self.scale_factor = scale_factor
        self.paired_imgs = paired_imgs
        self.crop_size = (crop_size, crop_size) if crop_size is not None else None
        self.random_rot = random_rot
        self.random_flip = random_flip
        self.enlarge_factor = enlarge_factor

        self.file_backend = FileBackend(file_backend_type, root)
        self.paths = self.get_paths()

        if subset == 0:
            self.n = len(self.paths)
        else:
            self.n = subset

    def get_paths(self):
        root = self.root
        if self.file_backend_type == 'LMDB':
            root = root.replace('_lmdb', '')
        if self.paired_imgs:
            return sorted(list(scandir(os.path.join(root, "hr"), suffix='png', recursive=True, full_path=False)))
        else:
            return sorted(list(scandir(root, suffix='png', recursive=True, full_path=False)))

    def __len__(self):
        return self.n * self.enlarge_factor

    def __getitem__(self, idx):
        idx = idx % self.n
        if self.paired_imgs:
            hr = self.file_backend.load(os.path.join("hr", self.paths[idx]))
            lr = self.file_backend.load(os.path.join("lr", self.paths[idx]))
        else:
            hr = self.file_backend.load(self.paths[idx])
            lr = bicubic_downscale_HWC(hr.transpose(1, 2, 0), self.scale_factor).transpose(2, 0, 1)

        if self.crop_size is not None:
            hr, lr = paired_random_crop(hr, lr, self.crop_size, scale=self.scale_factor)

        if self.random_flip:
            hr, lr = paired_random_flip_horizontal(hr, lr)
            hr, lr = paired_random_flip_vertical(hr, lr)

        if self.random_rot:
            hr, lr = paired_random_rotate(hr, lr)

        hr = torch.from_numpy(np.ascontiguousarray(hr))
        lr = torch.from_numpy(np.ascontiguousarray(lr))

        hr = normalize(hr)
        lr = normalize(lr)
        return hr, lr


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


if __name__ == '__main__':

    root = "C:/Users/Jack/Documents/GitHub/VisionTransformer/data/MANGA109"
    d = SuperResolutionDataset(root, 4)
    p = d.get_paths()
    print([os.path.splitext(img_file)[0] for img_file in p])



