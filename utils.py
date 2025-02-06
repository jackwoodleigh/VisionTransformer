import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms

class SuperResolutionDataset(Dataset):
    def __init__(self, root, scale_values, base_transforms, subset=None):
        self.root = root
        self.scale_values = scale_values

        self.blur_transforms = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))

        self.dataset = datasets.ImageFolder(root=root, transform=base_transforms)

        if subset is not None:
            subset_indices = list(range(subset))
            self.dataset = Subset(self.dataset, subset_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr_img, _ = self.dataset[idx]
        lr_img = self.blur_transforms(hr_img)
        lr_img = F.interpolate(lr_img.unsqueeze(0), scale_factor=(1 / self.scale_values), mode='area')

        return hr_img, lr_img.squeeze(0)


def save_images_comparison(input_tensor, output_tensor, filename="comparison.png"):
    comparison = torch.cat((input_tensor, output_tensor), dim=-1)
    vutils.save_image(comparison, filename, nrow=input_tensor.shape[0])
    print("Saved Images.")

def save_images(tensor, filename="images.png"):
    vutils.save_image(tensor, filename, nrow=tensor.shape[0])
    print("Saved Images.")

def tensor_to_pil(tensor):
    to_pil = transforms.ToPILImage()
    return [to_pil(t) for t in tensor]
