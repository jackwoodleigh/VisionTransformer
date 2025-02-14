import random
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from LMLTransformer import LMLTransformer
from ModelHelper import ModelHelper
from utils import SuperResolutionDataset
import yaml
import warnings
from utils import calculate_psnr, calculate_ssim
import torchvision.transforms.functional as F


# TODO add patch loss
# TODO add an EMA
# TODO add diffusion refinement

class PadTo1080:
    def __call__(self, img):
        w, h = img.size
        pad_w = max(0, 1080 - w)
        pad_h = max(0, 1080 - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        img = F.pad(img, padding, padding_mode='reflect')
        return img

def rotate_if_wide(img):
    if img.height > img.width:
        return img.rotate(-90, expand=True)
    return img

def initialize(config):
    warnings.filterwarnings("ignore", message=".*compiled with flash attention.*")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if config["training"]["transform_data"]:
        transform = transforms.Compose([
            PadTo1080(),
            transforms.RandomCrop(config["training"]["image_height"]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=[-90, 90]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(rotate_if_wide),
            transforms.CenterCrop((config["training"]["image_height"], config["training"]["image_width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])

    if config["training"]["data_subset"] != 0:
        print(f"Using a subset of {config["training"]["data_subset"]} images.")
        dataset = SuperResolutionDataset(root='./data/train', scale_values=config["model"]["scale_factor"], transform=transform, subset=config["training"]["data_subset"])
    else:
        dataset = SuperResolutionDataset(root='./data/train', scale_values=config["model"]["scale_factor"], transform=transform)

    test_size = int(len(dataset) * config["training"]["testing_data_split"])
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    model = LMLTransformer(
        n_blocks=config["model"]["n_blocks"],
        levels=config["model"]["levels"], window_size=config["model"]["n_blocks"],
        dim=config["model"]["dim"],
        scale_factor=config["model"]["scale_factor"]
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    helper = ModelHelper(model, optimizer, ema_beta=config["training"]["model_ema"])

    if config["tools"]["load_model_save_name"] != "":
        helper.load_model(config["tools"]["model_save_directory"], config["tools"]["load_model_save_name"], load_optimizer=config["tools"]["load_optimizer"])

    size = helper.get_parameter_count()
    print(f"Model Size: {size}")

    return model, helper, (dataset, train_loader, test_loader)

def training(config):
    model, helper, (dataset, train_loader, test_loader) = initialize(config)
    print("Running Training...")
    helper.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["training"]["epochs"],
        accumulation_steps=config["training"]["accumulation_steps"],
        pl_scale=config["training"]["perceptual_loss_scale"],
        fft_loss_scale=config["training"]["fft_loss_scale"],
        ema_start_epoch=config["training"]["ema_start_epoch"],
        log=config["tools"]["log"],
        save_model_every_i_epoch=config["tools"]["save_model_every_i_epoch"],
        save_path=config["tools"]["model_save_directory"],
        dataset=dataset
    )


def sample_images(config, count):
    model, helper, (dataset, train_loader, test_loader) = initialize(config)
    print("Running Image Sampling...")
    helper.sample_model(random_sample=count, dataset=dataset, save_img=True)

def test(config):
    model, helper, (dataset, train_loader, test_loader) = initialize(config)
    hr_p, hr = helper.sample_model(random_sample=3, dataset=dataset)
    t1 = calculate_psnr(hr_p, hr)
    t2 = calculate_ssim(hr_p, hr)
    from utils import save_images_comparison
    save_images_comparison(hr_p, hr)
    print(t1, t2)


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

    #test(config)
    training(config)

    #training(config)

    #sample_images(config, 1)
