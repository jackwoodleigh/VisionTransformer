import random

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from LMLTransformer import LMLTransformer
from ModelHelper import ModelHelper
from utils import SuperResolutionDataset
import yaml
import warnings

def rotate_if_wide(img):
    if img.height < img.width:
        return img.rotate(-90, expand=True)
    return img

def initialize(config):
    warnings.filterwarnings("ignore", message=".*compiled with flash attention.*")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    base_transforms = transforms.Compose([
        transforms.Lambda(rotate_if_wide),
        transforms.CenterCrop((config["training"]["image_height"], config["training"]["image_width"])),
        transforms.ToTensor()
    ])

    dataset = SuperResolutionDataset(root='./data/train', scale_values=config["model"]["scale_factor"], base_transforms=base_transforms, subset=100)

    test_size = int(len(dataset) * config["training"]["testing_data_split"])
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    model = LMLTransformer(
        n_blocks=config["model"]["n_blocks"],
        levels=config["model"]["levels"], window_size=config["model"]["n_blocks"],
        dim=config["model"]["dim"],
        scale_factor=config["model"]["scale_factor"]
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    trainer = ModelHelper(model, optimizer)

    if config["tools"]["load_model_directory"] != "":
        trainer.load_model(config["tools"]["load_model_directory"])

    size = trainer.get_parameter_count()
    print(f"Model Size: {size}")

    return model, trainer, (dataset, train_loader, test_loader)

def training(config):
    model, trainer, (dataset, train_loader, test_loader) = initialize(config)
    print("Running Training...")
    if config["training"]["is_training"]:
        trainer.train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config["training"]["epochs"],
            accumulation_steps=config["training"]["accumulation_steps"],
            pl_scale=config["training"]["perceptual_loss_scale"],
            fft_loss_scale=config["training"]["fft_loss_scale"],
            log=config["tools"]["log"],
            save_model_every_i_epoch=config["tools"]["save_model_every_i_epoch"],
            save_path=config["tools"]["model_save_directory"]
        )


def sample_images(config, count):
    model, trainer, (dataset, train_loader, test_loader) = initialize(config)
    print("Running Image Sampling...")
    r = random.randint(0, len(dataset))
    trainer.sample_model(torch.stack([dataset[i][1] for i in range(r, r+count)]).to("cuda"), save_img=True)

if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    #training(config)

    sample_images(config, 1)
