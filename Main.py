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



# TODO add patch loss
# TODO add an EMA
# TODO add diffusion refinement


def rotate_if_wide(img):
    if img.height > img.width:
        return img.rotate(-90, expand=True)
    return img

def initialize(config):
    warnings.filterwarnings("ignore", message=".*compiled with flash attention.*")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # TODO need to fix this so it doesnt add black pixels

    base_transforms = transforms.Compose([
        transforms.Lambda(rotate_if_wide),
        transforms.CenterCrop((config["training"]["image_height"], config["training"]["image_width"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])

    dataset = SuperResolutionDataset(root='./data/train', scale_values=config["model"]["scale_factor"], base_transforms=base_transforms)

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

    helper = ModelHelper(model, optimizer)

    if config["tools"]["load_model_save_name"] != "":
        helper.load_model(config["tools"]["model_save_directory"], config["tools"]["load_model_save_name"])

    size = helper.get_parameter_count()
    print(f"Model Size: {size}")

    return model, helper, (dataset, train_loader, test_loader)

def training(config):
    model, helper, (dataset, train_loader, test_loader) = initialize(config)
    print("Running Training...")
    if config["training"]["is_training"]:
        helper.train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config["training"]["epochs"],
            accumulation_steps=config["training"]["accumulation_steps"],
            pl_scale=config["training"]["perceptual_loss_scale"],
            fft_loss_scale=config["training"]["fft_loss_scale"],
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
    t1 = calculate_psnr(hr_p, hr).mean().item()
    t2 = calculate_ssim(hr_p, hr).item()
    print()


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
