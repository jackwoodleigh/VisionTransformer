import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from LMLTransformer import LMLTransformer
from ModelHelper import ModelHelper
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.rotate(-90, expand=True) if img.height < img.width else img),
    transforms.CenterCrop((config["training"]["image_height"], config["training"]["image_width"])),
    transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)

test_size = int(len(dataset) * config["training"]["testing_data_split"])
train_size = len(dataset) - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config["training"]["true_batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["training"]["true_batch_size"], shuffle=True)

model = LMLTransformer(
    n_blocks=config["model"]["n_blocks"],
    levels=config["model"]["levels"], window_size=config["model"]["n_blocks"],
    dim=config["model"]["dim"],
    scale_factor=config["model"]["scale_factor"]
)

optimzer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

trainer = ModelHelper(model, optimzer)

