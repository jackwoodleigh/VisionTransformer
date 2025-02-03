import torch
import wandb
from tqdm import tqdm
import torchvision.models as models
import torch
from torch import nn

class PerceptualLoss(nn.Module):
    def __init__(self, layer_index=16):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:layer_index])
    def forward(self, predicted, target):
        predicted_features = self.features(predicted)
        target_features = self.features(target)
        return nn.MSELoss(predicted_features, target_features)


class ModelHelper:
    def __init__(self, model, optimizer, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.perceptual_loss = PerceptualLoss().to(device)
        self.device = device

        # TODO add an EMA

    def train_model(self, train_loader, test_loader, epochs, batches_per_epoch, log=False, save_path="save.pt"):
        self.model.train()

        for e in range(epochs):
            epoch_training_loss = 0
            epoch_validation_loss = 0
            print(f"Epoch {e}...")

            # Training
            for images in tqdm(train_loader):
                self.optimizer.zero_grad()
                images = images.to(self.device)

                # TODO down scale image

                predicted = self.model(images)


            # Validation
            with torch.no_grad():
                for images in tqdm(test_loader):
                    loss = self.predict(images, self.EMA_model, learning=False)
                    epoch_validation_loss += loss

            epoch_validation_loss /= len(validation_loader)
            epoch_training_loss /= len(training_loader)

            print(f"Training Loss: {epoch_training_loss}, Validation Loss: {epoch_validation_loss}")
            # Epoch logging
            if log:
                pil_image = self.sample(8, torch.tensor([1], device=self.device))
                image = wandb.Image(pil_image, caption=f"class 2")
                wandb.log(
                    {"Training_Loss": epoch_training_loss, "Validation_Loss": epoch_validation_loss,
                     "Sample": image})
                torch.save(self.EMA_model.state_dict(), save_path)
                print("Model Saved.")

