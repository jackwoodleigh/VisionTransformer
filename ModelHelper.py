import torch


class ModelHelper:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimzer = optimizer

    def train_model(self, training_loader, validation_loader, epochs, log=False, save_path="save.pt"):

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(training_loader))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=len(training_loader)*5, eta_min=self.min_learning_rate)
        self.warm_up = len(training_loader) * 4
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup_lr)

        ema_start_step = len(training_loader) * 4
        self.model.train()
        self.EMA_model.eval()

        for e in range(epochs):
            epoch_training_loss = 0
            epoch_validation_loss = 0
            print(f"Epoch {e}...")

            # Training
            for images, labels in tqdm(training_loader):
                self.optimizer.zero_grad()
                loss = self.predict(images, labels + 1, self.model, learning=True)
                epoch_training_loss += loss
                self.EMA.step_ema(ema_model=self.EMA_model, model=self.model, start_step=ema_start_step)
                scheduler.step()

            # Validation
            with torch.no_grad():
                for images, labels in tqdm(validation_loader):
                    loss = self.predict(images, labels, self.EMA_model, learning=False)
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