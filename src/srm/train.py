import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.srm.dataset import SatelliteDataset
from src.srm.model import SRModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

LR_DIR = "data/raw/OLI2MSI/train_lr"
HR_DIR = "data/raw/OLI2MSI/train_hr"

BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4

MODEL_DIR = "models"
MODEL_PATH = "models/srm_model.pth"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = SatelliteDataset(
    lr_dir=LR_DIR,
    hr_dir=HR_DIR
)

print("Dataset size:", len(dataset))


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = SRModel(scale_factor=3)
model = model.to(device)


# --------------------------------------------------
# Loss function
# --------------------------------------------------

criterion = nn.L1Loss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_idx, (lr, hr) in enumerate(dataloader):

        lr = lr.to(device)
        hr = hr.to(device)

        # Forward pass
        sr = model(lr)

        # Calculate loss
        loss = criterion(sr, hr)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        running_loss += loss.item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f}"
            )

    epoch_loss = running_loss / len(dataloader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Average Loss: {epoch_loss:.6f}"
    )


# --------------------------------------------------
# Save trained model
# --------------------------------------------------

torch.save(model.state_dict(), MODEL_PATH)

print()
print("Training complete.")

print(f"Model saved to: {MODEL_PATH}")