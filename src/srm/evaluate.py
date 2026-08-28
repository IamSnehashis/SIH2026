import torch
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from src.srm.model import SRModel
from src.srm.dataset import SatelliteDataset


# --------------------------------------------------
# Paths
# --------------------------------------------------

LR_DIR = "data/raw/OLI2MSI/test_lr"
HR_DIR = "data/raw/OLI2MSI/test_hr"
MODEL_PATH = "models/srm_model.pth"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DEVICE:", device)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset = SatelliteDataset(
    lr_dir=LR_DIR,
    hr_dir=HR_DIR
)

print("TEST DATASET SIZE:", len(dataset))


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = SRModel(scale_factor=3)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("MODEL LOADED SUCCESSFULLY")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

total_mse = 0.0
total_psnr = 0.0
total_ssim = 0.0


with torch.no_grad():

    for i in range(len(dataset)):

        lr, hr = dataset[i]

        # Add batch dimension
        lr = lr.unsqueeze(0).to(device)
        hr = hr.unsqueeze(0).to(device)

        # Model prediction
        sr = model(lr)

        # Remove batch dimension
        sr = sr.squeeze(0).cpu().numpy()
        hr = hr.squeeze(0).cpu().numpy()

        # ------------------------------------------
        # MSE
        # ------------------------------------------

        mse = np.mean((sr - hr) ** 2)

        total_mse += mse

        # ------------------------------------------
        # PSNR
        # ------------------------------------------

        data_range = hr.max() - hr.min()

        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 10 * np.log10(
                (data_range ** 2) / mse
            )

        total_psnr += psnr

        # ------------------------------------------
        # SSIM
        # ------------------------------------------

        image_ssim = ssim(
            hr,
            sr,
            channel_axis=0,
            data_range=data_range
        )

        total_ssim += image_ssim

        # ------------------------------------------
        # Progress
        # ------------------------------------------

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"Evaluated {i + 1}/{len(dataset)} | "
                f"MSE: {mse:.6f} | "
                f"PSNR: {psnr:.4f} dB | "
                f"SSIM: {image_ssim:.4f}"
            )


# --------------------------------------------------
# Final results
# --------------------------------------------------

average_mse = total_mse / len(dataset)
average_psnr = total_psnr / len(dataset)
average_ssim = total_ssim / len(dataset)


print()
print("=" * 50)
print("EVALUATION COMPLETE")
print("=" * 50)

print(f"TEST IMAGES : {len(dataset)}")
print(f"AVERAGE MSE : {average_mse:.6f}")
print(f"AVERAGE PSNR: {average_psnr:.4f} dB")
print(f"AVERAGE SSIM: {average_ssim:.4f}")

print("=" * 50)