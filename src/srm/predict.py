import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

from src.srm.model import SRModel


# ============================================================
# PATHS
# ============================================================

LR_DIR = Path("data/raw/OLI2MSI/test_lr")
HR_DIR = Path("data/raw/OLI2MSI/test_hr")

MODEL_PATH = Path("models/srm_model.pth")

OUTPUT_DIR = Path("outputs/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DEVICE:", device)


# ============================================================
# LOAD MODEL
# ============================================================

model = SRModel(scale_factor=3)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

print("MODEL LOADED")


# ============================================================
# FIND TEST IMAGES
# ============================================================

lr_files = sorted(LR_DIR.glob("*.TIF"))

if len(lr_files) == 0:
    lr_files = sorted(LR_DIR.glob("*.tif"))

if len(lr_files) == 0:
    raise FileNotFoundError(
        f"No TIFF images found in {LR_DIR}"
    )


# ============================================================
# SELECT ONE TEST IMAGE
# ============================================================

lr_path = lr_files[0]

hr_path = HR_DIR / lr_path.name

print()
print("LR IMAGE :", lr_path)
print("HR IMAGE :", hr_path)


# ============================================================
# READ LR IMAGE
# ============================================================

with rasterio.open(lr_path) as src:
    lr_image = src.read().astype(np.float32)
    lr_profile = src.profile


print("LR SHAPE:", lr_image.shape)


# ============================================================
# MODEL INFERENCE
# ============================================================

lr_tensor = torch.from_numpy(lr_image).unsqueeze(0)

lr_tensor = lr_tensor.to(device)

with torch.no_grad():
    sr_tensor = model(lr_tensor)

sr_image = sr_tensor.squeeze(0).cpu().numpy()


print("SR SHAPE:", sr_image.shape)


# ============================================================
# SAVE SUPER-RESOLVED IMAGE
# ============================================================

output_path = OUTPUT_DIR / f"SR_{lr_path.name}"

profile = lr_profile.copy()

profile.update(
    height=sr_image.shape[1],
    width=sr_image.shape[2],
    count=sr_image.shape[0],
    dtype="float32"
)

with rasterio.open(
    output_path,
    "w",
    **profile
) as dst:

    dst.write(sr_image.astype(np.float32))


print("SR IMAGE SAVED:", output_path)


# ============================================================
# VISUALIZATION FUNCTION
# ============================================================

def make_rgb(image):

    # Convert CHW -> HWC
    rgb = np.transpose(image, (1, 2, 0))

    # Normalize for visualization
    min_value = rgb.min()
    max_value = rgb.max()

    if max_value > min_value:
        rgb = (rgb - min_value) / (max_value - min_value)

    return np.clip(rgb, 0, 1)


# ============================================================
# PREPARE LR / SR / HR FOR DISPLAY
# ============================================================

lr_display = make_rgb(lr_image)

sr_display = make_rgb(sr_image)


hr_display = None

if hr_path.exists():

    with rasterio.open(hr_path) as src:
        hr_image = src.read().astype(np.float32)

    hr_display = make_rgb(hr_image)

    print("HR SHAPE:", hr_image.shape)


# ============================================================
# CREATE COMPARISON FIGURE
# ============================================================

if hr_display is not None:

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(lr_display)
    plt.title("Low Resolution Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sr_display)
    plt.title("Super-Resolved Output")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(hr_display)
    plt.title("Ground Truth HR")
    plt.axis("off")

else:

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(lr_display)
    plt.title("Low Resolution Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sr_display)
    plt.title("Super-Resolved Output")
    plt.axis("off")


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_path = OUTPUT_DIR / "comparison.png"

plt.tight_layout()

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print()
print("COMPARISON SAVED:", comparison_path)
print()
print("PREDICTION COMPLETE")