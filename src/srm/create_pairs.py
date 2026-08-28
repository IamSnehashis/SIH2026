from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


def create_lr_image(hr_path, lr_path, scale=3):
    """
    Create a low-resolution image from a high-resolution GeoTIFF.

    Args:
        hr_path: Path to high-resolution GeoTIFF
        lr_path: Output path for low-resolution GeoTIFF
        scale: Downsampling factor
    """

    with rasterio.open(hr_path) as src:

        new_height = src.height // scale
        new_width = src.width // scale

        lr_image = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bicubic
        )

        profile = src.profile.copy()

        profile.update(
            height=new_height,
            width=new_width,
            transform=src.transform * src.transform.scale(
                src.width / new_width,
                src.height / new_height
            )
        )

        with rasterio.open(lr_path, "w", **profile) as dst:
            dst.write(lr_image)


def create_dataset_pairs(hr_dir, lr_dir, scale=4):

    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)

    lr_dir.mkdir(parents=True, exist_ok=True)

    hr_files = list(hr_dir.glob("*.tif"))

    print(f"Found {len(hr_files)} HR images.")

    for hr_path in hr_files:

        lr_path = lr_dir / hr_path.name

        create_lr_image(
            hr_path,
            lr_path,
            scale=scale
        )

        print(f"Created: {lr_path}")


if __name__ == "__main__":

    print("Dataset pair generator OK")