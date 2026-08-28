from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class SatelliteDataset(Dataset):
    """
    PyTorch dataset for paired satellite GeoTIFF images.

    LR images are the input.
    HR images are the target.

    Example:
        LR: 160 x 160 x 3
        HR: 480 x 480 x 3
    """

    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)

        self.lr_files = sorted(self.lr_dir.glob("*.tif"))
        self.hr_files = sorted(self.hr_dir.glob("*.tif"))

        if len(self.lr_files) != len(self.hr_files):
            raise ValueError(
                f"Number of LR images ({len(self.lr_files)}) "
                f"does not match HR images ({len(self.hr_files)})"
            )

        if len(self.lr_files) == 0:
            raise ValueError(
                f"No .tif files found in {self.lr_dir}"
            )

        # Verify that LR and HR filenames match
        for lr_path, hr_path in zip(self.lr_files, self.hr_files):
            if lr_path.name != hr_path.name:
                raise ValueError(
                    f"LR/HR filename mismatch:\n"
                    f"LR: {lr_path.name}\n"
                    f"HR: {hr_path.name}"
                )

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, index):
        lr_path = self.lr_files[index]
        hr_path = self.hr_files[index]

        # Read LR image
        with rasterio.open(lr_path) as src:
            lr = src.read().astype(np.float32)

        # Read HR image
        with rasterio.open(hr_path) as src:
            hr = src.read().astype(np.float32)

        # Convert from NumPy arrays:
        # (bands, height, width)
        # to PyTorch tensors
        lr = torch.from_numpy(lr)
        hr = torch.from_numpy(hr)

        return lr, hr


if __name__ == "__main__":
    print("SatelliteDataset module OK")