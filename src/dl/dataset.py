from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SRMTileDataset(Dataset):
    """Loads pre-tiled LR/HR .npy pairs produced by src/gis/preprocess.py."""

    def __init__(self, tiles_dir: str, split: str = "train", val_split: float = 0.15):
        self.tiles_dir = Path(tiles_dir)
        hr_files = sorted(self.tiles_dir.glob("*_hr.npy"))
        n_val = int(len(hr_files) * val_split)

        if split == "train":
            self.hr_files = hr_files[n_val:]
        else:
            self.hr_files = hr_files[:n_val]

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = hr_path.with_name(hr_path.name.replace("_hr.npy", "_lr.npy"))

        hr = np.load(hr_path)
        lr = np.load(lr_path)
        return torch.from_numpy(lr).float(), torch.from_numpy(hr).float()
