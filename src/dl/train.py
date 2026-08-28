import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from tqdm import tqdm

from src.dl.dataset import SRMTileDataset
from src.dl.models import build_model
from src.utils.config import load_config


def combined_loss(pred, target):
    l1 = torch.nn.functional.l1_loss(pred, target)
    s = 1 - ssim(pred.clamp(0, 1), target.clamp(0, 1), data_range=1.0, size_average=True)
    return l1 + 0.1 * s


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for lr, hr in tqdm(loader, leave=False):
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            loss = combined_loss(pred, hr)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * lr.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/utils/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg["dl"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_ds = SRMTileDataset(cfg["paths"]["tiles_dir"], "train", cfg["dl"]["val_split"])
    val_ds = SRMTileDataset(cfg["paths"]["tiles_dir"], "val", cfg["dl"]["val_split"])
    train_loader = DataLoader(train_ds, batch_size=cfg["dl"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg["dl"]["batch_size"], num_workers=2)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["dl"]["lr"])

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(cfg["dl"]["epochs"]):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        print(f"Epoch {epoch+1}/{cfg['dl']['epochs']}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), models_dir / cfg["dl"]["checkpoint_name"])
            print(f"  -> saved new best checkpoint (val={val_loss:.4f})")


if __name__ == "__main__":
    main()
