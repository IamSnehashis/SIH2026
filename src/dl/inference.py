"""
Applies the trained SR model to a real medium-res GeoTIFF scene and writes
out a georeferenced super-resolved GeoTIFF. This is the function the FastAPI
backend calls too (see src/backend/main.py) -- keep it side-effect-light and
reusable rather than script-only.
"""
import argparse
import numpy as np
import torch

from src.dl.models import build_model
from src.gis.preprocess import normalize
from src.gis.raster_utils import read_raster, write_sr_output
from src.utils.config import load_config


def run_inference(input_path: str, output_path: str, weights_path: str,
                   cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    device = torch.device(cfg["dl"]["device"] if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    arr, profile = read_raster(input_path, bands=cfg["gis"]["bands"])
    arr_norm = normalize(arr)

    # Bicubic pre-upsample to match the tiling convention used in training,
    # then let the model refine detail (see EDSR docstring in models.py).
    import cv2
    c, h, w = arr_norm.shape
    scale = cfg["gis"]["scale_factor"]
    upsampled = np.zeros((c, h * scale, w * scale), dtype=np.float32)
    for i in range(c):
        upsampled[i] = cv2.resize(arr_norm[i], (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    tensor = torch.from_numpy(upsampled).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).squeeze(0).clamp(0, 1).cpu().numpy()

    return write_sr_output(pred, profile, output_path, scale_factor=scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default="src/utils/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = run_inference(args.input, args.output, args.weights, cfg)
    print(f"Wrote super-resolved GeoTIFF to {out}")


if __name__ == "__main__":
    main()
