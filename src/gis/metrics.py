"""
Remote Sensing & GIS Quality Evaluation Metrics.

Computes both standard computer vision metrics (PSNR, SSIM, RMSE, MAE)
and specialized remote sensing radiometric fidelity metrics (SAM, ERGAS, UIQI).
Generates spatial difference error heatmaps and evaluation reports.
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import json
import numpy as np
import cv2
import matplotlib
from skimage.metrics import structural_similarity as ssim_fn

from src.gis.raster_utils import write_raster


def calculate_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher is better (>30 dB is standard good quality).
    """
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < eps:
        return 100.0
    psnr = 20.0 * np.log10(data_range / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Mean Structural Similarity Index Measure (SSIM) across all channels.
    Values range from -1.0 to 1.0 (1.0 is identical structure).
    """
    if pred.ndim == 3:
        c, h, w = pred.shape
        scores = []
        for i in range(c):
            s = ssim_fn(
                target[i], pred[i],
                data_range=data_range,
                win_size=min(7, min(h, w) if min(h, w) % 2 == 1 else min(h, w) - 1)
            )
            scores.append(s)
        return float(np.mean(scores))
    elif pred.ndim == 2:
        h, w = pred.shape
        return float(ssim_fn(target, pred, data_range=data_range, win_size=min(7, min(h, w))))
    return 0.0


def calculate_sam(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Spectral Angle Mapper (SAM) in degrees.
    Measures the spectral angle between target and predicted multi-band pixel vectors.
    Lower is better (0.0 degrees = perfect spectral fidelity).
    """
    if pred.ndim != 3 or pred.shape[0] < 2:
        return 0.0

    c, h, w = pred.shape
    p_flat = pred.reshape(c, -1).astype(np.float64)
    t_flat = target.reshape(c, -1).astype(np.float64)

    dot_prod = np.sum(p_flat * t_flat, axis=0)
    norm_p = np.sqrt(np.sum(p_flat ** 2, axis=0)) + eps
    norm_t = np.sqrt(np.sum(t_flat ** 2, axis=0)) + eps

    cos_theta = np.clip(dot_prod / (norm_p * norm_t), -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    angles_deg = np.degrees(angles_rad)

    return float(np.nanmean(angles_deg))


def calculate_ergas(
    pred: np.ndarray,
    target: np.ndarray,
    scale_factor: float = 4.0,
    eps: float = 1e-8,
) -> float:
    """
    Relative Dimensionless Global Error in Synthesis (ERGAS).
    Standard remote sensing benchmark metric. Lower is better (0.0 is perfect).
    """
    if pred.ndim != 3:
        return 0.0

    c, h, w = pred.shape
    sum_rmse_ratio = 0.0

    for i in range(c):
        p_b = pred[i].astype(np.float64)
        t_b = target[i].astype(np.float64)
        rmse_b = np.sqrt(np.mean((p_b - t_b) ** 2))
        mean_t = np.mean(t_b) + eps
        sum_rmse_ratio += (rmse_b / mean_t) ** 2

    ergas = (100.0 / scale_factor) * np.sqrt(sum_rmse_ratio / c)
    return float(ergas)


def calculate_uiqi(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Universal Image Quality Index (UIQI / Q-index).
    Range: [-1.0, 1.0], higher is better (1.0 is identical).
    """
    p = pred.astype(np.float64)
    t = target.astype(np.float64)

    mean_p = np.mean(p)
    mean_t = np.mean(t)
    var_p = np.var(p)
    var_t = np.var(t)
    cov_pt = np.mean((p - mean_p) * (t - mean_t))

    num = 4.0 * cov_pt * mean_p * mean_t
    den = (var_p + var_t + eps) * (mean_p ** 2 + mean_t ** 2 + eps)
    return float(num / den)


def evaluate_super_resolution(
    pred: np.ndarray,
    target: np.ndarray,
    scale_factor: float = 4.0,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """
    Compute comprehensive full-reference quality metrics.
    """
    p = np.clip(pred, 0.0, data_range)
    t = np.clip(target, 0.0, data_range)

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    psnr = calculate_psnr(p, t, data_range=data_range)
    ssim = calculate_ssim(p, t, data_range=data_range)
    sam = calculate_sam(p, t)
    ergas = calculate_ergas(p, t, scale_factor=scale_factor)
    uiqi = calculate_uiqi(p, t)

    return {
        "PSNR (dB)": round(psnr, 2),
        "SSIM": round(ssim, 4),
        "RMSE": round(rmse, 5),
        "MAE": round(mae, 5),
        "SAM (deg)": round(sam, 3),
        "ERGAS": round(ergas, 3),
        "UIQI": round(uiqi, 4),
    }


def generate_error_heatmap(
    pred: np.ndarray,
    target: np.ndarray,
    reference_profile: Dict[str, Any],
    output_dir: Union[str, Path],
    filename_prefix: str = "sr_error",
) -> Dict[str, Path]:
    """
    Generate pixel-wise spatial difference / absolute error heatmaps (GeoTIFF + PNG).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Channel-averaged absolute error
    diff = np.mean(np.abs(pred - target), axis=0).astype(np.float32)

    tif_path = output_dir / f"{filename_prefix}_diff.tif"
    png_path = output_dir / f"{filename_prefix}_diff.png"

    # Write GeoTIFF
    heat_profile = reference_profile.copy()
    heat_profile.update({
        "count": 1,
        "dtype": "float32",
        "height": diff.shape[0],
        "width": diff.shape[1],
    })
    write_raster(diff, heat_profile, tif_path, dtype="float32")

    # Write colorized PNG heatmap via OpenCV
    norm_diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
    try:
        cmap = matplotlib.colormaps["magma"]
    except Exception:
        cmap = matplotlib.pyplot.get_cmap("magma")

    rgba = (cmap(norm_diff) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(str(png_path), bgr)

    return {"geotiff": tif_path, "png": png_path}
