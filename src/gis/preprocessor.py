"""
GIS Pre-Processing Engine for Satellite Imagery.

Handles radiometric calibration, contrast stretching, percentile normalization,
TOA/Surface Reflectance scaling, NoData masking, and normalization manifest logging.
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import json
import numpy as np

from src.gis.raster_utils import read_raster


class Normalizer:
    """
    Stateful normalizer that records per-band statistics for deterministic
    normalization and exact inverse de-normalization after ML inference.
    """

    def __init__(
        self,
        method: str = "percentile",
        p_low: float = 2.0,
        p_high: float = 98.0,
        target_range: Tuple[float, float] = (0.0, 1.0),
        clip: bool = True,
    ):
        """
        Args:
            method: 'percentile' (2%-98% clip), 'minmax', 'reflectance' (0-10000 -> 0-1), or 'zscore'.
            p_low: Lower percentile for percentile method.
            p_high: Upper percentile for percentile method.
            target_range: Output target min and max range (e.g. (0.0, 1.0) or (-1.0, 1.0)).
            clip: Whether to clip output values to target_range.
        """
        self.method = method
        self.p_low = p_low
        self.p_high = p_high
        self.target_range = target_range
        self.clip = clip
        self.stats: List[Dict[str, float]] = []

    def fit_transform(self, arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute normalization parameters and normalize the input array.
        
        Args:
            arr: NumPy array of shape [C, H, W]
            mask: Optional boolean valid mask [H, W] (True for valid pixels, False for nodata)
            
        Returns:
            Normalized array [C, H, W] in target_range
        """
        c, h, w = arr.shape
        out = np.zeros_like(arr, dtype=np.float32)
        self.stats = []

        t_min, t_max = self.target_range
        t_range = t_max - t_min

        for i in range(c):
            band = arr[i]
            valid_pixels = band[mask] if mask is not None else band.ravel()

            # Handle edge case of all nodata or zero-variance
            if len(valid_pixels) == 0 or np.all(valid_pixels == valid_pixels[0]):
                b_min, b_max = float(band.min()), float(band.max())
                self.stats.append({"min": b_min, "max": b_max, "lo": b_min, "hi": b_max, "mean": b_min, "std": 1.0})
                out[i] = np.full_like(band, t_min, dtype=np.float32)
                continue

            b_min = float(valid_pixels.min())
            b_max = float(valid_pixels.max())
            b_mean = float(valid_pixels.mean())
            b_std = float(valid_pixels.std()) + 1e-7

            if self.method == "percentile":
                lo = float(np.percentile(valid_pixels, self.p_low))
                hi = float(np.percentile(valid_pixels, self.p_high))
                denom = hi - lo if (hi - lo) > 1e-6 else (b_max - b_min + 1e-6)
                norm_band = ((band - lo) / denom) * t_range + t_min

            elif self.method == "minmax":
                lo, hi = b_min, b_max
                denom = hi - lo if (hi - lo) > 1e-6 else 1.0
                norm_band = ((band - lo) / denom) * t_range + t_min

            elif self.method == "reflectance":
                # Typical Sentinel-2 / Landsat L2A surface reflectance integer scale (0-10000)
                lo, hi = 0.0, 10000.0
                norm_band = (band / 10000.0) * t_range + t_min

            elif self.method == "zscore":
                lo, hi = b_mean, b_std
                norm_band = (band - b_mean) / b_std
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")

            if self.clip and self.method != "zscore":
                norm_band = np.clip(norm_band, t_min, t_max)

            out[i] = norm_band
            self.stats.append({
                "band_idx": i + 1,
                "min": b_min,
                "max": b_max,
                "lo": lo,
                "hi": hi,
                "mean": b_mean,
                "std": b_std
            })

        return out

    def inverse_transform(self, norm_arr: np.ndarray) -> np.ndarray:
        """
        Reverse normalization using fitted per-band statistics.
        """
        if not self.stats:
            raise RuntimeError("Normalizer has not been fitted yet.")

        c, h, w = norm_arr.shape
        out = np.zeros_like(norm_arr, dtype=np.float32)
        t_min, t_max = self.target_range
        t_range = t_max - t_min

        for i in range(min(c, len(self.stats))):
            st = self.stats[i]
            band = norm_arr[i]

            if self.method in ["percentile", "minmax"]:
                denom = st["hi"] - st["lo"] if (st["hi"] - st["lo"]) > 1e-6 else 1.0
                unnorm = ((band - t_min) / t_range) * denom + st["lo"]

            elif self.method == "reflectance":
                unnorm = ((band - t_min) / t_range) * 10000.0

            elif self.method == "zscore":
                unnorm = (band * st["std"]) + st["mean"]
            else:
                unnorm = band

            out[i] = unnorm

        return out

    def export_manifest(self, filepath: Union[str, Path]):
        """Save normalization statistics to a JSON manifest for reproducibility."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "method": self.method,
            "p_low": self.p_low,
            "p_high": self.p_high,
            "target_range": list(self.target_range),
            "clip": self.clip,
            "stats": self.stats,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def from_manifest(cls, filepath: Union[str, Path]) -> "Normalizer":
        """Load normalizer configuration and statistics from a JSON manifest."""
        with open(filepath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        obj = cls(
            method=manifest["method"],
            p_low=manifest.get("p_low", 2.0),
            p_high=manifest.get("p_high", 98.0),
            target_range=tuple(manifest.get("target_range", (0.0, 1.0))),
            clip=manifest.get("clip", True),
        )
        obj.stats = manifest.get("stats", [])
        return obj


def preprocess_scene(
    scene_path: Union[str, Path],
    bands: Optional[List[int]] = None,
    method: str = "percentile",
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> Tuple[np.ndarray, Dict[str, Any], Normalizer, np.ndarray]:
    """
    Complete single-scene GIS pre-processing pipeline.
    
    Reads raster, extracts metadata, identifies nodata mask, and normalizes.
    
    Returns:
        (normalized_array [C, H, W], profile dict, normalizer object, valid_mask [H, W])
    """
    arr, profile, valid_mask = read_raster(scene_path, bands=bands, return_mask=True)
    normalizer = Normalizer(method=method, p_low=p_low, p_high=p_high)
    arr_norm = normalizer.fit_transform(arr, mask=valid_mask)
    return arr_norm, profile, normalizer, valid_mask
