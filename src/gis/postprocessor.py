"""
GIS Post-Processing Engine for Satellite Super-Resolution Mapping.

Features:
1. Seamless Patch Mosaicking with 2D Cosine / Gaussian / Linear Distance-Weighted Overlap Blending (Feathering).
2. Georeferencing & Spatial Resolution Re-scaling (Transform Matrix calculation).
3. Radiometric De-normalization and Histogram Matching against reference imagery.
4. Exporting fully georeferenced GeoTIFFs compatible with QGIS, ArcGIS, and Web GIS.
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
import rasterio

from src.gis.raster_utils import write_sr_output, write_raster, compute_sr_geotransform
from src.gis.preprocessor import Normalizer
from src.gis.tiling import TileMetadata


def generate_blending_weights(height: int, width: int, method: str = "cosine") -> np.ndarray:
    """
    Create a 2D weighting window [H, W] that tapers to zero at the tile boundaries.
    Used to blend overlapping tiles smoothly without visible seamlines.
    
    Methods:
      - 'cosine': Raised cosine window (Hann window in 2D) - smooth, continuous derivatives.
      - 'linear': 2D pyramidal/tent window.
      - 'gaussian': Normalized 2D Gaussian bell.
    """
    if method == "cosine":
        wy = np.sin(np.linspace(0, np.pi, height)) ** 2
        wx = np.sin(np.linspace(0, np.pi, width)) ** 2
        w2d = np.outer(wy, wx)
    elif method == "linear":
        wy = 1.0 - np.abs(np.linspace(-1, 1, height))
        wx = 1.0 - np.abs(np.linspace(-1, 1, width))
        w2d = np.outer(wy, wx)
    elif method == "gaussian":
        y, x = np.mgrid[-2:2:complex(0, height), -2:2:complex(0, width)]
        w2d = np.exp(-(x**2 + y**2) / 2.0)
    else:
        w2d = np.ones((height, width), dtype=np.float32)

    # Avoid divide by zero
    w2d = np.maximum(w2d, 1e-4).astype(np.float32)
    return w2d


class SceneMosaicker:
    """
    Reconstructs an arbitrary-sized full satellite scene from tiled predictions
    using weighted overlap blending (feathering).
    """

    def __init__(
        self,
        full_height: int,
        full_width: int,
        channels: int,
        scale_factor: float = 1.0,
        blend_method: str = "cosine",
    ):
        """
        Args:
            full_height: Target output height in pixels (original_h * scale_factor).
            full_width: Target output width in pixels (original_w * scale_factor).
            channels: Number of spectral channels/bands.
            scale_factor: Super-resolution upscale factor.
            blend_method: 'cosine', 'linear', 'gaussian', or 'none'.
        """
        self.target_h = int(round(full_height * scale_factor))
        self.target_w = int(round(full_width * scale_factor))
        self.channels = channels
        self.scale_factor = scale_factor
        self.blend_method = blend_method

        # Accumulators
        self.accumulator = np.zeros((channels, self.target_h, self.target_w), dtype=np.float32)
        self.weight_map = np.zeros((self.target_h, self.target_w), dtype=np.float32)

    def add_tile(self, pred_patch: np.ndarray, orig_x: int, orig_y: int):
        """
        Incorporate a predicted super-resolved tile into the full scene mosaic.
        
        Args:
            pred_patch: Array [C, H_sr, W_sr] (e.g. 480x480 or 128x128).
            orig_x: Top-left X coordinate of the tile in the ORIGINAL low-res scene.
            orig_y: Top-left Y coordinate of the tile in the ORIGINAL low-res scene.
        """
        c, ph, pw = pred_patch.shape
        dest_x = int(round(orig_x * self.scale_factor))
        dest_y = int(round(orig_y * self.scale_factor))

        # Check bounds
        h_end = min(dest_y + ph, self.target_h)
        w_end = min(dest_x + pw, self.target_w)
        patch_h = h_end - dest_y
        patch_w = w_end - dest_x

        if patch_h <= 0 or patch_w <= 0:
            return

        patch_crop = pred_patch[:, :patch_h, :patch_w]

        # Generate weights
        weights = generate_blending_weights(ph, pw, method=self.blend_method)[:patch_h, :patch_w]

        # Weighted accumulation
        for ch in range(self.channels):
            self.accumulator[ch, dest_y:h_end, dest_x:w_end] += patch_crop[ch] * weights

        self.weight_map[dest_y:h_end, dest_x:w_end] += weights

    def finalize(self) -> np.ndarray:
        """
        Normalize accumulated values by total overlapping weights to produce the final seamless scene.
        """
        valid_mask = self.weight_map > 1e-6
        safe_weights = np.where(valid_mask, self.weight_map, 1.0)

        result = np.zeros_like(self.accumulator)
        for ch in range(self.channels):
            result[ch] = np.where(valid_mask, self.accumulator[ch] / safe_weights, 0.0)

        return np.clip(result, 0.0, 1.0)


def postprocess_and_export(
    sr_array: np.ndarray,
    reference_profile: Dict[str, Any],
    output_path: Union[str, Path],
    scale_factor: float = 4.0,
    normalizer: Optional[Normalizer] = None,
    output_dtype: str = "float32",
    nodata: Optional[float] = None,
    compress: str = "deflate",
) -> Path:
    """
    Complete GIS post-processing pipeline:
      1. Optionally invert normalization back to physical reflectance / digital numbers.
      2. Re-register geospatial coordinates with scaled pixel resolution.
      3. Write out a georeferenced GeoTIFF ready for QGIS, ArcGIS, or web delivery.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalizer is not None:
        sr_export = normalizer.inverse_transform(sr_array)
    else:
        sr_export = sr_array

    if output_dtype == "uint8":
        sr_export = (np.clip(sr_export, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif output_dtype == "uint16":
        sr_export = (np.clip(sr_export, 0.0, 1.0) * 10000.0).astype(np.uint16)
    else:
        sr_export = sr_export.astype(np.float32)

    return write_sr_output(
        array=sr_export,
        reference_profile=reference_profile,
        dst_path=output_path,
        scale_factor=scale_factor,
        nodata=nodata,
        compress=compress,
    )
