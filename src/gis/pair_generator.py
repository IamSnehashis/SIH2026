"""
Dataset Pair Generator for Satellite Super-Resolution Mapping.

Supports:
1. Synthetic pair generation from high-resolution scenes with realistic satellite
   point spread function (PSF) Gaussian blur, decimation downsampling, and radiometric sensor noise.
2. Real-world paired scene alignment, reprojection, and co-registration (e.g. Landsat-8 OLI to Sentinel-2 MSI).
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
import cv2
import rasterio
from rasterio.enums import Resampling

from src.gis.raster_utils import read_raster, write_raster, compute_sr_geotransform
from src.gis.preprocessor import Normalizer
from src.gis.tiling import SceneTiler, TileMetadata


def apply_sensor_degradation(
    hr_patch: np.ndarray,
    scale_factor: int = 4,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.2,
    noise_sigma: float = 0.005,
    pre_upsample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate lower-resolution satellite sensor acquisition from high-res imagery.

    Steps:
      1. Point Spread Function (PSF) blur via Gaussian filter.
      2. Pixel decimation / area-averaging downsampling by scale_factor.
      3. Additive Gaussian sensor noise.
      4. (Optional) Bicubic upsample back to HR dimensions if the network expects
         pre-upsampled inputs (like standard SRCNN/EDSR architectures).

    Returns:
        (native_lr_patch [C, H//s, W//s], model_input_patch [C, H, W] if pre_upsample else [C, H//s, W//s])
    """
    c, h, w = hr_patch.shape
    lr_h = h // scale_factor
    lr_w = w // scale_factor

    native_lr = np.zeros((c, lr_h, lr_w), dtype=np.float32)

    for i in range(c):
        band = hr_patch[i]
        # 1. PSF Blur
        if blur_kernel_size > 0:
            k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
            blurred = cv2.GaussianBlur(band, (k, k), blur_sigma)
        else:
            blurred = band

        # 2. Subsample to lower sensor resolution
        down = cv2.resize(blurred, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

        # 3. Add sensor noise
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, down.shape).astype(np.float32)
            down = np.clip(down + noise, 0.0, 1.0)

        native_lr[i] = down

    if pre_upsample:
        # Pre-upsample with bicubic interpolation
        model_lr = np.zeros_like(hr_patch, dtype=np.float32)
        for i in range(c):
            model_lr[i] = cv2.resize(native_lr[i], (w, h), interpolation=cv2.INTER_CUBIC)
        return native_lr, model_lr
    else:
        return native_lr, native_lr


def generate_dataset_pairs_from_scenes(
    raw_dir: Union[str, Path],
    output_tiles_dir: Union[str, Path],
    tile_size: int = 128,
    overlap: int = 16,
    scale_factor: int = 4,
    bands: Optional[List[int]] = None,
    pre_upsample: bool = True,
    save_geotiffs: bool = False,
) -> Dict[str, Any]:
    """
    Process raw satellite scenes into paired LR and HR training/testing tiles.
    
    Saves `.npy` tile pairs (and optionally georeferenced `.tif` pairs) to `output_tiles_dir`.
    """
    raw_dir = Path(raw_dir)
    output_tiles_dir = Path(output_tiles_dir)
    output_tiles_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.TIF")))
    if not scenes:
        return {"num_scenes": 0, "num_pairs": 0, "pairs": []}

    tiler = SceneTiler(tile_size=tile_size, overlap=overlap, min_valid_ratio=0.3)
    normalizer = Normalizer(method="percentile", p_low=2.0, p_high=98.0)

    total_pairs = 0
    manifest = []

    for scene_path in scenes:
        arr, profile, mask = read_raster(scene_path, bands=bands, return_mask=True)
        arr_norm = normalizer.fit_transform(arr, mask=mask)

        tiles = tiler.extract_tiles_from_array(arr_norm, profile, scene_name=scene_path.stem, mask=mask)

        for hr_patch, meta in tiles:
            native_lr, lr_patch = apply_sensor_degradation(
                hr_patch,
                scale_factor=scale_factor,
                pre_upsample=pre_upsample,
            )

            stem = f"{scene_path.stem}_tile{meta.tile_id:05d}"
            hr_npy_path = output_tiles_dir / f"{stem}_hr.npy"
            lr_npy_path = output_tiles_dir / f"{stem}_lr.npy"

            np.save(hr_npy_path, hr_patch)
            np.save(lr_npy_path, lr_patch)

            record = {
                "stem": stem,
                "scene": scene_path.name,
                "tile_id": meta.tile_id,
                "hr_npy": str(hr_npy_path.name),
                "lr_npy": str(lr_npy_path.name),
                "bounds": meta.bounds,
                "grid_pos": {"x": meta.x, "y": meta.y},
            }

            if save_geotiffs:
                # Save georeferenced GeoTIFFs for GIS inspection
                hr_tif_path = output_tiles_dir / f"{stem}_hr.tif"
                lr_tif_path = output_tiles_dir / f"{stem}_lr.tif"
                
                tile_profile = profile.copy()
                tile_profile.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": meta.transform,
                    "count": hr_patch.shape[0],
                    "dtype": "float32",
                })
                write_raster(hr_patch, tile_profile, hr_tif_path)
                write_raster(lr_patch, tile_profile, lr_tif_path)
                record["hr_tif"] = str(hr_tif_path.name)
                record["lr_tif"] = str(lr_tif_path.name)

            manifest.append(record)
            total_pairs += 1

    # Save manifest
    normalizer.export_manifest(output_tiles_dir / "normalization_stats.json")
    import json
    with open(output_tiles_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"total_pairs": total_pairs, "scale_factor": scale_factor, "pairs": manifest}, f, indent=2)

    return {"num_scenes": len(scenes), "num_pairs": total_pairs, "manifest_path": str(output_tiles_dir / "dataset_manifest.json")}
