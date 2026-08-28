"""
Core Geospatial Raster Utilities built on Rasterio and NumPy.

Handles all remote-sensing I/O, coordinate reference system (CRS) conversions,
spatial resolution extraction, geotransform scaling, and GeoTIFF export.
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
import rasterio
from rasterio.transform import Affine, from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from rasterio.crs import CRS


VALID_CREATION_KEYS = {
    "driver", "width", "height", "count", "crs", "transform", "dtype",
    "nodata", "tiled", "blockxsize", "blockysize", "compress", "interleave",
    "photometric", "nbits", "predictor"
}


def sanitize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Filter dictionary to only retain valid rasterio GeoTIFF creation options."""
    return {k: v for k, v in profile.items() if k in VALID_CREATION_KEYS}


def get_raster_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract comprehensive geospatial and radiometric metadata from a raster file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        bounds = src.bounds
        res_x, res_y = src.res
        meta = {
            "path": str(path.resolve()),
            "filename": path.name,
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": src.crs.to_string() if src.crs else "Unspecified",
            "crs_epsg": src.crs.to_epsg() if src.crs else None,
            "is_projected": src.crs.is_projected if src.crs else False,
            "transform": src.transform,
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            },
            "resolution": (res_x, res_y),
            "nodata": src.nodata,
            "colorinterp": [ci.name for ci in src.colorinterp] if src.colorinterp else [],
            "tags": src.tags(),
        }
    return meta


def read_raster(
    path: Union[str, Path],
    bands: Optional[List[int]] = None,
    window: Optional[Window] = None,
    return_mask: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    """
    Read selected bands of a raster image.
    
    Args:
        path: Path to the GeoTIFF raster.
        bands: List of 1-indexed band indices (e.g. [1, 2, 3, 4]). None for all bands.
        window: Optional rasterio Window for windowed tile reading.
        return_mask: If True, returns a boolean mask (True for valid data, False for nodata).

    Returns:
        (array [C, H, W] as float32, profile dict, optional valid_mask [H, W])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        
        if window is not None:
            profile.update({
                "height": window.height,
                "width": window.width,
                "transform": rasterio.windows.transform(window, src.transform)
            })

        if bands is not None:
            for b in bands:
                if b < 1 or b > src.count:
                    raise ValueError(f"Band index {b} out of range (raster has {src.count} bands)")
            arr = src.read(bands, window=window)
            profile["count"] = len(bands)
        else:
            arr = src.read(window=window)

        # Handle nodata mask
        valid_mask = None
        if return_mask:
            if src.nodata is not None:
                if np.isnan(src.nodata):
                    valid_mask = ~np.isnan(arr).any(axis=0)
                else:
                    valid_mask = (arr != src.nodata).all(axis=0)
            else:
                valid_mask = np.ones((arr.shape[1], arr.shape[2]), dtype=bool)

    return arr.astype(np.float32), profile, valid_mask


def write_raster(
    array: np.ndarray,
    profile: Dict[str, Any],
    dst_path: Union[str, Path],
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
    compress: str = "deflate",
) -> Path:
    """
    Write an array to disk as a georeferenced GeoTIFF with standard compression.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        array = array[np.newaxis, ...]  # Shape [1, H, W]

    c, h, w = array.shape
    out_profile = sanitize_profile(profile)
    
    target_dtype = dtype or str(array.dtype)
    out_profile.update({
        "driver": "GTiff",
        "count": c,
        "height": h,
        "width": w,
        "dtype": target_dtype,
        "compress": compress,
        "tiled": True,
        "blockxsize": min(256, w),
        "blockysize": min(256, h),
    })

    if nodata is not None:
        out_profile["nodata"] = nodata

    with rasterio.open(dst_path, "w", **out_profile) as dst:
        for i in range(c):
            dst.write(array[i].astype(target_dtype), i + 1)

    return dst_path


def compute_sr_geotransform(
    reference_transform: Affine,
    scale_factor: float,
    origin_offset: Tuple[float, float] = (0.0, 0.0),
) -> Affine:
    """
    Compute the adjusted Affine geotransform for super-resolved imagery.
    Dividing pixel width/height by scale factor preserves the exact geographical extent.
    """
    a = reference_transform.a / scale_factor  # pixel width (m or deg)
    b = reference_transform.b / scale_factor  # row rotation
    c = reference_transform.c + origin_offset[0]  # x origin (top-left)
    d = reference_transform.d / scale_factor  # column rotation
    e = reference_transform.e / scale_factor  # pixel height (negative)
    f = reference_transform.f + origin_offset[1]  # y origin (top-left)
    return Affine(a, b, c, d, e, f)


def write_sr_output(
    array: np.ndarray,
    reference_profile: Dict[str, Any],
    dst_path: Union[str, Path],
    scale_factor: float = 4.0,
    nodata: Optional[float] = None,
    compress: str = "deflate",
) -> Path:
    """
    Write a super-resolved prediction array back as a georeferenced GeoTIFF.
    Accurately scales the spatial resolution and affine transform matrix.
    """
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    
    c, sr_h, sr_w = array.shape
    orig_transform = reference_profile["transform"]
    new_transform = compute_sr_geotransform(orig_transform, scale_factor)

    sr_profile = sanitize_profile(reference_profile)
    sr_profile.update({
        "driver": "GTiff",
        "height": sr_h,
        "width": sr_w,
        "count": c,
        "transform": new_transform,
        "dtype": str(array.dtype),
        "compress": compress,
        "tiled": True,
        "blockxsize": min(256, sr_w),
        "blockysize": min(256, sr_h),
    })

    if nodata is not None:
        sr_profile["nodata"] = nodata

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dst_path, "w", **sr_profile) as dst:
        for i in range(c):
            dst.write(array[i], i + 1)

    return dst_path


def reproject_raster(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    dst_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.bilinear,
    resolution: Optional[Tuple[float, float]] = None,
) -> Path:
    """
    Reproject a raster to a target CRS while preserving geospatial alignment.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        target_crs = CRS.from_user_input(dst_crs)
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=resolution
        )
        profile = sanitize_profile(src.profile)
        profile.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )
    return dst_path


def check_spatial_compatibility(path1: Union[str, Path], path2: Union[str, Path]) -> Dict[str, Any]:
    """
    Compare two rasters to check CRS alignment, spatial overlap, and resolution ratio.
    """
    m1 = get_raster_metadata(path1)
    m2 = get_raster_metadata(path2)

    crs_match = m1["crs"] == m2["crs"]
    b1 = m1["bounds"]
    b2 = m2["bounds"]

    overlap_left = max(b1["left"], b2["left"])
    overlap_right = min(b1["right"], b2["right"])
    overlap_bottom = max(b1["bottom"], b2["bottom"])
    overlap_top = min(b1["top"], b2["top"])
    
    has_overlap = (overlap_left < overlap_right) and (overlap_bottom < overlap_top)
    
    scale_ratio_x = m1["resolution"][0] / (m2["resolution"][0] + 1e-9)
    scale_ratio_y = m1["resolution"][1] / (m2["resolution"][1] + 1e-9)

    return {
        "crs_match": crs_match,
        "has_overlap": has_overlap,
        "overlap_bounds": (overlap_left, overlap_bottom, overlap_right, overlap_top) if has_overlap else None,
        "resolution_ratio": (round(scale_ratio_x, 2), round(scale_ratio_y, 2)),
        "meta1": m1,
        "meta2": m2,
    }
