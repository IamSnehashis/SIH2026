"""
Biophysical Spectral Index Engine for Satellite Imagery.

Calculates key remote sensing vegetation, water, and urban indices on super-resolved output:
- NDVI: Normalized Difference Vegetation Index (NIR - Red) / (NIR + Red)
- NDWI: Normalized Difference Water Index (Green - NIR) / (Green + NIR)
- NDBI: Normalized Difference Built-up Index (SWIR - NIR) / (SWIR + NIR)
- EVI:  Enhanced Vegetation Index 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
- SAVI: Soil-Adjusted Vegetation Index ((NIR - Red) / (NIR + Red + L)) * (1 + L)
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
import cv2
import matplotlib
import matplotlib.cm as cm

from src.gis.raster_utils import write_raster, read_raster


def calculate_ndvi(
    nir_band: np.ndarray,
    red_band: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index (NDVI).
    Values range from -1.0 to +1.0 (dense healthy vegetation > 0.5, water/soil < 0.1).
    """
    denom = nir_band + red_band + eps
    ndvi = (nir_band - red_band) / denom
    return np.clip(ndvi, -1.0, 1.0)


def calculate_ndwi(
    green_band: np.ndarray,
    nir_band: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute McFeeters Normalized Difference Water Index (NDWI).
    Values > 0 typically correspond to open water bodies (rivers, lakes, reservoirs).
    """
    denom = green_band + nir_band + eps
    ndwi = (green_band - nir_band) / denom
    return np.clip(ndwi, -1.0, 1.0)


def calculate_ndbi(
    swir_band: np.ndarray,
    nir_band: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute Normalized Difference Built-up Index (NDBI).
    Positive values indicate built-up urban / impervious surfaces.
    """
    denom = swir_band + nir_band + eps
    ndbi = (swir_band - nir_band) / denom
    return np.clip(ndbi, -1.0, 1.0)


def calculate_evi(
    nir_band: np.ndarray,
    red_band: np.ndarray,
    blue_band: np.ndarray,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute Enhanced Vegetation Index (EVI), optimized for high-biomass canopy regions.
    """
    denom = nir_band + (c1 * red_band) - (c2 * blue_band) + l + eps
    evi = g * ((nir_band - red_band) / denom)
    return np.clip(evi, -1.0, 1.5)


def calculate_savi(
    nir_band: np.ndarray,
    red_band: np.ndarray,
    l: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute Soil-Adjusted Vegetation Index (SAVI) with canopy background adjustment factor L.
    """
    denom = nir_band + red_band + l + eps
    savi = ((nir_band - red_band) / denom) * (1.0 + l)
    return np.clip(savi, -1.0, 1.0)


def compute_spectral_indices(
    raster_array: np.ndarray,
    band_mapping: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute all available spectral indices given a multi-band array [C, H, W].
    
    Args:
        raster_array: Array [C, H, W]
        band_mapping: Optional dict mapping band names ('blue', 'green', 'red', 'nir', 'swir')
                      to 0-indexed channel integers. Default assumes standard 4-band:
                      0: Blue, 1: Green, 2: Red, 3: NIR.
    """
    if band_mapping is None:
        c = raster_array.shape[0]
        if c >= 4:
            band_mapping = {"blue": 0, "green": 1, "red": 2, "nir": 3}
            if c >= 5:
                band_mapping["swir"] = 4
        elif c == 3:
            band_mapping = {"blue": 0, "green": 1, "red": 2}
        else:
            band_mapping = {"gray": 0}

    indices = {}

    if "nir" in band_mapping and "red" in band_mapping:
        nir = raster_array[band_mapping["nir"]]
        red = raster_array[band_mapping["red"]]
        indices["ndvi"] = calculate_ndvi(nir, red)
        indices["savi"] = calculate_savi(nir, red)

        if "blue" in band_mapping:
            blue = raster_array[band_mapping["blue"]]
            indices["evi"] = calculate_evi(nir, red, blue)

    if "green" in band_mapping and "nir" in band_mapping:
        green = raster_array[band_mapping["green"]]
        nir = raster_array[band_mapping["nir"]]
        indices["ndwi"] = calculate_ndwi(green, nir)

    if "swir" in band_mapping and "nir" in band_mapping:
        swir = raster_array[band_mapping["swir"]]
        nir = raster_array[band_mapping["nir"]]
        indices["ndbi"] = calculate_ndbi(swir, nir)

    return indices


def export_index_geotiff_and_png(
    index_array: np.ndarray,
    reference_profile: Dict[str, Any],
    output_dir: Union[str, Path],
    index_name: str = "ndvi",
    colormap_name: str = "RdYlGn",
) -> Dict[str, Path]:
    """
    Save calculated spectral index as a single-band GeoTIFF and a color-mapped PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_path = output_dir / f"{index_name}.tif"
    png_path = output_dir / f"{index_name}.png"

    # Save single-band float32 GeoTIFF
    idx_profile = reference_profile.copy()
    idx_profile.update({
        "count": 1,
        "dtype": "float32",
        "nodata": -9999.0,
    })
    write_raster(index_array, idx_profile, tif_path, dtype="float32", nodata=-9999.0)

    # Save colorized PNG directly via OpenCV (fast, headless, crash-proof)
    norm_val = (index_array - index_array.min()) / (index_array.max() - index_array.min() + 1e-6)
    
    try:
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = matplotlib.pyplot.get_cmap(colormap_name)

    rgba = (cmap(norm_val) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(str(png_path), bgr)

    return {"geotiff": tif_path, "png": png_path}
