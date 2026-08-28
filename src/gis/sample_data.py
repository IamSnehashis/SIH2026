"""
Sample Satellite Scene Generator.

Generates realistic multi-band GeoTIFF scenes (RGB + NIR) with geographic features:
- Winding river/water bodies (high NDWI, low NDVI)
- Dense forest/vegetation patches (high NDVI)
- Agricultural field grid patterns
- Urban / built-up clusters with road networks (high NDBI)
- Georeferenced CRS (UTM Zone 43N or WGS84) with exact bounding boxes
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS


def generate_synthetic_satellite_scene(
    output_path: Union[str, Path],
    height: int = 512,
    width: int = 512,
    crs: str = "EPSG:32643",  # UTM Zone 43N (meters)
    bounds: Tuple[float, float, float, float] = (500000.0, 3000000.0, 505120.0, 3005120.0), # 10m/px
    seed: int = 42,
) -> Path:
    """
    Synthesize a 4-band satellite image:
      Band 1: Blue
      Band 2: Green
      Band 3: Red
      Band 4: Near-Infrared (NIR)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # Base land reflectance (soil / background)
    blue = np.random.uniform(0.12, 0.18, (height, width)).astype(np.float32)
    green = np.random.uniform(0.15, 0.22, (height, width)).astype(np.float32)
    red = np.random.uniform(0.18, 0.26, (height, width)).astype(np.float32)
    nir = np.random.uniform(0.20, 0.30, (height, width)).astype(np.float32)

    # 1. Add Winding River / Water Body
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    river_center = (width * 0.45) + (width * 0.2) * np.sin(y_grid / (height * 0.15))
    river_mask = np.abs(x_grid - river_center) < (width * 0.04)

    # Water has high blue/green, low red, almost zero NIR
    blue[river_mask] = np.random.uniform(0.25, 0.35, np.sum(river_mask))
    green[river_mask] = np.random.uniform(0.20, 0.30, np.sum(river_mask))
    red[river_mask] = np.random.uniform(0.05, 0.10, np.sum(river_mask))
    nir[river_mask] = np.random.uniform(0.01, 0.04, np.sum(river_mask))

    # 2. Add Dense Forest Canopy / Nature Reserve (Top-Right)
    forest_mask = ((x_grid - width * 0.75)**2 + (y_grid - height * 0.3)**2) < ((width * 0.2)**2)
    # Forest has low red, very high NIR (high NDVI)
    blue[forest_mask] = np.random.uniform(0.04, 0.08, np.sum(forest_mask))
    green[forest_mask] = np.random.uniform(0.10, 0.18, np.sum(forest_mask))
    red[forest_mask] = np.random.uniform(0.03, 0.07, np.sum(forest_mask))
    nir[forest_mask] = np.random.uniform(0.65, 0.85, np.sum(forest_mask))

    # 3. Add Agricultural Fields (Bottom-Left Grid)
    field_mask = (x_grid < width * 0.4) & (y_grid > height * 0.5) & (~river_mask)
    grid_pattern = ((x_grid // 32) + (y_grid // 32)) % 2 == 0
    crop_mask = field_mask & grid_pattern
    blue[crop_mask] = np.random.uniform(0.06, 0.10, np.sum(crop_mask))
    green[crop_mask] = np.random.uniform(0.25, 0.38, np.sum(crop_mask))
    red[crop_mask] = np.random.uniform(0.08, 0.15, np.sum(crop_mask))
    nir[crop_mask] = np.random.uniform(0.55, 0.75, np.sum(crop_mask))

    # 4. Add Urban Cluster & Road Network (Bottom-Right)
    urban_mask = (x_grid > width * 0.6) & (y_grid > height * 0.6) & (~river_mask) & (~forest_mask)
    blue[urban_mask] = np.random.uniform(0.35, 0.55, np.sum(urban_mask))
    green[urban_mask] = np.random.uniform(0.38, 0.58, np.sum(urban_mask))
    red[urban_mask] = np.random.uniform(0.40, 0.60, np.sum(urban_mask))
    nir[urban_mask] = np.random.uniform(0.30, 0.45, np.sum(urban_mask))

    # Stack into 4-band array [C, H, W]
    scene_arr = np.stack([blue, green, red, nir], axis=0).astype(np.float32)

    # Scale to surface reflectance integer (0 - 10000) or float
    scene_reflectance = (scene_arr * 10000.0).astype(np.uint16)

    # Affine geotransform
    left, bottom, right, top = bounds
    transform = from_bounds(left, bottom, right, top, width, height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "uint16",
        "crs": CRS.from_user_input(crs),
        "transform": transform,
        "compress": "deflate",
        "nodata": 0,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for i in range(4):
            dst.write(scene_reflectance[i], i + 1)
        dst.set_band_description(1, "B02_Blue")
        dst.set_band_description(2, "B03_Green")
        dst.set_band_description(3, "B04_Red")
        dst.set_band_description(4, "B08_NIR")

    return output_path
