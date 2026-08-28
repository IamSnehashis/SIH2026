"""
Geospatial Vectorization & Feature Extraction Engine.

Converts raster segmentation / index masks (water bodies, urban extent, high-biomass canopy)
into vector polygon layers (GeoJSON and Shapefile) with geographic area statistics.
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import json
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union

from src.gis.raster_utils import get_raster_metadata


def vectorize_binary_mask(
    mask: np.ndarray,
    transform: rasterio.Affine,
    crs: str = "EPSG:4326",
    min_area_pixels: int = 10,
    feature_name: str = "feature",
) -> Dict[str, Any]:
    """
    Extract vector polygons from a 2D boolean or binary integer mask.
    
    Args:
        mask: 2D NumPy array [H, W] of boolean or binary (0 and 1) values.
        transform: Affine transform for georeferencing polygon vertices.
        crs: Coordinate Reference System string (e.g. 'EPSG:4326' or 'EPSG:32643').
        min_area_pixels: Minimum polygon pixel count to filter out noisy speckles.
        feature_name: Label for the extracted vector class.

    Returns:
        GeoJSON FeatureCollection dict with polygon geometries and area properties.
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    # Extract polygon generator from rasterio
    generator = shapes(mask_uint8, mask=(mask_uint8 == 1), transform=transform)

    features = []
    total_pixel_count = 0

    # Approximate pixel area in CRS units
    pixel_area = abs(transform.a * transform.e)

    for geom, value in generator:
        if value != 1:
            continue

        poly = shape(geom)
        if not poly.is_valid:
            poly = poly.buffer(0)

        area_units = poly.area
        pixel_count = int(round(area_units / (pixel_area + 1e-9)))

        if pixel_count < min_area_pixels:
            continue

        total_pixel_count += pixel_count

        # Compute metric area approximations
        # If CRS is geographic (degrees), area_units is in deg^2. Approx conversion: 1 deg ~ 111,320m
        is_degree = "4326" in crs or "WGS 84" in crs
        if is_degree:
            # Approximate conversion at latitude center
            lat_center = (poly.bounds[1] + poly.bounds[3]) / 2.0
            lat_m = 111320.0
            lon_m = 111320.0 * np.cos(np.radians(lat_center))
            area_m2 = area_units * lat_m * lon_m
        else:
            area_m2 = area_units  # Projected CRS in meters

        area_ha = area_m2 / 10000.0
        area_km2 = area_m2 / 1_000_000.0

        feat = {
            "type": "Feature",
            "properties": {
                "class": feature_name,
                "pixel_count": pixel_count,
                "area_m2": round(area_m2, 2),
                "area_hectares": round(area_ha, 4),
                "area_km2": round(area_km2, 6),
            },
            "geometry": mapping(poly),
        }
        features.append(feat)

    geojson_doc = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": crs}
        },
        "properties": {
            "class": feature_name,
            "total_features": len(features),
            "total_pixels": total_pixel_count,
        },
        "features": features,
    }
    return geojson_doc


def extract_features_from_indices(
    indices: Dict[str, np.ndarray],
    transform: rasterio.Affine,
    crs: str = "EPSG:4326",
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Extract segmented water bodies (NDWI > 0.0), dense vegetation (NDVI > 0.4),
    and built-up zones (NDBI > 0.0) as GeoJSON vector layers.
    """
    extracted = {}

    if "ndwi" in indices:
        water_mask = indices["ndwi"] > 0.05
        extracted["water_bodies"] = vectorize_binary_mask(
            water_mask, transform, crs, min_area_pixels=15, feature_name="water_body"
        )

    if "ndvi" in indices:
        veg_mask = indices["ndvi"] > 0.4
        extracted["vegetation"] = vectorize_binary_mask(
            veg_mask, transform, crs, min_area_pixels=20, feature_name="dense_vegetation"
        )

    if "ndbi" in indices:
        urban_mask = indices["ndbi"] > 0.05
        extracted["urban_builtup"] = vectorize_binary_mask(
            urban_mask, transform, crs, min_area_pixels=20, feature_name="builtup_area"
        )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, geojson_data in extracted.items():
            out_file = output_dir / f"{name}.geojson"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(geojson_data, f, indent=2)

    return extracted
