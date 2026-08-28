"""
Cloud-Optimized GeoTIFF (COG) & Image Pyramid Generator.

Creates web-streaming-ready GeoTIFFs with internal tiling, overviews,
and optimized headers for GIS map viewer performance and QGIS/ArcGIS fast rendering.
"""
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import rasterio
from rasterio.enums import Resampling


def build_pyramid_overviews(
    geotiff_path: Union[str, Path],
    overview_factors: Optional[List[int]] = None,
    resampling: Resampling = Resampling.average,
) -> Path:
    """
    Generate internal image pyramids/overviews for fast zooming and web map rendering.
    """
    geotiff_path = Path(geotiff_path)
    if not geotiff_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

    if overview_factors is None:
        overview_factors = [2, 4, 8, 16]

    with rasterio.open(geotiff_path, "r+") as dst:
        dst.build_overviews(overview_factors, resampling)
        dst.update_tags(ns="rio_overview", resampling=resampling.name)

    return geotiff_path


def export_as_cog(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    compress: str = "deflate",
    blocksize: int = 256,
) -> Path:
    """
    Convert a standard GeoTIFF into a tiled, compressed, overview-enabled GeoTIFF.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update({
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": min(blocksize, src.width),
            "blockysize": min(blocksize, src.height),
            "compress": compress,
            "interleave": "pixel",
        })

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
            
            # Build overviews directly
            if src.width >= 256 or src.height >= 256:
                factors = [2, 4, 8]
                dst.build_overviews(factors, Resampling.average)

    return dst_path
