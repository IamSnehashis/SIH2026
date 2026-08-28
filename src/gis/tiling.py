"""
Geospatial Tiling & Sliding Window Grid Generator.

Splits arbitrary-sized large satellite scenes into fixed-size tiles with
configurable overlap/stride, captures exact geotransforms for every patch,
filters out low-information/nodata patches, and produces GeoJSON tile catalogues.
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List, Generator
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

from src.gis.raster_utils import read_raster, get_raster_metadata


class TileMetadata:
    def __init__(
        self,
        tile_id: int,
        scene_name: str,
        x: int,
        y: int,
        width: int,
        height: int,
        transform: Affine,
        crs: str,
        bounds: Tuple[float, float, float, float],
        valid_pixel_ratio: float = 1.0,
    ):
        self.tile_id = tile_id
        self.scene_name = scene_name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.transform = transform
        self.crs = crs
        self.bounds = bounds  # (left, bottom, right, top)
        self.valid_pixel_ratio = valid_pixel_ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "scene_name": self.scene_name,
            "grid_pos": {"x": self.x, "y": self.y},
            "shape": {"height": self.height, "width": self.width},
            "transform": [self.transform.a, self.transform.b, self.transform.c,
                          self.transform.d, self.transform.e, self.transform.f],
            "crs": self.crs,
            "bounds": {
                "left": self.bounds[0],
                "bottom": self.bounds[1],
                "right": self.bounds[2],
                "top": self.bounds[3],
            },
            "valid_pixel_ratio": round(self.valid_pixel_ratio, 4),
        }

    def to_geojson_feature(self) -> Dict[str, Any]:
        left, bottom, right, top = self.bounds
        return {
            "type": "Feature",
            "properties": {
                "tile_id": self.tile_id,
                "scene_name": self.scene_name,
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
                "valid_pixel_ratio": self.valid_pixel_ratio,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                    [left, top],
                ]]
            }
        }


class SceneTiler:
    """
    Manages sliding-window grid generation and patch extraction over large satellite scenes.
    """

    def __init__(
        self,
        tile_size: int = 128,
        overlap: int = 16,
        min_valid_ratio: float = 0.5,
        pad_edges: bool = True,
    ):
        """
        Args:
            tile_size: Height and width of square patch in pixels.
            overlap: Overlap in pixels between adjacent tiles (stride = tile_size - overlap).
            min_valid_ratio: Discard tiles with valid data ratio below this threshold (0.0 to 1.0).
            pad_edges: If True, pads or adjusts final step to cover full scene edges.
        """
        if overlap >= tile_size:
            raise ValueError(f"Overlap ({overlap}) must be less than tile_size ({tile_size})")

        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.min_valid_ratio = min_valid_ratio
        self.pad_edges = pad_edges

    def compute_grid(self, height: int, width: int) -> List[Tuple[int, int]]:
        """
        Calculate top-left (y, x) coordinates for all sliding window patches.
        """
        y_coords = list(range(0, max(1, height - self.tile_size + 1), self.stride))
        if self.pad_edges and y_coords and (y_coords[-1] + self.tile_size < height):
            y_coords.append(height - self.tile_size)

        x_coords = list(range(0, max(1, width - self.tile_size + 1), self.stride))
        if self.pad_edges and x_coords and (x_coords[-1] + self.tile_size < width):
            x_coords.append(width - self.tile_size)

        # Handle scenes smaller than tile_size
        if height < self.tile_size:
            y_coords = [0]
        if width < self.tile_size:
            x_coords = [0]

        grid = []
        for y in y_coords:
            for x in x_coords:
                grid.append((y, x))
        return grid

    def extract_tiles_from_array(
        self,
        arr: np.ndarray,
        reference_profile: Dict[str, Any],
        scene_name: str = "scene",
        mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, TileMetadata]]:
        """
        Extract all tiles from an in-memory [C, H, W] array.
        """
        c, h, w = arr.shape
        grid = self.compute_grid(h, w)
        orig_transform = reference_profile["transform"]
        crs_str = reference_profile.get("crs", "Unspecified")
        if hasattr(crs_str, "to_string"):
            crs_str = crs_str.to_string()
        else:
            crs_str = str(crs_str)

        tiles: List[Tuple[np.ndarray, TileMetadata]] = []
        tile_idx = 0

        for y, x in grid:
            # Handle potential edge padding if scene is smaller than tile_size
            h_slice = min(self.tile_size, h - y)
            w_slice = min(self.tile_size, w - x)

            patch = arr[:, y:y + h_slice, x:x + w_slice]

            # Pad with reflect or edge if smaller than tile_size
            if h_slice < self.tile_size or w_slice < self.tile_size:
                pad_h = self.tile_size - h_slice
                pad_w = self.tile_size - w_slice
                patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

            # Check valid pixel ratio
            valid_ratio = 1.0
            if mask is not None:
                patch_mask = mask[y:y + h_slice, x:x + w_slice]
                valid_ratio = float(np.mean(patch_mask))
                if valid_ratio < self.min_valid_ratio:
                    continue

            # Compute tile transform & geographic bounding box
            tile_transform = rasterio.windows.transform(
                Window(x, y, self.tile_size, self.tile_size),
                orig_transform
            )
            left = tile_transform.c
            top = tile_transform.f
            right = left + (tile_transform.a * self.tile_size)
            bottom = top + (tile_transform.e * self.tile_size)
            bounds = (min(left, right), min(bottom, top), max(left, right), max(bottom, top))

            meta = TileMetadata(
                tile_id=tile_idx,
                scene_name=scene_name,
                x=x,
                y=y,
                width=self.tile_size,
                height=self.tile_size,
                transform=tile_transform,
                crs=crs_str,
                bounds=bounds,
                valid_pixel_ratio=valid_ratio,
            )
            tiles.append((patch, meta))
            tile_idx += 1

        return tiles

    def save_tile_catalogue(
        self,
        tile_metas: List[TileMetadata],
        output_dir: Union[str, Path],
        basename: str = "tiles",
    ):
        """
        Export tile index metadata as both JSON and GeoJSON.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_data = [tm.to_dict() for tm in tile_metas]
        with open(output_dir / f"{basename}_manifest.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        geojson_data = {
            "type": "FeatureCollection",
            "features": [tm.to_geojson_feature() for tm in tile_metas]
        }
        with open(output_dir / f"{basename}_footprints.geojson", "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)
