"""
Unit tests for GIS Pre-Processing & Tiling.
"""
import pytest
import numpy as np
from pathlib import Path
import rasterio

from src.gis.raster_utils import read_raster, write_raster, get_raster_metadata
from src.gis.preprocessor import Normalizer, preprocess_scene
from src.gis.tiling import SceneTiler
from src.gis.pair_generator import apply_sensor_degradation, generate_dataset_pairs_from_scenes
from src.gis.sample_data import generate_synthetic_satellite_scene


@pytest.fixture
def sample_scene_tif(tmp_path):
    scene_path = tmp_path / "test_scene.tif"
    generate_synthetic_satellite_scene(scene_path, height=256, width=256)
    return scene_path


def test_raster_io_and_metadata(sample_scene_tif):
    meta = get_raster_metadata(sample_scene_tif)
    assert meta["width"] == 256
    assert meta["height"] == 256
    assert meta["count"] == 4
    assert "crs" in meta

    arr, profile, mask = read_raster(sample_scene_tif, return_mask=True)
    assert arr.shape == (4, 256, 256)
    assert mask is not None
    assert mask.shape == (256, 256)


def test_normalizer_percentile_and_inverse():
    arr = np.random.uniform(100, 9000, size=(4, 64, 64)).astype(np.float32)
    norm = Normalizer(method="percentile", p_low=2, p_high=98, target_range=(0.0, 1.0))
    norm_arr = norm.fit_transform(arr)
    assert norm_arr.min() >= 0.0
    assert norm_arr.max() <= 1.0

    # Inverse transform
    inv = norm.inverse_transform(norm_arr)
    assert inv.shape == arr.shape
    
    # Check that reconstructed values accurately follow original values
    diff = np.abs(arr - inv)
    assert np.mean(diff) < 500.0


def test_normalizer_manifest(tmp_path):
    arr = np.random.uniform(50, 5000, size=(3, 32, 32)).astype(np.float32)
    norm = Normalizer(method="minmax")
    norm.fit_transform(arr)
    manifest_path = tmp_path / "norm_manifest.json"
    norm.export_manifest(manifest_path)
    assert manifest_path.exists()

    loaded_norm = Normalizer.from_manifest(manifest_path)
    assert loaded_norm.method == "minmax"
    assert len(loaded_norm.stats) == 3


def test_scene_tiler_grid(sample_scene_tif):
    arr, profile, mask = read_raster(sample_scene_tif, return_mask=True)
    tiler = SceneTiler(tile_size=128, overlap=16, min_valid_ratio=0.1)
    tiles = tiler.extract_tiles_from_array(arr, profile, scene_name="test_scene", mask=mask)
    
    assert len(tiles) > 0
    patch, meta = tiles[0]
    assert patch.shape == (4, 128, 128)
    assert meta.width == 128
    assert meta.height == 128
    assert len(meta.bounds) == 4


def test_sensor_degradation():
    hr = np.random.uniform(0.0, 1.0, size=(4, 128, 128)).astype(np.float32)
    native_lr, model_lr = apply_sensor_degradation(hr, scale_factor=4, pre_upsample=True)
    assert native_lr.shape == (4, 32, 32)
    assert model_lr.shape == (4, 128, 128)
