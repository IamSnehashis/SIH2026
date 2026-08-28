"""
End-to-End Integration Test for GIS Super-Resolution Pipeline.
"""
import pytest
from pathlib import Path
import rasterio

from src.gis.pipeline import GISPipeline
from src.gis.sample_data import generate_synthetic_satellite_scene


def test_full_gis_pipeline(tmp_path):
    # 1. Generate synthetic input scene
    input_scene = tmp_path / "raw_scene.tif"
    generate_synthetic_satellite_scene(input_scene, height=256, width=256)

    # 2. Run full pipeline
    output_sr = tmp_path / "outputs" / "sr_scene.tif"
    pipeline = GISPipeline(tile_size=128, overlap=16, scale_factor=2.0, blend_method="cosine")
    
    results = pipeline.run_full_scene(
        input_scene_path=input_scene,
        output_sr_path=output_sr,
        generate_indices=True,
        generate_vectors=True,
    )

    assert output_sr.exists()
    assert "stages" in results
    assert results["stages"]["preprocessing"]["num_tiles"] > 0
    assert results["stages"]["inference"]["output_shape"] == [4, 512, 512]

    # Verify georeferenced output GeoTIFF
    with rasterio.open(output_sr) as src:
        assert src.width == 512
        assert src.height == 512
        assert src.count == 4
        assert src.crs is not None

    # Check generated downstream product folders
    products_dir = tmp_path / "outputs" / "sr_scene_products"
    assert (products_dir / "normalization_manifest.json").exists()
    assert (products_dir / "tiling_manifest.json").exists()
    assert (products_dir / "indices" / "ndvi.tif").exists()
    assert (products_dir / "vectors" / "water_bodies.geojson").exists()
