"""
Unified GIS Command Line Interface (CLI) for Satellite Super-Resolution Mapping.

Usage:
  python -m src.gis.cli sample --output data/raw/sample_scene.tif
  python -m src.gis.cli preprocess --input data/raw --output data/tiles --scale 4
  python -m src.gis.cli pipeline --input data/raw/sample_scene.tif --output outputs/sr_result.tif --scale 4 --indices --vectorize
  python -m src.gis.cli indices --input outputs/sr_result.tif --output outputs/indices
  python -m src.gis.cli vectorize --input outputs/sr_result.tif --output outputs/vectors
  python -m src.gis.cli evaluate --pred outputs/sr_result.tif --target data/raw/hr_ground_truth.tif --output outputs/eval
"""
import argparse
import sys
from pathlib import Path

from src.gis.sample_data import generate_synthetic_satellite_scene
from src.gis.pair_generator import generate_dataset_pairs_from_scenes
from src.gis.pipeline import GISPipeline
from src.gis.indices import compute_spectral_indices, export_index_geotiff_and_png
from src.gis.vectorizer import extract_features_from_indices
from src.gis.metrics import evaluate_super_resolution, generate_error_heatmap
from src.gis.raster_utils import read_raster, get_raster_metadata


def cmd_sample(args):
    print(f"Generating synthetic 4-band satellite scene at: {args.output}")
    out = generate_synthetic_satellite_scene(
        output_path=args.output,
        height=args.height,
        width=args.width,
        crs=args.crs,
    )
    print(f"[OK] Created synthetic satellite GeoTIFF ({args.width}x{args.height}) -> {out}")


def cmd_preprocess(args):
    print(f"Starting GIS pre-processing & dataset pairing from: {args.input}")
    res = generate_dataset_pairs_from_scenes(
        raw_dir=args.input,
        output_tiles_dir=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        scale_factor=args.scale,
        save_geotiffs=args.save_tifs,
    )
    print(f"[OK] Pre-processed {res['num_scenes']} scenes into {res['num_pairs']} paired tiles.")
    print(f"     Manifest saved to: {res.get('manifest_path', args.output)}")


def cmd_pipeline(args):
    print(f"Running full GIS Super-Resolution Pipeline on: {args.input}")
    pipe = GISPipeline(
        tile_size=args.tile_size,
        overlap=args.overlap,
        scale_factor=args.scale,
        blend_method=args.blend,
    )
    res = pipe.run_full_scene(
        input_scene_path=args.input,
        output_sr_path=args.output,
        weights_path=args.weights,
        generate_indices=args.indices,
        generate_vectors=args.vectorize,
        ground_truth_hr_path=args.ground_truth,
    )
    print(f"\n=======================================================")
    print(f"[SUCCESS] Pipeline Complete! Total time: {res['total_elapsed_sec']}s")
    print(f"Output GeoTIFF: {res['output_sr_path']}")
    print(f"Model: {res['stages']['inference']['model_type']}")
    if "metrics" in res:
        print("\n--- Remote Sensing Quality Metrics ---")
        for k, v in res["metrics"].items():
            print(f"  {k}: {v}")
    if "extracted_vectors" in res:
        print("\n--- Extracted Vector Layers ---")
        for k, v in res["extracted_vectors"].items():
            print(f"  {k}: {v} polygons extracted")
    print(f"=======================================================\n")


def cmd_indices(args):
    print(f"Calculating spectral indices for: {args.input}")
    arr, profile, _ = read_raster(args.input)
    indices = compute_spectral_indices(arr)
    meta = get_raster_metadata(args.input)
    for name, arr_idx in indices.items():
        cmap = "RdYlGn" if "veg" in name or "ndvi" in name or "evi" in name else "Blues" if "water" in name or "ndwi" in name else "YlOrRd"
        export_index_geotiff_and_png(arr_idx, meta, args.output, index_name=name, colormap_name=cmap)
    print(f"[OK] Generated {list(indices.keys())} in {args.output}")


def cmd_vectorize(args):
    print(f"Extracting vector features from: {args.input}")
    arr, profile, _ = read_raster(args.input)
    indices = compute_spectral_indices(arr)
    meta = get_raster_metadata(args.input)
    vectors = extract_features_from_indices(
        indices, transform=meta["transform"], crs=meta["crs"], output_dir=args.output
    )
    print(f"[OK] Vector layers saved to {args.output}: {list(vectors.keys())}")


def cmd_evaluate(args):
    print(f"Evaluating SR output ({args.pred}) against Ground Truth HR ({args.target})")
    p_arr, _, _ = read_raster(args.pred)
    t_arr, t_prof, _ = read_raster(args.target)
    
    # Crop to common size
    min_h = min(p_arr.shape[1], t_arr.shape[1])
    min_w = min(p_arr.shape[2], t_arr.shape[2])
    p_crop = p_arr[:, :min_h, :min_w]
    t_crop = t_arr[:, :min_h, :min_w]

    metrics = evaluate_super_resolution(p_crop, t_crop, scale_factor=args.scale)
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    if args.output:
        generate_error_heatmap(p_crop, t_crop, t_prof, args.output)
        print(f"[OK] Error heatmap saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src.gis.cli",
        description="GIS Subsystem CLI for Satellite Super-Resolution Mapping (SRM)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Sample generator
    p_sample = subparsers.add_parser("sample", help="Generate synthetic satellite GeoTIFF scene")
    p_sample.add_argument("--output", "-o", default="data/raw/sample_scene.tif", help="Output GeoTIFF path")
    p_sample.add_argument("--height", type=int, default=512)
    p_sample.add_argument("--width", type=int, default=512)
    p_sample.add_argument("--crs", default="EPSG:32643")

    # Preprocess
    p_pre = subparsers.add_parser("preprocess", help="Preprocess scenes and create tile training pairs")
    p_pre.add_argument("--input", "-i", default="data/raw", help="Directory of raw satellite GeoTIFF scenes")
    p_pre.add_argument("--output", "-o", default="data/tiles", help="Directory for generated tile pairs")
    p_pre.add_argument("--tile-size", type=int, default=128)
    p_pre.add_argument("--overlap", type=int, default=16)
    p_pre.add_argument("--scale", type=int, default=4)
    p_pre.add_argument("--save-tifs", action="store_true", help="Also save individual GeoTIFF tiles")

    # Pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run end-to-end full scene super-resolution")
    p_pipe.add_argument("--input", "-i", required=True, help="Input medium-resolution GeoTIFF")
    p_pipe.add_argument("--output", "-o", default="outputs/sr_mosaic.tif", help="Output super-resolved GeoTIFF")
    p_pipe.add_argument("--weights", "-w", default=None, help="Trained DL model checkpoint (.pth)")
    p_pipe.add_argument("--scale", type=float, default=4.0)
    p_pipe.add_argument("--tile-size", type=int, default=128)
    p_pipe.add_argument("--overlap", type=int, default=16)
    p_pipe.add_argument("--blend", choices=["cosine", "linear", "gaussian", "none"], default="cosine")
    p_pipe.add_argument("--indices", action="store_true", default=True, help="Calculate NDVI/NDWI indices")
    p_pipe.add_argument("--vectorize", action="store_true", default=True, help="Extract GeoJSON vector features")
    p_pipe.add_argument("--ground-truth", "-gt", default=None, help="Ground truth HR GeoTIFF for metric evaluation")

    # Indices
    p_idx = subparsers.add_parser("indices", help="Compute spectral vegetation, water & urban indices")
    p_idx.add_argument("--input", "-i", required=True)
    p_idx.add_argument("--output", "-o", default="outputs/indices")

    # Vectorize
    p_vec = subparsers.add_parser("vectorize", help="Extract vector polygons from indices")
    p_vec.add_argument("--input", "-i", required=True)
    p_vec.add_argument("--output", "-o", default="outputs/vectors")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Compute SAM, ERGAS, PSNR, SSIM metrics")
    p_eval.add_argument("--pred", "-p", required=True)
    p_eval.add_argument("--target", "-t", required=True)
    p_eval.add_argument("--scale", type=float, default=4.0)
    p_eval.add_argument("--output", "-o", default="outputs/evaluation")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "sample": cmd_sample,
        "preprocess": cmd_preprocess,
        "pipeline": cmd_pipeline,
        "indices": cmd_indices,
        "vectorize": cmd_vectorize,
        "evaluate": cmd_evaluate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
