#!/usr/bin/env python3
"""
Predict habitat suitability for a species in a local area around a point.

Uses pre-trained classifier models for fast predictions.
Only loads the single tile containing the point (not the full mosaic).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from finder.methods import ClassifierMethod

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"

YEAR = 2024
TILE_SIZE = 0.1  # degrees


def meters_to_degrees(meters: float, lat: float) -> tuple[float, float]:
    """Convert meters to approximate degrees at a given latitude."""
    lat_deg = meters / 111000
    lon_deg = meters / (111000 * np.cos(np.radians(lat)))
    return lon_deg, lat_deg


def get_tile_coords(lon: float, lat: float) -> tuple[float, float]:
    """Get the tile center coordinates for a given point."""
    # Tiles are named by their center, offset by half step
    # Formula from EmbeddingMosaic: floor((coord + half_step) / step) * step - half_step
    half_step = TILE_SIZE / 2
    tile_lon = np.floor((lon + half_step) / TILE_SIZE) * TILE_SIZE - half_step
    tile_lat = np.floor((lat + half_step) / TILE_SIZE) * TILE_SIZE - half_step
    return round(tile_lon, 2), round(tile_lat, 2)


def load_single_tile(tile_lon: float, tile_lat: float) -> tuple[np.ndarray, rasterio.Affine] | None:
    """Load a single embedding tile. Returns (embeddings, transform) or None."""
    tile_dir = CACHE_DIR / str(YEAR)
    name = f"grid_{tile_lon:.2f}_{tile_lat:.2f}"
    npy_path = tile_dir / name / f"{name}.npy"
    scales_path = tile_dir / name / f"{name}_scales.npy"

    if not npy_path.exists() or not scales_path.exists():
        return None

    # Load and dequantize
    data = np.load(npy_path).astype(np.float32)
    scales = np.load(scales_path)
    embeddings = data * scales[:, :, np.newaxis]

    # Create transform for this tile
    h, w = embeddings.shape[:2]
    transform = rasterio.transform.from_bounds(
        tile_lon, tile_lat, tile_lon + TILE_SIZE, tile_lat + TILE_SIZE, w, h
    )

    return embeddings, transform


def predict_local(
    lat: float,
    lon: float,
    species_key: int,
    grid_size_m: int = 100,
) -> dict:
    """
    Get predictions for a grid around a point using pre-trained model.
    Only loads the single tile containing the point.
    """
    # Load pre-trained classifier
    model_path = MODELS_DIR / f"{species_key}.pkl"
    if not model_path.exists():
        raise ValueError(f"No trained model for species key {species_key}. Run train_models.py first.")

    classifier = ClassifierMethod.load(model_path)

    # Find and load only the tile containing this point
    tile_lon, tile_lat = get_tile_coords(lon, lat)
    tile_data = load_single_tile(tile_lon, tile_lat)

    if tile_data is None:
        return {
            "predictions": [],
            "species_key": species_key,
            "center": {"lon": lon, "lat": lat},
            "grid_size_m": grid_size_m,
            "n_pixels": 0,
            "error": f"No tile data at {tile_lon}, {tile_lat}",
        }

    embeddings, transform = tile_data
    h, w, _ = embeddings.shape

    # Calculate grid bounds in degrees
    lon_offset, lat_offset = meters_to_degrees(grid_size_m / 2, lat)
    min_lon = lon - lon_offset
    max_lon = lon + lon_offset
    min_lat = lat - lat_offset
    max_lat = lat + lat_offset

    # Convert corner coordinates to pixel indices within this tile
    min_row, min_col = rasterio.transform.rowcol(transform, min_lon, max_lat)
    max_row, max_col = rasterio.transform.rowcol(transform, max_lon, min_lat)

    # Clamp to valid range
    min_row = max(0, min(min_row, h - 1))
    max_row = max(0, min(max_row, h - 1))
    min_col = max(0, min(min_col, w - 1))
    max_col = max(0, min(max_col, w - 1))

    # Collect embeddings and coordinates
    embeddings_to_predict = []
    coords_to_predict = []

    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            emb = embeddings[row, col, :]
            if not np.allclose(emb, 0):
                px_lon, px_lat = rasterio.transform.xy(transform, row, col)
                embeddings_to_predict.append(emb)
                coords_to_predict.append((px_lon, px_lat))

    # Batch predict
    predictions = []
    if embeddings_to_predict:
        embeddings_array = np.array(embeddings_to_predict)
        scores = classifier.predict(embeddings_array)

        for (px_lon, px_lat), score in zip(coords_to_predict, scores):
            predictions.append({
                "lon": float(px_lon),
                "lat": float(px_lat),
                "score": float(score),
            })

    return {
        "predictions": predictions,
        "species_key": species_key,
        "center": {"lon": lon, "lat": lat},
        "grid_size_m": grid_size_m,
        "n_pixels": len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict local habitat suitability")
    parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    parser.add_argument("--species-key", type=int, required=True, help="GBIF species key")
    parser.add_argument("--grid-size", type=int, default=100, help="Grid size in meters")

    args = parser.parse_args()

    try:
        result = predict_local(
            lat=args.lat,
            lon=args.lon,
            species_key=args.species_key,
            grid_size_m=args.grid_size,
        )
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
