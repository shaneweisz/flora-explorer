"""
Main pipeline for finding candidate locations.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from .gbif import get_species_info, fetch_occurrences
from .embeddings import EmbeddingMosaic
from .methods import ClassifierMethod

logger = logging.getLogger(__name__)


# Predefined regions
REGIONS = {
    "cambridge": {
        "bbox": (0.03, 52.13, 0.22, 52.29),
        "description": "Cambridge, UK test region",
    },
}

# Default ratio of background samples to occurrences
NEGATIVE_RATIO = 5


@dataclass
class PredictionResult:
    """Container for prediction results."""

    species_name: str
    taxon_key: int
    n_occurrences: int
    n_background: int
    scores: np.ndarray  # (H, W) probability map
    transform: rasterio.transform.Affine
    bbox: tuple[float, float, float, float]

    def to_geojson(
        self,
        threshold: float = 0.5,
        max_points: int = 5000
    ) -> dict:
        """Convert high-scoring pixels to GeoJSON."""
        rows, cols = np.where(self.scores >= threshold)

        # Subsample if too many points
        if len(rows) > max_points:
            idx = np.random.choice(len(rows), max_points, replace=False)
            rows, cols = rows[idx], cols[idx]

        features = []
        for row, col in zip(rows, cols):
            lon, lat = rasterio.transform.xy(self.transform, row, col)
            features.append({
                "type": "Feature",
                "properties": {"probability": float(self.scores[row, col])},
                "geometry": {"type": "Point", "coordinates": [lon, lat]}
            })

        # Sort by probability (ascending, so high values rendered on top)
        features.sort(key=lambda f: f["properties"]["probability"])

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "species": self.species_name,
                "taxon_key": self.taxon_key,
                "n_occurrences": self.n_occurrences,
                "n_candidates": len(features),
                "threshold": threshold,
                "bbox": list(self.bbox),
            }
        }

    def save(self, output_dir: Path, threshold: float = 0.5) -> dict[str, Path]:
        """Save results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save probability raster as GeoTIFF
        tiff_path = output_dir / "probability.tif"
        with rasterio.open(
            tiff_path, "w",
            driver="GTiff",
            height=self.scores.shape[0],
            width=self.scores.shape[1],
            count=1,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=self.transform,
        ) as dst:
            dst.write(self.scores, 1)
        paths["raster"] = tiff_path
        logger.info(f"Saved probability raster: {tiff_path}")

        # Save candidates as GeoJSON
        geojson = self.to_geojson(threshold=threshold)
        geojson_path = output_dir / "candidates.geojson"
        with open(geojson_path, "w") as f:
            json.dump(geojson, f)
        paths["candidates"] = geojson_path
        logger.info(f"Saved {len(geojson['features'])} candidates: {geojson_path}")

        return paths


def sample_background(
    mosaic: EmbeddingMosaic,
    n_samples: int,
    exclude_coords: list[tuple[float, float]],
    seed: int = 42,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Sample random background points from the mosaic.

    Args:
        mosaic: Loaded embedding mosaic
        n_samples: Number of background samples to generate
        exclude_coords: Coordinates to exclude (occurrence locations)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (embeddings array, coordinates list)
    """
    rng = np.random.default_rng(seed)
    h, w, _ = mosaic.shape

    # Get pixel indices of exclusions
    exclude_pixels = set()
    for lon, lat in exclude_coords:
        row, col = mosaic.coords_to_pixel(lon, lat)
        exclude_pixels.add((row, col))

    coords = []
    embeddings = []
    attempts = 0
    max_attempts = n_samples * 20

    while len(coords) < n_samples and attempts < max_attempts:
        row = rng.integers(0, h)
        col = rng.integers(0, w)
        if (row, col) not in exclude_pixels:
            emb = mosaic.mosaic[row, col, :]
            if not np.allclose(emb, 0):  # Skip empty pixels
                lon, lat = mosaic.pixel_to_coords(row, col)
                coords.append((lon, lat))
                embeddings.append(emb)
                exclude_pixels.add((row, col))
        attempts += 1

    return np.array(embeddings), coords


def find_candidates(
    species_name: str,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    output_dir: Optional[Path] = None,
    negative_ratio: int = NEGATIVE_RATIO,
) -> PredictionResult:
    """
    Find candidate locations for a species using a classifier.

    Trains a logistic regression classifier on occurrence embeddings vs
    random background embeddings, then scores all pixels.

    Args:
        species_name: Scientific name of the species
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        cache_dir: Directory containing Tessera embeddings
        output_dir: If provided, save results to this directory
        negative_ratio: Ratio of background samples to occurrences

    Returns:
        PredictionResult with probability scores and metadata
    """
    logger.info("=" * 60)
    logger.info(f"Finding candidates for: {species_name}")
    logger.info("=" * 60)

    # 1. Get species info and occurrences
    logger.info("\n[1/5] Fetching GBIF data...")
    species_info = get_species_info(species_name)
    taxon_key = species_info["taxon_key"]
    logger.info(f"  Matched: {species_info['scientific_name']} (key: {taxon_key})")

    occurrences = fetch_occurrences(taxon_key, bbox)
    n_occurrences = len(occurrences)
    logger.info(f"  Found {n_occurrences} occurrences in region")

    if n_occurrences < 2:
        raise ValueError(f"Need at least 2 occurrences, found {n_occurrences}")

    # 2. Load embedding mosaic
    logger.info("\n[2/5] Loading embedding mosaic...")
    mosaic = EmbeddingMosaic(cache_dir, bbox)
    mosaic.load()
    h, w, c = mosaic.shape
    logger.info(f"  Mosaic shape: {h} x {w} x {c}")

    # 3. Sample embeddings at occurrence locations
    logger.info("\n[3/5] Sampling occurrence embeddings...")
    positive_embeddings, valid_coords = mosaic.sample_at_coords(occurrences)
    logger.info(f"  Valid occurrence samples: {len(positive_embeddings)}")

    if len(positive_embeddings) < 2:
        raise ValueError("Need at least 2 valid embeddings at occurrence locations")

    # 4. Sample background embeddings
    logger.info("\n[4/5] Sampling background embeddings...")
    n_background = len(positive_embeddings) * negative_ratio
    negative_embeddings, neg_coords = sample_background(
        mosaic, n_background, valid_coords
    )
    logger.info(f"  Background samples: {len(negative_embeddings)}")

    # 5. Train classifier and predict
    logger.info("\n[5/5] Training classifier and predicting...")
    classifier = ClassifierMethod()
    classifier.fit(positive_embeddings, negative_embeddings)

    all_embeddings = mosaic.get_all_embeddings()
    scores = classifier.predict(all_embeddings)
    scores_map = scores.reshape(h, w)

    # Log statistics
    logger.info(f"\n  Score range: {scores.min():.3f} - {scores.max():.3f}")
    high_score = (scores > 0.5).sum()
    logger.info(f"  High probability pixels (>0.5): {high_score:,} ({100*high_score/len(scores):.1f}%)")

    # Create result
    result = PredictionResult(
        species_name=species_info["canonical_name"],
        taxon_key=taxon_key,
        n_occurrences=len(valid_coords),
        n_background=len(negative_embeddings),
        scores=scores_map,
        transform=mosaic.transform,
        bbox=bbox,
    )

    # Save if output directory specified
    if output_dir:
        result.save(output_dir, threshold=0.5)

        # Also save occurrences
        occ_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "Point", "coordinates": [lon, lat]}
                }
                for lon, lat in valid_coords
            ]
        }
        occ_path = output_dir / "occurrences.geojson"
        with open(occ_path, "w") as f:
            json.dump(occ_geojson, f, indent=2)
        logger.info(f"Saved occurrences: {occ_path}")

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)

    return result
