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
from .methods import SimilarityMethod

logger = logging.getLogger(__name__)


# Predefined regions
REGIONS = {
    "cambridge": {
        "bbox": (0.03, 52.13, 0.22, 52.29),
        "description": "Cambridge, UK test region",
    },
}


@dataclass
class PredictionResult:
    """Container for prediction results."""

    species_name: str
    taxon_key: int
    n_occurrences: int
    scores: np.ndarray  # (H, W) similarity map
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


def find_candidates(
    species_name: str,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    output_dir: Optional[Path] = None,
) -> PredictionResult:
    """
    Find candidate locations for a species using habitat similarity.

    Args:
        species_name: Scientific name of the species
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        cache_dir: Directory containing Tessera embeddings
        output_dir: If provided, save results to this directory

    Returns:
        PredictionResult with similarity scores and metadata
    """
    logger.info("=" * 60)
    logger.info(f"Finding candidates for: {species_name}")
    logger.info("=" * 60)

    # 1. Get species info and occurrences
    logger.info("\n[1/4] Fetching GBIF data...")
    species_info = get_species_info(species_name)
    taxon_key = species_info["taxon_key"]
    logger.info(f"  Matched: {species_info['scientific_name']} (key: {taxon_key})")

    occurrences = fetch_occurrences(taxon_key, bbox)
    n_occurrences = len(occurrences)
    logger.info(f"  Found {n_occurrences} occurrences in region")

    if n_occurrences == 0:
        raise ValueError(f"No occurrences found for {species_name} in the specified region")

    # 2. Load embedding mosaic
    logger.info("\n[2/4] Loading embedding mosaic...")
    mosaic = EmbeddingMosaic(cache_dir, bbox)
    mosaic.load()
    h, w, c = mosaic.shape
    logger.info(f"  Mosaic shape: {h} x {w} x {c}")

    # 3. Sample embeddings at occurrence locations
    logger.info("\n[3/4] Sampling embeddings...")
    positive_embeddings, valid_coords = mosaic.sample_at_coords(occurrences)
    logger.info(f"  Valid occurrence samples: {len(positive_embeddings)}")

    if len(positive_embeddings) == 0:
        raise ValueError("No valid embeddings found at occurrence locations")

    # 4. Compute similarity
    logger.info("\n[4/4] Computing habitat similarity...")
    predictor = SimilarityMethod()
    predictor.fit(positive_embeddings)

    all_embeddings = mosaic.get_all_embeddings()
    scores = predictor.predict(all_embeddings)
    scores_map = scores.reshape(h, w)

    # Log statistics
    logger.info(f"\n  Score range: {scores.min():.3f} - {scores.max():.3f}")
    high_score = (scores > 0.6).sum()
    logger.info(f"  High similarity pixels (>0.6): {high_score:,} ({100*high_score/len(scores):.1f}%)")

    # Create result
    result = PredictionResult(
        species_name=species_info["canonical_name"],
        taxon_key=taxon_key,
        n_occurrences=len(valid_coords),
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
