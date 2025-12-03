#!/usr/bin/env python3
"""
Baseline comparison: Land cover only vs Tessera embeddings.

Downloads ESA WorldCover 10m data and trains a classifier using only
land cover class as the feature. This tests whether Tessera embeddings
add value beyond simple land cover classification.

ESA WorldCover classes:
    10: Tree cover
    20: Shrubland
    30: Grassland
    40: Cropland
    50: Built-up
    60: Bare / sparse vegetation
    70: Snow and ice
    80: Permanent water bodies
    90: Herbaceous wetland
    95: Mangroves
    100: Moss and lichen
"""

import logging
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from finder import get_species_info, fetch_occurrences, EmbeddingMosaic
from finder.pipeline import REGIONS
from experiment import (
    thin_occurrences,
    assign_spatial_blocks,
    spatial_block_split,
    sample_background_points,
    compute_auc,
    find_optimal_threshold,
    compute_random_baseline_auc,
    SPECIES_LIST,
    THINNING_RESOLUTION_DEG,
    SPATIAL_BLOCK_SIZE,
    NEGATIVE_RATIO,
    BASE_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
WORLDCOVER_DIR = PROJECT_ROOT / "cache" / "worldcover"

REGION = "cambridge"

# ESA WorldCover S3 bucket
# Cambridge is in tile N51E000 (covers 0-3°E, 51-54°N)
WORLDCOVER_TILE = "N51E000"
WORLDCOVER_URL = f"s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_{WORLDCOVER_TILE}_Map.tif"

# Land cover classes
LC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def download_worldcover():
    """Download ESA WorldCover tile for Cambridge region."""
    WORLDCOVER_DIR.mkdir(parents=True, exist_ok=True)
    local_path = WORLDCOVER_DIR / f"ESA_WorldCover_10m_2021_v200_{WORLDCOVER_TILE}_Map.tif"

    if local_path.exists():
        logger.info(f"WorldCover tile already exists: {local_path}")
        return local_path

    logger.info(f"Downloading WorldCover tile {WORLDCOVER_TILE}...")
    cmd = [
        "aws", "s3", "cp",
        WORLDCOVER_URL,
        str(local_path),
        "--no-sign-request"
    ]
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Downloaded: {local_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download WorldCover: {e}")
        raise
    except FileNotFoundError:
        logger.error("AWS CLI not found. Install with: pip install awscli")
        raise

    return local_path


def extract_landcover_at_coords(
    worldcover_path: Path,
    coords: list[tuple[float, float]],
) -> np.ndarray:
    """Extract land cover class at each coordinate."""
    with rasterio.open(worldcover_path) as src:
        lc_values = []
        for lon, lat in coords:
            try:
                # Sample the raster at this point
                row, col = src.index(lon, lat)
                value = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                lc_values.append(value)
            except Exception:
                # Outside raster bounds
                lc_values.append(0)

    return np.array(lc_values)


def encode_landcover(lc_values: np.ndarray) -> np.ndarray:
    """One-hot encode land cover classes."""
    # Reshape for sklearn
    lc_reshaped = lc_values.reshape(-1, 1)

    # Create encoder with all possible classes
    encoder = OneHotEncoder(categories=[LC_CLASSES], sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(lc_reshaped)

    return encoded


def run_baseline_experiment(worldcover_path: Path, mosaic: EmbeddingMosaic):
    """Run spatial block CV comparing land cover baseline to Tessera embeddings."""
    bbox = REGIONS[REGION]["bbox"]

    results = []

    for species_name in SPECIES_LIST:
        logger.info(f"\n{'='*60}")
        logger.info(f"Species: {species_name}")
        logger.info("=" * 60)

        # Get occurrences
        species_info = get_species_info(species_name)
        occurrences = fetch_occurrences(species_info["taxon_key"], bbox)
        logger.info(f"Total occurrences: {len(occurrences)}")

        # Get embeddings at occurrence locations
        all_occ_emb, valid_coords = mosaic.sample_at_coords(occurrences)
        n_before_thin = len(valid_coords)

        # Apply spatial thinning
        thinned_coords, thinned_emb = thin_occurrences(
            valid_coords, all_occ_emb, resolution_deg=THINNING_RESOLUTION_DEG
        )
        n_total = len(thinned_coords)
        logger.info(f"After thinning: {n_total} (from {n_before_thin})")

        if n_total < 15:
            logger.info("Not enough occurrences, skipping")
            continue

        # Get land cover at thinned locations
        thinned_lc = extract_landcover_at_coords(worldcover_path, thinned_coords)
        logger.info(f"Land cover classes present: {sorted(set(thinned_lc))}")

        # Assign spatial blocks
        n_blocks = SPATIAL_BLOCK_SIZE
        total_blocks = n_blocks * n_blocks
        block_ids = assign_spatial_blocks(thinned_coords, bbox, n_blocks)
        block_counts = {i: int(np.sum(block_ids == i)) for i in range(total_blocks)}

        # Run spatial block CV
        tessera_aucs = []
        landcover_aucs = []
        valid_folds = 0

        for test_block in range(total_blocks):
            if block_counts[test_block] < 2:
                continue

            # Split by spatial block
            train_emb, train_coords, test_emb, test_coords = spatial_block_split(
                thinned_coords, thinned_emb, bbox, test_block, n_blocks
            )

            if len(train_emb) < 5 or len(test_emb) < 2:
                continue

            # Also split land cover
            train_mask = block_ids != test_block
            test_mask = block_ids == test_block
            train_lc = thinned_lc[train_mask]
            test_pos_lc = thinned_lc[test_mask]

            # Sample background points
            rng = np.random.default_rng(BASE_SEED + test_block)
            n_train_neg = len(train_emb) * NEGATIVE_RATIO
            train_neg_emb, train_neg_coords = sample_background_points(
                mosaic, n_train_neg, train_coords + test_coords, rng
            )

            n_test_neg = len(test_emb)
            test_neg_emb, test_neg_coords = sample_background_points(
                mosaic, n_test_neg, train_coords + test_coords + train_neg_coords, rng
            )

            # Get land cover for background points
            train_neg_lc = extract_landcover_at_coords(worldcover_path, train_neg_coords)
            test_neg_lc = extract_landcover_at_coords(worldcover_path, test_neg_coords)

            # ===== TESSERA CLASSIFIER =====
            X_train_tessera = np.vstack([train_emb, train_neg_emb])
            y_train = np.array([1] * len(train_emb) + [0] * len(train_neg_emb))

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_tessera)

            clf_tessera = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf_tessera.fit(X_train_scaled, y_train)

            X_test_tessera = np.vstack([test_emb, test_neg_emb])
            X_test_scaled = scaler.transform(X_test_tessera)
            tessera_probs = clf_tessera.predict_proba(X_test_scaled)[:, 1]

            tessera_pos_scores = tessera_probs[:len(test_emb)]
            tessera_neg_scores = tessera_probs[len(test_emb):]
            tessera_auc = compute_auc(tessera_pos_scores, tessera_neg_scores)
            tessera_aucs.append(tessera_auc)

            # ===== LAND COVER CLASSIFIER =====
            # One-hot encode land cover
            train_lc_all = np.concatenate([train_lc, train_neg_lc])
            test_lc_all = np.concatenate([test_pos_lc, test_neg_lc])

            # Encode with all classes
            encoder = OneHotEncoder(categories=[LC_CLASSES], sparse_output=False, handle_unknown='ignore')
            X_train_lc = encoder.fit_transform(train_lc_all.reshape(-1, 1))
            X_test_lc = encoder.transform(test_lc_all.reshape(-1, 1))

            clf_lc = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf_lc.fit(X_train_lc, y_train)

            lc_probs = clf_lc.predict_proba(X_test_lc)[:, 1]
            lc_pos_scores = lc_probs[:len(test_pos_lc)]
            lc_neg_scores = lc_probs[len(test_pos_lc):]
            lc_auc = compute_auc(lc_pos_scores, lc_neg_scores)
            landcover_aucs.append(lc_auc)

            valid_folds += 1

        if valid_folds < 2:
            logger.info("Not enough valid folds, skipping")
            continue

        # Compute statistics
        tessera_mean = float(np.mean(tessera_aucs))
        tessera_std = float(np.std(tessera_aucs))
        lc_mean = float(np.mean(landcover_aucs))
        lc_std = float(np.std(landcover_aucs))

        # Random baseline
        baseline_auc, _ = compute_random_baseline_auc(10, 10)

        improvement = tessera_mean - lc_mean

        logger.info(f"\nResults ({valid_folds} folds):")
        logger.info(f"  Tessera AUC:    {tessera_mean:.3f} ± {tessera_std:.3f}")
        logger.info(f"  LandCover AUC:  {lc_mean:.3f} ± {lc_std:.3f}")
        logger.info(f"  Random AUC:     {baseline_auc:.3f}")
        logger.info(f"  Improvement:    {improvement:+.3f} ({improvement/lc_mean*100:+.1f}%)")

        results.append({
            "species": species_name,
            "n_occurrences": n_total,
            "valid_folds": valid_folds,
            "tessera_auc_mean": tessera_mean,
            "tessera_auc_std": tessera_std,
            "landcover_auc_mean": lc_mean,
            "landcover_auc_std": lc_std,
            "improvement": improvement,
            "tessera_vs_random": tessera_mean - baseline_auc,
            "landcover_vs_random": lc_mean - baseline_auc,
        })

    return results


def main():
    logger.info("=" * 60)
    logger.info("Baseline Comparison: Tessera vs Land Cover Only")
    logger.info("=" * 60)

    # Download WorldCover if needed
    try:
        worldcover_path = download_worldcover()
    except Exception as e:
        logger.error(f"Could not get WorldCover data: {e}")
        logger.info("\nTrying alternative: using curl to download...")
        # Try curl as fallback
        WORLDCOVER_DIR.mkdir(parents=True, exist_ok=True)
        local_path = WORLDCOVER_DIR / f"ESA_WorldCover_10m_2021_v200_{WORLDCOVER_TILE}_Map.tif"
        if not local_path.exists():
            https_url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{WORLDCOVER_TILE}_Map.tif"
            subprocess.run(["curl", "-o", str(local_path), https_url], check=True)
        worldcover_path = local_path

    # Load embedding mosaic
    bbox = REGIONS[REGION]["bbox"]
    logger.info(f"\nLoading embedding mosaic for {REGION}...")
    mosaic = EmbeddingMosaic(CACHE_DIR, bbox)
    mosaic.load()
    logger.info(f"Mosaic shape: {mosaic.shape}")

    # Run comparison
    results = run_baseline_experiment(worldcover_path, mosaic)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Tessera vs Land Cover Baseline")
    logger.info("=" * 60)
    logger.info(f"{'Species':<22} {'Tessera':>10} {'LandCover':>10} {'Δ AUC':>10} {'Winner':>10}")
    logger.info("-" * 66)

    tessera_wins = 0
    lc_wins = 0
    for r in results:
        winner = "Tessera" if r["improvement"] > 0 else "LandCover"
        if r["improvement"] > 0:
            tessera_wins += 1
        else:
            lc_wins += 1
        logger.info(
            f"{r['species']:<22} "
            f"{r['tessera_auc_mean']:>10.3f} "
            f"{r['landcover_auc_mean']:>10.3f} "
            f"{r['improvement']:>+10.3f} "
            f"{winner:>10}"
        )

    logger.info("-" * 66)
    if results:
        avg_tessera = np.mean([r["tessera_auc_mean"] for r in results])
        avg_lc = np.mean([r["landcover_auc_mean"] for r in results])
        avg_improvement = np.mean([r["improvement"] for r in results])
        logger.info(
            f"{'AVERAGE':<22} "
            f"{avg_tessera:>10.3f} "
            f"{avg_lc:>10.3f} "
            f"{avg_improvement:>+10.3f}"
        )
        logger.info(f"\nTessera wins: {tessera_wins}/{len(results)}, LandCover wins: {lc_wins}/{len(results)}")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
