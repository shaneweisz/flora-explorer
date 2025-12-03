#!/usr/bin/env python3
"""
Experiment: How does performance vary with training sample size?

This directly addresses the "data-deficient" use case by showing:
- How well does the model work with only 5-10 training samples?
- Where does performance plateau?
- Is there a minimum viable sample size?

Uses spatial block CV to prevent autocorrelation leakage, but subsamples
the training data within each fold to simulate data-deficient scenarios.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from finder import get_species_info, fetch_occurrences, EmbeddingMosaic
from finder.pipeline import REGIONS
from experiment import (
    thin_occurrences,
    assign_spatial_blocks,
    spatial_block_split,
    sample_background_points,
    compute_auc,
    compute_random_baseline_auc,
    THINNING_RESOLUTION_DEG,
    SPATIAL_BLOCK_SIZE,
    NEGATIVE_RATIO,
    BASE_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments" / "sample_size"

# Test species - use ones with enough data to subsample
SPECIES_LIST = [
    "Quercus robur",       # 44 after thinning
    "Fraxinus excelsior",  # 86 after thinning
    "Urtica dioica",       # 109 after thinning
    "Crataegus monogyna",  # Should have decent numbers
]

REGION = "cambridge"

# Sample sizes to test
N_TRAIN_VALUES = [2, 5, 10, 20, 50]

# Number of random subsamples per fold per n value
N_SUBSAMPLES = 5


def run_sample_size_experiment():
    """Run experiment varying training sample size within spatial CV."""

    logger.info("=" * 60)
    logger.info("Sample Size Experiment")
    logger.info("How does performance vary with training data amount?")
    logger.info("=" * 60)

    bbox = REGIONS[REGION]["bbox"]

    # Load mosaic
    logger.info("\nLoading embedding mosaic...")
    mosaic = EmbeddingMosaic(CACHE_DIR, bbox)
    mosaic.load()
    logger.info(f"Mosaic shape: {mosaic.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for species_name in SPECIES_LIST:
        logger.info(f"\n{'='*60}")
        logger.info(f"Species: {species_name}")
        logger.info("=" * 60)

        # Get occurrences
        species_info = get_species_info(species_name)
        occurrences = fetch_occurrences(species_info["taxon_key"], bbox)
        logger.info(f"Total occurrences: {len(occurrences)}")

        # Get embeddings
        all_occ_emb, valid_coords = mosaic.sample_at_coords(occurrences)

        # Thin
        thinned_coords, thinned_emb = thin_occurrences(
            valid_coords, all_occ_emb, resolution_deg=THINNING_RESOLUTION_DEG
        )
        n_total = len(thinned_coords)
        logger.info(f"After thinning: {n_total}")

        if n_total < 30:
            logger.info("Not enough data for sample size experiment, skipping")
            continue

        # Assign spatial blocks
        n_blocks = SPATIAL_BLOCK_SIZE
        total_blocks = n_blocks * n_blocks
        block_ids = assign_spatial_blocks(thinned_coords, bbox, n_blocks)
        block_counts = {i: int(np.sum(block_ids == i)) for i in range(total_blocks)}

        species_results = {
            "species": species_name,
            "n_total": n_total,
            "sample_sizes": [],
        }

        # For each sample size
        for n_train in N_TRAIN_VALUES:
            logger.info(f"\n  n_train = {n_train}")

            fold_aucs = []
            fold_aucs_vs_random = []
            valid_folds = 0

            # Spatial block CV
            for test_block in range(total_blocks):
                if block_counts[test_block] < 2:
                    continue

                # Split by block
                train_emb_full, train_coords_full, test_emb, test_coords = spatial_block_split(
                    thinned_coords, thinned_emb, bbox, test_block, n_blocks
                )

                n_train_available = len(train_emb_full)

                if n_train_available < n_train or len(test_emb) < 2:
                    continue

                # Run multiple subsamples of training data
                subsample_aucs = []

                for subsample_idx in range(N_SUBSAMPLES):
                    rng = np.random.default_rng(BASE_SEED + test_block * 100 + subsample_idx)

                    # Subsample training positives
                    train_indices = rng.choice(n_train_available, size=n_train, replace=False)
                    train_emb = train_emb_full[train_indices]
                    train_coords_sub = [train_coords_full[i] for i in train_indices]

                    # Sample background (proportional to training size)
                    n_neg = n_train * NEGATIVE_RATIO
                    train_neg_emb, train_neg_coords = sample_background_points(
                        mosaic, n_neg, train_coords_sub + test_coords, rng
                    )

                    # Test background (match test size)
                    test_neg_emb, test_neg_coords = sample_background_points(
                        mosaic, len(test_emb),
                        train_coords_sub + test_coords + train_neg_coords, rng
                    )

                    # Train classifier
                    X_train = np.vstack([train_emb, train_neg_emb])
                    y_train = np.array([1] * len(train_emb) + [0] * len(train_neg_emb))

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)

                    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
                    clf.fit(X_train_scaled, y_train)

                    # Test
                    X_test = np.vstack([test_emb, test_neg_emb])
                    X_test_scaled = scaler.transform(X_test)
                    probs = clf.predict_proba(X_test_scaled)[:, 1]

                    pos_scores = probs[:len(test_emb)]
                    neg_scores = probs[len(test_emb):]

                    auc = compute_auc(pos_scores, neg_scores)
                    subsample_aucs.append(auc)

                # Average across subsamples for this fold
                fold_auc = float(np.mean(subsample_aucs))
                fold_aucs.append(fold_auc)

                baseline_auc, _ = compute_random_baseline_auc(len(test_emb), len(test_neg_emb))
                fold_aucs_vs_random.append(fold_auc - baseline_auc)

                valid_folds += 1

            if valid_folds < 2:
                continue

            auc_mean = float(np.mean(fold_aucs))
            auc_std = float(np.std(fold_aucs))
            vs_random_mean = float(np.mean(fold_aucs_vs_random))

            logger.info(f"    AUC: {auc_mean:.3f} ± {auc_std:.3f} (vs random: {vs_random_mean:+.3f})")

            species_results["sample_sizes"].append({
                "n_train": n_train,
                "n_folds": valid_folds,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "auc_vs_random": vs_random_mean,
            })

        all_results.append(species_results)

        # Save per-species results
        slug = species_name.lower().replace(" ", "_")
        output_path = OUTPUT_DIR / f"{slug}.json"
        with open(output_path, "w") as f:
            json.dump(species_results, f, indent=2)
        logger.info(f"\nSaved: {output_path}")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: AUC by Training Sample Size")
    logger.info("=" * 70)

    # Header
    header = f"{'Species':<22}"
    for n in N_TRAIN_VALUES:
        header += f" n={n:>3}"
    logger.info(header)
    logger.info("-" * 70)

    for result in all_results:
        row = f"{result['species']:<22}"
        for size_result in result["sample_sizes"]:
            row += f" {size_result['auc_mean']:>.3f}"
        logger.info(row)

    # Save summary
    summary = {
        "n_train_values": N_TRAIN_VALUES,
        "n_subsamples": N_SUBSAMPLES,
        "negative_ratio": NEGATIVE_RATIO,
        "species": all_results,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary: {summary_path}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_sample_size_experiment()
