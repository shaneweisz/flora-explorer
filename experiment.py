#!/usr/bin/env python3
"""
Experiment: Validate classifier approach for habitat prediction.

Tests whether held-out occurrences score higher than random background points.
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from finder import get_species_info, fetch_occurrences, EmbeddingMosaic
from finder.pipeline import REGIONS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments"

# Experiment parameters
SPECIES_LIST = ["Quercus robur", "Fraxinus excelsior"]  # Oak, Ash
REGION = "cambridge"
N_POSITIVE_VALUES = [1, 2, 5, 10, 20, 50, 100]  # Positive training samples (matched by negatives)
SEED = 42


def compute_classifier(
    train_pos_emb: np.ndarray,
    train_neg_emb: np.ndarray,
    test_emb: np.ndarray
) -> np.ndarray:
    """Logistic regression classifier scores."""
    X_train = np.vstack([train_pos_emb, train_neg_emb])
    y_train = np.array([1] * len(train_pos_emb) + [0] * len(train_neg_emb))

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    return clf.predict_proba(test_emb)[:, 1]


def sample_background_points(
    mosaic: EmbeddingMosaic,
    n_points: int,
    exclude_coords: list[tuple[float, float]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Sample random background points."""
    h, w, _ = mosaic.shape

    exclude_pixels = set()
    for lon, lat in exclude_coords:
        row, col = mosaic.coords_to_pixel(lon, lat)
        exclude_pixels.add((row, col))

    coords = []
    embeddings = []
    attempts = 0
    max_attempts = n_points * 10

    while len(coords) < n_points and attempts < max_attempts:
        row = rng.integers(0, h)
        col = rng.integers(0, w)
        if (row, col) not in exclude_pixels:
            lon, lat = mosaic.pixel_to_coords(row, col)
            emb = mosaic.mosaic[row, col, :]
            if not np.allclose(emb, 0):
                coords.append((lon, lat))
                embeddings.append(emb)
                exclude_pixels.add((row, col))
        attempts += 1

    return np.array(embeddings), coords


def compute_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUC: P(random positive > random negative)."""
    n_comparisons = len(pos_scores) * len(neg_scores)
    if n_comparisons == 0:
        return 0.5
    n_correct = sum((p > n) for p in pos_scores for n in neg_scores)
    return n_correct / n_comparisons


def run_species_experiment(species_name: str, mosaic: EmbeddingMosaic, rng: np.random.Generator):
    """Run experiment for a single species."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Species: {species_name}")
    logger.info("=" * 60)

    bbox = REGIONS[REGION]["bbox"]
    species_info = get_species_info(species_name)
    occurrences = fetch_occurrences(species_info["taxon_key"], bbox)
    logger.info(f"Total occurrences: {len(occurrences)}")

    all_occ_emb, valid_coords = mosaic.sample_at_coords(occurrences)
    n_total = len(valid_coords)
    logger.info(f"Valid with embeddings: {n_total}")

    if n_total < 25:
        logger.info("Not enough occurrences, skipping")
        return None

    # Shuffle
    indices = rng.permutation(n_total)
    all_occ_emb = all_occ_emb[indices]
    valid_coords = [valid_coords[i] for i in indices]

    experiments = []

    for n_pos in N_POSITIVE_VALUES:
        if n_pos >= n_total - 10:  # Need at least 10 test samples
            continue

        logger.info(f"\nn_positive = {n_pos} (+ {n_pos} negative = {n_pos * 2} total training samples)")

        # Split occurrences
        train_emb = all_occ_emb[:n_pos]
        train_coords = valid_coords[:n_pos]
        test_pos_emb = all_occ_emb[n_pos:]
        test_pos_coords = valid_coords[n_pos:]
        n_test = len(test_pos_coords)

        # Sample background for training classifier (match positive training size)
        train_neg_emb, train_neg_coords = sample_background_points(
            mosaic, n_pos, valid_coords, rng
        )

        # Sample background for testing (match test size)
        test_neg_emb, test_neg_coords = sample_background_points(
            mosaic, n_test, valid_coords + train_neg_coords, rng
        )

        # Train classifier and score
        pos_scores = compute_classifier(train_emb, train_neg_emb, test_pos_emb)
        neg_scores = compute_classifier(train_emb, train_neg_emb, test_neg_emb)
        auc = compute_auc(pos_scores, neg_scores)

        logger.info(f"  AUC: {auc:.3f}")
        logger.info(f"  Mean positive score: {pos_scores.mean():.3f}")
        logger.info(f"  Mean negative score: {neg_scores.mean():.3f}")

        exp_data = {
            "n_positive": n_pos,
            "n_negative": n_pos,  # Always matches n_positive
            "n_test_positive": n_test,
            "n_test_negative": len(test_neg_coords),
            "auc": auc,
            "mean_positive": float(pos_scores.mean()),
            "mean_negative": float(neg_scores.mean()),
            "train_positive": [{"lon": lon, "lat": lat} for lon, lat in train_coords],
            "train_negative": [{"lon": lon, "lat": lat} for lon, lat in train_neg_coords],
            "test_positive": [
                {"lon": lon, "lat": lat, "score": float(s)}
                for (lon, lat), s in zip(test_pos_coords, pos_scores)
            ],
            "test_negative": [
                {"lon": lon, "lat": lat, "score": float(s)}
                for (lon, lat), s in zip(test_neg_coords, neg_scores)
            ],
        }
        experiments.append(exp_data)

    return {
        "species": species_name,
        "species_key": species_info["taxon_key"],
        "region": REGION,
        "n_occurrences": n_total,
        "experiments": experiments,
    }


def run_all_experiments():
    """Run experiments for all species."""
    logger.info("=" * 60)
    logger.info("Classifier Validation Experiment")
    logger.info("=" * 60)

    rng = np.random.default_rng(SEED)
    bbox = REGIONS[REGION]["bbox"]

    # Load mosaic once
    logger.info("\nLoading embedding mosaic...")
    mosaic = EmbeddingMosaic(CACHE_DIR, bbox)
    mosaic.load()
    logger.info(f"Mosaic shape: {mosaic.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Summary for all species
    summary = {
        "region": REGION,
        "seed": SEED,
        "n_positive_values": N_POSITIVE_VALUES,
        "species": [],
    }

    for species in SPECIES_LIST:
        result = run_species_experiment(species, mosaic, rng)

        if result:
            # Save per-species (full data with coordinates)
            slug = species.lower().replace(" ", "_")
            output_path = OUTPUT_DIR / f"{slug}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"\nSaved: {output_path}")

            # Add to summary (just metrics, no coordinates)
            species_summary = {
                "species": species,
                "n_occurrences": result["n_occurrences"],
                "results": [
                    {"n_positive": exp["n_positive"], "auc": exp["auc"]}
                    for exp in result["experiments"]
                ],
            }
            summary["species"].append(species_summary)

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary: {summary_path}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Species':<25} {'n_pos':>6} {'n_neg':>6} {'AUC':>10}")
    logger.info("-" * 52)
    for sp in summary["species"]:
        for r in sp["results"]:
            logger.info(f"{sp['species']:<25} {r['n_positive']:>6} {r['n_positive']:>6} {r['auc']:>10.3f}")
        logger.info("")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_all_experiments()
