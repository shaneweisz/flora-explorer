#!/usr/bin/env python3
"""
Experiment: Validate classifier approach for habitat prediction.

Tests whether held-out occurrences score higher than random background points.
Runs multiple trials per n value to reduce variance from random sampling.

Supports two model types:
- logistic: Logistic Regression (fast, simple)
- mlp: MLP with MC Dropout (provides uncertainty estimates)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from finder import get_species_info, fetch_occurrences, EmbeddingMosaic
from finder.pipeline import REGIONS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments"

# Experiment parameters
SPECIES_LIST = [
    "Quercus robur",       # Common Oak - widespread woodland
    "Fraxinus excelsior",  # Ash - woodland/hedgerows
    "Urtica dioica",       # Stinging Nettle - disturbed/nutrient-rich
    "Alnus glutinosa",     # Alder - wetland specialist
    "Salix caprea",        # Goat Willow - fewer occurrences (47)
]
REGION = "cambridge"
N_POSITIVE_VALUES = [1, 2, 5, 10, 20, 50, 100]  # Positive training samples
NEGATIVE_RATIO = 5  # Background samples per positive (aligned with production)
N_TRIALS = 5  # Number of random trials per n value
BASE_SEED = 42
SPATIAL_BLOCK_SIZE = 3  # 3x3 grid of spatial blocks for CV
THINNING_RESOLUTION_DEG = 0.005  # ~500m thinning grid

ModelType = Literal["logistic", "mlp"]


class MLPClassifier(nn.Module):
    """Simple MLP with dropout for experiments."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout_rate: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def train_mlp(
    train_pos_emb: np.ndarray,
    train_neg_emb: np.ndarray,
    n_epochs: int = 100,
    hidden_dim: int = 256,
    dropout_rate: float = 0.3,
    lr: float = 1e-3,
) -> Tuple[MLPClassifier, StandardScaler]:
    """Train MLP classifier."""
    X = np.vstack([train_pos_emb, train_neg_emb])
    y = np.array([1.0] * len(train_pos_emb) + [0.0] * len(train_neg_emb))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MLPClassifier(input_dim=X_scaled.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for _ in range(n_epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model, scaler


def predict_mlp(
    model: MLPClassifier,
    scaler: StandardScaler,
    test_emb: np.ndarray,
    n_mc_samples: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with MC Dropout, returning scores and uncertainties."""
    X_scaled = scaler.transform(test_emb)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.train()  # Keep dropout active
    all_preds = []

    with torch.no_grad():
        for _ in range(n_mc_samples):
            preds = model(X_tensor).numpy()
            all_preds.append(preds)

    all_preds = np.array(all_preds)
    scores = all_preds.mean(axis=0)
    uncertainties = all_preds.std(axis=0)

    return scores, uncertainties


def compute_classifier_logistic(
    train_pos_emb: np.ndarray,
    train_neg_emb: np.ndarray,
    test_emb: np.ndarray
) -> Tuple[np.ndarray, None]:
    """Logistic regression classifier scores."""
    X_train = np.vstack([train_pos_emb, train_neg_emb])
    y_train = np.array([1] * len(train_pos_emb) + [0] * len(train_neg_emb))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(test_emb)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train_scaled, y_train)

    return clf.predict_proba(test_scaled)[:, 1], None


def compute_classifier_mlp(
    train_pos_emb: np.ndarray,
    train_neg_emb: np.ndarray,
    test_emb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """MLP classifier with MC Dropout scores and uncertainties."""
    model, scaler = train_mlp(train_pos_emb, train_neg_emb)
    return predict_mlp(model, scaler, test_emb)


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


def compute_random_baseline_auc(n_pos: int, n_neg: int, n_trials: int = 100) -> tuple[float, float]:
    """Compute expected AUC for random predictions (should be ~0.5)."""
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(n_trials):
        pos_scores = rng.random(n_pos)
        neg_scores = rng.random(n_neg)
        aucs.append(compute_auc(pos_scores, neg_scores))
    return float(np.mean(aucs)), float(np.std(aucs))


def find_optimal_threshold(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[float, dict]:
    """Find threshold that maximizes F1 score."""
    thresholds = np.linspace(0.1, 0.9, 17)
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics = compute_classification_metrics(pos_scores, neg_scores, threshold=0.5)

    for thresh in thresholds:
        metrics = compute_classification_metrics(pos_scores, neg_scores, threshold=thresh)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = thresh
            best_metrics = metrics

    return float(best_threshold), best_metrics


def thin_occurrences(
    coords: list[tuple[float, float]],
    embeddings: np.ndarray,
    resolution_deg: float = 0.005,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """
    Thin occurrences to max one per grid cell to reduce spatial clustering.

    Args:
        coords: List of (lon, lat) coordinates
        embeddings: Corresponding embeddings array
        resolution_deg: Grid cell size in degrees (~500m at 0.005°)

    Returns:
        Thinned coordinates and embeddings
    """
    seen_cells = set()
    thinned_coords = []
    thinned_indices = []

    for i, (lon, lat) in enumerate(coords):
        cell = (int(lon / resolution_deg), int(lat / resolution_deg))
        if cell not in seen_cells:
            seen_cells.add(cell)
            thinned_coords.append((lon, lat))
            thinned_indices.append(i)

    thinned_emb = embeddings[thinned_indices] if thinned_indices else np.array([])
    return thinned_coords, thinned_emb


def assign_spatial_blocks(
    coords: list[tuple[float, float]],
    bbox: tuple[float, float, float, float],
    n_blocks: int = 3,
) -> np.ndarray:
    """
    Assign each coordinate to a spatial block for block cross-validation.

    Args:
        coords: List of (lon, lat) coordinates
        bbox: Region bounding box (min_lon, min_lat, max_lon, max_lat)
        n_blocks: Number of blocks per dimension (total blocks = n_blocks²)

    Returns:
        Array of block indices (0 to n_blocks²-1) for each coordinate
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_step = (max_lon - min_lon) / n_blocks
    lat_step = (max_lat - min_lat) / n_blocks

    block_ids = []
    for lon, lat in coords:
        col = min(int((lon - min_lon) / lon_step), n_blocks - 1)
        row = min(int((lat - min_lat) / lat_step), n_blocks - 1)
        block_id = row * n_blocks + col
        block_ids.append(block_id)

    return np.array(block_ids)


def spatial_block_split(
    coords: list[tuple[float, float]],
    embeddings: np.ndarray,
    bbox: tuple[float, float, float, float],
    test_block: int,
    n_blocks: int = 3,
) -> tuple[np.ndarray, list, np.ndarray, list]:
    """
    Split data by spatial blocks - one block for test, rest for train.

    Returns:
        train_emb, train_coords, test_emb, test_coords
    """
    block_ids = assign_spatial_blocks(coords, bbox, n_blocks)

    train_mask = block_ids != test_block
    test_mask = block_ids == test_block

    train_emb = embeddings[train_mask]
    test_emb = embeddings[test_mask]
    train_coords = [c for c, m in zip(coords, train_mask) if m]
    test_coords = [c for c, m in zip(coords, test_mask) if m]

    return train_emb, train_coords, test_emb, test_coords


def compute_classification_metrics(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """Compute precision, recall, F1 at a given threshold."""
    # True positives: positive samples predicted as positive
    tp = np.sum(pos_scores >= threshold)
    # False negatives: positive samples predicted as negative
    fn = np.sum(pos_scores < threshold)
    # False positives: negative samples predicted as positive
    fp = np.sum(neg_scores >= threshold)
    # True negatives: negative samples predicted as negative
    tn = np.sum(neg_scores < threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def run_single_trial_spatial_cv(
    train_emb: np.ndarray,
    train_coords: list[tuple[float, float]],
    test_pos_emb: np.ndarray,
    test_pos_coords: list[tuple[float, float]],
    mosaic: EmbeddingMosaic,
    rng: np.random.Generator,
    model_type: ModelType = "logistic",
    negative_ratio: int = NEGATIVE_RATIO,
) -> dict:
    """
    Run a single trial with pre-split spatial CV data.

    Uses spatial block CV: train and test points come from different geographic regions.
    """
    n_train = len(train_emb)
    n_test = len(test_pos_emb)

    if n_train < 2 or n_test < 2:
        return None  # Skip if not enough data

    # Sample background for training (aligned with production ratio)
    n_train_neg = n_train * negative_ratio
    train_neg_emb, train_neg_coords = sample_background_points(
        mosaic, n_train_neg, train_coords + test_pos_coords, rng
    )

    # Sample background for testing (match test size for balanced evaluation)
    test_neg_emb, test_neg_coords = sample_background_points(
        mosaic, n_test, train_coords + test_pos_coords + train_neg_coords, rng
    )

    # Combine test embeddings for single prediction call
    test_all_emb = np.vstack([test_pos_emb, test_neg_emb])

    # Train classifier and score based on model type
    if model_type == "mlp":
        all_scores, all_uncertainties = compute_classifier_mlp(
            train_emb, train_neg_emb, test_all_emb
        )
    else:
        all_scores, all_uncertainties = compute_classifier_logistic(
            train_emb, train_neg_emb, test_all_emb
        )

    pos_scores = all_scores[:len(test_pos_emb)]
    neg_scores = all_scores[len(test_pos_emb):]

    pos_uncertainties = None
    neg_uncertainties = None
    if all_uncertainties is not None:
        pos_uncertainties = all_uncertainties[:len(test_pos_emb)]
        neg_uncertainties = all_uncertainties[len(test_pos_emb):]

    # Compute metrics
    auc = compute_auc(pos_scores, neg_scores)

    # Use optimal threshold instead of fixed 0.5
    optimal_thresh, optimal_metrics = find_optimal_threshold(pos_scores, neg_scores)
    fixed_metrics = compute_classification_metrics(pos_scores, neg_scores, threshold=0.5)

    # Compute random baseline for comparison
    baseline_auc, baseline_std = compute_random_baseline_auc(len(pos_scores), len(neg_scores))

    result = {
        "auc": auc,
        "auc_vs_random": auc - baseline_auc,  # How much better than random
        "baseline_auc": baseline_auc,
        "optimal_threshold": optimal_thresh,
        "precision": optimal_metrics["precision"],
        "recall": optimal_metrics["recall"],
        "f1": optimal_metrics["f1"],
        "accuracy": optimal_metrics["accuracy"],
        "precision_at_0.5": fixed_metrics["precision"],
        "recall_at_0.5": fixed_metrics["recall"],
        "f1_at_0.5": fixed_metrics["f1"],
        "mean_positive": float(pos_scores.mean()),
        "mean_negative": float(neg_scores.mean()),
        "n_train_positive": n_train,
        "n_train_negative": len(train_neg_coords),
        "n_test_positive": n_test,
        "n_test_negative": len(test_neg_coords),
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

    # Add uncertainty data for MLP
    if pos_uncertainties is not None:
        result["mean_uncertainty_positive"] = float(pos_uncertainties.mean())
        result["mean_uncertainty_negative"] = float(neg_uncertainties.mean())
        for i, pt in enumerate(result["test_positive"]):
            pt["uncertainty"] = float(pos_uncertainties[i])
        for i, pt in enumerate(result["test_negative"]):
            pt["uncertainty"] = float(neg_uncertainties[i])

    return result


def run_species_experiment(
    species_name: str,
    mosaic: EmbeddingMosaic,
    model_type: ModelType = "logistic",
):
    """
    Run spatial block cross-validation experiment for a single species.

    Methodology improvements:
    1. Spatial thinning to reduce clustering bias
    2. Spatial block CV to prevent spatial autocorrelation leakage
    3. Aligned negative ratio with production (5:1)
    4. Random baseline comparison
    5. Optimal threshold selection
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Species: {species_name} (model: {model_type})")
    logger.info("=" * 60)

    bbox = REGIONS[REGION]["bbox"]
    species_info = get_species_info(species_name)
    occurrences = fetch_occurrences(species_info["taxon_key"], bbox)
    logger.info(f"Total occurrences: {len(occurrences)}")

    all_occ_emb, valid_coords = mosaic.sample_at_coords(occurrences)
    n_before_thin = len(valid_coords)
    logger.info(f"Valid with embeddings: {n_before_thin}")

    # Apply spatial thinning to reduce clustering
    thinned_coords, thinned_emb = thin_occurrences(
        valid_coords, all_occ_emb, resolution_deg=THINNING_RESOLUTION_DEG
    )
    n_total = len(thinned_coords)
    logger.info(f"After thinning (~500m): {n_total} (removed {n_before_thin - n_total})")

    # Need enough points for spatial block CV
    min_required = 15  # At least a few points in each block
    if n_total < min_required:
        logger.info(f"Not enough occurrences after thinning (need {min_required}), skipping")
        return None

    # Assign spatial blocks
    n_blocks = SPATIAL_BLOCK_SIZE
    total_blocks = n_blocks * n_blocks
    block_ids = assign_spatial_blocks(thinned_coords, bbox, n_blocks)

    # Count points per block
    block_counts = {i: int(np.sum(block_ids == i)) for i in range(total_blocks)}
    logger.info(f"Points per block: {dict(sorted(block_counts.items()))}")

    # Run spatial block CV (leave-one-block-out)
    # Each block serves as test set once
    trials = []
    aucs = []
    auc_vs_randoms = []
    f1s = []
    precisions = []
    recalls = []
    optimal_thresholds = []

    valid_folds = 0
    for test_block in range(total_blocks):
        if block_counts[test_block] < 2:
            continue  # Skip blocks with too few test points

        train_emb, train_coords, test_emb, test_coords = spatial_block_split(
            thinned_coords, thinned_emb, bbox, test_block, n_blocks
        )

        if len(train_emb) < 5 or len(test_emb) < 2:
            continue  # Need minimum training and test data

        rng = np.random.default_rng(BASE_SEED + test_block)

        trial_result = run_single_trial_spatial_cv(
            train_emb, train_coords,
            test_emb, test_coords,
            mosaic, rng,
            model_type=model_type,
            negative_ratio=NEGATIVE_RATIO,
        )

        if trial_result is None:
            continue

        trial_result["test_block"] = test_block
        trial_result["n_blocks"] = total_blocks
        trials.append(trial_result)

        aucs.append(trial_result["auc"])
        auc_vs_randoms.append(trial_result["auc_vs_random"])
        f1s.append(trial_result["f1"])
        precisions.append(trial_result["precision"])
        recalls.append(trial_result["recall"])
        optimal_thresholds.append(trial_result["optimal_threshold"])
        valid_folds += 1

    if valid_folds < 2:
        logger.info("Not enough valid spatial CV folds, skipping")
        return None

    # Aggregate metrics across folds
    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    auc_vs_random_mean = float(np.mean(auc_vs_randoms))
    f1_mean = float(np.mean(f1s))
    f1_std = float(np.std(f1s))
    precision_mean = float(np.mean(precisions))
    recall_mean = float(np.mean(recalls))
    threshold_mean = float(np.mean(optimal_thresholds))

    logger.info(f"\nSpatial Block CV Results ({valid_folds} folds):")
    logger.info(f"  AUC: {auc_mean:.3f} ± {auc_std:.3f} (vs random: +{auc_vs_random_mean:.3f})")
    logger.info(f"  F1:  {f1_mean:.3f} ± {f1_std:.3f}")
    logger.info(f"  P/R: {precision_mean:.3f}/{recall_mean:.3f}")
    logger.info(f"  Optimal threshold: {threshold_mean:.2f}")

    return {
        "species": species_name,
        "species_key": species_info["taxon_key"],
        "region": REGION,
        "model_type": model_type,
        "validation_method": "spatial_block_cv",
        "n_blocks": total_blocks,
        "n_valid_folds": valid_folds,
        "n_occurrences_raw": n_before_thin,
        "n_occurrences_thinned": n_total,
        "thinning_resolution_deg": THINNING_RESOLUTION_DEG,
        "negative_ratio": NEGATIVE_RATIO,
        "block_counts": block_counts,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_vs_random_mean": auc_vs_random_mean,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "precision_mean": precision_mean,
        "recall_mean": recall_mean,
        "optimal_threshold_mean": threshold_mean,
        "trials": trials,
    }


def run_all_experiments(model_type: ModelType = "logistic"):
    """Run spatial block CV experiments for all species."""
    logger.info("=" * 60)
    logger.info(f"Spatial Block CV Validation Experiment (model: {model_type})")
    logger.info(f"Validation: {SPATIAL_BLOCK_SIZE}x{SPATIAL_BLOCK_SIZE} spatial blocks, leave-one-out")
    logger.info(f"Thinning: {THINNING_RESOLUTION_DEG}° (~{THINNING_RESOLUTION_DEG * 111:.0f}m)")
    logger.info(f"Negative ratio: {NEGATIVE_RATIO}:1")
    logger.info("=" * 60)

    bbox = REGIONS[REGION]["bbox"]

    # Load mosaic once
    logger.info("\nLoading embedding mosaic...")
    mosaic = EmbeddingMosaic(CACHE_DIR, bbox)
    mosaic.load()
    logger.info(f"Mosaic shape: {mosaic.shape}")

    # Create output directory for this model type
    output_dir = OUTPUT_DIR / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary for all species
    summary = {
        "region": REGION,
        "model_type": model_type,
        "validation_method": "spatial_block_cv",
        "n_blocks": SPATIAL_BLOCK_SIZE * SPATIAL_BLOCK_SIZE,
        "thinning_resolution_deg": THINNING_RESOLUTION_DEG,
        "negative_ratio": NEGATIVE_RATIO,
        "base_seed": BASE_SEED,
        "species": [],
    }

    for species in SPECIES_LIST:
        result = run_species_experiment(species, mosaic, model_type=model_type)

        if result:
            # Save per-species (full data with coordinates)
            slug = species.lower().replace(" ", "_")
            output_path = output_dir / f"{slug}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"\nSaved: {output_path}")

            # Add to summary
            species_summary = {
                "species": species,
                "n_occurrences_raw": result["n_occurrences_raw"],
                "n_occurrences_thinned": result["n_occurrences_thinned"],
                "n_valid_folds": result["n_valid_folds"],
                "auc_mean": result["auc_mean"],
                "auc_std": result["auc_std"],
                "auc_vs_random": result["auc_vs_random_mean"],
                "f1_mean": result["f1_mean"],
                "precision_mean": result["precision_mean"],
                "recall_mean": result["recall_mean"],
                "optimal_threshold": result["optimal_threshold_mean"],
            }
            summary["species"].append(species_summary)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary: {summary_path}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info(f"SUMMARY - Spatial Block CV ({model_type})")
    logger.info("=" * 60)
    logger.info(f"{'Species':<22} {'Raw':>5} {'Thin':>5} {'Folds':>5} {'AUC':>7} {'±':>5} {'vs Rnd':>7} {'Thresh':>6}")
    logger.info("-" * 72)
    for sp in summary["species"]:
        logger.info(
            f"{sp['species']:<22} {sp['n_occurrences_raw']:>5} {sp['n_occurrences_thinned']:>5} "
            f"{sp['n_valid_folds']:>5} {sp['auc_mean']:>7.3f} {sp['auc_std']:>5.3f} "
            f"{sp['auc_vs_random']:>+7.3f} {sp['optimal_threshold']:>6.2f}"
        )

    logger.info("-" * 72)
    if summary["species"]:
        avg_auc = np.mean([sp["auc_mean"] for sp in summary["species"]])
        avg_vs_random = np.mean([sp["auc_vs_random"] for sp in summary["species"]])
        logger.info(f"{'AVERAGE':<22} {'':<5} {'':<5} {'':<5} {avg_auc:>7.3f} {'':<5} {avg_vs_random:>+7.3f}")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run classifier validation experiments"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic", "mlp", "both"],
        default="both",
        help="Model type to evaluate: logistic, mlp, or both (default: both)",
    )
    args = parser.parse_args()

    if args.model_type == "both":
        run_all_experiments(model_type="logistic")
        run_all_experiments(model_type="mlp")
    else:
        run_all_experiments(model_type=args.model_type)


if __name__ == "__main__":
    main()
