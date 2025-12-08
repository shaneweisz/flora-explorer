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
]
REGION = "cambridge"
N_POSITIVE_VALUES = [1, 2, 5, 10, 20, 50, 100]  # Positive training samples (matched by negatives)
N_TRIALS = 5  # Number of random trials per n value
BASE_SEED = 42

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


def run_single_trial(
    n_pos: int,
    all_occ_emb: np.ndarray,
    valid_coords: list[tuple[float, float]],
    mosaic: EmbeddingMosaic,
    rng: np.random.Generator,
    model_type: ModelType = "logistic",
) -> dict:
    """Run a single trial for a given n_positive value."""
    n_total = len(valid_coords)

    # Shuffle occurrences for this trial
    indices = rng.permutation(n_total)
    shuffled_emb = all_occ_emb[indices]
    shuffled_coords = [valid_coords[i] for i in indices]

    # Split occurrences
    train_emb = shuffled_emb[:n_pos]
    train_coords = shuffled_coords[:n_pos]
    test_pos_emb = shuffled_emb[n_pos:]
    test_pos_coords = shuffled_coords[n_pos:]
    n_test = len(test_pos_coords)

    # Sample background for training classifier (match positive training size)
    train_neg_emb, train_neg_coords = sample_background_points(
        mosaic, n_pos, shuffled_coords, rng
    )

    # Sample background for testing (match test size)
    test_neg_emb, test_neg_coords = sample_background_points(
        mosaic, n_test, shuffled_coords + train_neg_coords, rng
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
    metrics = compute_classification_metrics(pos_scores, neg_scores, threshold=0.5)

    result = {
        "auc": auc,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "mean_positive": float(pos_scores.mean()),
        "mean_negative": float(neg_scores.mean()),
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
        # Add uncertainty to test points
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
    """Run experiment for a single species with multiple trials per n."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Species: {species_name} (model: {model_type})")
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

    experiments = []

    for n_pos in N_POSITIVE_VALUES:
        if n_pos >= n_total - 10:  # Need at least 10 test samples
            continue

        logger.info(f"\nn_positive = {n_pos} (+ {n_pos} negative = {n_pos * 2} total training)")

        trials = []
        aucs = []
        f1s = []
        precisions = []
        recalls = []

        for trial_idx in range(N_TRIALS):
            # Different seed for each trial
            trial_seed = BASE_SEED + trial_idx
            rng = np.random.default_rng(trial_seed)

            trial_result = run_single_trial(
                n_pos, all_occ_emb, valid_coords, mosaic, rng, model_type=model_type
            )
            trial_result["seed"] = trial_seed
            trials.append(trial_result)
            aucs.append(trial_result["auc"])
            f1s.append(trial_result["f1"])
            precisions.append(trial_result["precision"])
            recalls.append(trial_result["recall"])

        auc_mean = float(np.mean(aucs))
        auc_std = float(np.std(aucs))
        f1_mean = float(np.mean(f1s))
        f1_std = float(np.std(f1s))
        precision_mean = float(np.mean(precisions))
        precision_std = float(np.std(precisions))
        recall_mean = float(np.mean(recalls))
        recall_std = float(np.std(recalls))

        logger.info(f"  AUC: {auc_mean:.3f} ± {auc_std:.3f}")
        logger.info(f"  F1:  {f1_mean:.3f} ± {f1_std:.3f}")
        logger.info(f"  P/R: {precision_mean:.3f}/{recall_mean:.3f}")

        exp_data = {
            "n_positive": n_pos,
            "n_negative": n_pos,
            "n_trials": N_TRIALS,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "precision_mean": precision_mean,
            "precision_std": precision_std,
            "recall_mean": recall_mean,
            "recall_std": recall_std,
            "trials": trials,
        }
        experiments.append(exp_data)

    return {
        "species": species_name,
        "species_key": species_info["taxon_key"],
        "region": REGION,
        "model_type": model_type,
        "n_occurrences": n_total,
        "n_trials": N_TRIALS,
        "experiments": experiments,
    }


def run_all_experiments(model_type: ModelType = "logistic"):
    """Run experiments for all species."""
    logger.info("=" * 60)
    logger.info(f"Classifier Validation Experiment (model: {model_type})")
    logger.info(f"({N_TRIALS} trials per n value)")
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
        "base_seed": BASE_SEED,
        "n_trials": N_TRIALS,
        "n_positive_values": N_POSITIVE_VALUES,
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

            # Add to summary (just metrics, no coordinates)
            species_summary = {
                "species": species,
                "n_occurrences": result["n_occurrences"],
                "results": [
                    {
                        "n_positive": exp["n_positive"],
                        "auc_mean": exp["auc_mean"],
                        "auc_std": exp["auc_std"],
                        "f1_mean": exp["f1_mean"],
                        "f1_std": exp["f1_std"],
                        "precision_mean": exp["precision_mean"],
                        "precision_std": exp["precision_std"],
                        "recall_mean": exp["recall_mean"],
                        "recall_std": exp["recall_std"],
                    }
                    for exp in result["experiments"]
                ],
            }
            summary["species"].append(species_summary)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary: {summary_path}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info(f"SUMMARY ({model_type})")
    logger.info("=" * 60)
    logger.info(f"{'Species':<20} {'n_pos':>6} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    logger.info("-" * 62)
    for sp in summary["species"]:
        for r in sp["results"]:
            logger.info(
                f"{sp['species']:<20} {r['n_positive']:>6} "
                f"{r['auc_mean']:>8.3f} {r['f1_mean']:>8.3f} "
                f"{r['precision_mean']:>8.3f} {r['recall_mean']:>8.3f}"
            )
        logger.info("")

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
