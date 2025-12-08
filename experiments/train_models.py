#!/usr/bin/env python3
"""
Train and save classifier models for all experiment species.

Supports two model types:
1. LogisticRegression (logistic) - Simple, fast, interpretable
2. MLP with MC Dropout (mlp) - Provides uncertainty estimates

Models are saved to separate directories for comparison:
- models/logistic/{taxon_key}.pkl
- models/mlp/{taxon_key}.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from finder import get_species_info, fetch_occurrences, EmbeddingMosaic
from finder.methods import ClassifierMethod, MLPClassifierMethod
from finder.pipeline import REGIONS, sample_background

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"

# Same species list as experiment.py
SPECIES_LIST = [
    "Quercus robur",
    "Fraxinus excelsior",
    "Alnus glutinosa",
    "Crataegus monogyna",
    "Urtica dioica",
    "Salix caprea",
    "Aesculus hippocastanum",
]
REGION = "cambridge"
NEGATIVE_RATIO = 5
SEED = 42

ModelType = Literal["logistic", "mlp", "both"]


def train_and_save_model(
    species_name: str,
    mosaic: EmbeddingMosaic,
    model_type: ModelType = "both",
) -> bool:
    """Train classifier(s) for a species and save them."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {species_name}")
    logger.info("=" * 60)

    try:
        # Get species info
        species_info = get_species_info(species_name)
        taxon_key = species_info["taxon_key"]
        logger.info(f"  Taxon key: {taxon_key}")

        # Fetch occurrences
        bbox = REGIONS[REGION]["bbox"]
        occurrences = fetch_occurrences(taxon_key, bbox)
        logger.info(f"  Occurrences: {len(occurrences)}")

        if len(occurrences) < 5:
            logger.info("  Not enough occurrences, skipping")
            return False

        # Sample embeddings
        positive_embeddings, valid_coords = mosaic.sample_at_coords(occurrences)
        logger.info(f"  Valid embeddings: {len(positive_embeddings)}")

        if len(positive_embeddings) < 5:
            logger.info("  Not enough valid embeddings, skipping")
            return False

        # Sample background
        n_background = len(positive_embeddings) * NEGATIVE_RATIO
        negative_embeddings, _ = sample_background(
            mosaic, n_background, valid_coords, seed=SEED
        )
        logger.info(f"  Background samples: {len(negative_embeddings)}")

        # Train and save Logistic Regression
        if model_type in ("logistic", "both"):
            logger.info("  Training Logistic Regression...")
            logistic_classifier = ClassifierMethod()
            logistic_classifier.fit(positive_embeddings, negative_embeddings)

            logistic_dir = MODELS_DIR / "logistic"
            logistic_dir.mkdir(parents=True, exist_ok=True)
            logistic_path = logistic_dir / f"{taxon_key}.pkl"
            logistic_classifier.save(logistic_path)
            logger.info(f"  Saved: {logistic_path}")

        # Train and save MLP with MC Dropout
        if model_type in ("mlp", "both"):
            logger.info("  Training MLP with MC Dropout...")
            mlp_classifier = MLPClassifierMethod(
                hidden_dim=256,
                dropout_rate=0.3,
                learning_rate=1e-3,
                n_epochs=100,
                batch_size=64,
            )
            mlp_classifier.fit(positive_embeddings, negative_embeddings, verbose=True)

            mlp_dir = MODELS_DIR / "mlp"
            mlp_dir.mkdir(parents=True, exist_ok=True)
            mlp_path = mlp_dir / f"{taxon_key}.pt"
            mlp_classifier.save(mlp_path)
            logger.info(f"  Saved: {mlp_path}")

        return True

    except Exception as e:
        logger.error(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train classifier models for species habitat prediction"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic", "mlp", "both"],
        default="both",
        help="Type of model to train: logistic, mlp, or both (default: both)",
    )
    args = parser.parse_args()

    model_type: ModelType = args.model_type

    logger.info("=" * 60)
    logger.info(f"Training Classifier Models (type: {model_type})")
    logger.info("=" * 60)

    # Load mosaic once
    bbox = REGIONS[REGION]["bbox"]
    logger.info(f"\nLoading embedding mosaic for {REGION}...")
    mosaic = EmbeddingMosaic(CACHE_DIR, bbox)
    mosaic.load()
    logger.info(f"Mosaic shape: {mosaic.shape}")

    # Train models for each species
    success_count = 0
    for species in SPECIES_LIST:
        if train_and_save_model(species, mosaic, model_type=model_type):
            success_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {success_count}/{len(SPECIES_LIST)} models saved")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
