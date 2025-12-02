#!/usr/bin/env python3
"""
Train and save classifier models for all experiment species.

These saved models can be loaded quickly for real-time predictions
without needing to reload the full mosaic or retrain.
"""

import logging
from pathlib import Path

import numpy as np

from finder import get_species_info, fetch_occurrences, EmbeddingMosaic
from finder.methods import ClassifierMethod
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


def train_and_save_model(species_name: str, mosaic: EmbeddingMosaic) -> bool:
    """Train a classifier for a species and save it."""
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

        # Train classifier
        classifier = ClassifierMethod()
        classifier.fit(positive_embeddings, negative_embeddings)

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{taxon_key}.pkl"
        classifier.save(model_path)
        logger.info(f"  Saved: {model_path}")

        return True

    except Exception as e:
        logger.error(f"  Error: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("Training Classifier Models")
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
        if train_and_save_model(species, mosaic):
            success_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {success_count}/{len(SPECIES_LIST)} models saved")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
