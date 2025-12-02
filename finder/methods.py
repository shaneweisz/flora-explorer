"""
Prediction method for finding candidate locations using habitat similarity.
"""

from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class SimilarityMethod:
    """
    Similarity-based approach using distance to centroid.

    Computes the centroid of positive embeddings and scores each pixel
    by its cosine similarity to the centroid. Works with any number of
    samples, including just 1.
    """

    def __init__(self):
        self._centroid: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None

    def fit(self, positive_embeddings: np.ndarray) -> None:
        """Compute centroid of positive embeddings."""
        if len(positive_embeddings) == 0:
            raise ValueError("Need at least 1 positive sample")

        # Normalize embeddings
        self._scaler = StandardScaler()
        normalized = self._scaler.fit_transform(positive_embeddings)

        # Compute and normalize centroid for cosine similarity
        centroid = normalized.mean(axis=0)
        self._centroid = centroid / np.linalg.norm(centroid)

    def predict(
        self,
        all_embeddings: np.ndarray,
        batch_size: int = 15000
    ) -> np.ndarray:
        """Compute similarity scores for all embeddings."""
        if self._centroid is None:
            raise ValueError("Must call fit() first")

        n_samples = len(all_embeddings)
        scores = np.zeros(n_samples, dtype=np.float32)

        for i in tqdm(range(0, n_samples, batch_size), desc="Computing similarity"):
            end = min(i + batch_size, n_samples)
            batch = all_embeddings[i:end]

            # Normalize batch
            batch_normalized = self._scaler.transform(batch)
            norms = np.linalg.norm(batch_normalized, axis=1, keepdims=True)
            norms[norms == 0] = 1
            batch_normalized = batch_normalized / norms

            # Cosine similarity, converted from [-1, 1] to [0, 1]
            similarities = batch_normalized @ self._centroid
            scores[i:end] = (similarities + 1) / 2

        return scores
