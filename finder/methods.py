"""
Prediction method for finding candidate locations using a classifier.
"""

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class ClassifierMethod:
    """
    Logistic regression classifier for habitat suitability.

    Trains on positive (occurrence) embeddings vs random background
    embeddings to learn a decision boundary. Outperforms similarity-based
    approaches, especially with more training samples.
    """

    def __init__(self):
        self._model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None

    def fit(
        self,
        positive_embeddings: np.ndarray,
        negative_embeddings: np.ndarray,
    ) -> None:
        """
        Train classifier on positive vs negative embeddings.

        Args:
            positive_embeddings: Embeddings at known occurrence locations
            negative_embeddings: Embeddings at random background locations
        """
        if len(positive_embeddings) < 2:
            raise ValueError("Need at least 2 positive samples")
        if len(negative_embeddings) < 2:
            raise ValueError("Need at least 2 negative samples")

        # Combine and create labels
        X = np.vstack([positive_embeddings, negative_embeddings])
        y = np.array([1] * len(positive_embeddings) + [0] * len(negative_embeddings))

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train classifier
        self._model = LogisticRegression(max_iter=1000, solver="lbfgs")
        self._model.fit(X_scaled, y)

    def predict(
        self,
        all_embeddings: np.ndarray,
        batch_size: int = 15000
    ) -> np.ndarray:
        """Predict probability of positive class for all embeddings."""
        if self._model is None:
            raise ValueError("Must call fit() first")

        n_samples = len(all_embeddings)
        scores = np.zeros(n_samples, dtype=np.float32)

        for i in tqdm(range(0, n_samples, batch_size), desc="Classifying"):
            end = min(i + batch_size, n_samples)
            batch = all_embeddings[i:end]

            batch_scaled = self._scaler.transform(batch)
            probs = self._model.predict_proba(batch_scaled)
            scores[i:end] = probs[:, 1]

        return scores

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model and scaler to a file."""
        if self._model is None or self._scaler is None:
            raise ValueError("Must call fit() before saving")

        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "scaler": self._scaler,
            }, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClassifierMethod":
        """Load a trained model from a file."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance._model = data["model"]
        instance._scaler = data["scaler"]
        return instance
