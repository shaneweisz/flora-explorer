"""
Prediction methods for finding candidate locations using classifiers.

Supports two approaches:
1. LogisticRegression - Simple, fast, interpretable
2. MLP with MC Dropout - Provides uncertainty estimates via Monte Carlo Dropout
"""

import pickle
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# PyTorch imports for MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


class MLPNetwork(nn.Module):
    """Simple MLP with dropout for MC Dropout uncertainty estimation."""

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


class MLPClassifierMethod:
    """
    MLP classifier with MC Dropout for habitat suitability prediction.

    Uses Monte Carlo Dropout to provide uncertainty estimates:
    - Mean of multiple forward passes = habitat suitability score
    - Std of multiple forward passes = model uncertainty (confidence)

    Lower uncertainty = higher confidence in the prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: Optional[MLPNetwork] = None
        self._scaler: Optional[StandardScaler] = None
        self._input_dim: int = 768

    def fit(
        self,
        positive_embeddings: np.ndarray,
        negative_embeddings: np.ndarray,
        verbose: bool = True,
    ) -> None:
        """
        Train MLP classifier on positive vs negative embeddings.

        Args:
            positive_embeddings: Embeddings at known occurrence locations
            negative_embeddings: Embeddings at random background locations
            verbose: Whether to show training progress
        """
        if len(positive_embeddings) < 2:
            raise ValueError("Need at least 2 positive samples")
        if len(negative_embeddings) < 2:
            raise ValueError("Need at least 2 negative samples")

        # Combine and create labels
        X = np.vstack([positive_embeddings, negative_embeddings])
        y = np.array([1.0] * len(positive_embeddings) + [0.0] * len(negative_embeddings))

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Store input dimension
        self._input_dim = X_scaled.shape[1]

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self._model = MLPNetwork(
            input_dim=self._input_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Training setup
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Training loop
        self._model.train()
        iterator = tqdm(range(self.n_epochs), desc="Training MLP") if verbose else range(self.n_epochs)

        for epoch in iterator:
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({"loss": epoch_loss / len(loader)})

    def predict(
        self,
        all_embeddings: np.ndarray,
        batch_size: int = 15000,
    ) -> np.ndarray:
        """Predict probability of positive class (no uncertainty)."""
        scores, _ = self.predict_with_uncertainty(all_embeddings, batch_size, n_samples=1)
        return scores

    def predict_with_uncertainty(
        self,
        all_embeddings: np.ndarray,
        batch_size: int = 15000,
        n_samples: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with MC Dropout uncertainty estimation.

        Args:
            all_embeddings: Embeddings to predict on
            batch_size: Batch size for prediction
            n_samples: Number of MC Dropout forward passes

        Returns:
            scores: Mean probability of positive class
            uncertainty: Standard deviation across forward passes (lower = more confident)
        """
        if self._model is None:
            raise ValueError("Must call fit() first")

        n_total = len(all_embeddings)
        all_preds = np.zeros((n_samples, n_total), dtype=np.float32)

        # Keep model in training mode to enable dropout
        self._model.train()

        with torch.no_grad():
            for sample_idx in range(n_samples):
                for i in range(0, n_total, batch_size):
                    end = min(i + batch_size, n_total)
                    batch = all_embeddings[i:end]

                    batch_scaled = self._scaler.transform(batch)
                    batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(self.device)

                    preds = self._model(batch_tensor).cpu().numpy()
                    all_preds[sample_idx, i:end] = preds

        # Compute mean and std across MC samples
        scores = all_preds.mean(axis=0)
        uncertainty = all_preds.std(axis=0)

        return scores, uncertainty

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model and scaler to a file."""
        if self._model is None or self._scaler is None:
            raise ValueError("Must call fit() before saving")

        path = Path(path)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "scaler": self._scaler,
            "input_dim": self._input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "MLPClassifierMethod":
        """Load a trained model from a file."""
        path = Path(path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        data = torch.load(path, map_location=device, weights_only=False)

        instance = cls(
            hidden_dim=data["hidden_dim"],
            dropout_rate=data["dropout_rate"],
            device=device,
        )
        instance._scaler = data["scaler"]
        instance._input_dim = data["input_dim"]

        instance._model = MLPNetwork(
            input_dim=data["input_dim"],
            hidden_dim=data["hidden_dim"],
            dropout_rate=data["dropout_rate"],
        ).to(device)
        instance._model.load_state_dict(data["model_state_dict"])

        return instance
