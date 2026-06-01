"""
models/anomaly_scorer.py
=========================
Abstract base class for all anomaly scoring models.

Every scorer operates on clean numpy arrays only — it knows nothing about
agent IDs, DataFrames, or imputation. That preprocessing happens upstream.

Contract
─────────
    fit(X)    → train on normal agents
    score(X)  → return anomaly scores (higher = more anomalous)
    save(path) / load(path) → persist and restore fitted state
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


class AnomalyScorer(ABC):
    """
    Abstract base for unsupervised anomaly scorers.

    All subclasses follow the same interface so they are interchangeable
    in the unified pipeline runner (main.py).
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the model on normal-agent feature matrix.

        Args:
            X: Clean scaled array of shape (n_samples, n_features).
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly scores. Higher = more anomalous.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1-D float array of length n_samples.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist fitted model to directory.

        Args:
            path: Local directory path.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load fitted model from directory.

        Args:
            path: Local directory path.
        """
        raise NotImplementedError