"""
models/sklearn_scorers.py
==========================
Three sklearn-based anomaly scorers used in the paper benchmark:

    GMMScorer            — Gaussian Mixture Model (diagonal covariance)
    IsolationForestScorer — Isolation Forest
    KNNScorer            — K-Nearest Neighbours mean distance

All follow the AnomalyScorer interface: fit / score / save / load.
Higher score = more anomalous in all cases.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from models.anomaly_scorer import AnomalyScorer

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GMM
# ─────────────────────────────────────────────────────────────────────────────

class GMMScorer(AnomalyScorer):
    """
    Gaussian Mixture Model anomaly scorer.
    Score = negated log-likelihood (higher = more anomalous).

    Diagonal covariance outperforms full on axis-aligned mobility anomalies
    and is significantly faster at the scale of 10k–200k agents.

    Args:
        n_components:    Number of mixture components. Default 50.
        covariance_type: GMM covariance type. Default 'diag'.
        n_init:          Number of EM initialisations. Default 3.
        max_iter:        Max EM iterations per init. Default 200.
        reg_covar:       Covariance regularisation. Default 1e-4.
        random_state:    Seed. Default 42.
    """

    def __init__(
        self,
        n_components:    int   = 50,
        covariance_type: str   = "diag",
        n_init:          int   = 3,
        max_iter:        int   = 200,
        reg_covar:       float = 1e-4,
        random_state:    int   = 42,
    ):
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.n_init          = n_init
        self.max_iter        = max_iter
        self.reg_covar       = reg_covar
        self.random_state    = random_state
        self._model: GaussianMixture | None = None

    def fit(self, X: np.ndarray) -> None:
        log.info(
            f"Fitting GMM ({self.covariance_type}, k={self.n_components}) "
            f"on {X.shape[0]:,} × {X.shape[1]} features"
        )
        self._model = GaussianMixture(
            n_components    = self.n_components,
            covariance_type = self.covariance_type,
            n_init          = self.n_init,
            max_iter        = self.max_iter,
            reg_covar       = self.reg_covar,
            random_state    = self.random_state,
        )
        self._model.fit(X)
        log.info(
            f"GMM converged={self._model.converged_}, "
            f"iterations={self._model.n_iter_}"
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GMMScorer not fitted. Call fit() first.")
        return -self._model.score_samples(X)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "gmm_scorer.pkl", "wb") as f:
            pickle.dump(self._model, f)
        log.info(f"GMMScorer saved → {p / 'gmm_scorer.pkl'}")

    def load(self, path: str) -> None:
        with open(Path(path) / "gmm_scorer.pkl", "rb") as f:
            self._model = pickle.load(f)
        log.info(f"GMMScorer loaded from {path}")


# Keep the old name as an alias so existing imports don't break
GMMDiagCovScorer = GMMScorer


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────

class IsolationForestScorer(AnomalyScorer):
    """
    Isolation Forest anomaly scorer.
    Score = negated decision_function (higher = more anomalous).

    Args:
        n_estimators:  Number of trees. Default 200.
        contamination: Expected fraction of anomalies. Default 'auto'.
        random_state:  Seed. Default 42.
        n_jobs:        Parallel jobs. Default -1 (all cores).
    """

    def __init__(
        self,
        n_estimators:  int   = 200,
        contamination        = "auto",
        random_state:  int   = 42,
        n_jobs:        int   = -1,
    ):
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.random_state  = random_state
        self.n_jobs        = n_jobs
        self._model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> None:
        log.info(
            f"Fitting IsolationForest (n_est={self.n_estimators}) "
            f"on {X.shape[0]:,} × {X.shape[1]} features"
        )
        self._model = IsolationForest(
            n_estimators  = self.n_estimators,
            contamination = self.contamination,
            random_state  = self.random_state,
            n_jobs        = self.n_jobs,
        )
        self._model.fit(X)
        log.info("IsolationForest fit complete.")

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsolationForestScorer not fitted. Call fit() first.")
        return -self._model.decision_function(X)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "iforest_scorer.pkl", "wb") as f:
            pickle.dump(self._model, f)
        log.info(f"IsolationForestScorer saved → {p / 'iforest_scorer.pkl'}")

    def load(self, path: str) -> None:
        with open(Path(path) / "iforest_scorer.pkl", "rb") as f:
            self._model = pickle.load(f)
        log.info(f"IsolationForestScorer loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# KNN
# ─────────────────────────────────────────────────────────────────────────────

class KNNScorer(AnomalyScorer):
    """
    K-Nearest Neighbours anomaly scorer.
    Score = mean distance to k nearest training agents (higher = more anomalous).

    Args:
        k:      Number of neighbours. Default 10.
        metric: Distance metric. Default 'euclidean'.
        n_jobs: Parallel jobs. Default -1.
    """

    def __init__(
        self,
        k:      int = 10,
        metric: str = "euclidean",
        n_jobs: int = -1,
    ):
        self.k      = k
        self.metric = metric
        self.n_jobs = n_jobs
        self._knn: NearestNeighbors | None = None

    def fit(self, X: np.ndarray) -> None:
        log.info(
            f"Fitting KNNScorer (k={self.k}) "
            f"on {X.shape[0]:,} × {X.shape[1]} features"
        )
        self._knn = NearestNeighbors(
            n_neighbors = self.k,
            metric      = self.metric,
            n_jobs      = self.n_jobs,
        )
        self._knn.fit(X)
        log.info("KNNScorer fit complete.")

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._knn is None:
            raise RuntimeError("KNNScorer not fitted. Call fit() first.")
        distances, _ = self._knn.kneighbors(X)
        scores = distances.mean(axis=1)
        log.info(
            f"KNN scored {X.shape[0]:,} agents. "
            f"Range: [{scores.min():.4f}, {scores.max():.4f}]"
        )
        return scores

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "knn_scorer.pkl", "wb") as f:
            pickle.dump({"knn": self._knn, "k": self.k, "metric": self.metric}, f)
        log.info(f"KNNScorer saved → {p / 'knn_scorer.pkl'}")

    def load(self, path: str) -> None:
        with open(Path(path) / "knn_scorer.pkl", "rb") as f:
            data = pickle.load(f)
        self._knn   = data["knn"]
        self.k      = data["k"]
        self.metric = data["metric"]
        log.info(f"KNNScorer loaded from {path}")