"""
models/xgboost_scorer.py
=========================
XGBoost supervised reference scorer.

Used as a ceiling to bound the best-possible performance on the 18-dim
feature vector when labels ARE available. Not part of the unsupervised
benchmark — included purely as a reference point in the paper figures.

Training strategy: 5-fold stratified cross-validation → out-of-fold
probability scores for the full dataset. This avoids train-set leakage
while still producing a score for every agent.

Args:
    n_estimators:  Boosting rounds. Default 300.
    max_depth:     Tree depth. Default 4.
    learning_rate: Step size. Default 0.05.
    subsample:     Row subsample ratio. Default 0.8.
    colsample:     Column subsample ratio. Default 0.8.
    n_splits:      CV folds. Default 5.
    random_state:  Seed. Default 42.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from models.anomaly_scorer import AnomalyScorer

log = logging.getLogger(__name__)


class XGBoostScorer(AnomalyScorer):
    """
    XGBoost supervised reference scorer (5-fold OOF).

    Requires labels at fit time. score() returns OOF probabilities
    produced during fit — call fit(X, y) not fit(X).

    NOTE: This scorer breaks the unsupervised contract (fit needs y).
    It is intentionally separated and labelled as "†" in all paper figures.
    """

    def __init__(
        self,
        n_estimators:  int   = 300,
        max_depth:     int   = 4,
        learning_rate: float = 0.05,
        subsample:     float = 0.8,
        colsample:     float = 0.8,
        n_splits:      int   = 5,
        random_state:  int   = 42,
    ):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self.colsample     = colsample
        self.n_splits      = n_splits
        self.random_state  = random_state

        self._oof_scores: np.ndarray | None = None
        self._models: list = []

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:  # type: ignore[override]
        """
        5-fold stratified cross-validation.
        Stores OOF probability scores for later retrieval via score().

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (1 = anomalous). REQUIRED for XGBoost.
        """
        if y is None:
            raise ValueError(
                "XGBoostScorer requires labels. "
                "Pass y=labels to fit(). "
                "This model is a supervised reference — not part of the unsupervised benchmark."
            )

        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "xgboost is required for XGBoostScorer. "
                "Install with: pip install xgboost"
            )

        log.info(
            f"XGBoost 5-fold CV on {X.shape[0]:,} × {X.shape[1]} features  "
            f"(pos={int(y.sum())}, neg={int((y==0).sum())})"
        )

        skf          = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                        random_state=self.random_state)
        oof          = np.zeros(len(y), dtype=np.float64)
        self._models = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va       = X[va_idx]

            # Handle class imbalance with scale_pos_weight
            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            spw   = n_neg / max(n_pos, 1)

            model = XGBClassifier(
                n_estimators  = self.n_estimators,
                max_depth     = self.max_depth,
                learning_rate = self.learning_rate,
                subsample     = self.subsample,
                colsample_bytree = self.colsample,
                scale_pos_weight = spw,
                eval_metric      = "aucpr",
                random_state     = self.random_state,
                verbosity        = 0,
            )
            model.fit(X_tr, y_tr)
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            self._models.append(model)
            log.info(f"  Fold {fold + 1}/{self.n_splits} done")

        self._oof_scores = oof
        log.info(
            f"XGBoost OOF complete. "
            f"Score range: [{oof.min():.4f}, {oof.max():.4f}]"
        )

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray = None) -> np.ndarray:  # type: ignore[override]
        """
        Return OOF scores computed during fit().
        X is ignored — pass None or the same X used in fit().
        """
        if self._oof_scores is None:
            raise RuntimeError("XGBoostScorer not fitted. Call fit(X, y) first.")
        return self._oof_scores

    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """
        Score new out-of-sample agents by averaging predictions
        across all fold models.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Probability scores (n_samples,).
        """
        if not self._models:
            raise RuntimeError("XGBoostScorer not fitted. Call fit(X, y) first.")
        probs = np.stack(
            [m.predict_proba(X)[:, 1] for m in self._models], axis=0
        )
        return probs.mean(axis=0)

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "xgboost_scorer.pkl", "wb") as f:
            pickle.dump({
                "models":     self._models,
                "oof_scores": self._oof_scores,
                "params": {
                    "n_estimators":  self.n_estimators,
                    "max_depth":     self.max_depth,
                    "learning_rate": self.learning_rate,
                    "subsample":     self.subsample,
                    "colsample":     self.colsample,
                    "n_splits":      self.n_splits,
                    "random_state":  self.random_state,
                }
            }, f)
        log.info(f"XGBoostScorer saved → {p / 'xgboost_scorer.pkl'}")

    def load(self, path: str) -> None:
        with open(Path(path) / "xgboost_scorer.pkl", "rb") as f:
            data = pickle.load(f)
        self._models     = data["models"]
        self._oof_scores = data["oof_scores"]
        for k, v in data["params"].items():
            setattr(self, k, v)
        log.info(f"XGBoostScorer loaded from {path}")