"""
models/lstmad_scorer.py
========================
LSTM-AD — Long Short-Term Memory Anomaly Detection.
Malhotra et al., ESANN 2015.

Trains stacked LSTM to predict the next T_pred timesteps.
Prediction errors fitted to a multivariate Gaussian.
Anomaly score = negative log-likelihood under that Gaussian.
Higher score = more anomalous.
"""

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from models.anomaly_scorer import AnomalyScorer

log = logging.getLogger(__name__)


class LSTMADScorer(AnomalyScorer):
    """
    LSTM-AD anomaly scorer on stay sequences.

    Args:
        hidden:       LSTM hidden size. Default 32.
        n_layers:     Stacked LSTM layers. Default 2.
        t_pred:       Steps ahead to predict. Default 3.
        epochs:       Training epochs. Default 60.
        lr:           Learning rate. Default 1e-3.
        batch_size:   Batch size. Default 64.
        max_len:      Max sequence length. Default 60.
        random_state: Seed. Default 42.
    """

    def __init__(self, hidden=32, n_layers=2, t_pred=3,
                 epochs=60, lr=1e-3, batch_size=64,
                 max_len=60, random_state=42):
        self.hidden       = hidden
        self.n_layers     = n_layers
        self.t_pred       = t_pred
        self.epochs       = epochs
        self.lr           = lr
        self.batch_size   = batch_size
        self.max_len      = max_len
        self.random_state = random_state
        self._model       = None
        self._mu          = None   # Gaussian mean of training errors
        self._cov_inv     = None   # Gaussian precision matrix
        self._input_size  = None
        self._device      = None

    def _build_model(self, input_size):
        import torch.nn as nn

        class PredLSTM(nn.Module):
            def __init__(self, input_size, hidden, n_layers, t_pred):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden, n_layers,
                    batch_first=True, dropout=0.2,
                )
                # Predict t_pred steps ahead simultaneously
                self.fc = nn.Linear(hidden, input_size * t_pred)
                self.t_pred = t_pred
                self.input_size = input_size

            def forward(self, x):
                # x: (B, L, F)
                out, _ = self.lstm(x)          # (B, L, H)
                pred   = self.fc(out)          # (B, L, F*t_pred)
                B, L, _ = pred.shape
                return pred.view(B, L,
                                 self.t_pred,
                                 self.input_size)  # (B, L, T, F)

        return PredLSTM(input_size, self.hidden,
                        self.n_layers, self.t_pred)

    def fit(self, sequences: List[np.ndarray]) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        self._input_size = sequences[0].shape[1]
        self._device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        log.info(
            f"Training LSTM-AD on {len(sequences):,} agents, "
            f"input_size={self._input_size}, "
            f"t_pred={self.t_pred}, "
            f"epochs={self.epochs}, "
            f"device={self._device}"
        )

        # Pad sequences
        N = len(sequences)
        X = np.zeros((N, self.max_len, self._input_size), dtype=np.float32)
        for i, seq in enumerate(sequences):
            L = min(len(seq), self.max_len)
            X[i, :L, :] = seq[-L:]

        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model  = self._build_model(self._input_size).to(self._device)
        opt    = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (x,) in loader:
                x = x.to(self._device)        # (B, L, F)
                L = x.shape[1]

                if L <= self.t_pred:
                    continue

                # Input: x[:, :-t_pred, :]
                # Target: x[:, t:t+t_pred, :] for each t
                inp  = x[:, :-self.t_pred, :]  # (B, L-T, F)
                preds = model(inp)             # (B, L-T, T, F)

                # Build multi-step targets
                targets = torch.stack([
                    x[:, t:L - self.t_pred + t, :]
                    for t in range(1, self.t_pred + 1)
                ], dim=2)                      # (B, L-T, T, F)

                loss = loss_fn(preds, targets)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                opt.step()
                epoch_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    f"  Epoch {epoch:>3}/{self.epochs}  "
                    f"loss={epoch_loss / len(loader):.5f}"
                )

        self._model = model

        # Fit Gaussian to training errors
        log.info("Fitting Gaussian to training prediction errors...")
        errors = self._get_errors(X)  # (N, error_dim)
        self._mu = errors.mean(axis=0)
        cov = np.cov(errors.T) + 1e-6 * np.eye(errors.shape[1])
        self._cov_inv = np.linalg.pinv(cov)
        log.info("LSTM-AD training complete.")

    def _get_errors(self, X: np.ndarray) -> np.ndarray:
        """Get flattened prediction error vectors for each agent."""
        import torch
        self._model.eval()
        tensor = torch.tensor(X, dtype=torch.float32)
        all_errors = []

        with torch.no_grad():
            for i in range(0, len(tensor), self.batch_size):
                x    = tensor[i: i + self.batch_size].to(self._device)
                L    = x.shape[1]
                inp  = x[:, :-self.t_pred, :]
                pred = self._model(inp)        # (B, L-T, T, F)

                targets = torch.stack([
                    x[:, t:L - self.t_pred + t, :]
                    for t in range(1, self.t_pred + 1)
                ], dim=2)

                err = (pred - targets) ** 2    # (B, L-T, T, F)
                # Mean over time and steps → (B, F)
                err_mean = err.mean(dim=(1, 2)).cpu().numpy()
                all_errors.append(err_mean)

        return np.concatenate(all_errors)      # (N, F)

    def score(self, sequences: List[np.ndarray]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LSTMADScorer not fitted.")

        N = len(sequences)
        X = np.zeros((N, self.max_len, self._input_size), dtype=np.float32)
        for i, seq in enumerate(sequences):
            L = min(len(seq), self.max_len)
            X[i, :L, :] = seq[-L:]

        errors = self._get_errors(X)           # (N, F)

        # Mahalanobis distance under fitted Gaussian
        diff   = errors - self._mu             # (N, F)
        scores = np.array([
            float(d @ self._cov_inv @ d)
            for d in diff
        ])
        log.info(
            f"LSTM-AD scored {N:,} agents. "
            f"Range: [{scores.min():.4f}, {scores.max():.4f}]"
        )
        return scores

    def save(self, path: str) -> None:
        import torch
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), d / "lstmad_state_dict.pt")
        with open(d / "lstmad_params.pkl", "wb") as fh:
            pickle.dump({
                "hidden": self.hidden, "n_layers": self.n_layers,
                "t_pred": self.t_pred, "epochs": self.epochs,
                "lr": self.lr, "max_len": self.max_len,
                "input_size": self._input_size,
                "mu": self._mu, "cov_inv": self._cov_inv,
            }, fh)
        log.info(f"LSTMADScorer saved to {d}")

    def load(self, path: str) -> None:
        import torch
        d = Path(path)
        with open(d / "lstmad_params.pkl", "rb") as fh:
            p = pickle.load(fh)
        self.hidden, self.n_layers = p["hidden"], p["n_layers"]
        self.t_pred, self.epochs   = p["t_pred"], p["epochs"]
        self.lr, self.max_len      = p["lr"],     p["max_len"]
        self._input_size = p["input_size"]
        self._mu, self._cov_inv = p["mu"], p["cov_inv"]
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = self._build_model(self._input_size).to(self._device)
        self._model.load_state_dict(
            torch.load(d / "lstmad_state_dict.pt",
                       map_location=self._device)
        )
        log.info(f"LSTMADScorer loaded from {d}")