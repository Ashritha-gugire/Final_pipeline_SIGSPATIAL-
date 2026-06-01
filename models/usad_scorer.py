"""
models/usad_scorer.py
======================
USAD — UnSupervised Anomaly Detection on Multivariate Time Series.
Audibert et al., KDD 2020.

Architecture:
  Shared encoder E + two decoders D1, D2.
  AE1 = (E, D1),  AE2 = (E, D2).

Two-phase training (combined per epoch n):
  L_AE1 = (1/n)||W - AE1(W)||^2 + (1 - 1/n)||W - AE2(AE1(W))||^2
  L_AE2 = (1/n)||W - AE2(W)||^2 - (1 - 1/n)||W - AE2(AE1(W))||^2

Anomaly score:
  A(W) = alpha * ||W - AE1(W)||^2 + beta * ||W - AE2(AE1(W))||^2

Falls back to sklearn MLPRegressor autoencoder if PyTorch is
unavailable (e.g. Python 3.14 on Windows).
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from models.anomaly_scorer import AnomalyScorer

log = logging.getLogger(__name__)

# Probe torch once at import time
_TORCH_AVAILABLE = False
try:
    import torch as _torch_probe
    _TORCH_AVAILABLE = True
    del _torch_probe
except (ImportError, OSError):
    log.warning(
        "torch could not be loaded (Python version or DLL incompatibility). "
        "USADScorer will use sklearn MLP fallback."
    )


class USADScorer(AnomalyScorer):
    """
    USAD anomaly scorer for agent-level feature vectors.

    PyTorch backend: full adversarial two-phase USAD training.
    sklearn fallback: MLPRegressor autoencoder with reconstruction MSE score.

    Args:
        latent_dim:   Encoder bottleneck size. Default 16.
        epochs:       Training epochs. Default 50.
        lr:           Learning rate (PyTorch only). Default 1e-3.
        batch_size:   Batch size. Default 256.
        alpha:        Weight on AE1 reconstruction error. Default 0.5.
        beta:         Weight on AE2(AE1) reconstruction error. Default 0.5.
        random_state: Seed. Default 42.
    """

    def __init__(
        self,
        latent_dim:   int   = 16,
        epochs:       int   = 50,
        lr:           float = 1e-3,
        batch_size:   int   = 256,
        alpha:        float = 0.5,
        beta:         float = 0.5,
        random_state: int   = 42,
    ):
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.lr           = lr
        self.batch_size   = batch_size
        self.alpha        = alpha
        self.beta         = beta
        self.random_state = random_state

        self._use_torch = _TORCH_AVAILABLE
        self._enc       = None
        self._dec1      = None
        self._dec2      = None
        self._mlp       = None
        self._device    = None
        self._input_dim = None
        self._x_min     = None
        self._x_max     = None

    # ── architecture ──────────────────────────────────────────────────────────

    def _build_encoder(self, input_dim: int, latent_dim: int):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2), nn.ReLU(),
            nn.Linear(input_dim * 2, latent_dim), nn.ReLU(),
        )

    def _build_decoder(self, input_dim: int, latent_dim: int):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(latent_dim, input_dim * 2), nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),  nn.Sigmoid(),
        )

    # ── normalisation ─────────────────────────────────────────────────────────

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        denom = self._x_max - self._x_min
        denom = np.where(denom == 0, 1.0, denom)
        return np.clip(
            (X - self._x_min) / denom, 0.0, 1.0
        ).astype(np.float32)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> None:
        self._input_dim = X.shape[1]
        self._x_min     = X.min(axis=0)
        self._x_max     = X.max(axis=0)

        log.info(
            f"Training USAD on {X.shape[0]:,} agents, "
            f"input_dim={self._input_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"epochs={self.epochs}, "
            f"backend={'pytorch' if self._use_torch else 'sklearn'}"
        )

        if self._use_torch:
            try:
                self._fit_pytorch(X)
            except (ImportError, OSError, RuntimeError) as exc:
                log.warning(
                    f"PyTorch failed ({type(exc).__name__}: {exc}). "
                    "Switching to sklearn fallback."
                )
                self._use_torch = False
                self._fit_sklearn(X)
        else:
            self._fit_sklearn(X)

    def _fit_pytorch(self, X: np.ndarray) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info(f"  Device: {self._device}")

        enc  = self._build_encoder(self._input_dim, self.latent_dim).to(self._device)
        dec1 = self._build_decoder(self._input_dim, self.latent_dim).to(self._device)
        dec2 = self._build_decoder(self._input_dim, self.latent_dim).to(self._device)

        opt1 = torch.optim.Adam(
            list(enc.parameters()) + list(dec1.parameters()), lr=self.lr
        )
        opt2 = torch.optim.Adam(
            list(enc.parameters()) + list(dec2.parameters()), lr=self.lr
        )

        X_norm = self._normalise(X)
        tensor = torch.tensor(X_norm, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for n in range(1, self.epochs + 1):
            epoch_l1 = epoch_l2 = 0.0
            for (w,) in loader:
                w = w.to(self._device)
                # AE1 loss
                z   = enc(w)
                w1  = dec1(z)
                w12 = dec2(enc(w1.detach()))

                l1 = (1.0 / n)       * ((w - w1)  ** 2).mean() \
                    + (1.0 - 1.0 / n) * ((w - w12) ** 2).mean()

            opt1.zero_grad()
            l1.backward()
            opt1.step()

    # AE2 loss — fresh forward pass, no shared graph with l1
            with torch.no_grad():
                z_  = enc(w)
                w1_ = dec1(z_)
            w2        = dec2(enc(w))
            w12_fresh = dec2(enc(w1_.detach()))

            l2 = (1.0 / n)       * ((w - w2)        ** 2).mean() \
                - (1.0 - 1.0 / n) * ((w - w12_fresh) ** 2).mean()

            opt2.zero_grad()
            l2.backward()
            opt2.step()

            epoch_l1 += l1.item()
            epoch_l2 += l2.item()
            if n % 10 == 0 or n == 1:
                log.info(
                    f"  Epoch {n:>3}/{self.epochs}  "
                    f"L_AE1={epoch_l1 / len(loader):.5f}  "
                    f"L_AE2={epoch_l2 / len(loader):.5f}"
                )

        self._enc  = enc
        self._dec1 = dec1
        self._dec2 = dec2
        log.info("USAD (PyTorch) training complete.")

    def _fit_sklearn(self, X: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor

        h = self.latent_dim
        log.info(
            f"  MLP autoencoder: "
            f"hidden=({h * 4}, {h * 2}, {h}, {h * 2}, {h * 4}), "
            f"max_iter={self.epochs * 5}"
        )
        X_norm    = self._normalise(X)
        self._mlp = MLPRegressor(
            hidden_layer_sizes=(h * 4, h * 2, h, h * 2, h * 4),
            activation="relu",
            max_iter=self.epochs * 5,
            random_state=self.random_state,
            early_stopping=True,
            verbose=False,
        )
        self._mlp.fit(X_norm, X_norm)
        log.info("USAD (sklearn) training complete.")

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._use_torch and self._enc is None:
            raise RuntimeError("USADScorer not fitted. Call fit() first.")
        if not self._use_torch and self._mlp is None:
            raise RuntimeError("USADScorer not fitted. Call fit() first.")
        if self._use_torch:
            return self._score_pytorch(X)
        return self._score_sklearn(X)

    def _score_pytorch(self, X: np.ndarray) -> np.ndarray:
        import torch
        self._enc.eval()
        self._dec1.eval()
        self._dec2.eval()

        X_norm = self._normalise(X)
        tensor = torch.tensor(X_norm, dtype=torch.float32).to(self._device)
        scores = []

        with torch.no_grad():
            for i in range(0, len(tensor), self.batch_size):
                w   = tensor[i: i + self.batch_size]
                z   = self._enc(w)
                w1  = self._dec1(z)
                w12 = self._dec2(self._enc(w1))
                e1  = ((w - w1)  ** 2).mean(dim=1)
                e2  = ((w - w12) ** 2).mean(dim=1)
                scores.append(
                    (self.alpha * e1 + self.beta * e2).cpu().numpy()
                )

        out = np.concatenate(scores)
        log.info(
            f"USAD scored {X.shape[0]:,} agents. "
            f"Range: [{out.min():.4f}, {out.max():.4f}]"
        )
        return out

    def _score_sklearn(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalise(X)
        recon  = self._mlp.predict(X_norm)
        out    = np.mean((X_norm - recon) ** 2, axis=1)
        log.info(
            f"USAD scored {X.shape[0]:,} agents. "
            f"Range: [{out.min():.4f}, {out.max():.4f}]"
        )
        return out

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)

        cfg = dict(
            latent_dim=self.latent_dim,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            alpha=self.alpha,
            beta=self.beta,
            random_state=self.random_state,
            input_dim=self._input_dim,
            use_torch=self._use_torch,
            x_min=self._x_min,
            x_max=self._x_max,
        )
        with open(d / "usad_config.pkl", "wb") as fh:
            pickle.dump(cfg, fh)

        if self._use_torch:
            import torch
            torch.save(self._enc.state_dict(),  d / "usad_enc.pt")
            torch.save(self._dec1.state_dict(), d / "usad_dec1.pt")
            torch.save(self._dec2.state_dict(), d / "usad_dec2.pt")
        else:
            with open(d / "usad_mlp.pkl", "wb") as fh:
                pickle.dump(self._mlp, fh)

        log.info(f"USADScorer saved to {d}")

    def load(self, path: str) -> None:
        d = Path(path)
        with open(d / "usad_config.pkl", "rb") as fh:
            cfg = pickle.load(fh)

        self.latent_dim   = cfg["latent_dim"]
        self.epochs       = cfg["epochs"]
        self.lr           = cfg["lr"]
        self.batch_size   = cfg["batch_size"]
        self.alpha        = cfg["alpha"]
        self.beta         = cfg["beta"]
        self.random_state = cfg["random_state"]
        self._input_dim   = cfg["input_dim"]
        self._use_torch   = cfg["use_torch"]
        self._x_min       = cfg["x_min"]
        self._x_max       = cfg["x_max"]

        if self._use_torch:
            import torch
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._enc  = self._build_encoder(
                self._input_dim, self.latent_dim
            ).to(self._device)
            self._dec1 = self._build_decoder(
                self._input_dim, self.latent_dim
            ).to(self._device)
            self._dec2 = self._build_decoder(
                self._input_dim, self.latent_dim
            ).to(self._device)
            self._enc.load_state_dict(
                torch.load(d / "usad_enc.pt",  map_location=self._device)
            )
            self._dec1.load_state_dict(
                torch.load(d / "usad_dec1.pt", map_location=self._device)
            )
            self._dec2.load_state_dict(
                torch.load(d / "usad_dec2.pt", map_location=self._device)
            )
        else:
            with open(d / "usad_mlp.pkl", "rb") as fh:
                self._mlp = pickle.load(fh)

        log.info(f"USADScorer loaded from {d}")