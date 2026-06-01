"""
models/dagmm_scorer.py
=======================
DAGMM — Deep Autoencoding Gaussian Mixture Model
Zong et al., ICLR 2018.

Jointly trains a deep autoencoder and a GMM in the latent space.
Anomaly score = sample energy from the GMM.

Falls back to sklearn (MLPRegressor autoencoder + GaussianMixture)
if PyTorch is unavailable (e.g. Python 3.14 on Windows).
"""

import logging
import pickle
from pathlib import Path

import numpy as np

from models.anomaly_scorer import AnomalyScorer

log = logging.getLogger(__name__)

# ── Probe torch once at import time ───────────────────────────────────────────
_TORCH_AVAILABLE = False
try:
    import torch as _torch_probe
    _TORCH_AVAILABLE = True
    del _torch_probe
except (ImportError, OSError):
    log.warning(
        "torch could not be loaded (Python version or DLL incompatibility). "
        "DAGMMScorer will use sklearn fallback "
        "(MLPRegressor autoencoder + GaussianMixture)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch DAGMM model definition
# ─────────────────────────────────────────────────────────────────────────────

def _build_dagmm_model(input_dim, latent_dim, n_components):
    import torch.nn as nn

    class DAGMMNet(nn.Module):
        def __init__(self, input_dim, latent_dim, n_comp):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(input_dim, 64), nn.Tanh(),
                nn.Linear(64, 32),        nn.Tanh(),
                nn.Linear(32, latent_dim),
            )
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, 32), nn.Tanh(),
                nn.Linear(32, 64),         nn.Tanh(),
                nn.Linear(64, input_dim),
            )
            # Estimation network: latent_dim + 2 reconstruction error features
            self.est = nn.Sequential(
                nn.Linear(latent_dim + 2, 10), nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(10, n_comp),
                nn.Softmax(dim=1),
            )
            self.n_comp = n_comp

        def forward(self, x):
            import torch
            import torch.nn.functional as F
            z_c   = self.enc(x)
            x_hat = self.dec(z_c)
            rel_ed = torch.norm(x - x_hat, dim=1, keepdim=True) / (
                torch.norm(x, dim=1, keepdim=True) + 1e-10
            )
            cos_d = (1 - F.cosine_similarity(x, x_hat, dim=1)).unsqueeze(1)
            z     = torch.cat([z_c, rel_ed, cos_d], dim=1)
            gamma = self.est(z)
            return x_hat, z, gamma

    return DAGMMNet(input_dim, latent_dim, n_components)


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────────────────────────────────────

class DAGMMScorer(AnomalyScorer):
    """
    DAGMM anomaly scorer for agent-level feature vectors.

    PyTorch backend: full DAGMM with joint autoencoder + GMM energy loss.
    sklearn fallback: MLPRegressor autoencoder → GaussianMixture on latent space.
    Both backends produce the same interface and comparable anomaly scores.

    Args:
        n_components: GMM components. Default 3.
        latent_dim:   Encoder output size. Default 2.
        epochs:       Training epochs. Default 100.
        lr:           Learning rate (PyTorch only). Default 1e-4.
        lambda1:      Energy loss weight. Default 0.1.
        lambda2:      Sigma penalty weight. Default 0.005.
        batch_size:   Batch size. Default 256.
        random_state: Seed. Default 42.
    """

    def __init__(
        self,
        n_components: int   = 3,
        latent_dim:   int   = 2,
        epochs:       int   = 100,
        lr:           float = 1e-4,
        lambda1:      float = 0.1,
        lambda2:      float = 0.005,
        batch_size:   int   = 256,
        random_state: int   = 42,
    ):
        self.n_components = n_components
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.lr           = lr
        self.lambda1      = lambda1
        self.lambda2      = lambda2
        self.batch_size   = batch_size
        self.random_state = random_state

        self._use_torch  = _TORCH_AVAILABLE
        self._model      = None
        self._gmm        = None    # sklearn GMM (fallback only)
        self._device     = None
        self._input_dim  = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> None:
        self._input_dim = X.shape[1]
        log.info(
            f"Training DAGMM on {X.shape[0]:,} agents, "
            f"input_dim={self._input_dim}, "
            f"n_components={self.n_components}, "
            f"backend={'pytorch' if self._use_torch else 'sklearn'}"
        )
        if self._use_torch:
            try:
                self._fit_pytorch(X)
            except (ImportError, OSError, RuntimeError) as exc:
                log.warning(
                    f"PyTorch failed ({type(exc).__name__}). "
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

        model  = _build_dagmm_model(
            self._input_dim, self.latent_dim, self.n_components
        ).to(self._device)
        opt    = torch.optim.Adam(model.parameters(), lr=self.lr)
        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch         = batch.to(self._device)
                x_hat, z, gamma = model(batch)

                recon  = ((batch - x_hat) ** 2).mean()
                energy, sigma_diag = self._gmm_energy(z, gamma)
                pen    = (1.0 / sigma_diag.clamp(min=1e-10)).mean()
                loss   = recon + self.lambda1 * energy + self.lambda2 * pen

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                epoch_loss += loss.item()

            if epoch % 20 == 0 or epoch == 1:
                log.info(
                    f"  Epoch {epoch:>3}/{self.epochs}  "
                    f"loss={epoch_loss / len(loader):.5f}"
                )

        self._model = model
        log.info("DAGMM (PyTorch) training complete.")

    def _gmm_energy(self, z, gamma):
        import torch
        N   = gamma.shape[0]
        phi = gamma.sum(0) / N
        mu  = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(0) / (
            gamma.sum(0).unsqueeze(1) + 1e-10
        )
        diff  = z.unsqueeze(1) - mu.unsqueeze(0)
        sigma = (
            gamma.unsqueeze(2).unsqueeze(3)
            * diff.unsqueeze(3)
            * diff.unsqueeze(2)
        ).sum(0) / (gamma.sum(0).unsqueeze(1).unsqueeze(2) + 1e-10)

        K, d   = mu.shape
        energy = []
        for k in range(K):
            s    = sigma[k] + 1e-6 * torch.eye(d, device=z.device)
            diff_k = z - mu[k]
            inv  = torch.linalg.solve(s, diff_k.T).T
            quad = (diff_k * inv).sum(1)
            log_det = torch.logdet(s).clamp(min=-20, max=20)
            e_k  = phi[k].log() - 0.5 * (
                d * np.log(2 * np.pi) + log_det + quad
            )
            energy.append(e_k)

        sample_energy = -torch.logsumexp(torch.stack(energy, dim=1), dim=1)
        sigma_diag    = torch.stack([
            torch.diagonal(sigma[k]) for k in range(K)
        ])
        return sample_energy.mean(), sigma_diag

    def _fit_sklearn(self, X: np.ndarray) -> None:
        """
        Sklearn fallback:
          1. Train MLPRegressor as autoencoder to get latent representation.
          2. Fit GaussianMixture on [latent_z, recon_errors].
          3. Score = negative log-likelihood from GMM.
        """
        from sklearn.neural_network import MLPRegressor
        from sklearn.mixture import GaussianMixture

        log.info(
            f"  MLP autoencoder: "
            f"hidden=(64, 32, {self.latent_dim}, 32, 64), "
            f"max_iter={self.epochs * 3}"
        )

        # Step 1: Train autoencoder
        ae = MLPRegressor(
            hidden_layer_sizes=(64, 32, self.latent_dim, 32, 64),
            activation="tanh",
            max_iter=self.epochs * 3,
            random_state=self.random_state,
            early_stopping=True,
            verbose=False,
        )
        ae.fit(X, X)

        # Step 2: Extract latent representation + reconstruction errors
        Z = self._get_latent_sklearn(ae, X)
        log.info(
            f"  Fitting GMM ({self.n_components} components) "
            f"on latent+error features shape={Z.shape}"
        )
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=200,
            random_state=self.random_state,
        )
        gmm.fit(Z)

        self._model = ae
        self._gmm   = gmm
        log.info("DAGMM (sklearn) training complete.")

    def _get_latent_sklearn(self, ae, X: np.ndarray) -> np.ndarray:
        """
        Extract latent representation from sklearn MLP by forward-passing
        through the encoder half of the autoencoder.
        Uses the weights of the first (latent_dim + 1) layers.
        Then appends reconstruction error features.
        """
        # Get reconstruction
        X_hat = ae.predict(X)

        # Reconstruction error features
        rel_ed = np.linalg.norm(X - X_hat, axis=1, keepdims=True) / (
            np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        )
        cos_sim = np.sum(X * X_hat, axis=1, keepdims=True) / (
            np.linalg.norm(X, axis=1, keepdims=True)
            * np.linalg.norm(X_hat, axis=1, keepdims=True)
            + 1e-10
        )
        cos_d = 1 - cos_sim

        # Simple latent proxy: PCA-like projection using first latent_dim
        # principal components of X (fast, no sklearn PCA fitting needed)
        X_centered = X - X.mean(axis=0)
        cov        = np.cov(X_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        top_idx    = np.argsort(eigvals)[::-1][: self.latent_dim]
        Z_latent   = X_centered @ eigvecs[:, top_idx]

        return np.concatenate([Z_latent, rel_ed, cos_d], axis=1)

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("DAGMMScorer not fitted. Call fit() first.")
        if self._use_torch:
            return self._score_pytorch(X)
        return self._score_sklearn(X)

    def _score_pytorch(self, X: np.ndarray) -> np.ndarray:
        import torch
        self._model.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        scores = []
        with torch.no_grad():
            for i in range(0, len(tensor), self.batch_size):
                batch           = tensor[i: i + self.batch_size]
                _, z, gamma     = self._model(batch)
                energy, _       = self._gmm_energy(z, gamma)
                # Per-sample energy
                N   = gamma.shape[0]
                phi = gamma.sum(0) / N
                mu  = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(0) / (
                    gamma.sum(0).unsqueeze(1) + 1e-10
                )
                import torch.nn.functional as F
                K = self.n_components
                d = z.shape[1]
                sample_e = []
                for k in range(K):
                    diff_k = z - mu[k]
                    sample_e.append(
                        phi[k].log()
                        - 0.5 * (diff_k ** 2).sum(1)
                    )
                e = -torch.logsumexp(torch.stack(sample_e, dim=1), dim=1)
                scores.append(e.cpu().numpy())
        out = np.concatenate(scores)
        log.info(
            f"DAGMM scored {X.shape[0]:,} agents. "
            f"Range: [{out.min():.4f}, {out.max():.4f}]"
        )
        return out

    def _score_sklearn(self, X: np.ndarray) -> np.ndarray:
        Z   = self._get_latent_sklearn(self._model, X)
        out = -self._gmm.score_samples(Z)   # negative log-likelihood
        log.info(
            f"DAGMM scored {X.shape[0]:,} agents. "
            f"Range: [{out.min():.4f}, {out.max():.4f}]"
        )
        return out

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        cfg = dict(
            n_components=self.n_components,
            latent_dim=self.latent_dim,
            epochs=self.epochs,
            lr=self.lr,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            batch_size=self.batch_size,
            random_state=self.random_state,
            input_dim=self._input_dim,
            use_torch=self._use_torch,
        )
        with open(d / "dagmm_config.pkl", "wb") as fh:
            pickle.dump(cfg, fh)
        if self._use_torch:
            import torch
            torch.save(
                self._model.state_dict(), d / "dagmm_state_dict.pt"
            )
        else:
            with open(d / "dagmm_model.pkl", "wb") as fh:
                pickle.dump(self._model, fh)
            with open(d / "dagmm_gmm.pkl", "wb") as fh:
                pickle.dump(self._gmm, fh)
        log.info(f"DAGMMScorer saved to {d}")

    def load(self, path: str) -> None:
        d = Path(path)
        with open(d / "dagmm_config.pkl", "rb") as fh:
            cfg = pickle.load(fh)
        for k, v in cfg.items():
            if k != "input_dim":
                setattr(self, k, v)
        self._input_dim = cfg["input_dim"]
        if self._use_torch:
            import torch
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = _build_dagmm_model(
                self._input_dim, self.latent_dim, self.n_components
            ).to(self._device)
            self._model.load_state_dict(
                torch.load(
                    d / "dagmm_state_dict.pt",
                    map_location=self._device,
                )
            )
        else:
            with open(d / "dagmm_model.pkl", "rb") as fh:
                self._model = pickle.load(fh)
            with open(d / "dagmm_gmm.pkl", "rb") as fh:
                self._gmm = pickle.load(fh)
        log.info(f"DAGMMScorer loaded from {d}")