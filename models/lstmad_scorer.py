"""
models/lstmad_scorer.py
========================
LSTM-AD — Long Short-Term Memory Anomaly Detection.
Malhotra et al., ESANN 2015.

Trains a stacked LSTM to predict the next T_pred timesteps of a stay
sequence. Prediction errors are fitted to a multivariate Gaussian.
Anomaly score = Mahalanobis distance under that Gaussian.
Higher score = more anomalous.

This version owns the full pipeline:
    - DataFrame → 7-dim stay feature engineering
    - StandardScaler fit on normal agent stays only
    - Sequence padding to max_len
    - LSTM training and scoring

No external sequence builder needed. bilstm_scorer.py not required.

7-dim stay feature vector (Section 4.2.2)
──────────────────────────────────────────
    [0] hour_of_day       — 0–23
    [1] day_of_week       — 0 (Mon) – 6 (Sun)
    [2] is_weekend        — binary 0/1
    [3] log_duration      — log(1 + seconds_in_bin)
    [4] coord_a           — latitude  (NUMOSIM) / x grid (YJMob)
    [5] coord_b           — longitude (NUMOSIM) / y grid (YJMob)
    [6] dist_to_prev      — haversine (NUMOSIM) / euclidean (YJMob)

Usage
─────
    from models.lstmad_scorer import LSTMADScorer

    scorer = LSTMADScorer(hidden=128, n_layers=2, t_pred=3, epochs=100)

    # Fit on normal agent stays only
    scorer.fit(
        stays_df      = normal_stays_df,   # past/future stays for normal agents
        is_yjmob      = False,             # True → euclidean coords
        poi           = poi_df,            # required for NUMOSIM lat/lon
    )

    # Score all agents — returns array aligned to stays_df agent order
    scores = scorer.score(
        stays_df = all_stays_df,
        agents   = all_agent_ids,          # desired output order
    )
"""

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.anomaly_scorer import AnomalyScorer

log = logging.getLogger(__name__)

SEQ_FEATURE_COLS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "log_duration",
    "coord_a",     # lat  (NUMOSIM) / x (YJMob)
    "coord_b",     # lon  (NUMOSIM) / y (YJMob)
    "dist_to_prev",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  (formerly in bilstm_scorer.build_agent_sequences)
# ─────────────────────────────────────────────────────────────────────────────

def _build_stay_features(
    stays_df: pd.DataFrame,
    is_yjmob: bool = False,
    poi: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Engineer the 7-dim stay feature vector from a canonical stays DataFrame.
    Returns one row per stay, sorted by agent then timestamp, with
    columns = ['agent', 'timestamp'] + SEQ_FEATURE_COLS.

    Args:
        stays_df: Canonical stay-sequence DataFrame.
                  Required columns: agent, geo_bin, timestamp, seconds_in_bin.
                  For NUMOSIM: also latitude/longitude (or supply poi).
                  For YJMob:   geo_bin decoded to x/y automatically.
        is_yjmob: True → YJMob100K (euclidean x/y coords).
        poi:      POI reference table [geo_bin, latitude, longitude].
                  Required for NUMOSIM when lat/lon absent.
    """
    from features.distance_utils import attach_coords, coord_cols, distance

    df = stays_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["agent", "timestamp"]).reset_index(drop=True)

    # Attach coordinates
    df = attach_coords(df, is_yjmob, poi)
    col_a, col_b = coord_cols(is_yjmob)

    # Temporal features
    df["hour_of_day"] = df["timestamp"].dt.hour.astype(np.float32)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.float32)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(np.float32)

    # Log duration
    df["log_duration"] = np.log1p(
        df["seconds_in_bin"].clip(lower=0).fillna(0)
    ).astype(np.float32)

    # Coordinates
    df["coord_a"] = df[col_a].fillna(0.0).astype(np.float32)
    df["coord_b"] = df[col_b].fillna(0.0).astype(np.float32)

    # Distance to previous stay within same agent
    df["prev_a"] = df.groupby("agent")["coord_a"].shift(1)
    df["prev_b"] = df.groupby("agent")["coord_b"].shift(1)
    mask = df["prev_a"].notna()
    df["dist_to_prev"] = 0.0
    if mask.any():
        df.loc[mask, "dist_to_prev"] = distance(
            df.loc[mask, "coord_a"].values,
            df.loc[mask, "coord_b"].values,
            df.loc[mask, "prev_a"].values,
            df.loc[mask, "prev_b"].values,
            is_yjmob,
        ).astype(np.float32)
    df = df.drop(columns=["prev_a", "prev_b"])

    return df[["agent", "timestamp"] + SEQ_FEATURE_COLS]


def _to_padded_sequences(
    flat_df: pd.DataFrame,
    max_len: int,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """
    Convert flat stay DataFrame to per-agent sequences, truncated to max_len.

    Returns:
        sequences: Dict[agent_id → np.ndarray (T, 7)]  T ≤ max_len
        agent_ids: List of agent IDs in DataFrame order
    """
    sequences = {}
    agent_ids = []
    for agent_id, grp in flat_df.groupby("agent"):
        arr = grp[SEQ_FEATURE_COLS].values.astype(np.float32)
        # Take last max_len stays (most recent activity)
        sequences[int(agent_id)] = arr[-max_len:]
        agent_ids.append(int(agent_id))
    return sequences, agent_ids


# ─────────────────────────────────────────────────────────────────────────────
# LSTM model definition
# ─────────────────────────────────────────────────────────────────────────────

def _build_lstm_model(input_size: int, hidden: int,
                      n_layers: int, t_pred: int):
    import torch.nn as nn

    class PredLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden, n_layers,
                batch_first=True, dropout=0.2,
            )
            self.fc      = nn.Linear(hidden, input_size * t_pred)
            self.t_pred  = t_pred
            self.in_size = input_size

        def forward(self, x):
            out, _ = self.lstm(x)              # (B, L, H)
            pred   = self.fc(out)              # (B, L, F*T)
            B, L, _ = pred.shape
            return pred.view(B, L, self.t_pred, self.in_size)  # (B, L, T, F)

    return PredLSTM()


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────────────────────────────────────

class LSTMADScorer(AnomalyScorer):
    """
    LSTM-AD anomaly scorer.

    Owns the full pipeline: stays DataFrame → feature engineering →
    StandardScaler (fit on normal agents only) → LSTM training → scoring.

    Args:
        hidden:       LSTM hidden size. Default 128.
        n_layers:     Stacked LSTM layers. Default 2.
        t_pred:       Steps ahead to predict. Default 3.
        epochs:       Training epochs. Default 100.
        lr:           Learning rate. Default 1e-3.
        batch_size:   Batch size. Default 64.
        max_len:      Max sequence length (stays per agent). Default 60.
        random_state: Seed. Default 42.
    """

    def __init__(
        self,
        hidden:       int   = 128,
        n_layers:     int   = 2,
        t_pred:       int   = 3,
        epochs:       int   = 100,
        lr:           float = 1e-3,
        batch_size:   int   = 64,
        max_len:      int   = 60,
        random_state: int   = 42,
    ):
        self.hidden       = hidden
        self.n_layers     = n_layers
        self.t_pred       = t_pred
        self.epochs       = epochs
        self.lr           = lr
        self.batch_size   = batch_size
        self.max_len      = max_len
        self.random_state = random_state

        self._model      = None
        self._scaler     = None    # StandardScaler fit on normal stays
        self._mu         = None    # Gaussian mean of training errors
        self._cov_inv    = None    # Gaussian precision matrix
        self._input_size = len(SEQ_FEATURE_COLS)
        self._device     = None
        self._is_yjmob   = False
        self._poi        = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        stays_df: pd.DataFrame,
        is_yjmob: bool = False,
        poi: pd.DataFrame = None,
    ) -> None:
        """
        Fit on normal agent stays.

        Args:
            stays_df: Canonical stays for NORMAL agents only (past or future period).
                      Required cols: agent, geo_bin, timestamp, seconds_in_bin.
            is_yjmob: True → YJMob100K euclidean coords.
            poi:      POI table for NUMOSIM lat/lon attachment.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._is_yjmob = is_yjmob
        self._poi      = poi
        torch.manual_seed(self.random_state)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        log.info(
            f"LSTM-AD fit — {stays_df['agent'].nunique():,} normal agents, "
            f"device={self._device}"
        )

        # ── Step 1: Feature engineering ───────────────────────────────────────
        log.info("  Building stay features...")
        flat_df = _build_stay_features(stays_df, is_yjmob, poi)

        # ── Step 2: StandardScaler fit on normal stays ─────────────────────
        log.info("  Fitting StandardScaler on normal stays...")
        self._scaler = StandardScaler()
        self._scaler.fit(flat_df[SEQ_FEATURE_COLS].fillna(0).values)

        # ── Step 3: Scale and build sequences ─────────────────────────────
        flat_df = flat_df.copy()
        flat_df[SEQ_FEATURE_COLS] = self._scaler.transform(
            flat_df[SEQ_FEATURE_COLS].fillna(0).values
        )
        seq_dict, _ = _to_padded_sequences(flat_df, self.max_len)
        sequences   = list(seq_dict.values())

        log.info(f"  {len(sequences):,} agent sequences built.")

        # ── Step 4: Pad to fixed length for batching ───────────────────────
        N  = len(sequences)
        X  = np.zeros((N, self.max_len, self._input_size), dtype=np.float32)
        for i, seq in enumerate(sequences):
            L = min(len(seq), self.max_len)
            X[i, :L, :] = seq[-L:]

        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size = self.batch_size,
            shuffle    = True,
        )

        # ── Step 5: Train LSTM ────────────────────────────────────────────
        model   = _build_lstm_model(
            self._input_size, self.hidden, self.n_layers, self.t_pred
        ).to(self._device)
        opt     = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (x,) in loader:
                x    = x.to(self._device)
                L    = x.shape[1]
                if L <= self.t_pred:
                    continue
                inp     = x[:, :-self.t_pred, :]          # (B, L-T, F)
                preds   = model(inp)                       # (B, L-T, T, F)
                targets = torch.stack([
                    x[:, t: L - self.t_pred + t, :]
                    for t in range(1, self.t_pred + 1)
                ], dim=2)                                  # (B, L-T, T, F)
                loss = loss_fn(preds, targets)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    f"  Epoch {epoch:>3}/{self.epochs}  "
                    f"loss={epoch_loss / max(len(loader), 1):.5f}"
                )

        self._model = model

        # ── Step 6: Fit Gaussian to training prediction errors ─────────────
        log.info("  Fitting Gaussian to training errors...")
        errors = self._prediction_errors(X)             # (N, F)
        self._mu      = errors.mean(axis=0)
        cov           = np.cov(errors.T) + 1e-6 * np.eye(errors.shape[1])
        self._cov_inv = np.linalg.pinv(cov)
        log.info("LSTM-AD training complete.")

    # ── score ─────────────────────────────────────────────────────────────────

    def score(
        self,
        stays_df: pd.DataFrame,
        agents: np.ndarray = None,
    ) -> np.ndarray:
        """
        Score agents. Returns anomaly scores aligned to `agents` order.

        Args:
            stays_df: Canonical stays for ALL agents to score.
            agents:   Ordered array of agent IDs for output alignment.
                      If None, scores are returned in groupby order.

        Returns:
            1-D float array of length len(agents) (or n_agents if agents=None).
            Higher = more anomalous.
        """
        if self._model is None:
            raise RuntimeError("LSTMADScorer not fitted. Call fit() first.")

        # Feature engineering + scaling (same pipeline as fit)
        flat_df = _build_stay_features(stays_df, self._is_yjmob, self._poi)
        flat_df = flat_df.copy()
        flat_df[SEQ_FEATURE_COLS] = self._scaler.transform(
            flat_df[SEQ_FEATURE_COLS].fillna(0).values
        )
        seq_dict, seq_ids = _to_padded_sequences(flat_df, self.max_len)

        N = len(seq_dict)
        X = np.zeros((N, self.max_len, self._input_size), dtype=np.float32)
        for i, agent_id in enumerate(seq_ids):
            seq = seq_dict[agent_id]
            L   = min(len(seq), self.max_len)
            X[i, :L, :] = seq[-L:]

        errors = self._prediction_errors(X)             # (N, F)
        diff   = errors - self._mu
        raw_scores = np.array([
            float(d @ self._cov_inv @ d) for d in diff
        ])

        log.info(
            f"LSTM-AD scored {N:,} agents. "
            f"Range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]"
        )

        # Align to requested agent order
        if agents is None:
            return raw_scores

        id_to_score = dict(zip(seq_ids, raw_scores))
        return np.array([
            id_to_score.get(int(a), 0.0) for a in agents
        ], dtype=np.float64)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _prediction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute mean squared prediction error per agent. Returns (N, F)."""
        import torch
        self._model.eval()
        tensor     = torch.tensor(X, dtype=torch.float32)
        all_errors = []

        with torch.no_grad():
            for i in range(0, len(tensor), self.batch_size):
                x    = tensor[i: i + self.batch_size].to(self._device)
                L    = x.shape[1]
                inp  = x[:, :-self.t_pred, :]
                pred = self._model(inp)            # (B, L-T, T, F)
                targets = torch.stack([
                    x[:, t: L - self.t_pred + t, :]
                    for t in range(1, self.t_pred + 1)
                ], dim=2)
                err      = (pred - targets) ** 2   # (B, L-T, T, F)
                err_mean = err.mean(dim=(1, 2)).cpu().numpy()   # (B, F)
                all_errors.append(err_mean)

        return np.concatenate(all_errors)          # (N, F)

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import torch
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), d / "lstmad_state_dict.pt")
        with open(d / "lstmad_params.pkl", "wb") as fh:
            pickle.dump({
                "hidden":       self.hidden,
                "n_layers":     self.n_layers,
                "t_pred":       self.t_pred,
                "epochs":       self.epochs,
                "lr":           self.lr,
                "batch_size":   self.batch_size,
                "max_len":      self.max_len,
                "random_state": self.random_state,
                "input_size":   self._input_size,
                "mu":           self._mu,
                "cov_inv":      self._cov_inv,
                "scaler":       self._scaler,
                "is_yjmob":     self._is_yjmob,
            }, fh)
        log.info(f"LSTMADScorer saved → {d}")

    def load(self, path: str) -> None:
        import torch
        d = Path(path)
        with open(d / "lstmad_params.pkl", "rb") as fh:
            p = pickle.load(fh)
        self.hidden       = p["hidden"]
        self.n_layers     = p["n_layers"]
        self.t_pred       = p["t_pred"]
        self.epochs       = p["epochs"]
        self.lr           = p["lr"]
        self.batch_size   = p["batch_size"]
        self.max_len      = p["max_len"]
        self.random_state = p["random_state"]
        self._input_size  = p["input_size"]
        self._mu          = p["mu"]
        self._cov_inv     = p["cov_inv"]
        self._scaler      = p["scaler"]
        self._is_yjmob    = p.get("is_yjmob", False)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = _build_lstm_model(
            self._input_size, self.hidden, self.n_layers, self.t_pred
        ).to(self._device)
        self._model.load_state_dict(
            torch.load(d / "lstmad_state_dict.pt", map_location=self._device)
        )
        log.info(f"LSTMADScorer loaded from {d}")