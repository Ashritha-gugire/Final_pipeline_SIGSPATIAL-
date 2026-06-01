"""
features/sequence_features.py
==============================
7-dimensional stay-level feature vector for sequence models (LSTM-AD, LM-TAD).

Per Section 4.2.2 of the paper — each stay becomes a 7-dim vector:
  [0] hour_of_day            — integer 0–23
  [1] day_of_week            — integer 0 (Mon) – 6 (Sun)
  [2] is_weekend             — binary 0/1
  [3] log_duration           — log(1 + duration_seconds)
  [4] latitude  / x          — lat for NUMOSIM; decoded x grid coord for YJMob
  [5] longitude / y          — lon for NUMOSIM; decoded y grid coord for YJMob
  [6] dist_to_prev_geobin    — haversine (NUMOSIM) or euclidean (YJMob) in same
                               units as features_18dim distance utilities

Usage
─────
    from features.sequence_features import SequenceFeatureBuilder

    builder = SequenceFeatureBuilder(
        stays    = stays_df,   # pd.DataFrame — flat parquet for the period to encode
        is_yjmob = False,      # True → YJMob100K, False → NUMOSIM
        poi      = poi_df,     # required for NUMOSIM if lat/lon absent; None for YJMob
    )

    # Returns Dict[agent_id → np.ndarray of shape (T, 7)]
    sequences = builder.build_sequences()

    # Or a flat DataFrame with all stays + 7 feature columns (useful for inspection)
    flat_df = builder.build_flat()
"""

import logging
import numpy as np
import pandas as pd

from features.distance_utils import attach_coords, coord_cols, distance

log = logging.getLogger(__name__)

# Column order — must match model input expectations exactly
SEQ_FEATURE_COLS = [
    "hour_of_day",        # 0
    "day_of_week",        # 1
    "is_weekend",         # 2
    "log_duration",       # 3
    "coord_a",            # 4  (latitude  for NUMOSIM  /  x  for YJMob)
    "coord_b",            # 5  (longitude for NUMOSIM  /  y  for YJMob)
    "dist_to_prev",       # 6
]


class SequenceFeatureBuilder:
    """
    Builds 7-dim stay-level feature sequences for LSTM-AD and LM-TAD.

    Args:
        stays:    Flat parquet DataFrame for the period to encode.
                  Required columns: agent, geo_bin, timestamp, seconds_in_bin.
                  For NUMOSIM: also latitude/longitude (or supply poi).
                  For YJMob:   geo_bin decoded to x/y automatically.
        is_yjmob: True → YJMob100K (euclidean). False → NUMOSIM (haversine).
        poi:      POI table [geo_bin, latitude, longitude]. Required for NUMOSIM
                  when lat/lon absent from stays; None for YJMob.
    """

    def __init__(
        self,
        stays:    pd.DataFrame,
        is_yjmob: bool = False,
        poi:      pd.DataFrame = None,
    ):
        self.is_yjmob      = is_yjmob
        self._col_a, self._col_b = coord_cols(is_yjmob)

        df = attach_coords(stays.copy(), is_yjmob, poi)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["agent", "timestamp"]).reset_index(drop=True)
        self._df = self._build_features(df)

    # ─────────────────────────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 7 stay-level features in-place. Returns enriched DataFrame."""
        col_a, col_b = self._col_a, self._col_b

        # ── Temporal features ─────────────────────────────────────────────────
        df["hour_of_day"] = df["timestamp"].dt.hour.astype(np.int32)
        df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.int32)
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(np.int32)

        # ── Log duration ──────────────────────────────────────────────────────
        df["log_duration"] = np.log1p(df["seconds_in_bin"].clip(lower=0)).astype(np.float32)

        # ── Coordinates (rename for uniform column names in output) ───────────
        df["coord_a"] = df[col_a].astype(np.float32)
        df["coord_b"] = df[col_b].astype(np.float32)

        # ── Distance to previous geo-bin (within agent, sequential) ──────────
        grp             = df.groupby("agent")
        df["prev_a"]    = grp[col_a].shift(1)
        df["prev_b"]    = grp[col_b].shift(1)

        # For the first stay in each agent sequence → distance = 0
        mask = df["prev_a"].notna()
        df["dist_to_prev"] = 0.0
        df.loc[mask, "dist_to_prev"] = distance(
            df.loc[mask, col_a].values,
            df.loc[mask, col_b].values,
            df.loc[mask, "prev_a"].values,
            df.loc[mask, "prev_b"].values,
            self.is_yjmob,
        ).astype(np.float32)

        df = df.drop(columns=["prev_a", "prev_b"])
        return df

    # ─────────────────────────────────────────────────────────────────────────

    def build_sequences(self) -> dict[int, np.ndarray]:
        """
        Returns a dict mapping agent_id → np.ndarray of shape (T, 7).
        T is the number of stays for that agent in the period.
        Column order follows SEQ_FEATURE_COLS.
        """
        log.info(
            f"Building 7-dim sequences — dataset: "
            f"{'YJMob100K' if self.is_yjmob else 'NUMOSIM'}, "
            f"{self._df['agent'].nunique()} agents"
        )
        sequences = {}
        for agent_id, grp in self._df.groupby("agent"):
            sequences[int(agent_id)] = grp[SEQ_FEATURE_COLS].values.astype(np.float32)
        log.info(f"  Done — {len(sequences)} sequences built.")
        return sequences

    def build_flat(self) -> pd.DataFrame:
        """
        Returns a flat DataFrame with all stays plus the 7 feature columns.
        Useful for inspection, saving to parquet, or feeding to LM-TAD directly.

        Columns: agent, geo_bin, timestamp, seconds_in_bin + SEQ_FEATURE_COLS
        """
        id_cols = ["agent", "geo_bin", "timestamp", "seconds_in_bin"]
        keep    = [c for c in id_cols if c in self._df.columns] + SEQ_FEATURE_COLS
        return self._df[keep].reset_index(drop=True)

    @property
    def feature_columns(self) -> list[str]:
        """Names of the 7 output feature columns (model input order)."""
        return SEQ_FEATURE_COLS

    @property
    def coord_column_meaning(self) -> dict[str, str]:
        """Human-readable mapping of coord_a/coord_b to physical meaning."""
        if self.is_yjmob:
            return {"coord_a": "x (grid column)", "coord_b": "y (grid row)"}
        return {"coord_a": "latitude (degrees)", "coord_b": "longitude (degrees)"}