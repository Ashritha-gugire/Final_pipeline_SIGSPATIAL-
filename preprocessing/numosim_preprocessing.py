"""
preprocessing/numosim_preprocessing.py
=======================================
Converts raw NUMOSIM parquet files into the canonical stay-sequence format.

Output schema
─────────────
    agent           int       unique agent identifier
    geo_bin         int       location identifier (poi_id)
    timestamp       datetime  stay start time
    seconds_in_bin  int       stay duration in seconds
    latitude        float     POI latitude
    longitude       float     POI longitude
    london          int       agent-specific frequency rank (0 = home proxy)
    period          str       'past' or 'future'

Usage
─────
    from preprocessing.numosim_preprocessing import NUMOSIMPreprocessor

    prep   = NUMOSIMPreprocessor(data_dir="data/numosim/raw",
                                  output_dir="data/numosim/processed")
    result = prep.run()
    # result keys: train_stays, test_stays, test_anom, poi, demo, keep_agents
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def compute_london(stays: pd.DataFrame) -> pd.DataFrame:
    """
    Rank each agent's visited locations by total time spent.
    london = 0 → most visited (home proxy).

    Returns DataFrame [agent, geo_bin, london].
    """
    durations = (
        stays.groupby(["agent", "geo_bin"], as_index=False)["seconds_in_bin"]
        .sum()
        .rename(columns={"seconds_in_bin": "total_seconds"})
    )
    durations = durations.sort_values(
        ["agent", "total_seconds", "geo_bin"],
        ascending=[True, False, True],
    )
    durations["london"] = durations.groupby("agent").cumcount()
    return durations[["agent", "geo_bin", "london"]]


# ─────────────────────────────────────────────────────────────────────────────

class NUMOSIMPreprocessor:
    """
    Loads raw NUMOSIM parquet files and converts to canonical stay-sequence format.

    Expected raw files under data_dir/
    ───────────────────────────────────
        poi.parquet                      — POI reference table
        stay_points_train.parquet        — training period stays
        stay_points_test_truth.parquet   — test period ground truth
        stay_points_test_anomalous.parquet — test period with inserted anomalies
        demographics.parquet             — agent demographics (optional)

    Args:
        data_dir:    Path to raw NUMOSIM parquet files.
        output_dir:  Path to save processed flat parquet files.
        n_normal:    Number of normal agents to sample (None = all).
        random_seed: Seed for reproducibility.
    """

    CANONICAL_COLS = [
        "agent", "geo_bin", "timestamp", "seconds_in_bin",
        "latitude", "longitude", "london", "period",
    ]

    def __init__(
        self,
        data_dir:    str = "data/numosim/raw",
        output_dir:  str = "data/numosim/processed",
        n_normal:    int = 20_000,
        random_seed: int = 42,
    ):
        self.data_dir    = Path(data_dir)
        self.output_dir  = Path(output_dir)
        self.n_normal    = n_normal
        self.random_seed = random_seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_raw(self) -> dict[str, pd.DataFrame]:
        log.info("Loading raw NUMOSIM files...")
        paths = {
            "poi":        self.data_dir / "poi.parquet",
            "train":      self.data_dir / "stay_points_train.parquet",
            "test_truth": self.data_dir / "stay_points_test_truth.parquet",
            "test_anom":  self.data_dir / "stay_points_test_anomalous.parquet",
            "demo":       self.data_dir / "demographics.parquet",
        }
        dfs = {}
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Expected file not found: {path}")
            dfs[key] = pd.read_parquet(path)
            log.info(f"  Loaded {key}: {dfs[key].shape}")

        # Ensure datetime columns are parsed
        for key in ("train", "test_truth", "test_anom"):
            for col in ("start_datetime", "end_datetime"):
                if col in dfs[key].columns:
                    dfs[key][col] = pd.to_datetime(dfs[key][col], utc=True)
        return dfs

    def _sample_agents(self, dfs: dict) -> np.ndarray:
        """Keep all anomalous agents + up to n_normal sampled normal agents."""
        anomalous  = dfs["test_anom"][dfs["test_anom"]["anomaly"] == True]["agent_id"].unique()
        all_agents = dfs["train"]["agent_id"].unique()
        normal     = np.setdiff1d(all_agents, anomalous)

        np.random.seed(self.random_seed)
        if self.n_normal is not None and self.n_normal < len(normal):
            sampled_normal = np.random.choice(normal, size=self.n_normal, replace=False)
        else:
            sampled_normal = normal

        keep = np.concatenate([anomalous, sampled_normal])
        log.info(f"  Anomalous agents  : {len(anomalous):,}")
        log.info(f"  Sampled normal    : {len(sampled_normal):,}")
        log.info(f"  Total kept        : {len(keep):,}")
        return keep

    def _to_canonical(self, stays: pd.DataFrame, poi: pd.DataFrame,
                      period: str) -> pd.DataFrame:
        """Rename columns, compute durations, attach coords and london rank."""
        df = stays.copy()
        df = df.rename(columns={
            "agent_id":       "agent",
            "poi_id":         "geo_bin",
            "start_datetime": "timestamp",
        })
        df["seconds_in_bin"] = (
            (df["end_datetime"] - df["timestamp"])
            .dt.total_seconds()
            .astype(int)
        )

        # Attach POI coordinates
        poi_coords = (
            poi[["poi_id", "latitude", "longitude"]]
            .rename(columns={"poi_id": "geo_bin"})
        )
        df = df.merge(poi_coords, on="geo_bin", how="left")

        # London rank (0 = home proxy)
        london_map = compute_london(df)
        df = df.merge(london_map, on=["agent", "geo_bin"], how="left")
        df["london"] = df["london"].fillna(0).astype(int)
        df["period"] = period

        df = df.drop(
            columns=["end_datetime", "anomaly", "anomaly_type"],
            errors="ignore",
        )
        return df[[c for c in self.CANONICAL_COLS if c in df.columns]]

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Full preprocessing pipeline.

        Returns dict with keys:
            past_stays, future_stays, test_anom, poi, demo, keep_agents
        """
        log.info("=" * 60)
        log.info("NUMOSIM Preprocessing Pipeline")
        log.info("=" * 60)

        dfs         = self._load_raw()
        keep_agents = self._sample_agents(dfs)
        poi         = dfs["poi"]

        train_raw = dfs["train"][dfs["train"]["agent_id"].isin(keep_agents)].copy()
        # Use test_anom (has inserted stays) as the future/test period
        test_raw  = dfs["test_anom"][dfs["test_anom"]["agent_id"].isin(keep_agents)].copy()
        test_anom = dfs["test_anom"][dfs["test_anom"]["agent_id"].isin(keep_agents)].copy()
        demo      = dfs["demo"][dfs["demo"]["agent_id"].isin(keep_agents)].copy() \
                    if "demo" in dfs else pd.DataFrame()

        log.info(f"Past stays     : {len(train_raw):,}")
        log.info(f"Future stays   : {len(test_raw):,}")
        log.info(f"Anomalous stays: {test_anom['anomaly'].sum():,}")

        log.info("Converting to canonical format...")
        past_stays   = self._to_canonical(train_raw, poi, "past")
        future_stays = self._to_canonical(test_raw,  poi, "future")

        log.info("Saving processed flat parquet files...")
        past_stays.to_parquet(  self.output_dir / "past_stays.parquet",   index=False)
        future_stays.to_parquet(self.output_dir / "future_stays.parquet", index=False)
        test_anom.to_parquet(   self.output_dir / "test_anom.parquet",    index=False)
        poi.to_parquet(         self.output_dir / "poi.parquet",           index=False)
        if not demo.empty:
            demo.to_parquet(    self.output_dir / "demo.parquet",          index=False)

        log.info(f"Preprocessing complete → {self.output_dir}")
        return {
            "past_stays":   past_stays,
            "future_stays": future_stays,
            "test_anom":    test_anom,
            "poi":          poi,
            "demo":         demo,
            "keep_agents":  keep_agents,
        }