"""
features/features_18dim.py
===========================
Unified 18-dimensional common feature vector.
Works identically on NUMOSIM (lat/lon, haversine) and YJMob100K (grid x/y, euclidean).

Table 1 — feature index reference
──────────────────────────────────
Change features (7)
  1  geobin_similarity          Jaccard similarity of visited locations: past vs future
  2  abandonment_score          Fraction of past locations absent from future visits
  3  kuiper_start_stat          Kuiper V on stay start-time distributions
  4  kuiper_start_pval          p-value of start-time Kuiper test
  5  kuiper_end_stat            Kuiper V on stay end-time distributions
  6  kuiper_end_pval            p-value of end-time Kuiper test
  7  rog_drift                  RoG_future - RoG_past

Drift features (5)  [future - past]
  8  gyration_score_drift       Change in gyration score
  9  max_pairwise_dist_drift    Change in max pairwise distance
 10  max_dist_home_drift        Change in max distance from home
 11  std_dist_home_drift        Change in std of home distance
 12  mean_dist_home_drift       Change in mean distance from home

Absolute feature (1)
 13  novel_location_rate        Fraction of test stays at unseen geo-bins

Entropy features (5)
 14  loc_entropy_future         H of geo-bin visits in test period
 15  temp_entropy_future        H of hour-of-day activity in test period
 16  loc_entropy_drift          H_future^loc - H_past^loc
 17  temp_entropy_drift         H_future^temp - H_past^temp
 18  entropy_ratio              H_future^loc / (H_past^loc + ε)

Usage
─────
    from features.features_18dim import FeatureBuilder18Dim

    builder = FeatureBuilder18Dim(
        past_stays  = past_df,     # pd.DataFrame — flat parquet, past period
        future_stays= future_df,   # pd.DataFrame — flat parquet, future period
        is_yjmob    = False,       # True → YJMob100K (euclidean), False → NUMOSIM (haversine)
        poi         = poi_df,      # required for NUMOSIM; None for YJMob
    )
    features_df = builder.compute_all()   # one row per agent, 18 feature columns
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

try:
    from astropy.stats import kuiper_two
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

from features.distance_utils import (
    attach_coords, coord_cols, distance,
    haversine_np, pairwise_max_distance,
)

log = logging.getLogger(__name__)

# ── Feature name constants (Table 1 order) ───────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Change features
    "geobin_similarity",        # 1
    "abandonment_score",        # 2
    "kuiper_start_stat",        # 3
    "kuiper_start_pval",        # 4
    "kuiper_end_stat",          # 5
    "kuiper_end_pval",          # 6
    "rog_drift",                # 7
    # Drift features
    "gyration_score_drift",     # 8
    "max_pairwise_dist_drift",  # 9
    "max_dist_home_drift",      # 10
    "std_dist_home_drift",      # 11
    "mean_dist_home_drift",     # 12
    # Absolute
    "novel_location_rate",      # 13
    # Entropy
    "loc_entropy_future",       # 14
    "temp_entropy_future",      # 15
    "loc_entropy_drift",        # 16
    "temp_entropy_drift",       # 17
    "entropy_ratio",            # 18
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(series: pd.Series) -> float:
    counts = series.value_counts()
    probs  = counts / counts.sum()
    return float(scipy_entropy(probs))


def _compute_home(stays: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """
    Per-agent home = geo_bin with the most total seconds_in_bin in the given period.
    Returns DataFrame [agent, home_a, home_b].
    """
    agg = (
        stays.groupby(["agent", "geo_bin"], as_index=False)["seconds_in_bin"]
        .sum()
        .sort_values(["agent", "seconds_in_bin"], ascending=[True, False])
        .drop_duplicates("agent")
    )
    coords = stays.drop_duplicates("geo_bin")[["geo_bin", col_a, col_b]]
    home   = (
        agg[["agent", "geo_bin"]]
        .merge(coords, on="geo_bin", how="left")
        .rename(columns={col_a: "home_a", col_b: "home_b"})
        [["agent", "home_a", "home_b"]]
    )
    return home


def _rog(stays: pd.DataFrame, col_a: str, col_b: str, is_yjmob: bool) -> pd.Series:
    """Weighted radius of gyration per agent (returns Series indexed by agent)."""
    df      = stays.dropna(subset=[col_a, col_b]).copy()
    df["w"] = df["seconds_in_bin"].clip(lower=1)
    df["w_a"] = df["w"] * df[col_a]
    df["w_b"] = df["w"] * df[col_b]
    c = df.groupby("agent").agg(wa=("w_a", "sum"), wb=("w_b", "sum"), wt=("w", "sum"))
    c["c_a"] = c["wa"] / c["wt"]
    c["c_b"] = c["wb"] / c["wt"]
    df = df.merge(c[["c_a", "c_b"]], on="agent")
    df["dist_sq"]   = distance(df[col_a], df[col_b], df["c_a"], df["c_b"], is_yjmob) ** 2
    df["w_dist_sq"] = df["w"] * df["dist_sq"]
    rog = df.groupby("agent").agg(wds=("w_dist_sq", "sum"), wt=("w", "sum"))
    return np.sqrt(rog["wds"] / rog["wt"]).rename("rog")


def _spatial_stats(stays: pd.DataFrame, home: pd.DataFrame,
                   col_a: str, col_b: str, is_yjmob: bool) -> pd.DataFrame:
    """
    Gyration score, max pairwise distance, and home-distance stats per agent.
    Returns DataFrame [agent, gyration_score, max_pairwise_dist,
                       max_dist_from_home, std_dist_from_home, mean_dist_from_home].
    """
    df = stays.dropna(subset=[col_a, col_b]).copy()

    # Centroid
    centroid = (
        df.groupby("agent")[[col_a, col_b]].mean().reset_index()
        .rename(columns={col_a: "c_a", col_b: "c_b"})
    )
    df = df.merge(centroid, on="agent", how="left")
    df["dist_centroid"] = distance(df[col_a], df[col_b], df["c_a"], df["c_b"], is_yjmob)

    gyration = (
        df.groupby("agent")["dist_centroid"]
        .apply(lambda x: float(np.sqrt((x ** 2).mean())))
        .rename("gyration_score")
        .reset_index()
    )
    max_pair = (
        df.groupby("agent")["dist_centroid"].max() * 2
    ).rename("max_pairwise_dist").reset_index()

    # Home distance
    df_h = df.merge(home, on="agent", how="left")
    df_h["dist_home"] = distance(df_h[col_a], df_h[col_b],
                                 df_h["home_a"], df_h["home_b"], is_yjmob)
    home_stats = (
        df_h.groupby("agent")["dist_home"]
        .agg(max_dist_from_home="max",
             std_dist_from_home="std",
             mean_dist_from_home="mean")
        .fillna(0)
        .reset_index()
    )

    result = gyration.merge(max_pair, on="agent").merge(home_stats, on="agent")
    return result


def _kuiper_stats(past_stays: pd.DataFrame, future_stays: pd.DataFrame,
                  bin_size: int = 300) -> pd.DataFrame:
    """
    Per-agent Kuiper V statistic and p-value for start-time and end-time distributions.
    bin_size: seconds per time bucket (default 300 = 5 min).
    Returns DataFrame [agent, kuiper_start_stat, kuiper_start_pval,
                                kuiper_end_stat,  kuiper_end_pval].
    """
    if not HAS_ASTROPY:
        log.warning("astropy not installed — Kuiper features will be 0.")
        agents = future_stays["agent"].unique()
        return pd.DataFrame({
            "agent": agents,
            "kuiper_start_stat": 0.0, "kuiper_start_pval": 1.0,
            "kuiper_end_stat":   0.0, "kuiper_end_pval":   1.0,
        })

    def _bucketise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        secs = df["timestamp"].dt.hour * 3600 + df["timestamp"].dt.minute * 60 + df["timestamp"].dt.second
        df["bucket_start"] = (secs // bin_size).astype(int)
        end_ts = df["timestamp"] + pd.to_timedelta(df["seconds_in_bin"], unit="s")
        secs_e = end_ts.dt.hour * 3600 + end_ts.dt.minute * 60 + end_ts.dt.second
        df["bucket_end"] = (secs_e // bin_size).astype(int)
        return df

    past_b   = _bucketise(past_stays)
    future_b = _bucketise(future_stays)

    agents = sorted(set(past_b["agent"]) & set(future_b["agent"]))
    results = []

    def _clean(val, is_pval=False):
        v = float(val.real if hasattr(val, "real") else val)
        return max(0.0, min(1.0, v)) if is_pval else v

    for agent_id in agents:
        try:
            p = past_b[past_b["agent"] == agent_id]
            f = future_b[future_b["agent"] == agent_id]
            if len(p) == 0 or len(f) == 0:
                continue
            ks = kuiper_two(p["bucket_start"].values, f["bucket_start"].values)
            ke = kuiper_two(p["bucket_end"].values,   f["bucket_end"].values)
            results.append({
                "agent":            agent_id,
                "kuiper_start_stat": _clean(ks[0]),
                "kuiper_start_pval": _clean(ks[1], True),
                "kuiper_end_stat":   _clean(ke[0]),
                "kuiper_end_pval":   _clean(ke[1], True),
            })
        except Exception as e:
            log.warning(f"Kuiper failed for agent {agent_id}: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["agent","kuiper_start_stat","kuiper_start_pval",
                 "kuiper_end_stat","kuiper_end_pval"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class FeatureBuilder18Dim:
    """
    Computes the full 18-dimensional common feature vector for all agents.

    Args:
        past_stays:   Flat parquet DataFrame for the PAST period.
                      Required columns: agent, geo_bin, timestamp, seconds_in_bin.
                      For NUMOSIM: also latitude, longitude (or supply poi).
                      For YJMob:   geo_bin is decoded to x/y automatically.
        future_stays: Same schema, FUTURE (test) period.
        is_yjmob:     True → YJMob100K (euclidean grid).
                      False → NUMOSIM (haversine lat/lon).
        poi:          POI reference table [geo_bin, latitude, longitude].
                      Required for NUMOSIM if lat/lon absent from stays DataFrames.
    """

    def __init__(
        self,
        past_stays:   pd.DataFrame,
        future_stays: pd.DataFrame,
        is_yjmob:     bool = False,
        poi:          pd.DataFrame = None,
    ):
        self.is_yjmob = is_yjmob
        self.poi      = poi

        self.past   = attach_coords(past_stays,   is_yjmob, poi)
        self.future = attach_coords(future_stays, is_yjmob, poi)

        self.past["timestamp"]   = pd.to_datetime(self.past["timestamp"])
        self.future["timestamp"] = pd.to_datetime(self.future["timestamp"])

        self._col_a, self._col_b = coord_cols(is_yjmob)

    # ── Feature 1: Geobin Similarity (Jaccard) ────────────────────────────────
    def _geobin_similarity(self) -> pd.Series:
        past_sets   = self.past.groupby("agent")["geo_bin"].apply(set)
        future_sets = self.future.groupby("agent")["geo_bin"].apply(set)
        df = pd.DataFrame({"past": past_sets, "future": future_sets}).dropna()
        df["geobin_similarity"] = df.apply(
            lambda r: len(r["past"] & r["future"]) / len(r["past"] | r["future"])
            if r["past"] | r["future"] else 0.0,
            axis=1,
        )
        return df["geobin_similarity"]

    # ── Feature 2: Abandonment Score ──────────────────────────────────────────
    def _abandonment_score(self) -> pd.Series:
        past_freq = (
            self.past.groupby(["agent", "geo_bin"]).size()
            .reset_index(name="past_count")
        )
        future_locs = self.future.groupby(["agent", "geo_bin"]).size().reset_index(name="fut")
        merged = past_freq.merge(future_locs, on=["agent", "geo_bin"], how="left")
        merged["abandoned"] = merged["fut"].isna()
        abandoned_w = (
            merged[merged["abandoned"]]
            .groupby("agent")["past_count"].sum()
            .rename("abandoned_raw")
        )
        total_locs = (
            self.past.groupby("agent")["geo_bin"].nunique()
            .rename("total_locs")
        )
        score = (abandoned_w / total_locs).fillna(0).rename("abandonment_score")
        return score

    # ── Features 3–6: Kuiper ──────────────────────────────────────────────────
    def _kuiper(self) -> pd.DataFrame:
        return _kuiper_stats(self.past, self.future)

    # ── Feature 7: RoG Drift ──────────────────────────────────────────────────
    def _rog_drift(self) -> pd.Series:
        rog_past   = _rog(self.past,   self._col_a, self._col_b, self.is_yjmob)
        rog_future = _rog(self.future, self._col_a, self._col_b, self.is_yjmob)
        return (rog_future - rog_past).rename("rog_drift")

    # ── Features 8–12: Spatial Drift ─────────────────────────────────────────
    def _spatial_drift(self) -> pd.DataFrame:
        home        = _compute_home(self.past, self._col_a, self._col_b)
        past_stats  = _spatial_stats(self.past,   home, self._col_a, self._col_b, self.is_yjmob)
        fut_stats   = _spatial_stats(self.future, home, self._col_a, self._col_b, self.is_yjmob)
        merged      = fut_stats.merge(past_stats, on="agent", suffixes=("_f", "_p"))
        drift_cols  = ["gyration_score", "max_pairwise_dist",
                       "max_dist_from_home", "std_dist_from_home", "mean_dist_from_home"]
        for c in drift_cols:
            merged[f"{c}_drift"] = merged[f"{c}_f"] - merged[f"{c}_p"]
        keep = ["agent"] + [f"{c}_drift" for c in drift_cols]
        return merged[keep].rename(columns={
            "gyration_score_drift":      "gyration_score_drift",
            "max_pairwise_dist_drift":   "max_pairwise_dist_drift",
            "max_dist_from_home_drift":  "max_dist_home_drift",
            "std_dist_from_home_drift":  "std_dist_home_drift",
            "mean_dist_from_home_drift": "mean_dist_home_drift",
        })

    # ── Feature 13: Novel Location Rate ──────────────────────────────────────
    def _novel_location_rate(self) -> pd.Series:
        past_sets = self.past.groupby("agent")["geo_bin"].apply(set).rename("past_locs")
        df = self.future.merge(past_sets, on="agent", how="left")
        df["past_locs"] = df["past_locs"].apply(lambda x: x if isinstance(x, set) else set())
        df["is_novel"]  = df.apply(lambda r: int(r["geo_bin"] not in r["past_locs"]), axis=1)
        return df.groupby("agent")["is_novel"].mean().rename("novel_location_rate")

    # ── Features 14–18: Entropy ───────────────────────────────────────────────
    def _entropy_features(self) -> pd.DataFrame:
        loc_past   = self.past.groupby("agent")["geo_bin"].apply(_shannon_entropy).rename("loc_ent_past")
        loc_future = self.future.groupby("agent")["geo_bin"].apply(_shannon_entropy).rename("loc_ent_future")

        self.past["_hour"]   = self.past["timestamp"].dt.hour
        self.future["_hour"] = self.future["timestamp"].dt.hour
        tmp_past   = self.past.groupby("agent")["_hour"].apply(_shannon_entropy).rename("tmp_ent_past")
        tmp_future = self.future.groupby("agent")["_hour"].apply(_shannon_entropy).rename("tmp_ent_future")

        df = pd.DataFrame({
            "loc_ent_past":   loc_past,
            "loc_ent_future": loc_future,
            "tmp_ent_past":   tmp_past,
            "tmp_ent_future": tmp_future,
        }).fillna(0)

        df["loc_entropy_future"]  = df["loc_ent_future"]
        df["temp_entropy_future"] = df["tmp_ent_future"]
        df["loc_entropy_drift"]   = df["loc_ent_future"] - df["loc_ent_past"]
        df["temp_entropy_drift"]  = df["tmp_ent_future"] - df["tmp_ent_past"]
        df["entropy_ratio"]       = (
            df["loc_ent_future"] / df["loc_ent_past"].replace(0, 1e-6)
        ).clip(0, 10)

        return df[["loc_entropy_future","temp_entropy_future",
                   "loc_entropy_drift","temp_entropy_drift","entropy_ratio"]].reset_index()

    # ── Master compute ────────────────────────────────────────────────────────
    def compute_all(self) -> pd.DataFrame:
        """
        Compute all 18 features. Returns one row per agent with columns = FEATURE_NAMES.
        """
        log.info(f"Computing 18-dim features — dataset: {'YJMob100K' if self.is_yjmob else 'NUMOSIM'}")
        agents = self.future[["agent"]].drop_duplicates()

        log.info("  [1/7] Geobin similarity...")
        geo_sim   = self._geobin_similarity().reset_index()

        log.info("  [2/7] Abandonment score...")
        abandon   = self._abandonment_score().reset_index()

        log.info("  [3/7] Kuiper features...")
        kuiper    = self._kuiper()

        log.info("  [4/7] RoG drift...")
        rog       = self._rog_drift().reset_index()

        log.info("  [5/7] Spatial drift features (8–12)...")
        spatial   = self._spatial_drift()

        log.info("  [6/7] Novel location rate...")
        novel     = self._novel_location_rate().reset_index()

        log.info("  [7/7] Entropy features (14–18)...")
        entropy   = self._entropy_features()

        # ── Join everything on agent ──────────────────────────────────────────
        result = agents
        for df in [geo_sim, abandon, kuiper, rog, spatial, novel, entropy]:
            if df is not None and len(df):
                result = result.merge(df, on="agent", how="left")

        result = result.fillna(0.0)

        # Return exactly FEATURE_NAMES columns (plus agent)
        missing = [f for f in FEATURE_NAMES if f not in result.columns]
        for col in missing:
            log.warning(f"  Feature '{col}' missing — filling with 0.")
            result[col] = 0.0

        log.info(f"  Done — {len(result)} agents, {len(FEATURE_NAMES)} features.")
        return result[["agent"] + FEATURE_NAMES].reset_index(drop=True)