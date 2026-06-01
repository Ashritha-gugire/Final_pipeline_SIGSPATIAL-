"""
features/distance_utils.py
===========================
Shared distance utilities for NUMOSIM (haversine) and YJMob100K (euclidean).
Import from here — never duplicate distance logic across feature files.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Core distance functions
# ─────────────────────────────────────────────────────────────────────────────

def haversine_np(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised Haversine distance in metres (NUMOSIM — lat/lon)."""
    R    = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def euclidean_np(x1, y1, x2, y2) -> np.ndarray:
    """Vectorised Euclidean distance in grid-cell units (YJMob100K — x/y)."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def pairwise_max_distance(coords_a: np.ndarray, coords_b: np.ndarray,
                          is_yjmob: bool = False,
                          max_sample: int = 200) -> float:
    """
    Maximum pairwise distance among a set of points.
    Uses haversine for NUMOSIM, euclidean for YJMob.

    Args:
        coords_a: Array of first coordinates (lat or x), shape (N,)
        coords_b: Array of second coordinates (lon or y), shape (N,)
        is_yjmob: If True, use euclidean; else haversine.
        max_sample: Subsample if N > this value for speed.
    """
    if len(coords_a) < 2:
        return 0.0
    if len(coords_a) > max_sample:
        idx      = np.random.choice(len(coords_a), max_sample, replace=False)
        coords_a = coords_a[idx]
        coords_b = coords_b[idx]

    dist_fn  = euclidean_np if is_yjmob else haversine_np
    n        = len(coords_a)
    max_dist = 0.0
    for i in range(n - 1):
        dists    = dist_fn(coords_a[i], coords_b[i], coords_a[i + 1:], coords_b[i + 1:])
        max_dist = max(max_dist, float(dists.max()))
    return max_dist


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate column helpers
# ─────────────────────────────────────────────────────────────────────────────

def coord_cols(is_yjmob: bool) -> tuple[str, str]:
    """Return (col_a, col_b) coordinate column names for the given dataset."""
    return ("x", "y") if is_yjmob else ("latitude", "longitude")


def distance(c1_a, c1_b, c2_a, c2_b, is_yjmob: bool) -> np.ndarray:
    """Dispatch to euclidean or haversine based on dataset flag."""
    return euclidean_np(c1_a, c1_b, c2_a, c2_b) if is_yjmob \
        else haversine_np(c1_a, c1_b, c2_a, c2_b)


# ─────────────────────────────────────────────────────────────────────────────
# YJMob geo_bin decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_geo_bin_yjmob(geo_bin: int) -> tuple[float, float]:
    """
    Decode a YJMob100K integer geo_bin into (x, y) grid coordinates.
    Encoding: geo_bin = x * 10000 + y  (both x and y are 0-9999).
    """
    x = geo_bin // 10_000
    y = geo_bin % 10_000
    return float(x), float(y)


def attach_coords(df: pd.DataFrame, is_yjmob: bool,
                  poi: pd.DataFrame = None) -> pd.DataFrame:
    """
    Ensure coordinate columns are present on a stays DataFrame.

    - NUMOSIM: merges lat/lon from POI table if not already present.
    - YJMob:   decodes geo_bin → (x, y) columns.

    Args:
        df:       Stay-sequence DataFrame (must have 'geo_bin').
        is_yjmob: Dataset flag.
        poi:      POI reference table with ['geo_bin','latitude','longitude'].
                  Required for NUMOSIM if lat/lon absent from df.
    """
    df = df.copy()
    if is_yjmob:
        if "x" not in df.columns or "y" not in df.columns:
            xy    = df["geo_bin"].map(decode_geo_bin_yjmob)
            df["x"] = xy.map(lambda t: t[0])
            df["y"] = xy.map(lambda t: t[1])
    else:
        if "latitude" not in df.columns or df["latitude"].isna().all():
            if poi is None:
                raise ValueError("POI table required for NUMOSIM coord attachment.")
            # Drop stale NaN coord columns to avoid merge suffixes
            drop = [c for c in ["latitude", "longitude"] if c in df.columns]
            if drop:
                df = df.drop(columns=drop)
            poi_cols = (
                poi[["poi_id", "latitude", "longitude"]]
                .rename(columns={"poi_id": "geo_bin"})
                if "poi_id" in poi.columns
                else poi[["geo_bin", "latitude", "longitude"]]
            )
            df = df.merge(poi_cols, on="geo_bin", how="left")
    return df