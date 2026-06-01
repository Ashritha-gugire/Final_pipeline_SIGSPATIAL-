"""
preprocessing/yjmob_stay_sequence.py
=====================================
Converts raw YJMob100K slot observations into the canonical stay-sequence format.

Core algorithm: iterate sorted rows, merge consecutive same-agent same-cell
slots into one stay, then rank geo-bins per agent by total duration.

Input assumption: the CSV is globally sorted by (uid, d, t).

Constants
─────────
    GRID_WIDTH   = 200   — YJMob uses a 200×200 grid of 500 m cells
    SLOT_SECONDS = 1800  — each time slot is 30 minutes
    SLOTS_PER_DAY = 48
    SPLIT_DAY    = 60    — d < 60 is "past",  d >= 60 is "future"
"""

from __future__ import annotations

from datetime import datetime, timedelta
from math import nan

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

GRID_WIDTH    = 200
SLOT_SECONDS  = 1_800
SLOTS_PER_DAY = 48
SPLIT_DAY     = 60
NUM_AGENTS_PER_PARTITION = 1_000

# Synthetic epoch — gives every slot a real timestamp for feature extraction
SYNTHETIC_EPOCH = datetime(2020, 1, 1)


# ── Geo-bin encoding / decoding ──────────────────────────────────────────────

def encode_geo_bin(x: int, y: int) -> int:
    """Row-major encoding of 1-based (x, y) into a single geo_bin integer."""
    return (y - 1) * GRID_WIDTH + (x - 1)


def decode_geo_bin(geo_bin: int) -> tuple[int, int]:
    """Inverse of encode_geo_bin: recover 1-based (x, y) from a geo_bin int."""
    x = (geo_bin % GRID_WIDTH) + 1
    y = (geo_bin // GRID_WIDTH) + 1
    return x, y


def slot_to_timestamp(day: int, slot: int) -> pd.Timestamp:
    """Convert (day, slot) to a synthetic UTC-anchored timestamp."""
    offset = (day * SLOTS_PER_DAY + slot) * SLOT_SECONDS
    return pd.Timestamp(SYNTHETIC_EPOCH + timedelta(seconds=offset))


# ── Core merging logic ────────────────────────────────────────────────────────

def merge_slots_to_stays(
    df: pd.DataFrame,
    split_day: int = SPLIT_DAY,
    tail: dict | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Merge consecutive same-agent same-cell slots into stay rows.

    Args:
        df:        Sorted observations with columns [uid, d, t, x, y].
        split_day: Day boundary separating past from future.
        tail:      Carry-over state from the previous chunk (None on first call).

    Returns:
        (stays DataFrame, new tail state for next chunk)
        The tail is the last in-progress stay that may continue into the next chunk.
    """
    stays: list[dict] = []

    for row in df.itertuples(index=False):
        uid, d, t, x, y = int(row.uid), int(row.d), int(row.t), int(row.x), int(row.y)
        geo_bin     = encode_geo_bin(x, y)
        global_slot = d * SLOTS_PER_DAY + t
        period      = "past" if d < split_day else "future"

        # Extend the current stay if same agent, same cell, same period, consecutive slot
        if (
            tail is not None
            and tail["uid"]      == uid
            and tail["geo_bin"]  == geo_bin
            and tail["period"]   == period
            and global_slot      == tail["end_slot"] + 1
        ):
            tail["end_slot"] = global_slot
            continue

        # Flush completed stay
        if tail is not None:
            stays.append(_finalize_stay(tail))

        # Start a new stay
        tail = {
            "uid":        uid,
            "geo_bin":    geo_bin,
            "x":          x,
            "y":          y,
            "period":     period,
            "start_slot": global_slot,
            "end_slot":   global_slot,
        }

    return pd.DataFrame(stays), tail


def flush_tail(tail: dict | None) -> pd.DataFrame:
    """Emit the final in-progress stay after all chunks are processed."""
    if tail is None:
        return pd.DataFrame()
    return pd.DataFrame([_finalize_stay(tail)])


def _finalize_stay(tail: dict) -> dict:
    """Convert a tail accumulator dict into one output stay row."""
    n_slots    = tail["end_slot"] - tail["start_slot"] + 1
    start_day, start_slot = divmod(tail["start_slot"], SLOTS_PER_DAY)
    return {
        "agent":          tail["uid"],
        "geo_bin":        tail["geo_bin"],
        "timestamp":      slot_to_timestamp(start_day, start_slot),
        "seconds_in_bin": n_slots * SLOT_SECONDS,
        "latitude":       nan,   # YJMob has no lat/lon — x/y decoded on demand
        "longitude":      nan,
        "agent_group":    tail["uid"] // NUM_AGENTS_PER_PARTITION,
        "period":         tail["period"],
    }


# ── Post-processing ───────────────────────────────────────────────────────────

def compute_london(stays: pd.DataFrame) -> pd.DataFrame:
    """
    Rank each agent's geo-bins by total duration (0 = most visited / home proxy).

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