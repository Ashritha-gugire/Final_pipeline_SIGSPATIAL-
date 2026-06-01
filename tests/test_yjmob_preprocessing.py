"""
tests/test_yjmob_preprocessing.py
===================================
Smoke test: verify stay merging, split boundary, and london ranking.

Run with:
    python tests/test_yjmob_preprocessing.py
    # or via pytest:
    pytest tests/test_yjmob_preprocessing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.yjmob_stay_sequence_preprocessing import (
    compute_london,
    flush_tail,
    merge_slots_to_stays,
    decode_geo_bin,
    encode_geo_bin,
    slot_to_timestamp,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_chunks() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two chunks to exercise cross-chunk merging and period boundary splits."""
    chunk_one = pd.DataFrame([
        {"uid": 10, "d": 59, "t": 46, "x": 5, "y": 7},  # past,   slot 2878
        {"uid": 10, "d": 59, "t": 47, "x": 5, "y": 7},  # past,   slot 2879 → merge
        {"uid": 10, "d": 60, "t":  0, "x": 5, "y": 7},  # future, slot 2880 → new stay (period break)
    ])
    chunk_two = pd.DataFrame([
        {"uid": 10, "d": 60, "t":  1, "x": 5, "y": 7},  # future, slot 2881 → merge
        {"uid": 10, "d": 60, "t":  3, "x": 5, "y": 7},  # future, slot 2883 → new stay (gap)
        {"uid": 42, "d":  0, "t":  0, "x": 2, "y": 3},  # past,   new agent
        {"uid": 42, "d":  0, "t":  1, "x": 2, "y": 3},  # past,   → merge
    ])
    return chunk_one, chunk_two


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_geo_bin_roundtrip():
    """encode → decode must be identity for all valid (x, y)."""
    for x in [1, 50, 100, 200]:
        for y in [1, 50, 100, 200]:
            assert decode_geo_bin(encode_geo_bin(x, y)) == (x, y), \
                f"Roundtrip failed for ({x}, {y})"
    print("PASSED test_geo_bin_roundtrip")


def test_slot_to_timestamp():
    """Day 0 slot 0 → synthetic epoch; day 1 slot 0 → +48 slots = +86400s."""
    t0 = slot_to_timestamp(0, 0)
    t1 = slot_to_timestamp(1, 0)
    assert (t1 - t0).total_seconds() == 86_400, "Day increment wrong"
    print("PASSED test_slot_to_timestamp")


def test_stay_merging():
    """Cross-chunk merging, period boundary, and gap splitting."""
    chunk_one, chunk_two = build_chunks()

    stays_1, tail = merge_slots_to_stays(chunk_one, split_day=60, tail=None)
    stays_2, tail = merge_slots_to_stays(chunk_two, split_day=60, tail=tail)
    stays_3       = flush_tail(tail)
    stays = pd.concat([stays_1, stays_2, stays_3], ignore_index=True)

    # Should produce exactly 4 stays
    assert len(stays) == 4, f"Expected 4 stays, got {len(stays)}"

    # Agent order
    assert list(stays["agent"]) == [10, 10, 10, 42], \
        f"Agent order wrong: {list(stays['agent'])}"

    # Period split
    assert list(stays["period"]) == ["past", "future", "future", "past"], \
        f"Period split wrong: {list(stays['period'])}"

    # Durations (in seconds)
    assert list(stays["seconds_in_bin"]) == [3600, 3600, 1800, 3600], \
        f"Durations wrong: {list(stays['seconds_in_bin'])}"

    print("PASSED test_stay_merging")


def test_london_ranking():
    """london=0 must be the most-visited geo_bin per agent."""
    chunk_one, chunk_two = build_chunks()
    stays_1, tail = merge_slots_to_stays(chunk_one, split_day=60, tail=None)
    stays_2, tail = merge_slots_to_stays(chunk_two, split_day=60, tail=tail)
    stays_3       = flush_tail(tail)
    stays = pd.concat([stays_1, stays_2, stays_3], ignore_index=True)

    london = compute_london(stays)
    stays  = stays.merge(london, on=["agent", "geo_bin"], how="left")

    # Agent 10: one unique bin → london=0
    assert stays.loc[0, "london"] == 0, "Agent 10 home rank wrong"
    # Agent 42: one unique bin → london=0
    assert stays.loc[3, "london"] == 0, "Agent 42 home rank wrong"

    print("PASSED test_london_ranking")


def test_full_pipeline_output():
    """End-to-end: output has all required canonical columns."""
    chunk_one, chunk_two = build_chunks()
    stays_1, tail = merge_slots_to_stays(chunk_one, split_day=60, tail=None)
    stays_2, tail = merge_slots_to_stays(chunk_two, split_day=60, tail=tail)
    stays_3       = flush_tail(tail)
    stays = pd.concat([stays_1, stays_2, stays_3], ignore_index=True)
    london = compute_london(stays)
    stays  = stays.merge(london, on=["agent", "geo_bin"], how="left")

    required = {"agent", "geo_bin", "timestamp", "seconds_in_bin",
                "latitude", "longitude", "agent_group", "period", "london"}
    missing  = required - set(stays.columns)
    assert not missing, f"Missing columns: {missing}"

    print("PASSED test_full_pipeline_output")


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("YJMob Preprocessing Smoke Tests")
    print("=" * 55)

    test_geo_bin_roundtrip()
    test_slot_to_timestamp()
    test_stay_merging()
    test_london_ranking()
    test_full_pipeline_output()

    print("=" * 55)
    print("ALL TESTS PASSED")
    print("=" * 55)

    # Pretty-print the stays for visual inspection
    chunk_one, chunk_two = build_chunks()
    stays_1, tail = merge_slots_to_stays(chunk_one, split_day=60, tail=None)
    stays_2, tail = merge_slots_to_stays(chunk_two, split_day=60, tail=tail)
    stays_3       = flush_tail(tail)
    stays = pd.concat([stays_1, stays_2, stays_3], ignore_index=True)
    london = compute_london(stays)
    stays  = stays.merge(london, on=["agent", "geo_bin"], how="left")
    print()
    print(stays[["agent", "geo_bin", "period", "seconds_in_bin", "london"]].to_string(index=False))


if __name__ == "__main__":
    main()