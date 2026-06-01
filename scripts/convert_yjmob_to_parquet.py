"""
scripts/convert_yjmob_to_parquet.py
=====================================
Convert a raw YJMob100K CSV (sorted by uid, d, t) into two flat parquet files:
    data/yjmob/processed/past_stays.parquet
    data/yjmob/processed/future_stays.parquet

Memory-efficient: processes one agent_group (~1000 agents) at a time.
Peak memory ≈ one group's stays in RAM.

Usage
─────
    python scripts/convert_yjmob_to_parquet.py \\
        data/yjmob/raw/yjmob100k-dataset2.csv.gz \\
        data/yjmob/processed \\
        --split-day 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.yjmob_stay_sequence_preprocessing import (
    NUM_AGENTS_PER_PARTITION,
    SPLIT_DAY,
    compute_london,
    flush_tail,
    merge_slots_to_stays,
)

# ── Config ────────────────────────────────────────────────────────────────────

DTYPES = {"uid": "int32", "d": "int16", "t": "int16", "x": "int16", "y": "int16"}

OUTPUT_COLUMNS = [
    "agent", "geo_bin", "timestamp", "seconds_in_bin",
    "latitude", "longitude", "london", "agent_group",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv",  type=Path, help="Path to raw YJMob CSV or CSV.gz")
    parser.add_argument("output_dir", type=Path, help="Output directory for flat parquet files")
    parser.add_argument("--chunk-size", type=int, default=2_000_000,
                        help="CSV rows per chunk (default: 2M)")
    parser.add_argument("--split-day",  type=int, default=SPLIT_DAY,
                        help=f"Day boundary past/future (default: {SPLIT_DAY})")
    return parser.parse_args()


def process_group(stays: pd.DataFrame) -> pd.DataFrame:
    """Attach london rank and return canonical columns for one agent group."""
    if stays.empty:
        return stays
    london = compute_london(stays)
    stays  = stays.merge(london, on=["agent", "geo_bin"], how="left")
    return stays[OUTPUT_COLUMNS + ["period"]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate all stays in memory (two lists: past / future)
    # For very large files this is still manageable because we process
    # group-by-group and only concatenate at the end.
    past_parts:   list[pd.DataFrame] = []
    future_parts: list[pd.DataFrame] = []

    current_group: int | None = None
    group_stays:   list[pd.DataFrame] = []
    tail = None

    total_stays = total_agents = past_count = future_count = 0

    def flush_group() -> None:
        nonlocal total_stays, total_agents, past_count, future_count
        if not group_stays:
            return
        stays = pd.concat(group_stays, ignore_index=True)
        stays = process_group(stays)
        total_stays  += len(stays)
        total_agents += stays["agent"].nunique()

        past_part   = stays[stays["period"] == "past"].drop(columns="period")
        future_part = stays[stays["period"] == "future"].drop(columns="period")
        past_count   += len(past_part)
        future_count += len(future_part)

        if not past_part.empty:
            past_parts.append(past_part)
        if not future_part.empty:
            future_parts.append(future_part)

        group_stays.clear()

    print(f"Reading {args.input_csv} ...")
    for chunk in pd.read_csv(args.input_csv, dtype=DTYPES, chunksize=args.chunk_size):
        stays, tail = merge_slots_to_stays(chunk, split_day=args.split_day, tail=tail)
        if stays.empty:
            continue

        for ag, ag_df in stays.groupby("agent_group", sort=True):
            ag = int(ag)
            if current_group is not None and ag != current_group:
                flush_group()
            current_group = ag
            group_stays.append(ag_df)

    # Flush final in-progress stay
    final = flush_tail(tail)
    if not final.empty:
        ag = int(final["agent_group"].iloc[0])
        if current_group is not None and ag != current_group:
            flush_group()
        current_group = ag
        group_stays.append(final)

    flush_group()

    # ── Write flat parquet files ───────────────────────────────────────────
    print("Writing flat parquet files...")
    past_path   = args.output_dir / "past_stays.parquet"
    future_path = args.output_dir / "future_stays.parquet"

    pd.concat(past_parts,   ignore_index=True).to_parquet(past_path,   index=False)
    pd.concat(future_parts, ignore_index=True).to_parquet(future_path, index=False)

    # ── Write manifest ────────────────────────────────────────────────────
    manifest = {
        "source":         str(args.input_csv),
        "split_day":      args.split_day,
        "total_stays":    total_stays,
        "past_stays":     past_count,
        "future_stays":   future_count,
        "unique_agents":  total_agents,
        "past_parquet":   str(past_path),
        "future_parquet": str(future_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nDone!")
    print(f"  Total stays   : {total_stays:,}")
    print(f"  Past stays    : {past_count:,}  → {past_path}")
    print(f"  Future stays  : {future_count:,} → {future_path}")
    print(f"  Unique agents : {total_agents:,}")
    print(f"  Manifest      : {manifest_path}")


if __name__ == "__main__":
    main()