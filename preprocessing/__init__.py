"""
preprocessing/
===============
Raw → canonical stay-sequence conversion for both datasets.

    from preprocessing.numosim_preprocessing import NUMOSIMPreprocessor
    from preprocessing.yjmob_stay_sequence   import merge_slots_to_stays, compute_london

Canonical stay-sequence schema (shared by both datasets)
─────────────────────────────────────────────────────────
    agent           int       unique agent identifier
    geo_bin         int       location identifier
    timestamp       datetime  stay start time
    seconds_in_bin  int       stay duration in seconds
    latitude        float     POI latitude  (NaN for YJMob — x/y decoded on demand)
    longitude       float     POI longitude (NaN for YJMob)
    london          int       frequency rank per agent (0 = home proxy)
    period          str       'past' or 'future'
"""

from preprocessing.numosim_preprocessing import NUMOSIMPreprocessor, compute_london as numosim_compute_london
from preprocessing.yjmob_stay_sequence_preprocessing   import (
    merge_slots_to_stays,
    flush_tail,
    compute_london as yjmob_compute_london,
    encode_geo_bin,
    decode_geo_bin,
    slot_to_timestamp,
    SPLIT_DAY,
    SLOT_SECONDS,
    SLOTS_PER_DAY,
    GRID_WIDTH,
)

__all__ = [
    "NUMOSIMPreprocessor",
    "merge_slots_to_stays",
    "flush_tail",
    "encode_geo_bin",
    "decode_geo_bin",
    "slot_to_timestamp",
    "SPLIT_DAY",
    "SLOT_SECONDS",
    "SLOTS_PER_DAY",
    "GRID_WIDTH",
]