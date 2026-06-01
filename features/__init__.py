"""
features/
=========
Unified feature extraction for the SIGSPATIAL 2026 anomaly detection pipeline.
Works identically on NUMOSIM (lat/lon) and YJMob100K (grid x/y).

Public API
──────────
    from features import FeatureBuilder18Dim, SequenceFeatureBuilder, FEATURE_NAMES, SEQ_FEATURE_COLS

    # 18-dim common feature vector (Table 1)
    builder  = FeatureBuilder18Dim(past_stays, future_stays, is_yjmob=False, poi=poi_df)
    feat_df  = builder.compute_all()          # shape: (n_agents, 18)

    # 7-dim sequence features (LSTM-AD / LM-TAD)
    seq      = SequenceFeatureBuilder(stays, is_yjmob=False, poi=poi_df)
    seqs     = seq.build_sequences()          # Dict[agent_id → np.ndarray (T, 7)]
    flat_df  = seq.build_flat()               # flat DataFrame for inspection
"""

from features.features_18dim   import FeatureBuilder18Dim, FEATURE_NAMES
from features.sequence_features import SequenceFeatureBuilder, SEQ_FEATURE_COLS
from features.distance_utils    import haversine_np, euclidean_np, attach_coords

__all__ = [
    "FeatureBuilder18Dim",
    "SequenceFeatureBuilder",
    "FEATURE_NAMES",
    "SEQ_FEATURE_COLS",
    "haversine_np",
    "euclidean_np",
    "attach_coords",
]