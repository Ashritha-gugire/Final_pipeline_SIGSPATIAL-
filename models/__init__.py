"""
models/
========
All anomaly scoring models for the SIGSPATIAL 2026 benchmark.

Unsupervised agent-level models (18-dim feature vector)
────────────────────────────────────────────────────────
    GMMScorer              density    sklearn
    IsolationForestScorer  ensemble   sklearn
    KNNScorer              proximity  sklearn
    DAGMMScorer            deep       pytorch (sklearn fallback)
    USADScorer             deep       pytorch (sklearn fallback)

Sequence models (7-dim stay-level vectors)
───────────────────────────────────────────
    LSTMADScorer           sequence   pytorch
    LMTADScorer            sequence   pytorch

Supervised reference (labels required)
───────────────────────────────────────
    XGBoostScorer          supervised xgboost (5-fold OOF)
"""

from models.anomaly_scorer  import AnomalyScorer
from models.sklearn_scorers import (
    GMMScorer, GMMDiagCovScorer,
    IsolationForestScorer, KNNScorer,
)
from models.xgboost_scorer import XGBoostScorer

DAGMMScorer = USADScorer = LSTMADScorer = LMTADScorer = None

try:
    from models.dagmm_scorer  import DAGMMScorer
except Exception as _e:
    import logging; logging.getLogger(__name__).warning(f"DAGMMScorer unavailable: {_e}")

try:
    from models.usad_scorer   import USADScorer
except Exception as _e:
    import logging; logging.getLogger(__name__).warning(f"USADScorer unavailable: {_e}")

try:
    from models.lstmad_scorer import LSTMADScorer
except Exception as _e:
    import logging; logging.getLogger(__name__).warning(f"LSTMADScorer unavailable: {_e}")

try:
    from models.lmtad_scorer  import LMTADScorer
except Exception as _e:
    import logging; logging.getLogger(__name__).warning(f"LMTADScorer unavailable: {_e}")


UNSUPERVISED_MODELS: dict = {
    "GMM":              GMMScorer,
    "Isolation Forest": IsolationForestScorer,
    "KNN":              KNNScorer,
}
if DAGMMScorer: UNSUPERVISED_MODELS["DAGMM"] = DAGMMScorer
if USADScorer:  UNSUPERVISED_MODELS["USAD"]  = USADScorer

SEQUENCE_MODELS: dict = {}
if LSTMADScorer: SEQUENCE_MODELS["LSTM-AD"] = LSTMADScorer
if LMTADScorer:  SEQUENCE_MODELS["LM-TAD"]  = LMTADScorer

DEFAULT_HYPERPARAMS: dict = {
    "GMM":              {"n_components": 10, "covariance_type": "diag", "n_init": 5, "random_state": 42},
    "Isolation Forest": {"n_estimators": 500, "contamination": "auto", "random_state": 42},
    "KNN":              {"k": 50},
    "DAGMM":            {"n_components": 8, "latent_dim": 4, "epochs": 200, "lr": 1e-3,
                         "lambda1": 0.001, "lambda2": 0.0005, "batch_size": 256, "random_state": 42},
    "USAD":             {"latent_dim": 16, "epochs": 50, "lr": 1e-3, "batch_size": 256,
                         "alpha": 0.3, "beta": 0.7, "random_state": 42},
    "LSTM-AD":          {"hidden": 128, "n_layers": 2, "t_pred": 3, "epochs": 100,
                         "lr": 1e-3, "batch_size": 64, "max_len": 60, "random_state": 42},
    "LM-TAD":           {"max_vocab": 5_000, "embed_dim": 64, "n_heads": 4, "n_layers": 3,
                         "max_len": 60, "epochs": 75, "lr": 1e-3, "batch_size": 128, "random_state": 42},
    "XGBoost†":         {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                         "subsample": 0.8, "colsample": 0.8, "n_splits": 5, "random_state": 42},
}

MODEL_FAMILY: dict = {
    "GMM": "density", "Isolation Forest": "tree", "KNN": "proximity",
    "DAGMM": "deep ", "USAD": "deep ",
    "LSTM-AD": "sequence", "LM-TAD": "sequence",
    "XGBoost†": " supervised",
}


def build_unsupervised_scorers(include_deep: bool = True, custom_params: dict = None) -> dict:
    """Instantiate all 5 unsupervised scorers with paper hyperparameters."""
    import logging
    log = logging.getLogger(__name__)
    params = {k: dict(v) for k, v in DEFAULT_HYPERPARAMS.items()}
    if custom_params:
        for name, overrides in custom_params.items():
            if name in params:
                params[name].update(overrides)
    scorers = {}
    for name, cls in UNSUPERVISED_MODELS.items():
        if not include_deep and name in ("DAGMM", "USAD"):
            continue
        if cls is None:
            continue
        valid_keys = set(cls.__init__.__code__.co_varnames)
        kw = {k: v for k, v in params.get(name, {}).items() if k in valid_keys}
        try:
            scorers[name] = cls(**kw)
        except Exception as e:
            log.warning(f"Could not instantiate {name}: {e}")
    return scorers