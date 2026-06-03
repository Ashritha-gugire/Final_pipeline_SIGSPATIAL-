"""
main.py
========
Unified entry point for the SIGSPATIAL 2026 anomaly detection pipeline.

Runs the full benchmark for NUMOSIM, YJMob100K, or both datasets with
a single command. All steps share the same code — only data paths and
the is_yjmob flag differ between datasets.

Pipeline stages
---------------
  1. Preprocessing   raw files -> canonical past/future stays parquet
  2. Features        stays -> 18-dim feature matrix (cached after first run)
  3. Models          fit unsupervised scorers on normal agents, score all
  4. Evaluation      AUC, AP, F1, bootstrap 95% CIs, results table
  5. Plots           grid curves, bar chart, CI table, rate line plot
  6. Sequence models LSTM-AD + LM-TAD  (--sequence-models flag)
  7. Sensitivity     NUMOSIM at 5%% and 10%% contamination (--sensitivity flag)

Usage examples
--------------
  python main.py --dataset numosim
  python main.py --dataset numosim --skip-preprocessing --skip-features --sequence-models --sensitivity ( for full run at 1% 5% 10%)
  python main.py --dataset yjmob
  python main.py --dataset all
  python main.py --dataset numosim --skip-preprocessing
  python main.py --dataset numosim --skip-preprocessing --skip-features
  python main.py --dataset numosim --skip-preprocessing --skip-features --skip-training
  python main.py --dataset all --sequence-models
  python main.py --dataset all --no-deep
  python main.py --dataset numosim --sensitivity
  python main.py --dataset all --sensitivity --sequence-models --no-bootstrap
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import importlib
import logging
import subprocess
import sys
import time
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Project imports ───────────────────────────────────────────────────────────
from features.features_18dim import FeatureBuilder18Dim
from features.features_18dim import FEATURE_NAMES as FEATURE_COLS  # single source of truth
from features.sequence_features import SequenceFeatureBuilder, SEQ_FEATURE_COLS
from evaluation.metrics import build_results_table
from evaluation.plots import (
    make_grid_figure,
    make_bar_chart,
    make_ci_table,
    make_rate_line_plot,
)
from preprocessing.numosim_preprocessing import NUMOSIMPreprocessor
from models.sklearn_scorers import GMMScorer, IsolationForestScorer, KNNScorer
from models.xgboost_scorer import XGBoostScorer


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# YJMob contamination sweep rates
CONTAMINATION_RATES  = [0.01, 0.05, 0.10]
CONTAMINATION_LABELS = ["1%", "5%", "10%"]

# NUMOSIM sensitivity rates (supplementary — natural rate is primary)
SENSITIVITY_RATES  = [0.05, 0.10]
SENSITIVITY_LABELS = ["5%", "10%"]

TOP_K_LIST       = [50, 100, 200]
N_BOOTSTRAP      = 1000
RANDOM_SEED      = 42
MAX_AGENTS_YJMOB = 10_000   # agents per YJMob contamination sample
N_NORMAL_NUMOSIM = 20_000   # normal agents to sample for NUMOSIM

# Model family labels for results tables
MODEL_FAMILY = {
    "GMM":              "density",
    "Isolation Forest": "ensemble",
    "KNN":              "proximity",
    "DAGMM":            "deep",
    "USAD":             "deep",
    "LSTM-AD":          "sequence",
    "LM-TAD":           "sequence",
    "XGBoost†":         "supervised",
}

# Hardcoded tuned hyperparameters — validated, no re-tuning on test labels
HYPERPARAMS = {
    "GMM":              {"n_components": 10, "covariance_type": "diag",
                         "n_init": 5, "random_state": RANDOM_SEED},
    "Isolation Forest": {"n_estimators": 500, "contamination": "auto",
                         "random_state": RANDOM_SEED},
    "KNN":              {"k": 50},
    "DAGMM":            {"n_components": 8, "latent_dim": 4, "epochs": 200,
                         "lr": 1e-3, "lambda1": 0.001, "lambda2": 0.0005,
                         "batch_size": 256, "random_state": RANDOM_SEED},
    "USAD":             {"latent_dim": 16, "epochs": 50, "lr": 1e-3,
                         "batch_size": 256, "alpha": 0.3, "beta": 0.7,
                         "random_state": RANDOM_SEED},
    "LSTM-AD":          {"hidden": 128, "n_layers": 2, "t_pred": 3,
                         "epochs": 100, "lr": 1e-3, "batch_size": 64,
                         "max_len": 60, "random_state": RANDOM_SEED},
    "LM-TAD":           {"max_vocab": 5_000, "embed_dim": 64, "n_heads": 4,
                         "n_layers": 3, "max_len": 60, "epochs": 75,
                         "lr": 1e-3, "batch_size": 128,
                         "random_state": RANDOM_SEED},
    "XGBoost†":         {"n_estimators": 300, "max_depth": 4,
                         "learning_rate": 0.05, "subsample": 0.8,
                         "colsample": 0.8, "n_splits": 5,
                         "random_state": RANDOM_SEED},
}


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run.log"
    fmt      = "%(asctime)s  %(levelname)-8s  %(name)s - %(message)s"
    datefmt  = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt,
                        datefmt=datefmt, handlers=handlers)
    return logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clip_by_train_percentiles(
    scores:       np.ndarray,
    train_scores: np.ndarray,
) -> np.ndarray:
    """Clip scores to 1st-99th percentile of training scores."""
    return np.clip(scores,
                   np.percentile(train_scores, 1),
                   np.percentile(train_scores, 99))


def _orient(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Ensure higher score = more anomalous."""
    s = np.where(np.isnan(scores), 0.0, scores.astype(float))
    return -s if roc_auc_score(labels, s) < 0.5 else s


def _print_summary(
    df:  pd.DataFrame,
    label: str,
    log: logging.Logger,
) -> None:
    log.info(f"\n{'='*65}")
    log.info(f"RESULTS - {label}")
    log.info(f"{'='*65}")
    cols = ["Algorithm", "AUC", "AP", "F1"]
    if "AUC_95CI" in df.columns:
        cols.insert(2, "AUC_95CI")
    log.info("\n" + df[cols].to_string(index=False))
    log.info("=" * 65)


def _build_eval_matrices(
    feat_df:   pd.DataFrame,
    avail_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build scaled feature matrices and agent arrays from a labelled feature DataFrame.
    Scaler is always fit on normal agents only — no leakage.

    Returns: X_norm_sc, X_all_sc, labels, normal_agents, all_agents
    """
    labels        = feat_df["is_anomalous"].values
    normal_mask   = labels == 0
    all_agents    = feat_df["agent"].values
    normal_agents = feat_df.loc[normal_mask, "agent"].values

    X_all     = feat_df[avail_cols].fillna(0).values
    scaler    = StandardScaler()
    scaler.fit(X_all[normal_mask])          # fit on normals only
    X_all_sc  = scaler.transform(X_all)
    X_norm_sc = X_all_sc[normal_mask]

    return X_norm_sc, X_all_sc, labels, normal_agents, all_agents


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 - Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def stage_preprocessing_numosim(cfg: dict, log: logging.Logger) -> dict:
    """
    NUMOSIM raw parquets -> canonical past/future stays.
    Always runs fresh — never loads stale cached data.
    Use --skip-preprocessing to load from disk instead.
    """
    log.info("[NUMOSIM] Running preprocessing...")
    prep = NUMOSIMPreprocessor(
        data_dir    = str(cfg["raw_dir"]),
        output_dir  = str(cfg["processed_dir"]),
        n_normal    = N_NORMAL_NUMOSIM,
        random_seed = RANDOM_SEED,
    )
    results = prep.run()
    log.info("[NUMOSIM] Preprocessing complete.")
    return results


def stage_preprocessing_yjmob(cfg: dict, log: logging.Logger) -> dict:
    """
    YJMob raw CSVs -> canonical past/future stays (flat parquet).
    Always runs fresh — never loads stale cached data.
    Use --skip-preprocessing to load from disk instead.
    """
    log.info("[YJMob] Running preprocessing (~30 min)...")

    for dataset_num, label in [(1, "nonanom"), (2, "anom")]:
        csv_path = cfg["raw_dir"] / f"yjmob100k-dataset{dataset_num}.csv"
        out_dir  = cfg["processed_dir"] / label

        if not csv_path.exists():
            raise FileNotFoundError(
                f"YJMob raw CSV not found: {csv_path}\n"
                f"Place yjmob100k-dataset{dataset_num}.csv in {cfg['raw_dir']}"
            )

        log.info(f"[YJMob] Converting dataset{dataset_num} -> {out_dir}")
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts" / "convert_yjmob_to_parquet.py"),
            str(csv_path), str(out_dir),
            "--split-day", "60",
        ], check=True)

    log.info("[YJMob] Preprocessing complete.")
    return _load_processed_yjmob(cfg)


def _load_processed_numosim(cfg: dict) -> dict:
    """Load already-processed NUMOSIM stays from disk."""
    proc = cfg["processed_dir"]
    return {
        "past_stays":   pd.read_parquet(proc / "past_stays.parquet"),
        "future_stays": pd.read_parquet(proc / "future_stays.parquet"),
        "test_anom":    pd.read_parquet(proc / "test_anom.parquet"),
        "poi":          pd.read_parquet(proc / "poi.parquet"),
    }


def _load_processed_yjmob(cfg: dict) -> dict:
    """Load already-processed YJMob stays from disk."""
    proc = cfg["processed_dir"]
    return {
        "past_nonanom":   pd.read_parquet(proc / "nonanom" / "past_stays.parquet"),
        "future_nonanom": pd.read_parquet(proc / "nonanom" / "future_stays.parquet"),
        "past_anom":      pd.read_parquet(proc / "anom"    / "past_stays.parquet"),
        "future_anom":    pd.read_parquet(proc / "anom"    / "future_stays.parquet"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 - Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def stage_features(
    past_stays:    pd.DataFrame,
    future_stays:  pd.DataFrame,
    is_yjmob:      bool,
    cache_path:    Path,
    poi:           pd.DataFrame,
    log:           logging.Logger,
    dataset_label: str = "",
) -> pd.DataFrame:
    """
    Compute 18-dim feature vector. Loads from cache if available.
    Returns DataFrame with columns = FEATURE_COLS + ['agent'].
    """
    if cache_path.exists():
        log.info(f"[{dataset_label}] Features cache found - loading.")
        return pd.read_parquet(cache_path)

    log.info(f"[{dataset_label}] Computing 18-dim features (~30-45 min first run)...")
    builder = FeatureBuilder18Dim(
        past_stays   = past_stays,
        future_stays = future_stays,
        is_yjmob     = is_yjmob,
        poi          = poi,
    )
    feat_df = builder.compute_all()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(cache_path, index=False)
    log.info(f"[{dataset_label}] Features cached -> {cache_path}")
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 - Model Training and Scoring
# ─────────────────────────────────────────────────────────────────────────────

def stage_models(
    X_normal:      np.ndarray,
    X_all:         np.ndarray,
    labels:        np.ndarray,
    checkpoints:   Path,
    args,
    log:           logging.Logger,
    dataset_label: str = "",
) -> dict[str, np.ndarray]:
    """
    Fit each unsupervised scorer on X_normal, score X_all.
    Returns dict: model_name -> oriented anomaly scores (len = len(X_all)).
    """
    checkpoints.mkdir(parents=True, exist_ok=True)
    scores = {}

    # ── Sklearn models ────────────────────────────────────────────────────────
    sklearn_models = {
        "GMM":              GMMScorer(**HYPERPARAMS["GMM"]),
        "Isolation Forest": IsolationForestScorer(**HYPERPARAMS["Isolation Forest"]),
        "KNN":              KNNScorer(**HYPERPARAMS["KNN"]),
    }
    for name, scorer in sklearn_models.items():
        ckpt = checkpoints / name.replace(" ", "_").lower()
        if args.skip_training and ckpt.exists():
            log.info(f"[{dataset_label}] Loading {name} from checkpoint...")
            scorer.load(str(ckpt))
        else:
            log.info(f"[{dataset_label}] Fitting {name}...")
            t0 = time.time()
            scorer.fit(X_normal)
            scorer.save(str(ckpt))
            log.info(f"[{dataset_label}] {name} done in {time.time()-t0:.1f}s")
        scores[name] = _orient(scorer.score(X_all), labels)

    # ── Deep models (DAGMM, USAD) ─────────────────────────────────────────────
    if not args.no_deep:
        for name, module in [("DAGMM", "models.dagmm_scorer"),
                              ("USAD",  "models.usad_scorer")]:
            try:
                mod    = importlib.import_module(module)
                cls    = getattr(mod, f"{name}Scorer")
                scorer = cls(**HYPERPARAMS[name])
                ckpt   = checkpoints / name.lower()

                if args.skip_training and ckpt.exists():
                    log.info(f"[{dataset_label}] Loading {name} from checkpoint...")
                    scorer.load(str(ckpt))
                    raw = scorer.score(X_all)
                else:
                    log.info(f"[{dataset_label}] Fitting {name} "
                             f"(epochs={HYPERPARAMS[name]['epochs']})...")
                    t0 = time.time()
                    scorer.fit(X_normal)
                    raw_train = scorer.score(X_normal)
                    raw       = _clip_by_train_percentiles(
                        scorer.score(X_all), raw_train
                    )
                    scorer.save(str(ckpt))
                    log.info(f"[{dataset_label}] {name} done in {time.time()-t0:.1f}s")

                scores[name] = _orient(raw, labels)

            except (ImportError, AttributeError) as e:
                log.warning(f"[{dataset_label}] {name} unavailable: {e}")

    # ── XGBoost supervised reference ──────────────────────────────────────────
    log.info(f"[{dataset_label}] Fitting XGBoost† (5-fold OOF)...")
    xgb = XGBoostScorer(**HYPERPARAMS["XGBoost†"])
    xgb.fit(X_all, labels)
    scores["XGBoost†"] = _orient(xgb.score(), labels)
    xgb.save(str(checkpoints / "xgboost"))

    return scores


def _reload_scores(
    checkpoints: Path,
    X_all:       np.ndarray,
    labels:      np.ndarray,
    args,
    log:         logging.Logger,
) -> dict[str, np.ndarray]:
    """Reload fitted scorers from checkpoints and re-score."""
    scores = {}
    for name, cls, slug in [
        ("GMM",              GMMScorer,             "gmm"),
        ("Isolation Forest", IsolationForestScorer, "isolation_forest"),
        ("KNN",              KNNScorer,             "knn"),
    ]:
        ckpt = checkpoints / slug
        if ckpt.exists():
            scorer = cls()
            scorer.load(str(ckpt))
            scores[name] = _orient(scorer.score(X_all), labels)
            log.info(f"  Reloaded {name}")

    if not args.no_deep:
        for name, module in [("DAGMM", "models.dagmm_scorer"),
                              ("USAD",  "models.usad_scorer")]:
            ckpt = checkpoints / name.lower()
            if ckpt.exists():
                try:
                    mod    = importlib.import_module(module)
                    scorer = getattr(mod, f"{name}Scorer")()
                    scorer.load(str(ckpt))
                    scores[name] = _orient(scorer.score(X_all), labels)
                    log.info(f"  Reloaded {name}")
                except Exception as e:
                    log.warning(f"  Could not reload {name}: {e}")

    xgb_ckpt = checkpoints / "xgboost"
    if xgb_ckpt.exists():
        xgb = XGBoostScorer()
        xgb.load(str(xgb_ckpt))
        scores["XGBoost†"] = _orient(xgb.score(), labels)
        log.info("  Reloaded XGBoost†")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 - Sequence Models
# ─────────────────────────────────────────────────────────────────────────────

def stage_sequence_models(
    past_stays:    pd.DataFrame,
    future_stays:  pd.DataFrame,
    normal_agents: np.ndarray,
    all_agents:    np.ndarray,
    labels:        np.ndarray,
    is_yjmob:      bool,
    checkpoints:   Path,
    args,
    log:           logging.Logger,
    dataset_label: str = "",
    poi:           pd.DataFrame = None,
) -> dict[str, np.ndarray]:
    """
    Fit LSTM-AD and LM-TAD on normal agent sequences.

    NOTE: past_stays and future_stays agent IDs must already match
    the IDs in normal_agents and all_agents. For YJMob, apply the
    max_id offset to stays DataFrames before calling this function.

    Returns dict: model_name -> oriented anomaly scores aligned to all_agents.
    """
    checkpoints.mkdir(parents=True, exist_ok=True)
    seq_scores = {}

    # ── LSTM-AD ───────────────────────────────────────────────────────────────
    try:
        from models.lstmad_scorer import LSTMADScorer

        normal_stays_df = future_stays[future_stays["agent"].isin(normal_agents)]
        log.info(f"[{dataset_label}] Fitting LSTM-AD "
                 f"({normal_stays_df['agent'].nunique():,} normal agents)...")
        t0     = time.time()
        lstmad = LSTMADScorer(**HYPERPARAMS["LSTM-AD"])
        lstmad.fit(stays_df=normal_stays_df, is_yjmob=is_yjmob, poi=poi)
        raw_train = lstmad.score(stays_df=normal_stays_df, agents=normal_agents)
        raw_all   = lstmad.score(stays_df=future_stays,    agents=all_agents)
        raw_all   = _clip_by_train_percentiles(raw_all, raw_train)
        lstmad.save(str(checkpoints / "lstmad"))
        log.info(f"[{dataset_label}] LSTM-AD done in {time.time()-t0:.1f}s")
        seq_scores["LSTM-AD"] = _orient(raw_all, labels)

    except ImportError as e:
        log.warning(f"[{dataset_label}] LSTMADScorer unavailable: {e}")

    # ── LM-TAD ────────────────────────────────────────────────────────────────
    try:
        from models.lmtad_scorer import LMTADScorer, build_vocab, build_lmtad_sequences

        log.info(f"[{dataset_label}] Building LM-TAD token sequences...")
        vocab      = build_vocab(past_stays, max_vocab=5_000)
        train_bins = set(past_stays["geo_bin"].unique())

        normal_stays  = future_stays[future_stays["agent"].isin(normal_agents)]
        train_seqs, _ = build_lmtad_sequences(
            normal_stays, vocab, train_bins,
            max_len=HYPERPARAMS["LM-TAD"]["max_len"],
        )
        test_seqs, test_ids = build_lmtad_sequences(
            future_stays, vocab, train_bins,
            max_len=HYPERPARAMS["LM-TAD"]["max_len"],
        )

        log.info(f"[{dataset_label}] Fitting LM-TAD ({len(train_seqs):,} normal seqs)...")
        t0    = time.time()
        lmtad = LMTADScorer(**HYPERPARAMS["LM-TAD"])
        lmtad.fit(train_seqs, vocab, train_bins)
        raw_scores = lmtad.score(test_seqs)
        lmtad.save(str(checkpoints / "lmtad"))
        log.info(f"[{dataset_label}] LM-TAD done in {time.time()-t0:.1f}s")

        id_to_score = dict(zip(test_ids, raw_scores))
        aligned     = np.array([id_to_score.get(int(a), 0.0) for a in all_agents])
        seq_scores["LM-TAD"] = _orient(aligned, labels)

    except ImportError as e:
        log.warning(f"[{dataset_label}] LMTADScorer unavailable: {e}")

    return seq_scores


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 - Evaluation and Plots
# ─────────────────────────────────────────────────────────────────────────────

def stage_evaluation(
    all_scores:    dict[str, np.ndarray],
    labels:        np.ndarray,
    rate_label:    str,
    results_dir:   Path,
    dataset_name:  str,
    args,
    log:           logging.Logger,
) -> pd.DataFrame:
    """Compute metrics + CIs, save CSV, generate grid figure and CI table."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    log.info(f"[{dataset_name}] Evaluating {len(all_scores)} models at {rate_label}...")

    n_bootstrap = N_BOOTSTRAP if not args.no_bootstrap else 0
    df = build_results_table(
        results     = {n: (MODEL_FAMILY.get(n, "unknown"), s)
                       for n, s in all_scores.items()},
        labels      = labels,
        rate_label  = rate_label,
        top_k_list  = TOP_K_LIST,
        n_bootstrap = n_bootstrap,
    )

    rate_slug = (rate_label.replace("%", "pct").replace(".", "")
                           .replace(" ", "_").replace("(", "").replace(")", ""))
    csv_path  = results_dir / f"results_{rate_slug}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"[{dataset_name}] Results saved -> {csv_path}")

    make_grid_figure(
        scores       = all_scores,
        labels       = labels,
        rate_label   = rate_label,
        dataset_name = dataset_name,
        out_path     = str(results_dir / "figures" / f"fig_grid_curves_{rate_slug}.png"),
    )
    make_ci_table(
        df, dataset_name,
        str(results_dir / "figures" / f"fig_ci_table_{rate_slug}.png"),
    )
    return df


def stage_summary_plots(
    all_results:  list[pd.DataFrame],
    results_dir:  Path,
    dataset_name: str,
    log:          logging.Logger,
) -> None:
    """Bar chart and rate line plot across all contamination rates."""
    if not all_results:
        return
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_results, ignore_index=True)

    make_bar_chart(combined, dataset_name,
                   str(results_dir / "figures" / "fig_bar_chart_all_rates.png"))
    log.info(f"[{dataset_name}] Bar chart saved.")

    if len(all_results) > 1:
        make_rate_line_plot(combined, dataset_name,
                            str(results_dir / "figures" / "fig_rate_line_plot.png"))
        log.info(f"[{dataset_name}] Rate line plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Shared contamination sweep
# ─────────────────────────────────────────────────────────────────────────────

def _run_contamination_sweep(
    feat_df:       pd.DataFrame,
    past_stays:    pd.DataFrame,
    future_stays:  pd.DataFrame,
    avail_cols:    list[str],
    rates:         list[float],
    rate_labels:   list[str],
    results_dir:   Path,
    dataset_name:  str,
    is_yjmob:      bool,
    max_id:        int,           # 0 for NUMOSIM, DS2 offset for YJMob
    n_anomalous:   int,           # fixed anomalous count (NUMOSIM) or 0 (YJMob)
    total_agents:  int,           # 0 for NUMOSIM (variable), MAX_AGENTS_YJMOB for YJMob
    args,
    log:           logging.Logger,
    poi:           pd.DataFrame = None,
) -> list[pd.DataFrame]:
    """
    Shared sweep loop used by NUMOSIM sensitivity and YJMob contamination.

    NUMOSIM mode (is_yjmob=False, max_id=0):
        - feat_df contains both normal and anomalous agents
        - n_anomalous is fixed (381), n_normal computed from rate
        - Stays DataFrames have no offset

    YJMob mode (is_yjmob=True, max_id>0):
        - feat_df contains both datasets with DS2 agent IDs already offset
        - n_anom + n_norm = total_agents (10,000)
        - Stays DataFrames for anomalous agents must be offset by max_id
          before passing to sequence models — handled inside this function

    Returns list of results DataFrames, one per rate.
    """
    all_rate_results = []

    for rate, rate_label in zip(rates, rate_labels):
        log.info("-" * 65)
        log.info(f"[{dataset_name}] Rate: {rate_label}")
        log.info("-" * 65)

        # ── Sample evaluation set ─────────────────────────────────────────────
        if is_yjmob:
            # YJMob: fixed total, split by rate
            n_anom = int(total_agents * rate)
            n_norm = total_agents - n_anom
            eval_df = pd.concat([
                feat_df[feat_df["is_anomalous"] == 0].sample(
                    n=min(n_norm, (feat_df["is_anomalous"] == 0).sum()),
                    random_state=RANDOM_SEED),
                feat_df[feat_df["is_anomalous"] == 1].sample(
                    n=min(n_anom, (feat_df["is_anomalous"] == 1).sum()),
                    random_state=RANDOM_SEED),
            ], ignore_index=True)
        else:
            # NUMOSIM: fixed anomalous, compute normal count from rate
            n_norm = int(n_anomalous / rate) - n_anomalous
            eval_df = pd.concat([
                feat_df[feat_df["is_anomalous"] == 0].sample(
                    n=min(n_norm, (feat_df["is_anomalous"] == 0).sum()),
                    random_state=RANDOM_SEED),
                feat_df[feat_df["is_anomalous"] == 1],
            ], ignore_index=True)

        # ── Build matrices ────────────────────────────────────────────────────
        X_norm_sc, X_all_sc, labels, normal_agents, all_agents = \
            _build_eval_matrices(eval_df, avail_cols)

        log.info(f"  {(labels==1).sum()} anomalous / {(labels==0).sum():,} normal "
                 f"/ {len(eval_df):,} total (rate={labels.mean()*100:.2f}%)")

        # ── Save canonical agent list ─────────────────────────────────────────
        rate_slug = rate_label.replace("%", "pct")
        results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"agent": all_agents, "y_true": labels}).to_parquet(
            results_dir / f"canonical_agents_{rate_slug}.parquet", index=False
        )

        # ── Models ────────────────────────────────────────────────────────────
        rate_ckpts = results_dir / "checkpoints" / rate_slug
        if not args.skip_training:
            all_scores = stage_models(
                X_normal      = X_norm_sc,
                X_all         = X_all_sc,
                labels        = labels,
                checkpoints   = rate_ckpts,
                args          = args,
                log           = log,
                dataset_label = f"{dataset_name}-{rate_label}",
            )
        else:
            all_scores = _reload_scores(rate_ckpts, X_all_sc, labels, args, log)

        # ── Sequence models ───────────────────────────────────────────────────
        if args.sequence_models:
            if is_yjmob:
                # For YJMob: filter stays to sampled agents, apply offset to anom
                norm_ids      = set(eval_df.loc[eval_df["is_anomalous"]==0, "agent"])
                orig_anom_ids = {a - max_id for a in
                                 eval_df.loc[eval_df["is_anomalous"]==1, "agent"]}

                past_norm_s   = past_stays[past_stays["agent"].isin(norm_ids)]
                future_norm_s = future_stays[future_stays["agent"].isin(norm_ids)]

                # The anom stays DataFrames passed in have original IDs
                # We need to find the right stays — but past_stays/future_stays
                # here are already the combined (nonanom + anom) DataFrames
                # Filter anom stays and apply offset
                past_anom_s   = past_stays[
                    past_stays["agent"].isin(orig_anom_ids)].copy()
                future_anom_s = future_stays[
                    future_stays["agent"].isin(orig_anom_ids)].copy()
                past_anom_s["agent"]   += max_id
                future_anom_s["agent"] += max_id

                seq_past   = pd.concat([past_norm_s,   past_anom_s],   ignore_index=True)
                seq_future = pd.concat([future_norm_s, future_anom_s], ignore_index=True)
            else:
                # NUMOSIM: stays have same IDs as feat_df, no offset needed
                sampled_agents = set(all_agents)
                seq_past   = past_stays[past_stays["agent"].isin(sampled_agents)]
                seq_future = future_stays[future_stays["agent"].isin(sampled_agents)]

            seq_scores = stage_sequence_models(
                past_stays    = seq_past,
                future_stays  = seq_future,
                normal_agents = normal_agents,
                all_agents    = all_agents,
                labels        = labels,
                is_yjmob      = is_yjmob,
                checkpoints   = rate_ckpts / "sequence",
                args          = args,
                log           = log,
                dataset_label = f"{dataset_name}-{rate_label}",
                poi           = poi,
            )
            all_scores.update(seq_scores)

        # ── Evaluation ────────────────────────────────────────────────────────
        rate_results = stage_evaluation(
            all_scores   = all_scores,
            labels       = labels,
            rate_label   = rate_label,
            results_dir  = results_dir,
            dataset_name = dataset_name,
            args         = args,
            log          = log,
        )
        all_rate_results.append(rate_results)
        _print_summary(rate_results, f"{dataset_name} {rate_label}", log)

    return all_rate_results


# ─────────────────────────────────────────────────────────────────────────────
# Dataset runners
# ─────────────────────────────────────────────────────────────────────────────

def run_numosim(args, log: logging.Logger) -> None:
    """Full NUMOSIM pipeline at the natural anomaly rate (~1.87%)."""
    t_start = time.time()
    log.info("=" * 65)
    log.info("NUMOSIM PIPELINE")
    log.info("=" * 65)

    cfg = {
        "raw_dir":       ROOT / "data" / "numosim" / "raw",
        "processed_dir": ROOT / "data" / "numosim" / "processed",
        "features_dir":  ROOT / "data" / "numosim" / "features",
        "results_dir":   ROOT / "results" / "numosim",
        "checkpoints":   ROOT / "results" / "numosim" / "checkpoints",
    }

    # Stage 1: Preprocessing
    data = (stage_preprocessing_numosim(cfg, log)
            if not args.skip_preprocessing
            else _load_processed_numosim(cfg))

    past_stays   = data["past_stays"]
    future_stays = data["future_stays"]
    test_anom    = data["test_anom"]
    poi          = data.get("poi")

    for df in [past_stays, future_stays]:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Stage 2: Features
    if not args.skip_features:
        feat_df = stage_features(
            past_stays=past_stays, future_stays=future_stays,
            is_yjmob=False,
            cache_path=cfg["features_dir"] / "features.parquet",
            poi=poi, log=log, dataset_label="NUMOSIM",
        )
    else:
        feat_df = pd.read_parquet(cfg["features_dir"] / "features.parquet")

    # Build labels — all anomalous (Types 1+2) vs normal
    agent_labels = (
        test_anom.groupby("agent_id")["anomaly"].max().astype(int)
        .reset_index().rename(columns={"agent_id": "agent", "anomaly": "is_anomalous"})
    )
    feat_df = feat_df.merge(agent_labels, on="agent", how="left")
    feat_df["is_anomalous"] = feat_df["is_anomalous"].fillna(0).astype(int)

    avail_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    missing    = [c for c in FEATURE_COLS if c not in feat_df.columns]
    if missing:
        log.warning(f"[NUMOSIM] Missing feature columns: {missing}")

    X_norm_sc, X_all_sc, labels, normal_agents, all_agents = \
        _build_eval_matrices(feat_df, avail_cols)

    log.info(f"[NUMOSIM] {(labels==1).sum()} anomalous / "
             f"{(labels==0).sum():,} normal | {len(avail_cols)} features")

    # Stage 3: Models
    if not args.skip_training:
        all_scores = stage_models(
            X_norm_sc, X_all_sc, labels,
            cfg["checkpoints"], args, log, "NUMOSIM",
        )
    else:
        all_scores = _reload_scores(cfg["checkpoints"], X_all_sc, labels, args, log)

    # Stage 4: Sequence models
    if args.sequence_models:
        all_scores.update(stage_sequence_models(
            past_stays, future_stays, normal_agents, all_agents, labels,
            is_yjmob=False,
            checkpoints=cfg["checkpoints"] / "sequence",
            args=args, log=log, dataset_label="NUMOSIM", poi=poi,
        ))

    # Stage 5: Evaluation — natural rate
    natural_rate = f"natural ({labels.mean():.2%})"
    results_df   = stage_evaluation(
        all_scores, labels, natural_rate,
        cfg["results_dir"], "NUMOSIM", args, log,
    )
    stage_summary_plots([results_df], cfg["results_dir"], "NUMOSIM", log)

    log.info(f"[NUMOSIM] Complete in {(time.time()-t_start)/60:.1f} min.")
    _print_summary(results_df, "NUMOSIM", log)

    # Stage 7 (optional): Sensitivity sweep at 5% and 10%
    if args.sensitivity:
        log.info("=" * 65)
        log.info("NUMOSIM SENSITIVITY - 5% and 10%")
        log.info("=" * 65)
        n_anomalous = int((feat_df["is_anomalous"] == 1).sum())
        sens_results = _run_contamination_sweep(
            feat_df       = feat_df,
            past_stays    = past_stays,
            future_stays  = future_stays,
            avail_cols    = avail_cols,
            rates         = SENSITIVITY_RATES,
            rate_labels   = SENSITIVITY_LABELS,
            results_dir   = cfg["results_dir"] / "sensitivity",
            dataset_name  = "NUMOSIM (sensitivity)",
            is_yjmob      = False,
            max_id        = 0,
            n_anomalous   = n_anomalous,
            total_agents  = 0,
            args          = args,
            log           = log,
            poi           = poi,
        )
        stage_summary_plots(
            sens_results,
            cfg["results_dir"] / "sensitivity",
            "NUMOSIM (sensitivity)", log,
        )


def run_yjmob(args, log: logging.Logger) -> None:
    """Full YJMob100K pipeline — sweeps 1%, 5%, 10% contamination."""
    t_start = time.time()
    log.info("=" * 65)
    log.info("YJMob100K PIPELINE")
    log.info("=" * 65)

    cfg = {
        "raw_dir":       ROOT / "data" / "yjmob" / "raw",
        "processed_dir": ROOT / "data" / "yjmob" / "processed",
        "features_dir":  ROOT / "data" / "yjmob" / "features",
        "results_dir":   ROOT / "results" / "yjmob",
        "checkpoints":   ROOT / "results" / "yjmob" / "checkpoints",
    }

    # Stage 1: Preprocessing
    data = (stage_preprocessing_yjmob(cfg, log)
            if not args.skip_preprocessing
            else _load_processed_yjmob(cfg))

    past_nonanom   = data["past_nonanom"]
    future_nonanom = data["future_nonanom"]
    past_anom      = data["past_anom"]
    future_anom    = data["future_anom"]

    # Stage 2: Features
    if not args.skip_features:
        feat_normal = stage_features(
            past_nonanom, future_nonanom, True,
            cfg["features_dir"] / "features_nonanom.parquet",
            None, log, "YJMob-DS1",
        )
        feat_anom = stage_features(
            past_anom, future_anom, True,
            cfg["features_dir"] / "features_anom.parquet",
            None, log, "YJMob-DS2",
        )
    else:
        feat_normal = pd.read_parquet(cfg["features_dir"] / "features_nonanom.parquet")
        feat_anom   = pd.read_parquet(cfg["features_dir"] / "features_anom.parquet")

    feat_normal["is_anomalous"] = 0
    feat_anom["is_anomalous"]   = 1

    # Offset DS2 agent IDs to avoid collision with DS1
    # This offset is applied to BOTH the feature DataFrame AND the stays DataFrames
    # so that agent IDs are consistent throughout the entire pipeline
    max_id    = int(feat_normal["agent"].max()) + 1
    feat_anom = feat_anom.copy()
    feat_anom["agent"] = feat_anom["agent"] + max_id

    # Combine into one feature DataFrame for the sweep
    feat_df    = pd.concat([feat_normal, feat_anom], ignore_index=True)
    avail_cols = [c for c in FEATURE_COLS if c in feat_df.columns]

    # Combine stays DataFrames — anom stays are passed with original IDs
    # The offset is applied inside _run_contamination_sweep when filtering
    # for sequence models (orig_anom_ids = offset_id - max_id)
    past_combined   = pd.concat([past_nonanom,   past_anom],   ignore_index=True)
    future_combined = pd.concat([future_nonanom, future_anom], ignore_index=True)

    log.info(f"[YJMob] DS1: {len(feat_normal):,} normal | "
             f"DS2: {len(feat_anom):,} anomalous | "
             f"ID offset: {max_id:,}")

    # Stage 3-5: Contamination sweep
    all_rate_results = _run_contamination_sweep(
        feat_df       = feat_df,
        past_stays    = past_combined,
        future_stays  = future_combined,
        avail_cols    = avail_cols,
        rates         = CONTAMINATION_RATES,
        rate_labels   = CONTAMINATION_LABELS,
        results_dir   = cfg["results_dir"],
        dataset_name  = "YJMob100K",
        is_yjmob      = True,
        max_id        = max_id,
        n_anomalous   = 0,
        total_agents  = MAX_AGENTS_YJMOB,
        args          = args,
        log           = log,
        poi           = None,
    )

    stage_summary_plots(all_rate_results, cfg["results_dir"], "YJMob100K", log)

    log.info(f"[YJMob] Complete in {(time.time()-t_start)/60:.1f} min.")
    for df, label in zip(all_rate_results, CONTAMINATION_LABELS):
        _print_summary(df, f"YJMob100K {label}", log)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SIGSPATIAL 2026 - Unified anomaly detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", choices=["numosim", "yjmob", "all"],
                   default="all", help="Dataset(s) to run (default: all)")
    p.add_argument("--skip-preprocessing", action="store_true",
                   help="Load processed parquets instead of preprocessing")
    p.add_argument("--skip-features", action="store_true",
                   help="Load cached features instead of recomputing")
    p.add_argument("--skip-training", action="store_true",
                   help="Reload models from checkpoints")
    p.add_argument("--sequence-models", action="store_true",
                   help="Run LSTM-AD and LM-TAD (slow, requires PyTorch)")
    p.add_argument("--no-deep", action="store_true",
                   help="Skip DAGMM and USAD")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip bootstrap CIs (faster iteration)")
    p.add_argument("--sensitivity", action="store_true",
                   help="Also run NUMOSIM at 5%% and 10%% contamination")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    log  = setup_logging(ROOT / "results")

    log.info("SIGSPATIAL 2026 - Anomaly Detection Pipeline")
    log.info(f"Dataset  : {args.dataset}")
    log.info(f"Flags    : skip_preprocessing={args.skip_preprocessing}  "
             f"skip_features={args.skip_features}  "
             f"skip_training={args.skip_training}  "
             f"sequence_models={args.sequence_models}  "
             f"no_deep={args.no_deep}  "
             f"no_bootstrap={args.no_bootstrap}  "
             f"sensitivity={args.sensitivity}")

    t0 = time.time()

    if args.dataset in ("numosim", "all"):
        run_numosim(args, log)

    if args.dataset in ("yjmob", "all"):
        run_yjmob(args, log)

    log.info(f"\nTotal pipeline time: {(time.time()-t0)/60:.1f} min")
    log.info(f"Results written to: {ROOT / 'results'}/")


if __name__ == "__main__":
    main()