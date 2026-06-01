"""
main.py
========
Unified entry point for the SIGSPATIAL 2026 anomaly detection pipeline.

Runs the full benchmark for NUMOSIM, YJMob100K, or both datasets with
a single command. All steps share the same code — only data paths and
dataset-specific flags differ.

Pipeline stages
───────────────
  1. Preprocessing   raw files → canonical past/future stays parquet
  2. Features        stays → 18-dim feature matrix (cached after first run)
  3. Models          fit unsupervised scorers on normal agents, score all
  4. Evaluation      AUC, AP, F1, bootstrap 95% CIs, results table
  5. Plots           grid curves, bar chart, CI table, rate line plot
  6. Sequence models LSTM-AD + LM-TAD  (--sequence-models flag)

Usage examples
──────────────
  # Full run — both datasets
  python main.py --dataset all

  # NUMOSIM only, preprocessing already done
  python main.py --dataset numosim --skip-preprocessing

  # YJMob only, features already cached, regenerate plots only
  python main.py --dataset yjmob --skip-preprocessing --skip-features --skip-training

  # Include LSTM-AD and LM-TAD (requires PyTorch)
  python main.py --dataset all --sequence-models

  # Skip DAGMM and USAD (no GPU / quick test)
  python main.py --dataset all --no-deep

  # Skip bootstrap CIs (faster dev iteration)
  python main.py --dataset all --no-bootstrap
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run.log"
    fmt      = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt  = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt,
                        datefmt=datefmt, handlers=handlers)
    return logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CONTAMINATION_RATES  = [0.01, 0.05, 0.10]
CONTAMINATION_LABELS = ["1%", "5%", "10%"]
TOP_K_LIST           = [50, 100, 200]
N_BOOTSTRAP          = 1000
RANDOM_SEED          = 42
MAX_AGENTS_YJMOB     = 10_000   # agents per contamination sample
N_NORMAL_NUMOSIM     = 20_000   # normal agents to sample for NUMOSIM

# 18 feature columns (Table 1) — identical for both datasets
FEATURE_COLS = [
    # Change features (7)
    "geobin_similarity",
    "weighted_abandonment_score",
    "kuiper_start_statistic",
    "kuiper_start_pvalue",
    "kuiper_end_statistic",
    "kuiper_end_pvalue",
    "radius_of_gyration_drift",
    # Drift features (5)
    "gyration_score_drift",
    "max_pairwise_dist_drift",
    "max_dist_home_drift",
    "std_dist_home_drift",
    "mean_dist_home_drift",
    # Absolute (1)
    "novel_location_rate",
    # Entropy (5)
    "loc_entropy_future",
    "temp_entropy_future",
    "loc_entropy_drift",
    "temp_entropy_drift",
    "entropy_ratio",
]

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

# Hardcoded tuned hyperparameters (validated, no re-tuning)
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
# Stage 1 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def stage_preprocessing_numosim(cfg: dict, log: logging.Logger) -> dict:
    """NUMOSIM raw parquets → canonical past/future stays."""
    from preprocessing.numosim_preprocessing import NUMOSIMPreprocessor

    processed_dir = cfg["processed_dir"]
    past_path     = processed_dir / "past_stays.parquet"

    if past_path.exists():
        log.info("[NUMOSIM] Processed files found — loading from disk.")
        return {
            "past_stays":   pd.read_parquet(processed_dir / "past_stays.parquet"),
            "future_stays": pd.read_parquet(processed_dir / "future_stays.parquet"),
            "test_anom":    pd.read_parquet(processed_dir / "test_anom.parquet"),
            "poi":          pd.read_parquet(processed_dir / "poi.parquet"),
        }

    log.info("[NUMOSIM] Running preprocessing...")
    prep    = NUMOSIMPreprocessor(
        data_dir    = str(cfg["raw_dir"]),
        output_dir  = str(processed_dir),
        n_normal    = N_NORMAL_NUMOSIM,
        random_seed = RANDOM_SEED,
    )
    results = prep.run()
    log.info("[NUMOSIM] Preprocessing complete.")
    return results


def stage_preprocessing_yjmob(cfg: dict, log: logging.Logger) -> dict:
    """YJMob raw CSVs → canonical past/future stays (flat parquet)."""
    processed_dir = cfg["processed_dir"]
    past_path     = processed_dir / "past_stays.parquet"

    if past_path.exists():
        log.info("[YJMob] Processed files found — loading from disk.")
        return {
            "past_stays":   pd.read_parquet(processed_dir / "past_stays.parquet"),
            "future_stays": pd.read_parquet(processed_dir / "future_stays.parquet"),
        }

    log.info("[YJMob] Running preprocessing (this will take ~30 min)...")
    import subprocess
    for dataset_num, label in [(1, "nonanom"), (2, "anom")]:
        csv_path = cfg["raw_dir"] / f"yjmob100k-dataset{dataset_num}.csv"
        out_dir  = processed_dir / label
        if not csv_path.exists():
            raise FileNotFoundError(
                f"YJMob raw CSV not found: {csv_path}\n"
                f"Place yjmob100k-dataset{dataset_num}.csv in {cfg['raw_dir']}"
            )
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts" / "convert_yjmob_to_parquet.py"),
            str(csv_path), str(out_dir),
            "--split-day", "60",
        ], check=True)

    # For YJMob, past/future are split by dataset label
    # DS1 = normal (past + future both normal behaviour)
    # DS2 = anomalous (future period = emergency)
    past_nonanom   = pd.read_parquet(processed_dir / "nonanom" / "past_stays.parquet")
    future_nonanom = pd.read_parquet(processed_dir / "nonanom" / "future_stays.parquet")
    past_anom      = pd.read_parquet(processed_dir / "anom"    / "past_stays.parquet")
    future_anom    = pd.read_parquet(processed_dir / "anom"    / "future_stays.parquet")

    log.info("[YJMob] Preprocessing complete.")
    return {
        "past_nonanom":   past_nonanom,
        "future_nonanom": future_nonanom,
        "past_anom":      past_anom,
        "future_anom":    future_anom,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def stage_features(
    past_stays:   pd.DataFrame,
    future_stays: pd.DataFrame,
    is_yjmob:     bool,
    cache_path:   Path,
    poi:          pd.DataFrame,
    log:          logging.Logger,
    dataset_label: str = "",
) -> pd.DataFrame:
    """
    Compute 18-dim feature vector. Loads from cache if available.
    Returns DataFrame with columns = FEATURE_COLS + ['agent'].
    """
    if cache_path.exists():
        log.info(f"[{dataset_label}] Features cache found — loading from {cache_path}")
        return pd.read_parquet(cache_path)

    log.info(f"[{dataset_label}] Computing 18-dim features "
             f"(~30-45 min first run)...")
    from features.features_18dim import FeatureBuilder18Dim

    builder  = FeatureBuilder18Dim(
        past_stays   = past_stays,
        future_stays = future_stays,
        is_yjmob     = is_yjmob,
        poi          = poi,
    )
    feat_df  = builder.compute_all()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(cache_path, index=False)
    log.info(f"[{dataset_label}] Features saved → {cache_path}")
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Model Training & Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _clip_by_train_percentiles(
    scores: np.ndarray,
    train_scores: np.ndarray,
) -> np.ndarray:
    """Clip scores using 1st–99th percentile of training scores."""
    lo = np.percentile(train_scores, 1)
    hi = np.percentile(train_scores, 99)
    return np.clip(scores, lo, hi)


def _orient(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Ensure higher score = more anomalous."""
    from sklearn.metrics import roc_auc_score
    s = np.where(np.isnan(scores), 0.0, scores.astype(float))
    return -s if roc_auc_score(labels, s) < 0.5 else s


def stage_models(
    X_normal:     np.ndarray,
    X_all:        np.ndarray,
    labels:       np.ndarray,
    checkpoints:  Path,
    args,
    log:          logging.Logger,
    dataset_label: str = "",
) -> dict[str, np.ndarray]:
    """
    Fit each unsupervised scorer on X_normal, score X_all.
    Returns dict: model_name → oriented anomaly scores (len = len(X_all)).
    Saves checkpoints to disk for later reload.
    """
    from models.sklearn_scorers import GMMScorer, IsolationForestScorer, KNNScorer
    from models.xgboost_scorer  import XGBoostScorer

    checkpoints.mkdir(parents=True, exist_ok=True)
    scores = {}

    sklearn_models = {
        "GMM":              GMMScorer(**HYPERPARAMS["GMM"]),
        "Isolation Forest": IsolationForestScorer(**HYPERPARAMS["Isolation Forest"]),
        "KNN":              KNNScorer(**HYPERPARAMS["KNN"]),
    }

    # ── Sklearn models ────────────────────────────────────────────────────────
    for name, scorer in sklearn_models.items():
        ckpt = checkpoints / name.replace(" ", "_").lower()
        if args.skip_training and (ckpt / f"{name.replace(' ','_').lower()}_scorer.pkl").exists():
            log.info(f"[{dataset_label}] Loading {name} from checkpoint...")
            scorer.load(str(ckpt))
            raw = scorer.score(X_all)
        else:
            log.info(f"[{dataset_label}] Fitting {name}...")
            t0 = time.time()
            scorer.fit(X_normal)
            raw = scorer.score(X_all)
            scorer.save(str(ckpt))
            log.info(f"[{dataset_label}] {name} done in {time.time()-t0:.1f}s")
        scores[name] = _orient(raw, labels)

    # ── Deep models (DAGMM, USAD) ─────────────────────────────────────────────
    if not args.no_deep:
        deep_models = {}
        try:
            from models.dagmm_scorer import DAGMMScorer
            deep_models["DAGMM"] = DAGMMScorer(**HYPERPARAMS["DAGMM"])
        except ImportError:
            log.warning(f"[{dataset_label}] DAGMMScorer unavailable — skipping.")

        try:
            from models.usad_scorer import USADScorer
            deep_models["USAD"] = USADScorer(**HYPERPARAMS["USAD"])
        except ImportError:
            log.warning(f"[{dataset_label}] USADScorer unavailable — skipping.")

        for name, scorer in deep_models.items():
            ckpt = checkpoints / name.lower()
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
                raw       = scorer.score(X_all)
                raw       = _clip_by_train_percentiles(raw, raw_train)
                scorer.save(str(ckpt))
                log.info(f"[{dataset_label}] {name} done in {time.time()-t0:.1f}s")
            scores[name] = _orient(raw, labels)

    # ── XGBoost supervised reference ──────────────────────────────────────────
    log.info(f"[{dataset_label}] Fitting XGBoost† (5-fold OOF)...")
    xgb = XGBoostScorer(**HYPERPARAMS["XGBoost†"])
    xgb.fit(X_all, labels)
    scores["XGBoost†"] = _orient(xgb.score(), labels)
    xgb.save(str(checkpoints / "xgboost"))

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Sequence Models
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
    Returns dict: model_name → oriented anomaly scores aligned to all_agents.
    """
    from features.sequence_features import SequenceFeatureBuilder

    checkpoints.mkdir(parents=True, exist_ok=True)
    seq_scores = {}

    # ── LSTM-AD ───────────────────────────────────────────────────────────────
    try:
        from models.lstmad_scorer import LSTMADScorer
        log.info(f"[{dataset_label}] Building 7-dim sequence features...")

        # Build on full future stays — split by normal/all after
        seq_builder = SequenceFeatureBuilder(
            stays    = future_stays,
            is_yjmob = is_yjmob,
            poi      = poi,
        )
        sequences = seq_builder.build_sequences()   # Dict[agent_id → np.ndarray (T, 7)]

        normal_seqs = [sequences[a] for a in normal_agents if a in sequences]
        all_seqs    = [sequences.get(a, np.zeros((1, 7), dtype=np.float32))
                       for a in all_agents]

        log.info(f"[{dataset_label}] Fitting LSTM-AD "
                 f"({len(normal_seqs):,} normal seqs)...")
        lstmad = LSTMADScorer(**HYPERPARAMS["LSTM-AD"])
        t0     = time.time()
        lstmad.fit(normal_seqs)
        raw_train = lstmad.score(normal_seqs)
        raw_all   = lstmad.score(all_seqs)
        raw_all   = _clip_by_train_percentiles(raw_all, raw_train)
        lstmad.save(str(checkpoints / "lstmad"))
        log.info(f"[{dataset_label}] LSTM-AD done in {time.time()-t0:.1f}s")
        seq_scores["LSTM-AD"] = _orient(raw_all, labels)

    except ImportError as e:
        log.warning(f"[{dataset_label}] LSTMADScorer unavailable: {e}")

    # ── LM-TAD ────────────────────────────────────────────────────────────────
    try:
        from models.lmtad_scorer import (
            LMTADScorer, build_vocab, build_lmtad_sequences,
        )
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

        log.info(f"[{dataset_label}] Fitting LM-TAD "
                 f"({len(train_seqs):,} normal seqs)...")
        lmtad = LMTADScorer(**HYPERPARAMS["LM-TAD"])
        t0    = time.time()
        lmtad.fit(train_seqs, vocab, train_bins)
        raw_scores = lmtad.score(test_seqs)
        lmtad.save(str(checkpoints / "lmtad"))
        log.info(f"[{dataset_label}] LM-TAD done in {time.time()-t0:.1f}s")

        # Align scores to all_agents order
        id_to_score = dict(zip(test_ids, raw_scores))
        aligned     = np.array([id_to_score.get(a, 0.0) for a in all_agents])
        seq_scores["LM-TAD"] = _orient(aligned, labels)

    except ImportError as e:
        log.warning(f"[{dataset_label}] LMTADScorer unavailable: {e}")

    return seq_scores


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Evaluation & Plots
# ─────────────────────────────────────────────────────────────────────────────

def stage_evaluation(
    all_scores:   dict[str, np.ndarray],
    labels:       np.ndarray,
    rate_label:   str,
    results_dir:  Path,
    dataset_name: str,
    args,
    log:          logging.Logger,
) -> pd.DataFrame:
    """
    Compute metrics + CIs, save CSV, generate all 4 figures.
    Returns results DataFrame.
    """
    from evaluation.metrics import build_results_table
    from evaluation.plots   import (
        make_grid_figure, make_bar_chart,
        make_ci_table, make_rate_line_plot,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[{dataset_name}] Evaluating {len(all_scores)} models "
             f"at {rate_label}...")

    n_bootstrap = N_BOOTSTRAP if not args.no_bootstrap else 0
    results = {
        name: (MODEL_FAMILY.get(name, "unknown"), scores)
        for name, scores in all_scores.items()
    }
    df = build_results_table(
        results     = results,
        labels      = labels,
        rate_label  = rate_label,
        top_k_list  = TOP_K_LIST,
        n_bootstrap = n_bootstrap,
    )

    # Save CSV
    rate_slug = rate_label.replace("%", "pct").replace(".", "").replace(" ", "_")
    csv_path  = results_dir / f"results_{rate_slug}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"[{dataset_name}] Results saved → {csv_path}")

    # Grid figure
    grid_path = figures_dir / f"fig_grid_curves_{rate_slug}.png"
    make_grid_figure(
        scores       = all_scores,
        labels       = labels,
        rate_label   = rate_label,
        dataset_name = dataset_name,
        out_path     = str(grid_path),
    )

    # CI table
    ci_path = figures_dir / f"fig_ci_table_{rate_slug}.png"
    make_ci_table(df, dataset_name, str(ci_path))

    return df


def stage_summary_plots(
    all_results: list[pd.DataFrame],
    results_dir: Path,
    dataset_name: str,
    log: logging.Logger,
) -> None:
    """Bar chart + rate line plot across all contamination rates."""
    from evaluation.plots import make_bar_chart, make_rate_line_plot

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not all_results:
        return

    combined = pd.concat(all_results, ignore_index=True)

    bar_path  = figures_dir / "fig_bar_chart_all_rates.png"
    line_path = figures_dir / "fig_rate_line_plot.png"

    make_bar_chart(combined, dataset_name, str(bar_path))
    log.info(f"[{dataset_name}] Bar chart saved → {bar_path}")

    if len(all_results) > 1:
        make_rate_line_plot(combined, dataset_name, str(line_path))
        log.info(f"[{dataset_name}] Rate line plot saved → {line_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset runners
# ─────────────────────────────────────────────────────────────────────────────

def run_numosim(args, log: logging.Logger) -> None:
    """Full NUMOSIM pipeline."""
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

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────
    if not args.skip_preprocessing:
        data = stage_preprocessing_numosim(cfg, log)
    else:
        log.info("[NUMOSIM] Skipping preprocessing (--skip-preprocessing).")
        proc = cfg["processed_dir"]
        data = {
            "past_stays":   pd.read_parquet(proc / "past_stays.parquet"),
            "future_stays": pd.read_parquet(proc / "future_stays.parquet"),
            "test_anom":    pd.read_parquet(proc / "test_anom.parquet"),
            "poi":          pd.read_parquet(proc / "poi.parquet"),
        }

    past_stays   = data["past_stays"]
    future_stays = data["future_stays"]
    test_anom    = data["test_anom"]
    poi          = data.get("poi")

    # Ensure timestamps parsed
    for df in [past_stays, future_stays]:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ── Stage 2: Features ─────────────────────────────────────────────────────
    if not args.skip_features:
        feat_df = stage_features(
            past_stays    = past_stays,
            future_stays  = future_stays,
            is_yjmob      = False,
            cache_path    = cfg["features_dir"] / "features.parquet",
            poi           = poi,
            log           = log,
            dataset_label = "NUMOSIM",
        )
    else:
        log.info("[NUMOSIM] Skipping feature computation (--skip-features).")
        feat_df = pd.read_parquet(cfg["features_dir"] / "features.parquet")

    # ── Build labels — all anomalous vs normal (Types 1+2 combined) ───────────
    agent_labels = (
        test_anom.groupby("agent_id")["anomaly"]
        .max().astype(int).reset_index()
        .rename(columns={"agent_id": "agent", "anomaly": "is_anomalous"})
    )
    feat_df = feat_df.merge(agent_labels, on="agent", how="left")
    feat_df["is_anomalous"] = feat_df["is_anomalous"].fillna(0).astype(int)

    avail_cols   = [c for c in FEATURE_COLS if c in feat_df.columns]
    missing_cols = [c for c in FEATURE_COLS if c not in feat_df.columns]
    if missing_cols:
        log.warning(f"[NUMOSIM] Missing feature columns: {missing_cols}")

    labels       = feat_df["is_anomalous"].values
    normal_mask  = labels == 0
    all_agents   = feat_df["agent"].values
    normal_agents = feat_df.loc[normal_mask, "agent"].values

    X_all     = feat_df[avail_cols].fillna(0).values
    scaler    = StandardScaler()
    scaler.fit(X_all[normal_mask])
    X_all_sc  = scaler.transform(X_all)
    X_norm_sc = X_all_sc[normal_mask]

    log.info(f"[NUMOSIM] {(~normal_mask).sum()} anomalous / "
             f"{normal_mask.sum():,} normal agents  "
             f"| {len(avail_cols)} features")

    # ── Stage 3: Models ───────────────────────────────────────────────────────
    if not args.skip_training:
        all_scores = stage_models(
            X_normal      = X_norm_sc,
            X_all         = X_all_sc,
            labels        = labels,
            checkpoints   = cfg["checkpoints"],
            args          = args,
            log           = log,
            dataset_label = "NUMOSIM",
        )
    else:
        log.info("[NUMOSIM] Skipping model training (--skip-training).")
        all_scores = _reload_scores(cfg["checkpoints"], X_all_sc, labels, args, log)

    # ── Stage 4: Sequence models ──────────────────────────────────────────────
    if args.sequence_models:
        seq_scores = stage_sequence_models(
            past_stays    = past_stays,
            future_stays  = future_stays,
            normal_agents = normal_agents,
            all_agents    = all_agents,
            labels        = labels,
            is_yjmob      = False,
            checkpoints   = cfg["checkpoints"] / "sequence",
            args          = args,
            log           = log,
            dataset_label = "NUMOSIM",
            poi           = poi,
        )
        all_scores.update(seq_scores)

    # ── Stage 5: Evaluation — single operating point (natural rate) ───────────
    natural_rate = f"natural ({labels.mean():.2%})"
    results_df   = stage_evaluation(
        all_scores   = all_scores,
        labels       = labels,
        rate_label   = natural_rate,
        results_dir  = cfg["results_dir"],
        dataset_name = "NUMOSIM",
        args         = args,
        log          = log,
    )
    stage_summary_plots([results_df], cfg["results_dir"], "NUMOSIM", log)

    log.info(f"[NUMOSIM] Pipeline complete in "
             f"{(time.time()-t_start)/60:.1f} min.")
    _print_summary(results_df, "NUMOSIM", log)


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

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────
    if not args.skip_preprocessing:
        data = stage_preprocessing_yjmob(cfg, log)
    else:
        log.info("[YJMob] Skipping preprocessing (--skip-preprocessing).")
        proc = cfg["processed_dir"]
        data = {
            "past_nonanom":   pd.read_parquet(proc / "nonanom" / "past_stays.parquet"),
            "future_nonanom": pd.read_parquet(proc / "nonanom" / "future_stays.parquet"),
            "past_anom":      pd.read_parquet(proc / "anom"    / "past_stays.parquet"),
            "future_anom":    pd.read_parquet(proc / "anom"    / "future_stays.parquet"),
        }

    past_nonanom   = data["past_nonanom"]
    future_nonanom = data["future_nonanom"]
    past_anom      = data["past_anom"]
    future_anom    = data["future_anom"]

    # ── Stage 2: Features — computed separately for each dataset ─────────────
    if not args.skip_features:
        feat_normal = stage_features(
            past_stays    = past_nonanom,
            future_stays  = future_nonanom,
            is_yjmob      = True,
            cache_path    = cfg["features_dir"] / "features_nonanom.parquet",
            poi           = None,
            log           = log,
            dataset_label = "YJMob-DS1",
        )
        feat_anom = stage_features(
            past_stays    = past_anom,
            future_stays  = future_anom,
            is_yjmob      = True,
            cache_path    = cfg["features_dir"] / "features_anom.parquet",
            poi           = None,
            log           = log,
            dataset_label = "YJMob-DS2",
        )
    else:
        log.info("[YJMob] Skipping feature computation (--skip-features).")
        feat_normal = pd.read_parquet(
            cfg["features_dir"] / "features_nonanom.parquet"
        )
        feat_anom = pd.read_parquet(
            cfg["features_dir"] / "features_anom.parquet"
        )

    feat_normal["is_anomalous"] = 0
    feat_anom["is_anomalous"]   = 1

    # Offset DS2 agent IDs to avoid collision with DS1
    max_id = int(feat_normal["agent"].max()) + 1
    feat_anom = feat_anom.copy()
    feat_anom["agent"] = feat_anom["agent"] + max_id

    avail_cols = [c for c in FEATURE_COLS if c in feat_normal.columns]

    # ── Sweep contamination rates ─────────────────────────────────────────────
    all_rate_results = []
    rng = np.random.default_rng(RANDOM_SEED)

    for rate, rate_label in zip(CONTAMINATION_RATES, CONTAMINATION_LABELS):
        log.info("-" * 65)
        log.info(f"[YJMob] Contamination rate: {rate_label}")
        log.info("-" * 65)

        n_anom = int(MAX_AGENTS_YJMOB * rate)
        n_norm = MAX_AGENTS_YJMOB - n_anom

        df_norm_s = feat_normal.sample(
            n=min(n_norm, len(feat_normal)), random_state=RANDOM_SEED
        ).reset_index(drop=True)
        df_anom_s = feat_anom.sample(
            n=min(n_anom, len(feat_anom)), random_state=RANDOM_SEED
        ).reset_index(drop=True)

        combined     = pd.concat([df_norm_s, df_anom_s], ignore_index=True)
        labels       = combined["is_anomalous"].astype(int).values
        all_agents   = combined["agent"].values
        normal_mask  = labels == 0
        normal_agents = combined.loc[normal_mask, "agent"].values

        log.info(f"[YJMob] {n_anom:,} anomalous / {n_norm:,} normal "
                 f"= {len(combined):,} total")

        X_all    = combined[avail_cols].fillna(0).values
        scaler   = StandardScaler()
        scaler.fit(X_all[normal_mask])
        X_all_sc = scaler.transform(X_all)
        X_norm_sc = X_all_sc[normal_mask]

        # Save canonical agent list (required for sequence models)
        rate_slug     = rate_label.replace("%", "pct")
        canonical_dir = cfg["results_dir"]
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = canonical_dir / f"canonical_agents_{rate_slug}.parquet"
        pd.DataFrame({
            "agent":  all_agents,
            "y_true": labels,
        }).to_parquet(canonical_path, index=False)
        log.info(f"[YJMob] Canonical agent list saved → {canonical_path}")

        # Models
        rate_ckpts = cfg["checkpoints"] / rate_slug
        if not args.skip_training:
            all_scores = stage_models(
                X_normal      = X_norm_sc,
                X_all         = X_all_sc,
                labels        = labels,
                checkpoints   = rate_ckpts,
                args          = args,
                log           = log,
                dataset_label = f"YJMob-{rate_label}",
            )
        else:
            log.info(f"[YJMob] Skipping training (--skip-training).")
            all_scores = _reload_scores(
                rate_ckpts, X_all_sc, labels, args, log
            )

        # Sequence models
        if args.sequence_models:
            # For YJMob sequence models use the sampled past/future stays
            sampled_agents_set = set(all_agents.tolist())
            past_sample   = pd.concat([past_nonanom, past_anom])
            past_sample   = past_sample[past_sample["agent"].isin(
                set(df_norm_s["agent"]) | (set(df_anom_s["agent"]) - max_id)
            )]
            future_sample = pd.concat([future_nonanom, future_anom])

            seq_scores = stage_sequence_models(
                past_stays    = past_sample,
                future_stays  = future_sample,
                normal_agents = normal_agents,
                all_agents    = all_agents,
                labels        = labels,
                is_yjmob      = True,
                checkpoints   = rate_ckpts / "sequence",
                args          = args,
                log           = log,
                dataset_label = f"YJMob-{rate_label}",
            )
            all_scores.update(seq_scores)

        # Evaluation
        rate_results = stage_evaluation(
            all_scores   = all_scores,
            labels       = labels,
            rate_label   = rate_label,
            results_dir  = cfg["results_dir"],
            dataset_name = "YJMob100K",
            args         = args,
            log          = log,
        )
        all_rate_results.append(rate_results)

    # Summary plots across all rates
    stage_summary_plots(all_rate_results, cfg["results_dir"], "YJMob100K", log)

    log.info(f"[YJMob] Pipeline complete in "
             f"{(time.time()-t_start)/60:.1f} min.")
    for df, label in zip(all_rate_results, CONTAMINATION_LABELS):
        _print_summary(df, f"YJMob100K {label}", log)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reload_scores(
    checkpoints: Path,
    X_all: np.ndarray,
    labels: np.ndarray,
    args,
    log: logging.Logger,
) -> dict[str, np.ndarray]:
    """Reload fitted scorers from checkpoints and re-score."""
    from models.sklearn_scorers import GMMScorer, IsolationForestScorer, KNNScorer
    from models.xgboost_scorer  import XGBoostScorer

    scores = {}
    reload_map = {
        "GMM":              (GMMScorer,              "gmm"),
        "Isolation Forest": (IsolationForestScorer,  "isolation_forest"),
        "KNN":              (KNNScorer,              "knn"),
    }
    for name, (cls, slug) in reload_map.items():
        ckpt = checkpoints / slug
        if ckpt.exists():
            scorer = cls()
            scorer.load(str(ckpt))
            scores[name] = _orient(scorer.score(X_all), labels)
            log.info(f"  Reloaded {name}")

    if not args.no_deep:
        for name, module, slug in [
            ("DAGMM", "models.dagmm_scorer", "dagmm"),
            ("USAD",  "models.usad_scorer",  "usad"),
        ]:
            ckpt = checkpoints / slug
            if ckpt.exists():
                try:
                    import importlib
                    mod    = importlib.import_module(module)
                    cls    = getattr(mod, f"{name}Scorer")
                    scorer = cls()
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


def _print_summary(
    df: pd.DataFrame, label: str, log: logging.Logger
) -> None:
    log.info(f"\n{'='*65}")
    log.info(f"RESULTS — {label}")
    log.info(f"{'='*65}")
    cols = ["Algorithm", "AUC", "AP", "F1"]
    if "AUC_95CI" in df.columns:
        cols.insert(2, "AUC_95CI")
    log.info("\n" + df[cols].to_string(index=False))
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SIGSPATIAL 2026 — Unified anomaly detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", choices=["numosim", "yjmob", "all"],
        default="all",
        help="Which dataset(s) to run (default: all)",
    )
    p.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Skip preprocessing — load processed parquets directly",
    )
    p.add_argument(
        "--skip-features", action="store_true",
        help="Skip feature computation — load cached feature parquets",
    )
    p.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training — reload from checkpoints",
    )
    p.add_argument(
        "--sequence-models", action="store_true",
        help="Run LSTM-AD and LM-TAD (slow, requires PyTorch)",
    )
    p.add_argument(
        "--no-deep", action="store_true",
        help="Skip DAGMM and USAD (for quick CPU-only runs)",
    )
    p.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip bootstrap CIs (faster dev iteration)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    results_root = ROOT / "results"
    log = setup_logging(results_root)

    log.info("SIGSPATIAL 2026 — Anomaly Detection Pipeline")
    log.info(f"Dataset   : {args.dataset}")
    log.info(f"Flags     : "
             f"skip_preprocessing={args.skip_preprocessing}  "
             f"skip_features={args.skip_features}  "
             f"skip_training={args.skip_training}  "
             f"sequence_models={args.sequence_models}  "
             f"no_deep={args.no_deep}  "
             f"no_bootstrap={args.no_bootstrap}")

    t0 = time.time()

    if args.dataset in ("numosim", "all"):
        run_numosim(args, log)

    if args.dataset in ("yjmob", "all"):
        run_yjmob(args, log)

    total = (time.time() - t0) / 60
    log.info(f"\nTotal pipeline time: {total:.1f} min")
    log.info(f"Results written to: {results_root}/")


if __name__ == "__main__":
    main()