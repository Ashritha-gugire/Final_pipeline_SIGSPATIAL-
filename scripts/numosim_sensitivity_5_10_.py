"""
scripts/numosim_sensitivity_5_10_.py
================================
NUMOSIM contamination rate sensitivity analysis — supplementary results.

Evaluates all 8 models at 5% and 10% contamination rates by subsampling
normal agents while keeping all 381 anomalous agents fixed:

    5%   ->  381 anomalous +  7,239 normal  =   7,620 total
    10%  ->  381 anomalous +  3,429 normal  =   3,810 total

The natural rate (1.87%) is the primary result in main.py — not repeated here.

Prerequisites
-------------
Run main.py first to generate processed stays and cached features:
    python main.py --dataset numosim --skip-preprocessing

Then run this script:
    python scripts/numosim_sensitivity_5_10_.py

    # Skip sequence models if you want a faster run
    python scripts/numosim_sensitivity_5_10_.py --no-sequence-models
    # Skip deep models too
    python scripts/numosim_sensitivity_5_10_.py --no-sequence-models --no-deep

Outputs
-------
    results/numosim/sensitivity/results_5pct.csv
    results/numosim/sensitivity/results_10pct.csv
    results/numosim/sensitivity/figures/fig_grid_curves_5pct.png
    results/numosim/sensitivity/figures/fig_grid_curves_10pct.png
    results/numosim/sensitivity/figures/fig_rate_line_plot.png
    results/numosim/sensitivity/figures/fig_ci_table_*.png
    results/numosim/sensitivity/figures/fig_bar_chart_all_rates.png
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (
    FEATURE_COLS,
    RANDOM_SEED,
    _print_summary,
    setup_logging,
    stage_evaluation,
    stage_models,
    stage_sequence_models,
    stage_summary_plots,
)

# ── Sensitivity sweep config ──────────────────────────────────────────────────

# Natural rate (1.87%) is in main.py — only supplementary rates here
SENSITIVITY_RATES  = [0.05,  0.10]
SENSITIVITY_LABELS = ["5%", "10%"]


def compute_n_normal(rate: float, n_anomalous: int) -> int:
    """Number of normal agents needed to hit the target contamination rate."""
    return int(n_anomalous / rate) - n_anomalous


# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity(args) -> None:
    results_dir = ROOT / "results" / "numosim" / "sensitivity"
    log         = setup_logging(results_dir)

    log.info("=" * 65)
    log.info("NUMOSIM CONTAMINATION SENSITIVITY ANALYSIS")
    log.info("Rates: 5%  |  10%")
    log.info("381 anomalous agents fixed — normal agents subsampled")
    log.info("Natural rate (1.87%) is in main.py")
    log.info("=" * 65)

    t_start = time.time()

    # ── Load preprocessed stays and features (must exist from main.py) ───────
    proc_dir = ROOT / "data" / "numosim" / "processed"
    feat_dir = ROOT / "data" / "numosim" / "features"

    for path in [
        proc_dir / "past_stays.parquet",
        proc_dir / "future_stays.parquet",
        proc_dir / "test_anom.parquet",
        proc_dir / "poi.parquet",
        feat_dir / "features.parquet",
    ]:
        if not path.exists():
            log.error(f"Required file not found: {path}")
            log.error("Run main.py first:  "
                      "python main.py --dataset numosim --skip-preprocessing")
            sys.exit(1)

    log.info("Loading processed stays and features...")
    past_stays   = pd.read_parquet(proc_dir / "past_stays.parquet")
    future_stays = pd.read_parquet(proc_dir / "future_stays.parquet")
    test_anom    = pd.read_parquet(proc_dir / "test_anom.parquet")
    poi          = pd.read_parquet(proc_dir / "poi.parquet")
    feat_df      = pd.read_parquet(feat_dir / "features.parquet")

    for df in [past_stays, future_stays]:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ── Build labels ──────────────────────────────────────────────────────────
    agent_labels = (
        test_anom.groupby("agent_id")["anomaly"]
        .max().astype(int).reset_index()
        .rename(columns={"agent_id": "agent", "anomaly": "is_anomalous"})
    )
    feat_df = feat_df.merge(agent_labels, on="agent", how="left")
    feat_df["is_anomalous"] = feat_df["is_anomalous"].fillna(0).astype(int)

    avail_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    missing    = [c for c in FEATURE_COLS if c not in feat_df.columns]
    if missing:
        log.warning(f"Missing feature columns: {missing}")

    n_anomalous = int((feat_df["is_anomalous"] == 1).sum())
    log.info(f"Total anomalous agents : {n_anomalous}")
    log.info(f"Total normal agents    : {(feat_df['is_anomalous'] == 0).sum():,}")
    log.info(f"Features available     : {len(avail_cols)} / {len(FEATURE_COLS)}")

    # ── Print sampling plan ───────────────────────────────────────────────────
    log.info("\nSampling plan:")
    for rate, label in zip(SENSITIVITY_RATES, SENSITIVITY_LABELS):
        n_norm  = compute_n_normal(rate, n_anomalous)
        n_total = n_norm + n_anomalous
        actual  = n_anomalous / n_total * 100
        log.info(f"  {label:<6}  {n_anomalous} anomalous + "
                 f"{n_norm:>6,} normal = {n_total:>7,} total "
                 f"(actual rate = {actual:.2f}%)")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    all_rate_results = []

    for rate, rate_label in zip(SENSITIVITY_RATES, SENSITIVITY_LABELS):
        log.info("")
        log.info("-" * 65)
        log.info(f"Rate: {rate_label}")
        log.info("-" * 65)

        # Subsample normal agents to hit target rate
        n_normal_target = compute_n_normal(rate, n_anomalous)
        normal_pool     = feat_df[feat_df["is_anomalous"] == 0]
        anom_df         = feat_df[feat_df["is_anomalous"] == 1]

        normal_sample = normal_pool.sample(
            n            = min(n_normal_target, len(normal_pool)),
            random_state = RANDOM_SEED,
        )
        eval_df = pd.concat([normal_sample, anom_df], ignore_index=True)

        labels        = eval_df["is_anomalous"].values
        normal_mask   = labels == 0
        normal_agents = eval_df.loc[normal_mask, "agent"].values
        all_agents    = eval_df["agent"].values

        log.info(f"Eval set: {(~normal_mask).sum()} anomalous / "
                 f"{normal_mask.sum():,} normal / "
                 f"{len(eval_df):,} total  "
                 f"(actual rate = {labels.mean()*100:.2f}%)")

        # Scale — fit on normal agents only, no leakage
        X_all     = eval_df[avail_cols].fillna(0).values
        scaler    = StandardScaler()
        scaler.fit(X_all[normal_mask])
        X_all_sc  = scaler.transform(X_all)
        X_norm_sc = X_all_sc[normal_mask]

        # Separate checkpoint folder per rate
        rate_slug  = rate_label.replace("%", "pct")
        rate_ckpts = results_dir / "checkpoints" / rate_slug

        # ── Unsupervised models + XGBoost ─────────────────────────────────────
        all_scores = stage_models(
            X_normal      = X_norm_sc,
            X_all         = X_all_sc,
            labels        = labels,
            checkpoints   = rate_ckpts,
            args          = args,
            log           = log,
            dataset_label = f"NUMOSIM-sensitivity-{rate_label}",
        )

        # ── Sequence models (on by default) ───────────────────────────────────
        if not args.no_sequence_models:
            seq_scores = stage_sequence_models(
                past_stays    = past_stays,
                future_stays  = future_stays,
                normal_agents = normal_agents,
                all_agents    = all_agents,
                labels        = labels,
                is_yjmob      = False,
                checkpoints   = rate_ckpts / "sequence",
                args          = args,
                log           = log,
                dataset_label = f"NUMOSIM-sensitivity-{rate_label}",
                poi           = poi,
            )
            all_scores.update(seq_scores)
        else:
            log.info(f"[NUMOSIM-sensitivity-{rate_label}] "
                     "Skipping sequence models (--no-sequence-models).")

        # ── Evaluation + figures ───────────────────────────────────────────────
        rate_results = stage_evaluation(
            all_scores   = all_scores,
            labels       = labels,
            rate_label   = rate_label,
            results_dir  = results_dir,
            dataset_name = "NUMOSIM (sensitivity)",
            args         = args,
            log          = log,
        )
        all_rate_results.append(rate_results)
        _print_summary(rate_results, f"NUMOSIM sensitivity {rate_label}", log)

    # ── Summary plots across both rates ───────────────────────────────────────
    stage_summary_plots(
        all_rate_results, results_dir,
        "NUMOSIM (sensitivity)", log,
    )

    log.info(f"\nSensitivity analysis complete in "
             f"{(time.time()-t_start)/60:.1f} min.")
    log.info(f"Results -> {results_dir}/")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NUMOSIM contamination rate sensitivity analysis (5% and 10%)"
    )
    p.add_argument(
        "--no-sequence-models", action="store_true",
        help="Skip LSTM-AD and LM-TAD (sequence models run by default)",
    )
    p.add_argument(
        "--no-deep", action="store_true",
        help="Skip DAGMM and USAD",
    )
    p.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip bootstrap CIs (faster dev iteration)",
    )
    p.add_argument(
        "--skip-training", action="store_true",
        help="Reload models from checkpoints instead of retraining",
    )
    # Always True for this script — features must exist from main.py
    p.add_argument("--skip-preprocessing", action="store_true", default=True)
    p.add_argument("--skip-features",      action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sensitivity(args)