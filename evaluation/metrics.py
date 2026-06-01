"""
evaluation/metrics.py
======================
Unified evaluation utilities for both NUMOSIM and YJMob100K.

Public API
──────────
    evaluate(labels, scores)          → dict of all metrics
    bootstrap_ci(labels, scores)      → dict with AUC/AP 95% CIs
    build_results_table(results, ...) → pd.DataFrame, one row per model
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    accuracy_score, precision_recall_curve,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    labels:     np.ndarray,
    scores:     np.ndarray,
    top_k_list: list[int] = None,
) -> dict:
    """
    Compute the full evaluation suite for one model at one operating point.

    Metrics returned
    ─────────────────
        auc        AUC-ROC
        ap         Average Precision (AUC-PR)
        f1         F1 at the F1-optimal threshold
        precision  Precision at the F1-optimal threshold
        recall     Recall at the F1-optimal threshold
        accuracy   Accuracy at the F1-optimal threshold
        lift       AP / baseline_ap
        baseline_ap  Fraction of positives (random AP)
        threshold  The F1-optimal decision threshold
        top{k}_prec  Precision@K for each k in top_k_list
        scores     Oriented scores (higher = more anomalous)

    Args:
        labels:     Binary ground-truth array (1 = anomalous, 0 = normal).
        scores:     Continuous anomaly scores, same length as labels.
        top_k_list: K values for Precision@K. Default [50, 100, 200].

    Returns:
        dict of metric name → value.
    """
    if top_k_list is None:
        top_k_list = [50, 100, 200]

    labels = np.array(labels, dtype=int)
    scores = np.where(np.isnan(scores), 0.0, np.array(scores, dtype=float))

    # Guard: need both classes present
    if len(np.unique(labels)) < 2:
        return {
            "auc": 0.5, "ap": float(labels.mean()), "f1": 0.0,
            "precision": 0.0, "recall": 0.0,
            "accuracy": float((labels == 0).mean()),
            "lift": 1.0, "baseline_ap": float(labels.mean()),
            "threshold": 0.5, "scores": scores,
            **{f"top{k}_prec": 0.0 for k in top_k_list},
        }

    auc = roc_auc_score(labels, scores)
    # Ensure higher score = more anomalous
    if auc < 0.5:
        scores = -scores
        auc    = 1.0 - auc

    ap          = average_precision_score(labels, scores)
    baseline_ap = float(labels.mean())
    lift        = ap / baseline_ap if baseline_ap > 0 else 0.0

    # F1-optimal threshold (post-hoc on same set — standard practice)
    prec_c, rec_c, thresholds = precision_recall_curve(labels, scores)
    f1_curve = (
        2 * prec_c[:-1] * rec_c[:-1]
        / (prec_c[:-1] + rec_c[:-1] + 1e-10)
    )
    best_idx    = int(np.argmax(f1_curve))
    best_thresh = float(thresholds[best_idx])
    preds       = (scores >= best_thresh).astype(int)

    # Precision@K
    sorted_idx = np.argsort(-scores)
    topk_prec  = {
        f"top{k}_prec": float(labels[sorted_idx[:min(k, len(labels))]].mean())
        for k in top_k_list
    }

    return {
        "auc":         float(auc),
        "ap":          float(ap),
        "f1":          float(f1_score(labels, preds, zero_division=0)),
        "precision":   float(precision_score(labels, preds, zero_division=0)),
        "recall":      float(recall_score(labels, preds, zero_division=0)),
        "accuracy":    float(accuracy_score(labels, preds)),
        "threshold":   best_thresh,
        "lift":        float(lift),
        "baseline_ap": baseline_ap,
        "scores":      scores,
        **topk_prec,
    }


# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    labels:      np.ndarray,
    scores:      np.ndarray,
    n_bootstrap: int   = 1000,
    ci_level:    float = 0.95,
) -> dict:
    """
    Stratified bootstrap confidence intervals for AUC-ROC and AP.

    Stratified = positives and negatives resampled separately,
    preserving class balance in each bootstrap replicate.

    Args:
        labels:      Binary ground-truth.
        scores:      Anomaly scores (higher = more anomalous).
        n_bootstrap: Number of bootstrap replicates. Default 1000.
        ci_level:    Confidence level. Default 0.95.

    Returns:
        dict: auc, auc_lo, auc_hi, ap, ap_lo, ap_hi
    """
    labels = np.array(labels, dtype=int)
    scores = np.where(np.isnan(scores), 0.0, np.array(scores, dtype=float))

    if len(np.unique(labels)) < 2:
        base_ap = float(labels.mean())
        return dict(auc=0.5, auc_lo=0.5, auc_hi=0.5,
                    ap=base_ap, ap_lo=base_ap, ap_hi=base_ap)

    # Orient scores
    if roc_auc_score(labels, scores) < 0.5:
        scores = -scores

    alpha   = (1 - ci_level) / 2
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    auc_boots, ap_boots = [], []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        idx = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
        y_b, s_b = labels[idx], scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_boots.append(roc_auc_score(y_b, s_b))
            ap_boots.append(average_precision_score(y_b, s_b))
        except Exception:
            continue

    # Fallback to point estimate if bootstrapping collapsed
    if len(auc_boots) == 0:
        base_auc = float(roc_auc_score(labels, scores))
        base_ap  = float(average_precision_score(labels, scores))
        return dict(auc=base_auc, auc_lo=base_auc, auc_hi=base_auc,
                    ap=base_ap,   ap_lo=base_ap,   ap_hi=base_ap)

    auc_boots = np.array(auc_boots)
    ap_boots  = np.array(ap_boots)

    return {
        "auc":    float(np.mean(auc_boots)),
        "auc_lo": float(np.percentile(auc_boots, alpha * 100)),
        "auc_hi": float(np.percentile(auc_boots, (1 - alpha) * 100)),
        "ap":     float(np.mean(ap_boots)),
        "ap_lo":  float(np.percentile(ap_boots, alpha * 100)),
        "ap_hi":  float(np.percentile(ap_boots, (1 - alpha) * 100)),
    }


# ─────────────────────────────────────────────────────────────────────────────

def build_results_table(
    results:      dict[str, tuple[str, np.ndarray]],
    labels:       np.ndarray,
    rate_label:   str  = "",
    top_k_list:   list[int] = None,
    n_bootstrap:  int  = 1000,
) -> pd.DataFrame:
    """
    Build a results DataFrame from multiple model scores.

    Args:
        results:     Dict mapping model_name → (family_str, scores_array).
                     family_str is e.g. "density", "proximity", "ensemble".
        labels:      Binary ground-truth array.
        rate_label:  Contamination rate label for display e.g. "1%" or "natural (0.19%)".
        top_k_list:  K values for Precision@K. Default [50, 100, 200].
        n_bootstrap: Bootstrap replicates for CIs (0 = skip CIs).

    Returns:
        pd.DataFrame with one row per model, columns:
            Algorithm, Family, Rate, AUC, AP, F1, Precision, Recall,
            Accuracy, Lift, AUC_95CI, AP_95CI, P@50, P@100, P@200
    """
    if top_k_list is None:
        top_k_list = [50, 100, 200]

    rows = []
    for model_name, (family, scores) in results.items():
        log.info(f"  Evaluating {model_name}...")
        ev = evaluate(labels, scores, top_k_list)

        row = {
            "Algorithm": model_name,
            "Family":    family,
            "Rate":      rate_label,
            "AUC":       round(ev["auc"],       4),
            "AP":        round(ev["ap"],         4),
            "F1":        round(ev["f1"],         4),
            "Precision": round(ev["precision"],  4),
            "Recall":    round(ev["recall"],     4),
            "Accuracy":  round(ev["accuracy"],   4),
            "Lift":      f"{ev['lift']:.1f}×",
        }

        if n_bootstrap > 0:
            ci = bootstrap_ci(labels, ev["scores"], n_bootstrap)
            row["AUC_95CI"] = f"[{ci['auc_lo']:.3f}, {ci['auc_hi']:.3f}]"
            row["AP_95CI"]  = f"[{ci['ap_lo']:.4f}, {ci['ap_hi']:.4f}]"

        for k in top_k_list:
            row[f"P@{k}"] = round(ev.get(f"top{k}_prec", 0.0), 4)

        rows.append(row)

    return pd.DataFrame(rows)