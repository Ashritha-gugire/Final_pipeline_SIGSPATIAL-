"""
evaluation/plots.py
====================
Unified plotting module for both NUMOSIM and YJMob100K.

Every function takes the same inputs regardless of dataset — the caller
passes dataset_name="NUMOSIM" or "YJMob100K" for titles/labels only.

Public API
──────────
    make_grid_figure(scores, labels, rate_label, dataset_name, out_path)
        → ROC / PR / F1-vs-Threshold grid, one row per model

    make_bar_chart(results_df, dataset_name, out_path)
        → Grouped bar chart: AUC / AP / F1 across contamination rates

    make_ci_table(results_df, dataset_name, out_path)
        → Formatted table image with 95% CIs highlighted

    make_rate_line_plot(results_df, dataset_name, out_path)
        → Line plot: AUC / AP / F1 vs contamination rate (YJMob sweep)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from evaluation.metrics import evaluate

log = logging.getLogger(__name__)

# ── Shared style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "savefig.dpi":      300,
})

# Colours per model — consistent across all figures
# These are the 5 unsupervised + 2 sequence + 1 supervised reference
MODEL_COLORS: dict[str, str] = {
    "GMM":              "#9C27B0",   # purple  — density
    "Isolation Forest": "#FF9800",   # orange  — ensemble
    "KNN":              "#2196F3",   # blue    — proximity
    "DAGMM":            "#009688",   # teal    — deep
    "USAD":             "#4CAF50",   # green   — deep
    "LSTM-AD":          "#E91E63",   # pink    — sequence
    "LM-TAD":           "#F44336",   # red     — sequence
    "XGBoost†":         "#607D8B",   # grey    — supervised reference (dashed)
}
DEFAULT_COLOR = "#546E7A"

# Panel colours for the grid figure
_GRID_COL = {"roc": "#1565C0", "pr": "#B71C1C", "f1": "#2E7D32"}

RATE_LABELS   = ["1%", "5%", "10%"]
RATE_MARKERS  = {"1%": "o", "5%": "s", "10%": "^"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _orient(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    s = np.where(np.isnan(scores), 0.0, np.array(scores, dtype=float))
    return -s if roc_auc_score(labels, s) < 0.5 else s


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, DEFAULT_COLOR)


def _is_supervised(name: str) -> bool:
    return "†" in name or name.lower() == "xgboost"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grid figure  (ROC | PR | F1-vs-Threshold)
# ─────────────────────────────────────────────────────────────────────────────

def make_grid_figure(
    scores:       dict[str, np.ndarray],
    labels:       np.ndarray,
    rate_label:   str,
    dataset_name: str,
    out_path:     str,
) -> None:
    """
    Publication-quality grid figure: one row per model, three panels each.

    Panels: ROC curve | Precision-Recall curve | F1 vs normalised threshold

    Row order: XGBoost† (dashed, supervised reference) at top,
               then unsupervised models sorted by F1 descending.

    Args:
        scores:       Dict model_name → oriented anomaly scores array.
        labels:       Binary ground-truth (1 = anomaly).
        rate_label:   e.g. "1%", "5%", "natural (0.19%)"
        dataset_name: "NUMOSIM" or "YJMob100K"
        out_path:     Full path to save PNG.
    """
    labels   = np.array(labels, dtype=int)
    rand_ap  = float(labels.mean())

    # Orient all scores
    scores = {n: _orient(s, labels) for n, s in scores.items()}

    # Evaluate once
    ev = {n: evaluate(labels, s) for n, s in scores.items()}

    # Sort: XGBoost† first (dashed), rest by F1 desc
    sup   = [n for n in scores if _is_supervised(n)]
    unsup = sorted([n for n in scores if not _is_supervised(n)],
                   key=lambda n: ev[n]["f1"], reverse=True)
    row_order = sup + unsup

    n_rows  = len(row_order)
    label_w = 2.8; cell_w = 2.4; cell_h = 0.92; hdr_h = 0.48
    fig_w   = label_w + cell_w * 3
    fig_h   = hdr_h + cell_h * n_rows + 0.3
    left_f  = label_w / fig_w
    top_f   = 1.0 - hdr_h / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        n_rows, 3, figure=fig,
        left=left_f, right=0.97,
        top=top_f, bottom=0.06,
        hspace=0.42, wspace=0.28,
    )
    axes = {(i, j): fig.add_subplot(gs[i, j])
            for i in range(n_rows) for j in range(3)}

    # Column headers
    col_titles = ["ROC Curve", "Precision-Recall", "F1 vs Threshold"]
    col_keys   = ["roc", "pr", "f1"]
    for j, (ttl, key) in enumerate(zip(col_titles, col_keys)):
        pos  = axes[(0, j)].get_position()
        xctr = pos.x0 + pos.width / 2
        fig.text(xctr, top_f + (1 - top_f) * 0.60, ttl,
                 ha="center", va="center",
                 fontsize=10.5, fontweight="bold", color=_GRID_COL[key])

    for i, name in enumerate(row_order):
        is_sup  = _is_supervised(name)
        s       = scores[name]
        e       = ev[name]
        ls      = (0, (5, 3)) if is_sup else "-"
        lw      = 1.6
        alpha   = 0.55 if is_sup else 1.0
        row_bg  = "#F4F4F4" if is_sup else ("#F7F9FC" if i % 2 == 0 else "#FFFFFF")
        col     = _GRID_COL   # panel-specific colour, not model colour

        for j, key in enumerate(col_keys):
            ax = axes[(i, j)]
            ax.set_facecolor(row_bg)
            c  = col[key]

            if key == "roc":
                fpr, tpr, _ = roc_curve(labels, s)
                if not is_sup:
                    ax.fill_between(fpr, tpr, alpha=0.07, color=c)
                ax.plot(fpr, tpr, color=c, lw=lw, ls=ls, alpha=alpha, zorder=3)
                ax.plot([0,1],[0,1], color="#CCCCCC", lw=0.7, ls="--", zorder=1)
                ax.text(0.97, 0.06, f"AUC={e['auc']:.3f}",
                        ha="right", va="bottom", fontsize=7.5,
                        fontweight="bold" if not is_sup else "normal",
                        color=c, transform=ax.transAxes)
                if i == n_rows - 1:
                    ax.set_xlabel("FPR", fontsize=8)

            elif key == "pr":
                prec, rec, _ = precision_recall_curve(labels, s)
                if not is_sup:
                    ax.fill_between(rec, prec, alpha=0.07, color=c)
                ax.plot(rec, prec, color=c, lw=lw, ls=ls, alpha=alpha, zorder=3)
                ax.axhline(rand_ap, color="#CCCCCC", lw=0.7, ls="--", zorder=1)
                ax.text(0.97, 0.96, f"AP={e['ap']:.3f}",
                        ha="right", va="top", fontsize=7.5,
                        fontweight="bold" if not is_sup else "normal",
                        color=c, transform=ax.transAxes)
                if i == n_rows - 1:
                    ax.set_xlabel("Recall", fontsize=8)

            else:  # f1 vs threshold
                p2, r2, thr = precision_recall_curve(labels, s)
                p2 = p2[:-1]; r2 = r2[:-1]
                denom = p2 + r2
                f1_c  = np.where(denom > 0, 2 * p2 * r2 / denom, 0.0)
                t_n   = (thr - thr.min()) / (thr.max() - thr.min() + 1e-10)
                if not is_sup:
                    ax.fill_between(t_n, f1_c, alpha=0.07, color=c)
                ax.plot(t_n, f1_c, color=c, lw=lw, ls=ls, alpha=alpha, zorder=3)
                if len(f1_c):
                    ax.axvline(t_n[np.argmax(f1_c)], color=c,
                               lw=0.8, ls=":", alpha=0.5, zorder=2)
                ax.text(0.97, 0.96, f"F1={e['f1']:.3f}",
                        ha="right", va="top", fontsize=7.5,
                        fontweight="bold" if not is_sup else "normal",
                        color=c, transform=ax.transAxes)
                if i == n_rows - 1:
                    ax.set_xlabel("Threshold (norm.)", fontsize=8)

            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=6.5, pad=1)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)

        # Row label (left margin)
        pos  = axes[(i, 0)].get_position()
        ymid = (pos.y0 + pos.y1) / 2
        nx   = left_f - 0.012
        if is_sup:
            fig.text(nx, ymid, name, ha="right", va="center",
                     fontsize=9, fontstyle="italic", color="#666666")
        else:
            rank = unsup.index(name) + 1
            fig.text(nx, ymid + cell_h / fig_h * 0.22, f"#{rank}",
                     ha="right", va="center", fontsize=7.5, color="#AAAAAA")
            fig.text(nx, ymid - cell_h / fig_h * 0.16, name,
                     ha="right", va="center", fontsize=9, color="#222222",
                     fontweight="bold" if rank == 1 else "normal")

    # Separator between supervised and unsupervised rows
    if sup and unsup:
        y_sep = (axes[(0, 0)].get_position().y0 +
                 axes[(1, 0)].get_position().y1) / 2
        fig.add_artist(Line2D(
            [left_f * 0.05, 0.97], [y_sep, y_sep],
            transform=fig.transFigure,
            color="#BBBBBB", lw=0.9, ls="--",
        ))

    # Title and footnote
    n_anom   = int(labels.sum())
    n_total  = len(labels)
    fig.text(
        0.5, top_f + (1 - top_f) * 0.88,
        f"{dataset_name}  —  {len(unsup)} unsupervised models  |  "
        f"Contamination = {rate_label}  |  Ordered by F1",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color="#1B2A4A",
    )
    fig.text(
        0.5, 0.018,
        f"Random baseline AP = {rand_ap:.4f}  |  "
        f"N = {n_total:,}  |  Anomalous = {n_anom:,} ({rand_ap*100:.2f}%)"
        + ("  |  † supervised reference only" if sup else ""),
        ha="center", va="bottom", fontsize=7, color="#999999",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"Grid figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bar chart  (AUC / AP / F1 grouped by model, coloured by rate)
# ─────────────────────────────────────────────────────────────────────────────

def make_bar_chart(
    results_df:   pd.DataFrame,
    dataset_name: str,
    out_path:     str,
    metrics:      list[str] = None,
) -> None:
    """
    Grouped bar chart: models on x-axis, bars coloured by contamination rate.

    Args:
        results_df:   Output of build_results_table(), may contain multiple rates.
        dataset_name: "NUMOSIM" or "YJMob100K"
        out_path:     Full path to save PNG.
        metrics:      Columns to plot. Default ["AUC", "AP", "F1"].
    """
    if metrics is None:
        metrics = ["AUC", "AP", "F1"]

    rates   = results_df["Rate"].unique().tolist()
    models  = [m for m in results_df["Algorithm"].unique()
               if not _is_supervised(m)]

    # Colour per rate
    rate_colors = {
        rates[0]: "#1565C0",
        rates[1]: "#E65100",
        rates[2]: "#2E7D32",
    } if len(rates) >= 3 else {r: c for r, c in zip(
        rates, ["#1565C0", "#E65100", "#2E7D32"][:len(rates)]
    )}

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5.5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(
        f"{dataset_name} — Model Performance Across Contamination Rates",
        fontsize=13, fontweight="bold", y=1.01,
    )

    x     = np.arange(len(models))
    width = 0.22

    for ax, metric in zip(axes, metrics):
        offsets = np.linspace(
            -(len(rates) - 1) / 2 * width,
             (len(rates) - 1) / 2 * width,
            len(rates),
        )
        for offset, rate in zip(offsets, rates):
            vals = []
            for model in models:
                row = results_df[
                    (results_df["Algorithm"] == model) &
                    (results_df["Rate"] == rate)
                ]
                vals.append(float(row[metric].values[0]) if not row.empty else 0.0)

            bars = ax.bar(x + offset, vals, width,
                          label=rate, color=rate_colors[rate],
                          alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                    color=rate_colors[rate],
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, title="Contamination rate",
               loc="lower center", ncol=len(rates),
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"Bar chart saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CI table  (formatted image with 95% CI columns highlighted)
# ─────────────────────────────────────────────────────────────────────────────

def make_ci_table(
    results_df:   pd.DataFrame,
    dataset_name: str,
    out_path:     str,
) -> None:
    """
    Formatted table image showing AUC, AP, F1, AUC_95CI, AP_95CI per model/rate.
    Best value per column is highlighted in green.

    Args:
        results_df:   Output of build_results_table() with CIs computed.
        dataset_name: "NUMOSIM" or "YJMob100K"
        out_path:     Full path to save PNG.
    """
    display_cols = ["Algorithm", "Rate", "AUC", "AP", "F1",
                    "Precision", "Recall", "AUC_95CI", "AP_95CI"]
    cols = [c for c in display_cols if c in results_df.columns]
    df   = results_df[cols].copy()

    n_rows   = len(df)
    fig_h    = max(3.0, 0.45 * n_rows + 1.5)
    fig, ax  = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    rates = results_df["Rate"].unique()
    rate_str = " / ".join(str(r) for r in rates)
    ax.set_title(
        f"{dataset_name} — Results with 95% Bootstrap CIs  |  Rate: {rate_str}",
        fontsize=12, fontweight="bold", pad=16,
    )

    col_labels = [c.replace("_95CI", " 95% CI") for c in cols]
    cell_data  = df.values.tolist()

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)

    # Header styling
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor("#E3F2FD")
        cell.set_text_props(fontweight="bold")

    # Highlight best numeric value per column
    numeric_cols = [c for c in cols if c in ("AUC", "AP", "F1", "Precision", "Recall")]
    for col_name in numeric_cols:
        j = cols.index(col_name)
        vals = []
        for i in range(n_rows):
            try:
                vals.append(float(cell_data[i][j]))
            except (ValueError, TypeError):
                vals.append(-1.0)
        best_i = int(np.argmax(vals))
        table[best_i + 1, j].set_facecolor("#E8F5E9")
        table[best_i + 1, j].set_text_props(fontweight="bold")

    # Alternate row shading
    for i in range(n_rows):
        if i % 2 == 0:
            for j in range(len(cols)):
                c = table[i + 1, j]
                if c.get_facecolor()[0] > 0.9:  # not already highlighted
                    c.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"CI table saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rate line plot  (AUC / AP / F1 vs contamination rate — YJMob sweep)
# ─────────────────────────────────────────────────────────────────────────────

def make_rate_line_plot(
    results_df:   pd.DataFrame,
    dataset_name: str,
    out_path:     str,
    metrics:      list[str] = None,
) -> None:
    """
    Line plot: metric vs contamination rate, one line per model.
    Designed for YJMob sweep (multiple rates). Also works for NUMOSIM
    if multiple rates are provided.

    Args:
        results_df:   Output of build_results_table() with multiple rates.
        dataset_name: "NUMOSIM" or "YJMob100K"
        out_path:     Full path to save PNG.
        metrics:      Columns to plot. Default ["AUC", "AP", "F1"].
    """
    if metrics is None:
        metrics = ["AUC", "AP", "F1"]

    models = [m for m in results_df["Algorithm"].unique()
              if not _is_supervised(m)]
    rates  = sorted(results_df["Rate"].unique(),
                    key=lambda r: float(str(r).replace("%", "").split()[0]))

    # Convert rate labels to numeric for x-axis
    def _rate_to_float(r):
        try:
            return float(str(r).replace("%", "").split()[0])
        except ValueError:
            return 0.0

    x_vals = [_rate_to_float(r) for r in rates]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(
        f"{dataset_name} — Performance vs Contamination Rate",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, metric in zip(axes, metrics):
        for model in models:
            subset = results_df[results_df["Algorithm"] == model]
            ys = []
            for rate in rates:
                row = subset[subset["Rate"] == rate]
                ys.append(float(row[metric].values[0]) if not row.empty else np.nan)

            color = _color(model)
            ax.plot(x_vals, ys, color=color, linewidth=2.2,
                    marker="o", markersize=7,
                    markerfacecolor="white", markeredgewidth=2,
                    markeredgecolor=color, label=model, zorder=3)

            # Annotate last point
            if not np.isnan(ys[-1]):
                ax.text(x_vals[-1] + 0.2, ys[-1], model,
                        fontsize=8, color=color, va="center")

        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{r}%" for r in x_vals], fontsize=10)
        ax.set_xlabel("Contamination Rate", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlim([x_vals[0] - 0.5, x_vals[-1] + 2.0])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"Rate line plot saved → {out_path}")