# Unsupervised Anomaly Detection in Human Mobility Data
### A Multi-Model Benchmarking Framework
**ACM SIGSPATIAL 2026** · Deloitte / IARPA · GMU CARE Center

---

## Overview

This repository benchmarks **seven fully unsupervised anomaly detection models** on two real-world human mobility datasets. The pipeline is designed as a single unified entry point — one command runs preprocessing, feature extraction, model training, evaluation, and figure generation for either or both datasets.

| Dataset | Type | Anomaly definition |
|---|---|---|
| **NUMOSIM** | Synthetic activity-based mobility | Inserted stay sequences (Type 2 compound anomaly agents) |
| **YJMob100K** | Real pseudonymized mobility (Yahoo Japan) | Behavioural shift during a declared state of emergency |

---

## Models

| Model | Family | Backend |
|---|---|---|
| GMM | Density | scikit-learn |
| Isolation Forest | tree| scikit-learn |
| KNN | distance Proximity | scikit-learn |
| DAGMM | Deep generative | PyTorch |
| USAD | Adversarial autoencoder | PyTorch |
| LSTM-AD | Predictive sequence | PyTorch |
| LM-TAD | Causal transformer | PyTorch |
| XGBoost† | Supervised ceiling | xgboost (reference only — not a benchmark competitor) |

---

## Repository Structure

```
Final_Repository/
│
├── main.py                          ← Single pipeline entry point
│
├── features/
│   ├── __init__.py
│   ├── distance_utils.py            ← Haversine / euclidean utilities
│   ├── features_18dim.py            ← 18-dim common feature vector (Table 1)
│   └── sequence_features.py        ← 7-dim stay vectors for LM-TAD
│
├── models/
│   ├── __init__.py                  ← Model registry + build_unsupervised_scorers()
│   ├── anomaly_scorer.py            ← Abstract base class
│   ├── sklearn_scorers.py           ← GMM, Isolation Forest, KNN
│   ├── classical_scorers.py         ← (reserved)
│   ├── dagmm_scorer.py              ← DAGMM (PyTorch + sklearn fallback)
│   ├── usad_scorer.py               ← USAD (PyTorch + sklearn fallback)
│   ├── lstmad_scorer.py             ← LSTM-AD (owns full DataFrame pipeline)
│   ├── lmtad_scorer.py              ← LM-TAD (causal transformer)
│   └── xgboost_scorer.py            ← XGBoost supervised reference
│
├── preprocessing/
│   ├── __init__.py
│   ├── numosim_preprocessing.py     ← Raw NUMOSIM parquets → canonical stays
│   └── yjmob_stay_sequence.py       ← YJMob slot observations → stay sequences
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   ← AUC, AP, F1, bootstrap 95% CIs
│   └── plots.py                     ← Grid curves, bar chart, CI table, line plot
│
├── scripts/
│   └── convert_yjmob_to_parquet.py  ← CLI: raw YJMob CSV → flat parquet
│
├── tests/
│   └── test_yjmob_preprocessing.py  ← Smoke tests for YJMob preprocessing
│
├── data/                            ← gitignored — place raw files here
│   ├── numosim/
│   │   ├── raw/                     ← Raw NUMOSIM parquet files
│   │   ├── processed/               ← Auto-generated canonical stays
│   │   └── features/                ← Cached 18-dim feature parquet
│   └── yjmob/
│       ├── raw/                     ← Raw YJMob CSV files
│       ├── processed/               ← Auto-generated canonical stays
│       └── features/                ← Cached 18-dim feature parquets
│
├── results/                         ← gitignored — all outputs written here
│   ├── run.log                      ← Full pipeline log with timestamps
│   ├── numosim/
│   │   ├── results_natural_*.csv    ← Metrics table
│   │   ├── checkpoints/             ← Saved model weights
│   │   └── figures/                 ← All publication figures
│   └── yjmob/
│       ├── results_1pct.csv
│       ├── results_5pct.csv
│       ├── results_10pct.csv
│       ├── canonical_agents_*.parquet
│       ├── checkpoints/
│       └── figures/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Setup

### NUMOSIM

Place the following files in `data/numosim/raw/`:
- [Paper (ACM DL)](https://doi.org/10.1145/3589132.3625590)
- [Dataset download (OSF)](https://osf.io/sjyfr/)
- [arXiv preprint](https://arxiv.org/abs/2409.03024)
```
poi.parquet
stay_points_train.parquet
stay_points_test_truth.parquet
stay_points_test_anomalous.parquet
demographics.parquet
```

### YJMob100K

Place the following files in `data/yjmob/raw/`:
- [Paper (Scientific Data)](https://doi.org/10.1038/s41597-024-03237-9)
- [Dataset download (Zenodo)](https://zenodo.org/records/13237029)

```
yjmob100k-dataset1.csv    ← 75,000 normal agents (business-as-usual)
yjmob100k-dataset2.csv    ← 25,000 anomalous agents (emergency period)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3
astropy>=5.3
matplotlib>=3.7
torch>=2.0
xgboost>=2.0
pyarrow>=12.0
```

---

## How to Run

All commands are run from the project root.

### Quick start — NUMOSIM only, no deep models

```bash
python main.py --dataset numosim --no-deep
```

Runs: preprocessing → 18-dim features → GMM + IForest + KNN + XGBoost → evaluation → figures.
First run ~45 min (feature computation cached after). Subsequent runs ~5 min.

### Full NUMOSIM benchmark

```bash
# First run — computes and caches everything
python main.py --dataset numosim

# Subsequent runs — skip preprocessing and features
python main.py --dataset numosim --skip-preprocessing --skip-features
```

### With sequence models (LSTM-AD + LM-TAD)

```bash
python main.py --dataset numosim --skip-preprocessing --skip-features --sequence-models
```

### YJMob100K — sweeps 1%, 5%, 10% contamination

```bash
python main.py --dataset yjmob
```

### Both datasets — full benchmark

```bash
python main.py --dataset all --sequence-models
```

### Skip model training — reload from checkpoints

```bash
python main.py --dataset all --skip-preprocessing --skip-features --skip-training
```

---

## CLI Reference

```
python main.py [--dataset {numosim,yjmob,all}]
               [--skip-preprocessing]
               [--skip-features]
               [--skip-training]
               [--sequence-models]
               [--no-deep]
               [--no-bootstrap]
```

| Flag | Effect |
|---|---|
| `--dataset` | Which dataset(s) to run. Default: `all` |
| `--skip-preprocessing` | Load processed parquets directly (already ran once) |
| `--skip-features` | Load cached feature parquets (already computed) |
| `--skip-training` | Reload fitted models from checkpoints |
| `--sequence-models` | Run LSTM-AD and LM-TAD (slow, requires PyTorch) |
| `--no-deep` | Skip DAGMM and USAD (fast CPU-only run) |
| `--no-bootstrap` | Skip 95% CI computation (faster dev iteration) |

---

## Feature Vector

### 18-Dimensional Common Feature Vector (Table 1)

Computed identically on NUMOSIM and YJMob100K via `FeatureBuilder18Dim(is_yjmob=...)`.

| # | Feature | Description |
|---|---|---|
| 1 | `geobin_similarity` | Jaccard similarity of visited locations: past vs future |
| 2 | `abandonment_score` | Fraction of past locations absent from future visits |
| 3 | `kuiper_start_stat` | Kuiper V on stay start-time distributions |
| 4 | `kuiper_start_pval` | p-value of start-time Kuiper test |
| 5 | `kuiper_end_stat` | Kuiper V on stay end-time distributions |
| 6 | `kuiper_end_pval` | p-value of end-time Kuiper test |
| 7 | `rog_drift` | RoG_future − RoG_past |
| 8 | `gyration_score_drift` | Change in gyration score |
| 9 | `max_pairwise_dist_drift` | Change in max pairwise distance |
| 10 | `max_dist_home_drift` | Change in max distance from home |
| 11 | `std_dist_home_drift` | Change in std of home distance |
| 12 | `mean_dist_home_drift` | Change in mean distance from home |
| 13 | `novel_location_rate` | Fraction of test stays at unseen geo-bins |
| 14 | `loc_entropy_future` | Shannon entropy of geo-bin visits in test period |
| 15 | `temp_entropy_future` | Shannon entropy of hour-of-day activity in test period |
| 16 | `loc_entropy_drift` | H_future^loc − H_past^loc |
| 17 | `temp_entropy_drift` | H_future^temp − H_past^temp |
| 18 | `entropy_ratio` | H_future^loc / (H_past^loc + ε) |

### 7-Dimensional Sequence Feature Vector (Section 4.2.2)

Used by LSTM-AD and LM-TAD. Each stay → one 7-dim vector.

| # | Feature | NUMOSIM | YJMob100K |
|---|---|---|---|
| 0 | `hour_of_day` | 0–23 | 0–23 |
| 1 | `day_of_week` | 0–6 | 0–6 |
| 2 | `is_weekend` | 0/1 | 0/1 |
| 3 | `log_duration` | log(1 + seconds) | log(1 + seconds) |
| 4 | `coord_a` | latitude | x grid coordinate |
| 5 | `coord_b` | longitude | y grid coordinate |
| 6 | `dist_to_prev` | haversine (metres) | euclidean (grid units) |

---

## Evaluation Protocol

### NUMOSIM

Evaluated at the dataset's **natural anomaly prevalence (~0.19%)**, preserving
ground-truth label integrity. All anomalous agents (Types 1 + 2) vs normal agents.

### YJMob100K

Evaluated at **three synthetic contamination rates** — 1%, 5%, 10% — each using
a fixed sample of 10,000 agents:

| Rate | Normal (DS1) | Anomalous (DS2) | Total |
|---|---|---|---|
| 1% | 9,900 | 100 | 10,000 |
| 5% | 9,500 | 500 | 10,000 |
| 10% | 9,000 | 1,000 | 10,000 |

All models fit on **normal agents only** (no label leakage). Labels used only
for post-hoc evaluation.

Primary metric: **AUC-PR (Average Precision)**. All metrics accompanied by
bootstrapped 95% confidence intervals (1,000 resamples, stratified).

---

## Outputs

After a full run, `results/` contains:

```
results/
├── run.log                              ← timestamped pipeline log
│
├── numosim/
│   ├── results_natural_0.19pct.csv      ← metrics table (all models)
│   ├── figures/
│   │   ├── fig_grid_curves_natural.png  ← ROC / PR / F1 grid per model
│   │   ├── fig_ci_table_natural.png     ← results table with 95% CIs
│   │   └── fig_bar_chart_all_rates.png  ← AUC / AP / F1 bar chart
│   └── checkpoints/                     ← saved model weights
│
└── yjmob/
    ├── results_1pct.csv
    ├── results_5pct.csv
    ├── results_10pct.csv
    ├── canonical_agents_1pct.parquet    ← fixed evaluation population
    ├── canonical_agents_5pct.parquet
    ├── canonical_agents_10pct.parquet
    ├── figures/
    │   ├── fig_grid_curves_1pct.png
    │   ├── fig_grid_curves_5pct.png
    │   ├── fig_grid_curves_10pct.png
    │   ├── fig_bar_chart_all_rates.png
    │   └── fig_rate_line_plot.png
    └── checkpoints/
```

---

## Design Principles

- **No code duplication** — all stages share the same functions across both datasets.
  The only difference is `is_yjmob=True/False` which controls distance metric
  (euclidean vs haversine) and coordinate handling (grid x/y vs lat/lon).
- **Caching** — feature computation is expensive (~30–45 min). Results are cached
  to parquet after the first run and reloaded automatically.
- **Checkpointing** — fitted model weights are saved to disk. Use `--skip-training`
  to reload and re-evaluate without retraining.
- **No label leakage** — scalers and models always fit on normal agents only.
  Labels are accessed only inside evaluation functions.

---

## Hyperparameters

All hyperparameters are hardcoded from a validated tuning run. No re-tuning
on test labels.

| Model | Key hyperparameters |
|---|---|
| GMM | n_components=10, covariance_type=diag, n_init=5 |
| Isolation Forest | n_estimators=500 |
| KNN | k=50 |
| DAGMM | n_components=8, latent_dim=4, epochs=200, λ1=0.001, λ2=0.0005 |
| USAD | latent_dim=16, epochs=50, α=0.3, β=0.7 |
| LSTM-AD | hidden=128, n_layers=2, t_pred=3, epochs=100, max_len=60 |
| LM-TAD | embed_dim=64, n_layers=3, epochs=75, max_vocab=5000, max_len=60 |

---

## Running Tests

```bash
python tests/test_yjmob_preprocessing.py
```


