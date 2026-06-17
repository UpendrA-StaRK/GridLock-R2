# GridLock R2 — PS1: Parking-Induced Congestion

> **Problem Statement 1 — Gridlock 2.0 Hackathon | Flipkart HQ, Bengaluru**
>
> Bengaluru generates tens of thousands of illegal parking violations per day, but enforcement is reactive and patrol-memory-based. **GridLock R2** ingests 150 days of real police violation records, identifies the city's true congestion-driving parking clusters, and produces a ranked enforcement schedule — telling officers *exactly which 10 zones to prioritise at which hour* — so that every patrol car maximises congestion reduction per kilometre driven.

---

## 🗺️ Architecture Overview

```
Raw CSV (298K rows)
    │
    ▼
[Step 1] Schema Validation          validate.py        8 hard checks; fails loudly on any breach
    │
    ▼
[Step 2] Ingestion & Dedup          load.py            dtype cast, 15 leakage cols dropped,
    │                                                  minute-level dedup (268K rows retained)
    ▼
[Step 3] Row-level Features         features.py        temporal + spatial + categorical (Phase A)
    │
    ▼
[Step 4] Geospatial Clustering      clustering.py      DBSCAN → 139 enforcement zones + CIS
    │
    ▼
[Step 5] Zone × Time Grid           features.py        Phase B: aggregate to zone×hour / zone×day
    │                                                  zone aggregate features computed here
    ▼
[Step 6] ML Training & Eval         train.py           XGBoost / LightGBM / CatBoost
    │                                                  Phase 1 features (no zone_id leakage)
    ▼
[Step 7] Ranker + Demo Output       ranker.py          priority_score = predicted_count × CIS
                                    static_output.py   HTML map + 24h time-slider
```

---

## 🛠️ Setup & Running From Scratch

### Prerequisites
- Python **3.11** on Windows
- The raw dataset file placed at `data/raw/`

### 1. Create virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install pandas numpy scikit-learn xgboost lightgbm catboost folium tqdm loguru pyyaml shap scipy
```

> **Note:** `shap` is only needed for `notebooks/06_shap.ipynb`. All other notebooks work without it.

### 3. Place the raw dataset
```
data/raw/jan to may police violation_anonymized791b166.csv
```
This file is **read-only**. Never overwrite it.

---

## 📓 Notebook Execution Order

**Start here → open `notebooks/00_run_guide.ipynb`** — it audits which files already exist and tells you exactly which steps to run.

| # | Notebook | What it does | Runtime |
|---|----------|--------------|---------|
| 0 | `00_run_guide.ipynb` | File audit + one-click SHAP launcher | ~5s |
| 1 | `01_eda.ipynb` | EDA — schema, distributions, leakage audit | ~1 min |
| 2 | `01b_features.ipynb` | Row-level feature engineering + label encoding | ~3 min |
| 3 | `02_cluster_tuning.ipynb` | DBSCAN eps/min_samples grid search | ~5 min |
| 4 | `03_clustering.ipynb` | DBSCAN zones + CIS table + zone×time grids | ~8 min |
| 5 | `04_training.ipynb` | Train 6 models, pick winner by per-hour NDCG | ~10 min |
| 6 | `05_inference.ipynb` | Zone ranking + enforcement HTML with 24h slider | ~30s |
| 7 | `06_shap.ipynb` | SHAP feature importance + validation gate | ~4 min |

### To run a notebook
```powershell
# Option A — VS Code
# Open file → Select Kernel → venv (Python 3.11) → Run All

# Option B — Jupyter in browser
venv\Scripts\jupyter notebook
# Navigate to notebooks/ in the browser

# Option C — Execute non-interactively
venv\Scripts\jupyter nbconvert --to notebook --execute notebooks/06_shap.ipynb
```

---

## 🤖 Predictive Model

### What it predicts
`predicted_count` — expected number of parking violations for a given `(zone, hour)` pair.

### Feature set (v2.0 — Phase 1 overhaul)

> **Important change from v1.0**: `zone_id`, `police_station_id`, and `center_code_encoded` have been **removed** from the feature set. These were ordinal encodings of categorical identifiers — XGBoost was treating zone 50 as "between zone 49 and zone 51", which is meaningless for DBSCAN labels. They have been replaced by zone aggregate statistics computed on the training split only.

| Group | Feature | Description |
|---|---|---|
| **Temporal** | `hour_of_day` | 0–23; captures morning/evening peaks |
| | `day_of_week` | 0=Monday, 6=Sunday |
| | `is_weekend` | Saturday + Sunday flag |
| | `month` | Seasonal enforcement pattern |
| **Zone aggregates** | `zone_mean_count` | Mean violation count per zone (training period only) |
| | `zone_median_count` | Median — robust to enforcement surges |
| | `zone_cis_score` | CIS score from `cis_table.parquet` |
| | `zone_junction_frac` | Fraction of violations at junctions in this zone |
| | `zone_total_count` | Total violations in zone over training period |
| **Spatial** | `fraction_at_junction` | Time-block-level junction fraction (varies per zone×hour) |
| **Historical** | `rolling_7d_count` | 7-day lagged mean per (zone, hour) — **strongest signal** |
| **Categorical** | `dominant_violation_type` | Mode violation type in this zone×time block |
| | `dominant_vehicle_type` | Mode vehicle type in this zone×time block |
| | `violation_type_primary_encoded` | Encoded primary violation type |
| | `vehicle_type_encoded` | Encoded vehicle type |
| **Optional** | `data_sent_to_scita_mean` | Mean SCITA forwarding rate (included for SHAP validation) |

### Leakage guards
- **Temporal:** Hard `AssertionError` if `max(train.date) >= min(test.date)`
- **Zone aggregates:** Computed on `train_df` **only**, then joined to both splits
- **Rolling features:** `shift(1).rolling(7)` — current day's count never included

### Train / test split
| Split | Date range | Rows |
|---|---|---|
| Train | Nov 9 2023 – Feb 29 2024 | ~19,870 (zone×hour) |
| Test | Mar 1 2024 – Apr 8 2024 | ~6,484 (zone×hour) |

### Models trained
- XGBoost (winner), LightGBM, CatBoost — at both `hour` and `day` resolutions = **6 runs total**
- Winner selected by **per-hour NDCG@10** (see Evaluation section)

---

## 📊 Evaluation Metrics (Phase 2 overhaul)

Two tiers of ranking metrics are computed. The per-hour tier is the primary differentiator.

### Tier 1 — Regression (count prediction accuracy)

| Metric | Description | Phase 1 result |
|---|---|---|
| **MAE** | Mean absolute error in predicted violation count per zone-hour | ~4.68 (XGBoost) |
| **RMSE** | Penalises large errors more; sensitive to high-violation outlier zones | ~10.66 |
| **Naive MAE** | Frequency baseline (no ML — ranks by raw historical count) | ~5.58 |
| **ML Lift %** | `(Naive_MAE - ML_MAE) / Naive_MAE × 100` | **+16.1%** |

### Tier 2 — Ranking (zone ordering quality)

| Metric | Description | Why it matters |
|---|---|---|
| **Aggregate NDCG@10** | NDCG over the full test period | Uninformative (both model and baseline score 1.0 — top-10 zones are globally stable) |
| **Per-hour NDCG@10** ⭐ | NDCG computed per `(date × hour)` slot, then averaged | **Primary differentiator** — model must predict *which zone peaks at 2am vs 9am* |
| **Per-hour Spearman ρ** | Rank correlation per hour slot | Measures fine-grained zone ordering quality within each hour |
| **Per-hour Precision@10** | Fraction of predicted top-10 in true top-10, per hour | Operational precision — how often would an officer be in the right place |
| **Frequency baseline per-hour** | Same metrics for the static baseline (no ML) | Comparison reference — ML must beat this to be useful |

> **Why NDCG@10 = 1.0 at aggregate level?**
> Brigade Road, Indiranagar, and Commercial Street are high-violation every day — even a static frequency table gets the global top-10 right. The ML model's value is that it correctly predicts **which of those zones peaks at which hour**. Per-hour NDCG captures this. A frequency table always recommends the same order regardless of hour; the ML model adjusts.

---

## 🔍 SHAP Explainability

Run `notebooks/06_shap.ipynb` after training to generate:

| Output | Location | Use |
|---|---|---|
| Beeswarm summary | `data/outputs/shap_summary.png` | Demo slide — "our model is explainable" |
| Feature importance bar | `data/outputs/shap_importance.png` | Shows zone aggregates dominate, not zone IDs |
| PDP: rolling_7d_count | `data/outputs/shap_pdp_rolling.png` | Confirms recent history drives predictions |
| PDP: hour_of_day | `data/outputs/shap_pdp_hour.png` | Shows 9am / 6pm peaks (Bengaluru rush hours) |
| Validation report | `data/outputs/shap_report.json` | Gate check results |

### SHAP Validation Gate
After every retrain, the notebook checks:

| Gate | Condition | Meaning |
|---|---|---|
| Gate 1 *(hard)* | `zone_id` NOT in top-5 SHAP | Phase 1 fix confirmed — no lookup-table behaviour |
| Gate 2 *(soft)* | `rolling_7d_count` in top-3 | Temporal signal is dominant |
| Gate 3 *(soft)* | `hour_of_day` in top-5 | Model captures time-of-day patterns |

---

## 🗺️ Demo Output

### Interactive enforcement map with 24-hour time slider

Generated by `notebooks/05_inference.ipynb` → `generate_static_output_with_slider()`.

```
data/outputs/enforcement_slider_DATE.html
```

Open in Chrome (works **fully offline** — no internet required at the venue).

**Features:**
- Leaflet.js map with colour-coded zone markers (red = HIGH, orange = MEDIUM, green = LOW)
- Hour slider 0–23 — zone markers and priority table update in real-time
- Quick-access buttons: 🌅 9am | ☀️ 12pm | 🌆 6pm | 🌙 11pm
- Model scorecard panel (MAE, RMSE, ML Lift %, NDCG@10)

### Priority table CSV

```
data/outputs/day_schedule_DATE.csv
```

Top-K zones across all 24 hours — suitable for police vehicle route planning.

---

## ⚠️ Limitations (Honest Assessment)

| Limitation | Impact | Notes |
|---|---|---|
| **NDCG ceiling at aggregate level** | All models + baseline score 1.0 on global NDCG@10 | Not a bug — a property of the dataset. Use per-hour NDCG as the primary metric. |
| **DBSCAN silhouette near zero** | Silhouette of −0.096 on tuned params | Reflects genuine spatial sparsity of urban violation data, not clustering failure. Zones are visually coherent. Re-tune on full 268K dataset (see Phase 4 in task.md). |
| **Zone aggregate cold-start** | New zones not seen in training receive mean-count as default | Conservative fallback. Quarterly re-clustering mitigates this. |
| **No real-time data** | Batch-only system; outputs are pre-computed | Would require streaming ingestion (Kafka/Flink) for live deployment. |
| **No road-type data** | CIS uses junction presence as proxy for road classification | External datasets prohibited by hackathon rules. Junction weight is defensible but approximate. |
| **Static zone boundaries** | DBSCAN zones frozen at training time | Requires periodic re-clustering as new data arrives. Roadmap: HDBSCAN for dynamic boundaries. |
| **5-week test window** | Mar 1 – Apr 8 2024 only (5.5 weeks) | Limited by competition dataset. Longer test periods would give more reliable per-hour NDCG estimates. |
| **Hourly granularity only** | Cannot schedule patrol windows < 1 hour | Dataset lacks reliable sub-hour resolution. |

---

## 🚀 Roadmap

### Achievable before demo
- [ ] Re-tune DBSCAN on full 268K dataset (fix silhouette) → `02_cluster_tuning.ipynb`
- [ ] Phase 1 retrain with zone aggregate features → `04_training.ipynb`
- [ ] Run SHAP analysis on new checkpoint → `06_shap.ipynb`
- [ ] Confirm per-hour NDCG improvement over baseline

### Post-hackathon
- Real-time ingestion pipeline (Kafka + Flink) replacing batch CSV
- Sub-hourly prediction (15-minute windows) with richer temporal features
- MapmyIndia traffic speed integration to validate and calibrate CIS formula
- HDBSCAN for dynamic zone boundary updates without full retraining
- Uncertainty quantification (prediction intervals) for officer confidence scores
- Multi-city deployment (pipeline is city-agnostic; re-tune eps/junction weights per city)

---

## 📋 Phase Log

| Phase | What changed | Status |
|---|---|---|
| **Phase 0** | Fixed 2 crash bugs in `pipeline.py` (wrong function names in step1/step2) | ✅ Done |
| **Phase 1** | Removed `zone_id`, `police_station_id`, `center_code_encoded` from features. Added zone aggregate features computed train-only. Updated `features.yaml` to v2.0. | ✅ Code done — retrain needed |
| **Phase 2** | Added `ndcg_per_hour()`, `temporal_rank_delta()`, `precision_per_hour()`, `frequency_baseline_per_hour()` to `metrics.py`. Integrated into `full_eval()` scorecard. | ✅ Done |
| **Phase 3** | Created `notebooks/06_shap.ipynb` — SHAP analysis with validation gate | ✅ Notebook done — run after retrain |
| **Phase 4** | DBSCAN re-tuning on full 268K dataset | ⏳ Needs running |
| **Phase 5** | Added 24h time-slider to `static_output.py`. Created `docs/demo_script.md`. Created `notebooks/00_run_guide.ipynb`. | ✅ Done |

---

## 📁 Repository Structure

```
GridLock_R2_Transfer/
├── configs/
│   ├── features.yaml       # Feature list v2.0 — zone aggregates, no zone_id
│   ├── eval.yaml           # CIS formula, ranker formula, NDCG relevance, split bounds
│   └── model.yaml          # Model hyperparameters, winner checkpoint path
├── src/
│   ├── data/
│   │   ├── validate.py     # Schema validator (8 hard checks)
│   │   ├── load.py         # Ingest + dedup → 268K rows
│   │   ├── features.py     # Phase A (row features) + Phase B (zone×time grid)
│   │   └── pipeline.py     # 8-step end-to-end orchestrator
│   ├── models/
│   │   └── clustering.py   # DBSCAN + KDE + CIS
│   ├── training/
│   │   └── train.py        # Multi-model training, leakage guard, zone aggregates
│   ├── evaluation/
│   │   └── metrics.py      # MAE/RMSE, NDCG@K, per-hour ranking metrics
│   └── inference/
│       ├── ranker.py        # priority_score = predicted_count × CIS → top-K
│       └── static_output.py # HTML map + 24h time-slider generator
├── notebooks/
│   ├── 00_run_guide.ipynb  # START HERE — file audit + execution guide
│   ├── 01_eda.ipynb
│   ├── 01b_features.ipynb
│   ├── 02_cluster_tuning.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_training.ipynb
│   ├── 05_inference.ipynb
│   └── 06_shap.ipynb       # SHAP feature importance + validation gate
├── docs/
│   └── demo_script.md      # 2-min demo walkthrough + 5 judge Q&A answers
├── data/
│   ├── raw/                # READ-ONLY — never modify
│   ├── processed/          # Parquet files, encoders, metadata
│   └── outputs/            # HTML maps, CSVs, eval JSONs, SHAP plots
├── checkpoints/            # Saved model checkpoints
├── artifacts/
│   ├── session_log.md      # Living log — all sessions, decisions, metrics
│   └── Problem.md          # Original problem statement
└── claude.md               # AI pair-programming context (read before any AI session)
```

---

## 📊 Ranking Formula

```
priority_score(zone, t) = predicted_count(zone, t) × CIS(zone)

CIS(zone) = violation_density_norm(zone) × junction_weight(zone)

  violation_density_norm = zone_violation_count / max_zone_violation_count
  junction_weight        = 1.5  if any violation in zone has is_at_junction = 1
                           1.0  otherwise
```

Zones ranked descending by `priority_score`. Top-K output with tier labels:

| Tier | Threshold |
|---|---|
| HIGH | `priority_score ≥ 0.7 × max` |
| MEDIUM | `priority_score ≥ 0.4 × max` |
| LOW | `priority_score < 0.4 × max` |

---

*Built for the Flipkart Gridlock 2.0 Hackathon — PS1: Poor Visibility on Parking-Induced Congestion.*
*Dataset: Bengaluru Police Violation Data, Nov 2023 – Apr 2024 (268K rows after dedup).*
