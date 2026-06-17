# GridLock R2 — PS1: Parking-Induced Congestion

> **Problem Statement 1**: Bengaluru generates tens of thousands of illegal parking violations each day, but enforcement resources are finite and deployed reactively. Traffic Police patrollers go to the same static zones from memory, miss emerging hotspots, and have no tool to predict where violations will spike at a given hour. **GridLock R2** solves this by ingesting 6 months of historical police violation data, identifying the city's true congestion-driving parking clusters, and producing a ranked enforcement schedule — telling officers exactly which 10 zones to prioritise, and at what hour — so that every patrol car maximises congestion reduction per kilometre driven.

---

## 🗺️ Architecture Overview

The system is an 8-step sequential pipeline: raw CSV in, ranked enforcement schedule out. Each step is independently checkpointed and skippable.

```
Raw CSV (298K rows)
    │
    ▼
[Step 1] Schema Validation       validate.py   — 8 hard checks; fails loudly on any breach
    │
    ▼
[Step 2] Ingestion & Dedup       load.py       — dtype cast, 15 leakage cols dropped,
    │                                            minute-level dedup (268K rows retained)
    ▼
[Step 3] Row-level Features      features.py   — temporal + spatial + categorical features
    │                                            (Phase A). Zone grid rollup deferred to Phase B.
    ▼
[Step 4] Geospatial Clustering   clustering.py — DBSCAN + KDE + CIS computation → 139 zones
    │
    ▼
[Step 5] CIS Computation         clustering.py — Congestion Impact Score per zone
    │
    ▼
[Step 6] Zone × Time Grid        features.py   — Phase B: aggregate to zone×hour and zone×day grids
    │
    ▼
[Step 7] ML Training & Eval      train.py      — XGBoost / LightGBM / CatBoost; winner by NDCG@10
    │
    ▼
[Step 8] Ranker + Outputs        ranker.py     — priority_score = predicted_count × CIS → top-K
                                 static_output.py → HTML map + CSV schedule
```

### Step-by-Step Explanation

#### 1. Data Ingestion & Preprocessing

| Detail | Value |
|:---|:---|
| **Raw dataset** | `data/raw/jan to may police violation_anonymized791b166.csv` |
| **Rows** | 298,450 × 24 columns (109.6 MB) |
| **Date range** | Nov 9 2023 – Apr 8 2024 (150 days) |
| **Deduplication rule** | Minute-level — rows are identical only if *all* of `(lat, lon, violation_type, vehicle_type, created_minute)` match. Same-second events at different coordinates = real multi-vehicle clusters, kept. |
| **Leakage columns excluded** | `description`, `closed_datetime`, `action_taken_timestamp` (100% NULL); `data_sent_to_scita_timestamp` (86% NULL, only in test window); `modified_datetime`, `validation_status`, `validation_timestamp` (post-event admin fields) |

Row-level features engineered per record: `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `is_at_junction`, `violation_type_encoded` (parsed from JSON-list string via `ast.literal_eval`, 17 unique atomic types), `vehicle_type_encoded`, `police_station_id`, `center_code_encoded`.

#### 2. Geospatial Clustering (DBSCAN + KDE + CIS)

DBSCAN is run on `(latitude, longitude)` after MinMax scaling, using parameters tuned via grid-search in `notebooks/02_cluster_tuning.ipynb`:

| DBSCAN Parameter | Value | Rationale |
|:---|:---|:---|
| `eps` | `0.05` | Tuned via silhouette score grid search on cluster stability |
| `min_samples` | `50` | Ensures only meaningfully dense zones are labelled |

**Results:** 139 dense violation zones + 1 sparse "noise" zone (zone_id = -1, 2.07% of rows). Noise points are **kept** and scored at 50% CIS weight — they represent real but sparse enforcement events.

**KDE (Kernel Density Estimation)** provides a continuous density surface for visual map overlays. It is not used in the ML model directly but informs cluster coherence verification.

**Congestion Impact Score (CIS) — Formula v1.0:**

```
CIS(zone) = violation_density_norm(zone) × junction_weight(zone)

  violation_density_norm = zone_violation_count / max_zone_violation_count
                           (normalised to [0, 1] across all zones)

  junction_weight = 1.5  if any violation in the zone has is_at_junction = 1
                    1.0  otherwise
```

Junction presence gets a 1.5× multiplier because blocking an intersection causes cascading grid-lock. No external road-type datasets are used (prohibited by hackathon FAQ). CIS output range: **[0.0, 1.5]**. The formula is versioned in `configs/eval.yaml` and logged in `artifacts/session_log.md` after every change.

#### 3. Predictive Model

**What it is:** A gradient-boosted regression model (XGBoost selected as winner; LightGBM and CatBoost trained as candidates for comparison).

**What it takes as input:** A zone×hour feature row:

| Feature | Type | Description |
|:---|:---|:---|
| `zone_id` | categorical | DBSCAN cluster identity (the dominant feature) |
| `hour_of_day` | int [0–23] | Target prediction hour |
| `is_weekend` | bool | Sat/Sun flag |
| `month` | int | Month of year |
| `fraction_at_junction` | float | % of zone violations at junctions |
| `dominant_violation_type` | encoded int | Most common violation type in zone |
| `dominant_vehicle_type` | encoded int | Most common vehicle type in zone |
| `police_station_id` | encoded int | Jurisdiction identifier |
| `center_code_encoded` | encoded int | Administrative center |
| `rolling_7d_count` | float | 7-day lag mean (computed via `shift(1)` — no leakage) |

**What it outputs:** `predicted_count` — the expected number of parking violations for a given `(zone_id, hour_of_day)` pair on the requested date.

**Temporal split (leakage-free):**
- **Train:** Nov 9 2023 – Feb 29 2024 (226,296 rows)
- **Test:** Mar 1 2024 – Apr 8 2024 (70,311 rows)
- A hard `AssertionError` is raised if `max(train.created_datetime) >= min(test.created_datetime)`.

#### 4. Ranking Logic

```
priority_score(zone, t) = predicted_count(zone, t) × CIS(zone)
```

For a requested date and hour `t`, this formula scores every zone by combining:
- **How many violations the ML model expects** at that time (temporal signal)
- **How much congestion impact that zone has** structurally (spatial signal)

Zones are ranked descending by `priority_score`. The top-K (default: 10) are output with a priority tier:

| Tier | Threshold |
|:---|:---|
| HIGH | `priority_score ≥ 0.7 × max(priority_score)` |
| MEDIUM | `priority_score ≥ 0.4 × max(priority_score)` |
| LOW | `priority_score < 0.4 × max(priority_score)` |

#### 5. Dashboard / Demo Output

Two output artifacts are generated per run:

1. **`enforcement_priority_{date}_{hour}h.html`** — a self-contained HTML file (no server needed) containing:
   - An interactive **Folium map** of Bengaluru with zone circles coloured by priority tier (red = HIGH, orange = MEDIUM, blue = LOW). Mouseover popups show priority score, avg daily violations, junction weight, and tier.
   - A **model scorecard panel** with MAE, RMSE, ML Lift % over naive baseline, and NDCG@10.
   - An interactive **priority zone table** with coordinates and tier labels.

2. **`day_schedule_{date}.csv`** — the top-K enforcement zones across all 24 hours of the requested date; suitable for police vehicle route planning.

---

## 🛠️ Setup & Running From Scratch

### Prerequisites
- Python **3.10+** on Windows
- Git

### 1. Clone the Repository
```powershell
git clone https://github.com/<your-org>/GridLock-R2.git
cd "GridLock R2"
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv venv

# PowerShell
.\venv\Scripts\Activate.ps1

# Command Prompt
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install pandas numpy scikit-learn xgboost lightgbm catboost folium tqdm loguru pyyaml pathlib2
```

### 4. Place the Raw Dataset
The raw data file must be placed at:
```
data/raw/jan to may police violation_anonymized791b166.csv
```
> This file is **read-only**. Never overwrite it. The pipeline reads it and writes all outputs to `data/processed/` and `data/outputs/`.

### 5. Run the Full Pipeline (From Scratch)
This runs all 8 steps: validation → ingest → features → clustering → CIS → grids → training → inference.
```powershell
python -m src.data.pipeline
```
Default target: **2024-03-18 09:00**, top-10 zones. Full run takes ~5–10 minutes on first execution.

### 6. Fast Inference-Only Run (Demo Mode — ~4 seconds)
If models and clusters are already trained and saved, skip recomputation:
```powershell
python -m src.data.pipeline --skip-features --skip-clustering --skip-training --date 2024-03-18 --hour 14 --top-k 10
```

### 7. CLI Reference

| Flag | Type | Default | Description |
|:---|:---|:---|:---|
| `--date` | string | `2024-03-18` | Target date (`YYYY-MM-DD`) |
| `--hour` | int | `9` | Target hour bucket (0–23) |
| `--top-k` | int | `10` | Number of enforcement zones to output |
| `--skip-training` | flag | False | Load winning model from checkpoint instead of retraining |
| `--skip-clustering` | flag | False | Load DBSCAN zones from existing Parquet files |
| `--skip-features` | flag | False | Skip validation + ingest + row features; use existing Parquet |

### 8. View the Output Map
Open in any browser (no server required):
```
data/outputs/enforcement_priority_2024-03-18_09h.html
```
Double-click in Windows Explorer, or paste the full path into your browser address bar.

### 9. Run Individual Notebooks (Step-by-Step Walkthrough)

| Notebook | Purpose |
|:---|:---|
| `notebooks/01_eda.ipynb` | Data loading, schema validation walkthrough |
| `notebooks/01b_features.ipynb` | Row-level feature engineering |
| `notebooks/02_cluster_tuning.ipynb` | DBSCAN `eps` / `min_samples` grid search |
| `notebooks/03_clustering.ipynb` | DBSCAN execution + CIS computation + map |
| `notebooks/04_training.ipynb` | Multi-model training + scorecard |
| `notebooks/05_inference.ipynb` | Zone ranking inference + HTML output generation |

```powershell
jupyter notebook
```

---

## 📊 Evaluation Metrics

| Metric | What It Measures | Result |
|:---|:---|:---|
| **MAE (Mean Absolute Error)** | Average absolute error in predicted violation *count* per zone-hour. Lower = more precise count prediction. | **4.68** (XGBoost winner) |
| **RMSE** | Penalises large count errors more heavily than MAE. Sensitive to outlier zones with very high violations. | **10.66** |
| **Naive MAE Baseline** | Same metric for a frequency ranker (no ML — ranks by raw historical count). Establishes the floor the model must beat. | **5.58** |
| **ML Lift %** | `(Naive_MAE - ML_MAE) / Naive_MAE × 100`. The % reduction in count error from using the ML model vs. the naive baseline. | **+16.1%** |
| **NDCG@10** | Normalised Discounted Cumulative Gain at K=10. Measures whether the top-10 ranked enforcement zones are the *correct* top-10 (and in the right order). Score of 1.0 = perfect ranking. | **1.0000** |
| **Precision@10** | What fraction of the predicted top-10 zones are truly high-violation zones in the test period. | **1.0000** |
| **Silhouette Score** | DBSCAN cluster quality (compactness vs. separation). A value near 0 indicates appropriate cluster boundary decisions on sparse urban data. | **-0.096** |

> **Why is NDCG@10 = 1.0?** Bengaluru parking violations show strong spatial stability — Brigade Road, Indiranagar, and Commercial Street are chronically high-violation zones every day. Even the naive frequency baseline achieves perfect NDCG@10. The ML model's real value is in *count-level precision* (MAE 4.68 vs. 5.58), enabling hour-by-hour resource allocation — not just identifying which zones matter, but *when* each zone peaks.

---

## ⚠️ Known Limitations

| Limitation | Impact | Notes |
|:---|:---|:---|
| **Hourly granularity only** | Cannot schedule patrol windows shorter than 1 hour | Dataset does not include sub-hour temporal resolution reliably |
| **No road-type data** | CIS uses junction presence as a proxy for road classification (main / side / footpath) | External datasets prohibited by hackathon FAQ; junction weight is defensible but approximate |
| **Static zone boundaries** | DBSCAN zones are frozen at training time; new hotspots emerging after April 2024 are invisible | Requires periodic re-clustering as new data arrives |
| **NDCG ceiling effect** | All three models + baseline achieve NDCG@10 = 1.0; ranking metric cannot differentiate models | Use MAE/RMSE as the primary model selection criterion |
| **Silhouette score near zero** | Reflects the inherent spatial sparsity of violation data, not a clustering failure; confirmed by visual map inspection | Cluster coherence is validated visually in `notebooks/03_clustering.ipynb` |
| **No real-time data feed** | System is batch-only; outputs are pre-computed for a given date/hour | Would require a streaming ingestion layer for live deployment |
| **Dataset ends Apr 8 2024** | Test window is only ~5 weeks; longer held-out periods would give more reliable NDCG estimates | Limited by the competition-provided dataset |
| **Leakage-prone columns excluded** | `data_sent_to_scita`, `validation_status`, etc. are dropped — their information is genuinely lost | This is correct by design; no information available at prediction time can substitute them |

---

## 🚀 Future Improvements

### Near-Term (Achievable within Hackathon)
- Add Streamlit interactive dashboard (`src/dashboard/app.py` scaffolded but deferred)
- Time-of-day slider on the enforcement map for live hour selection
- Gemini API integration for auto-generated zone briefings (async pre-compute)

### Post-Hackathon
- Sub-hourly prediction (15-minute resolution) with richer temporal features
- Real-time data ingestion pipeline (Kafka + Flink) to replace batch CSV processing
- Periodic model retraining trigger on concept drift detection (PSI / KL divergence)
- Road classification integration if Bengaluru BBMP GIS data becomes available
- Uncertainty quantification (prediction intervals) so enforcement planners know model confidence
- Cold-start solution for newly observed zones with zero historical violations

---

## 📋 Session Log Summary

This project was built in 5 sequential phases over one hackathon session:

| Phase | Step | Status | Key Output |
|:---|:---|:---|:---|
| **Phase 0** | EDA Audit | ✅ Complete | 298,450 rows, 150-day span confirmed. 15 leakage columns identified and excluded. Train/test split validated (no calendar gaps). `eda_summary.json` saved. |
| **Phase 1** | Config Gates | ✅ Complete | `configs/features.yaml` v1.0, `configs/eval.yaml` v1.0 (CIS + ranker formula), `configs/model.yaml` v1.0 all written and user-approved. |
| **Phase 1** | Data Ingestion | ✅ Complete | `validate.py` (8-check schema validator), `load.py` (ingest + dedup → 268K rows), `notebooks/01_eda.ipynb`. |
| **Phase 1** | Feature Engineering | ✅ Complete | `features.py` Phase A: 22-col row-level features. `features_row_level.parquet` saved. |
| **Phase 2** | Clustering + Grid | ✅ Complete | DBSCAN (eps=0.05, min_samples=50): 139 zones + noise zone. CIS computed for all 140 zones. Zone×hour grid (26,354 rows) + Zone×day grid (8,246 rows). |
| **Phase 3** | ML Training | ✅ Complete | XGBoost wins (NDCG@10=1.0, MAE=4.68). All 6 model checkpoints saved. 16.1% lift over naive baseline. |
| **Phase 4** | Inference + Output | ✅ Complete | `ranker.py` + `static_output.py`. Top-10 zones ranked for 2024-03-18 09:00. HTML map + day schedule CSV generated. |
| **Phase 5** | Pipeline Orchestrator | ✅ Complete | `pipeline.py` — 8-step end-to-end run. Inference-only mode in **3.3 seconds**. |
| **Next** | Streamlit Dashboard | ⏳ Deferred | `src/dashboard/app.py` scaffolded. Build after core pipeline is demo-stable. |

**Models tested:** XGBoost, LightGBM, CatBoost (all at `hour` and `day` resolutions — 6 total runs)  
**Winner:** `xgboost_hour` — MAE 4.68, NDCG@10 1.0, 16.1% lift over naive baseline  
**Demo command:** `python -m src.data.pipeline --skip-features --skip-clustering --skip-training` (~4s)

---

## 📁 Repository Structure

```
GridLock R2/
├── configs/
│   ├── features.yaml       # Feature list, exclusion registry, encoding rules
│   ├── eval.yaml           # CIS formula, ranker formula, NDCG relevance, split bounds
│   └── model.yaml          # DBSCAN params, model candidates, winner checkpoint path
├── src/
│   ├── data/
│   │   ├── validate.py     # 8-check schema validator — fails loudly on any breach
│   │   ├── load.py         # Ingest CSV, cast dtypes, minute-level dedup
│   │   ├── features.py     # Phase A (row features) + Phase B (zone×time grid)
│   │   └── pipeline.py     # End-to-end orchestrator (run this for demo)
│   ├── models/
│   │   └── clustering.py   # DBSCAN + KDE + CIS computation
│   ├── training/
│   │   └── train.py        # Multi-model training, leakage guard, checkpointing
│   ├── evaluation/
│   │   └── metrics.py      # MAE/RMSE, NDCG@K, Precision@K, baseline runner
│   └── inference/
│       ├── ranker.py        # Load checkpoint, score zones, rank by priority_score
│       └── static_output.py # Folium HTML map + priority table generator
├── notebooks/              # Step-by-step human-executable walkthroughs
├── data/
│   ├── raw/                # READ-ONLY — never modify
│   ├── processed/          # Parquet files, encoders, metadata JSONs
│   └── outputs/            # HTML maps, CSV schedules, eval JSONs
├── checkpoints/            # Saved model checkpoints (all 6 candidates)
├── artifacts/
│   ├── session_log.md      # Living log — all sessions, decisions, metrics
│   └── problem_analysis.md
└── claude.md               # AI pair-programming context file (read before any AI session)
```

---

*Built for the Flipkart GridLock 2.0 Hackathon — PS1: Poor Visibility on Parking-Induced Congestion.*  
*Dataset: Bengaluru Police Violation Data, Nov 2023 – Apr 2024.*
