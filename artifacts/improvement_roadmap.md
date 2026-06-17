# GridLock R2 — Improvement Roadmap

**Tied to the current architecture. Every suggestion references a specific file or component.**

---

## Part A — Temporal Resolution Improvements

### Current State
The model trains and predicts at **hourly granularity** (`zone_hour_grid` → `hour_of_day` feature).
The finest native temporal signal in the dataset is `created_datetime` at second-level precision,
but the feature engineering pipeline (`features.py` Phase B) rolls everything up to the
zone × hour bucket before training.

---

### A1 — Going from Hourly → 15-Minute Resolution

**What it would take:**

| Component | Change Required |
|:---|:---|
| `configs/features.yaml` | Add `time_resolution: 15min`. Change aggregation key from `hour_of_day` to `quarter_hour` (0–95 buckets) |
| `src/data/features.py` Phase B | Replace `df.groupby(['zone_id', 'hour_of_day'])` with `df.groupby(['zone_id', 'quarter_hour'])` where `quarter_hour = (hour * 60 + minute) // 15` |
| `src/training/train.py` | Add `quarter_hour` as a feature instead of `hour_of_day`. The 4× increase in time buckets means the zone×time grid grows from ~26K to ~104K rows. Training time increases proportionally but remains sub-minute for gradient boosting. |
| `configs/model.yaml` | Add `time_resolution: "15min"` variant. Run both and compare NDCG@10 and MAE. |
| `src/inference/ranker.py` | `rank_zones()` signature changes: accept `hour` + `minute` instead of just `hour`. Build scaffold for the 15-min bucket instead of the full hour. |

**What data is needed:**
- The existing dataset already contains `created_datetime` at second precision.
- No new data required — only a feature engineering change.
- However, the 15-minute grid will be **4× sparser** than the hourly grid. If the hourly grid is already ~94% sparse (confirmed in session log), 15-minute buckets will have ~98%+ sparsity.

**Architectural changes required:**
- The leakage guard in `train.py` (`assert max(train) < min(test)`) is time-agnostic and works unchanged.
- The `rolling_7d_count` lag feature needs to be recomputed at the 15-minute level (7d × 24h × 4 = 672 prior buckets instead of 168).
- Consider adding `time_sin` / `time_cos` cyclic encoding of `quarter_hour` to help the model learn intra-day patterns (currently not needed at hourly level because `hour_of_day` is directly encoded).

**Tradeoffs:**

| Pro | Con |
|:---|:---|
| Scheduling precision: tell officers to arrive at 08:45 not just "09:00" | Grid sparsity explodes: most zone×15min cells will have 0 violations in training |
| Better capture of rush-hour onset (7:45 vs 8:15 matters) | XGBoost handles sparsity well but MAE will increase due to fewer training samples per bucket |
| Enables tighter enforcement windows → more efficient patrol routing | Requires more data: 15-min resolution is only reliable if each zone has ≥5 violations per quarter-hour bucket on average across training days |

**Feasibility:** ✅ **Achievable within hackathon.** No new dependencies. Change `features.py` Phase B + `model.yaml` + `ranker.py` only.

---

### A2 — Going from 15-Minute → Minute-Level

**What it would take:**
- The zone×minute grid would have ~8,400 time buckets per day per zone.
- At 139 zones × 8,400 = **1.17M rows** in the grid. Still tractable for XGBoost.
- However, most zone×minute cells will have exactly 0 or 1 violations. The regression target becomes effectively a binary outcome — this signals a model architecture shift.

**What data is needed:**
- The raw data contains `created_datetime` at second precision.
- Minute-level aggregation is technically possible with the existing dataset.
- But: with 268K violations spread over 150 days × 1440 minutes × 139 zones, the average violation density per zone-minute is **0.009** — extreme sparsity.

**Architectural changes required:**
- Switch from **regression** (predict count) to **Poisson regression** or a **zero-inflated model** (Poisson/NB) to handle the near-zero integer counts properly.
- XGBoost supports `objective: count:poisson` — a direct config change in `model.yaml`.
- The NDCG@10 ranking metric remains valid; MAE as a regression metric becomes less meaningful (replace with Log-Loss or CRPS for count distributions).
- `static_output.py` and `ranker.py` need no structural changes — they just consume whatever count the model produces.

**Tradeoffs:**

| Pro | Con |
|:---|:---|
| Maximum precision for enforcement scheduling | Model likely underfits: not enough signal per minute bucket |
| Enables "arrive at 08:47" precision | NDCG@10 will degrade — ranking stability of top-10 zones decreases when counts are near-zero everywhere |
| | Sparse target distribution requires Poisson/NB loss — adds complexity |
| | No practical benefit if officers cannot act on sub-5-minute windows |

**Verdict:** ⚠️ **Not recommended for the current dataset.** Minute-level resolution exceeds what 150 days of data can reliably support. Use 15-minute granularity as the practical floor.

---

### A3 — Going to Second-Level (Real-Time)

**What it would take:**
This is no longer a modeling problem — it becomes a **streaming system** problem.

- Second-level prediction requires a **live data feed** (the existing CSV is a historical batch export).
- The current `pipeline.py` batch architecture is fundamentally incompatible with second-by-second prediction.
- The model itself cannot be meaningfully used at second-level because:
  - Feature computation (`rolling_7d_count`, zone-level aggregates) requires a lookback window.
  - Zone assignment (DBSCAN) was frozen at training time — a violation 1 second ago cannot retroactively update zone boundaries.

**Architectural requirements for second-level:**
1. **Streaming ingestion:** Kafka or Pub/Sub topic receiving violation events from police officer mobile apps in real time.
2. **Sliding window aggregation:** Flink or Spark Streaming to compute rolling 5-min / 15-min violation counts per zone in memory.
3. **Online feature store:** Redis or DynamoDB to store pre-computed zone features that can be retrieved in <10ms.
4. **Model serving:** The trained XGBoost model is loaded as a REST endpoint (FastAPI + `xgboost.Booster.predict()`). Each incoming zone-second query hits the endpoint with pre-fetched features.
5. **Re-clustering:** DBSCAN zones must be recomputed periodically (weekly or monthly) as violation geography shifts.

**Tradeoffs:**

| Pro | Con |
|:---|:---|
| True real-time enforcement routing | Requires full streaming infrastructure (Kafka, Flink, Redis) |
| Responds to sudden incident-driven parking spikes | 3–6 months of engineering work beyond the hackathon |
| Enables GPS-integrated officer dispatch | Officer reaction time is minutes, not seconds — second-level adds no practical value |

**Verdict:** ❌ **Out of scope for the hackathon.** Adds infrastructure complexity with near-zero practical benefit given officer reaction times. See Part B for the production roadmap.

---

## Part B — Architectural Improvements for Real-World Deployment

### Ranking by Impact × Effort

| # | Improvement | Impact | Effort | Timeline |
|:---|:---|:---:|:---:|:---|
| B1 | Real-time data ingestion pipeline | 🔴 Critical | High | Post-hackathon (3–6 months) |
| B2 | Automated model retraining trigger | 🔴 Critical | Medium | Post-hackathon (1–2 months) |
| B3 | Uncertainty quantification | 🟠 High | Low | ✅ Hackathon (1–2 days) |
| B4 | Concept drift detection | 🟠 High | Medium | Post-hackathon (1 month) |
| B5 | Cold-start zone handling | 🟡 Medium | Low | ✅ Hackathon (hours) |
| B6 | Scalability (100+ cities) | 🟡 Medium | High | Post-hackathon (6+ months) |
| B7 | Edge case: zone boundary drift | 🟡 Medium | Medium | Post-hackathon (2–3 months) |
| B8 | Officer feedback loop | 🟢 Nice to have | Medium | Post-hackathon |

---

### B1 — Real-Time Data Ingestion (Replaces Batch CSV)

**Current architecture:** `src/data/load.py` reads a single static CSV. The entire ingestion,
feature engineering, and inference pipeline runs as a batch job (`pipeline.py`).

**What needs to change:**

```
CURRENT:  CSV file → load.py → features.py → pipeline.py → HTML output (once per run)

PRODUCTION:
  Officer mobile app
       │  (violation recorded)
       ▼
  Kafka topic: "violation_events"
       │
       ▼
  Flink stream processor
       │  (rolling zone counts, feature computation)
       ▼
  Feature store (Redis)
       │
       ▼
  Model serving endpoint (FastAPI + XGBoost)
       │
       ▼
  Enforcement dashboard (live Folium/Plotly map, auto-refreshing)
```

**Files affected:**
- `src/data/load.py` — replace CSV reader with Kafka consumer
- `src/data/features.py` Phase B — replace batch groupby with stateful rolling window computation
- `src/inference/ranker.py` — expose as a REST endpoint instead of a CLI function
- New: `src/serving/api.py` — FastAPI app wrapping `ranker.rank_zones()`

**Key decisions:**
- Keep `pipeline.py` as the batch fallback for testing and demo.
- The model artifact (`checkpoints/`) is unchanged — only the data ingestion layer changes.

**Hackathon scope:** ❌ Not achievable. Requires Kafka, Flink, Redis setup.

---

### B2 — Automated Model Retraining Pipeline

**The problem:** The current model is trained once on Nov 2023 – Feb 2024 data.
As the city changes (new construction, events, road closures), violation patterns shift.
After 3–6 months, the frozen DBSCAN zones and the trained model will both degrade.

**What needs to change:**

```
Current: Manual → notebooks/04_training.ipynb → run by developer

Production retraining trigger:
  New month of data arrives
       │
       ▼
  Drift detection (PSI on violation_count distribution)
       │  (if drift > threshold)
       ▼
  Re-run pipeline.py (all steps, no skip flags)
       │
       ▼
  Compare new model NDCG@10 vs. current production model
       │  (if new model wins)
       ▼
  Swap checkpoint → update configs/model.yaml → redeploy serving endpoint
```

**Files affected:**
- `src/data/pipeline.py` — add `--full-retrain` flag that triggers all steps without skips
- `src/evaluation/metrics.py` — add drift detection: Population Stability Index (PSI) on zone-level violation counts
- New: `src/monitoring/drift_detector.py` — monitors incoming violation distributions vs. training baseline
- New: `configs/retraining.yaml` — defines drift threshold, retraining frequency, champion/challenger evaluation criteria

**Key decision:** Champion/Challenger pattern — new model must beat the production model on
NDCG@10 and MAE *before* replacing it. Never auto-replace without comparison.

**Hackathon scope:** ⚠️ Partial. Can add PSI drift detection to `metrics.py` within hackathon.
Full retraining automation (scheduling, champion/challenger) is post-hackathon.

---

### B3 — Uncertainty Quantification on Predictions

**The problem:** The current ranker outputs a single `predicted_count` with no confidence bound.
A traffic inspector cannot know whether "Zone 2: predicted 47 violations" is reliable or noisy.

**What needs to change:**

Replace point predictions with **prediction intervals** using one of:

1. **XGBoost Quantile Regression** (easiest — no new library):
   - Add `objective: "reg:quantile"` with `alpha: [0.1, 0.9]` in `model.yaml`
   - Train 3 models: P10 (lower bound), P50 (median), P90 (upper bound)
   - All 3 checkpoints saved to `checkpoints/`

2. **Conformal Prediction** (calibration-only, no retraining):
   - Use `MAPIE` library on the existing XGBoost predictions
   - Calibrate on a held-out calibration split (e.g., Feb 2024)
   - Produces valid coverage intervals guaranteed to contain the true count `1 - α` % of the time

**Files affected:**
- `configs/model.yaml` — add `quantile_alphas: [0.1, 0.5, 0.9]`
- `src/training/train.py` — train quantile variants alongside the point model
- `src/inference/ranker.py` — return `(predicted_count, lower_bound, upper_bound)` per zone
- `src/inference/static_output.py` — add confidence interval column to the priority table; visually show "Zone 2: 47 ± 12 violations" in the HTML popup

**Impact on demo:** A judge who asks "how reliable is this?" gets a concrete answer: "We're 80% confident violations will be between 35 and 59." This is significantly more useful than a point estimate.

**Hackathon scope:** ✅ **Achievable within hackathon (1–2 days).** Quantile regression is a
`model.yaml` config change + minor updates to `train.py` and `static_output.py`.

---

### B4 — Concept Drift Handling

**The problem:** Violation patterns shift seasonally (monsoon affects parking behaviour),
event-driven (IPL matches at Chinnaswamy Stadium spike nearby zones), and structurally
(new construction closes zones). The current model is blind to all of these.

**Current architecture gap:** `train.py` asserts temporal split correctness but does not
monitor whether the violation distribution in the test window has drifted from training.

**What needs to change:**

1. **PSI (Population Stability Index)** on `zone_hour_violation_count`:
   ```python
   # Add to src/evaluation/metrics.py
   def population_stability_index(train_counts, test_counts, bins=10):
       # PSI < 0.1 = stable, 0.1–0.25 = minor drift, >0.25 = major drift
   ```
   Compute PSI after every new month of data. If PSI > 0.25, trigger retraining.

2. **Zone-level drift flag:** Flag individual zones where test-period violation counts
   deviate > 2σ from the training distribution. Output as a warning in the ranker.

3. **Time-weighted training:** Give more weight to recent data during training
   (e.g., Nov–Jan weight = 1.0, Feb weight = 2.0, March = 3.0).
   This is a single parameter in `model.yaml` (`sample_weight_by_month: true`).
   XGBoost and LightGBM both support `sample_weight` in their fit() calls.

**Files affected:**
- `src/evaluation/metrics.py` — add `population_stability_index()`
- `src/training/train.py` — add optional `sample_weight` by recency
- `configs/model.yaml` — add `sample_weight_by_month: false` (off by default, toggle to enable)

**Hackathon scope:** ⚠️ PSI metric is achievable. Full drift-triggered retraining pipeline is post-hackathon.

---

### B5 — Cold-Start Zone Handling

**The problem:** DBSCAN zones are frozen at training time. If a new illegal parking hotspot
emerges after the training cutoff (e.g., a new commercial building opens), it will not have
a `zone_id` and the ranker will assign it to the noise zone (`zone_id = -1`) with a reduced
CIS weight of 0.5. This understates the risk of a genuinely new dense zone.

**Current handling:** Noise zone is kept but scored at 50% CIS weight (correct for sparse violations;
incorrect for a genuinely new dense cluster that simply has no historical data).

**What needs to change:**

1. **New-zone detection rule** in `clustering.py`:
   - If a geographic area accumulates > `min_samples` (50) violations within a rolling 30-day window
     and is currently mapped to `zone_id = -1`, flag it as a **candidate new zone**.
   - Output the candidate zone list as `data/outputs/emerging_zones.csv`.

2. **Incremental DBSCAN update** (approximate):
   - Monthly: re-run DBSCAN on the last 30 days of data only.
   - Compare cluster centres against existing zones using haversine distance.
   - New clusters (distance > 500m from all existing zones) = new zones.
   - Assign them temporary `zone_id = 200+` and score with a default CIS of 0.75
     (conservative — no historical data, but geographically active).

3. **Cold-start feature imputation** in `ranker.py`:
   - For zones with < 7 days of history, use the city-wide median zone-hour count as the predicted count instead of the model's prediction.
   - Flag these zones in the output with `cold_start: true`.

**Files affected:**
- `src/models/clustering.py` — add `detect_emerging_zones()` function
- `src/inference/ranker.py` — add cold-start imputation branch
- `configs/model.yaml` — add `cold_start_cis_default: 0.75`

**Hackathon scope:** ✅ **Cold-start imputation in `ranker.py` is achievable within hackathon.**
Incremental DBSCAN update is post-hackathon.

---

### B6 — Scalability to Multiple Cities

**The problem:** The entire pipeline is hardcoded to Bengaluru's bounding box and
DBSCAN parameters that were tuned on Bengaluru data. Deploying to Chennai or Hyderabad
would require a full manual re-parameterisation.

**What needs to change:**

1. **City config files** — replace the hardcoded Bengaluru bbox in `validate.py` and the
   tuned DBSCAN parameters in `model.yaml` with per-city config files:
   ```yaml
   # configs/cities/bengaluru.yaml
   bounding_box: {lat_min: 12.7, lat_max: 13.2, lon_min: 77.4, lon_max: 77.8}
   dbscan: {eps: 0.05, min_samples: 50}
   ```

2. **Auto-tuned DBSCAN** — replace fixed `eps`/`min_samples` with a grid search that runs
   automatically at pipeline startup for any new city. The grid search notebook
   (`notebooks/02_cluster_tuning.ipynb`) already does this manually.

3. **Shared model architecture, city-specific checkpoints** — train one model class
   per city; store checkpoints in `checkpoints/{city}/best_checkpoint.pkl`.

**Files affected:**
- `configs/model.yaml` — add `city: bengaluru` parameter
- `src/data/validate.py` — load bbox from city config instead of hardcoded constants
- `src/data/pipeline.py` — add `--city` CLI flag
- New: `configs/cities/` directory with per-city YAML files

**Hackathon scope:** ❌ Nice-to-have. Not required for PS1 demo.

---

### B7 — Zone Boundary Drift (DBSCAN Staleness)

**The problem:** DBSCAN zones are computed once and frozen as Parquet files.
Real-world parking violation geography shifts as roads change. A zone defined
in Nov 2023 may no longer be the actual hotspot boundary in Jun 2025.

**What needs to change:**
- Monthly re-clustering run using the last 90 days of data.
- Compare new cluster centres to existing zones (haversine distance).
- Zones that have shifted > 200m get their `zone_id` remapped.
- Historical checkpoints are preserved — models can be retrained on the updated zone map.

**Cadence:** Monthly (triggered by `--full-retrain` flag in `pipeline.py`).

**Hackathon scope:** ⚠️ Post-hackathon. The existing frozen zones are valid for the demo period.

---

### B8 — Officer Feedback Loop

**The problem:** The ranker outputs top-10 zones but has no signal on whether enforcement
actually happened and whether it reduced violations. There is no feedback from field officers
back to the model.

**What needs to change:**
- Police officers mark zones as "enforced" or "not enforced" via a mobile form.
- Enforcement events are logged with timestamp, zone_id, and outcome (violations dispersed? Y/N).
- This data feeds back into the training pipeline as a new feature: `enforcement_history_7d`
  (how many times this zone was enforced in the last 7 days).
- Zones that are repeatedly enforced but still show high violations = structural problem zones
  (need physical infrastructure intervention, not just patrolling).

**Files affected:**
- New: `src/data/feedback_ingestion.py`
- `configs/features.yaml` — add `enforcement_history_7d` to feature list
- `src/data/features.py` Phase A — compute enforcement lag feature from feedback log

**Hackathon scope:** ❌ Requires mobile app + database integration. Post-hackathon.

---

## Summary Table

| Improvement | Hackathon? | Files to Change | Impact |
|:---|:---:|:---|:---|
| **A1** — 15-min resolution | ✅ Yes | `features.py`, `model.yaml`, `ranker.py` | Medium — better schedule precision |
| **A2** — Minute-level | ⚠️ Risky | `features.py`, `model.yaml` | Low — sparsity kills signal |
| **A3** — Second-level (real-time) | ❌ No | Full streaming rewrite | N/A for batch system |
| **B1** — Real-time ingestion | ❌ No | `load.py`, `features.py`, new `serving/api.py` | Critical for production |
| **B2** — Auto retraining | ⚠️ Partial | `metrics.py`, `pipeline.py`, new `monitoring/` | Critical for production |
| **B3** — Uncertainty quantification | ✅ Yes | `model.yaml`, `train.py`, `ranker.py`, `static_output.py` | High — demo credibility |
| **B4** — Concept drift detection | ⚠️ Partial | `metrics.py`, `train.py`, `model.yaml` | High for production |
| **B5** — Cold-start zones | ✅ Yes | `clustering.py`, `ranker.py`, `model.yaml` | Medium |
| **B6** — Multi-city scalability | ❌ No | `validate.py`, `model.yaml`, new `configs/cities/` | Low for hackathon |
| **B7** — Zone boundary drift | ❌ No | `clustering.py`, `pipeline.py` | Medium for production |
| **B8** — Officer feedback loop | ❌ No | New `feedback_ingestion.py`, `features.py` | High for production value |

---

*This roadmap is tied to GridLock R2's current architecture as of the Flipkart Grid 6.0 hackathon session (Jun 2026).*
*All file references are relative to the project root.*
