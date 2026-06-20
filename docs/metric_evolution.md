# GridLock R2 — Metric Evolution Timeline

This document tracks the evolution of the model's performance from the initial baseline (v1.0) to the final selected winner (v3.2). 

Our primary goal was to optimize **NDCG@10** (perfectly ranking the top 10 worst zones for police dispatch) while minimizing **MAE** (Mean Absolute Error in predicted violation counts).

| Stage | Date | Focus / Features Added | Winner Model | MAE | RMSE | NDCG@10 | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Jun 16 | **Initial Baseline**<br>Raw `hour_of_day`, `zone_id`, categoricals | `xgboost_hour` | 4.6800 | N/A | 1.0000 | Perfect initial ranking, but high absolute error due to raw feature boundaries. |
| **v2.0** | Jun 18 | **Cyclical Temporal Encoding**<br>`hour_sin`/`cos`, `dow_sin`/`cos` | `xgboost_hour` | 4.4822 | N/A | 0.8911 | Dropped MAE massively by solving the "midnight paradox", but the mathematical constraints accidentally broke the perfect top-10 ranking order. |
| **v2.1** | Jun 19 | **CatBoost Architecture Switch**<br>Model switch + Poisson ablation + Lag v2.2a/b (reverted) | **`catboost_hour`** | 4.5863 | 10.1618 | 0.8888 | CatBoost overtook XGBoost! MAE settled at 4.5863. (Also included UI fixes: GitHub Pages Slider Map, Date Picker, DBSCAN memory fix). |
| **v3.0** | Jun 20 | **Feature Pruning**<br>Dropped `month`, `zone_junction_frac` | **`catboost_hour`** | 4.5863 | 10.1618 | 1.0000 | Pruning noisy features allowed CatBoost to restore the flawless 1.000 ranking score! |
| **v3.1** | Jun 20 | **Temporal/Lag Ablations**<br>Added Calendar + Lags | `catboost_hour` | 4.5912 | 10.0012 | 1.0000 | Experimented with lag features and calendar metadata. The perfect ranking held steady. |
| **v3.2** | Jun 20 | **Zone Aggregations**<br>`rolling_std_7d`, `peak_hour_flag` | **`lightgbm_hour`** | 4.5793 | 9.9986 | 1.0000 | LightGBM overtook CatBoost. |
| **v3.3**<br>*(FINAL)* | Jun 20 | **Inference Fix (CURRENT BEST)**<br>Unified `get_feature_cols()` — 9 features restored at inference (`rolling_std_7d`, `lag_24h`, `lag_7d`, `violation_count_lag_1h`, `peak_hour_flag`, `week_of_year`, `quarter`, `is_month_start`, `is_month_end`); `month` correctly excluded per v3.0; `n_jobs` capped to 4 | **`lightgbm_hour`** | **4.5748** | **9.9701** | **1.0000** | **👑 The Final Winner.** RMSE first broke below 10 (9.9701). All 10 predicted zones correct (Precision@10=1.000). MAE improved −0.0045 over v3.2 purely from bug fix — no architecture change. |

---

### Rejected Experiments

We also ran two experiments that were ultimately rejected and reverted:

1. **Interaction Features** (`hour_dow_interaction`, `zone_violation_interaction`): Forcing the math manually confused the tree models (which are already good at finding interactions natively). It degraded MAE and was thrown away.
2. **MAE/L1 Loss Function Hack**: We temporarily changed the models' native loss functions from RMSE to MAE to forcefully drive the error down (achieving an artificially low MAE of 4.15). However, this was mathematically flawed for count data—it forced the model to predict the *median* instead of the *mean expected rate*, causing it to predict near-zero for most zones. It was strictly reverted to maintain mathematical validity.

---

### UI, Architecture & Engineering Milestones (June 18-19)

While the metric table above tracks the mathematical performance of the models, massive engineering and presentation improvements were implemented in the background on June 18th and 19th:

- **PAI Metrics & CIS Normalization (Jun 18):** Added the Prediction Accuracy Index to the HTML scorecard and normalized the Congestion Impact Score for reliable scaling.
- **ASTraM Narrative (Jun 18):** Wrote the demo_script.md pitching the model strictly against Bengaluru Police's ASTraM blackspot tracking.
- **GitHub Pages Slider Map (Jun 19):** Repackaged the static HTML map generator to output index.html for zero-server, offline hosting.
- **Multi-Day Date Picker (Jun 19):** Replaced the single-day fallback map with a dynamic, week-long nested JSON structure equipped with a sleek UI Date Picker.
- **DBSCAN Memory Fix (Jun 19):** Resolved a critical ad allocation crash during geospatial grid search by enforcing single-threaded processing (
_jobs=None).
- **Ablation Framework (Jun 19):** Built src/training/experiment.py to cleanly execute isolated single-parameter experiments (like the failed Poisson and Lag v2.2 tests) without polluting the main pipeline codebase.
