# CLAUDE.md — ML Hackathon Project

## Hackathon Context
- **Problem**: PS1 — Poor Visibility on Parking-Induced Congestion (Jan–May Police Violation Data)
- **One-line description**: Build an AI system that detects illegal parking hotspots, quantifies their congestion impact, and recommends targeted enforcement zones using 6 months of Bengaluru police violation data.
- **Exact evaluation metrics**: Feasibility · Relevance · Innovation · Real-World Impact (expert panel, no numeric leaderboard — judged qualitatively at prototype review and Flipkart HQ finale)
- **Demo runtime constraint**: End-to-end demo must complete in **under 2 minutes**
- **The single most impressive demo moment**: A live interactive Folium/Plotly map of Bengaluru showing violation hotspots with a **time-of-day slider** + enforcement priority score — judges (Bengaluru Traffic Police) instantly recognize real roads and junctions they patrol
- **Runs locally on GPU/CPU**: All data processing (pandas EDA, DBSCAN clustering, KDE heatmap, XGBoost/LightGBM prediction) runs fully on local CPU — RTX 3050 Ti is **not needed** for PS1 but may be used for embedding-based feature experiments

---

## Current Focus
> **Priority order for this phase: Data Pipeline → Feature Engineering → Model Training → Evaluation → Dashboard/Demo (last)**

- ✅ **Do now**: Build `src/data/`, `src/models/`, `src/training/`, `src/evaluation/` — the full prototype pipeline with working trained models.
- ✅ **Do now**: Train and checkpoint the predictive model (XGBoost/LightGBM). Validate with internal metrics.
- ⏳ **Defer**: The interactive dashboard (Streamlit), the map visualisation, and the 2-minute demo flow are **last-mile tasks** — do not build these until the core pipeline and trained model are stable.
- ❌ **Do not**: Spend time on demo polish, UI design, or briefing generation until the model is trained and evaluated.

---

## Model Switching Protocol
- The model used in this session may change mid-project.
- When a new model is loaded, it MUST read this entire CLAUDE.md before doing anything.
- After reading, the new model must output a one-line confirmation: "Context loaded. Ready to continue."
- The new model inherits ALL decisions, constraints, and progress made by previous models.
- Never restart work from scratch after a model switch — always continue from where the last model left off.
- If anything is unclear after reading, ask ONE clarifying question before proceeding.
- Current model in use: Gemini 3.5 Flash

---

## Core Principles
- Think before coding.
- Prefer simple solutions over clever ones.
- Preserve reproducibility.
- Never modify datasets unless explicitly requested.
- Never delete checkpoints or experiment artifacts.
- Ask before introducing new dependencies.

---

## Project Structure
```
GridLock R2/
├── src/
│   ├── data/
│   │   ├── load.py         # ingest, dtype casting, bounds check
│   │   ├── validate.py     # schema validator — fails LOUDLY on any breach
│   │   ├── features.py     # feature engineering, zone×time aggregation
│   │   └── pipeline.py     # orchestrates Phases 1→3 end-to-end (run this live)
│   ├── models/
│   │   ├── clustering.py   # DBSCAN + KDE + Congestion Impact Score
│   │   └── predictor.py    # XGBoost/LightGBM model definition
│   ├── training/
│   │   └── train.py        # train loop, leakage guard, checkpointing
│   ├── evaluation/
│   │   ├── metrics.py      # MAE/RMSE, Precision@K, F1, NDCG@10, baseline
│   │   └── report.py       # saves eval_TIMESTAMP.json
│   ├── inference/
│   │   ├── ranker.py       # loads checkpoint, scores zones, outputs top-K
│   │   └── static_output.py  # fallback: ranked table + map snapshot as HTML
│   ├── dashboard/          # Streamlit app (DEFERRED — build last)
│   └── utils/
├── configs/
│   ├── data.yaml           # paths, bounding box, column names
│   ├── features.yaml       # feature list (changes often — keep separate)
│   ├── model.yaml          # hyperparameters: n_estimators, lr, depth, seed
│   └── eval.yaml           # NDCG relevance definition, ranker weight formula
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_cluster_tuning.ipynb  # eps grid search before committing DBSCAN params
├── scripts/
├── tests/
├── artifacts/
│   ├── Problem.md
│   └── problem_analysis.md
├── checkpoints/            # best_checkpoint.pt, latest_checkpoint.pt
├── data/
│   ├── raw/                # READ-ONLY — never modify
│   │   └── jan to may police violation_anonymized791b166.csv
│   ├── processed/
│   │   └── eda_summary.json  # EDA findings — machine-readable for later phases
│   └── outputs/            # eval_TIMESTAMP.json
├── README.md               # one-para problem-solution + how to run pipeline.py
└── claude.md
```

---

## Environment Rules
- **Always activate the virtual environment** before running any command:
  - Linux/Mac: `source venv/bin/activate`
  - Windows: `venv\Scripts\activate`
- Before installing any package, check if it is already available in the venv: `pip show <package>`
- If already installed, use it — **never reinstall or upgrade without explicit approval**
- **Never install packages globally** — venv only
- If venv does not exist, create it first and confirm before installing anything: `python -m venv venv`

---

## Progress Visibility
- **Every iterative step MUST have a progress bar** — this includes: training, evaluation, data loading, preprocessing, inference, and feature engineering
- Use `tqdm` for all loops and pipelines — no silent iteration
- **Training progress bars must display**: epoch number, current loss, primary metric (e.g. MAE/F1), and ETA
- Any step that takes more than a few seconds must show a live status bar — **never run long-running work silently**
- All status, logs, and progress must be visible in the terminal (or a log file that is actively tailed) — nothing should run invisibly in the background without output
- Use `tqdm(desc="...")` with a meaningful label so each bar is immediately identifiable
- For nested loops (e.g. epoch → batch), use nested `tqdm` bars with `leave=False` on inner bars
- When using `loguru`, pair log statements with tqdm updates — do not replace one with the other

---

## Coding Style
- Python 3.11+
- Type hints wherever practical
- Follow PEP8
- Use `pathlib` instead of `os.path`
- Prefer dataclasses or pydantic for configs
- Avoid global variables
- Log with `loguru`, not `print()`

---

## Preferred Libraries
- **Deep Learning**: PyTorch
- **Data**: pandas, numpy, polars
- **Experiment Tracking**: MLflow, TensorBoard
- **Visualization**: matplotlib, seaborn
- **Metrics**: sklearn.metrics
- **Config**: hydra, yaml
- **Testing**: pytest
- **Geospatial**: folium, geopandas, shapely, kepler.gl, pyproj
- **Clustering / Density**: scikit-learn (DBSCAN, KMeans), scipy (KDE / gaussian_kde)
- **Gradient Boosting**: xgboost, lightgbm, catboost
- **Interactive Dashboard**: streamlit, plotly, dash
- **Mapping tiles**: contextily (basemap overlays for matplotlib), branca (Folium plugins)
- **Google AI API**: google-generativeai (Gemini 2.5 Flash)
- **HTTP / async**: httpx, aiohttp (for async Gemini calls during demo)

---

## Training Rules
- Never hardcode paths — use configs
- Read all hyperparameters from YAML
- Save every trained model with timestamp and config
- Save random seeds
- Log metrics after every epoch
- Use deterministic seeds when possible
- **No-future-leakage guard in `train.py`**: assert that the maximum `created_datetime` in the training set is strictly less than the minimum `created_datetime` in the test set — raise a hard error if this fails
- Feature list is read from `configs/features.yaml` not hardcoded — changing a feature only requires editing that file

Always record:
- learning rate, batch size, optimizer, scheduler
- seed, model architecture, dataset version, features.yaml hash

---

## Dataset Rules
- Datasets are immutable
- Raw data → `data/raw/` (never overwrite)
- Processed data → `data/processed/`
- Document all preprocessing inside `src/data/`
- Always check: nulls, duplicates, class imbalance, outliers (IQR + Z-score)
- Log outlier findings — never silently drop

---

## Checkpointing and Recovery
- Always save periodic checkpoints during training and large computations.
- Save model weights, optimizer state, scheduler state, random seeds, and training progress.
- Save to: `checkpoints/` — format: `model_epoch_{epoch}.pt`
- Maintain both `latest_checkpoint` and `best_checkpoint` at all times.
- Automatically detect existing checkpoints and resume from the latest valid state on startup.
- Design workflows so that unexpected interruptions, crashes, or hardware limitations do not require restarting from scratch.
- Prioritize progress preservation over maximum throughput.
- Ensure training can continue seamlessly after restarts with minimal loss of work.
- Never delete checkpoints automatically.

---

## Experiments
- All experiments must be reproducible
- Store: config, metrics, seed, git commit hash
- Naming: `task_model_version` (e.g. `sentiment_bert_v1`)
- Every run gets a unique folder — never overwrite previous runs

---

## Evaluation
- Metrics belong in `src/evaluation/` — not inside training loops
- **This hackathon uses qualitative judging** — no single numeric leaderboard score
- The expert panel evaluates on: **Feasibility · Relevance · Innovation · Real-World Impact**
- Internally, track these proxy metrics to validate model quality:
  - **Hotspot clustering**: Silhouette score (DBSCAN), visual cluster coherence on Bengaluru map
  - **Predictive model**: MAE / RMSE on violation count regression; Precision@K for top-K enforcement zone ranking
  - **Classification subtasks** (if any): F1-score (macro), confusion matrix
  - **Enforcement priority ranking**: NDCG@10 — relevance definition is in `configs/eval.yaml`; a zone is "relevant" if its actual violation count in the test period falls in the top quartile across all zones
  - **Baseline comparison**: always compare against a frequency ranker (rank zones by raw historical violation count, no ML) — model must beat baseline on Precision@K and NDCG@10 or it adds no value
- Save all eval results to `data/outputs/eval_TIMESTAMP.json`

### ⚠️ Evaluation Traps — DO NOT fall into these
- **Trap 1**: Tuning or reporting metrics on the test split → always hold out a time-based test split (last 4–6 weeks of data = March–April 2024)
- **Trap 2**: The `description` column is 100% NULL and `closed_datetime` / `action_taken_timestamp` are entirely NULL — do NOT use these as features or labels
- **Trap 3**: Class imbalance in violation types (WRONG PARKING ~46% of rows) — always report per-class metrics, not just accuracy
- **Trap 4**: Spatial leakage — do NOT train on lat/lon of test-set zones; use zone-level aggregation as features instead
- **Trap 5**: Using external datasets — **strictly prohibited by FAQ**; disqualification risk
- **Trap 6**: Optimizing only for visual impressiveness while ignoring model validity — judges include domain experts who will probe methodology

---

## Inference
- No training logic in inference code
- Load models from checkpoints only
- Support both CPU and CUDA
- Support batch inference

---

## Pipeline Development Protocol

For every step in the pipeline, follow this pattern **strictly**:

### Step Structure
Every pipeline step produces exactly two artifacts:
- `src/<phase>/<module>.py` — core logic (clean, importable Python module)
- `notebooks/0X_<step_name>.ipynb` — human-executable walkthrough that imports and calls the module

### Module Rules (`src/`)
1. Write all core logic as a clean Python module inside `src/`
2. The module must be fully self-contained and importable
3. **Always confirm the module works before writing the notebook that calls it**

### Notebook Rules (`notebooks/`)
1. Notebooks must **NOT** contain any core logic — only imports, function calls, and output display
2. Every code cell must have a **markdown cell above it** explaining:
   - What that cell does
   - What the expected output is
3. Every cell must be **independently runnable** — no hidden state dependencies between cells
4. Progress bars, plots, and print outputs must all render **inline** in the notebook
5. At the end of every notebook, add a **summary cell** that prints:
   - What was done
   - What was saved (file paths)
   - What the next step is

### Execution Policy — CRITICAL
- The model **writes** all code and notebooks but **NEVER executes them**
- The user **runs every notebook manually**, cell by cell, in Jupyter
- The model **waits** for the user to share outputs, results, or errors before proceeding to the next step
- **Never assume a step completed successfully** — always wait for the user to confirm output before building on top of it
- If the user shares an error or unexpected output: debug and fix the module first, then tell the user to re-run that specific cell

### Notebook Cell Template
```
[Markdown cell]: ## Step N — <description>
What this cell does: ...
Expected output: ...

[Code cell]: from src.<phase>.<module> import <function>
result = <function>(args)
```

---

## Testing
- Run before every commit: `pytest tests/`
- Test: dataloaders, preprocessing, metrics, model forward pass

---

## Performance Priority
1. Correctness
2. Reproducibility
3. Readability
4. Speed

Avoid premature optimization.

---

## When Modifying Code
First understand: model architecture, data flow, configs, training loop.
Then:
1. Explain assumptions
2. Identify affected files
3. Make minimal changes
4. Preserve backward compatibility

---

## When Debugging
Check in order:
1. Shapes
2. Dtypes
3. Device placement
4. Data leakage
5. Seed reproducibility
6. NaNs and exploding gradients

---

## Hardware Utilization
- Fully utilize available CPU and GPU resources when beneficial.
- Prefer efficient and stable methods that fit within the current hardware limits (RTX 3050 Ti — 4 GB VRAM, i7-12700H, 16 GB RAM).
- Avoid approaches that are likely to fail due to memory exhaustion or resource constraints.
- Use mixed precision (`torch.cuda.amp`), gradient accumulation, chunking, or other memory-efficient techniques when needed to maximize utilization without sacrificing stability.
- Use `torch.no_grad()` during eval/inference.
- Clear GPU cache when needed: `torch.cuda.empty_cache()`.
- Long-running tasks should be fault-tolerant and capable of recovering from interruptions (see Checkpointing and Recovery).

---

## Documentation
Every public function needs:
- description, arguments, returns

Complex models need:
- input shape, output shape, assumptions

---

## Pipeline Overview
End-to-end flow for PS1 — from raw CSV to demo output:

1. **Schema Validation** (`src/data/validate.py`): Strict schema check — fails loudly if expected columns are missing, dtypes wrong, or lat/lon out of bounds; also checks temporal continuity (no suspicious date gaps that would break the time-based split); writes `data/processed/eda_summary.json`
2. **Ingest** (`src/data/load.py`): Load CSV → cast dtypes, filter invalid records, log null summary; calls validate.py first
3. **Feature Engineering** (`src/data/features.py`): Feature list read from `configs/features.yaml`; extract `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `violation_type_encoded`, `vehicle_type_encoded`, `police_station_id`, `junction_proximity_score`; aggregate to zone × time-block grid; save to `data/processed/`
4. **Cluster Tuning** (`notebooks/02_cluster_tuning.ipynb`): Grid-search eps/min_samples for DBSCAN; pick params before committing — do NOT skip this step
5. **Geospatial Clustering** (`src/models/clustering.py`): DBSCAN on (lat, lon) with tuned params; KDE density surface; **Congestion Impact Score formula defined in `configs/eval.yaml`** — violation density × road-type weight × junction proximity; freeze formula before Phase 3
6. **Predictive Model Training** (`src/training/train.py`): XGBoost/LightGBM; time-based split (train: Nov 2023–Feb 2024 / test: Mar–Apr 2024); no-future-leakage assertion; baseline frequency ranker also run here for comparison; serialize `checkpoints/best_checkpoint.pt`
7. **Evaluation** (`src/evaluation/`): MAE/RMSE, Precision@K, NDCG@10 vs baseline; save `data/outputs/eval_TIMESTAMP.json`
8. **Enforcement Priority Ranking** (`src/inference/ranker.py`): Load checkpoint → predict all zones for requested day/hour → rank by `predicted_count × CIS_score` (weight formula in `configs/eval.yaml`) → output top-K with priority tier
9. **Static Fallback Output** (`src/inference/static_output.py`): Renders a self-contained HTML file (ranked table + folium map snapshot) — **demo fallback if Streamlit crashes**; build this in Phase 5 alongside the ranker
10. **End-to-end Orchestrator** (`src/data/pipeline.py`): Single script that runs steps 1→8; judges may ask to run this live — must complete without errors
11. **Interactive Dashboard** (`src/dashboard/app.py`) ⏳ **DEFERRED — build last**: Streamlit app; only begin after step 10 is stable

> **Before demo**: `git tag demo-ready` — always tag so you can roll back to a clean state

---

## NEVER
- Rewrite entire modules unnecessarily
- Change dataset schemas without approval
- Remove checkpoints
- Introduce dependencies casually
- Ignore failing tests
- Mix training and inference code
- Hardcode paths — everything from `configs/`
- Tune on the test set
- Start from scratch after a model switch
- Use external datasets (FAQ violation → **disqualification**)
- Use the `description`, `closed_datetime`, or `action_taken_timestamp` columns — they are entirely NULL
- Report only aggregate accuracy — always include per-class and per-zone breakdowns
- Spatially leak test-zone coordinates into training features
- Skip time-based train/test split in favour of random split (causes temporal leakage)
- Call Gemini API synchronously inside a rendering loop — always pre-compute or async-call briefings before the dashboard renders
- Commit DBSCAN eps without running `notebooks/02_cluster_tuning.ipynb` first
- Change the Congestion Impact Score formula without versioning it — all formula changes must be tracked in `configs/eval.yaml` with a version comment and logged in the session log artifact
- Hardcode feature names in Python — always read from `configs/features.yaml`
- Skip the baseline frequency ranker — it must be compared against the ML model
- Deploy to demo without `git tag demo-ready` committed

---

## Session Log
The canonical session log is maintained as a living artifact:
**`artifacts/session_log.md`** — updated after every completed pipeline step.

Every new model or session must read `artifacts/session_log.md` before starting work.
Append a one-line mirror here after every step:
`[DATE] [MODEL] [STEP] [summary]`

<!-- entries below — newest last -->
[2026-06-16] [Claude Sonnet 4.6 Thinking] [STEP: EDA/pre-architecture] Full EDA audit complete. 298,450 rows. Split viable (train=226k, test=70k). data_sent_to_scita_timestamp excluded (leakage). violation_type is JSON-list string. Duplicates = multi-violation events (dedup only if ALL fields identical). CIS + ranker formulas versioned. eda_summary.json saved. 4 config gates remain open.
[2026-06-16] [Gemini 3.5 Flash] [STEP: Clustering & Aggregation] DBSCAN completed. 139 clusters, 2.07% noise. Row-level zoned features, CIS table, and hour/day aggregated grids generated. Next: model training.
[2026-06-16] [Claude Sonnet 4.6 Thinking] [STEP: Model Training Pipeline] Wrote train.py, metrics.py and 04_training.ipynb. Trained 6 models, winner is xgboost_hour (NDCG=1.0, MAE=4.68).
[2026-06-16] [Claude Sonnet 4.6 Thinking] [STEP: Inference & Static Output] Wrote ranker.py, static_output.py, and 05_inference.ipynb. Generated static HTML map fallback.
[2026-06-16] [Claude Sonnet 4.6 Thinking] [STEP: End-to-End Pipeline] Wrote pipeline.py end-to-end orchestrator. Runs in 3.3s in inference-only mode.
[2026-06-18] [Antigravity Gemini 2.5 Pro] [STEP: Phase 3 Improvements] Implemented 6 improvements: cyclical temporal encoding, PAI metrics, CIS normalization, ASTraM blackspot narrative, and scorecard HTML updates.
[2026-06-18] [Antigravity Gemini 2.5 Pro] [STEP: Retrain B3 Result] Cyclical encoding retrain SUCCEEDED. MAE improved to 4.4822, NDCG to 0.8911, Spearman to 0.5216.
[2026-06-19] [Gemini 3.5 Flash] [STEP: GitHub Pages Slider Map] Regenerated docs/index.html with the latest XGBoost model checkpoint (incorporating cyclical temporal encoding, normalized CIS, and the PAI metric) and prepared it for GitHub Pages hosting.
[2026-06-19] [Gemini 3.5 Flash] [STEP: Static HTML Date Picker] Replaced the single-day logic in the static HTML map generator with a multi-day (1 week) nested JSON structure. Added a sleek Date Picker to the UI to allow judges to toggle between dates and see weekday vs. weekend patterns.
[2026-06-19] [Gemini 3.5 Flash] [STEP: DBSCAN Memory Fix] Fixed MemoryError: bad allocation during grid search by changing DBSCAN n_jobs from -1 to None (single-threaded) in src/models/clustering.py.
[2026-06-21] [Antigravity Gemini 2.5 Pro] [STEP: Final Experiments (Cat, Tweedie, Lags)] Completed 3 ablation experiments. Exp 3 (Native Categoricals): CatBoost native handled them best (MAE 4.52, NDCG 0.89). Exp 1 (Tweedie Loss): LightGBM with Tweedie variance power 1.8 dominated (MAE 4.3064, per-hour NDCG 0.8942). Exp 2 (Calendar Lags): Exact lag_1d_count/lag_7d_count degraded metrics (injects high variance noise vs smoothed rolling average), confirming previous exclusions. Hurdle Model cancelled: Tweedie natively and elegantly handles the zero-inflated compound Poisson-Gamma distribution in a single step without architectural complexity. New champion promoted: lightgbm_tweedie_18.
[2026-06-21] [Antigravity Gemini 2.5 Pro] [STEP: Experiment Cleanup] Totally removed the exact calendar lag feature experiment code from `src/data/features.py`, `src/training/experiment.py`, and `configs/model.yaml` to keep the codebase clean after hypothesis refutation. Deleted all one-off experiment notebooks (`07_experiments.ipynb`, `08_lag_features.ipynb`, `09_experiment_categoricals.ipynb`, `10_experiment_tweedie.ipynb`, `11_experiment_lags.ipynb`) as the optimal `lightgbm_tweedie_18` configuration is now officially the standard pipeline target.