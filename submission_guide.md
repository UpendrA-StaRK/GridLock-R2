# Final Submission Packaging Guide

Based on the `solution.md` requirements for a **Source Code** attachment (e.g., zip, max 50MB), you need to package the clean repository. 

Follow these steps to create your final `Source_Code.zip` file.

### ✅ What to INCLUDE in the ZIP

Include the core pipeline, configurations, and clean documentation:
- `src/` (Entire folder — contains the data pipeline, model definitions, training loops, inference, and dashboard)
- `configs/` (Entire folder — contains `data.yaml`, `features.yaml`, `model.yaml`, `eval.yaml`)
- `docs/` (Entire folder — contains `demo_script.md`, `metric_evolution.md`, `index.html`)
- `README.md` (The final clean overview of your architecture)
- `run_project.md` (The step-by-step instructions for judges to run the project)
- `requirements.txt` (For environment setup)
- `solution.md` (If required to be bundled)

### ❌ What to EXCLUDE from the ZIP

Do **not** include development artifacts, environment folders, raw data (if it exceeds limits), or exploratory AI logs. Exclude the following before zipping:

- **Environments & Git:**
  - `venv/`
  - `.git/`, `.gitignore`, `.gitattributes`
- **AI Context & Logs:**
  - `claude.md`
  - `AGENTS.md`
  - `artifacts/session_log.md`
  - `artifacts/experiment_log.md`
  - `artifacts/Problem.md`
  - `artifacts/final_review.md`
- **Training Junk:**
  - `catboost_info/`
  - `checkpoints/` (Keep ONLY the final winning `lightgbm_tweedie_18_<timestamp>` folder. Delete all older baseline/XGBoost/CatBoost checkpoints to save massive amounts of space).
- **Exploratory Notebooks (Already Deleted):**
  - Make sure `notebooks/07_experiments.ipynb`, `08_lag_features.ipynb`, `09_experiment_categoricals.ipynb`, `10_experiment_tweedie.ipynb`, and `11_experiment_lags.ipynb` are NOT included.
  - *Note: Keep `01b_features.ipynb` since it's step 3 in `run_project.md`.*
- **Data (Optional depending on size):**
  - If `data/raw/` CSV and `data/processed/` parquets cause the zip to exceed 50MB, exclude them. Tell the judges to place the raw CSV manually as per `run_project.md`.

### Final Zip Command Example (Windows PowerShell)

If you have 7-Zip installed, or you can just right-click -> "Compress to ZIP file" after carefully selecting only the target folders.

Make sure to double check that the final ZIP is **under 50MB**!
