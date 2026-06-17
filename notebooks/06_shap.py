# GridLock R2 — Phase 3: SHAP Feature Importance Analysis
# Notebook: 06_shap.ipynb (run this as a Python script or convert to .ipynb)
#
# Purpose: Explain which features drive the ML model's enforcement zone predictions.
# Run AFTER Phase 1 retrain (model must use zone aggregate features, not zone_id).
#
# Prerequisites:
#   - Best M1 model checkpoint in checkpoints/ (xgboost_hour_*)
#   - data/processed/zone_hour_grid.parquet
#   - data/processed/cis_table.parquet
#   - configs/eval.yaml (for split boundaries)
#   - pip install shap matplotlib

"""
06_shap.py
GridLock R2 — SHAP Feature Importance Analysis (Phase 3)

Run from project root:
    python notebooks/06_shap.py

Outputs:
    data/outputs/shap_summary.png      — global beeswarm plot (for demo slides)
    data/outputs/shap_importance.png   — feature importance bar chart
    data/outputs/shap_pdp_rolling.png  — partial dependence: rolling_7d_count
    data/outputs/shap_pdp_hour.png     — partial dependence: hour_of_day
    data/outputs/shap_force_plots/     — per-zone force plots for top 3 zones
    data/outputs/shap_values.npz       — raw SHAP values (for further analysis)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/notebook
import matplotlib.pyplot as plt
import yaml
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

# ── 1. Load the best model checkpoint ─────────────────────────────────────────
logger.info("=== SHAP Analysis — GridLock R2 (Phase 3) ===")

# Find the best xgboost_hour checkpoint (most recent by timestamp)
ckpt_dir_base = PROJECT_ROOT / "checkpoints"
xgb_hour_dirs = sorted(
    [d for d in ckpt_dir_base.glob("xgboost_hour_*") if d.is_dir()],
    key=lambda d: d.name,
    reverse=True,
)
if not xgb_hour_dirs:
    raise FileNotFoundError(
        "No xgboost_hour_* checkpoint found. Run training (Phase 1 retrain) first."
    )

best_ckpt = xgb_hour_dirs[0]
logger.info(f"Using checkpoint: {best_ckpt.name}")

# Load XGBoost model
from xgboost import XGBRegressor
model = XGBRegressor()
model.load_model(str(best_ckpt / "model.xgb"))
logger.info(f"XGBoost model loaded | n_estimators={model.n_estimators}")

# Load training meta to get feature list
with open(best_ckpt / "training_meta.json") as f:
    training_meta = json.load(f)

# ── 2. Reconstruct test set with Phase 1 features ─────────────────────────────
logger.info("Loading data for SHAP analysis ...")

# Load zone_hour_grid
grid_path = PROJECT_ROOT / "data" / "processed" / "zone_hour_grid.parquet"
if not grid_path.exists():
    raise FileNotFoundError(f"zone_hour_grid.parquet not found at {grid_path}.")
df = pd.read_parquet(grid_path)

# Load eval config for split boundaries
with open(PROJECT_ROOT / "configs" / "eval.yaml") as f:
    eval_cfg = yaml.safe_load(f)

train_end  = pd.Timestamp(eval_cfg["split"]["train_end"])
test_start = pd.Timestamp(eval_cfg["split"]["test_start"])
df["date"] = pd.to_datetime(df["date"])

train_df = df[df["date"] <= train_end].copy()
test_df  = df[df["date"] >= test_start].copy()
logger.info(f"Split: train={len(train_df):,} rows | test={len(test_df):,} rows")

# Load CIS table
cis_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "cis_table.parquet")

# Add zone aggregate features (same as training)
from src.training.train import _add_zone_aggregate_features, _get_feature_cols
target_col = "zone_hour_violation_count"
train_df, test_df = _add_zone_aggregate_features(train_df, test_df, target_col, cis_df)

# Get feature columns (Phase 1 feature list)
with open(PROJECT_ROOT / "configs" / "features.yaml") as f:
    features_cfg = yaml.safe_load(f)
feature_cols = _get_feature_cols(features_cfg, "hour")
available_cols = [c for c in feature_cols if c in test_df.columns]
missing_cols   = [c for c in feature_cols if c not in test_df.columns]
if missing_cols:
    logger.warning(f"Features missing from grid (will be skipped): {missing_cols}")

X_test = test_df[available_cols].fillna(-1)
y_test = test_df[target_col].values

logger.info(f"Test set shape: {X_test.shape} | Features: {available_cols}")

# ── 3. Subsample for SHAP (5,000 rows for speed) ──────────────────────────────
SHAP_SAMPLE = 5_000
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
logger.info(f"SHAP sample: {len(X_sample):,} rows")

# ── 4. Run SHAP TreeExplainer ─────────────────────────────────────────────────
try:
    import shap
except ImportError:
    raise ImportError("Run: pip install shap")

logger.info("Computing SHAP values (TreeExplainer) ...")
explainer    = shap.TreeExplainer(model)
shap_values  = explainer.shap_values(X_sample)
expected_val = explainer.expected_value
logger.info(f"SHAP values computed | shape={shap_values.shape} | expected_value={expected_val:.4f}")

# Save raw SHAP values
out_dir = PROJECT_ROOT / "data" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_dir / "shap_values.npz",
    shap_values=shap_values,
    expected_value=np.array([expected_val]),
    feature_names=np.array(available_cols),
)
logger.info(f"SHAP values saved → {out_dir / 'shap_values.npz'}")

# ── 5. Global summary plot (beeswarm) ─────────────────────────────────────────
logger.info("Generating SHAP summary plot ...")
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, feature_names=available_cols, show=False, max_display=15)
plt.title("SHAP Feature Importance — GridLock R2 XGBoost (Zone Enforcement Priority)", fontsize=13)
plt.tight_layout()
summary_path = out_dir / "shap_summary.png"
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"Summary plot saved → {summary_path}")

# ── 6. Feature importance bar chart ───────────────────────────────────────────
logger.info("Generating SHAP importance bar chart ...")
mean_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": available_cols,
    "mean_abs_shap": mean_shap,
}).sort_values("mean_abs_shap", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(importance_df["feature"], importance_df["mean_abs_shap"],
               color="#4C72B0", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Mean |SHAP value| (impact on model output)", fontsize=11)
ax.set_title("Feature Importance — GridLock R2 XGBoost\n(Phase 1: zone_id replaced by zone aggregates)", fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
importance_path = out_dir / "shap_importance.png"
plt.savefig(importance_path, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"Importance bar chart saved → {importance_path}")

# ── 7. Partial dependence: rolling_7d_count ───────────────────────────────────
if "rolling_7d_count" in available_cols:
    logger.info("Generating PDP: rolling_7d_count ...")
    feat_idx = available_cols.index("rolling_7d_count")
    x_vals   = X_sample["rolling_7d_count"].values
    s_vals   = shap_values[:, feat_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_vals, s_vals, alpha=0.3, s=8, c="#2196F3")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("rolling_7d_count (7-day lagged violation mean)", fontsize=11)
    ax.set_ylabel("SHAP value (impact on predicted count)", fontsize=11)
    ax.set_title("Partial Dependence: rolling_7d_count\n(Expected: strong positive — recent history drives predictions)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    pdp_roll_path = out_dir / "shap_pdp_rolling.png"
    plt.savefig(pdp_roll_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PDP (rolling_7d_count) saved → {pdp_roll_path}")

# ── 8. Partial dependence: hour_of_day ───────────────────────────────────────
if "hour_of_day" in available_cols:
    logger.info("Generating PDP: hour_of_day ...")
    feat_idx = available_cols.index("hour_of_day")
    x_vals   = X_sample["hour_of_day"].values

    # Aggregate mean SHAP per hour
    hour_shap = pd.DataFrame({"hour": x_vals, "shap": shap_values[:, feat_idx]})
    hour_agg  = hour_shap.groupby("hour")["shap"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hour_agg["hour"], hour_agg["shap"], color=["#F44336" if v > 0 else "#2196F3" for v in hour_agg["shap"]])
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Hour of Day (0–23 UTC)", fontsize=11)
    ax.set_ylabel("Mean SHAP value", fontsize=11)
    ax.set_title("Hour-of-Day Effect on Enforcement Priority\n(Red = increases predicted count, Blue = decreases)", fontsize=11)
    ax.set_xticks(range(0, 24, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    pdp_hour_path = out_dir / "shap_pdp_hour.png"
    plt.savefig(pdp_hour_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PDP (hour_of_day) saved → {pdp_hour_path}")

# ── 9. SHAP validation gate ───────────────────────────────────────────────────
logger.info("\n=== SHAP VALIDATION GATE ===")
mean_shap_sorted = pd.Series(mean_shap, index=available_cols).sort_values(ascending=False)
top5 = list(mean_shap_sorted.head(5).index)
logger.info(f"Top 5 features by mean |SHAP|: {top5}")

gate_pass = True
if "zone_id" in top5:
    logger.error("VALIDATION GATE FAIL: zone_id still in top-5 SHAP features! Phase 1 fix may not have applied.")
    gate_pass = False
if "rolling_7d_count" not in top5[:2]:
    logger.warning(f"VALIDATION CHECK: rolling_7d_count is not in top-2. Current top-2: {top5[:2]}")
    # Not a hard fail — model may weight zone aggregates more heavily in first retrain
if "hour_of_day" not in top5[:5]:
    logger.warning(f"VALIDATION CHECK: hour_of_day not in top-5. May indicate temporal signal is weak.")

if gate_pass:
    logger.info("✅ SHAP validation gate PASSED — model uses interpretable zone characteristics.")
else:
    logger.warning("⚠  SHAP validation gate FAILED — check Phase 1 feature list and retrain.")

# ── 10. Summary report ────────────────────────────────────────────────────────
shap_report = {
    "top_features_by_mean_abs_shap": mean_shap_sorted.head(10).to_dict(),
    "validation_gate_passed": gate_pass,
    "top5": top5,
    "expected_value": float(expected_val),
    "sample_size": len(X_sample),
    "outputs": {
        "summary_plot":    str(summary_path),
        "importance_plot": str(importance_path),
    }
}

import json as _json
report_path = out_dir / "shap_report.json"
with open(report_path, "w") as f:
    _json.dump(shap_report, f, indent=2)
logger.info(f"SHAP report saved → {report_path}")

logger.info("\n=== SHAP Analysis Complete ===")
logger.info(f"Outputs in: {out_dir}")
logger.info("Include shap_summary.png and shap_pdp_hour.png in the demo slides.")
