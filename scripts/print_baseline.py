"""Print current best model metrics — baseline to beat after retraining."""
import json
from pathlib import Path
import glob

# Find latest eval JSON
eval_files = sorted(glob.glob("data/outputs/eval_*.json"), reverse=True)
if not eval_files:
    print("No eval JSON found. Run 04_training.ipynb first.")
    raise SystemExit(1)

eval_path = eval_files[0]
print(f"Reading: {eval_path}\n")

with open(eval_path) as f:
    data = json.load(f)

# data contains model results AND metadata keys (winner, comparison_table, etc.)
# Filter to only model result dicts (they have a 'regression' key)
best_key, best = max(
    ((k, v) for k, v in data.items() if isinstance(v, dict) and "regression" in v),
    key=lambda kv: kv[1].get("ranking_per_hour", {}).get("model_ndcg", {}).get("mean_ndcg", 0),
)

print("=" * 50)
print("  CURRENT BEST MODEL (baseline to beat)")
print("=" * 50)

# Always show the primary inference model (xgboost_hour) first
primary_key = "xgboost_hour"
model_results = {k: v for k, v in data.items() if isinstance(v, dict) and "regression" in v}

if primary_key in model_results:
    best = model_results[primary_key]
    best_key = primary_key
else:
    best_key, best = max(
        model_results.items(),
        key=lambda kv: kv[1].get("ranking_per_hour", {}).get("model_ndcg", {}).get("mean_ndcg", 0),
    )

print(f"  Key          : {best_key}")
print(f"  Model        : {best['model']} / {best['time_resolution']}")
print(f"  MAE          : {best['regression']['mae']:.4f}")
print(f"  Naive MAE    : {best['naive_baseline_reg']['mae']:.4f}")
print(f"  ML Lift      : {best['mae_lift_vs_naive_pct']:+.1f}%")
rph = best.get("ranking_per_hour", {})
print(f"  Per-hr NDCG  : {rph.get('model_ndcg',{}).get('mean_ndcg',0):.4f}")
print(f"  Baseline NDCG: {rph.get('baseline_ndcg',{}).get('mean_ndcg',0):.4f}")
print(f"  Spearman rho : {rph.get('model_spearman',{}).get('mean_spearman',0):.4f}")
print("=" * 50)
print("\nShare the above block as your BEFORE snapshot.")
