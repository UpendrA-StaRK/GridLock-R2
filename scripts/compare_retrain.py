"""
Print comparison: BEFORE vs AFTER retraining with cyclical features.
Run this AFTER 04_training.ipynb has completed.
"""
import json
import glob

PRIMARY = "xgboost_hour"

def load_best(eval_file: str) -> dict:
    with open(eval_file) as f:
        data = json.load(f)
    model_results = {k: v for k, v in data.items() if isinstance(v, dict) and "regression" in v}
    if PRIMARY in model_results:
        return model_results[PRIMARY]
    _, best = max(
        model_results.items(),
        key=lambda kv: kv[1].get("ranking_per_hour", {}).get("model_ndcg", {}).get("mean_ndcg", 0),
    )
    return best

eval_files = sorted(glob.glob("data/outputs/eval_*.json"), reverse=True)

if len(eval_files) < 2:
    print(f"Only {len(eval_files)} eval file(s) found.")
    print("BEFORE eval: eval_20260618_074728.json  (the old one)")
    print("AFTER  eval: run 04_training.ipynb first, then re-run this script.")
    raise SystemExit(0)

after_file  = eval_files[0]   # newest
before_file = eval_files[1]   # second newest (or old one)

print(f"BEFORE: {before_file}")
print(f"AFTER : {after_file}\n")

before = load_best(before_file)
after  = load_best(after_file)

def rph(m):
    return m.get("ranking_per_hour", {})

b_ndcg  = rph(before).get("model_ndcg", {}).get("mean_ndcg", 0)
a_ndcg  = rph(after).get("model_ndcg", {}).get("mean_ndcg", 0)
b_mae   = before["regression"]["mae"]
a_mae   = after["regression"]["mae"]
b_spr   = rph(before).get("model_spearman", {}).get("mean_spearman", 0)
a_spr   = rph(after).get("model_spearman", {}).get("mean_spearman", 0)

ndcg_delta = a_ndcg - b_ndcg
mae_delta  = a_mae  - b_mae
spr_delta  = a_spr  - b_spr

def arrow(delta, lower_is_better=False):
    better = delta < 0 if lower_is_better else delta > 0
    return ("✅ +" if better else "❌ ") + f"{delta:+.4f}"

print("=" * 60)
print("  BEFORE vs AFTER — Cyclical Encoding Retrain")
print("=" * 60)
print(f"{'Metric':<22} {'BEFORE':>10} {'AFTER':>10}  {'Delta':>15}")
print("-" * 60)
print(f"{'MAE (lower=better)':<22} {b_mae:>10.4f} {a_mae:>10.4f}  {arrow(mae_delta, lower_is_better=True)}")
print(f"{'Per-hr NDCG@10':<22} {b_ndcg:>10.4f} {a_ndcg:>10.4f}  {arrow(ndcg_delta)}")
print(f"{'Spearman rho':<22} {b_spr:>10.4f} {a_spr:>10.4f}  {arrow(spr_delta)}")
print("=" * 60)

ndcg_wins = a_ndcg > b_ndcg
mae_wins  = a_mae  < b_mae

if ndcg_wins and mae_wins:
    print("\n✅ KEEP new checkpoint. Cyclical encoding improved both metrics.")
    print("   Next: run 05_inference.ipynb → 06_shap.ipynb → git tag demo-ready")
elif ndcg_wins and not mae_wins:
    print("\n⚠️  NDCG improved but MAE degraded.")
    print("   Recommendation: KEEP if NDCG improvement > 0.005. Ask agent to decide.")
elif not ndcg_wins and mae_wins:
    print("\n⚠️  MAE improved but NDCG degraded.")
    print("   Recommendation: REVERT features.yaml to v2.0 (keep old checkpoint).")
else:
    print("\n❌ REVERT: Both metrics degraded with cyclical encoding.")
    print("   Action: tell agent to revert configs/features.yaml to v2.0.")
