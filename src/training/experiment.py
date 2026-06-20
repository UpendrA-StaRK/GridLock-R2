"""
src/training/experiment.py
GridLock R2 — PS1: Parking-Induced Congestion

Focused experiment runner for single-factor ablation studies.

Runs ONLY the experiments defined in configs/model.yaml under the
`xgboost_poisson` and `catboost_poisson` variant blocks — NOT the
full 6-model comparison grid (that is train.py's job).

Each experiment:
  - Inherits base model hyperparameters from its `base` block in model.yaml
  - Overrides EXACTLY the params listed in `changes`
  - Saves a dedicated checkpoint directory
  - Writes a structured results JSON to data/outputs/

Usage (from notebooks/07_experiments.ipynb):
    from src.training.experiment import run_experiments
    results = run_experiments(project_root=PROJECT_ROOT)

Rules (AGENTS.md):
  - All hyperparameters read from configs/model.yaml — never hardcoded
  - Feature list read from configs/features.yaml — never hardcoded
  - No-future-leakage assertion (inherited from train.py split logic)
  - Checkpoint every variant to checkpoints/{variant}_{resolution}_{timestamp}/
  - tqdm progress on all loops
  - loguru for all logging
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

# Re-use all shared helpers from train.py — single source of truth
from src.training.train import (
    _add_cyclical_temporal_features,
    _add_zone_aggregate_features,
    _build_catboost,
    _build_xgboost,
    _get_feature_cols,
    _get_target_col,
    _load_configs,
    _make_checkpoint_dir,
    _save_checkpoint,
    _split_data,
    _train_one,
    _yaml_hash,
)


# ── Experiment variant resolver ────────────────────────────────────────────────

def _resolve_variant_config(
    model_cfg: dict[str, Any],
    variant_key: str,
) -> tuple[str, dict[str, Any]]:
    """
    Resolve a variant config block into (model_name, merged_config).

    The variant block specifies:
      base: xgboost | catboost
      changes:
        objective: count:poisson   # or any other param override

    Returns the base model name and a config dict that is the base config
    with the `changes` overrides applied.

    Args:
        model_cfg:    Full model.yaml dict (configs["model"]).
        variant_key:  e.g. "xgboost_poisson" or "catboost_poisson".

    Returns:
        (base_model_name, merged_params_dict)
    """
    variant = model_cfg.get(variant_key)
    if variant is None:
        raise KeyError(
            f"Variant '{variant_key}' not found in model.yaml. "
            f"Add it under '# Experiment Variants' section."
        )

    base_name = variant.get("base")
    if base_name not in ("xgboost", "catboost", "lightgbm"):
        raise ValueError(
            f"Variant '{variant_key}' has unsupported base='{base_name}'. "
            f"Expected 'xgboost', 'catboost', or 'lightgbm'."
        )

    # Deep copy base config and apply overrides
    base_cfg = copy.deepcopy(model_cfg.get(base_name, {}))
    changes = variant.get("changes", {})
    base_cfg.update(changes)

    logger.info(
        f"Variant '{variant_key}': base='{base_name}' | "
        f"overrides={changes}"
    )
    return base_name, base_cfg


def _build_variant_model(
    base_name: str,
    merged_cfg: dict[str, Any],
    full_model_cfg: dict[str, Any],
    seed: int,
) -> Any:
    """
    Build a model instance from a merged (base + override) config dict.

    We temporarily patch full_model_cfg[base_name] with merged_cfg,
    call the standard builder, then restore the original. This ensures
    the builder functions receive a correctly structured dict.

    Args:
        base_name:     "xgboost" | "catboost" | "lightgbm"
        merged_cfg:    Base config + variant overrides merged dict.
        full_model_cfg: The full model.yaml dict (needed by builders).
        seed:          Random seed.

    Returns:
        Fitted model instance (before training).
    """
    # Patch the base block temporarily
    original = full_model_cfg.get(base_name, {})
    full_model_cfg[base_name] = merged_cfg

    try:
        if base_name == "xgboost":
            model = _build_xgboost(full_model_cfg, seed)
        elif base_name == "catboost":
            model = _build_catboost(full_model_cfg, seed)
        else:
            from src.training.train import _build_lightgbm
            model = _build_lightgbm(full_model_cfg, seed)
    finally:
        # Restore original config (do not mutate caller's state)
        full_model_cfg[base_name] = original

    return model


# ── Main experiment entry point ────────────────────────────────────────────────

# Variants to run — these must exist in model.yaml experiment section
EXPERIMENT_VARIANTS = ["xgboost_poisson", "catboost_poisson"]


def run_experiments(
    project_root: str | Path = ".",
    variants: list[str] | None = None,
    time_resolution: str = "hour",
) -> dict[str, Any]:
    """
    Run focused single-factor ablation experiments defined in model.yaml.

    For each variant in `variants`:
      1. Resolve base model + override params from model.yaml
      2. Load the same zone_hour_grid.parquet used in standard training
      3. Apply the same time-based split + leakage guard
      4. Apply the same zone aggregate features + cyclical temporal features
      5. Train and evaluate
      6. Save checkpoint to checkpoints/{variant}_{resolution}_{timestamp}/
      7. Compare against baseline results from the most recent eval JSON

    Args:
        project_root:    Path to the project root.
        variants:        List of variant keys from model.yaml to run.
                         Defaults to EXPERIMENT_VARIANTS.
        time_resolution: "hour" or "day". Defaults to "hour" (winner resolution).

    Returns:
        Dict keyed by variant name with full eval results + comparison vs baseline.
    """
    from src.evaluation.metrics import full_eval, save_eval_results

    project_root = Path(project_root)
    if variants is None:
        variants = EXPERIMENT_VARIANTS

    configs = _load_configs(project_root)
    model_cfg    = configs["model"]
    eval_cfg     = configs["eval"]
    features_cfg = configs["features"]

    seed = int(model_cfg.get("seed", 42))
    np.random.seed(seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(
        f"\n{'='*60}\n"
        f"  EXPERIMENT RUN | timestamp={timestamp}\n"
        f"  Variants: {variants}\n"
        f"  Resolution: {time_resolution}\n"
        f"{'='*60}"
    )

    # Load shared data once
    cis_path = project_root / "data" / "processed" / "cis_table.parquet"
    if not cis_path.exists():
        raise FileNotFoundError(f"cis_table.parquet not found at '{cis_path}'")
    cis_df = pd.read_parquet(cis_path)
    logger.info(f"CIS table loaded: {len(cis_df)} zones")

    target_col   = _get_target_col(time_resolution)
    feature_cols = _get_feature_cols(features_cfg, time_resolution)

    grid_file = f"zone_{time_resolution}_grid.parquet"
    grid_path = project_root / "data" / "processed" / grid_file
    if not grid_path.exists():
        raise FileNotFoundError(f"'{grid_path}' not found. Run 03_clustering.ipynb first.")
    df = pd.read_parquet(grid_path)
    logger.info(f"Grid loaded: {len(df):,} rows from '{grid_file}'")

    # Time-based split (same as train.py — reuse exact same split)
    train_df, test_df = _split_data(df, eval_cfg)

    # Zone aggregate features (train-only, same as train.py)
    train_df, test_df = _add_zone_aggregate_features(
        train_df, test_df, target_col, cis_df
    )

    # Cyclical temporal features (same as train.py)
    train_df = _add_cyclical_temporal_features(train_df)
    test_df  = _add_cyclical_temporal_features(test_df)

    # Build feature matrix
    available_cols = [c for c in feature_cols if c in train_df.columns]
    missing_cols   = [c for c in feature_cols if c not in train_df.columns]
    if missing_cols:
        logger.warning(f"Feature columns not found in {grid_file}: {missing_cols} — skipping.")

    X_train = train_df[available_cols].fillna(-1)
    y_train = train_df[target_col].astype(float)
    X_val   = test_df[available_cols].fillna(-1)
    y_val   = test_df[target_col].astype(float)

    logger.info(
        f"X_train: {X_train.shape}  y_train mean={y_train.mean():.2f} max={y_train.max():.0f}\n"
        f"X_val:   {X_val.shape}    y_val mean={y_val.mean():.2f} max={y_val.max():.0f}"
    )

    # ── Run each variant ──────────────────────────────────────────────────────
    experiment_results: dict[str, Any] = {}

    for variant_key in tqdm(variants, desc="Experiment variants", unit="variant"):
        logger.info(f"\n{'─'*60}\n  Experiment: {variant_key}\n{'─'*60}")

        # Resolve variant config
        base_name, merged_cfg = _resolve_variant_config(model_cfg, variant_key)

        # Build model with overridden params
        model = _build_variant_model(base_name, merged_cfg, model_cfg, seed)

        # Train
        with tqdm(
            total=1,
            desc=f"Fitting {variant_key}",
            unit="model",
            leave=False,
        ) as pbar:
            model, eval_history = _train_one(
                base_name, model, X_train, y_train, X_val, y_val
            )
            pbar.update(1)

        # Predict (clip negatives — Poisson objectives can produce tiny negatives at boundary)
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, None)

        # Full evaluation
        eval_result = full_eval(
            model_name      = variant_key,   # use variant key so it's distinct in the JSON
            time_resolution = time_resolution,
            y_true          = y_val,
            y_pred          = y_pred,
            test_df         = test_df,
            train_df        = train_df,
            cis_df          = cis_df,
            eval_config     = eval_cfg,
            eval_history    = eval_history,
        )

        # Save checkpoint (variant name used as model_name for distinct directory)
        ckpt_dir = _make_checkpoint_dir(
            project_root, variant_key, time_resolution, timestamp
        )
        eval_result["time_resolution"] = time_resolution
        eval_result["variant_config"] = {
            "base": base_name,
            "overrides": model_cfg.get(variant_key, {}).get("changes", {}),
        }

        # Zone stats for ranker compatibility
        _zone_stats = (
            train_df.groupby("zone_id", observed=True)
            .agg(
                zone_mean_count   =(target_col, "mean"),
                zone_median_count =(target_col, "median"),
                zone_total_count  =(target_col, "sum"),
                zone_junction_frac=("fraction_at_junction", "mean"),
            )
            .reset_index()
            .merge(
                cis_df[["zone_id", "cis_score"]].rename(
                    columns={"cis_score": "zone_cis_score"}
                ),
                on="zone_id",
                how="left",
            )
        )
        _zone_stats["zone_cis_score"] = _zone_stats["zone_cis_score"].fillna(0.0)

        _save_checkpoint(
            ckpt_dir, model, base_name, eval_result, configs, project_root, _zone_stats
        )

        experiment_results[variant_key] = eval_result
        logger.info(
            f"  [{variant_key}] DONE | "
            f"MAE={eval_result['regression']['mae']:.4f} | "
            f"RMSE={eval_result['regression']['rmse']:.4f} | "
            f"Per-hour NDCG={eval_result['ranking_per_hour']['model_ndcg']['mean_ndcg']:.4f}"
        )

    # ── Load baseline results for comparison ──────────────────────────────────
    baseline = _load_baseline_results(project_root, time_resolution)
    experiment_results["baseline_comparison"] = _build_comparison_table(
        experiment_results, baseline, time_resolution
    )
    experiment_results["experiment_timestamp"] = timestamp
    experiment_results["variants_run"] = variants

    # ── Save experiment results ───────────────────────────────────────────────
    out_path = project_root / "data" / "outputs" / f"experiment_{timestamp}.json"
    _save_experiment_results(experiment_results, out_path)

    # ── Print scorecard ───────────────────────────────────────────────────────
    _print_experiment_scorecard(experiment_results, baseline)

    logger.info(f"\nExperiment results saved → '{out_path}'")
    return experiment_results


# ── Comparison helpers ─────────────────────────────────────────────────────────

def _load_baseline_results(
    project_root: Path,
    time_resolution: str,
) -> dict[str, Any]:
    """
    Load the most recent standard eval JSON and extract baseline model results.

    Returns a dict keyed by model name with their metrics, so we can compare
    experiment variants against the standard 6-model run.
    """
    outputs_dir = project_root / "data" / "outputs"
    eval_files = sorted(outputs_dir.glob("eval_*.json"))

    if not eval_files:
        logger.warning("No baseline eval JSON found — comparison will be empty.")
        return {}

    latest_eval = eval_files[-1]
    logger.info(f"Baseline results loaded from: '{latest_eval.name}'")

    with latest_eval.open("r", encoding="utf-8") as f:
        d = json.load(f)

    baseline: dict[str, Any] = {}
    for key in [f"xgboost_{time_resolution}", f"lightgbm_{time_resolution}",
                f"catboost_{time_resolution}"]:
        if key in d:
            baseline[key] = d[key]

    return baseline


def _build_comparison_table(
    experiment_results: dict[str, Any],
    baseline: dict[str, Any],
    time_resolution: str,
) -> list[dict[str, Any]]:
    """Build a flat comparison table across all variants + baseline models."""
    rows: list[dict[str, Any]] = []

    def _extract_row(name: str, result: dict[str, Any]) -> dict[str, Any]:
        reg  = result.get("regression", {})
        rph  = result.get("ranking_per_hour", {})
        ndcg = rph.get("model_ndcg", {})
        spear = rph.get("model_spearman", {})
        pai  = result.get("spatial_pai", {})
        var  = result.get("variant_config", {})
        return {
            "name":            name,
            "is_experiment":   bool(var),
            "base_model":      var.get("base", name.replace(f"_{time_resolution}", "")),
            "overrides":       var.get("overrides", {}),
            "mae":             round(reg.get("mae", float("inf")), 4),
            "rmse":            round(reg.get("rmse", float("inf")), 4),
            "per_hour_ndcg":   round(ndcg.get("mean_ndcg", 0.0), 4),
            "per_hour_spear":  round(spear.get("mean_spearman", 0.0), 4),
            "pai":             round(pai.get("pai", 0.0), 2),
            "beats_baseline_per_hour": rph.get("beats_baseline_per_hour_ndcg", False),
        }

    # Experiment variants
    for k, v in experiment_results.items():
        if k in ("baseline_comparison", "experiment_timestamp", "variants_run"):
            continue
        if isinstance(v, dict) and "regression" in v:
            rows.append(_extract_row(k, v))

    # Baseline models
    for k, v in baseline.items():
        if isinstance(v, dict) and "regression" in v:
            rows.append(_extract_row(k, v))

    # Sort by per_hour_ndcg desc, then mae asc
    rows.sort(key=lambda x: (-x["per_hour_ndcg"], x["mae"]))
    return rows


def _print_experiment_scorecard(
    experiment_results: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    """Print a formatted comparison table to the terminal."""
    table = experiment_results.get("baseline_comparison", [])
    if not table:
        return

    logger.info("\n" + "=" * 80)
    logger.info("  EXPERIMENT SCORECARD")
    logger.info("=" * 80)
    logger.info(
        f"  {'Model':<30} {'MAE':>7} {'RMSE':>8} "
        f"{'NDCG/hr':>9} {'Spear':>7} {'PAI':>6} {'Beats BL':>9}"
    )
    logger.info("  " + "-" * 78)

    for row in table:
        tag = " [EXP]" if row["is_experiment"] else "      "
        beat = "✓" if row["beats_baseline_per_hour"] else "✗"
        logger.info(
            f"  {row['name']:<30}{tag} "
            f"{row['mae']:>7.4f} {row['rmse']:>8.4f} "
            f"{row['per_hour_ndcg']:>9.4f} {row['per_hour_spear']:>7.4f} "
            f"{row['pai']:>6.2f} {beat:>9}"
        )

    logger.info("=" * 80)

    # Determine winner from experiment variants only
    exp_rows = [r for r in table if r["is_experiment"]]
    if exp_rows:
        best = exp_rows[0]
        logger.info(
            f"\n  Best experiment variant: {best['name']}\n"
            f"    MAE={best['mae']}  RMSE={best['rmse']}  "
            f"Per-hour NDCG={best['per_hour_ndcg']}"
        )


def _save_experiment_results(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Save experiment results dict to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Experiment results saved → '{output_path}'")
