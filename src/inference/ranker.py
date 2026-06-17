"""
src/inference/ranker.py
GridLock R2 — PS1: Parking-Induced Congestion

Enforcement Priority Ranker.

Given a requested day and hour, loads the winning checkpoint, predicts violation
counts for every zone, multiplies by CIS, and returns a ranked top-K table.

Formula (configs/eval.yaml ranker v1.0):
    priority_score(zone, t) = predicted_count(zone, t) × CIS(zone)

Usage (from notebooks/05_inference.ipynb):
    from src.inference.ranker import load_ranker, rank_zones
    ranker = load_ranker(project_root=PROJECT_ROOT)
    top10   = rank_zones(ranker, day_of_week=0, hour=9)

Rules (from claude.md):
  - No training logic here — loads from checkpoint only
  - Always load saved label_encoders.pkl from checkpoint (never re-fit)
  - Support CPU inference only (no GPU needed for tree models)
  - feature list read from checkpoint's features.yaml copy, not from current disk
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def find_best_checkpoint(
    project_root: str | Path = ".",
    model_name: str | None = None,
    time_resolution: str | None = None,
) -> Path:
    """
    Find the best checkpoint directory to use for inference.

    Priority order:
      1. Use primary_model + primary_time_resolution from configs/model.yaml
      2. If not set, pick the most recent checkpoint dir that matches model/resolution filters

    Args:
        project_root:     Project root directory.
        model_name:       Override model name (e.g. "xgboost"). None = read from model.yaml.
        time_resolution:  Override resolution ("hour"/"day"). None = read from model.yaml.

    Returns:
        Path to the best checkpoint directory.
    """
    project_root = Path(project_root)
    model_cfg_path = project_root / "configs" / "model.yaml"

    with model_cfg_path.open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # Determine target model + resolution
    if model_name is None:
        model_name = model_cfg.get("primary_model")
    if time_resolution is None:
        time_resolution = model_cfg.get("primary_time_resolution")

    ckpt_root = project_root / "checkpoints"
    if not ckpt_root.exists():
        raise FileNotFoundError(
            f"Checkpoints directory not found at '{ckpt_root}'. "
            "Run notebooks/04_training.ipynb first."
        )

    # Filter candidate dirs
    prefix = f"{model_name}_{time_resolution}_" if (model_name and time_resolution) else ""
    candidates = sorted(
        [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        reverse=True,  # newest first (timestamp suffix)
    )

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found matching '{prefix}*' in '{ckpt_root}'. "
            "Run notebooks/04_training.ipynb first."
        )

    chosen = candidates[0]
    logger.info(f"Using checkpoint: '{chosen.name}'")
    return chosen


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model(ckpt_dir: Path, model_name: str) -> Any:
    """Load model weights from checkpoint directory."""
    if model_name == "xgboost":
        from xgboost import XGBRegressor
        m = XGBRegressor()
        m.load_model(str(ckpt_dir / "model.xgb"))
        return m
    elif model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(ckpt_dir / "model.lgb"))
    elif model_name == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(str(ckpt_dir / "model.cbm"))
        return m
    else:
        import pickle
        with open(ckpt_dir / "model.pkl", "rb") as f:
            return pickle.load(f)


def load_ranker(
    project_root: str | Path = ".",
    model_name: str | None = None,
    time_resolution: str | None = None,
    ckpt_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load the trained ranker from the best (or specified) checkpoint.

    Returns a ranker dict containing:
        model:            Loaded model object (XGBRegressor / lgb.Booster / CatBoostRegressor)
        model_name:       e.g. "xgboost"
        time_resolution:  "hour" or "day"
        cis_df:           CIS table DataFrame
        zone_hour_df:     Aggregated zone-hour grid (for feature scaffolding)
        feature_cols:     Ordered list of feature column names
        ckpt_dir:         Path to the checkpoint directory
        meta:             training_meta.json contents

    Args:
        project_root:     Project root directory.
        model_name:       Override model name. None = read from model.yaml.
        time_resolution:  Override resolution. None = read from model.yaml.
        ckpt_dir:         Explicit checkpoint dir path (overrides auto-discovery).

    Returns:
        ranker dict — pass this to rank_zones().
    """
    project_root = Path(project_root)

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
    else:
        ckpt_dir = find_best_checkpoint(project_root, model_name, time_resolution)

    # Load training metadata
    meta_path = ckpt_dir / "training_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    model_name_eff       = meta["model_name"]
    time_resolution_eff  = meta["time_resolution"]

    logger.info(
        f"Loading ranker: model={model_name_eff} | resolution={time_resolution_eff} | "
        f"checkpoint={ckpt_dir.name}"
    )

    with tqdm(total=5, desc="Loading ranker", unit="step", leave=True) as pbar:

        # Load model
        pbar.set_description("Loading model weights")
        model = _load_model(ckpt_dir, model_name_eff)
        pbar.update(1)

        # Load CIS table (from processed data, not checkpoint)
        pbar.set_description("Loading CIS table")
        cis_path = project_root / "data" / "processed" / "cis_table.parquet"
        cis_df = pd.read_parquet(cis_path)
        pbar.update(1)

        # Load zone-hour grid for feature scaffolding
        pbar.set_description("Loading zone grid")
        grid_file = f"zone_{time_resolution_eff}_grid.parquet"
        grid_path = project_root / "data" / "processed" / grid_file
        zone_grid_df = pd.read_parquet(grid_path)
        pbar.update(1)

        # Load zone aggregate stats lookup (saved by train.py _save_checkpoint)
        # Falls back to computing from grid if not in checkpoint (old checkpoints)
        pbar.set_description("Loading zone stats")
        zone_stats_path = ckpt_dir / "zone_stats.parquet"
        if zone_stats_path.exists():
            zone_stats_df = pd.read_parquet(zone_stats_path)
            logger.info(f"  zone_stats loaded from checkpoint: {len(zone_stats_df)} zones")
        else:
            logger.warning(
                "zone_stats.parquet not found in checkpoint — computing from grid (fallback). "
                "Re-run training to fix this permanently."
            )
            target_col = (
                "zone_hour_violation_count"
                if time_resolution_eff == "hour"
                else "zone_day_violation_count"
            )
            cis_path_fb = project_root / "data" / "processed" / "cis_table.parquet"
            cis_fb = pd.read_parquet(cis_path_fb) if cis_path_fb.exists() else pd.DataFrame(columns=["zone_id", "cis_score"])
            zone_stats_df = (
                zone_grid_df.groupby("zone_id", observed=True)
                .agg(
                    zone_mean_count   =(target_col, "mean"),
                    zone_median_count =(target_col, "median"),
                    zone_total_count  =(target_col, "sum"),
                    zone_junction_frac=("fraction_at_junction", "mean"),
                )
                .reset_index()
                .merge(
                    cis_fb[["zone_id", "cis_score"]].rename(columns={"cis_score": "zone_cis_score"}),
                    on="zone_id",
                    how="left",
                )
            )
            zone_stats_df["zone_cis_score"] = zone_stats_df["zone_cis_score"].fillna(0.0)
            logger.info(f"  zone_stats computed from grid fallback: {len(zone_stats_df)} zones")
        pbar.update(1)

        # Determine feature columns (same logic as train.py)
        pbar.set_description("Building feature list")
        feature_cols = _get_feature_cols(time_resolution_eff, zone_grid_df)
        pbar.update(1)

    logger.info(
        f"✓ Ranker loaded: {model_name_eff}/{time_resolution_eff} | "
        f"{len(cis_df)} CIS zones | {len(feature_cols)} features"
    )

    return {
        "model":           model,
        "model_name":      model_name_eff,
        "time_resolution": time_resolution_eff,
        "cis_df":          cis_df,
        "zone_grid_df":    zone_grid_df,
        "zone_stats_df":   zone_stats_df,
        "feature_cols":    feature_cols,
        "ckpt_dir":        ckpt_dir,
        "meta":            meta,
    }


# ── Feature column builder (must match train.py exactly) ─────────────────────

def _get_feature_cols(time_resolution: str, zone_grid_df: pd.DataFrame) -> list[str]:
    """
    Build feature column list consistent with training (train.py _get_feature_cols).

    Phase 1 update: Uses zone aggregate features instead of raw zone_id/police_station_id/
    center_code_encoded. Must match train.py _get_feature_cols() exactly.
    """
    candidates = []
    # Temporal
    if time_resolution == "hour":
        candidates.append("hour_of_day")
    candidates += ["day_of_week", "is_weekend", "month"]

    # Zone aggregate features (Phase 1 — replaces zone_id lookup table pattern)
    candidates += [
        "zone_mean_count",
        "zone_median_count",
        "zone_cis_score",
        "zone_junction_frac",
        "zone_total_count",
    ]

    # Spatial (time-block-level junction fraction)
    candidates.append("fraction_at_junction")

    # Historical
    candidates.append("rolling_7d_count")

    # Categorical (aggregated versions from Phase B)
    candidates += [
        "dominant_violation_type",
        "dominant_vehicle_type",
        "violation_type_primary_encoded",
        "vehicle_type_encoded",
    ]

    # Optional
    candidates.append("data_sent_to_scita_mean")

    # Return all candidates — the scaffold's fillna(0) handles genuinely missing cols.
    # Do NOT filter by zone_grid_df.columns: zone aggregate cols (zone_mean_count etc.)
    # live in zone_stats_df, not the grid, so filtering would silently drop them.
    return candidates


# ── Zone scaffold builder ─────────────────────────────────────────────────────

def _build_zone_scaffold(
    ranker: dict[str, Any],
    target_date: pd.Timestamp,
    target_hour: int | None,
) -> pd.DataFrame:
    """
    Build a one-row-per-zone feature DataFrame for a specific date/hour.

    For each zone, we use the median feature values observed in the training data
    (excluding the target column). The temporal fields (hour_of_day, is_weekend,
    month) are overridden with the requested values.

    Args:
        ranker:      Loaded ranker dict from load_ranker().
        target_date: The date to predict for (pd.Timestamp).
        target_hour: Hour of day [0–23]. Only used if time_resolution='hour'.

    Returns:
        scaffold_df: One row per zone_id with all feature columns filled in.
    """
    zone_grid_df    = ranker["zone_grid_df"]
    zone_stats_df   = ranker["zone_stats_df"]
    feature_cols    = ranker["feature_cols"]
    time_resolution = ranker["time_resolution"]

    # Start from one-row-per-zone using the zone stats lookup.
    # zone_stats_df has columns: zone_id, zone_mean_count, zone_median_count,
    # zone_total_count, zone_junction_frac, zone_cis_score
    scaffold = zone_stats_df.copy()

    # Add time-block-level median features from the grid (fraction_at_junction,
    # rolling_7d_count, dominant_violation_type, etc.) — use overall zone medians
    temporal_skip = {"hour_of_day", "is_weekend", "month", "day_of_week",
                     "zone_id", "zone_hour_violation_count", "zone_day_violation_count", "date"}
    agg_dict: dict[str, str] = {
        col: "median"
        for col in zone_grid_df.columns
        if col not in temporal_skip and pd.api.types.is_numeric_dtype(zone_grid_df[col])
    }
    if agg_dict:
        grid_medians = (
            zone_grid_df.groupby("zone_id", observed=True)
            .agg(agg_dict)
            .reset_index()
        )
        scaffold = scaffold.merge(grid_medians, on="zone_id", how="left")

    # Override temporal features with the requested values
    if time_resolution == "hour" and target_hour is not None:
        scaffold["hour_of_day"] = int(target_hour)

    scaffold["is_weekend"]  = int(target_date.dayofweek >= 5)
    scaffold["day_of_week"] = int(target_date.dayofweek)
    scaffold["month"]       = int(target_date.month)

    # Ensure all model feature_cols are present; fill any missing with 0
    for col in feature_cols:
        if col not in scaffold.columns:
            scaffold[col] = 0
    scaffold = scaffold.fillna(0)

    return scaffold


# ── Ranker entry point ────────────────────────────────────────────────────────

def rank_zones(
    ranker: dict[str, Any],
    target_date: str | pd.Timestamp = "2024-03-15",
    target_hour: int | None = 9,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Score all zones for a requested date/hour and return the top-K ranked table.

    Formula (eval.yaml ranker v1.0):
        priority_score(zone, t) = predicted_count(zone, t) × CIS(zone)

    Args:
        ranker:      Loaded ranker dict from load_ranker().
        target_date: Date string or Timestamp to predict for.
        target_hour: Hour of day [0–23]. Ignored for day-resolution models.
        top_k:       Number of top zones to return (default 10).

    Returns:
        top_k_df: DataFrame with columns:
            zone_id, predicted_count, cis_score, priority_score,
            priority_tier, has_junction, formula_version
    """
    model           = ranker["model"]
    model_name      = ranker["model_name"]
    time_resolution = ranker["time_resolution"]
    cis_df          = ranker["cis_df"]
    feature_cols    = ranker["feature_cols"]

    target_date = pd.Timestamp(target_date)
    hour_label  = f" hour={target_hour}" if time_resolution == "hour" else ""
    logger.info(
        f"Ranking zones: model={model_name}/{time_resolution} | "
        f"date={target_date.date()}{hour_label} | top_k={top_k}"
    )

    with tqdm(total=3, desc="Ranking zones", unit="step", leave=True) as pbar:

        # Step 1: Build feature scaffold (one row per zone)
        pbar.set_description("Building zone scaffold")
        scaffold_df = _build_zone_scaffold(ranker, target_date, target_hour)
        X = scaffold_df[feature_cols].fillna(-1)
        zone_ids = scaffold_df["zone_id"].values
        pbar.update(1)

        # Step 2: Predict violation counts per zone
        pbar.set_description("Predicting counts")
        if model_name == "lightgbm":
            # lgb.Booster.predict() takes numpy array
            y_pred = model.predict(X.values)
        else:
            y_pred = model.predict(X)
        y_pred = np.clip(y_pred, 0, None)  # counts cannot be negative
        pbar.update(1)

        # Step 3: Merge with CIS and compute priority_score
        pbar.set_description("Computing priority scores")
        pred_df = pd.DataFrame({
            "zone_id":         zone_ids,
            "predicted_count": y_pred.astype(float).round(2),
        })

        result_df = pred_df.merge(
            cis_df[["zone_id", "cis_score", "has_junction", "priority_tier", "formula_version"]],
            on="zone_id",
            how="left",
        )
        result_df["cis_score"] = result_df["cis_score"].fillna(0.0)

        # priority_score = predicted_count × CIS
        result_df["priority_score"] = (
            result_df["predicted_count"] * result_df["cis_score"]
        ).round(4)

        # Re-compute priority_tier based on final priority_score
        max_score = result_df["priority_score"].max()
        if max_score > 0:
            result_df["priority_tier"] = pd.cut(
                result_df["priority_score"],
                bins=[-0.001, 0.4 * max_score, 0.7 * max_score, max_score + 0.001],
                labels=["LOW", "MEDIUM", "HIGH"],
            ).astype(str)
        else:
            result_df["priority_tier"] = "LOW"

        # Sort by priority_score descending → top-K
        top_k_df = (
            result_df
            .sort_values("priority_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        top_k_df.index = top_k_df.index + 1  # rank 1-based
        top_k_df.index.name = "rank"
        pbar.update(1)

    logger.info(
        f"✓ Ranking complete: top zone={int(top_k_df.iloc[0]['zone_id'])} | "
        f"priority_score={top_k_df.iloc[0]['priority_score']:.4f} | "
        f"tier={top_k_df.iloc[0]['priority_tier']}"
    )

    return top_k_df


# ── Batch ranking (multiple hours) ────────────────────────────────────────────

def rank_day_schedule(
    ranker: dict[str, Any],
    target_date: str | pd.Timestamp = "2024-03-15",
    hours: list[int] | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Generate enforcement schedules for multiple hours in a day.

    Runs rank_zones() for each requested hour and concatenates results.
    Useful for generating a full-day enforcement deployment plan.

    Args:
        ranker:      Loaded ranker dict.
        target_date: Date to plan for.
        hours:       List of hours to rank. Default: [7,8,9,10,17,18,19,20] (morning + evening rush).
        top_k:       Top zones per hour.

    Returns:
        schedule_df: DataFrame with hour, rank, zone_id, predicted_count, priority_score, priority_tier.
    """
    if hours is None:
        hours = [7, 8, 9, 10, 17, 18, 19, 20]  # Morning + evening rush hours

    target_date = pd.Timestamp(target_date)
    all_rows: list[pd.DataFrame] = []

    logger.info(
        f"Generating day schedule: date={target_date.date()} | "
        f"{len(hours)} hours | top_k={top_k}"
    )

    for hour in tqdm(hours, desc="Ranking hours", unit="hour"):
        hour_df = rank_zones(ranker, target_date=target_date, target_hour=hour, top_k=top_k)
        hour_df = hour_df.reset_index()  # bring rank back as column
        hour_df.insert(0, "hour", hour)
        all_rows.append(hour_df)

    schedule_df = pd.concat(all_rows, ignore_index=True)
    logger.info(f"✓ Day schedule: {len(schedule_df)} rows ({len(hours)} hours × {top_k} zones)")
    return schedule_df
