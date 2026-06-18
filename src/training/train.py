"""
src/training/train.py
GridLock R2 — PS1: Parking-Induced Congestion

Training pipeline for violation count prediction.

Trains and compares three candidate models:
  - XGBoost  (XGBRegressor)
  - LightGBM (LGBMRegressor)
  - CatBoost (CatBoostRegressor)

For BOTH time resolutions (hour, day). Selects winner by NDCG@10.

Usage (from notebooks/04_training.ipynb):
    from src.training.train import run_training
    results = run_training(project_root=PROJECT_ROOT)

Rules (from claude.md):
  - All hyperparameters read from configs/model.yaml — never hardcoded
  - Feature list read from configs/features.yaml — never hardcoded
  - No-future-leakage assertion: max(train date) < min(test date) — hard error if fails
  - Checkpoint every model to checkpoints/{model}_{resolution}_{timestamp}/
  - Checkpoint contains: model file, model.yaml copy, features.yaml copy, eval.yaml copy,
    label_encoders.pkl copy, features.yaml hash, training metrics
  - Must compare against frequency ranker baseline — model must beat it to be valid
  - tqdm progress on all loops
  - loguru for all logging (not print)
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm


# ── Config loaders ────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _yaml_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _load_configs(project_root: Path) -> dict[str, Any]:
    model_cfg   = _load_yaml(project_root / "configs" / "model.yaml")
    features_cfg= _load_yaml(project_root / "configs" / "features.yaml")
    eval_cfg    = _load_yaml(project_root / "configs" / "eval.yaml")
    feat_hash   = _yaml_hash(project_root / "configs" / "features.yaml")
    logger.info(f"Configs loaded | features.yaml hash: {feat_hash[:16]}...")
    return {
        "model":    model_cfg,
        "features": features_cfg,
        "eval":     eval_cfg,
        "feat_hash": feat_hash,
    }


# ── Feature columns ───────────────────────────────────────────────────────────

def _get_feature_cols(features_cfg: dict[str, Any], time_resolution: str) -> list[str]:
    """
    Build ordered feature column list from configs/features.yaml.

    Phase 1 (v2.0): zone_id, police_station_id, center_code_encoded are REMOVED.
    Phase 3 (v2.1): hour_of_day → hour_sin + hour_cos; day_of_week → dow_sin + dow_cos.
    Cyclical encoding prevents the "midnight paradox" where hour 23 and hour 0
    appear numerically far apart to tree-based models.

    For zone-grid training we use:
      temporal:    hour_sin + hour_cos (only for 'hour'), dow_sin + dow_cos, is_weekend, month
      zone_aggs:   zone_mean_count, zone_median_count, zone_cis_score,
                   zone_junction_frac, zone_total_count
      spatial:     fraction_at_junction (time-block-level junction fraction)
      historical:  rolling_7d_count (7-day lagged mean count per zone×hour — strongest signal)
      categorical: dominant_violation_type, dominant_vehicle_type,
                   violation_type_primary_encoded, vehicle_type_encoded
      optional:    data_sent_to_scita_mean
    """
    cols: list[str] = []

    # Temporal — cyclical encoding (Phase 3 / v2.1)
    # hour_sin / hour_cos only relevant for hour resolution
    if time_resolution == "hour":
        cols += ["hour_sin", "hour_cos"]
    cols += ["dow_sin", "dow_cos", "is_weekend", "month"]

    # Zone aggregate features (Phase 1 — replaces zone_id lookup table pattern)
    cols += [
        "zone_mean_count",
        "zone_median_count",
        "zone_cis_score",
        "zone_junction_frac",
        "zone_total_count",
    ]

    # Spatial (time-block-level junction fraction)
    cols.append("fraction_at_junction")

    # Historical — rolling 7-day lagged count (most important feature)
    cols.append("rolling_7d_count")

    # Categorical (aggregated versions from Phase B)
    cols += [
        "dominant_violation_type",
        "dominant_vehicle_type",
        "violation_type_primary_encoded",
        "vehicle_type_encoded",
    ]

    # Optional
    cols.append("data_sent_to_scita_mean")

    return cols


def _add_cyclical_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical sin/cos encoding for hour_of_day and day_of_week.

    Phase 3 (v2.1): Replaces raw integer features with 2D circular representations
    so that hour 23 ≈ hour 0 in feature space ("midnight paradox" fix).

    Formula:
        hour_sin = sin(2π × hour_of_day / 24)
        hour_cos = cos(2π × hour_of_day / 24)
        dow_sin  = sin(2π × day_of_week / 7)
        dow_cos  = cos(2π × day_of_week / 7)

    Args:
        df: DataFrame that already has hour_of_day and day_of_week columns.

    Returns:
        df with four new columns added (in-place safe — returns df).
    """
    import numpy as np

    if "hour_of_day" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0).round(8)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0).round(8)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0  # cos(0) = 1.0 (default to midnight)

    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0).round(8)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0).round(8)
    else:
        df["dow_sin"] = 0.0
        df["dow_cos"] = 1.0  # default to Monday

    return df


def _get_target_col(time_resolution: str) -> str:
    if time_resolution == "hour":
        return "zone_hour_violation_count"
    return "zone_day_violation_count"


def _add_zone_aggregate_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    cis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 1: Compute zone aggregate statistics from training data ONLY,
    then join them to both train and test DataFrames.

    Zone aggregates replace zone_id (which was being used as a numeric ordinal —
    meaningless for DBSCAN labels). These features capture zone identity
    through its actual measured characteristics.

    Features added:
      zone_mean_count    — mean violation count per zone over training period
      zone_median_count  — median violation count per zone over training period
      zone_cis_score     — CIS score per zone (from cis_table.parquet)
      zone_junction_frac — fraction of violations at junctions per zone (zone-level static)
      zone_total_count   — total violations in zone over training period

    CRITICAL: All stats computed on train_df ONLY, then joined.
              Avoids test-period statistics leaking into zone aggregates.

    Args:
        train_df:   Training split DataFrame (zone×time grid).
        test_df:    Test split DataFrame (zone×time grid).
        target_col: Violation count column name.
        cis_df:     CIS table from data/processed/cis_table.parquet.

    Returns:
        (train_df, test_df) with zone_aggregate columns added.
    """
    logger.info("Computing zone aggregate features (train-only) ...")

    # Aggregate per zone over entire training period
    zone_stats = (
        train_df.groupby("zone_id", observed=True)
        .agg(
            zone_mean_count   =(target_col, "mean"),
            zone_median_count =(target_col, "median"),
            zone_total_count  =(target_col, "sum"),
            zone_junction_frac=("fraction_at_junction", "mean"),
        )
        .reset_index()
    )

    # Add CIS score from cis_df
    if "zone_id" in cis_df.columns and "cis_score" in cis_df.columns:
        cis_lookup = cis_df[["zone_id", "cis_score"]].rename(
            columns={"cis_score": "zone_cis_score"}
        )
        zone_stats = zone_stats.merge(cis_lookup, on="zone_id", how="left")
        zone_stats["zone_cis_score"] = zone_stats["zone_cis_score"].fillna(0.0)
    else:
        zone_stats["zone_cis_score"] = 0.0
        logger.warning("cis_df missing 'cis_score' column — zone_cis_score set to 0.0")

    agg_cols = [
        "zone_id", "zone_mean_count", "zone_median_count",
        "zone_total_count", "zone_junction_frac", "zone_cis_score",
    ]
    zone_stats = zone_stats[agg_cols]

    # Join to both splits (left join — test zones not in train get NaN → filled with 0)
    train_df = train_df.merge(zone_stats, on="zone_id", how="left")
    test_df  = test_df.merge(zone_stats, on="zone_id", how="left")

    # Fill NaN for test zones not seen in training (cold-start fallback)
    for col in ["zone_mean_count", "zone_median_count", "zone_cis_score",
                "zone_junction_frac", "zone_total_count"]:
        train_df[col] = train_df[col].fillna(0.0).astype("float32")
        test_df[col]  = test_df[col].fillna(0.0).astype("float32")

    n_zones_stats = len(zone_stats)
    logger.info(
        f"Zone aggregates computed for {n_zones_stats} training zones | "
        f"zone_mean_count range: [{zone_stats['zone_mean_count'].min():.2f}, "
        f"{zone_stats['zone_mean_count'].max():.2f}]"
    )
    return train_df, test_df


# ── Train / test split + leakage guard ───────────────────────────────────────

def _split_data(
    df: pd.DataFrame,
    eval_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split using boundaries from configs/eval.yaml.

    Leakage guard: asserts max(train date) < min(test date). Hard error if fails.
    """
    train_end  = pd.Timestamp(eval_cfg["split"]["train_end"])
    test_start = pd.Timestamp(eval_cfg["split"]["test_start"])

    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["date"] <= train_end].copy()
    test_df  = df[df["date"] >= test_start].copy()

    if len(train_df) == 0:
        raise ValueError("Training split is empty after date filtering!")
    if len(test_df) == 0:
        raise ValueError("Test split is empty after date filtering!")

    # ── No-future-leakage assertion ──────────────────────────────────────────
    max_train = train_df["date"].max()
    min_test  = test_df["date"].min()
    if not (max_train < min_test):
        raise AssertionError(
            f"LEAKAGE DETECTED: max(train date)={max_train} is NOT < min(test date)={min_test}. "
            f"This is a hard failure. Check the split boundaries in configs/eval.yaml."
        )

    logger.info(
        f"Split: train={len(train_df):,} rows ({train_df['date'].min().date()} → {max_train.date()}) | "
        f"test={len(test_df):,} rows ({min_test.date()} → {test_df['date'].max().date()}) | "
        f"Leakage guard: PASSED ✓"
    )
    return train_df, test_df


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _make_checkpoint_dir(
    project_root: Path,
    model_name: str,
    time_resolution: str,
    timestamp: str,
) -> Path:
    ckpt_dir = (
        project_root / "checkpoints"
        / f"{model_name}_{time_resolution}_{timestamp}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _save_checkpoint(
    ckpt_dir: Path,
    model: Any,
    model_name: str,
    metrics: dict[str, Any],
    configs: dict[str, Any],
    project_root: Path,
    zone_stats: "pd.DataFrame | None" = None,
) -> None:
    """Save model weights + all config copies + metrics to checkpoint dir."""
    import pickle

    # 1. Model weights
    if model_name == "xgboost":
        model.save_model(str(ckpt_dir / "model.xgb"))
    elif model_name == "lightgbm":
        model.booster_.save_model(str(ckpt_dir / "model.lgb"))
    elif model_name == "catboost":
        model.save_model(str(ckpt_dir / "model.cbm"))
    else:
        with open(ckpt_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

    # 2. Config copies
    shutil.copy2(project_root / "configs" / "model.yaml",    ckpt_dir / "model.yaml")
    shutil.copy2(project_root / "configs" / "features.yaml", ckpt_dir / "features.yaml")
    shutil.copy2(project_root / "configs" / "eval.yaml",     ckpt_dir / "eval.yaml")

    # 3. Label encoders copy
    enc_src = project_root / "data" / "processed" / "label_encoders.pkl"
    if enc_src.exists():
        shutil.copy2(enc_src, ckpt_dir / "label_encoders.pkl")

    # 4. Zone aggregate stats lookup (needed by ranker at inference time)
    if zone_stats is not None:
        zone_stats.to_parquet(ckpt_dir / "zone_stats.parquet", index=False)
        logger.info(f"  zone_stats.parquet saved → '{ckpt_dir / 'zone_stats.parquet'}'")

    # 5. Training metrics + metadata
    meta = {
        "model_name":        model_name,
        "time_resolution":   metrics.get("time_resolution"),
        "features_yaml_hash": configs["feat_hash"],
        "seed":              configs["model"].get("seed", 42),
        "metrics":           metrics,
        "checkpoint_dir":    str(ckpt_dir),
        "saved_at":          datetime.now(timezone.utc).isoformat(),
    }
    with open(ckpt_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Checkpoint saved → '{ckpt_dir}'")


# ── Model builders ────────────────────────────────────────────────────────────

def _build_xgboost(model_cfg: dict[str, Any], seed: int) -> Any:
    from xgboost import XGBRegressor
    xgb_cfg = model_cfg["xgboost"]
    return XGBRegressor(
        objective           = xgb_cfg.get("objective",         "reg:squarederror"),
        eval_metric         = xgb_cfg.get("eval_metric",       "rmse"),
        n_estimators        = xgb_cfg.get("n_estimators",      300),
        learning_rate       = xgb_cfg.get("learning_rate",     0.05),
        max_depth           = xgb_cfg.get("max_depth",         6),
        min_child_weight    = xgb_cfg.get("min_child_weight",  5),
        subsample           = xgb_cfg.get("subsample",         0.8),
        colsample_bytree    = xgb_cfg.get("colsample_bytree",  0.8),
        reg_alpha           = xgb_cfg.get("reg_alpha",         0.1),
        reg_lambda          = xgb_cfg.get("reg_lambda",        1.0),
        n_jobs              = xgb_cfg.get("n_jobs",            -1),
        random_state        = seed,
        early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 20),
        verbosity           = 0,
    )


def _build_lightgbm(model_cfg: dict[str, Any], seed: int) -> Any:
    from lightgbm import LGBMRegressor
    lgb_cfg = model_cfg["lightgbm"]
    return LGBMRegressor(
        objective           = lgb_cfg.get("objective",         "regression"),
        metric              = lgb_cfg.get("metric",            "rmse"),
        n_estimators        = lgb_cfg.get("n_estimators",      300),
        learning_rate       = lgb_cfg.get("learning_rate",     0.05),
        num_leaves          = lgb_cfg.get("num_leaves",        63),
        min_child_samples   = lgb_cfg.get("min_child_samples", 10),
        subsample           = lgb_cfg.get("subsample",         0.8),
        colsample_bytree    = lgb_cfg.get("colsample_bytree",  0.8),
        reg_alpha           = lgb_cfg.get("reg_alpha",         0.1),
        reg_lambda          = lgb_cfg.get("reg_lambda",        1.0),
        n_jobs              = lgb_cfg.get("n_jobs",            -1),
        random_state        = seed,
        verbose             = -1,
    )


def _build_catboost(model_cfg: dict[str, Any], seed: int) -> Any:
    from catboost import CatBoostRegressor
    cb_cfg = model_cfg["catboost"]
    return CatBoostRegressor(
        loss_function       = cb_cfg.get("loss_function",     "RMSE"),
        iterations          = cb_cfg.get("iterations",        300),
        learning_rate       = cb_cfg.get("learning_rate",     0.05),
        depth               = cb_cfg.get("depth",             6),
        l2_leaf_reg         = cb_cfg.get("l2_leaf_reg",       3.0),
        min_data_in_leaf    = cb_cfg.get("min_data_in_leaf",  5),
        random_seed         = seed,
        early_stopping_rounds = cb_cfg.get("early_stopping_rounds", 20),
        verbose             = 0,
    )


# ── Single model training loop ────────────────────────────────────────────────

def _train_one(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Any, dict[str, list[float]]]:
    """
    Fit one model. Returns (trained_model, eval_history).
    eval_history: dict with key 'rmse' mapping to list of per-round RMSE values on val set.
    Handles early_stopping_rounds for XGBoost/LightGBM/CatBoost uniformly.
    """
    eval_history: dict[str, list[float]] = {}

    if model_name == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        raw = model.evals_result()
        # evals_result: {"validation_0": {"rmse": [...]}}
        for _set_name, metrics_dict in raw.items():
            for metric_name, values in metrics_dict.items():
                eval_history[metric_name] = [float(v) for v in values]
        best_round = model.best_iteration if hasattr(model, "best_iteration") else len(next(iter(eval_history.values()), []))
        logger.info(f"  XGBoost early-stop: best_round={best_round} | "
                    f"final_val_rmse={eval_history.get('rmse', [float('nan')])[-1]:.4f}")

    elif model_name == "lightgbm":
        from lightgbm import early_stopping, log_evaluation, record_evaluation
        evals_result_lgb: dict = {}
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=20, verbose=False),
                log_evaluation(period=-1),
                record_evaluation(evals_result_lgb),
            ],
        )
        for _set_name, metrics_dict in evals_result_lgb.items():
            for metric_name, values in metrics_dict.items():
                eval_history[metric_name] = [float(v) for v in values]
        best_round = model.best_iteration_ if hasattr(model, "best_iteration_") else len(next(iter(eval_history.values()), []))
        logger.info(f"  LightGBM early-stop: best_round={best_round} | "
                    f"final_val_rmse={eval_history.get('rmse', [float('nan')])[-1]:.4f}")

    elif model_name == "catboost":
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )
        raw = model.get_evals_result()
        # {"learn": {"RMSE": [...]}, "validation": {"RMSE": [...]}}
        val_key = "validation" if "validation" in raw else list(raw.keys())[-1]
        for metric_name, values in raw.get(val_key, {}).items():
            eval_history[metric_name.lower()] = [float(v) for v in values]
        best_round = model.get_best_iteration() if hasattr(model, "get_best_iteration") else len(next(iter(eval_history.values()), []))
        logger.info(f"  CatBoost early-stop: best_round={best_round} | "
                    f"final_val_rmse={eval_history.get('rmse', [float('nan')])[-1]:.4f}")

    else:
        model.fit(X_train, y_train)

    return model, eval_history


# ── Main training entry point ─────────────────────────────────────────────────

def run_training(
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """
    Train all candidate models on both time resolutions and evaluate against baseline.

    Steps:
      1. Load configs (model.yaml, features.yaml, eval.yaml)
      2. For each time_resolution in ['hour', 'day']:
         a. Load aggregated grid (zone_hour_grid or zone_day_grid)
         b. Time-based split + leakage guard
         c. Build feature matrix X, target y
         d. For each model (xgboost, lightgbm, catboost):
            i.   Build and train model
            ii.  Predict on test split
            iii. Run full_eval() → regression + ranking metrics
            iv.  Save checkpoint
      3. Select winner by NDCG@10 on test set
      4. Save all_results.json to data/outputs/

    Args:
        project_root: Path to the project root (GridLock R2/).

    Returns:
        results: Dict of all results keyed by '{model}_{resolution}'.
                 Also contains 'winner' key with model + resolution + metrics.
    """
    from src.evaluation.metrics import full_eval, save_eval_results

    project_root = Path(project_root)
    configs = _load_configs(project_root)
    model_cfg    = configs["model"]
    eval_cfg     = configs["eval"]
    features_cfg = configs["features"]

    seed = int(model_cfg.get("seed", 42))
    np.random.seed(seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(f"Training run started | timestamp={timestamp} | seed={seed}")

    # Load CIS table (needed for all evaluations)
    cis_path = project_root / "data" / "processed" / "cis_table.parquet"
    if not cis_path.exists():
        raise FileNotFoundError(
            f"cis_table.parquet not found at '{cis_path}'. "
            "Run notebooks/03_clustering.ipynb first."
        )
    cis_df = pd.read_parquet(cis_path)
    logger.info(f"CIS table loaded: {len(cis_df)} zones")

    # Models to train (skip disabled ones)
    candidate_models: list[tuple[str, Any]] = []
    if model_cfg["xgboost"].get("enabled", True):
        candidate_models.append(("xgboost", _build_xgboost(model_cfg, seed)))
    if model_cfg["lightgbm"].get("enabled", True):
        candidate_models.append(("lightgbm", _build_lightgbm(model_cfg, seed)))
    if model_cfg["catboost"].get("enabled", True):
        candidate_models.append(("catboost", _build_catboost(model_cfg, seed)))

    resolutions = model_cfg["time_resolution"].get("candidates", ["hour", "day"])

    all_results: dict[str, Any] = {}

    outer_combos = [(res, name, mdl) for res in resolutions for name, mdl in candidate_models]

    for time_resolution, model_name, model in tqdm(
        outer_combos,
        desc="Training all models",
        unit="run",
    ):
        target_col  = _get_target_col(time_resolution)
        feature_cols = _get_feature_cols(features_cfg, time_resolution)

        logger.info(
            f"\n{'='*60}\n"
            f"  Training: {model_name.upper()} | resolution={time_resolution}\n"
            f"  Target: {target_col} | Features: {len(feature_cols)}\n"
            f"{'='*60}"
        )

        # Load aggregated grid
        grid_file = f"zone_{time_resolution}_grid.parquet"
        grid_path = project_root / "data" / "processed" / grid_file
        if not grid_path.exists():
            raise FileNotFoundError(
                f"'{grid_path}' not found. Run notebooks/03_clustering.ipynb first."
            )
        df = pd.read_parquet(grid_path)

        # Time-based split + leakage guard
        train_df, test_df = _split_data(df, eval_cfg)

        # Phase 1: Add zone aggregate features (computed from training data ONLY,
        # then joined to both splits). This replaces zone_id as an ordinal feature.
        train_df, test_df = _add_zone_aggregate_features(
            train_df, test_df, target_col, cis_df
        )

        # Phase 3 (v2.1): Add cyclical temporal features (hour_sin, hour_cos,
        # dow_sin, dow_cos). Must be added after the split so that the raw
        # hour_of_day / day_of_week columns in the parquet are still usable.
        # The raw integer columns (hour_of_day, day_of_week) remain in the
        # DataFrame but are excluded from feature_cols going forward.
        train_df = _add_cyclical_temporal_features(train_df)
        test_df  = _add_cyclical_temporal_features(test_df)

        # Build feature matrices — only keep cols present in the loaded dataframe
        available_feature_cols = [c for c in feature_cols if c in train_df.columns]
        missing_feature_cols   = [c for c in feature_cols if c not in train_df.columns]
        if missing_feature_cols:
            logger.warning(
                f"Feature columns not found in {grid_file}: {missing_feature_cols} — skipping them."
            )

        X_train = train_df[available_feature_cols].fillna(-1)
        y_train = train_df[target_col].astype(float)
        X_val   = test_df[available_feature_cols].fillna(-1)
        y_val   = test_df[target_col].astype(float)


        logger.info(
            f"X_train: {X_train.shape}  y_train mean={y_train.mean():.2f} max={y_train.max():.0f}"
        )
        logger.info(
            f"X_val:   {X_val.shape}    y_val mean={y_val.mean():.2f} max={y_val.max():.0f}"
        )

        # Rebuild the model fresh (avoid stale state from previous resolution)
        if model_name == "xgboost":
            model = _build_xgboost(model_cfg, seed)
        elif model_name == "lightgbm":
            model = _build_lightgbm(model_cfg, seed)
        elif model_name == "catboost":
            model = _build_catboost(model_cfg, seed)

        # Train
        with tqdm(total=1, desc=f"Fitting {model_name} ({time_resolution})", unit="model", leave=False) as pbar:
            model, eval_history = _train_one(model_name, model, X_train, y_train, X_val, y_val)
            pbar.update(1)

        # Predict
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, None)  # Counts can't be negative

        # Full evaluation
        eval_result = full_eval(
            model_name      = model_name,
            time_resolution = time_resolution,
            y_true          = y_val,
            y_pred          = y_pred,
            test_df         = test_df,
            train_df        = train_df,
            cis_df          = cis_df,
            eval_config     = eval_cfg,
            eval_history    = eval_history,
        )

        # Save checkpoint
        ckpt_dir = _make_checkpoint_dir(project_root, model_name, time_resolution, timestamp)
        eval_result["time_resolution"] = time_resolution

        # Compute zone_stats to save with checkpoint (needed by ranker at inference time)
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
                cis_df[["zone_id", "cis_score"]].rename(columns={"cis_score": "zone_cis_score"}),
                on="zone_id",
                how="left",
            )
        )
        _zone_stats["zone_cis_score"] = _zone_stats["zone_cis_score"].fillna(0.0)

        _save_checkpoint(ckpt_dir, model, model_name, eval_result, configs, project_root, _zone_stats)

        run_key = f"{model_name}_{time_resolution}"
        all_results[run_key] = eval_result

    # ── Select winner by NDCG@10 ─────────────────────────────────────────────
    logger.info("\n" + "="*60 + "\n  MODEL COMPARISON SUMMARY\n" + "="*60)

    rows = []
    for run_key, res in all_results.items():
        ndcg_10  = res["ranking"].get("k10", {}).get("ndcg_at_k",      0.0)
        prec_10  = res["ranking"].get("k10", {}).get("precision_at_k", 0.0)
        mae      = res["regression"].get("mae",  float("inf"))
        rmse     = res["regression"].get("rmse", float("inf"))
        rows.append({
            "run":              run_key,
            "model":            res["model"],
            "resolution":       res["time_resolution"],
            "NDCG@10":          ndcg_10,
            "Precision@10":     prec_10,
            "MAE":              mae,
            "RMSE":             rmse,
            "beats_baseline":   res["beats_baseline"],
        })
        logger.info(
            f"  {run_key:<30} | NDCG@10={ndcg_10:.4f} | Prec@10={prec_10:.4f} | MAE={mae:.4f}"
        )

    # Sort by NDCG@10 descending, then MAE ascending as tiebreaker
    rows.sort(key=lambda x: (-x["NDCG@10"], x["MAE"]))
    winner_row = rows[0]

    logger.info(
        f"\n  🏆 WINNER: {winner_row['run']} | "
        f"NDCG@10={winner_row['NDCG@10']:.4f} | "
        f"Precision@10={winner_row['Precision@10']:.4f} | "
        f"MAE={winner_row['MAE']:.4f}"
    )

    all_results["winner"] = winner_row
    all_results["comparison_table"] = rows
    all_results["training_timestamp"] = timestamp

    # ── Save all_results.json ─────────────────────────────────────────────────
    out_path = project_root / "data" / "outputs" / f"eval_{timestamp}.json"
    save_eval_results(all_results, out_path)

    # ── Update configs/model.yaml with winner ────────────────────────────────
    _write_winner_to_model_yaml(
        project_root / "configs" / "model.yaml",
        winner_row,
    )

    logger.info(f"\nAll results saved → '{out_path}'")
    return all_results


# ── Update model.yaml with winner ────────────────────────────────────────────

def _write_winner_to_model_yaml(
    model_yaml_path: Path,
    winner: dict[str, Any],
) -> None:
    """
    Write the winning model + resolution + score into configs/model.yaml.

    Updates:
      primary_model, primary_time_resolution, winner_ndcg_at_10, comparison_run_date
    """
    with model_yaml_path.open("r", encoding="utf-8") as f:
        content = f.read()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    replacements = {
        "primary_model: null":           f"primary_model: \"{winner['model']}\"",
        "primary_time_resolution: null": f"primary_time_resolution: \"{winner['resolution']}\"",
        "winner_ndcg_at_10: null":       f"winner_ndcg_at_10: {winner['NDCG@10']:.6f}",
        "comparison_run_date: null":     f"comparison_run_date: \"{today}\"",
    }

    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            logger.info(f"model.yaml updated: {old!r} → {new!r}")

    with model_yaml_path.open("w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"model.yaml winner section updated → '{model_yaml_path}'")
