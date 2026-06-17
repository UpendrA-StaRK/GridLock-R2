"""
src/data/features.py
GridLock R2 — PS1: Parking-Induced Congestion

Feature engineering pipeline. Two phases:

  Phase A — extract_row_features(df):
    Operates on the clean DataFrame from load.py (268k rows, 12 cols).
    Produces one row per violation event with all ML-ready features except zone_id.
    Steps:
      1. Parse violation_type JSON list → violation_type_primary (first atomic type)
      2. Derive is_at_junction = (junction_name != 'No Junction').astype(int8)
      3. Extract temporal features from created_datetime (UTC)
      4. Impute center_code nulls with mode per police_station group
      5. LabelEncode: violation_type_primary, vehicle_type, police_station, center_code
      6. Save label encoders to data/processed/label_encoders.pkl

  Phase B — aggregate_to_zone_grid(df_with_zones, time_resolution):
    Called AFTER clustering.py assigns zone_id via DBSCAN.
    Aggregates to zone × time-block grid and produces the regression target.
    time_resolution: 'hour' or 'day'

Pipeline protocol:
  - Called from notebooks/01_eda.ipynb (Phase A) and later notebooks (Phase B).
  - Never hardcode feature names — always read from configs/features.yaml.
  - Save label encoders alongside data so inference can load them without re-fitting.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# ── Config loader ─────────────────────────────────────────────────────────────

def load_features_config(
    config_path: str | Path = "configs/features.yaml",
) -> dict[str, Any]:
    """
    Load and return configs/features.yaml.

    Args:
        config_path: Path to features.yaml (relative to project root).

    Returns:
        Parsed YAML as a dict.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"configs/features.yaml not found at '{path.resolve()}'. "
            f"Run from the project root."
        )
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded features config v{cfg.get('version', '?')} from '{path}'")
    return cfg


def features_yaml_hash(config_path: str | Path = "configs/features.yaml") -> str:
    """
    Compute SHA-256 hash of configs/features.yaml.
    Logged in every training run to detect feature-list drift.

    Returns:
        Hex digest string.
    """
    path = Path(config_path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ── Phase A: Row-level feature extraction ────────────────────────────────────

def extract_row_features(
    df: pd.DataFrame,
    features_config_path: str | Path = "configs/features.yaml",
    encoder_save_path: str | Path = "data/processed/label_encoders.pkl",
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], dict[str, Any]]:
    """
    Extract all row-level ML features from the clean load.py DataFrame.

    Does NOT require zone_id — that comes from clustering.py.
    Call aggregate_to_zone_grid() after DBSCAN assigns zone_id.

    Steps (with tqdm progress bar):
      1. Parse violation_type JSON list → violation_type_primary
      2. Derive is_at_junction from junction_name
      3. Extract: hour_of_day, day_of_week, is_weekend, month
      4. Impute center_code nulls (mode per police_station group)
      5. LabelEncode: violation_type_primary, vehicle_type, police_station, center_code
      6. Save label encoders to encoder_save_path

    Args:
        df: Clean DataFrame from load_raw() (must have created_datetime as UTC datetime64).
        features_config_path: Path to configs/features.yaml.
        encoder_save_path: Where to save the fitted LabelEncoder dict (pickle).

    Returns:
        df_feat: DataFrame with all new feature columns added (original columns preserved).
        encoders: Dict mapping col_name → fitted LabelEncoder (for inference use).
        metadata: Dict with stats, encoder classes, features.yaml hash.

    Raises:
        ValueError: If any required source column is missing from df.
    """
    cfg = load_features_config(features_config_path)
    feat_hash = features_yaml_hash(features_config_path)
    logger.info(f"features.yaml hash: {feat_hash[:16]}...")

    _check_required_source_cols(df)

    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    metadata: dict[str, Any] = {
        "features_yaml_hash": feat_hash,
        "input_rows": len(df),
        "input_cols": len(df.columns),
    }

    steps = [
        "Parse violation_type",
        "Derive is_at_junction",
        "Temporal features",
        "Impute center_code",
        "Label encode",
    ]

    with tqdm(total=len(steps), desc="Feature engineering", unit="step", leave=True) as pbar:

        # ── Step 1: Parse violation_type ──────────────────────────────────
        pbar.set_description("Parse violation_type")
        df, vt_stats = _parse_violation_type(df)
        metadata["violation_type_stats"] = vt_stats
        logger.info(
            f"✓ violation_type parsed: {vt_stats['unique_primary_types']} primary types, "
            f"{vt_stats['parse_failures']} parse failures → set to 'UNKNOWN'"
        )
        pbar.update(1)

        # ── Step 2: Derive is_at_junction ─────────────────────────────────
        pbar.set_description("Derive is_at_junction")
        df["is_at_junction"] = (
            (df["junction_name"].str.strip() != "No Junction")
            .astype("int8")
        )
        n_junction = int(df["is_at_junction"].sum())
        metadata["is_at_junction_count"] = n_junction
        metadata["is_at_junction_pct"] = round(n_junction / len(df) * 100, 2)
        logger.info(
            f"✓ is_at_junction derived: {n_junction:,} junction rows "
            f"({metadata['is_at_junction_pct']:.1f}%)"
        )
        pbar.update(1)

        # ── Step 3: Temporal features ─────────────────────────────────────
        pbar.set_description("Temporal features")
        df = _extract_temporal(df)
        logger.info(
            "✓ Temporal features extracted: hour_of_day, day_of_week, is_weekend, month"
        )
        pbar.update(1)

        # ── Step 4: Impute center_code nulls ──────────────────────────────
        pbar.set_description("Impute center_code")
        df, impute_stats = _impute_center_code(df)
        metadata["center_code_imputation"] = impute_stats
        logger.info(
            f"✓ center_code imputed: {impute_stats['null_before']:,} nulls → "
            f"{impute_stats['null_after']} remaining"
        )
        pbar.update(1)

        # ── Step 5: Label encoding ────────────────────────────────────────
        pbar.set_description("Label encoding")
        df, encoders = _label_encode_all(df)
        enc_classes = {k: list(enc.classes_) for k, enc in encoders.items()}
        metadata["encoder_classes"] = enc_classes
        for col, enc in encoders.items():
            logger.info(
                f"  ✓ Encoded '{col}': {len(enc.classes_)} unique classes"
            )
        pbar.update(1)

    # ── Save label encoders ───────────────────────────────────────────────────
    _save_encoders(encoders, encoder_save_path)
    metadata["encoder_save_path"] = str(encoder_save_path)

    metadata["output_rows"] = len(df)
    metadata["output_cols"] = len(df.columns)
    metadata["new_columns"] = [
        "violation_type_primary",
        "violation_type_primary_encoded",
        "is_at_junction",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "month",
        "vehicle_type_encoded",
        "police_station_id",
        "center_code_encoded",
    ]

    logger.info(
        f"─── extract_row_features() complete: {len(df):,} rows × {len(df.columns)} cols ───"
    )
    return df, encoders, metadata


# ── Phase B: Zone × time-block aggregation ────────────────────────────────────

def aggregate_to_zone_grid(
    df: pd.DataFrame,
    time_resolution: str = "hour",
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Aggregate row-level features to a zone × time-block grid.

    REQUIRES zone_id column (from DBSCAN clustering). Call this AFTER
    clustering.py has assigned zone_id to every row.

    The output has one row per (zone_id, time_block) combination where at least
    one violation occurred. Zero-violation combinations are NOT added here —
    the training loop can optionally add them.

    Target column produced:
      zone_hour_violation_count  (if time_resolution='hour')
      zone_day_violation_count   (if time_resolution='day')

    Aggregation features (zone-level summaries for the time-block):
      - fraction_at_junction (fraction of violations at a junction in this block)
      - dominant_violation_type (mode of violation_type_primary_encoded)
      - dominant_vehicle_type (mode of vehicle_type_encoded)
      - police_station_id (mode — one station per zone assumed)
      - center_code_encoded (mode)
      - data_sent_to_scita_mean (mean of 0/1 flag)
      - is_weekend, day_of_week, month
      - rolling_7d_count: 7-day trailing mean per (zone_id, hour_of_day),
        shifted 1 day to avoid leakage (most important new feature)

    Args:
        df: Feature-engineered DataFrame (must have zone_id, created_datetime, and
            all feature columns from extract_row_features()).
        time_resolution: 'hour' → group by (zone_id, date, hour_of_day)
                         'day'  → group by (zone_id, date)
        save_path: If provided, save the aggregated DataFrame as parquet.

    Returns:
        agg_df: Aggregated DataFrame with target column and zone-level feature summary.

    Raises:
        ValueError: If zone_id column is missing (DBSCAN not yet run).
    """
    if "zone_id" not in df.columns:
        raise ValueError(
            "zone_id column is missing. Run clustering.py (DBSCAN) first to assign "
            "zone_id to every row, then call aggregate_to_zone_grid()."
        )
    if time_resolution not in ("hour", "day"):
        raise ValueError(f"time_resolution must be 'hour' or 'day', got '{time_resolution}'")

    df = df.copy()
    df["_date"] = df["created_datetime"].dt.date

    logger.info(f"Aggregating to zone × {time_resolution} grid ...")

    if time_resolution == "hour":
        group_keys = ["zone_id", "_date", "hour_of_day"]
        target_col = "zone_hour_violation_count"
    else:
        group_keys = ["zone_id", "_date"]
        target_col = "zone_day_violation_count"

    # Count rows per group = violation count per zone per time block
    with tqdm(total=4, desc=f"Aggregating (zone×{time_resolution})", unit="step", leave=True) as pbar:

        counts = (
            df.groupby(group_keys, observed=True)
            .size()
            .reset_index(name=target_col)
        )
        pbar.update(1)

        # Zone-level feature summary aggregated per time block
        agg_ops: dict = {
            "fraction_at_junction":           ("is_at_junction", "mean"),
            "dominant_violation_type":        ("violation_type_primary_encoded", _mode),
            "dominant_vehicle_type":          ("vehicle_type_encoded", _mode),
            # Phase 1: add mode-encoded categoricals directly (for train.py feature list)
            "violation_type_primary_encoded": ("violation_type_primary_encoded", _mode),
            "vehicle_type_encoded":           ("vehicle_type_encoded", _mode),
            # Phase 1: police_station_id and center_code_encoded are kept in the grid
            # for reference/backward-compatibility but are NOT in train.py's feature list.
            "police_station_id":              ("police_station_id", _mode),
            "center_code_encoded":            ("center_code_encoded", _mode),
            "data_sent_to_scita_mean":        ("data_sent_to_scita", "mean"),
            "is_weekend":                     ("is_weekend", _mode),
            "day_of_week":                    ("day_of_week", _mode),
            "month":                          ("month", _mode),
        }
        agg_features = (
            df.groupby(group_keys, observed=True)
            .agg(**agg_ops)
            .reset_index()
        )

        pbar.update(1)

        agg_df = counts.merge(agg_features, on=group_keys, how="left")

        # Rename _date to a cleaner name
        agg_df = agg_df.rename(columns={"_date": "date"})

        # Cast types
        agg_df["zone_id"]   = agg_df["zone_id"].astype("int32")
        agg_df[target_col]  = agg_df[target_col].astype("int32")
        agg_df["date"]      = pd.to_datetime(agg_df["date"])
        pbar.update(1)

        # ── Rolling 7-day historical count (leakage-free) ─────────────────
        # For each (zone_id, hour_of_day) — or just zone_id for day resolution —
        # compute the trailing 7-day rolling mean of the target, shifted 1 day
        # forward so the current day's count is never included.
        # This is the strongest predictive signal — captures recent zone activity.
        agg_df = agg_df.sort_values(["zone_id"] + (
            ["hour_of_day"] if time_resolution == "hour" else []
        ) + ["date"]).reset_index(drop=True)

        if time_resolution == "hour":
            roll_groups = ["zone_id", "hour_of_day"]
        else:
            roll_groups = ["zone_id"]

        agg_df["rolling_7d_count"] = (
            agg_df.groupby(roll_groups, observed=True)[target_col]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
            .fillna(0.0)
            .astype("float32")
        )
        pbar.update(1)

    n_zones = agg_df["zone_id"].nunique()
    n_rows = len(agg_df)
    roll_nonzero = int((agg_df["rolling_7d_count"] > 0).sum())
    logger.info(
        f"✓ Aggregation complete ({time_resolution}): {n_rows:,} zone×time rows | "
        f"{n_zones} unique zones | target='{target_col}' "
        f"(mean={agg_df[target_col].mean():.2f}, max={agg_df[target_col].max()}) | "
        f"rolling_7d_count non-zero: {roll_nonzero:,}/{n_rows:,}"
    )

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_parquet(out, index=False)
        logger.info(f"Aggregated grid saved → '{out}'")

    return agg_df


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_feature_metadata(
    metadata: dict[str, Any],
    output_path: str | Path = "data/processed/feature_metadata.json",
) -> None:
    """Persist feature engineering metadata to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Feature metadata saved → '{out}'")


def load_encoders(
    encoder_path: str | Path = "data/processed/label_encoders.pkl",
) -> dict[str, LabelEncoder]:
    """
    Load saved label encoders for inference.

    Args:
        encoder_path: Path saved by extract_row_features().

    Returns:
        Dict mapping column name → fitted LabelEncoder.
    """
    path = Path(encoder_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Label encoders not found at '{path}'. "
            f"Run extract_row_features() first."
        )
    with path.open("rb") as f:
        encoders = pickle.load(f)
    logger.info(f"Label encoders loaded from '{path}': {list(encoders.keys())}")
    return encoders


# ── Private helpers ───────────────────────────────────────────────────────────

def _check_required_source_cols(df: pd.DataFrame) -> None:
    """Raise ValueError if any source column expected by this module is missing."""
    required = [
        "created_datetime",
        "violation_type",
        "vehicle_type",
        "police_station",
        "center_code",
        "junction_name",
        "data_sent_to_scita",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"extract_row_features() — missing source columns: {missing}. "
            f"Run load_raw() first."
        )


def _parse_violation_type(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Parse violation_type JSON list strings → violation_type_primary.

    Rule (EDA-confirmed):
      - ast.literal_eval(value) → Python list
      - Take index[0] as the primary (first) atomic violation type
      - On parse failure: set to 'UNKNOWN'

    Returns:
        df: With new column violation_type_primary (str).
        stats: Parse stats dict.
    """
    parse_failures = 0
    primary_types: list[str] = []

    for val in tqdm(df["violation_type"], desc="  Parsing violation_type", leave=False):
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list) and len(parsed) > 0:
                primary_types.append(str(parsed[0]).strip())
            else:
                primary_types.append("UNKNOWN")
                parse_failures += 1
        except (ValueError, SyntaxError):
            primary_types.append("UNKNOWN")
            parse_failures += 1

    df["violation_type_primary"] = primary_types

    unique_types = df["violation_type_primary"].unique().tolist()
    stats = {
        "parse_failures": parse_failures,
        "unique_primary_types": len(unique_types),
        "primary_types": sorted(unique_types),
        "top5": df["violation_type_primary"].value_counts().head(5).to_dict(),
    }
    return df, stats


def _extract_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from created_datetime (UTC).

    Adds: hour_of_day, day_of_week, is_weekend, month
    All computed in UTC (consistent with how data is stored).
    """
    dt = df["created_datetime"].dt
    df["hour_of_day"] = dt.hour.astype("int8")        # [0, 23]
    df["day_of_week"] = dt.dayofweek.astype("int8")   # [0=Mon, 6=Sun]
    df["is_weekend"]  = (dt.dayofweek >= 5).astype("int8")  # Sat=5, Sun=6
    df["month"]       = dt.month.astype("int8")       # [1, 12]
    return df


def _impute_center_code(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Impute center_code nulls with mode per police_station group.

    Rationale: center_code is a geographic sub-zone ID correlated with police_station.
    Mode per station is the simplest defensible imputation (3.77% null).
    Any remaining nulls (stations where ALL rows have null center_code) → global mode.
    """
    null_before = int(df["center_code"].isna().sum())

    # Compute mode per police_station
    station_mode = (
        df.dropna(subset=["center_code"])
        .groupby("police_station")["center_code"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
    )

    def _fill(row: pd.Series) -> Any:
        if pd.isna(row["center_code"]):
            return station_mode.get(row["police_station"], None)
        return row["center_code"]

    df["center_code"] = df.apply(_fill, axis=1)

    # Fallback: global mode for any still-null
    global_mode = df["center_code"].dropna().mode()
    if not global_mode.empty and df["center_code"].isna().any():
        df["center_code"] = df["center_code"].fillna(global_mode.iloc[0])
        logger.warning(
            "Some center_code nulls remained after per-station imputation — "
            "filled with global mode."
        )

    null_after = int(df["center_code"].isna().sum())
    stats = {
        "null_before": null_before,
        "null_after": null_after,
        "imputed_count": null_before - null_after,
    }
    return df, stats


def _label_encode_all(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    LabelEncode all categorical columns per features.yaml encoding registry.

    Columns encoded:
      violation_type_primary → violation_type_primary_encoded
      vehicle_type           → vehicle_type_encoded
      police_station         → police_station_id
      center_code            → center_code_encoded  (str-cast before encoding)

    Returns:
        df: With new encoded columns added.
        encoders: Dict of fitted LabelEncoders (save these for inference).
    """
    encoders: dict[str, LabelEncoder] = {}

    encode_map: list[tuple[str, str]] = [
        ("violation_type_primary", "violation_type_primary_encoded"),
        ("vehicle_type",           "vehicle_type_encoded"),
        ("police_station",         "police_station_id"),
        ("center_code",            "center_code_encoded"),
    ]

    for src_col, dst_col in tqdm(encode_map, desc="  LabelEncoding", leave=False):
        le = LabelEncoder()
        # Cast to string to handle mixed types (center_code can be float strings)
        values = df[src_col].astype(str).fillna("UNKNOWN")
        df[dst_col] = le.fit_transform(values).astype("int16")
        encoders[dst_col] = le

    return df, encoders


def _save_encoders(
    encoders: dict[str, LabelEncoder],
    save_path: str | Path,
) -> None:
    """Pickle the encoder dict to disk."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(encoders, f)
    logger.info(f"Label encoders saved → '{out}'")


def _mode(series: pd.Series) -> Any:
    """Return mode of a series (for use in groupby.agg). Returns NaN if empty."""
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan
