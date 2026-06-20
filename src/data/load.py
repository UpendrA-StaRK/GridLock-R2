"""
src/data/load.py
GridLock R2 — PS1: Parking-Induced Congestion

Ingest the raw police violation CSV:
  1. Validate schema (calls validate.py — fails loudly on any breach)
  2. Cast dtypes (datetime → UTC, lat/lon → float64, bool columns)
  3. Drop excluded / leakage / identifier columns
  4. Log null summary for all retained columns
  5. Deduplicate per the confirmed rule:
       Drop ONLY when ALL of (latitude, longitude, violation_type, vehicle_type,
       created_datetime_minute) are identical.
       Same-second events at different lat/lon = real multi-violation events — KEEP.
  6. Return clean DataFrame + metadata dict

Pipeline protocol:
  - This module is imported and called from notebooks/01_eda.ipynb (cell by cell, user-run).
  - Do NOT add any training or feature engineering logic here.
  - All config paths are read from configs/eval.yaml and features.yaml — never hardcoded.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

from src.data.validate import (
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    NULL_COLUMNS,
    load_eval_config,
    save_validation_report,
    validate_schema,
)


# ── Column drop registry ──────────────────────────────────────────────────────
# Assembled from EDA findings. Source of truth is features.yaml (excluded section)
# but we also define it here so load.py is self-contained and fails loudly if
# a column that must be dropped is accidentally used upstream.
COLUMNS_TO_DROP: list[str] = (
    NULL_COLUMNS          # 100% null
    + LEAKAGE_COLUMNS     # temporal leakage / post-event
    + IDENTIFIER_COLUMNS  # id / vehicle_number / location
)


def load_raw(
    csv_path: str | Path,
    eval_config_path: str | Path = "configs/eval.yaml",
    save_report: bool = True,
    report_path: str | Path = "data/processed/validation_report.json",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load and validate the raw police violation CSV.

    Steps:
        1. Read CSV (all columns as object dtype initially)
        2. Run validate_schema() — raises ValueError on any schema breach
        3. Cast dtypes: created_datetime → UTC datetime64, lat/lon → float64,
           data_sent_to_scita → bool (Int8), violation_type kept as string for now
        4. Drop excluded columns (leakage + null + identifier)
        5. Log null summary for all retained columns
        6. Deduplicate (minute-level rule)
        7. Return (clean_df, metadata)

    Args:
        csv_path: Path to the raw CSV (e.g. "data/raw/jan to may police violation_anonymized791b166.csv").
        eval_config_path: Path to configs/eval.yaml (for split boundaries used in validation).
        save_report: If True, save the validation report JSON to report_path.
        report_path: Destination for validation_report.json.

    Returns:
        df: Clean DataFrame with dtypes cast and leakage columns removed.
        metadata: Dict with row counts, null summary, dedup stats, file hash, timestamps.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If schema validation fails (propagated from validate_schema).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: '{csv_path.resolve()}'. "
            f"Expected at: data/raw/jan to may police violation_anonymized791b166.csv"
        )

    metadata: dict[str, Any] = {
        "source_file": str(csv_path),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Step 1: Read CSV ─────────────────────────────────────────────────────
    logger.info(f"Reading CSV: '{csv_path}' ...")
    with tqdm(total=1, desc="Reading CSV", unit="file", leave=True) as pbar:
        df_raw = pd.read_csv(csv_path, dtype=str, low_memory=False)
        pbar.update(1)

    metadata["rows_raw"] = len(df_raw)
    metadata["cols_raw"] = len(df_raw.columns)
    metadata["file_hash_sha256"] = _sha256(csv_path)
    logger.info(
        f"Raw CSV loaded: {len(df_raw):,} rows × {len(df_raw.columns)} columns "
        f"| SHA-256: {metadata['file_hash_sha256'][:16]}..."
    )

    # ── Step 2: Schema validation ─────────────────────────────────────────────
    eval_cfg = load_eval_config(eval_config_path)
    report = validate_schema(df_raw, eval_cfg, strict=True)

    if save_report:
        save_validation_report(report, report_path)

    metadata["validation_passed"] = report["passed"]
    metadata["validation_warnings"] = report["warnings"]

    # ── Step 3: Cast dtypes ───────────────────────────────────────────────────
    logger.info("Casting dtypes ...")
    df = df_raw.copy()

    with tqdm(total=4, desc="Casting dtypes", unit="col-group", leave=True) as pbar:

        # created_datetime → UTC-aware datetime64[ns]
        df["created_datetime"] = pd.to_datetime(
            df["created_datetime"], errors="coerce", utc=True
        )
        # Drop the known ≤10 parse failures (EDA baseline = 5). Log count explicitly.
        n_dt_null = df["created_datetime"].isna().sum()
        if n_dt_null > 0:
            logger.warning(
                f"Dropping {n_dt_null} rows where created_datetime could not be parsed "
                f"(EDA baseline = 5 — documented in eda_summary.json)"
            )
            df = df[df["created_datetime"].notna()].copy()
        pbar.update(1)

        # latitude / longitude → float64
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        pbar.update(1)

        # data_sent_to_scita → int8 (0/1). 0% null confirmed in EDA.
        # Stored as int8 (not bool) to keep numpy-friendly for XGBoost.
        if "data_sent_to_scita" in df.columns:
            df["data_sent_to_scita"] = (
                df["data_sent_to_scita"]
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype("int8")
            )
        pbar.update(1)

        # center_code → keep as string (nullable category) — encoded later in features.py
        # violation_type → keep as string — parsed in features.py via ast.literal_eval
        # vehicle_type → keep as string — encoded in features.py
        # police_station → keep as string — encoded in features.py
        pbar.update(1)

    logger.info("✓ Dtypes cast")

    # ── Step 4: Drop excluded columns ────────────────────────────────────────
    cols_to_drop_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    cols_not_found = [c for c in COLUMNS_TO_DROP if c not in df.columns]

    if cols_not_found:
        logger.debug(f"Columns already absent (not in CSV): {cols_not_found}")

    with tqdm(total=1, desc="Dropping excluded columns", unit="op", leave=True) as pbar:
        df = df.drop(columns=cols_to_drop_present, errors="ignore")
        pbar.update(1)

    metadata["cols_dropped"] = cols_to_drop_present
    logger.info(
        f"✓ Dropped {len(cols_to_drop_present)} excluded columns: {cols_to_drop_present}"
    )
    logger.info(f"  Retained columns ({len(df.columns)}): {list(df.columns)}")

    # ── Step 5: Null summary (retained columns only) ──────────────────────────
    null_summary = _null_summary(df)
    metadata["null_summary"] = null_summary

    logger.info("Null summary (retained columns):")
    for col, stats in null_summary.items():
        if stats["null_count"] > 0:
            logger.warning(
                f"  ⚠  {col}: {stats['null_count']:,} nulls "
                f"({stats['null_pct']:.1f}%)"
            )
        else:
            logger.debug(f"  ✓ {col}: 0 nulls")

    # ── Step 6: Deduplication ─────────────────────────────────────────────────
    # Rule (EDA-confirmed):
    #   Deduplicate ONLY when ALL of (latitude, longitude, violation_type,
    #   vehicle_type, created_datetime rounded to minute) are identical.
    #   Same-second events at different lat/lon = real multi-violation events — KEEP.
    logger.info("Deduplicating (minute-level rule) ...")
    rows_before_dedup = len(df)

    with tqdm(total=2, desc="Deduplication", unit="step", leave=True) as pbar:
        df["_created_minute"] = df["created_datetime"].dt.floor("min")
        pbar.update(1)

        dedup_keys = ["latitude", "longitude", "violation_type", "vehicle_type", "_created_minute"]
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
        df = df.drop(columns=["_created_minute"])
        pbar.update(1)

    rows_after_dedup = len(df)
    rows_dropped_dedup = rows_before_dedup - rows_after_dedup

    metadata["rows_before_dedup"] = rows_before_dedup
    metadata["rows_after_dedup"] = rows_after_dedup
    metadata["rows_dropped_dedup"] = rows_dropped_dedup

    logger.info(
        f"✓ Deduplication complete: {rows_before_dedup:,} → {rows_after_dedup:,} rows "
        f"({rows_dropped_dedup:,} exact duplicates removed)"
    )

    # ── IQR + Z-score outlier detection on lat/lon (I-1: AGENTS.md mandate) ────
    import numpy as np
    for _col in ["latitude", "longitude"]:
        if _col in df.columns:
            q1, q3 = df[_col].quantile(0.25), df[_col].quantile(0.75)
            iqr = q3 - q1
            n_iqr = int(((df[_col] < q1 - 1.5 * iqr) | (df[_col] > q3 + 1.5 * iqr)).sum())
            z_scores = (df[_col] - df[_col].mean()) / df[_col].std()
            n_z = int((z_scores.abs() > 3).sum())
            if n_iqr > 0 or n_z > 0:
                logger.warning(f"  ⚠ {_col}: IQR outliers={n_iqr:,}, Z-score>3={n_z:,} (logged; not dropped)")
            else:
                logger.debug(f"  ✓ {_col}: no IQR/Z-score outliers")


    # ── Final metadata ────────────────────────────────────────────────────────
    metadata["rows_final"] = len(df)
    metadata["cols_final"] = len(df.columns)
    metadata["columns_retained"] = list(df.columns)

    # Row counts by split window (informational — actual split done in train.py)
    if "created_datetime" in df.columns:
        split = eval_cfg.get("split", {})
        train_end = pd.Timestamp(split.get("train_end", "2024-02-29"), tz="UTC")
        test_start = pd.Timestamp(split.get("test_start", "2024-03-01"), tz="UTC")
        test_end = pd.Timestamp(split.get("test_end", "2024-04-08"), tz="UTC")

        n_train = int((df["created_datetime"] <= train_end).sum())
        n_test = int(
            ((df["created_datetime"] >= test_start) & (df["created_datetime"] <= test_end)).sum()
        )
        metadata["preview_train_rows"] = n_train
        metadata["preview_test_rows"] = n_test

        logger.info(
            f"Split preview → train (≤{train_end.date()}): {n_train:,} rows | "
            f"test ({test_start.date()}–{test_end.date()}): {n_test:,} rows"
        )

    logger.info(
        f"─── load_raw() complete: {len(df):,} rows × {len(df.columns)} columns ───"
    )
    return df, metadata


def save_load_metadata(
    metadata: dict[str, Any],
    output_path: str | Path = "data/processed/load_metadata.json",
) -> None:
    """
    Persist the load metadata dict to disk as JSON.

    Args:
        metadata: Output metadata dict from load_raw().
        output_path: Destination JSON path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Load metadata saved → '{out}'")


# ── Private helpers ───────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file for reproducibility tracking."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _null_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Compute per-column null statistics.

    Returns:
        Dict mapping column name → {null_count, null_pct, dtype}.
    """
    n = len(df)
    summary: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        summary[col] = {
            "null_count": null_count,
            "null_pct": round(null_count / n * 100, 2) if n > 0 else 0.0,
            "dtype": str(df[col].dtype),
        }
    return summary
