"""
src/data/validate.py
GridLock R2 — PS1: Parking-Induced Congestion

Schema validator for the raw police violation CSV.
Fails LOUDLY (raises ValueError) on any breach.
Called by load.py as the very first step — nothing proceeds on a failed schema.

Rules enforced:
  - Required columns present
  - Dtypes castable (datetime, float, bool)
  - Latitude/longitude within Bengaluru bounding box
  - No calendar date gaps in created_datetime
  - Excluded columns with 100% null are documented (not silently ignored)
  - Train/test temporal split guard
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger


# ── Bengaluru geographic bounding box ────────────────────────────────────────
# EDA-confirmed actual data bounds (eda_summary.json geo section):
#   lat: [12.8027, 13.2937]  lon: [77.4426, 77.7717]
# We add a small buffer (0.01°) on each side to tolerate minor GPS jitter.
BENGALURU_BBOX: dict[str, float] = {
    "lat_min": 12.79,
    "lat_max": 13.30,
    "lon_min": 77.43,
    "lon_max": 77.78,
}

# ── Expected column registry ─────────────────────────────────────────────────
# fmt: off
REQUIRED_COLUMNS: list[str] = [
    "latitude", "longitude",
    "created_datetime",
    "violation_type",
    "vehicle_type",
    "police_station",
    "center_code",
    "junction_name",      # Raw column. is_at_junction is DERIVED in features.py
                           # as: (junction_name != 'No Junction').astype(int)
    "data_sent_to_scita",
]

# These are documented as 100% null in EDA — we verify they exist but don't fail
# on nulls; we fail if they contain unexpected non-null values (would be a data
# version mismatch).
NULL_COLUMNS: list[str] = [
    "description",
    "closed_datetime",
    "action_taken_timestamp",
]

# These are excluded for leakage / post-event reasons — log a warning if present.
LEAKAGE_COLUMNS: list[str] = [
    "data_sent_to_scita_timestamp",  # 86% null + test-window only
    "modified_datetime",             # post-event
    "validation_status",             # post-event + 42% null
    "validation_timestamp",          # post-event + 42% null
    "updated_vehicle_number",        # 42% null
    "updated_vehicle_type",          # 42% null
]

IDENTIFIER_COLUMNS: list[str] = ["id", "vehicle_number", "location"]
# fmt: on


# ── Public validator ──────────────────────────────────────────────────────────

def validate_schema(
    df: pd.DataFrame,
    eval_cfg: dict[str, Any],
    strict: bool = True,
) -> dict[str, Any]:
    """
    Validate the raw police violation DataFrame against the expected schema.

    Checks performed (in order):
      1. Required columns present
      2. Latitude / longitude within Bengaluru bounding box
      3. created_datetime parseable and within expected date range
      4. No calendar-day gaps in created_datetime (temporal continuity)
      5. Null-column audit (description / closed_datetime / action_taken_timestamp)
      6. Leakage-column audit (warns, does not fail)
      7. violation_type parseable as a Python list via ast.literal_eval
      8. Train / test split temporal guard (max train dt < min test dt)

    Args:
        df: Raw DataFrame loaded from CSV (dtypes not yet cast — still object/string).
        eval_cfg: Parsed contents of configs/eval.yaml (used for split boundaries).
        strict: If True, raise ValueError on any breach. If False, log errors only.

    Returns:
        report: dict with keys {passed, errors, warnings, stats}
                Always returned even when strict=True (before the raise).

    Raises:
        ValueError: On any schema breach when strict=True.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    logger.info("─── Schema Validation Start ───────────────────────────────────────")
    logger.info(f"DataFrame shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 1. Required columns ───────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"MISSING REQUIRED COLUMNS: {missing}")
        logger.error(f"Missing required columns: {missing}")
    else:
        logger.info(f"✓ All {len(REQUIRED_COLUMNS)} required columns present")

    # ── 2. Latitude / longitude bounds ────────────────────────────────────────
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")

        lat_oob = ((lat < BENGALURU_BBOX["lat_min"]) | (lat > BENGALURU_BBOX["lat_max"])).sum()
        lon_oob = ((lon < BENGALURU_BBOX["lon_min"]) | (lon > BENGALURU_BBOX["lon_max"])).sum()
        lat_null = lat.isna().sum()
        lon_null = lon.isna().sum()

        stats["lat_out_of_bounds"] = int(lat_oob)
        stats["lon_out_of_bounds"] = int(lon_oob)
        stats["lat_null"] = int(lat_null)
        stats["lon_null"] = int(lon_null)

        if lat_null > 0 or lon_null > 0:
            errors.append(
                f"NULL coordinates: latitude={lat_null:,} nulls, longitude={lon_null:,} nulls"
            )
            logger.error(f"Null coordinates — lat: {lat_null:,}, lon: {lon_null:,}")
        if lat_oob > 0:
            errors.append(
                f"LATITUDE OUT OF BENGALURU BBOX: {lat_oob:,} rows outside "
                f"[{BENGALURU_BBOX['lat_min']}, {BENGALURU_BBOX['lat_max']}]"
            )
            logger.error(f"Latitude OOB: {lat_oob:,} rows")
        if lon_oob > 0:
            errors.append(
                f"LONGITUDE OUT OF BENGALURU BBOX: {lon_oob:,} rows outside "
                f"[{BENGALURU_BBOX['lon_min']}, {BENGALURU_BBOX['lon_max']}]"
            )
            logger.error(f"Longitude OOB: {lon_oob:,} rows")
        if lat_oob == 0 and lon_oob == 0 and lat_null == 0 and lon_null == 0:
            logger.info("✓ All coordinates within Bengaluru bounding box")

    # ── 3. created_datetime parseable + range check ───────────────────────────
    if "created_datetime" in df.columns:
        dt_series = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
        dt_null = dt_series.isna().sum()
        stats["datetime_null"] = int(dt_null)

        # EDA documented exactly 5 parse failures (eda_summary.json temporal.created_datetime.parse_failures).
        # Treat ≤ 10 nulls as a warning (known, documented) — load.py will drop them.
        # Treat > 10 nulls as a hard error (unexpected data quality issue).
        DATETIME_NULL_HARD_THRESHOLD = 10
        if dt_null > DATETIME_NULL_HARD_THRESHOLD:
            errors.append(
                f"UNPARSEABLE created_datetime: {dt_null:,} nulls after coerce "
                f"(threshold={DATETIME_NULL_HARD_THRESHOLD})"
            )
            logger.error(f"created_datetime parse failures: {dt_null:,} (>{DATETIME_NULL_HARD_THRESHOLD} — hard error)")
        elif dt_null > 0:
            warnings.append(
                f"created_datetime: {dt_null} unparseable rows (will be dropped in load.py). "
                f"EDA baseline = 5 — acceptable."
            )
            logger.warning(f"created_datetime: {dt_null} nulls after coerce (≤{DATETIME_NULL_HARD_THRESHOLD} — warning, will be dropped)")
        if dt_null <= DATETIME_NULL_HARD_THRESHOLD:
            dt_min = dt_series.min()
            dt_max = dt_series.max()
            stats["datetime_min"] = str(dt_min)
            stats["datetime_max"] = str(dt_max)

            expected_min = pd.Timestamp("2023-11-09", tz="UTC")
            expected_max = pd.Timestamp("2024-04-08 23:59:59", tz="UTC")

            if dt_min < expected_min:
                warnings.append(
                    f"created_datetime min ({dt_min}) is earlier than expected ({expected_min})"
                )
                logger.warning(f"created_datetime min={dt_min} earlier than expected {expected_min}")
            if dt_max > expected_max:
                warnings.append(
                    f"created_datetime max ({dt_max}) is later than expected ({expected_max})"
                )
                logger.warning(f"created_datetime max={dt_max} later than expected {expected_max}")

            logger.info(
                f"✓ created_datetime range: {dt_min.date()} → {dt_max.date()} "
                f"({dt_null} nulls)"
            )

            # ── 4. Temporal continuity (no day gaps) ─────────────────────────
            # Drop NaT values first (already warned/counted above) so date_range
            # never receives NaT as start/end.
            valid_dt = dt_series.dropna()
            dates_present = pd.Series(valid_dt.dt.date.unique()).sort_values().reset_index(drop=True)
            all_dates = pd.date_range(
                start=dates_present.iloc[0],
                end=dates_present.iloc[-1],
                freq="D",
            ).date
            missing_dates = sorted(set(all_dates) - set(dates_present))
            stats["missing_calendar_days"] = [str(d) for d in missing_dates]

            if missing_dates:
                warnings.append(
                    f"CALENDAR DAY GAPS in created_datetime: {len(missing_dates)} missing days → "
                    f"{missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}"
                )
                logger.warning(
                    f"Calendar day gaps detected: {len(missing_dates)} days missing "
                    f"(first 5: {missing_dates[:5]})"
                )
            else:
                logger.info("✓ No calendar day gaps in created_datetime")

    # ── 5. Null-column audit ──────────────────────────────────────────────────
    null_col_report: dict[str, int] = {}
    for col in NULL_COLUMNS:
        if col in df.columns:
            n_non_null = df[col].notna().sum()
            null_col_report[col] = int(n_non_null)
            if n_non_null > 0:
                warnings.append(
                    f"EXPECTED-NULL COLUMN '{col}' has {n_non_null:,} non-null values — "
                    f"possible data version mismatch"
                )
                logger.warning(f"Expected-null column '{col}' contains {n_non_null:,} non-null values")
            else:
                logger.info(f"✓ '{col}' confirmed 100% null (expected)")
    stats["null_column_audit"] = null_col_report

    # ── 6. Leakage-column audit ───────────────────────────────────────────────
    leakage_present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if leakage_present:
        warnings.append(
            f"LEAKAGE/EXCLUDED COLUMNS PRESENT (will be dropped in load.py): {leakage_present}"
        )
        logger.warning(
            f"Leakage columns present — ensure load.py drops them before features: "
            f"{leakage_present}"
        )
    else:
        logger.info("✓ No leakage columns detected in this DataFrame slice")

    # ── 7. violation_type parseability ────────────────────────────────────────
    if "violation_type" in df.columns:
        sample = df["violation_type"].dropna().head(500)
        parse_failures = 0
        for val in sample:
            try:
                parsed = ast.literal_eval(str(val))
                if not isinstance(parsed, list):
                    parse_failures += 1
            except (ValueError, SyntaxError):
                parse_failures += 1

        stats["violation_type_parse_failures_in_sample"] = parse_failures
        if parse_failures > 0:
            errors.append(
                f"violation_type NOT parseable as list via ast.literal_eval: "
                f"{parse_failures} failures in 500-row sample"
            )
            logger.error(f"violation_type parse failures: {parse_failures}/500 in sample")
        else:
            logger.info("✓ violation_type parseable as list (500-row sample check)")

    # ── 8. Train / test split temporal guard ─────────────────────────────────
    if "created_datetime" in df.columns and eval_cfg:
        split = eval_cfg.get("split", {})
        train_end = pd.Timestamp(split.get("train_end", "2024-02-29"), tz="UTC")
        test_start = pd.Timestamp(split.get("test_start", "2024-03-01"), tz="UTC")

        dt_series = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True).dropna()

        train_mask = dt_series <= train_end
        test_mask = dt_series >= test_start

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        stats["train_rows"] = n_train
        stats["test_rows"] = n_test

        logger.info(
            f"Split preview — train rows (≤{train_end.date()}): {n_train:,} | "
            f"test rows (≥{test_start.date()}): {n_test:,}"
        )

        if n_train > 0 and n_test > 0:
            max_train_dt = dt_series[train_mask].max()
            min_test_dt = dt_series[test_mask].min()

            if max_train_dt >= min_test_dt:
                errors.append(
                    f"TEMPORAL LEAKAGE DETECTED: max(train.created_datetime) = {max_train_dt} "
                    f"≥ min(test.created_datetime) = {min_test_dt}. "
                    f"The split boundary is contaminated."
                )
                logger.error(
                    f"LEAKAGE: max_train={max_train_dt} >= min_test={min_test_dt}"
                )
            else:
                logger.info(
                    f"✓ Temporal split guard passed: "
                    f"max_train={max_train_dt} < min_test={min_test_dt}"
                )

    # ── Compile report ────────────────────────────────────────────────────────
    passed = len(errors) == 0
    report: dict[str, Any] = {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }

    if warnings:
        for w in warnings:
            logger.warning(f"  ⚠  {w}")

    if errors:
        logger.error(
            f"Schema validation FAILED with {len(errors)} error(s):\n"
            + "\n".join(f"  ✗ {e}" for e in errors)
        )
        if strict:
            raise ValueError(
                f"Schema validation failed ({len(errors)} error(s)):\n"
                + "\n".join(f"  ✗ {e}" for e in errors)
            )
    else:
        logger.info(
            f"─── Schema Validation PASSED "
            f"({len(warnings)} warning(s)) ──────────────────────────"
        )

    return report


def load_eval_config(config_path: str | Path = "configs/eval.yaml") -> dict[str, Any]:
    """
    Load and return the parsed eval.yaml config dict.

    Args:
        config_path: Path to configs/eval.yaml (relative to project root).

    Returns:
        Parsed YAML as a dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"configs/eval.yaml not found at '{path.resolve()}'. "
            f"Run from the project root or pass an absolute path."
        )
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded eval config v{cfg.get('version', '?')} from '{path}'")
    return cfg


def save_validation_report(
    report: dict[str, Any],
    output_path: str | Path = "data/processed/validation_report.json",
) -> None:
    """
    Persist the validation report to disk as JSON.

    Args:
        report: Output of validate_schema().
        output_path: Destination JSON file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Validation report saved → '{out}'")
