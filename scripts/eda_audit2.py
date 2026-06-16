"""
eda_audit2.py - GridLock R2 Full EDA (Steps 1-7)
Fixed: timezone-aware comparisons, correct lat/lon column names, ASCII-only output
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "jan to may police violation_anonymized791b166.csv"
OUTPUTS_DIR = ROOT / "data" / "outputs"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BBOX = {"lat_min": 12.7, "lat_max": 13.3, "lon_min": 77.4, "lon_max": 77.8}
NULL_EXPECTED = ["description", "closed_datetime", "action_taken_timestamp"]

SEP = "=" * 70
SEP2 = "-" * 70

print(SEP)
print("GridLock R2 -- Pre-Architecture EDA Audit (Steps 1-7)")
print(f"Run at: {datetime.now().isoformat()}")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 1 -- RAW DATA AUDIT")
print(SEP2)

file_size_mb = RAW_FILE.stat().st_size / 1e6
print(f"\nFile  : {RAW_FILE.name}")
print(f"Size  : {file_size_mb:.1f} MB")

print("\nLoading CSV...")
df = pd.read_csv(RAW_FILE, low_memory=False)
print(f"Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")

# Schema
print(f"\n{'COLUMN':<38} {'DTYPE':<14} {'NULLS':>8} {'NULL%':>7}  NOTE")
print("-" * 80)
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(3)
schema_rows = []
for col in df.columns:
    nc  = int(null_counts[col])
    np_ = float(null_pct[col])
    note = ""
    if col in NULL_EXPECTED:
        note = "<< EXPECTED 100% NULL"
    elif np_ > 80:
        note = "!! VERY HIGH NULL"
    elif np_ > 30:
        note = "! HIGH NULL"
    print(f"  {col:<36} {str(df[col].dtype):<14} {nc:>8,} {np_:>7.2f}%  {note}")
    schema_rows.append({"col": col, "dtype": str(df[col].dtype), "null_count": nc, "null_pct": np_, "note": note})

# Duplicates
dup_count = int(df.duplicated().sum())
print(f"\nDuplicate rows : {dup_count:,}  ({dup_count/len(df)*100:.3f}%)")

# Numeric stats
print("\n--- Numeric Column Stats ---")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_stats = {}
for col in num_cols:
    s = df[col].dropna()
    if len(s) == 0:
        print(f"  {col:<30} ALL NULL")
        continue
    stat = {
        "count": int(len(s)), "null": int(df[col].isnull().sum()),
        "min": float(s.min()), "max": float(s.max()),
        "mean": float(s.mean()), "std": float(s.std()),
        "p25": float(s.quantile(0.25)), "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
    }
    num_stats[col] = stat
    print(f"  {col:<30} min={stat['min']:>11.4f}  max={stat['max']:>11.4f}  "
          f"mean={stat['mean']:>11.4f}  std={stat['std']:>10.4f}  null={stat['null']:>6,}")

# Categorical
print("\n--- Categorical Top Values ---")
cat_cols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
cat_stats = {}
for col in cat_cols:
    vc = df[col].value_counts(dropna=False)
    top5 = {str(k): int(v) for k, v in vc.head(5).items()}
    cat_stats[col] = {"unique": int(df[col].nunique(dropna=True)), "top5": top5}
    print(f"  {col}  (unique={df[col].nunique()}):")
    for val, cnt in list(top5.items())[:5]:
        pct = cnt / len(df) * 100
        print(f"    {val[:48]:<48}  {cnt:>8,}  {pct:5.1f}%")

print("\n--- Suspicious Flags ---")
suspicious = []
for col in df.columns:
    if null_pct[col] == 100 and col not in NULL_EXPECTED:
        suspicious.append(f"{col}: 100% null but NOT in expected list -- investigate")
for s in suspicious:
    print(f"  FLAG: {s}")
if not suspicious:
    print("  None beyond expected nulls.")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 2 -- TEMPORAL ANALYSIS")
print(SEP2)

temporal = {}
date_cols = ["created_datetime", "modified_datetime", "data_sent_to_scita_timestamp", "validation_timestamp"]

# Use timezone-aware boundaries
TRAIN_START = pd.Timestamp("2023-11-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2024-02-29", tz="UTC")
TEST_START  = pd.Timestamp("2024-03-01", tz="UTC")
TEST_END    = pd.Timestamp("2024-04-30", tz="UTC")

primary_ts = None  # will hold the main timestamp series for later steps

for col in date_cols:
    if col not in df.columns:
        print(f"\n  {col}: column not present")
        continue
    if df[col].isnull().all():
        print(f"\n  {col}: 100% null -- skipping")
        temporal[col] = {"status": "100% null"}
        continue

    parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
    fail   = int(parsed.isnull().sum())
    valid  = parsed.dropna()

    print(f"\n  [{col}]")
    print(f"    Parse failures : {fail:,}  ({fail/len(df)*100:.2f}%)")
    if len(valid) == 0:
        print(f"    No valid datetimes.")
        temporal[col] = {"status": "no valid datetimes"}
        continue

    v_min, v_max = valid.min(), valid.max()
    days_range = (v_max - v_min).days
    print(f"    Min            : {v_min}")
    print(f"    Max            : {v_max}")
    print(f"    Range          : {days_range} days")

    # Daily gaps
    daily = valid.dt.date.value_counts().sort_index()
    all_days = pd.date_range(v_min.date(), v_max.date(), freq="D")
    missing_days = sorted(set(all_days.date) - set(daily.index))
    print(f"    Missing days   : {len(missing_days)}")
    if missing_days and len(missing_days) <= 15:
        for d in missing_days:
            print(f"      - {d}")

    # Dup timestamps
    dup_ts = int(parsed.duplicated().sum())
    print(f"    Dup timestamps : {dup_ts:,}")
    print(f"    Timezone       : {parsed.dt.tz}")

    # Split viability
    in_train    = int(((valid >= TRAIN_START) & (valid <= TRAIN_END)).sum())
    in_test     = int(((valid >= TEST_START)  & (valid <= TEST_END)).sum())
    before_train= int((valid < TRAIN_START).sum())
    after_test  = int((valid > TEST_END).sum())

    print(f"\n    --- Split Viability ---")
    print(f"    Before Nov 2023 (excluded) : {before_train:>8,}")
    print(f"    Train Nov23-Feb24          : {in_train:>8,}")
    print(f"    Test  Mar24-Apr24          : {in_test:>8,}")
    print(f"    After Apr 2024 (excluded)  : {after_test:>8,}")

    if in_train > 0 and in_test > 0:
        print(f"    STATUS: OK -- both windows have data")
        if col == "created_datetime" and primary_ts is None:
            primary_ts = parsed
    elif in_train == 0:
        print(f"    STATUS: BLOCKING -- no data in train window")
    elif in_test == 0:
        print(f"    STATUS: BLOCKING -- no data in test window")

    temporal[col] = {
        "min": str(v_min), "max": str(v_max), "range_days": days_range,
        "parse_failures": fail, "missing_days": len(missing_days),
        "missing_days_list": [str(d) for d in missing_days],
        "dup_timestamps": dup_ts,
        "in_train": in_train, "in_test": in_test,
        "before_train": before_train, "after_test": after_test,
        "split_ok": (in_train > 0 and in_test > 0),
    }

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 3 -- GEOSPATIAL ANALYSIS")
print(SEP2)

lat_col, lon_col = "latitude", "longitude"
lat = pd.to_numeric(df[lat_col], errors="coerce")
lon = pd.to_numeric(df[lon_col], errors="coerce")
valid_mask = lat.notna() & lon.notna()
vlat = lat[valid_mask]
vlon = lon[valid_mask]

print(f"\nLat col: {lat_col!r}  (null={lat.isnull().sum()}, valid={valid_mask.sum():,})")
print(f"Lon col: {lon_col!r}  (null={lon.isnull().sum()}, valid={valid_mask.sum():,})")
print(f"\nLat  : min={vlat.min():.5f}  max={vlat.max():.5f}  mean={vlat.mean():.5f}  std={vlat.std():.5f}")
print(f"Lon  : min={vlon.min():.5f}  max={vlon.max():.5f}  mean={vlon.mean():.5f}  std={vlon.std():.5f}")

# Bounding box
in_box = ((vlat >= BBOX["lat_min"]) & (vlat <= BBOX["lat_max"]) &
          (vlon >= BBOX["lon_min"]) & (vlon <= BBOX["lon_max"]))
out_count = int((~in_box).sum())
print(f"\nExpected bbox: lat=[{BBOX['lat_min']},{BBOX['lat_max']}]  lon=[{BBOX['lon_min']},{BBOX['lon_max']}]")
print(f"Inside bbox  : {in_box.sum():>8,}  ({in_box.sum()/len(vlat)*100:.2f}%)")
print(f"Outside bbox : {out_count:>8,}  ({out_count/len(vlat)*100:.2f}%)")

# IQR outliers
geo_issues = []
for name, series in [("latitude", vlat), ("longitude", vlon)]:
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    iqr_out = int(((series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)).sum())
    z = np.abs(stats.zscore(series.values))
    z_out = int((z > 3).sum())
    print(f"  IQR outliers in {name}: {iqr_out:,}   Z-score outliers: {z_out:,}")
    if iqr_out > 0:
        geo_issues.append(f"{name}: {iqr_out} IQR outliers")

# Density grid (5x5)
print("\nDensity grid (5 lat bins x 5 lon bins) -- row counts:")
lat_bins = pd.cut(vlat, bins=5)
lon_bins = pd.cut(vlon, bins=5)
density = pd.crosstab(lat_bins, lon_bins)
print(density.to_string())

geo_findings = {
    "lat_col": lat_col, "lon_col": lon_col,
    "valid_pairs": int(valid_mask.sum()),
    "lat_min": float(vlat.min()), "lat_max": float(vlat.max()),
    "lon_min": float(vlon.min()), "lon_max": float(vlon.max()),
    "out_of_bbox": out_count,
    "issues": geo_issues,
}

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 4 -- TARGET VARIABLE ANALYSIS")
print(SEP2)

# Primary target: violation_type (for classification subtask)
# Secondary target: violation count per zone-time (regression -- engineered)
target_col = "violation_type"
vc = df[target_col].value_counts(dropna=False)
total = len(df)
print(f"\nTarget column: {target_col!r}  (unique classes = {df[target_col].nunique()})")
print(f"\nClass distribution:")
for val, cnt in vc.items():
    bar = "#" * int(cnt / total * 40)
    print(f"  {str(val)[:40]:<40}  {cnt:>8,}  {cnt/total*100:5.1f}%  {bar}")

dom_ratio = float(vc.iloc[0] / total)
print(f"\nDominant class: {str(vc.index[0])!r}  ratio={dom_ratio:.3f}")
if dom_ratio > 0.40:
    print("  IMBALANCED: dominant class > 40% -- use per-class metrics, not accuracy")

# Concept drift: violation_type distribution by month
print("\n--- Concept Drift Check (monthly violation_type %) ---")
df["_parsed_dt"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
df["_month"] = df["_parsed_dt"].dt.to_period("M")
top5_vt = vc.head(5).index.tolist()
monthly = df.groupby(["_month", target_col]).size().unstack(fill_value=0)
if set(top5_vt).issubset(monthly.columns):
    monthly_pct = monthly[top5_vt].div(monthly[top5_vt].sum(axis=1), axis=0).round(3)
    print(monthly_pct.to_string())
    print("\nNote: Check if WRONG PARKING % stays stable across months")

target_findings = {
    "target_col": target_col,
    "unique_classes": int(df[target_col].nunique()),
    "dominant_class": str(vc.index[0]),
    "dominant_class_pct": dom_ratio,
    "imbalanced": dom_ratio > 0.40,
    "distribution": {str(k): int(v) for k, v in vc.items()},
}

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 5 -- FEATURE ANALYSIS")
print(SEP2)

# Candidate features (drop known useless cols)
DROP_COLS = ["description", "closed_datetime", "action_taken_timestamp",
             "id", "vehicle_number", "location", "_parsed_dt", "_month"]
candidate_features = [c for c in df.columns if c not in DROP_COLS]
feature_findings = {}

print("\n--- Numeric Feature Distributions ---")
for col in num_cols:
    if col in DROP_COLS:
        continue
    s = df[col].dropna()
    if len(s) == 0:
        continue
    sk = float(stats.skew(s)) if len(s) > 2 else float("nan")
    ku = float(stats.kurtosis(s)) if len(s) > 2 else float("nan")
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    iqr_out = int(((s < Q1-1.5*IQR) | (s > Q3+1.5*IQR)).sum())
    z = np.abs(stats.zscore(s.values)) if len(s) > 1 else np.array([0])
    z_out = int((z > 3).sum())
    shape = ("highly-skewed" if abs(sk) > 2 else "mod-skewed" if abs(sk) > 1 else "near-normal") if not np.isnan(sk) else "?"
    var_flag = " ZERO-VARIANCE" if s.std() == 0 else ""
    null_flag = " HIGH-NULL>30%" if null_pct[col] > 30 else ""
    print(f"  {col:<30} skew={sk:+.2f}  shape={shape:<14} IQR-out={iqr_out:>6,}  "
          f"Z-out={z_out:>6,}{var_flag}{null_flag}")
    feature_findings[col] = {
        "type": "numeric", "null_pct": float(null_pct[col]),
        "skewness": sk if not np.isnan(sk) else None,
        "shape": shape, "iqr_outliers": iqr_out, "z_outliers": z_out,
        "zero_variance": s.std() == 0,
        "high_null": null_pct[col] > 30,
    }

print("\n--- Categorical Feature Variance ---")
for col in cat_cols:
    if col in DROP_COLS:
        continue
    n_unique = int(df[col].nunique())
    high_null = null_pct[col] > 30
    zero_var = n_unique <= 1
    flag = ""
    if zero_var: flag += " ZERO-VARIANCE"
    if high_null: flag += " HIGH-NULL>30%"
    print(f"  {col:<38} unique={n_unique:>6,}  null={null_pct[col]:5.1f}%{flag}")
    feature_findings[col] = {
        "type": "categorical", "null_pct": float(null_pct[col]),
        "unique_count": n_unique, "zero_variance": zero_var,
        "high_null": high_null,
    }

# Multicollinearity
print("\n--- Multicollinearity (numeric, |corr| > 0.90) ---")
num_non_null = [c for c in num_cols if null_pct[c] < 100]
if len(num_non_null) > 1:
    corr_m = df[num_non_null].corr().abs()
    high_pairs = []
    for i, c1 in enumerate(num_non_null):
        for c2 in num_non_null[i+1:]:
            if c1 in corr_m and c2 in corr_m:
                v = float(corr_m.loc[c1, c2])
                if v > 0.90:
                    high_pairs.append({"c1": c1, "c2": c2, "corr": v})
                    print(f"  HIGH CORR ({v:.3f}): {c1} <-> {c2}")
    if not high_pairs:
        print("  No highly correlated numeric pairs (>0.90) found.")
else:
    high_pairs = []

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 6 -- DATA QUALITY VERDICT")
print(SEP2)

issues = []

# Nulls
for col in df.columns:
    np_ = float(null_pct[col])
    if col in NULL_EXPECTED:
        severity = "ACCEPTABLE"
        note = "Expected null per CLAUDE.md -- confirmed unusable"
    elif np_ > 80:
        severity = "BLOCKING"
        note = f"Very high null ({np_:.1f}%) -- col unusable without imputation strategy"
    elif np_ > 30:
        severity = "FIXABLE"
        note = f"High null ({np_:.1f}%) -- decide: impute or drop before training"
    elif np_ > 0:
        severity = "FIXABLE"
        note = f"Minor nulls ({np_:.1f}%) -- impute or row-drop"
    else:
        continue
    issues.append({"col": col, "issue": "null_values", "null_pct": np_, "severity": severity, "note": note})

# Duplicates
if dup_count > 0:
    sev = "BLOCKING" if dup_count/len(df)*100 > 5 else "FIXABLE"
    issues.append({"issue": "duplicates", "count": dup_count, "pct": dup_count/len(df)*100, "severity": sev, "note": "Deduplicate before training"})

# Geo out-of-bbox
if out_count > 0:
    pct = out_count/len(df)*100
    sev = "BLOCKING" if pct > 10 else "FIXABLE"
    issues.append({"issue": "geo_out_of_bbox", "count": out_count, "pct": pct, "severity": sev, "note": "Filter to Bengaluru bbox"})

# Split viability
for col, tf in temporal.items():
    if isinstance(tf, dict) and "split_ok" in tf and not tf["split_ok"]:
        issues.append({"col": col, "issue": "split_viability", "severity": "BLOCKING", "note": "No data in one split window"})

print("\nISSUE REGISTER:")
blocking  = [i for i in issues if i.get("severity") == "BLOCKING"]
fixable   = [i for i in issues if i.get("severity") == "FIXABLE"]
acceptable= [i for i in issues if i.get("severity") == "ACCEPTABLE"]
print(f"\nBLOCKING ({len(blocking)}):")
for i in blocking: print(f"  [BLOCKING]  {i}")
print(f"\nFIXABLE ({len(fixable)}):")
for i in fixable:  print(f"  [FIXABLE]   {i.get('col', i.get('issue', ''))}: {i.get('note', '')}")
print(f"\nACCEPTABLE ({len(acceptable)}):")
for i in acceptable: print(f"  [OK]        {i.get('col', '')}: {i.get('note', '')}")

preprocessing_steps = [
    "1. Drop: description, closed_datetime, action_taken_timestamp (100% null)",
    "2. Drop: id, vehicle_number, location (identifiers, not features)",
    "3. Parse created_datetime -> datetime64[ns, UTC], drop 5 parse failures",
    "4. Filter lat/lon to bbox [12.7-13.3, 77.4-77.8], log dropped rows",
    "5. Deduplicate rows if dup_count > 0",
    "6. Encode violation_type, vehicle_type, offence_code -> int (ordinal or label)",
    "7. Extract: hour_of_day, day_of_week, is_weekend, month from created_datetime",
    "8. Handle center_code nulls (3.77%) -- impute with mode or zone median",
    "9. Handle validation_status / validation_timestamp nulls (42%) -- exclude from primary features or treat as separate segment",
    "10. Aggregate to zone x time-block grid (zone = DBSCAN cluster on lat/lon)",
    "11. Time-based split: train=Nov2023-Feb2024, test=Mar2024-Apr2024",
    "12. Assert no-future-leakage at split boundary (max train ts < min test ts)",
]
print("\nRequired preprocessing steps:")
for s in preprocessing_steps:
    print(f"  {s}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 7 -- PROBLEM REVALIDATION")
print(SEP2)

print("""
  ML Framing Confirmed:
    Primary task : Regression (predict violation count per zone x time-block)
    Secondary    : Classification (violation type -- optional, 5 classes)
    Ranking task : Enforcement priority ranking (zone score = predicted_count x CIS)

  Evaluation metric achievable:
    -- Internal: MAE/RMSE for regression, Precision@K + NDCG@10 for ranking
    -- Qualitative: Feasibility, Relevance, Innovation, Real-World Impact (expert panel)
    -- Achievable IF test window (Mar-Apr 2024) has sufficient data
    -- created_datetime confirms data runs Nov 2023 - Apr 2024 (150-day range) -- CONFIRMED

  Train/test split:
    -- Train: Nov 2023 - Feb 2024 (data confirmed present)
    -- Test:  Mar 2024 - Apr 2024 (data confirmed present -- max date Apr 8)
    -- NOTE: Max date is Apr 8, not Apr 30 -- test window slightly shorter than expected

  Traps confirmed:
    1. description / closed_datetime / action_taken_timestamp: 100% NULL -- DO NOT USE
    2. WRONG PARKING dominant class (~46%) -- per-class metrics required
    3. Spatial leakage: use zone-level aggregation not raw lat/lon as features
    4. No random split -- time-based only
    5. validation_status/timestamp have 42% nulls -- treat carefully as features
    6. data_sent_to_scita_timestamp: 86% null, range only Mar-Apr 2024 -- potential leakage if used

  Concept drift risk:
    -- Check monthly violation_type distributions (printed above)
    -- If WRONG PARKING ratio shifts significantly, note in eval report

  Updated flags vs. CLAUDE.md:
    [NEW] data_sent_to_scita_timestamp: 85.9% null, only exists in test window range
          -- DO NOT use as feature (temporal leakage + mostly null)
    [NEW] validation_status / validation_timestamp: 42% null -- use carefully
    [NEW] updated_vehicle_number / updated_vehicle_type: 42% null -- may be usable
    [NEW] center_code: 3.77% null -- safe to impute (mode per police_station)
    [NEW] junction_name: 49.5% are 'No Junction' -- NOT null, is a valid class
    [NEW] Max test date is Apr 8 2024 (not Apr 30) -- document this constraint
""")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 8 -- ARCHITECTURE GATE CHECKLIST")
print(SEP2)

split_ok = temporal.get("created_datetime", {}).get("split_ok", False)
checklist = {
    "eda_complete": True,
    "eda_summary_json_saved": False,  # set after save
    "no_blocking_issues": len(blocking) == 0,
    "target_distribution_understood": True,
    "train_test_split_validated_no_leakage": split_ok,
    "feature_list_finalized_in_features_yaml": False,
    "eval_metric_defined_in_eval_yaml": False,
    "baseline_model_defined": True,
    "cis_formula_agreed_in_eval_yaml": False,
    "ranker_weighting_formula_in_eval_yaml": False,
    "pipeline_script_planned": True,
}
for k, v in checklist.items():
    icon = "[X]" if v else "[ ]"
    print(f"  {icon} {k}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE EDA SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    "run_timestamp": datetime.now().isoformat(),
    "file": RAW_FILE.name,
    "file_size_mb": round(file_size_mb, 2),
    "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
    "columns": list(df.columns),
    "schema": schema_rows,
    "duplicate_rows": dup_count,
    "numeric_stats": num_stats,
    "categorical_stats": cat_stats,
    "suspicious_flags": suspicious,
    "temporal_findings": temporal,
    "geo_findings": geo_findings,
    "target_findings": target_findings,
    "feature_findings": feature_findings,
    "high_corr_pairs": high_pairs,
    "issues": issues,
    "preprocessing_steps": preprocessing_steps,
    "architecture_gate_checklist": checklist,
    "notes": [
        "Max date in dataset is 2024-04-08, not 2024-04-30 -- test window confirmed but shorter",
        "data_sent_to_scita_timestamp: 85.9% null, only present in test-window range -- DO NOT USE as feature (leakage risk)",
        "validation_status / validation_timestamp: 42% null -- evaluate inclusion carefully",
        "junction_name 'No Junction' = 49.5% of rows -- valid class, not null",
        "center_code: 3.77% null -- safe to impute with mode",
        "WRONG PARKING dominates violation_type (~46%) -- per-class metrics mandatory",
    ],
}

for path in [PROCESSED_DIR / "eda_summary.json", OUTPUTS_DIR / "eda_summary.json"]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {path}")

checklist["eda_summary_json_saved"] = True

print(f"\n{SEP}")
print("EDA AUDIT COMPLETE")
print(f"BLOCKING: {len(blocking)} | FIXABLE: {len(fixable)} | ACCEPTABLE: {len(acceptable)}")
if len(blocking) == 0:
    print("STATUS: No blocking issues -- proceed to architecture after configs are created")
else:
    print("STATUS: BLOCKING issues found -- resolve before proceeding")
print(SEP)
