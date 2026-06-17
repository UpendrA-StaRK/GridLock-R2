"""
eda_audit.py — GridLock R2 Pre-Architecture EDA
Covers STEPS 1–7 from the architecture gate checklist.
Outputs: data/outputs/eda_summary.json
Run: python scripts/eda_audit.py
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "jan to may police violation_anonymized791b166.csv"
OUTPUTS_DIR = ROOT / "data" / "outputs"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = PROCESSED_DIR / "eda_summary.json"

# ── Expected schema from CLAUDE.md ────────────────────────────────────────────
EXPECTED_BBOX = {"lat_min": 12.7, "lat_max": 13.2, "lon_min": 77.4, "lon_max": 77.8}
TRAIN_END   = pd.Timestamp("2024-02-29")
TEST_START  = pd.Timestamp("2024-03-01")
TEST_END    = pd.Timestamp("2024-04-30")

NULL_COLS_EXPECTED = ["description", "closed_datetime", "action_taken_timestamp"]

print("=" * 70)
print("GridLock R2 — Pre-Architecture EDA Audit")
print(f"Run at: {datetime.now().isoformat()}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — RAW DATA AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 1 — RAW DATA AUDIT")
print("─" * 70)

print(f"\nFile: {RAW_FILE.name}")
file_size_mb = RAW_FILE.stat().st_size / (1024 * 1024)
print(f"Size: {file_size_mb:.2f} MB")

# Load with low_memory=False to avoid mixed-type warnings
print("\nLoading CSV … (may take a moment for 100+ MB file)")
df = pd.read_csv(RAW_FILE, low_memory=False)

print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn names ({df.shape[1]}):")
for c in df.columns:
    print(f"  {c!r}")

# ── Dtypes ────────────────────────────────────────────────────────────────────
print("\n--- Dtypes ---")
dtype_map = {}
for col in df.columns:
    dtype_map[col] = str(df[col].dtype)
    print(f"  {col:45s} {str(df[col].dtype)}")

# ── Null audit ────────────────────────────────────────────────────────────────
print("\n--- Null Audit ---")
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(2)
null_df = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
null_df = null_df[null_df["null_count"] > 0].sort_values("null_pct", ascending=False)
if null_df.empty:
    print("  No nulls found.")
else:
    for col, row in null_df.iterrows():
        flag = " ⚠️  EXPECTED-NULL" if col in NULL_COLS_EXPECTED else (
               " 🚨 HIGH NULL" if row["null_pct"] > 30 else "")
        print(f"  {col:45s} {row['null_count']:>8,}  ({row['null_pct']:6.2f}%){flag}")

# ── Duplicate rows ─────────────────────────────────────────────────────────────
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count:,} ({dup_count/len(df)*100:.3f}%)")

# ── Numeric columns — stats ───────────────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n--- Numeric Columns ({len(num_cols)}) — Descriptive Stats ---")
num_stats = {}
for col in num_cols:
    s = df[col].dropna()
    stat = {
        "count": int(s.count()),
        "null_count": int(df[col].isnull().sum()),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
    }
    num_stats[col] = stat
    print(f"  {col:45s} min={stat['min']:>12.4f}  max={stat['max']:>12.4f}  "
          f"mean={stat['mean']:>12.4f}  std={stat['std']:>12.4f}")

# ── Categorical columns — unique value counts ──────────────────────────────────
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"\n--- Categorical Columns ({len(cat_cols)}) — Unique Value Counts ---")
cat_stats = {}
for col in cat_cols:
    vc = df[col].value_counts(dropna=False)
    top5 = vc.head(5).to_dict()
    cat_stats[col] = {
        "unique_count": int(df[col].nunique(dropna=True)),
        "top_values": {str(k): int(v) for k, v in top5.items()},
    }
    print(f"  {col:45s} unique={df[col].nunique(dropna=True):>8,}")
    for val, cnt in list(top5.items())[:5]:
        print(f"      {str(val)[:50]:50s} {cnt:>8,}  ({cnt/len(df)*100:5.1f}%)")

# ── Suspicious column flags ───────────────────────────────────────────────────
suspicious = []
for col in df.columns:
    if null_pct.get(col, 0) == 100:
        suspicious.append(f"{col}: 100% null — completely empty column")
    elif col in NULL_COLS_EXPECTED and null_pct.get(col, 0) < 90:
        suspicious.append(f"{col}: Expected ~100% null but only {null_pct[col]:.1f}% — investigate")
print("\n--- Suspicious / Flagged Columns ---")
if suspicious:
    for s in suspicious:
        print(f"  🚨 {s}")
else:
    print("  None flagged.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 2 — TEMPORAL ANALYSIS")
print("─" * 70)

# Identify date columns (object cols that parse as datetime)
date_cols_found = []
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower() or "datetime" in col.lower():
        date_cols_found.append(col)

print(f"\nDate/time candidate columns: {date_cols_found}")

temporal_findings = {}

for col in date_cols_found:
    if df[col].isnull().all():
        print(f"\n  {col}: 100% null — skipping")
        temporal_findings[col] = {"status": "100% null — skipped"}
        continue

    try:
        parsed = pd.to_datetime(df[col], errors="coerce")
        parse_fail = parsed.isnull().sum()
        valid = parsed.dropna()
        print(f"\n  Column: {col}")
        print(f"    Parse failures: {parse_fail:,} ({parse_fail/len(df)*100:.2f}%)")
        if len(valid) == 0:
            print(f"    No valid datetimes found — skipping")
            temporal_findings[col] = {"status": "no valid datetimes"}
            continue
        print(f"    Min date:  {valid.min()}")
        print(f"    Max date:  {valid.max()}")
        date_range_days = (valid.max() - valid.min()).days
        print(f"    Range: {date_range_days} days")

        # Date gaps (daily)
        daily_counts = valid.dt.date.value_counts().sort_index()
        all_days = pd.date_range(valid.min().date(), valid.max().date(), freq="D")
        missing_days = set(all_days.date) - set(daily_counts.index)
        print(f"    Missing days in range: {len(missing_days)}")
        if missing_days and len(missing_days) <= 20:
            for d in sorted(missing_days):
                print(f"      Missing: {d}")

        # Duplicate timestamps
        dup_ts = parsed.duplicated().sum()
        print(f"    Duplicate timestamps: {dup_ts:,}")

        # Timezone check
        tz_info = parsed.dt.tz
        print(f"    Timezone: {tz_info}")

        # Check train/test split viability
        in_train = ((valid >= "2023-11-01") & (valid <= TRAIN_END)).sum()
        in_test  = ((valid >= TEST_START)   & (valid <= TEST_END)).sum()
        before_train = (valid < "2023-11-01").sum()
        after_test   = (valid > TEST_END).sum()

        print(f"\n    --- Split Viability ---")
        print(f"    Before Nov 2023 (excluded):    {before_train:>8,}")
        print(f"    Train window (Nov23–Feb24):    {in_train:>8,}")
        print(f"    Test window  (Mar24–Apr24):    {in_test:>8,}")
        print(f"    After Apr 2024 (excluded):     {after_test:>8,}")

        # Leakage check
        if in_train > 0 and in_test > 0:
            print(f"    ✅ Both train and test windows have data — split viable")
        elif in_train == 0:
            print(f"    🚨 NO data in train window — BLOCKING: split must be reconsidered")
        elif in_test == 0:
            print(f"    🚨 NO data in test window — BLOCKING: split must be reconsidered")

        temporal_findings[col] = {
            "min": str(valid.min()),
            "max": str(valid.max()),
            "range_days": date_range_days,
            "missing_days": len(missing_days),
            "dup_timestamps": int(dup_ts),
            "parse_failures": int(parse_fail),
            "in_train_window": int(in_train),
            "in_test_window": int(in_test),
        }
    except Exception as e:
        print(f"  ERROR processing {col}: {e}")
        temporal_findings[col] = {"status": f"error: {e}"}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — GEOSPATIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 3 — GEOSPATIAL ANALYSIS")
print("─" * 70)

geo_findings = {}
lat_col = lon_col = None

# Auto-detect lat/lon columns
for col in df.columns:
    cl = col.lower()
    if any(x in cl for x in ["lat", "latitude"]):
        lat_col = col
    if any(x in cl for x in ["lon", "lng", "longitude"]):
        lon_col = col

print(f"\nDetected: lat_col={lat_col!r}, lon_col={lon_col!r}")

if lat_col and lon_col:
    lat = pd.to_numeric(df[lat_col], errors="coerce").dropna()
    lon = pd.to_numeric(df[lon_col], errors="coerce").dropna()

    # Valid mask (both not null)
    mask = df[lat_col].notna() & df[lon_col].notna()
    coords = df[mask][[lat_col, lon_col]].apply(pd.to_numeric, errors="coerce").dropna()

    print(f"\nValid coordinate pairs: {len(coords):,} / {len(df):,}")
    print(f"\nLat  — min:{coords[lat_col].min():.5f}  max:{coords[lat_col].max():.5f}  "
          f"mean:{coords[lat_col].mean():.5f}  std:{coords[lat_col].std():.5f}")
    print(f"Lon  — min:{coords[lon_col].min():.5f}  max:{coords[lon_col].max():.5f}  "
          f"mean:{coords[lon_col].mean():.5f}  std:{coords[lon_col].std():.5f}")

    # Bounding box check
    bbox = EXPECTED_BBOX
    in_box = (
        (coords[lat_col] >= bbox["lat_min"]) & (coords[lat_col] <= bbox["lat_max"]) &
        (coords[lon_col] >= bbox["lon_min"]) & (coords[lon_col] <= bbox["lon_max"])
    )
    out_of_box = (~in_box).sum()
    print(f"\nExpected bounding box: lat [{bbox['lat_min']}, {bbox['lat_max']}], "
          f"lon [{bbox['lon_min']}, {bbox['lon_max']}]")
    print(f"Points INSIDE bounding box:   {in_box.sum():>8,} ({in_box.sum()/len(coords)*100:.2f}%)")
    print(f"Points OUTSIDE bounding box:  {out_of_box:>8,} ({out_of_box/len(coords)*100:.2f}%)")

    if out_of_box > 0:
        outlier_coords = coords[~in_box]
        print("\n  Sample out-of-box coordinates (up to 10):")
        for _, row in outlier_coords.head(10).iterrows():
            print(f"    lat={row[lat_col]:.5f}, lon={row[lon_col]:.5f}")

    # IQR outlier detection on lat/lon
    for col_name, series in [(lat_col, coords[lat_col]), (lon_col, coords[lon_col])]:
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
        print(f"  IQR outliers in {col_name}: {outlier_mask.sum():,}")

    # Density summary: grid bin counts
    lat_bins = pd.cut(coords[lat_col], bins=5)
    lon_bins = pd.cut(coords[lon_col], bins=5)
    density_grid = pd.crosstab(lat_bins, lon_bins)
    print(f"\nDensity grid (5×5 bins — row=lat, col=lon):")
    print(density_grid.to_string())

    geo_findings = {
        "lat_col": lat_col,
        "lon_col": lon_col,
        "valid_pairs": int(len(coords)),
        "out_of_bbox_count": int(out_of_box),
        "lat_min": float(coords[lat_col].min()),
        "lat_max": float(coords[lat_col].max()),
        "lon_min": float(coords[lon_col].min()),
        "lon_max": float(coords[lon_col].max()),
    }
else:
    print("  🚨 No lat/lon columns found — BLOCKING")
    geo_findings = {"status": "lat/lon not found"}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TARGET VARIABLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 4 — TARGET VARIABLE ANALYSIS")
print("─" * 70)

target_findings = {}

# Primary target: violation_type (and/or violation count per zone×time)
# Look for violation type / subtype columns
viol_cols = [c for c in df.columns if "violat" in c.lower() or "offence" in c.lower()
             or "offense" in c.lower() or "type" in c.lower()]
print(f"\nViolation-related candidate columns: {viol_cols}")

for col in viol_cols:
    if col not in df.columns:
        continue
    vc = df[col].value_counts(dropna=False)
    total = len(df)
    top_class = vc.index[0] if len(vc) > 0 else None
    top_count = vc.iloc[0] if len(vc) > 0 else 0
    imbalance_ratio = top_count / total if total > 0 else 0

    print(f"\n  Target candidate: {col!r}")
    print(f"  Unique classes: {df[col].nunique()}")
    print(f"  Class distribution (top 10):")
    for val, cnt in vc.head(10).items():
        bar = "█" * int(cnt / total * 50)
        print(f"    {str(val)[:40]:40s} {cnt:>8,} ({cnt/total*100:5.1f}%) {bar}")
    print(f"  Dominant class ratio: {imbalance_ratio:.3f}")
    if imbalance_ratio > 0.40:
        print(f"  ⚠️  IMBALANCED — dominant class > 40%")

    target_findings[col] = {
        "unique_classes": int(df[col].nunique()),
        "dominant_class": str(top_class),
        "dominant_class_count": int(top_count),
        "dominant_class_ratio": float(imbalance_ratio),
        "imbalanced": bool(imbalance_ratio > 0.40),
    }

# Temporal drift check for target
print("\n--- Concept Drift Check (violation type distribution over time) ---")
if date_cols_found:
    primary_date_col = date_cols_found[0]
    if not df[primary_date_col].isnull().all():
        df["_parsed_date"] = pd.to_datetime(df[primary_date_col], errors="coerce")
        df["_month"] = df["_parsed_date"].dt.to_period("M")

        for col in viol_cols[:2]:  # limit to first 2
            if df[col].nunique() < 20:
                monthly = df.groupby(["_month", col]).size().unstack(fill_value=0)
                monthly_pct = monthly.div(monthly.sum(axis=1), axis=0)
                print(f"\n  {col} — monthly % (top 3 classes):")
                top3 = df[col].value_counts().head(3).index.tolist()
                if top3:
                    sub = monthly_pct[top3].dropna()
                    print(sub.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 5 — FEATURE ANALYSIS")
print("─" * 70)

feature_findings = {}

candidate_features = [c for c in df.columns if c not in NULL_COLS_EXPECTED
                      and c not in ["_parsed_date", "_month"]]

print(f"\nCandidate features for analysis: {len(candidate_features)}")

for col in candidate_features:
    f = {"col": col, "dtype": str(df[col].dtype)}

    null_p = null_pct.get(col, 0.0)
    f["null_pct"] = float(null_p)

    if null_p > 30:
        f["flag_high_null"] = True
        print(f"\n  🚨 HIGH NULL ({null_p:.1f}%): {col}")
        feature_findings[col] = f
        continue

    if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        s = df[col].dropna()
        if len(s) == 0:
            continue

        # Variance check
        if s.std() == 0:
            f["flag_zero_variance"] = True
            print(f"\n  🚨 ZERO VARIANCE: {col}")

        # Distribution shape
        try:
            sk = float(stats.skew(s))
            ku = float(stats.kurtosis(s))
        except Exception:
            sk = ku = float("nan")
        f["skewness"] = sk
        f["kurtosis"] = ku

        if abs(sk) > 2:
            shape = "highly skewed"
        elif abs(sk) > 1:
            shape = "moderately skewed"
        else:
            shape = "approx. normal"
        f["distribution_shape"] = shape

        # IQR outliers
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = ((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum()
        f["iqr_outliers"] = int(iqr_outliers)

        # Z-score outliers
        z = np.abs(stats.zscore(s))
        z_outliers = (z > 3).sum()
        f["zscore_outliers"] = int(z_outliers)

        print(f"  {col:45s} skew={sk:+.2f}  kurt={ku:+.2f}  shape={shape:20s}  "
              f"IQR-out={iqr_outliers:>6,}  Z-out={z_outliers:>6,}")
    else:
        # Categorical
        n_unique = df[col].nunique()
        f["unique_count"] = int(n_unique)
        if n_unique == 1:
            f["flag_zero_variance"] = True
            print(f"\n  🚨 ZERO VARIANCE (single value): {col}")

    feature_findings[col] = f

# Multicollinearity check among numeric columns
print("\n--- Multicollinearity Check (numeric columns, |corr| > 0.90) ---")
num_df = df[num_cols].dropna()
if len(num_df) > 0 and len(num_cols) > 1:
    corr_matrix = num_df.corr().abs()
    high_corr_pairs = []
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1:]:
            if c1 in corr_matrix.columns and c2 in corr_matrix.columns:
                val = corr_matrix.loc[c1, c2]
                if val > 0.90:
                    high_corr_pairs.append((c1, c2, float(val)))
                    print(f"  🚨 HIGH CORR ({val:.3f}): {c1} ↔ {c2}")
    if not high_corr_pairs:
        print("  No highly correlated numeric pairs (>0.90) found.")
else:
    high_corr_pairs = []

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — DATA QUALITY VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 6 — DATA QUALITY VERDICT")
print("─" * 70)

issues = []

# Issue 1: Null analysis
for col, row in null_df.iterrows():
    if col in NULL_COLS_EXPECTED:
        severity = "ACCEPTABLE"
        note = "Expected null per CLAUDE.md — confirmed unusable"
    elif row["null_pct"] > 30:
        severity = "BLOCKING" if row["null_pct"] > 80 else "FIXABLE"
        note = f"High null ({row['null_pct']:.1f}%) — needs decision: impute or drop"
    else:
        severity = "FIXABLE"
        note = f"Minor nulls ({row['null_pct']:.1f}%) — impute or drop"
    issues.append({"column": col, "issue": "null_values", "null_pct": float(row["null_pct"]),
                   "severity": severity, "note": note})

# Issue 2: Duplicates
if dup_count > 0:
    dup_pct = dup_count / len(df) * 100
    severity = "BLOCKING" if dup_pct > 5 else "FIXABLE"
    issues.append({"issue": "duplicate_rows", "count": int(dup_count),
                   "pct": float(dup_pct), "severity": severity,
                   "note": "Deduplicate before training"})

# Issue 3: Geo
if geo_findings.get("out_of_bbox_count", 0) > 0:
    pct = geo_findings["out_of_bbox_count"] / len(df) * 100
    severity = "BLOCKING" if pct > 10 else "FIXABLE"
    issues.append({"issue": "geo_out_of_bbox", "count": geo_findings["out_of_bbox_count"],
                   "pct": float(pct), "severity": severity,
                   "note": "Filter coordinates outside Bengaluru bounding box"})

# Issue 4: Temporal split
for col, tf in temporal_findings.items():
    if isinstance(tf, dict):
        if tf.get("in_train_window", 1) == 0:
            issues.append({"issue": f"no_train_data_{col}", "severity": "BLOCKING",
                           "note": "No data in train window — split must change"})
        if tf.get("in_test_window", 1) == 0:
            issues.append({"issue": f"no_test_data_{col}", "severity": "BLOCKING",
                           "note": "No data in test window — split must change"})

print("\n--- Issue Register ---")
blocking = [i for i in issues if i.get("severity") == "BLOCKING"]
fixable  = [i for i in issues if i.get("severity") == "FIXABLE"]
acceptable = [i for i in issues if i.get("severity") == "ACCEPTABLE"]

print(f"\n  BLOCKING  ({len(blocking)}):")
for i in blocking:
    print(f"    🔴 {i}")
print(f"\n  FIXABLE   ({len(fixable)}):")
for i in fixable:
    print(f"    🟡 {i}")
print(f"\n  ACCEPTABLE({len(acceptable)}):")
for i in acceptable:
    print(f"    🟢 {i}")

# Required preprocessing steps
print("\n--- Required Preprocessing Steps ---")
preprocessing_steps = [
    "1. Drop/ignore columns: description, closed_datetime, action_taken_timestamp (100% null)",
    "2. Parse created_datetime → datetime64; drop rows with parse failures",
    "3. Filter lat/lon to Bengaluru bounding box — drop out-of-range rows",
    "4. Deduplicate rows (if duplicate count > 0)",
    "5. Encode violation_type, vehicle_type → integer/ordinal",
    "6. Extract temporal features: hour_of_day, day_of_week, is_weekend, month",
    "7. Aggregate to zone × time-block grid (zone = DBSCAN cluster label)",
    "8. Time-based split: train Nov2023–Feb2024, test Mar2024–Apr2024",
    "9. Assert no-future-leakage at split boundary",
]
for s in preprocessing_steps:
    print(f"  {s}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — PROBLEM REVALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 7 — PROBLEM REVALIDATION")
print("─" * 70)

revalidation = {
    "ml_framing": "PENDING — depends on split viability; likely regression (violation count prediction per zone×time) + ranking",
    "evaluation_achievable": "PENDING — depends on test window data availability",
    "split_leakage_risk": "LOW — time-based split with strict boundary assertion in train.py",
    "traps_confirmed": [
        "description / closed_datetime / action_taken_timestamp: 100% null — confirmed unusable",
        "WRONG PARKING dominant class — per-class metrics required",
        "Spatial leakage: use zone-level aggregation not raw lat/lon as features",
        "No random split — only time-based split permitted",
        "No external datasets — FAQ disqualification risk",
    ],
}

print("\n  ML Framing: Regression (violation count per zone×time block) + Ranking")
print("  Evaluation: Qualitative judging + internal Precision@K, NDCG@10, MAE/RMSE")
print("  Baseline: Frequency ranker (rank zones by historical count — no ML)")
print("  Split: Train Nov2023–Feb2024, Test Mar2024–Apr2024 — time-based ONLY")
print("  Leakage guards: (1) time-based split boundary assertion, (2) zone agg not raw lat/lon")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — ARCHITECTURE CHECKLIST STATUS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 8 — ARCHITECTURE GATE CHECKLIST (status after this EDA)")
print("─" * 70)

checklist = {
    "eda_complete": True,
    "eda_summary_saved": False,  # set True after saving
    "no_blocking_issues": len(blocking) == 0,
    "target_distribution_understood": True,
    "split_validated": all(
        tf.get("in_train_window", 0) > 0 and tf.get("in_test_window", 0) > 0
        for col, tf in temporal_findings.items()
        if isinstance(tf, dict) and "in_train_window" in tf
    ) if temporal_findings else False,
    "feature_list_finalized": False,  # needs configs/features.yaml
    "eval_metric_defined": False,     # needs configs/eval.yaml
    "baseline_model_defined": True,   # frequency ranker — in CLAUDE.md
    "cis_formula_agreed": False,      # pending eval.yaml
    "ranker_formula_agreed": False,   # pending eval.yaml
    "pipeline_script_planned": True,  # in CLAUDE.md pipeline overview
}

for item, status in checklist.items():
    icon = "✅" if status else "❌"
    print(f"  {icon} {item}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE EDA SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
summary = {
    "run_timestamp": datetime.now().isoformat(),
    "file": str(RAW_FILE.name),
    "file_size_mb": round(file_size_mb, 2),
    "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
    "columns": list(df.columns),
    "dtypes": dtype_map,
    "null_summary": {col: {"null_count": int(null_counts[col]), "null_pct": float(null_pct[col])}
                     for col in df.columns},
    "duplicate_rows": int(dup_count),
    "numeric_stats": {k: {sk2: (v2 if not (isinstance(v2, float) and np.isnan(v2)) else None)
                          for sk2, v2 in v.items()}
                      for k, v in num_stats.items()},
    "categorical_stats": cat_stats,
    "suspicious_flags": suspicious,
    "temporal_findings": temporal_findings,
    "geo_findings": geo_findings,
    "target_findings": target_findings,
    "feature_findings": {k: {fk: (fv if not (isinstance(fv, float) and np.isnan(fv)) else None)
                              for fk, fv in v.items()}
                         for k, v in feature_findings.items()},
    "high_corr_pairs": high_corr_pairs,
    "issues": issues,
    "preprocessing_steps": preprocessing_steps,
    "revalidation": revalidation,
    "architecture_gate_checklist": checklist,
}

# Save to BOTH locations (CLAUDE.md says data/processed/, request says data/outputs/)
for save_path in [SUMMARY_PATH, OUTPUTS_DIR / "eda_summary.json"]:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✅ EDA summary saved → {save_path}")

checklist["eda_summary_saved"] = True
print("\n" + "=" * 70)
print("EDA AUDIT COMPLETE")
print(f"Total issues: {len(blocking)} BLOCKING | {len(fixable)} FIXABLE | {len(acceptable)} ACCEPTABLE")
if blocking:
    print("⛔ BLOCKING ISSUES FOUND — resolve before architecture decision")
else:
    print("✅ No blocking issues — proceed to architecture after reviewing report")
print("=" * 70)
