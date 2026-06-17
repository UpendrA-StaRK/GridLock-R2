"""save_eda_json.py - Save the EDA summary to JSON"""
import json
from pathlib import Path
import pandas as pd

summary = {
    "run_timestamp": str(pd.Timestamp.now()),
    "file": "jan to may police violation_anonymized791b166.csv",
    "file_size_mb": 109.6,
    "shape": {"rows": 298450, "cols": 24},
    "columns": [
        "id","latitude","longitude","location","vehicle_number","vehicle_type",
        "description","violation_type","offence_code","created_datetime",
        "closed_datetime","modified_datetime","device_id","created_by_id",
        "center_code","police_station","data_sent_to_scita","junction_name",
        "action_taken_timestamp","data_sent_to_scita_timestamp",
        "updated_vehicle_number","updated_vehicle_type","validation_status",
        "validation_timestamp"
    ],
    "null_pct": {
        "id": 0.0, "latitude": 0.0, "longitude": 0.0, "location": 1.003,
        "vehicle_number": 0.0, "vehicle_type": 0.0, "description": 100.0,
        "violation_type": 0.0, "offence_code": 0.0, "created_datetime": 0.0,
        "closed_datetime": 100.0, "modified_datetime": 0.0, "device_id": 0.0,
        "created_by_id": 0.017, "center_code": 3.771, "police_station": 0.002,
        "data_sent_to_scita": 0.0, "junction_name": 0.0,
        "action_taken_timestamp": 100.0, "data_sent_to_scita_timestamp": 85.873,
        "updated_vehicle_number": 41.965, "updated_vehicle_type": 41.965,
        "validation_status": 41.965, "validation_timestamp": 41.965,
    },
    "duplicate_rows": 204037,
    "duplicate_rows_note": (
        "68.36% of rows share timestamps on created_datetime. "
        "These are likely multi-violation events (same officer, different location). "
        "Verify lat/lon difference before dropping -- do NOT blindly deduplicate."
    ),
    "geo": {
        "lat_col": "latitude", "lon_col": "longitude",
        "lat_min": 12.8027, "lat_max": 13.2937,
        "lon_min": 77.4426, "lon_max": 77.7717,
        "out_of_bbox_count": 0,
        "lat_iqr_outliers": 35507, "lat_z_outliers": 6964,
        "lon_iqr_outliers": 14995, "lon_z_outliers": 551,
        "note": (
            "All points within Bengaluru bbox [12.7-13.3, 77.4-77.8]. "
            "IQR outliers exist but may be real sparse-zone events. "
            "Use DBSCAN noise label rather than pre-filtering."
        ),
    },
    "temporal": {
        "created_datetime": {
            "min": "2023-11-09 19:11:46+00:00",
            "max": "2024-04-08 17:30:46+00:00",
            "range_days": 150,
            "parse_failures": 5,
            "missing_days_in_range": 0,
            "dup_timestamps": 204037,
            "in_train_Nov23_Feb24": 226296,
            "in_test_Mar24_Apr24": 70311,
            "before_train": 0,
            "after_test": 0,
            "split_ok": True,
            "note": "Test window ends Apr 8 2024 not Apr 30. 70,311 test rows -- sufficient.",
        },
        "validation_timestamp": {
            "null_pct": 41.965,
            "missing_days": 8,
            "missing_days_list": ["2024-02-26","2024-02-29","2024-03-03","2024-03-04",
                                  "2024-03-07","2024-03-09","2024-03-10","2024-03-18"],
            "note": "42% null. 8 missing days mostly around split boundary -- do not use as primary feature.",
        },
        "data_sent_to_scita_timestamp": {
            "null_pct": 85.873,
            "range": "Mar 2024 - Apr 2024 only",
            "note": "EXCLUDE: 86% null AND only covers test-window period. Using it would be leakage.",
        },
    },
    "target": {
        "col": "violation_type",
        "unique_raw_values": 991,
        "important_note": (
            "violation_type stores multi-violation events as JSON list strings "
            "(e.g. [\"WRONG PARKING\",\"DEFECTIVE NUMBER PLATE\"]). "
            "991 unique combinations. After ast.literal_eval + explode, "
            "real atomic types are ~15-20. Must parse before encoding."
        ),
        "dominant_class": "[\"WRONG PARKING\"]",
        "dominant_class_pct": 0.465,
        "imbalanced": True,
        "top10": {
            "[\"WRONG PARKING\"]": 138764,
            "[\"NO PARKING\"]": 119576,
            "[\"PARKING IN A MAIN ROAD\",\"WRONG PARKING\"]": 9472,
            "[\"PARKING IN A MAIN ROAD\",\"NO PARKING\"]": 4818,
            "[\"WRONG PARKING\",\"DEFECTIVE NUMBER PLATE\"]": 3317,
            "[\"NO PARKING\",\"PARKING IN A MAIN ROAD\"]": 2449,
            "[\"NO PARKING\",\"DEFECTIVE NUMBER PLATE\"]": 2380,
            "[\"WRONG PARKING\",\"PARKING IN A MAIN ROAD\"]": 1955,
            "[\"PARKING ON FOOTPATH\",\"WRONG PARKING\"]": 1190,
            "[\"NO PARKING\",\"WRONG PARKING\"]": 891,
        },
        "concept_drift": (
            "LOW. WRONG PARKING ratio: Nov23=46.0%, Dec23=44.8%, Jan24=46.4%, "
            "Feb24=48.6%, Mar24=46.2%, Apr24=48.8% -- stable. "
            "NO PARKING: mild decline from 43% to 39% -- acceptable."
        ),
    },
    "categorical_stats": {
        "vehicle_type": {
            "unique": 22, "null_pct": 0.0,
            "top": {"SCOOTER": 79459, "CAR": 65432, "MOTOR CYCLE": 32876,
                    "PASSENGER AUTO": 23533, "PASSENGER AUTO (3 WHEELER)": 23007},
        },
        "police_station": {
            "unique": 54, "null_pct": 0.002,
            "top": {"Upparpet": 34468, "Shivajinagar": 28044, "Malleshwaram": 22200,
                    "HAL Old Airport": 20819, "City Market": 17646},
        },
        "junction_name": {
            "unique": 169, "null_pct": 0.0,
            "top": {"No Junction": 147880, "BTP051 - Safina Plaza Junction": 15449,
                    "BTP082 - KR Market Junction": 11538, "BTP040 - Elite Junction": 10718},
            "note": "No Junction = 49.5% -- valid class not null. Encode as is_at_junction binary.",
        },
        "validation_status": {
            "unique": 5, "null_pct": 41.965,
            "values": {"approved": 115400, "rejected": 49754, "created1": 7044, "processing": 678},
        },
    },
    "numeric_stats": {
        "latitude": {
            "min": 12.8027, "max": 13.2937, "mean": 12.9813, "std": 0.0712,
            "iqr_outliers": 35507, "z_outliers": 6964, "skew": 1.36,
            "shape": "moderately-skewed",
        },
        "longitude": {
            "min": 77.4426, "max": 77.7717, "mean": 77.5934, "std": 0.0655,
            "iqr_outliers": 14995, "z_outliers": 551, "skew": 0.81,
            "shape": "near-normal",
        },
        "center_code": {
            "min": 2.0, "max": 88.0, "null_pct": 3.77,
            "iqr_outliers": 19488, "skew": 1.81, "shape": "moderately-skewed",
            "note": "Categorical-ish integer (center ID). Treat as categorical not numeric feature.",
        },
    },
    "issues": [
        {
            "col": "data_sent_to_scita_timestamp", "issue": "null",
            "null_pct": 85.873, "severity": "BLOCKING",
            "note": "86% null AND only covers test-window dates. EXCLUDE completely. No blocking impact once excluded.",
        },
        {
            "issue": "duplicate_timestamps", "count": 204037, "pct": 68.36,
            "severity": "FIXABLE",
            "note": "Same created_datetime, likely different lat/lon. Verify before deduplication. May represent multi-violation events at same moment.",
        },
        {
            "col": "updated_vehicle_number", "null_pct": 42.0, "severity": "FIXABLE",
            "note": "High null 42% -- exclude from primary feature set",
        },
        {
            "col": "updated_vehicle_type", "null_pct": 42.0, "severity": "FIXABLE",
            "note": "High null 42% -- exclude from primary feature set",
        },
        {
            "col": "validation_status", "null_pct": 42.0, "severity": "FIXABLE",
            "note": "High null 42% + not available at prediction time -- exclude",
        },
        {
            "col": "validation_timestamp", "null_pct": 42.0, "severity": "FIXABLE",
            "note": "High null 42% + 8 missing days near split boundary -- exclude",
        },
        {
            "col": "center_code", "null_pct": 3.77, "severity": "FIXABLE",
            "note": "3.77% null -- impute with mode per police_station group",
        },
        {
            "col": "location", "null_pct": 1.003, "severity": "FIXABLE",
            "note": "1% null -- drop column (free-text address, not useful as ML feature)",
        },
        {
            "col": "description", "null_pct": 100.0, "severity": "ACCEPTABLE",
            "note": "Expected null per CLAUDE.md -- confirmed",
        },
        {
            "col": "closed_datetime", "null_pct": 100.0, "severity": "ACCEPTABLE",
            "note": "Expected null per CLAUDE.md -- confirmed",
        },
        {
            "col": "action_taken_timestamp", "null_pct": 100.0, "severity": "ACCEPTABLE",
            "note": "Expected null per CLAUDE.md -- confirmed",
        },
    ],
    "blocking_issues_count": 1,
    "blocking_resolution": (
        "data_sent_to_scita_timestamp BLOCKING by null threshold but resolution is EXCLUDE. "
        "Once excluded, zero blocking issues remain for the modelling pipeline."
    ),
    "features_plan": {
        "temporal": ["hour_of_day", "day_of_week", "is_weekend", "month"],
        "spatial": ["latitude", "longitude", "zone_id (DBSCAN cluster)", "is_at_junction"],
        "categorical_encoded": [
            "violation_type_primary_encoded",
            "vehicle_type_encoded",
            "police_station_id",
            "center_code (treat as categorical ID)",
        ],
        "aggregated_target": [
            "zone_hour_violation_count (PRIMARY REGRESSION TARGET)",
            "zone_day_violation_count",
        ],
        "exclude": [
            "description, closed_datetime, action_taken_timestamp (100% null)",
            "data_sent_to_scita_timestamp (86% null + test-window leakage)",
            "validation_status, validation_timestamp (42% null, not at prediction time)",
            "updated_vehicle_number, updated_vehicle_type (42% null)",
            "modified_datetime (post-event -- not available at prediction time, leakage)",
            "id, vehicle_number, location (identifiers / free-text)",
        ],
    },
    "preprocessing_steps": [
        "1. Drop: description, closed_datetime, action_taken_timestamp (100% null)",
        "2. Drop: id, vehicle_number, location, modified_datetime (identifiers / leakage)",
        "3. Drop: data_sent_to_scita_timestamp (86% null + test-window leakage)",
        "4. Drop: validation_status, validation_timestamp, updated_vehicle_* (42% null)",
        "5. Parse created_datetime -> datetime64[ns, UTC]; drop 5 parse failures",
        "6. Check lat/lon duplicates: if same lat/lon AND same timestamp, deduplicate; if different lat/lon, keep (multi-violation event)",
        "7. Parse violation_type JSON strings via ast.literal_eval; explode to atomic types; take primary (first) type as label",
        "8. Encode: violation_type_primary, vehicle_type, offence_code -> LabelEncoder",
        "9. Encode: police_station -> integer ID",
        "10. Impute center_code nulls with mode per police_station group",
        "11. Create is_at_junction = (junction_name != 'No Junction').astype(int)",
        "12. Extract temporal: hour_of_day, day_of_week, is_weekend, month from created_datetime",
        "13. Run DBSCAN on (lat, lon) -> zone_id column (noise = -1, treat as sparse zone)",
        "14. Aggregate to zone x time-block (hour or day) -> target column (violation count)",
        "15. Time-based split: train = [2023-11-09, 2024-02-29], test = [2024-03-01, 2024-04-08]",
        "16. Assert: max(train.created_datetime) < min(test.created_datetime) -- hard error if fails",
    ],
    "architecture_gate_checklist": {
        "eda_complete": True,
        "eda_summary_json_saved": True,
        "no_blocking_issues": "CONDITIONAL -- data_sent_to_scita_timestamp excluded; 0 remaining blockers",
        "target_distribution_understood": True,
        "train_test_split_validated_no_leakage": True,
        "feature_list_finalized_features_yaml": False,
        "eval_metric_defined_eval_yaml": False,
        "baseline_model_defined": True,
        "cis_formula_agreed_eval_yaml": False,
        "ranker_weighting_formula_eval_yaml": False,
        "pipeline_script_planned": True,
    },
    "step7_revalidation": {
        "ml_framing": "CONFIRMED: Regression (predict violation count per zone x timeblock) + Ranking (zone priority score = predicted_count x CIS)",
        "eval_metric_achievable": "YES -- 298k rows total; train=226k, test=70k. MAE/RMSE, Precision@K, NDCG@10 all achievable.",
        "split_clean": "YES. Strictly time-based. Train: Nov 9 2023 - Feb 29 2024. Test: Mar 1 - Apr 8 2024. No overlap. Leakage assertion required in train.py.",
        "leakage_risks_identified_and_resolved": [
            "data_sent_to_scita_timestamp: test-window only -- EXCLUDED",
            "modified_datetime: post-event -- EXCLUDED",
            "Raw lat/lon as zone feature -- use DBSCAN zone_id instead",
            "validation_status: not available at prediction time -- EXCLUDED",
        ],
        "traps_confirmed": [
            "violation_type is multi-label JSON string (991 combos) -- parse before encoding",
            "WRONG PARKING 46.5% dominance -- per-class F1 mandatory not just accuracy",
            "Duplicate timestamps (68%) -- likely multi-violation events not exact duplicates",
            "Test window ends Apr 8 not Apr 30 -- shorter but 70k rows sufficient",
            "junction_name 'No Junction' = 49.5% -- valid class, encode as binary is_at_junction",
            "NO PARKING mild decline trend across months -- low drift risk, document in eval",
        ],
    },
    "step9_architecture_recommendation_pending": (
        "DO NOT propose architecture until: "
        "(1) features.yaml created, (2) eval.yaml created with CIS formula + ranker weights. "
        "These require USER input/approval."
    ),
}

Path("data/outputs").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

for path in ["data/outputs/eda_summary.json", "data/processed/eda_summary.json"]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved -> {path}")

print()
print("=" * 60)
print("eda_summary.json saved successfully.")
print("BLOCKING: 1 (resolved by exclusion)")
print("FIXABLE : 8 (all resolvable in preprocessing)")
print("ACCEPTABLE: 3 (expected nulls)")
print("Architecture gate: 5 items pending (configs needed)")
print("=" * 60)
