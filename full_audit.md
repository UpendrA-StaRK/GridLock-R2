# GridLock R2 — Full Project Audit

> **Auditor:** Claude Opus 4.6 (Thinking)  
> **Date:** 2026-06-17  
> **Scope:** Every source file, config, notebook, output, and artifact in the project  
> **Files reviewed:** 30+ files across `src/`, `configs/`, `data/`, `notebooks/`, `artifacts/`, `checkpoints/`

---

## AUDIT 1 — DATA REVIEW

### 1.1 Is the data good enough?

**Yes, with caveats.**

- **298,450 rows** × 24 columns, 109.6 MB. After dedup: **268,281 rows**. This is more than enough for gradient-boosted tree regression on ~140 zones.
- **150 days** of data (Nov 9 2023 → Apr 8 2024). No calendar gaps. No missing days.
- All coordinates are within the Bengaluru bounding box — no geographic filtering was needed.
- The target (violation count per zone × time-block) is well-defined and derived cleanly.

> [!WARNING]
> **The data has one fundamental limitation:** there is no ground truth for "congestion impact." The problem asks for quantifying impact on traffic flow, but the dataset contains only violation records — not traffic speed, density, or delay data. The CIS formula is an **educated proxy** (density × junction weight), not a measured output. This ceiling is inherent to the dataset, not a pipeline bug.

### 1.2 Outliers, nulls, and class imbalance

| Issue | Handling | Verdict |
|---|---|---|
| **100% null columns** (description, closed_datetime, action_taken_timestamp) | Correctly identified, documented, dropped | ✅ Correct |
| **86% null** (data_sent_to_scita_timestamp) | Excluded as temporal leakage — only exists in test window | ✅ Correct |
| **42% null** (validation_*, updated_*) | Excluded — post-event fields not available at prediction time | ✅ Correct |
| **3.77% null** (center_code) | Imputed with mode per police_station group, fallback to global mode | ✅ Reasonable |
| **5 parse failures** (created_datetime) | Dropped with logging | ✅ Acceptable |
| **Lat IQR outliers** (11.9%, 35,507 rows) | Not filtered — correctly delegated to DBSCAN noise label | ✅ Correct |
| **Class imbalance** (WRONG PARKING ≈ 46.5%) | Documented. Per-class reporting mandated in eval.yaml | ✅ Correct |

**Verdict: Data cleaning is thorough and well-documented. No silent drops.**

### 1.3 Train/test split — zero leakage?

**Yes. The split is clean.**

- **Train:** Nov 9 2023 → Feb 29 2024 (226,296 rows → after dedup ~200k+)
- **Test:** Mar 1 → Apr 8 2024 (70,311 rows → after dedup ~68k)
- Hard assertion in [train.py:149](file:///c:/Users/palur/OneDrive/Desktop/GridLock_R2_Transfer/src/training/train.py#L149): `max(train date) < min(test date)` — raises `AssertionError` if violated.
- The temporal leakage column (`data_sent_to_scita_timestamp`) was correctly excluded.
- `rolling_7d_count` uses `shift(1)` before rolling — current day's count is never included. ✅

> [!NOTE]
> One subtlety: the `rolling_7d_count` in the test set uses values from the training period (the last 7 days before March 1). This is **correct behavior** — it represents "what happened in the recent past" which would be available at prediction time. Not leakage.

### 1.4 Feature set quality

| Feature | Source | Signal | Verdict |
|---|---|---|---|
| `hour_of_day` | Temporal | High — captures rush-hour patterns | ✅ Essential |
| `day_of_week` | Temporal | Medium — weekday/weekend effect | ✅ Keep |
| `is_weekend` | Temporal | Low-medium — correlated with `day_of_week` | ⚠️ Redundant but harmless |
| `month` | Temporal | Low — only 5 months, low variance | ⚠️ Marginal |
| `zone_id` | Spatial | **Dominant** — directly encodes geographic identity | ✅ Essential but see §1.5 |
| `fraction_at_junction` | Spatial | Medium — congestion proxy | ✅ Keep |
| `rolling_7d_count` | Historical | **Strongest signal** — lagged zone activity | ✅ Essential |
| `dominant_violation_type` | Categorical | Low-medium — most zones have stable dominant type | ⚠️ Marginal |
| `dominant_vehicle_type` | Categorical | Low — scooter/car dominance is zone-level constant | ⚠️ Marginal |
| `police_station_id` | Categorical | Low — collinear with `zone_id` | ⚠️ Likely redundant |
| `center_code_encoded` | Categorical | Low — collinear with `police_station_id` | ⚠️ Likely redundant |
| `data_sent_to_scita_mean` | Optional | Unknown — pending SHAP analysis | ⚠️ Unvalidated |

### 1.5 Suspicious findings

> [!CAUTION]
> **Finding #1: `zone_id` as a direct feature is problematic.**
> 
> `zone_id` is a label-encoded DBSCAN cluster label (0–138). It is being fed directly into XGBoost as a numeric feature. The model is treating it as an ordinal number — "zone 50 is between zone 49 and zone 51" — which is meaningless. DBSCAN labels have no numeric ordering.
> 
> **Impact:** The model likely memorizes zone_id → count mappings, making it a fancy lookup table. This is why **all models achieve NDCG@10 = 1.0 and the baseline also scores 1.0**. The model isn't learning temporal patterns — it's memorizing zone identities.
> 
> **Fix:** Either (a) one-hot encode zone_id (not feasible with 140 zones in XGBoost) or (b) replace zone_id with zone-level aggregate features (mean historical count, CIS score, zone area, zone violation density) as separate columns and **drop zone_id entirely**. This forces the model to generalize from zone characteristics rather than memorizing zone identity.

> [!CAUTION]
> **Finding #2: `police_station_id` and `center_code_encoded` are collinear with `zone_id`.**
> 
> Each DBSCAN zone is dominated by a single police station and center code. Including all three is adding correlated proxies for the same information: "which geographic area is this?" This inflates the model's apparent feature count without adding independent signal.

> [!WARNING]
> **Finding #3: No SHAP feature importance analysis was performed.**
> 
> CLAUDE.md and features.yaml both say "check SHAP importance after training" for `data_sent_to_scita` and other optional features. This was never done. The session log confirms training completed but no importance analysis followed.

---

## AUDIT 2 — MODEL REVIEW

### 2.1 Is XGBoost the right tool?

**Yes, for this problem, gradient-boosted trees are the right family.** The data is tabular, the target is a count, and the feature set is a mix of categorical and numerical. XGBoost/LightGBM/CatBoost are the standard workhorses here.

However, the **specific formulation** is questionable — see §2.3.

### 2.2 Loss curve analysis

From [eval_20260616_162216.json](file:///c:/Users/palur/OneDrive/Desktop/GridLock_R2_Transfer/data/outputs/eval_20260616_162216.json):

**XGBoost (hour resolution) — the winner:**
- 120 rounds trained (early stopping at patience=20, did NOT trigger — ran all 300? No — the eval_history shows ~120 entries, so training did converge and stop)
- Validation RMSE: 22.01 → 10.66 (final). Steady decrease, mild plateau from round ~60 onward.
- **No signs of overfitting** — val RMSE doesn't increase at end.

**LightGBM (hour):**
- ~105 rounds. Final val RMSE: 10.61. Slightly better RMSE than XGBoost.
- Same pattern: smooth descent, no overfitting.

**CatBoost (hour):**
- ~300 rounds (ran to completion, no early stop). Final val RMSE: 11.35.
- Slowest to converge. Mildly worse than XGBoost/LightGBM.

**Day resolution models:**
- All significantly worse (MAE ~10.5–13.4 vs ~4.7–5.0 for hourly).
- XGBoost day early-stopped at round ~55 and started increasing — mild overfit.

### 2.3 Did it actually learn anything?

**Partially — and this is the most important finding.**

| Metric | XGBoost (hour) | Naive Baseline | ML Lift |
|---|---|---|---|
| MAE | 4.68 | 6.97 | +32.8% ✅ |
| RMSE | 10.66 | 15.90 | +33.0% ✅ |
| NDCG@10 | 1.000 | 1.000 | 0% ❌ |
| Precision@10 | 1.000 | 1.000 | 0% ❌ |

> [!CAUTION]
> **The ML model does NOT beat the frequency baseline on the ranking task.**
> 
> Both the ML model and the "just count historical violations per zone" baseline achieve **perfect NDCG@10 and Precision@10**. The model adds **zero value** for the actual downstream task (ranking enforcement zones).
> 
> The model **does** improve count prediction accuracy by ~33% — which matters for telling an officer "expect ~5 violations" vs "expect ~7" — but for the question "where should you go?", a simple lookup table of historical zone frequencies gives identical results.
> 
> The session log acknowledges this (line 271-278) and attributes it to "spatial stability" — the top zones are consistently the same. This is a valid explanation but doesn't change the implication: **the ML model's value proposition for ranking is zero**.

### 2.4 Overfitting, underfitting, or generalizing?

- **Not overfitting**: val RMSE curves are smooth, no divergence from train RMSE trajectory.
- **Not significantly underfitting**: 33% MAE improvement over naive baseline shows the model learned something.
- **But**: it learned the **wrong thing** — it learned zone_id → count mappings, not temporal dynamics that differentiate the model from a lookup table.

### 2.5 Would a simpler model work?

**For ranking: yes, a frequency table does the same job.**  
**For count prediction: the ML model genuinely helps.** But count prediction accuracy is not what judges will evaluate — they care about "which zones should we enforce?" (ranking).

---

## AUDIT 3 — ARCHITECTURE REVIEW

### 3.1 End-to-end pipeline walkthrough

```
CSV (raw) → validate.py → load.py → features.py (Phase A)
    → clustering.py (DBSCAN) → features.py (Phase B: zone aggregation)
    → train.py (XGBoost/LightGBM/CatBoost)
    → metrics.py (full_eval)
    → ranker.py (inference)
    → static_output.py (HTML map)
    → pipeline.py (orchestrator)
```

**Every link in the chain works.** The pipeline ran end-to-end successfully with outputs saved. There are no broken imports, no missing files, no dangling references.

### 3.2 Broken links or missing steps

| Issue | Severity | Detail |
|---|---|---|
| `pipeline.py` calls `validate_raw()` — function doesn't exist in validate.py | 🔴 **Breaking** | `validate.py` exports `validate_schema()`, not `validate_raw()`. Pipeline step 1 will crash. |
| `pipeline.py` calls `load_raw(csv_path, metadata_output_path=meta_path)` — `load_raw()` doesn't accept that kwarg | 🔴 **Breaking** | `load.py` signature: `load_raw(csv_path, eval_config_path, save_report, report_path)`. Pipeline step 2 will crash. |
| No `src/evaluation/report.py` | 🟡 Minor | CLAUDE.md says it should exist but it doesn't. `metrics.py` handles saving instead. Functional but inconsistent with docs. |
| No `src/models/predictor.py` | 🟡 Minor | CLAUDE.md says it should exist. `train.py` builds models inline instead. Works but inconsistent with docs. |
| `pipeline.py --skip-features --skip-clustering --skip-training` works | ✅ | Confirmed in session log: runs in 3.3s (inference-only mode). |
| Dashboard (`src/dashboard/app.py`) | ⏳ | Marked as deferred. Not built. Acceptable for current phase. |
| `day_of_week` added as feature in `train.py` but session log notes it was "missing — FIXED" | ✅ | Confirmed present in both `_get_feature_cols()` functions. |

> [!CAUTION]
> **The pipeline.py orchestrator will CRASH if run without skip flags.**
> 
> Steps 1 and 2 call functions with wrong names/signatures. This means `python -m src.data.pipeline` (full run) — which CLAUDE.md says "judges may ask to run live" — **will fail**.
> 
> The skip-flag mode (inference-only) works because it bypasses steps 1-3. But a full cold run is broken.

### 3.3 Unnecessary complexity

- The 3-model comparison (XGBoost + LightGBM + CatBoost) × 2 resolutions = 6 training runs. For a hackathon this is overkill — all models give essentially identical ranking results. Could have trained one model.
- However, having the comparison is a **good thing for judge Q&A** — you can show you evaluated alternatives. Keep it.

### 3.4 Architecture rating

**Rating: 7/10**

**Justification:**
- (+3) Clean separation of concerns: data → features → clustering → training → evaluation → inference → output. Each module is importable and testable.
- (+2) Excellent documentation and config-driven design. CLAUDE.md, session_log.md, three YAML configs, all cross-referenced.
- (+1) Checkpointing is thorough — every model saves weights + config copies + hash.
- (+1) The static HTML fallback output is a pragmatic choice for demo reliability.
- (-1) pipeline.py has broken function calls that would crash on a full run.
- (-1) The ranking evaluation is meaningless (all 1.0) — should have been caught and addressed.
- (-1) No feature importance analysis (SHAP) despite being planned.
- (-1) DBSCAN silhouette score is **-0.0955** (negative!) — this means clusters are overlapping. The tuning notebook chose eps=0.05 with silhouette 0.3415 on a 50k sample, but the full 268k-row run produces -0.0955. This discrepancy was logged but not investigated.

### 3.5 Top 3 weakest points

1. **The ranking task is solved by a lookup table.** The ML model adds zero value over the frequency baseline for the primary downstream use case. This is a fundamental problem, not a bug.

2. **DBSCAN clusters have negative silhouette on the full dataset** (-0.0955). The zones overlap geographically, which means zone_id assignments are unstable at boundaries. Clusters tuned on a 50k sample don't generalize to the full dataset.

3. **pipeline.py will crash on full (non-skip) run.** Function signatures in step1 and step2 don't match the actual module APIs. This is a demo risk.

---

## AUDIT 4 — EVALUATION REVIEW

### 4.1 Are metrics aligned with judging criteria?

The hackathon judges evaluate on: **Feasibility · Relevance · Innovation · Real-World Impact**

| Metric Used | Judges Care? | Alignment |
|---|---|---|
| MAE/RMSE | Partially — shows prediction accuracy | ✅ Supports "Feasibility" |
| NDCG@10 | Yes — shows ranking quality | ⚠️ Currently meaningless (all 1.0) |
| Precision@10 | Yes — shows how many recommendations are good | ⚠️ Currently meaningless (all 1.0) |
| Silhouette score | Indirectly — cluster quality | ⚠️ Negative score undermines "Innovation" |
| CIS formula | Yes — judges want to see congestion quantification | ✅ Defensible proxy |
| Baseline comparison | Yes — "does ML add value?" | ❌ Doesn't beat baseline on ranking |

> [!WARNING]
> **The evaluation story has a hole:** the main demo metric (NDCG@10) is 1.0 for everything including the dumb baseline. A judge asking "why do I need ML?" will correctly note that a frequency table does the same job for zone ranking.
>
> The honest answer is: "ML improves count prediction accuracy by 33%, which matters for resource allocation (how many officers to send), even though the zone priority order is stable."

### 4.2 Is there a proper baseline?

**Yes.** Two baselines are computed:
1. **Naive mean-per-zone** (regression baseline) — MAE 6.97 vs model MAE 4.68. ML wins.
2. **Frequency ranker** (ranking baseline) — NDCG@10 = 1.0 vs model NDCG@10 = 1.0. Tie.

The baselines are well-designed. The problem is that the model can't beat the ranking baseline, not that the baseline is missing.

### 4.3 Would current scores hold on unseen data?

**Mostly yes for regression, likely yes for ranking.**

- The time-based split is clean (no leakage confirmed).
- Violation patterns are stable across months (concept drift is LOW per EDA).
- The top-10 zones are geographically stable — they'd likely stay the same even with new months of data.
- MAE might increase slightly on truly future data (e.g., monsoon season) due to seasonal effects not captured in 5 months.
- **Risk:** If a new commercial development creates a new hotspot after April 2024, the frozen DBSCAN zones won't capture it. This is acknowledged in the roadmap (B5/B7).

---

## AUDIT 5 — IMPROVEMENT ROADMAP VALIDATION

### Existing roadmap items — still relevant?

| Item | Still Relevant? | Comment |
|---|---|---|
| **A1** — 15-min resolution | ⚠️ **Low priority** | Won't fix the core problem (ranking = lookup table). Only improves scheduling granularity, which is nice but not the bottleneck. |
| **A2** — Minute-level | ❌ **Remove** | Correctly flagged as "not recommended" in roadmap. Still true. |
| **A3** — Second-level | ❌ **Remove** | Correctly flagged as out of scope. Still true. |
| **B1** — Real-time ingestion | ✅ Still relevant | Post-hackathon. Correct priority. |
| **B2** — Auto retraining | ✅ Still relevant | Post-hackathon. Correct priority. |
| **B3** — Uncertainty quantification | ✅ **High priority** | Achievable within hackathon. Would significantly improve demo credibility. |
| **B4** — Concept drift detection | ⚠️ Medium | PSI metric is achievable; full pipeline is post-hackathon. |
| **B5** — Cold-start zones | ⚠️ Medium | Relevant but not the most impactful improvement for the demo. |
| **B6** — Multi-city scalability | ❌ **Remove from hackathon scope** | Nice story for judges but zero code impact needed. |
| **B7** — Zone boundary drift | ❌ **Remove from hackathon scope** | Post-hackathon. Correct. |
| **B8** — Officer feedback loop | ❌ **Remove from hackathon scope** | Post-hackathon. Correct. |

### What's genuinely MISSING from the roadmap

> [!IMPORTANT]
> **Missing #1: Fix the ranking evaluation to be non-trivial.**
> 
> The current NDCG@10 = 1.0 for everything (including baseline) means the ranking metric provides zero information. This is the single biggest weakness in the evaluation story.
> 
> **Fix:** Evaluate ranking at finer granularity — per-hour or per-day ranking, not aggregate-over-entire-test-period ranking. Zone 2 might be the top zone overall, but at 2am it should rank lower than zone 50. Evaluate NDCG@10 for **each hour slot** separately, then average. This will produce meaningful differentiation between ML model and frequency baseline. Add an `ndcg_per_hour()` function to `metrics.py`.

> [!IMPORTANT]
> **Missing #2: SHAP feature importance analysis.**
> 
> This was planned (CLAUDE.md, features.yaml) but never executed. It would:
> - Validate whether `data_sent_to_scita` should be kept or dropped
> - Reveal whether `zone_id` is dominating (spoiler: it almost certainly is)
> - Provide a compelling visualization for the demo ("these are the factors driving enforcement priority")
> 
> **Fix:** Add a `06_shap.ipynb` notebook. Run `shap.TreeExplainer(model)` on the test set. Generate summary plot + per-zone force plots.

> [!IMPORTANT]
> **Missing #3: Fix pipeline.py to actually run end-to-end.**
> 
> Steps 1 and 2 call non-existent function signatures. This is a demo blocker.

> [!TIP]
> **Missing #4 (nice to have): Interactive time slider for the demo.**
> 
> CLAUDE.md identifies the "single most impressive demo moment" as a time-of-day slider on the map. The static HTML output has no slider — it shows one hour at a time. The Streamlit dashboard (deferred) would have it. If time permits, building even a minimal hour-selector in the static HTML would significantly improve the demo impact.

---

## VERDICT

### Is this ready to demo as-is?

**Conditionally yes.** The inference-only pipeline works (`--skip-features --skip-clustering --skip-training`). The static HTML output with Folium map renders correctly. The ranked enforcement table is clear. A 2-minute demo walkthrough is achievable.

**But** if a judge asks to run the full pipeline from scratch, it will crash. And if a judge asks "does your ML model actually beat a simple frequency table?", the honest answer is "for ranking, no."

### 3 things that MUST be fixed before submission

1. **Fix `pipeline.py` function calls** (steps 1 and 2). Either update `pipeline.py` to call the correct function names/signatures from `validate.py` and `load.py`, or accept that full cold runs are broken and only demo with skip flags. The first option is a 10-minute fix.

2. **Add per-hour NDCG evaluation** to `metrics.py`. The current aggregate NDCG@10 = 1.0 is an evaluation gap that will not survive a methodology probe from judges. Adding per-hour ranking evaluation will show that the ML model **does** outperform the frequency baseline at predicting which zones are hottest at specific times of day, even if the overall zone ranking is stable. This is the model's real value proposition — it tells you not just where, but when.

3. **Run SHAP analysis and include it in the demo.** A single summary plot showing "rolling_7d_count and zone_id are the top features, hour_of_day captures temporal dynamics" is compelling evidence for methodology and transparency. Judges include domain experts who will probe this.

### Honest ceiling of this solution

Given the current data and model:

- **Ranking accuracy:** The top-10 enforcement zones are geographically stable and any reasonable model (including frequency counting) identifies them correctly. The ML ceiling for ranking improvement is low because the problem is spatially dominated.

- **Count prediction:** MAE ≈ 4.7 violations per zone-hour. This means the model is off by about 5 violations per zone per hour on average, against a mean of ~7 violations. That's useful but not precise.

- **Congestion impact quantification:** Limited by the absence of actual traffic flow data. The CIS formula is a defensible proxy but it's not measuring real congestion — it's measuring violation density weighted by junction presence. A judge who asks "how do you know illegal parking actually causes congestion here?" will correctly note that you're inferring impact, not measuring it.

- **Innovation ceiling:** The core idea (cluster violations → predict counts → rank zones × CIS) is solid and appropriate for the data available. The innovation is in the integration — tying prediction to enforcement prioritization with a clear formula. That's the pitch.

- **Realistic demo strength:** Strong visual output (Folium map with ranked zones). Clean pipeline. Well-documented methodology. The weakness is that the ML "value add" over simple statistics is genuinely small for this specific dataset. Frame the demo around the **system** (data pipeline → prediction → enforcement ranking → officer scheduling) rather than individual model accuracy numbers.

---

## Summary Scorecard

| Audit Area | Score | Key Issue |
|---|---|---|
| **Data Quality** | 8/10 | Clean, well-documented. No ground truth for congestion impact. |
| **Model Quality** | 6/10 | Learns count prediction (33% lift). Fails to differentiate from baseline on ranking. |
| **Architecture** | 7/10 | Clean design, but pipeline.py has breaking bugs. |
| **Evaluation** | 5/10 | NDCG@10 = 1.0 everywhere is an evaluation failure, not a success. No SHAP. |
| **Roadmap** | 7/10 | Well-structured but missing the most critical items (per-hour eval, SHAP). |
| **Demo Readiness** | 6/10 | Works in skip mode. Full cold run crashes. No time slider. |

**Overall: 6.5/10 — Competent prototype with a significant evaluation blind spot.**
