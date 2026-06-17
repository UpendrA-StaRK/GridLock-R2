"""
src/evaluation/metrics.py
GridLock R2 — PS1: Parking-Induced Congestion

Evaluation metrics for the enforcement priority ranking system.

Functions:
    regression_metrics()          → MAE, RMSE for violation count prediction
    ndcg_at_k()                   → NDCG@K using graded relevance (0/1/2)
    precision_at_k()              → Precision@K for top-K zone ranking
    compute_relevance()           → Assign graded relevance labels to zones
    frequency_baseline()          → Rank zones by historical count × CIS (no ML)
    full_eval()                   → Run all metrics and return structured dict
    ndcg_per_hour()               → PHASE 2: Per-hour NDCG@K (primary differentiation metric)
    temporal_rank_delta()         → PHASE 2: Per-hour Spearman rank correlation
    precision_per_hour()          → PHASE 2: Per-hour Precision@K
    frequency_baseline_per_hour() → PHASE 2: Per-hour NDCG@K for the static frequency baseline

Rules (from claude.md):
  - Metrics belong here — NOT inside training loops
  - Always compare against the frequency ranker baseline
  - Always report per-zone and per-hour breakdowns (report_per_zone=True)
  - NDCG relevance definition lives in configs/eval.yaml — never hardcoded here
  - Per-class breakdown on violation type prediction if classification subtask added
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Config loader ─────────────────────────────────────────────────────────────

def load_eval_config(config_path: str | Path = "configs/eval.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"configs/eval.yaml not found at '{path.resolve()}'")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Regression metrics ────────────────────────────────────────────────────────

def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    label: str = "",
) -> dict[str, float]:
    """
    Compute MAE and RMSE for violation count regression.

    Args:
        y_true: Ground-truth violation counts.
        y_pred: Model-predicted violation counts.
        label:  Optional label for logging (e.g. model name).

    Returns:
        Dict with keys: mae, rmse.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    prefix = f"[{label}] " if label else ""
    logger.info(f"{prefix}MAE={mae:.4f}  RMSE={rmse:.4f}")
    return {"mae": mae, "rmse": rmse}


# ── Relevance labelling (graded, from eval.yaml) ──────────────────────────────

def compute_relevance(
    zone_actual_counts: pd.Series,
    eval_config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/eval.yaml",
) -> pd.Series:
    """
    Assign graded relevance labels to zones based on actual test-period violation counts.

    Relevance definition (from configs/eval.yaml ndcg_relevance):
        top_quartile (≥ 75th pct)  → 2  (highly relevant)
        second_quartile (50–75th)  → 1  (relevant)
        bottom_half (< 50th pct)   → 0  (not relevant)

    Args:
        zone_actual_counts: Series indexed by zone_id with total actual counts in test period.
        eval_config:        Pre-loaded eval.yaml dict (loads from file if None).
        config_path:        Path to eval.yaml (used only if eval_config is None).

    Returns:
        Series of relevance grades (0/1/2), same index as zone_actual_counts.
    """
    if eval_config is None:
        eval_config = load_eval_config(config_path)

    ndcg_cfg = eval_config.get("ndcg_relevance", {})
    grade_top    = int(ndcg_cfg.get("graded_relevance", {}).get("top_quartile",    2))
    grade_second = int(ndcg_cfg.get("graded_relevance", {}).get("second_quartile", 1))
    grade_bottom = int(ndcg_cfg.get("graded_relevance", {}).get("bottom_half",     0))

    q75 = zone_actual_counts.quantile(0.75)
    q50 = zone_actual_counts.quantile(0.50)

    relevance = pd.Series(grade_bottom, index=zone_actual_counts.index, name="relevance")
    relevance[zone_actual_counts >= q75] = grade_top
    relevance[(zone_actual_counts >= q50) & (zone_actual_counts < q75)] = grade_second

    n2 = int((relevance == 2).sum())
    n1 = int((relevance == 1).sum())
    n0 = int((relevance == 0).sum())
    logger.info(
        f"Relevance assigned: grade=2 → {n2} zones | grade=1 → {n1} zones | grade=0 → {n0} zones"
        f" (q50={q50:.1f}, q75={q75:.1f})"
    )
    return relevance


# ── NDCG@K ───────────────────────────────────────────────────────────────────

def ndcg_at_k(
    zone_scores: pd.Series,
    relevance: pd.Series,
    k: int = 10,
) -> float:
    """
    Compute NDCG@K for a ranked list of zones.

    Uses graded relevance (0/1/2). Both series must share the same zone_id index.

    Args:
        zone_scores: Series indexed by zone_id containing the predicted priority score.
                     Higher score = higher rank.
        relevance:   Series indexed by zone_id with graded relevance labels (0/1/2).
        k:           Number of top positions to evaluate.

    Returns:
        NDCG@K score in [0.0, 1.0].
    """
    common_idx = zone_scores.index.intersection(relevance.index)
    scores = zone_scores.loc[common_idx]
    rels   = relevance.loc[common_idx]

    # Sort zones by predicted score descending → get top-K relevance grades
    ranked_idx  = scores.sort_values(ascending=False).index[:k]
    ranked_rels = rels.loc[ranked_idx].values.astype(float)

    # Ideal ranking = sort all zones by relevance descending → top-K
    ideal_rels = np.sort(rels.values)[::-1][:k].astype(float)

    def _dcg(rels_arr: np.ndarray) -> float:
        """Compute DCG for an array of relevance grades."""
        gains = (2.0 ** rels_arr - 1.0)
        discounts = np.log2(np.arange(2, len(rels_arr) + 2))
        return float(np.sum(gains / discounts))

    dcg  = _dcg(ranked_rels)
    idcg = _dcg(ideal_rels)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return round(ndcg, 6)


# ── Precision@K ──────────────────────────────────────────────────────────────

def precision_at_k(
    zone_scores: pd.Series,
    relevance: pd.Series,
    k: int = 10,
    relevant_threshold: int = 2,
) -> float:
    """
    Compute Precision@K — fraction of top-K recommended zones that are relevant.

    A zone is "relevant" if its relevance grade >= relevant_threshold (default=2 = top quartile).

    Args:
        zone_scores:          Series indexed by zone_id with predicted priority scores.
        relevance:            Series indexed by zone_id with graded relevance (0/1/2).
        k:                    Number of top positions to consider.
        relevant_threshold:   Minimum relevance grade to count as relevant.

    Returns:
        Precision@K in [0.0, 1.0].
    """
    common_idx = zone_scores.index.intersection(relevance.index)
    scores = zone_scores.loc[common_idx]
    rels   = relevance.loc[common_idx]

    top_k_idx   = scores.sort_values(ascending=False).index[:k]
    top_k_rels  = rels.loc[top_k_idx]
    n_relevant  = int((top_k_rels >= relevant_threshold).sum())

    prec = n_relevant / min(k, len(top_k_idx))
    return round(prec, 6)


# ── Naive mean-per-zone baseline (regression benchmark) ───────────────────────

def naive_mean_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> dict[str, float]:
    """
    Naive mean-per-zone baseline: predict each zone's mean training count for every
    test row, ignoring time-of-day entirely.

    This is the honest regression benchmark. If the ML model cannot beat this,
    it has learned nothing temporal — it is equivalent to a lookup table.

    Args:
        train_df:   Training grid (must have zone_id and target_col).
        test_df:    Test grid (must have zone_id and target_col).
        target_col: Name of the violation count column.

    Returns:
        Dict with 'mae', 'rmse' of the naive baseline predictions on the test split.
    """
    zone_train_means = train_df.groupby("zone_id")[target_col].mean()
    y_pred_naive = test_df["zone_id"].map(zone_train_means).fillna(0.0).values
    y_true = test_df[target_col].values.astype(float)

    mae  = float(mean_absolute_error(y_true, y_pred_naive))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_naive)))
    logger.info(f"Naive mean-per-zone baseline: MAE={mae:.4f}  RMSE={rmse:.4f}")
    return {"mae": mae, "rmse": rmse}


# ── Frequency-ranker baseline ─────────────────────────────────────────────────

def frequency_baseline(
    train_df: pd.DataFrame,
    cis_df: pd.DataFrame,
    target_col: str = "zone_hour_violation_count",
    eval_config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/eval.yaml",
) -> pd.Series:
    """
    Compute priority scores for the frequency ranker baseline.

    Formula (from configs/eval.yaml ranker.formula):
        priority_score(zone) = historical_count(zone) × CIS(zone)

    Where historical_count is the total violation count in the TRAINING period.
    No ML — pure frequency heuristic. Must be beaten by the ML model.

    Args:
        train_df:   Training aggregated grid (zone_hour_grid or zone_day_grid).
        cis_df:     CIS table from data/processed/cis_table.parquet.
        target_col: Name of the violation count column.
        eval_config: Pre-loaded eval.yaml dict (loads from file if None).
        config_path: Path to eval.yaml.

    Returns:
        Series indexed by zone_id with baseline priority_score.
    """
    if eval_config is None:
        eval_config = load_eval_config(config_path)

    # Sum historical counts per zone over training period
    hist_counts = (
        train_df.groupby("zone_id")[target_col]
        .sum()
        .rename("historical_count")
    )

    # Merge CIS scores
    cis_lookup = cis_df.set_index("zone_id")["cis_score"]
    merged = hist_counts.to_frame().join(cis_lookup, how="left")
    merged["cis_score"] = merged["cis_score"].fillna(0.0)

    # priority_score = historical_count × CIS
    merged["priority_score"] = merged["historical_count"] * merged["cis_score"]

    logger.info(
        f"Frequency baseline: {len(merged)} zones | "
        f"top-3 zones: {merged['priority_score'].nlargest(3).to_dict()}"
    )
    return merged["priority_score"]


# ── Full evaluation runner ────────────────────────────────────────────────────

def full_eval(
    model_name: str,
    time_resolution: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    cis_df: pd.DataFrame,
    eval_config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/eval.yaml",
    k_values: list[int] | None = None,
    eval_history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """
    Run complete evaluation for one model × time_resolution combination.

    Steps:
      1. Regression metrics (MAE, RMSE)
      2. Aggregate predicted counts per zone → zone-level priority score (pred × CIS)
      3. Compute actual zone counts in test period
      4. Assign relevance grades based on actual counts
      5. NDCG@K and Precision@K for the model
      6. Frequency baseline NDCG@K and Precision@K
      7. Return structured results dict

    Args:
        model_name:       e.g. "xgboost", "lightgbm", "catboost"
        time_resolution:  "hour" or "day"
        y_true:           Ground-truth counts (test split rows).
        y_pred:           Predicted counts (test split rows).
        test_df:          Test split DataFrame (must have zone_id, target col, date).
        train_df:         Training split DataFrame (for baseline).
        cis_df:           CIS table from data/processed/cis_table.parquet.
        eval_config:      Pre-loaded eval.yaml dict.
        config_path:      Path to eval.yaml (used if eval_config is None).
        k_values:         List of K values for NDCG and Precision. Default: [5, 10].

    Returns:
        eval_results: Nested dict with all metrics and metadata.
    """
    if eval_config is None:
        eval_config = load_eval_config(config_path)

    if k_values is None:
        k_values = eval_config.get("metrics", {}).get("ranking", {}).get("k_values", [5, 10])

    target_col = (
        "zone_hour_violation_count" if time_resolution == "hour"
        else "zone_day_violation_count"
    )

    logger.info(
        f"=== Evaluating: {model_name} | {time_resolution} | target='{target_col}' ==="
    )

    # 1. Regression metrics
    reg = regression_metrics(y_true, y_pred, label=f"{model_name}/{time_resolution}")

    # 2. Naive mean-per-zone baseline (honest regression benchmark)
    naive_reg = naive_mean_baseline(train_df, test_df, target_col)
    mae_lift_vs_naive = naive_reg["mae"] - reg["mae"]   # positive = ML is better
    mae_lift_pct = (mae_lift_vs_naive / naive_reg["mae"] * 100) if naive_reg["mae"] > 0 else 0.0
    if mae_lift_vs_naive > 0:
        logger.info(
            f"  BEATS naive baseline: MAE {reg['mae']:.4f} vs {naive_reg['mae']:.4f} "
            f"({mae_lift_pct:+.1f}% improvement)"
        )
    else:
        logger.warning(
            f"  DOES NOT beat naive baseline: MAE {reg['mae']:.4f} vs naive {naive_reg['mae']:.4f} "
            f"({mae_lift_pct:+.1f}%) -- check rolling_7d_count feature"
        )

    # 3. Build zone-level predicted count (sum predictions per zone in test period)
    test_df = test_df.copy()
    test_df["_pred"] = np.asarray(y_pred, dtype=float)
    test_df["_pred"] = test_df["_pred"].clip(lower=0)  # predictions can't be negative counts

    zone_pred_counts = test_df.groupby("zone_id")["_pred"].sum()

    # 3. Actual zone counts in test period
    zone_true_counts = (
        test_df.groupby("zone_id")[target_col]
        .sum()
    )

    # 4. Graded relevance from actual counts
    relevance = compute_relevance(zone_true_counts, eval_config=eval_config)

    # 5. CIS lookup → model priority score = pred_count × CIS
    cis_lookup = cis_df.set_index("zone_id")["cis_score"]
    model_priority = zone_pred_counts * cis_lookup.reindex(zone_pred_counts.index).fillna(0.0)

    # 6. Ranking metrics for ML model
    ranking_results: dict[str, dict[str, float]] = {}
    for k in k_values:
        ranking_results[f"k{k}"] = {
            "ndcg_at_k":    ndcg_at_k(model_priority, relevance, k=k),
            "precision_at_k": precision_at_k(model_priority, relevance, k=k),
        }
        logger.info(
            f"  [{model_name}] NDCG@{k}={ranking_results[f'k{k}']['ndcg_at_k']:.4f}  "
            f"Precision@{k}={ranking_results[f'k{k}']['precision_at_k']:.4f}"
        )

    # 7. Frequency baseline
    baseline_priority = frequency_baseline(
        train_df, cis_df, target_col=target_col, eval_config=eval_config
    )
    baseline_results: dict[str, dict[str, float]] = {}
    for k in k_values:
        baseline_results[f"k{k}"] = {
            "ndcg_at_k":      ndcg_at_k(baseline_priority, relevance, k=k),
            "precision_at_k": precision_at_k(baseline_priority, relevance, k=k),
        }
        logger.info(
            f"  [freq-baseline] NDCG@{k}={baseline_results[f'k{k}']['ndcg_at_k']:.4f}  "
            f"Precision@{k}={baseline_results[f'k{k}']['precision_at_k']:.4f}"
        )

    # 9. Beat-baseline flags
    primary_k = f"k{max(k_values)}"
    beats_baseline_ndcg = (
        ranking_results[primary_k]["ndcg_at_k"] > baseline_results[primary_k]["ndcg_at_k"]
    )
    beats_baseline_prec = (
        ranking_results[primary_k]["precision_at_k"] > baseline_results[primary_k]["precision_at_k"]
    )

    if beats_baseline_ndcg and beats_baseline_prec:
        logger.info(f"  [{model_name}] BEATS frequency ranker on NDCG and Precision.")
    else:
        logger.warning(
            f"  [{model_name}] does NOT beat freq ranker on "
            f"{'NDCG' if not beats_baseline_ndcg else 'Precision'}."
        )

    # 10. Per-hour ranking metrics (Phase 2) — the primary differentiation signal
    # Aggregate NDCG@10=1.0 for both model and baseline is uninformative.
    # Per-hour NDCG shows whether the model predicts WHICH ZONES ARE HOTTEST AT
    # SPECIFIC HOURS, which is the actual downstream use case.
    logger.info(f"  [{model_name}] Computing per-hour ranking metrics ...")
    per_hour_ndcg = ndcg_per_hour(
        test_df, y_pred, cis_df, eval_config=eval_config
    )
    per_hour_spearman = temporal_rank_delta(
        test_df, y_pred, cis_df
    )
    per_hour_prec = precision_per_hour(
        test_df, y_pred, cis_df, eval_config=eval_config
    )
    baseline_per_hour_ndcg = frequency_baseline_per_hour(
        train_df, test_df, cis_df, eval_config=eval_config
    )

    beats_baseline_per_hour_ndcg = (
        per_hour_ndcg["mean_ndcg"] > baseline_per_hour_ndcg["mean_ndcg"]
    )

    # -- Model Scorecard -------------------------------------------------------
    rounds_trained = len(next(iter((eval_history or {}).values()), []))
    ndcg10 = ranking_results.get("k10", {}).get("ndcg_at_k", 0.0)
    prec10 = ranking_results.get("k10", {}).get("precision_at_k", 0.0)
    b_ndcg = baseline_results.get("k10", {}).get("ndcg_at_k", 0.0)
    b_prec = baseline_results.get("k10", {}).get("precision_at_k", 0.0)
    logger.info(
        f"\n{'─'*58}\n"
        f"  SCORECARD -- {model_name.upper()} / {time_resolution}\n"
        f"{'─'*58}\n"
        f"  Regression (test set):\n"
        f"    MAE         : {reg['mae']:.4f}\n"
        f"    RMSE        : {reg['rmse']:.4f}\n"
        f"    Naive MAE   : {naive_reg['mae']:.4f}  (zone mean, no time signal)\n"
        f"    ML lift     : {mae_lift_pct:+.1f}%  "
        f"{'[BEATS NAIVE]' if mae_lift_vs_naive > 0 else '[NO IMPROVEMENT OVER NAIVE]'}\n"
        f"  Ranking — Aggregate (test set):\n"
        f"    NDCG@10     : {ndcg10:.4f}  (freq-baseline: {b_ndcg:.4f})\n"
        f"    Prec@10     : {prec10:.4f}  (freq-baseline: {b_prec:.4f})\n"
        f"  Ranking — Per-hour [PRIMARY METRIC] (Phase 2):\n"
        f"    NDCG@10 mean: {per_hour_ndcg['mean_ndcg']:.4f}  "
        f"(baseline: {baseline_per_hour_ndcg['mean_ndcg']:.4f})  "
        f"{'[ML WINS]' if beats_baseline_per_hour_ndcg else '[BASELINE WINS]'}\n"
        f"    NDCG@10 std : {per_hour_ndcg['std_ndcg']:.4f}\n"
        f"    NDCG@10 min : {per_hour_ndcg['min_ndcg']:.4f}\n"
        f"    Spearman ρ  : {per_hour_spearman['mean_spearman']:.4f}  "
        f"(std={per_hour_spearman['std_spearman']:.4f})\n"
        f"    Prec@10 mean: {per_hour_prec['mean_precision']:.4f}\n"
        f"  Training:\n"
        f"    Rounds      : {rounds_trained if rounds_trained else 'N/A'}\n"
        f"{'─'*58}"
    )

    return {
        "model":                 model_name,
        "time_resolution":       time_resolution,
        "regression":            reg,
        "naive_baseline_reg":    naive_reg,
        "mae_lift_vs_naive_pct": round(mae_lift_pct, 4),
        "beats_naive_baseline":  bool(mae_lift_vs_naive > 0),
        "ranking":               ranking_results,
        "baseline":              baseline_results,
        "beats_baseline": {
            "ndcg":      beats_baseline_ndcg,
            "precision": beats_baseline_prec,
        },
        # Phase 2 additions — per-hour ranking (primary differentiation metrics)
        "ranking_per_hour": {
            "model_ndcg":    per_hour_ndcg,
            "model_spearman": per_hour_spearman,
            "model_prec":    per_hour_prec,
            "baseline_ndcg": baseline_per_hour_ndcg,
            "beats_baseline_per_hour_ndcg": beats_baseline_per_hour_ndcg,
        },
        "n_test_rows":  int(len(test_df)),
        "n_test_zones": int(test_df["zone_id"].nunique()),
        "eval_history": eval_history or {},
    }



# ── Save eval results ─────────────────────────────────────────────────────────

def save_eval_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save evaluation results dict to JSON (data/outputs/eval_TIMESTAMP.json).

    Args:
        results:     Eval results dict from full_eval().
        output_path: Absolute path to save the JSON file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Eval results saved → '{out}'")


# ── Per-hour ranking metrics (Phase 2) ────────────────────────────────────────
# These replace the aggregate NDCG@10=1.0 metric (which was meaningless — both
# model and baseline scored 1.0). Per-hour ranking differentiates the ML model
# from a static frequency table: zone 2 may be #1 overall but should rank lower
# at 2am. A temporal-aware model captures intra-day rank shifts.

def ndcg_per_hour(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    cis_df: pd.DataFrame,
    eval_config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/eval.yaml",
    k: int = 10,
) -> dict[str, Any]:
    """
    Compute NDCG@k for each (date × hour_of_day) slot separately, then aggregate.

    This produces meaningful differentiation between the ML model and the frequency
    baseline — the overall NDCG@10 = 1.0 (aggregate over full test period) is
    uninformative because the top-10 zones are globally stable. But per-hour,
    the ranking shifts (zone 2 peaks at 9am, zone 50 peaks at 7pm) and the ML
    model should predict these shifts better than a static frequency count.

    Args:
        test_df:     Test split DataFrame (must have zone_id, date, hour_of_day,
                     and the target column).
        y_pred:      Predicted violation counts (same row order as test_df).
        cis_df:      CIS table for computing priority_score = pred_count × CIS.
        eval_config: Pre-loaded eval.yaml dict (loads from file if None).
        config_path: Path to eval.yaml.
        k:           Number of top positions to evaluate.

    Returns:
        dict with keys:
          mean_ndcg      — mean NDCG@k across all hours (primary metric)
          std_ndcg       — standard deviation across hours
          min_ndcg       — worst-hour NDCG (reliability floor)
          n_hours        — number of unique hours evaluated
          per_hour       — list of {hour, date, ndcg} dicts (for drill-down)
    """
    if eval_config is None:
        eval_config = load_eval_config(config_path)

    test_df = test_df.copy()
    test_df["_pred"] = np.asarray(y_pred, dtype=float).clip(min=0)

    # Determine target column
    target_col = (
        "zone_hour_violation_count" if "zone_hour_violation_count" in test_df.columns
        else "zone_day_violation_count"
    )

    # CIS lookup
    cis_lookup = cis_df.set_index("zone_id")["cis_score"] if "cis_score" in cis_df.columns \
        else pd.Series(dtype=float)

    # Group by (date, hour_of_day) — if hour_of_day not present, use date only
    group_cols = ["date"] + (["hour_of_day"] if "hour_of_day" in test_df.columns else [])

    ndcg_scores: list[float] = []
    per_hour_records: list[dict[str, Any]] = []

    for group_key, group in test_df.groupby(group_cols, observed=True):
        if len(group) < 2:
            continue  # Can't rank with fewer than 2 zones

        # Zone-level predicted counts for this time slot
        zone_pred = group.groupby("zone_id")["_pred"].sum()
        zone_true = group.groupby("zone_id")[target_col].sum()

        # Priority score = pred × CIS
        if not cis_lookup.empty:
            zone_pred_priority = zone_pred * cis_lookup.reindex(zone_pred.index).fillna(0.0)
        else:
            zone_pred_priority = zone_pred

        # Relevance based on actual counts in this hour slot
        relevance = compute_relevance(zone_true, eval_config=eval_config)

        # NDCG@k for this hour
        score = ndcg_at_k(zone_pred_priority, relevance, k=k)
        ndcg_scores.append(score)

        record = {"ndcg": score}
        if isinstance(group_key, tuple):
            for col, val in zip(group_cols, group_key):
                record[col] = str(val)
        else:
            record[group_cols[0]] = str(group_key)
        per_hour_records.append(record)

    if not ndcg_scores:
        logger.warning("ndcg_per_hour: no hour groups found — returning zeros")
        return {"mean_ndcg": 0.0, "std_ndcg": 0.0, "min_ndcg": 0.0, "n_hours": 0, "per_hour": []}

    arr = np.array(ndcg_scores)
    result = {
        "mean_ndcg": float(np.mean(arr)),
        "std_ndcg":  float(np.std(arr)),
        "min_ndcg":  float(np.min(arr)),
        "n_hours":   len(arr),
        "per_hour":  per_hour_records,
    }
    logger.info(
        f"NDCG@{k} per-hour: mean={result['mean_ndcg']:.4f} "
        f"std={result['std_ndcg']:.4f} "
        f"min={result['min_ndcg']:.4f} "
        f"(n_hours={result['n_hours']})"
    )
    return result


def temporal_rank_delta(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    cis_df: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """
    Compute per-hour Spearman rank correlation between true and predicted zone rankings.

    For each (date, hour) slot, computes Spearman ρ between the true zone ranking
    (by actual count) and the model's predicted zone ranking. A higher mean ρ means
    the model is better at capturing within-slot relative zone ordering.

    Args:
        test_df: Test split DataFrame.
        y_pred:  Predicted counts (same row order as test_df).
        cis_df:  CIS table for priority_score computation.
        k:       Only consider top-k zones by true count per slot.

    Returns:
        dict with mean_spearman, std_spearman, n_hours.
    """
    from scipy.stats import spearmanr  # type: ignore[import]

    test_df = test_df.copy()
    test_df["_pred"] = np.asarray(y_pred, dtype=float).clip(min=0)

    target_col = (
        "zone_hour_violation_count" if "zone_hour_violation_count" in test_df.columns
        else "zone_day_violation_count"
    )

    cis_lookup = cis_df.set_index("zone_id")["cis_score"] if "cis_score" in cis_df.columns \
        else pd.Series(dtype=float)

    group_cols = ["date"] + (["hour_of_day"] if "hour_of_day" in test_df.columns else [])
    spearman_scores: list[float] = []

    for _, group in test_df.groupby(group_cols, observed=True):
        if len(group) < 3:
            continue

        zone_pred = group.groupby("zone_id")["_pred"].sum()
        zone_true = group.groupby("zone_id")[target_col].sum()

        # Align on common zones
        common = zone_pred.index.intersection(zone_true.index)
        if len(common) < 3:
            continue

        if not cis_lookup.empty:
            zone_pred_priority = zone_pred.loc[common] * cis_lookup.reindex(common).fillna(0.0)
        else:
            zone_pred_priority = zone_pred.loc[common]

        rho, _ = spearmanr(zone_true.loc[common].values, zone_pred_priority.values)
        if not np.isnan(rho):
            spearman_scores.append(float(rho))

    if not spearman_scores:
        return {"mean_spearman": 0.0, "std_spearman": 0.0, "n_hours": 0}

    arr = np.array(spearman_scores)
    result = {
        "mean_spearman": float(np.mean(arr)),
        "std_spearman":  float(np.std(arr)),
        "n_hours":       len(arr),
    }
    logger.info(
        f"Temporal Spearman ρ: mean={result['mean_spearman']:.4f} "
        f"std={result['std_spearman']:.4f} "
        f"(n_hours={result['n_hours']})"
    )
    return result


def precision_per_hour(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    cis_df: pd.DataFrame,
    eval_config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/eval.yaml",
    k: int = 10,
) -> dict[str, float]:
    """
    Compute per-hour Precision@k (fraction of top-k predicted zones in true top-k).

    Args:
        test_df:     Test split DataFrame.
        y_pred:      Predicted counts (same row order as test_df).
        cis_df:      CIS table.
        eval_config: Pre-loaded eval.yaml.
        config_path: Path to eval.yaml.
        k:           Top-k threshold.

    Returns:
        dict with mean_precision, std_precision, n_hours.
    """
    if eval_config is None:
        eval_config = load_eval_config(config_path)

    test_df = test_df.copy()
    test_df["_pred"] = np.asarray(y_pred, dtype=float).clip(min=0)

    target_col = (
        "zone_hour_violation_count" if "zone_hour_violation_count" in test_df.columns
        else "zone_day_violation_count"
    )

    cis_lookup = cis_df.set_index("zone_id")["cis_score"] if "cis_score" in cis_df.columns \
        else pd.Series(dtype=float)

    group_cols = ["date"] + (["hour_of_day"] if "hour_of_day" in test_df.columns else [])
    precision_scores: list[float] = []

    for _, group in test_df.groupby(group_cols, observed=True):
        if len(group) < 2:
            continue

        zone_pred = group.groupby("zone_id")["_pred"].sum()
        zone_true = group.groupby("zone_id")[target_col].sum()

        if not cis_lookup.empty:
            zone_pred_priority = zone_pred * cis_lookup.reindex(zone_pred.index).fillna(0.0)
        else:
            zone_pred_priority = zone_pred

        relevance = compute_relevance(zone_true, eval_config=eval_config)
        prec = precision_at_k(zone_pred_priority, relevance, k=k, relevant_threshold=2)
        precision_scores.append(prec)

    if not precision_scores:
        return {"mean_precision": 0.0, "std_precision": 0.0, "n_hours": 0}

    arr = np.array(precision_scores)
    result = {
        "mean_precision": float(np.mean(arr)),
        "std_precision":  float(np.std(arr)),
        "n_hours":        len(arr),
    }
    logger.info(
        f"Precision@{k} per-hour: mean={result['mean_precision']:.4f} "
        f"std={result['std_precision']:.4f} "
        f"(n_hours={result['n_hours']})"
    )
    return result


def frequency_baseline_per_hour(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cis_df: pd.DataFrame,
    eval_config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/eval.yaml",
    k: int = 10,
) -> dict[str, Any]:
    """
    Compute per-hour NDCG@k for the frequency baseline (no ML).

    The frequency baseline ranks zones by historical count × CIS — it is completely
    static (ignores time-of-day). Per-hour, it will perform worse than the ML model
    on slots where the historical ranking doesn't match the actual hour's ranking.

    Args:
        train_df:    Training split DataFrame.
        test_df:     Test split DataFrame.
        cis_df:      CIS table.
        eval_config: Pre-loaded eval.yaml.
        config_path: Path to eval.yaml.
        k:           Top-k threshold.

    Returns:
        dict with mean_ndcg, std_ndcg, min_ndcg, n_hours.
    """
    if eval_config is None:
        eval_config = load_eval_config(config_path)

    target_col = (
        "zone_hour_violation_count" if "zone_hour_violation_count" in test_df.columns
        else "zone_day_violation_count"
    )

    # Frequency baseline: static historical count × CIS (same score for every hour)
    baseline_priority = frequency_baseline(
        train_df, cis_df, target_col=target_col, eval_config=eval_config
    )

    cis_lookup = cis_df.set_index("zone_id")["cis_score"] if "cis_score" in cis_df.columns \
        else pd.Series(dtype=float)

    group_cols = ["date"] + (["hour_of_day"] if "hour_of_day" in test_df.columns else [])
    ndcg_scores: list[float] = []

    for _, group in test_df.groupby(group_cols, observed=True):
        if len(group) < 2:
            continue

        zone_true = group.groupby("zone_id")[target_col].sum()
        relevance = compute_relevance(zone_true, eval_config=eval_config)

        # Baseline priority is static — same for all hours
        common_idx = baseline_priority.index.intersection(relevance.index)
        if len(common_idx) < 2:
            continue

        score = ndcg_at_k(baseline_priority.loc[common_idx], relevance.loc[common_idx], k=k)
        ndcg_scores.append(score)

    if not ndcg_scores:
        return {"mean_ndcg": 0.0, "std_ndcg": 0.0, "min_ndcg": 0.0, "n_hours": 0}

    arr = np.array(ndcg_scores)
    result = {
        "mean_ndcg": float(np.mean(arr)),
        "std_ndcg":  float(np.std(arr)),
        "min_ndcg":  float(np.min(arr)),
        "n_hours":   len(arr),
    }
    logger.info(
        f"Freq-baseline per-hour NDCG@{k}: mean={result['mean_ndcg']:.4f} "
        f"std={result['std_ndcg']:.4f} min={result['min_ndcg']:.4f}"
    )
    return result

