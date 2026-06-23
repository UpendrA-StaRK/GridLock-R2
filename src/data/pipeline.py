"""
src/data/pipeline.py
GridLock R2 — PS1: Parking-Induced Congestion

End-to-end orchestrator. Runs all 8 pipeline steps in sequence:

  Step 1: Schema validation     (src/data/validate.py)
  Step 2: Data ingest           (src/data/load.py)
  Step 3: Feature engineering   (src/data/features.py)
  Step 4: DBSCAN clustering     (src/models/clustering.py)
  Step 5: CIS computation       (src/models/clustering.py)
  Step 6: Zone-grid aggregation (src/data/features.py)
  Step 7: Model Training        (src/training/train.py)
  Step 8: Inference / ranking   (src/inference/ranker.py + static_output.py)

Usage:
    python -m src.data.pipeline                   # full run, default paths
    python -m src.data.pipeline --skip-training   # skip training, load existing checkpoint
    python -m src.data.pipeline --date 2024-03-18 --hour 9  # custom inference target

Rules (from claude.md):
  - All params from configs/ — never hardcoded here
  - Each step is fault-tolerant: wraps in try/except, logs clearly, exits with non-zero on failure
  - tqdm progress on every step
  - loguru for all logging
  - Judges may ask to run this live — it must complete without errors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(log_file: Path | None = None) -> None:
    # Fix Windows cp1252 terminal — reconfigure stdout to UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
        colorize=False,
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_file), level="DEBUG", rotation="10 MB")
        logger.info(f"Logging to file: '{log_file}'")


# ── Step wrappers ─────────────────────────────────────────────────────────────

def _step(name: str, step_num: int, total: int) -> None:
    """Print a visible step header."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Step {step_num}/{total}: {name}")
    logger.info(f"{'='*60}")


def step1_validate(project_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Step 1 — Schema validation."""
    # Call validate_schema.
    # validate_schema(df, eval_cfg, strict=True) is the actual exported signature.
    from src.data.validate import validate_schema, load_eval_config, save_validation_report

    csv_path    = project_root / "data" / "raw" / cfg["raw_filename"]
    report_path = project_root / "data" / "processed" / "validation_report.json"
    eval_cfg    = load_eval_config(project_root / "configs" / "eval.yaml")

    import pandas as pd
    df_raw = pd.read_csv(csv_path, dtype=str, low_memory=False)
    logger.info(f"Raw file loaded for validation: {len(df_raw):,} rows")

    report = validate_schema(df_raw, eval_cfg=eval_cfg, strict=True)
    save_validation_report(report, report_path)
    return {"validation_report": report, "csv_path": csv_path}


def step2_ingest(project_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Step 2 — Data ingestion + dedup."""
    # load_raw() loads data without metadata export.
    # Correct signature: load_raw(csv_path, eval_config_path, save_report, report_path)
    from src.data.load import load_raw, save_load_metadata

    csv_path    = project_root / "data" / "raw" / cfg["raw_filename"]
    meta_path   = project_root / "data" / "processed" / "load_metadata.json"
    eval_cfg_path = project_root / "configs" / "eval.yaml"

    df, metadata = load_raw(
        csv_path,
        eval_config_path = eval_cfg_path,
        save_report      = True,
        report_path      = project_root / "data" / "processed" / "validation_report.json",
    )
    save_load_metadata(metadata, meta_path)
    logger.info(f"Ingested: {len(df):,} rows × {len(df.columns)} cols")
    return {"df": df, "load_metadata": metadata}


def step3_features(project_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Step 3 — Row-level feature engineering."""
    from src.data.features import extract_row_features, save_feature_metadata

    df = state["df"]
    feat_path = project_root / "data" / "processed" / "features_row_level.parquet"
    enc_path  = project_root / "data" / "processed" / "label_encoders.pkl"
    meta_path = project_root / "data" / "processed" / "feature_metadata.json"

    df_feat, encoders, feat_meta = extract_row_features(
        df,
        features_config_path=project_root / "configs" / "features.yaml",
        encoder_save_path=enc_path,
    )
    save_feature_metadata(feat_meta, meta_path)
    df_feat.to_parquet(feat_path, index=False)
    logger.info(f"Features saved: {feat_path} ({feat_path.stat().st_size / 1e6:.1f} MB)")
    return {"df_feat": df_feat, "encoders": encoders, "feat_meta": feat_meta}


def step4_cluster(project_root: Path, state: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Step 4 — DBSCAN clustering on full dataset."""
    from src.models.clustering import run_clustering, save_cluster_stats

    df_feat = state["df_feat"]
    eps     = model_cfg["dbscan"]["eps"]
    min_s   = model_cfg["dbscan"]["min_samples"]
    seed    = model_cfg.get("seed", 42)

    df_zoned, cluster_stats = run_clustering(df_feat, eps=eps, min_samples=min_s, random_state=seed)
    stats_path = project_root / "data" / "processed" / "cluster_stats.json"
    save_cluster_stats(cluster_stats, stats_path)

    zones_path = project_root / "data" / "processed" / "features_with_zones.parquet"
    df_zoned.to_parquet(zones_path, index=False)
    logger.info(
        f"Clustering complete: {cluster_stats['n_clusters']} clusters | "
        f"{cluster_stats['noise_pct']}% noise | features_with_zones saved"
    )
    return {"df_zoned": df_zoned, "cluster_stats": cluster_stats}


def step5_cis(project_root: Path, state: dict[str, Any], eval_cfg: dict[str, Any]) -> dict[str, Any]:
    """Step 5 — CIS computation per zone."""
    from src.models.clustering import compute_cis, save_cis_table

    df_zoned = state["df_zoned"]
    cis_df   = compute_cis(df_zoned, eval_cfg)
    cis_path = project_root / "data" / "processed" / "cis_table.parquet"
    save_cis_table(cis_df, cis_path)
    logger.info(f"CIS computed: {len(cis_df)} zones | saved → '{cis_path}'")
    return {"cis_df": cis_df}


def step6_grids(project_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Step 6 — Zone × Hour and Zone × Day aggregation."""
    from src.data.features import aggregate_to_zone_grid

    df_zoned = state["df_zoned"]
    hour_path = project_root / "data" / "processed" / "zone_hour_grid.parquet"
    day_path  = project_root / "data" / "processed" / "zone_day_grid.parquet"

    zone_hour_df = aggregate_to_zone_grid(df_zoned, time_resolution="hour", save_path=hour_path)
    zone_day_df  = aggregate_to_zone_grid(df_zoned, time_resolution="day",  save_path=day_path)

    logger.info(
        f"Grids saved: zone×hour={len(zone_hour_df):,} rows | zone×day={len(zone_day_df):,} rows"
    )
    return {"zone_hour_df": zone_hour_df, "zone_day_df": zone_day_df}




def step7_train(project_root: Path) -> dict[str, Any]:
    """Step 7 — Train predictive models."""
    from src.training.train import run_training
    results = run_training(project_root)
    return {"training_results": results, "winner": results.get("winner", {})}


def step8_infer(
    project_root: Path,
    state: dict[str, Any],
    target_date: str,
    target_hour: int,
    top_k: int = 10,
    ckpt_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Step 8 — Inference: rank zones and generate static HTML output."""
    import json as _json
    from src.inference.ranker import load_ranker, rank_zones, rank_day_schedule
    from src.inference.static_output import build_zone_centroids, generate_static_output

    # Load ranker (auto-discovers winner from model.yaml or uses explicit ckpt_dir)
    ranker = load_ranker(project_root=project_root, ckpt_dir=ckpt_dir)

    # Load eval_metrics from the most recent eval JSON (for scorecard display)
    import glob
    eval_metrics: dict[str, Any] | None = None
    try:
        eval_files = sorted(glob.glob(str(project_root / "data" / "outputs" / "eval_*.json")), reverse=True)
        if eval_files:
            with open(eval_files[0], "r", encoding="utf-8") as _f:
                ev_data = _json.load(_f)
            model_key = f"{ranker['model_name']}_{ranker['time_resolution']}"
            eval_metrics = ev_data.get(model_key)
            if eval_metrics:
                logger.info(f"Eval metrics loaded from '{Path(eval_files[0]).name}' for scorecard")
    except Exception as _exc:
        logger.warning(f"Could not load eval_metrics for scorecard: {_exc}")

    # Rank zones for requested date/hour
    top_k_df = rank_zones(ranker, target_date=target_date, target_hour=target_hour, top_k=top_k)

    # Build zone centroids
    centroids_df = build_zone_centroids(
        project_root / "data" / "processed" / "features_with_zones.parquet"
    )

    # Generate static HTML output
    html_path = (
        project_root / "data" / "outputs"
        / f"enforcement_priority_{target_date}_{target_hour:02d}h.html"
    )
    generate_static_output(
        top_k_df        = top_k_df,
        centroids_df    = centroids_df,
        target_date     = target_date,
        target_hour     = target_hour,
        output_path     = html_path,
        model_name      = ranker["model_name"],
        time_resolution = ranker["time_resolution"],
        eval_metrics    = eval_metrics,
    )

    # Generate 24-hour interactive slider map (Demo Dashboard)
    from src.inference.static_output import generate_static_output_with_slider
    
    logger.info("Updating frontend dashboard scorecard with latest model metrics...")
    
    slider_html_path = project_root / "docs" / "index.html"
    generate_static_output_with_slider(
        all_dates_hours_data={},
        centroids_df=centroids_df,
        target_dates=[],
        output_path=slider_html_path,
        model_name=ranker["model_name"],
        time_resolution=ranker["time_resolution"],
        eval_metrics=eval_metrics,
    )

    # Generate day schedule
    schedule_df = rank_day_schedule(ranker, target_date=target_date, top_k=5)
    csv_path = project_root / "data" / "outputs" / f"day_schedule_{target_date}.csv"
    schedule_df.to_csv(csv_path, index=False)

    logger.info(
        f"Inference complete | Top zone: {int(top_k_df.iloc[0]['zone_id'])} | "
        f"score={top_k_df.iloc[0]['priority_score']:.4f} | "
        f"HTML: '{html_path.name}'"
    )
    return {
        "top_k_df":    top_k_df,
        "schedule_df": schedule_df,
        "html_path":   html_path,
        "slider_html_path": slider_html_path,
        "csv_path":    csv_path,
    }


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_pipeline(
    project_root: str | Path = ".",
    target_date:  str | None = None,
    target_hour:  int = 9,
    top_k:        int = 10,
    skip_training: bool = False,
    skip_clustering: bool = False,
    skip_features: bool = False,
    ckpt_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run the full GridLock R2 pipeline end-to-end.

    Steps:
      1. Schema validation
      2. Data ingest + dedup
      3. Feature engineering (row level)
      4. DBSCAN clustering → zone_id assignment
      5. CIS computation per zone
      6. Zone × Hour + Zone × Day grid aggregation
      7. Inference → top-K enforcement ranking + static HTML + day schedule

    Args:
        project_root:    Project root directory (GridLock R2/).
        target_date:     Date for inference output (YYYY-MM-DD).
        target_hour:     Hour for inference output [0–23].
        top_k:           Top-K zones to rank in output.
        skip_training:   If True, skip Step 7 (use existing checkpoint).
        skip_clustering: If True, skip Steps 4–6 (use existing parquet files).
        skip_features:   If True, skip Steps 1–3 (use existing features_row_level.parquet).
        ckpt_dir:        Explicit checkpoint directory path to use for inference.

    Returns:
        state: Dict containing outputs of all pipeline steps.
    """
    import yaml

    project_root = Path(project_root)
    _setup_logging(project_root / "data" / "outputs" / "pipeline.log")

    t0 = time.time()
    logger.info(
        f"\n{'#'*60}\n"
        f"  GridLock R2 — End-to-End Pipeline\n"
        f"  project_root = {project_root}\n"
        f"  target       = {target_date} hour={target_hour}\n"
        f"{'#'*60}"
    )

    # Load configs
    with open(project_root / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open(project_root / "configs" / "eval.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    # Raw filename (assume single CSV in data/raw/)
    raw_files = list((project_root / "data" / "raw").glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError("No CSV found in data/raw/")
    cfg = {"raw_filename": raw_files[0].name}
    logger.info(f"Raw data file: {cfg['raw_filename']}")

    # ── Dynamic Split Configuration ─────────────────────────────────────────
    if not skip_features:  # Only auto-update if we are starting fresh from the raw file
        logger.info("Auto-configuring train/test split dates from dataset...")
        try:
            import pandas as pd
            import re
            
            df_dt = pd.read_csv(raw_files[0], usecols=["created_datetime"], low_memory=False)
            dt_series = pd.to_datetime(df_dt["created_datetime"], errors="coerce", utc=True).dropna()
            
            if not dt_series.empty:
                dt_min = dt_series.min()
                dt_max = dt_series.max()
                total_days = (dt_max - dt_min).days
                
                # 70% Train / 30% Test split
                train_days = int(total_days * 0.70)
                train_end_dt = dt_min + pd.Timedelta(days=train_days)
                test_start_dt = train_end_dt + pd.Timedelta(days=1)
                
                split_cfg = {
                    "train_start": dt_min.strftime("%Y-%m-%d"),
                    "train_end": train_end_dt.strftime("%Y-%m-%d"),
                    "test_start": test_start_dt.strftime("%Y-%m-%d"),
                    "test_end": dt_max.strftime("%Y-%m-%d")
                }
                
                logger.info(f"Detected dataset range: {split_cfg['train_start']} to {split_cfg['test_end']}")
                logger.info(f"Computed Split -> Train: up to {split_cfg['train_end']} | Test: from {split_cfg['test_start']}")
                
                # Update eval.yaml using regex to preserve comments
                eval_yaml_path = project_root / "configs" / "eval.yaml"
                with open(eval_yaml_path, "r", encoding="utf-8") as f:
                    eval_content = f.read()
                
                eval_content = re.sub(r'train_start:\s*".*?"', f'train_start: "{split_cfg["train_start"]}"', eval_content)
                eval_content = re.sub(r'train_end:\s*".*?"',   f'train_end:   "{split_cfg["train_end"]}"', eval_content)
                eval_content = re.sub(r'test_start:\s*".*?"',  f'test_start:  "{split_cfg["test_start"]}"', eval_content)
                eval_content = re.sub(r'test_end:\s*".*?"',    f'test_end:    "{split_cfg["test_end"]}"', eval_content)
                
                with open(eval_yaml_path, "w", encoding="utf-8") as f:
                    f.write(eval_content)
                
                # Reload eval_cfg into memory since we just updated the file on disk
                with open(eval_yaml_path, "r", encoding="utf-8") as f:
                    eval_cfg = yaml.safe_load(f)
            else:
                logger.warning("Could not find valid dates to auto-configure split.")
        except Exception as e:
            logger.warning(f"Auto-split configuration failed: {e}")

    if target_date is None:
        target_date = eval_cfg.get("split", {}).get("test_start", "2024-03-01")
        logger.info(f"No target_date provided. Defaulting to first test day: {target_date}")

    state: dict[str, Any] = {}
    total_steps = 8

    STEPS = [
        ("Schema Validation",        1),
        ("Data Ingest",              2),
        ("Feature Engineering",      3),
        ("DBSCAN Clustering",        4),
        ("CIS Computation",          5),
        ("Zone Grid Aggregation",    6),
        ("Model Training",           7),
        ("Inference & Output",       8),
    ]

    for step_name, step_num in tqdm(STEPS, desc="Pipeline", unit="step"):
        _step(step_name, step_num, total_steps)
        t_step = time.time()

        try:
            if step_num == 1:
                if skip_features:
                    logger.info("Skipping Step 1 (skip_features=True)")
                else:
                    state.update(step1_validate(project_root, cfg))

            elif step_num == 2:
                if skip_features:
                    logger.info("Skipping Step 2 (skip_features=True)")
                else:
                    state.update(step2_ingest(project_root, cfg))

            elif step_num == 3:
                if skip_features:
                    logger.info("Skipping Step 3 — loading existing features_row_level.parquet")
                    import pandas as pd
                    feat_path = project_root / "data" / "processed" / "features_row_level.parquet"
                    state["df_feat"] = pd.read_parquet(feat_path)
                else:
                    state.update(step3_features(project_root, state))

            elif step_num == 4:
                if skip_clustering:
                    logger.info("Skipping Step 4 — loading existing features_with_zones.parquet")
                    import pandas as pd
                    state["df_zoned"] = pd.read_parquet(
                        project_root / "data" / "processed" / "features_with_zones.parquet"
                    )
                else:
                    state.update(step4_cluster(project_root, state, model_cfg))

            elif step_num == 5:
                if skip_clustering:
                    logger.info("Skipping Step 5 — loading existing cis_table.parquet")
                    import pandas as pd
                    state["cis_df"] = pd.read_parquet(
                        project_root / "data" / "processed" / "cis_table.parquet"
                    )
                else:
                    state.update(step5_cis(project_root, state, eval_cfg))

            elif step_num == 6:
                if skip_clustering:
                    logger.info("Skipping Step 6 — loading existing zone grids")
                    import pandas as pd
                    state["zone_hour_df"] = pd.read_parquet(
                        project_root / "data" / "processed" / "zone_hour_grid.parquet"
                    )
                    state["zone_day_df"] = pd.read_parquet(
                        project_root / "data" / "processed" / "zone_day_grid.parquet"
                    )
                else:
                    state.update(step6_grids(project_root, state))

            elif step_num == 7:
                if skip_training:
                    logger.info("Skipping Step 7 (skip_training=True) — using existing checkpoints")
                else:
                    state.update(step7_train(project_root))

            elif step_num == 8:
                state.update(step8_infer(project_root, state, target_date, target_hour, top_k, ckpt_dir))

        except Exception as exc:
            logger.error(f"Step {step_num} FAILED: {exc}")
            raise

        elapsed = time.time() - t_step
        logger.info(f"  Step {step_num} complete in {elapsed:.1f}s")

    total_elapsed = time.time() - t0
    logger.info(
        f"\n{'#'*60}\n"
        f"  [OK] Pipeline complete in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)\n"
        f"{'#'*60}"
    )

    # Print final summary
    _print_summary(state, target_date, target_hour)
    return state


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(state: dict[str, Any], target_date: str, target_hour: int) -> None:
    top = state.get("top_k_df")
    winner = state.get("winner", {})

    print("\n" + "="*60)
    print("  GRIDLOCK R2 — PIPELINE SUMMARY")
    print("="*60)
    if "load_metadata" in state:
        meta = state["load_metadata"]
        print(f"  Raw rows ingested   : {meta.get('rows_after_dedup', '?'):,}")
    if "cluster_stats" in state:
        cs = state["cluster_stats"]
        print(f"  DBSCAN clusters     : {cs.get('n_clusters', '?')}")
        print(f"  Noise pct           : {cs.get('noise_pct', '?')}%")
    if winner:
        print(f"  Winner model        : {winner.get('run', '?')}")
        print(f"  NDCG@10             : {winner.get('NDCG@10', '?'):.4f}")
        print(f"  MAE                 : {winner.get('MAE', '?'):.4f}")
    if top is not None and len(top) > 0:
        row = top.iloc[0]
        print(f"  Top enforcement zone: Zone {int(row['zone_id'])} "
              f"(score={row['priority_score']:.4f}, tier={row['priority_tier']})")
    if "html_path" in state:
        print(f"  Demo HTML           : {state['html_path'].name}")
    if "slider_html_path" in state:
        print(f"  Interactive Demo    : {state['slider_html_path'].name} (in docs/)")
    if "csv_path" in state:
        print(f"  Schedule CSV        : {state['csv_path'].name}")
    print("="*60 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GridLock R2 — End-to-End Pipeline (PS1: Parking-Induced Congestion)"
    )
    parser.add_argument(
        "--date", default=None,
        help="Target date for inference output (YYYY-MM-DD). Default: test_start from eval.yaml"
    )
    parser.add_argument(
        "--hour", type=int, default=9,
        help="Target hour for inference output [0-23]. Default: 9"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top enforcement zones to output. Default: 10"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training and use existing checkpoint."
    )
    parser.add_argument(
        "--skip-clustering", action="store_true",
        help="Skip clustering + grid aggregation and use existing parquet files."
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip validation + ingest + feature engineering and use existing features_row_level.parquet."
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default=None,
        help="Explicit checkpoint directory to load for inference."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        project_root     = Path(__file__).resolve().parent.parent.parent,
        target_date      = args.date,
        target_hour      = args.hour,
        top_k            = args.top_k,
        skip_training    = args.skip_training,
        skip_clustering  = args.skip_clustering,
        skip_features    = args.skip_features,
        ckpt_dir         = args.ckpt_dir,
    )
