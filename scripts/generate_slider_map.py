import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger
from src.inference.ranker import load_ranker, rank_zones
from src.inference.static_output import build_zone_centroids, generate_static_output_with_slider

def main():
    logger.info("Loading ranker...")
    ranker = load_ranker(project_root=PROJECT_ROOT)
    
    # Load eval_metrics from checkpoint
    eval_metrics = None
    try:
        import json
        ckpt_dir_path = ranker.get("ckpt_dir")
        if ckpt_dir_path is not None:
            meta_path = Path(ckpt_dir_path) / "training_meta.json"
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                eval_metrics = meta.get("metrics", None)
                if eval_metrics:
                    logger.info(f"Loaded training metrics successfully.")
    except Exception as exc:
        logger.warning(f"Could not load eval_metrics: {exc}")
        
    target_date = "2024-03-18"
    logger.info(f"Ranking zones for all 24 hours of {target_date}...")
    
    all_hours_data = {}
    for hour in range(24):
        hour_df = rank_zones(ranker, target_date=target_date, target_hour=hour, top_k=10)
        all_hours_data[hour] = hour_df
        
    logger.info("Computing centroids...")
    centroids_df = build_zone_centroids(
        PROJECT_ROOT / "data" / "processed" / "features_with_zones.parquet"
    )
    
    output_path = PROJECT_ROOT / "docs" / "index.html"
    logger.info(f"Generating time-slider output to {output_path}...")
    generate_static_output_with_slider(
        all_hours_data  = all_hours_data,
        centroids_df    = centroids_df,
        target_date     = target_date,
        output_path     = output_path,
        model_name      = ranker["model_name"],
        time_resolution = ranker["time_resolution"],
        eval_metrics    = eval_metrics,
    )
    logger.info("Time-slider output generation complete.")

if __name__ == "__main__":
    main()
