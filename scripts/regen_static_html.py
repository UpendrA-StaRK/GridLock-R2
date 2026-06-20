from src.inference.ranker import load_ranker, rank_zones
from src.inference.static_output import build_zone_centroids, generate_static_output
from pathlib import Path
import json
import glob

project_root = Path('.')
ranker = load_ranker(project_root)
top10 = rank_zones(ranker, target_date='2024-03-18', target_hour=9, top_k=10)
centroids_df = build_zone_centroids(project_root / 'data' / 'processed' / 'features_with_zones.parquet')

html_path = project_root / 'data' / 'outputs' / 'enforcement_priority_2024-03-18_09h.html'

eval_files = sorted(glob.glob(str(project_root / 'data' / 'outputs' / 'eval_*.json')), reverse=True)
eval_metrics = None
if eval_files:
    with open(eval_files[0], 'r', encoding='utf-8') as f:
        ev_data = json.load(f)
    model_key = f"{ranker['model_name']}_{ranker['time_resolution']}"
    eval_metrics = ev_data.get(model_key)

generate_static_output(
    top_k_df=top10,
    centroids_df=centroids_df,
    target_date='2024-03-18',
    target_hour=9,
    output_path=html_path,
    model_name=ranker['model_name'],
    time_resolution=ranker['time_resolution'],
    eval_metrics=eval_metrics
)
print('Static HTML regenerated successfully.')
