"""
src/inference/static_output.py
GridLock R2 — PS1: Parking-Induced Congestion

Static HTML output generator — demo fallback.

Produces a self-contained HTML file containing:
  1. A ranked enforcement priority table (top-K zones)
  2. A Folium map of Bengaluru with zone markers coloured by priority tier
     (HIGH=red, MEDIUM=orange, LOW=green) and CIS/count annotations
  3. A time-of-day bar chart (if day-schedule mode)

This is the FALLBACK output — build this alongside the ranker so that if
Streamlit crashes during the demo, we can serve this static file instead.

Rules (from claude.md):
  - No training logic here
  - All styling via inline CSS only (self-contained, no external CDN dependency)
  - Folium map embedded as iframe inside the HTML
  - Must work offline (judges' venue may have no internet)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


# ── Zone centroid lookup ──────────────────────────────────────────────────────

def build_zone_centroids(
    features_with_zones_path: str | Path,
) -> pd.DataFrame:
    """
    Compute the geographic centroid (mean lat/lon) of each DBSCAN zone.

    Args:
        features_with_zones_path: Path to data/processed/features_with_zones.parquet

    Returns:
        DataFrame with columns: zone_id, lat_centroid, lon_centroid
    """
    df = pd.read_parquet(features_with_zones_path, columns=["zone_id", "latitude", "longitude"])
    centroids = (
        df.groupby("zone_id")
        .agg(lat_centroid=("latitude", "mean"), lon_centroid=("longitude", "mean"))
        .reset_index()
    )
    logger.info(f"Zone centroids computed: {len(centroids)} zones")
    return centroids


# ── Folium map builder ────────────────────────────────────────────────────────

def build_folium_map(
    top_k_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    target_date: str,
    target_hour: int | None,
    map_zoom: int = 12,
) -> str:
    """
    Build a Folium map with zone markers coloured by priority tier.

    Marker colours:
        HIGH   → red
        MEDIUM → orange
        LOW    → green

    Each marker popup shows:
        Zone ID, Rank, Priority Score, Predicted Count, CIS Score, Tier

    Args:
        top_k_df:    Output of rank_zones() — ranked top-K zones.
        centroids_df: Zone centroid lookup (from build_zone_centroids()).
        target_date: Date string for map title.
        target_hour: Hour for map title (None if day resolution).
        map_zoom:    Initial zoom level (12 = city scale).

    Returns:
        html_str: Self-contained HTML string of the Folium map.
    """
    try:
        import folium
    except ImportError:
        raise ImportError(
            "folium is not installed. Run: venv\\Scripts\\python.exe -m pip install folium"
        )

    # Bengaluru city centre
    BENGALURU_LAT = 12.9716
    BENGALURU_LON = 77.5946

    hour_label = f" — Hour {target_hour:02d}:00" if target_hour is not None else ""
    title = f"GridLock R2 — Enforcement Priority Map | {target_date}{hour_label}"

    m = folium.Map(
        location=[BENGALURU_LAT, BENGALURU_LON],
        zoom_start=map_zoom,
        tiles="OpenStreetMap",
    )

    colour_map = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}

    # Merge rank table with centroids
    merged = top_k_df.reset_index().merge(centroids_df, on="zone_id", how="left")

    for _, row in merged.iterrows():
        if pd.isna(row.get("lat_centroid")) or pd.isna(row.get("lon_centroid")):
            continue

        tier   = str(row.get("priority_tier", "LOW"))
        colour = colour_map.get(tier, "blue")
        rank   = int(row.get("rank", 0))

        popup_html = f"""
        <div style='font-family:Arial; font-size:13px; min-width:180px'>
          <b>Rank #{rank} — Zone {int(row['zone_id'])}</b><br>
          <hr style='margin:4px 0'>
          Priority Score : <b>{row['priority_score']:.4f}</b><br>
          Tier           : <b style='color:{colour}'>{tier}</b><br>
          Predicted Count: {row['predicted_count']:.1f}<br>
          CIS Score      : {row['cis_score']:.4f}<br>
          Junction       : {'Yes' if row.get('has_junction') else 'No'}
        </div>
        """

        folium.CircleMarker(
            location=[row["lat_centroid"], row["lon_centroid"]],
            radius=12 + (10 - rank),   # top rank = bigger marker
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"#{rank} Zone {int(row['zone_id'])} ({tier})",
        ).add_to(m)

        # Add rank number label
        folium.Marker(
            location=[row["lat_centroid"], row["lon_centroid"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:bold;color:white;'
                     f'text-align:center;line-height:20px">#{rank}</div>',
                icon_size=(20, 20),
                icon_anchor=(10, 10),
            ),
        ).add_to(m)

    map_html = m._repr_html_()
    return map_html


# ── Priority table HTML ───────────────────────────────────────────────────────

def _build_table_html(top_k_df: pd.DataFrame) -> str:
    """Build a styled HTML table from the top-K ranked zones DataFrame."""
    tier_colours = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#27ae60"}

    rows_html = ""
    for rank, row in top_k_df.reset_index().iterrows():
        tier   = str(row.get("priority_tier", "LOW"))
        colour = tier_colours.get(tier, "#555")
        rows_html += f"""
        <tr>
          <td style='text-align:center;font-weight:bold'>{int(row.get('rank', rank+1))}</td>
          <td style='text-align:center'>{int(row['zone_id'])}</td>
          <td style='text-align:center;font-weight:bold;color:{colour}'>{tier}</td>
          <td style='text-align:right'>{row['priority_score']:.4f}</td>
          <td style='text-align:right'>{row['predicted_count']:.1f}</td>
          <td style='text-align:right'>{row['cis_score']:.4f}</td>
          <td style='text-align:center'>{'✓' if row.get('has_junction') else '—'}</td>
        </tr>"""

    table_html = f"""
    <div class="table-wrapper">
      <table style='border-collapse:collapse; width:100%; font-family:Arial; font-size:14px'>
        <thead>
          <tr style='background:#2c3e50; color:white'>
            <th style='padding:8px'>Rank</th>
            <th style='padding:8px'>Zone ID</th>
            <th style='padding:8px'>Priority</th>
            <th style='padding:8px'>Priority Score</th>
            <th style='padding:8px'>Predicted Count</th>
            <th style='padding:8px'>CIS Score</th>
            <th style='padding:8px'>Junction</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>"""

    return table_html


# ── Model scorecard HTML ─────────────────────────────────────────────────────

def _build_scorecard_html(
    eval_metrics: dict | None,
    model_name: str,
    time_resolution: str,
) -> str:
    """Build a styled model evaluation scorecard HTML block."""
    if not eval_metrics:
        return (
            "<p style='color:#999;font-size:13px'>No evaluation metrics available — "
            "run training first to populate scores.</p>"
        )

    reg      = eval_metrics.get("regression", {})
    naive    = eval_metrics.get("naive_baseline_reg", {})
    ranking  = eval_metrics.get("ranking", {})
    baseline = eval_metrics.get("baseline", {})
    lift_pct = eval_metrics.get("mae_lift_vs_naive_pct", 0.0)
    beats_naive   = eval_metrics.get("beats_naive_baseline", False)
    rounds   = len(next(iter(eval_metrics.get("eval_history", {}).values()), []))

    mae      = reg.get("mae",  float("nan"))
    rmse     = reg.get("rmse", float("nan"))
    naive_mae = naive.get("mae", float("nan"))
    ndcg10   = ranking.get("k10", {}).get("ndcg_at_k",      0.0)
    prec10   = ranking.get("k10", {}).get("precision_at_k", 0.0)
    b_ndcg10 = baseline.get("k10", {}).get("ndcg_at_k",     0.0)
    b_prec10 = baseline.get("k10", {}).get("precision_at_k",0.0)

    lift_colour  = "#27ae60" if beats_naive  else "#e74c3c"
    lift_label   = f"+{lift_pct:.1f}% vs naive" if beats_naive else f"{lift_pct:.1f}% vs naive"
    ndcg_colour  = "#27ae60" if ndcg10 > b_ndcg10  else "#e67e22"
    prec_colour  = "#27ae60" if prec10 > b_prec10  else "#e67e22"

    def _bar(value: float, max_val: float = 1.0, colour: str = "#3498db") -> str:
        pct = min(max(value / max_val * 100, 0), 100)
        return (
            f"<div style='background:#eee;border-radius:4px;height:8px;margin:4px 0'>"
            f"<div style='background:{colour};width:{pct:.1f}%;height:100%;border-radius:4px'></div>"
            f"</div>"
        )

    return f"""
    <div class='scorecard-grid'>
      <div class='score-block'>
        <div class='score-label'>MAE (test)</div>
        <div class='score-value'>{mae:.3f}</div>
        {_bar(1 / (1 + mae), colour="#3498db")}
        <div class='score-sub'>Naive: {naive_mae:.3f}
          &nbsp;<span style='color:{lift_colour};font-weight:bold'>{lift_label}</span>
        </div>
      </div>
      <div class='score-block'>
        <div class='score-label'>RMSE (test)</div>
        <div class='score-value'>{rmse:.3f}</div>
        {_bar(1 / (1 + rmse), colour="#8e44ad")}
        <div class='score-sub'>Model: {model_name.upper()} | Res: {time_resolution}</div>
      </div>
      <div class='score-block'>
        <div class='score-label'>NDCG@10</div>
        <div class='score-value' style='color:{ndcg_colour}'>{ndcg10:.4f}</div>
        {_bar(ndcg10, colour=ndcg_colour)}
        <div class='score-sub'>Freq baseline: {b_ndcg10:.4f}</div>
      </div>
      <div class='score-block'>
        <div class='score-label'>Precision@10</div>
        <div class='score-value' style='color:{prec_colour}'>{prec10:.4f}</div>
        {_bar(prec10, colour=prec_colour)}
        <div class='score-sub'>Freq baseline: {b_prec10:.4f}</div>
      </div>
      <div class='score-block'>
        <div class='score-label'>ML Lift vs Naive</div>
        <div class='score-value' style='color:{lift_colour}'>{lift_pct:+.1f}%</div>
        <div class='score-sub'>
          {'\u2713 Beats naive predictor' if beats_naive else '\u2717 No improvement over naive'}
        </div>
      </div>
      <div class='score-block'>
        <div class='score-label'>Rounds Trained</div>
        <div class='score-value'>{rounds if rounds else 'N/A'}</div>
        <div class='score-sub'>Early-stop @ 20 patience</div>
      </div>
    </div>"""


# ── Main static output generator ─────────────────────────────────────────────

def generate_static_output(
    top_k_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    target_date: str,
    target_hour: int | None,
    output_path: str | Path,
    model_name: str = "xgboost",
    time_resolution: str = "hour",
    eval_metrics: dict | None = None,
) -> Path:
    """
    Generate a self-contained HTML file with enforcement priority map + table.

    Args:
        top_k_df:        Output of rank_zones() — ranked top-K zones.
        centroids_df:    Zone centroids from build_zone_centroids().
        target_date:     Date string (e.g. "2024-03-15").
        target_hour:     Hour of day [0–23] or None for day resolution.
        output_path:     Where to save the HTML file.
        model_name:      Model used (for header display).
        time_resolution: "hour" or "day" (for header display).
        eval_metrics:    Optional dict from full_eval() to display model scorecard.

    Returns:
        Path to the saved HTML file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    hour_label = f"{target_hour:02d}:00–{target_hour+1:02d}:00" if target_hour is not None else "Full Day"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with tqdm(total=4, desc="Building static output", unit="step", leave=True) as pbar:

        # Build Folium map
        pbar.set_description("Building Folium map")
        map_html = build_folium_map(top_k_df, centroids_df, target_date, target_hour)
        pbar.update(1)

        # Build priority table
        pbar.set_description("Building priority table")
        table_html = _build_table_html(top_k_df)
        pbar.update(1)

        # Build scorecard panel
        pbar.set_description("Building model scorecard")
        scorecard_html = _build_scorecard_html(eval_metrics, model_name, time_resolution)
        pbar.update(1)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GridLock R2 — Enforcement Priority | {target_date}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #2c3e50; }}
    header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      color: white; padding: 24px 32px; display: flex; justify-content: space-between; align-items: center;
    }}
    header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }}
    header p {{ font-size: 13px; color: #a0b4c8; margin-top: 4px; }}
    .badge {{
      background: #e74c3c; color: white; padding: 4px 12px;
      border-radius: 20px; font-size: 12px; font-weight: bold;
    }}
    .meta-bar {{
      background: white; border-bottom: 1px solid #ddd;
      padding: 12px 32px; display: flex; gap: 32px; font-size: 13px; color: #555;
    }}
    .meta-bar span {{ font-weight: 600; color: #2c3e50; }}
    .container {{ padding: 24px 32px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .card {{
      background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      overflow: hidden;
    }}
    .card-header {{
      background: #2c3e50; color: white; padding: 12px 16px;
      font-size: 14px; font-weight: 600; letter-spacing: 0.3px;
    }}
    .card-body {{ padding: 16px; overflow: auto; }}
    .map-card {{ grid-column: 1 / 3; }}
    .map-card .card-body {{ padding: 0; height: 520px; }}
    .map-card iframe {{ width: 100%; height: 100%; border: none; }}
    .formula-box {{
      background: #f8f9fa; border-left: 4px solid #3498db;
      padding: 10px 14px; margin-top: 12px; font-size: 13px; border-radius: 0 4px 4px 0;
    }}
    .scorecard-grid {{
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; padding: 4px 0;
    }}
    .score-block {{
      background: #f8f9fa; border-radius: 8px; padding: 14px 16px;
      border: 1px solid #e8ecef;
    }}
    .score-label {{ font-size: 11px; font-weight: 600; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }}
    .score-value {{ font-size: 28px; font-weight: 700; color: #2c3e50; margin: 6px 0 2px; }}
    .score-sub {{ font-size: 11px; color: #95a5a6; margin-top: 4px; }}
    footer {{
      text-align: center; padding: 16px; font-size: 12px; color: #999;
      border-top: 1px solid #ddd; background: white;
    }}
    @media (max-width: 900px) {{
      .container {{ grid-template-columns: 1fr; padding: 16px; gap: 16px; }}
      .map-card {{ grid-column: 1; }}
      .scorecard-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .card {{ grid-column: 1 !important; }}
    }}
    @media (max-width: 600px) {{
      header {{ flex-direction: column; align-items: flex-start; gap: 12px; padding: 16px; }}
      .meta-bar {{ flex-direction: column; gap: 8px; padding: 12px 16px; }}
      .scorecard-grid {{ grid-template-columns: 1fr; }}
      .map-card .card-body {{ height: 350px; }}
      .table-wrapper {{ overflow-x: auto; display: block; width: 100%; -webkit-overflow-scrolling: touch; }}
    }}
  </style>
</head>
<body>

<header>
  <div>
    <h1>🚨 GridLock R2 — Enforcement Priority Map</h1>
    <p>PS1: Parking-Induced Congestion | Bengaluru Traffic Police</p>
  </div>
  <div>
    <div class="badge">DEMO OUTPUT</div>
    <p style="font-size:12px;color:#a0b4c8;margin-top:6px">{generated_at}</p>
  </div>
</header>

<div class="meta-bar">
  <div>Date: <span>{target_date}</span></div>
  <div>Time slot: <span>{hour_label}</span></div>
  <div>Model: <span>{model_name.upper()} ({time_resolution})</span></div>
  <div>Zones ranked: <span>{len(top_k_df)}</span></div>
  <div>Formula: <span>priority = predicted_count × CIS</span></div>
</div>

<div class="container">

  <div class="card map-card">
    <div class="card-header">📍 Interactive Enforcement Map — Bengaluru</div>
    <div class="card-body">
      {map_html}
    </div>
  </div>

  <div class="card" style="grid-column: 1 / 3;">
    <div class="card-header">📊 Model Evaluation Scorecard — {model_name.upper()} / {time_resolution}</div>
    <div class="card-body">
      {scorecard_html}
    </div>
  </div>

  <div class="card" style="grid-column: 1 / 3;">
    <div class="card-header">🏆 Top {len(top_k_df)} Enforcement Priority Zones</div>
    <div class="card-body">
      {table_html}
      <div class="formula-box">
        <strong>Ranker formula v1.0:</strong>
        priority_score(zone, t) = predicted_violation_count(zone, t) × CIS(zone)<br>
        <strong>CIS formula v1.0:</strong>
        CIS(zone) = violation_density_norm(zone) × junction_weight (1.5 at junction, 1.0 otherwise)
      </div>
    </div>
  </div>

</div>

<footer>
  GridLock R2 — Bengaluru Parking Violation AI | Prototype Demo | Data: Jan–May 2024 Police Violations
</footer>

</body>
</html>"""

        out.write_text(html, encoding="utf-8")
        pbar.update(1)

    logger.info(f"✓ Static output saved → '{out}' ({out.stat().st_size / 1e3:.1f} KB)")
    return out
