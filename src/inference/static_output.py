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

    import base64
    html_bytes = m.get_root().render().encode("utf-8")
    b64 = base64.b64encode(html_bytes).decode("utf-8")
    map_html = f'<iframe src="data:text/html;base64,{b64}" style="width:100%; height:100%; border:none;"></iframe>'
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
    # Pre-compute conditional string to avoid backslash-in-f-string (Python < 3.12)
    status_label = "✓ Beats naive predictor" if beats_naive else "✗ No improvement over naive"

    def _bar(value: float, max_val: float = 1.0, colour: str = "#3498db") -> str:
        pct = min(max(value / max_val * 100, 0), 100)
        return (
            f"<div style='background:#eee;border-radius:4px;height:8px;margin:4px 0'>"
            f"<div style='background:{colour};width:{pct:.1f}%;height:100%;border-radius:4px'></div>"
            f"</div>"
        )

    # PAI block — Phase 3 addition (backward-compatible: absent if not in eval_metrics)
    pai_data = eval_metrics.get("spatial_pai", {})
    pai_block = ""
    if pai_data and pai_data.get("pai", 0.0) > 0:
        pai        = pai_data.get("pai", 0.0)
        hit_rate   = pai_data.get("hit_rate", 0.0)
        area_frac  = pai_data.get("area_fraction", 0.0)
        pai_colour = "#27ae60" if pai >= 2.0 else ("#e67e22" if pai >= 1.0 else "#e74c3c")
        # Cap bar at PAI=10 for display purposes
        pai_block = f"""
      <div class='score-block' style='grid-column: 1 / -1; border-left: 4px solid {pai_colour};'>
        <div class='score-label'>PAI — Spatial Accuracy Index (police standard metric)</div>
        <div class='score-value' style='color:{pai_colour}'>{pai:.2f}×</div>
        {_bar(min(pai / 10.0, 1.0), colour=pai_colour)}
        <div class='score-sub'>
          Top-{pai_data.get('k', 10)} zones cover <b>{area_frac*100:.1f}%</b> of all zones
          but capture <b>{hit_rate*100:.1f}%</b> of test violations
          &nbsp;→&nbsp; <b>{pai:.2f}× better than random patrolling</b>
          &nbsp;(PAI&nbsp;&gt;&nbsp;1.0&nbsp;=&nbsp;better than random;&nbsp;PAI&nbsp;&gt;&nbsp;2.0&nbsp;=&nbsp;strong)
        </div>
      </div>"""

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
          {status_label}
        </div>
      </div>
      <div class='score-block'>
        <div class='score-label'>Rounds Trained</div>
        <div class='score-value'>{rounds if rounds else 'N/A'}</div>
        <div class='score-sub'>Early-stop @ 20 patience</div>
      </div>
      {pai_block}
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
      .map-card .card-body {{ height: 350px; padding: 0; margin-bottom: -5px; }}
      .map-card iframe {{ display: block; }}
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


# ── Time-slider output generator (Phase 5) ────────────────────────────────────

def generate_static_output_with_slider(
    all_dates_hours_data: dict[str, dict[int, pd.DataFrame]],
    centroids_df: pd.DataFrame,
    target_dates: list[str],
    output_path: str | Path,
    model_name: str = "xgboost",
    time_resolution: str = "hour",
    eval_metrics: dict | None = None,
) -> Path:
    """
    Phase 5: Generate a self-contained HTML file with a 24-hour interactive time slider.

    The slider updates zone markers and the priority table in real-time (JavaScript,
    no server required). All 24 hours of ranking data are embedded as a JSON object.

    This is the primary demo output — it shows the temporal dynamics of enforcement
    priority across the day, which is the key differentiator between ML and a static
    frequency table.

    Args:
        all_dates_hours_data: Nested Dict: date_str -> hour (0-23) -> rank_zones() DataFrame.
        centroids_df:   Zone centroids from build_zone_centroids().
        target_dates:   List of date strings (e.g. ["2024-03-18", "2024-03-19"]).
        output_path:    Where to save the HTML file.
        model_name:     Model used (for header display).
        time_resolution: Resolution (for header display).
        eval_metrics:   Optional dict from full_eval() to display model scorecard.

    Returns:
        Path to the saved HTML file.
    """
    import json as _json

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build centroid lookup: zone_id → {lat, lon}
    centroid_lookup: dict[int, dict] = {}
    for _, row in centroids_df.iterrows():
        centroid_lookup[int(row["zone_id"])] = {
            "lat": float(row["lat_centroid"]),
            "lon": float(row["lon_centroid"]),
        }

    # Build JS-embeddable data structure: date -> hour -> list of zone records
    dates_json_data: dict[str, dict[str, list[dict]]] = {}
    for date_str, hours_dict in all_dates_hours_data.items():
        dates_json_data[date_str] = {}
        for hour in range(24):
            df = hours_dict.get(hour)
            if df is None or len(df) == 0:
                dates_json_data[date_str][str(hour)] = []
                continue
            records = []
            for _, row in df.reset_index().iterrows():
                zone_id = int(row["zone_id"])
                centroid = centroid_lookup.get(zone_id, {})
                records.append({
                    "zone_id":        zone_id,
                    "rank":           int(row.get("rank", 0)),
                    "priority_score": float(row.get("priority_score", 0)),
                    "predicted_count": float(row.get("predicted_count", 0)),
                    "cis_score":      float(row.get("cis_score", 0)),
                    "priority_tier":  str(row.get("priority_tier", "LOW")),
                    "has_junction":   bool(row.get("has_junction", False)),
                    "lat":            centroid.get("lat", 0),
                    "lon":            centroid.get("lon", 0),
                })
            dates_json_data[date_str][str(hour)] = records

    data_json_str = _json.dumps(dates_json_data)

    # Build scorecard (reuse existing helper)
    scorecard_html = _build_scorecard_html(eval_metrics, model_name, time_resolution)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GridLock R2 — Enforcement Priority | {target_dates[0]} | Time Slider</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #2c3e50; }}
    header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      color: white; padding: 20px 28px; display: flex; justify-content: space-between; align-items: center;
    }}
    header h1 {{ font-size: 20px; font-weight: 700; }}
    header p {{ font-size: 12px; color: #a0b4c8; margin-top: 3px; }}
    .badge {{ background: #e74c3c; color: white; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: bold; }}

    .slider-bar {{
      background: white; border-bottom: 2px solid #3498db;
      padding: 16px 28px; display: flex; align-items: center; gap: 24px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .slider-label {{ font-size: 13px; font-weight: 700; color: #2c3e50; white-space: nowrap; }}
    #hour-slider {{ flex: 1; -webkit-appearance: none; height: 6px; background: #dfe6e9; border-radius: 3px; outline: none; }}
    #hour-slider::-webkit-slider-thumb {{
      -webkit-appearance: none; width: 22px; height: 22px;
      border-radius: 50%; background: #3498db; cursor: pointer;
      box-shadow: 0 2px 6px rgba(52,152,219,0.5); transition: background 0.2s;
    }}
    #hour-slider::-webkit-slider-thumb:hover {{ background: #2980b9; }}
    #hour-display {{
      font-size: 22px; font-weight: 800; color: #3498db;
      background: #eaf4fd; padding: 6px 16px; border-radius: 8px;
      min-width: 100px; text-align: center; white-space: nowrap;
    }}
    .time-icons {{ display: flex; gap: 8px; }}
    .time-icon {{ cursor: pointer; font-size: 22px; padding: 4px; border-radius: 6px; transition: background 0.15s; }}
    .time-icon:hover {{ background: #eaf4fd; }}

    .container {{ padding: 20px 28px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    #map-container {{ grid-column: 1 / 3; height: 500px; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.1); }}
    #leaflet-map {{ width: 100%; height: 100%; }}
    .card {{
      background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      overflow: hidden;
    }}
    .card-header {{
      background: #2c3e50; color: white; padding: 11px 16px;
      font-size: 13px; font-weight: 600;
    }}
    .card-body {{ padding: 16px; overflow: auto; max-height: 360px; }}
    #zone-table-card {{ grid-column: 1 / 3; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    thead tr {{ background: #2c3e50; color: white; }}
    thead th {{ padding: 8px 12px; text-align: center; font-weight: 600; }}
    tbody tr:nth-child(even) {{ background: #f8f9fa; }}
    tbody td {{ padding: 7px 12px; text-align: center; border-bottom: 1px solid #eee; }}
    .tier-HIGH {{ color: #e74c3c; font-weight: bold; }}
    .tier-MEDIUM {{ color: #e67e22; font-weight: bold; }}
    .tier-LOW {{ color: #27ae60; font-weight: bold; }}

    .scorecard-grid {{
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; padding: 4px 0;
    }}
    .score-block {{ background: #f8f9fa; border-radius: 8px; padding: 12px 14px; border: 1px solid #e8ecef; }}
    .score-label {{ font-size: 10px; font-weight: 600; color: #7f8c8d; text-transform: uppercase; }}
    .score-value {{ font-size: 24px; font-weight: 700; color: #2c3e50; margin: 4px 0 2px; }}
    .score-sub {{ font-size: 10px; color: #95a5a6; margin-top: 3px; }}

    footer {{ text-align: center; padding: 14px; font-size: 11px; color: #999; border-top: 1px solid #ddd; background: white; }}
    @media (max-width: 900px) {{ .container {{ grid-template-columns: 1fr; }} #map-container, #zone-table-card {{ grid-column: 1; }} }}
  </style>
</head>
<body>

<header>
  <div>
    <h1>🚨 GridLock R2 — Enforcement Priority Map (24h)</h1>
    <p>PS1: Parking-Induced Congestion | Bengaluru Traffic Police | <span id="header-date">{target_dates[0]}</span></p>
  </div>
  <div style="text-align:right">
    <div class="badge">DEMO — TIME SLIDER</div>
    <p style="font-size:11px;color:#a0b4c8;margin-top:5px">{generated_at}</p>
  </div>
</header>

<!-- ── Time slider ── -->
<div class="slider-bar">
  <span class="slider-label">📅 Date:</span>
  <select id="date-select" style="padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 13px; outline: none; cursor: pointer; color: #2c3e50; font-weight: 600;">
    {''.join(f'<option value="{d}">{d}</option>' for d in target_dates)}
  </select>

  <span class="slider-label" style="margin-left: 16px;">⏱ Hour of Day:</span>
  <input type="range" id="hour-slider" min="0" max="23" value="9" step="1">
  <div id="hour-display">09:00</div>
  <div class="time-icons">
    <span class="time-icon" title="Morning rush (9am)" onclick="setHour(9)">🌅</span>
    <span class="time-icon" title="Midday (12pm)" onclick="setHour(12)">☀️</span>
    <span class="time-icon" title="Evening rush (18:00)" onclick="setHour(18)">🌆</span>
    <span class="time-icon" title="Night (23:00)" onclick="setHour(23)">🌙</span>
  </div>
</div>

<div class="container">

  <!-- Map -->
  <div id="map-container">
    <div id="leaflet-map"></div>
  </div>

  <!-- Model scorecard -->
  <div class="card" style="grid-column: 1 / 3;">
    <div class="card-header">📊 Model Evaluation Scorecard — {model_name.upper()} / {time_resolution}</div>
    <div class="card-body">{scorecard_html}</div>
  </div>

  <!-- Zone table -->
  <div class="card" id="zone-table-card">
    <div class="card-header">🏆 Top 10 Enforcement Zones — <span id="table-hour-label">Hour 09:00</span></div>
    <div class="card-body">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Zone ID</th><th>Priority</th>
            <th>Score</th><th>Predicted Count</th><th>CIS</th><th>Junction</th>
          </tr>
        </thead>
        <tbody id="zone-tbody"></tbody>
      </table>
    </div>
  </div>

</div>

<footer>
  GridLock R2 — Bengaluru Parking Violation AI | Prototype Demo | Data: Jan–May 2024 Police Violations
</footer>

<script>
// ── All dates/hours data (embedded at build time) ─────────────────────────────
const ALL_DATA = {data_json_str};

let currentDate = "{target_dates[0]}";
let currentHour = 9;

// ── Leaflet map setup ─────────────────────────────────────────────────────────
const BENG_LAT = 12.9716, BENG_LON = 77.5946;
const map = L.map('leaflet-map').setView([BENG_LAT, BENG_LON], 12);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© OpenStreetMap contributors'
}}).addTo(map);

let markers = [];

function tierColor(tier) {{
  if (tier === 'HIGH')   return '#e74c3c';
  if (tier === 'MEDIUM') return '#e67e22';
  return '#27ae60';
}}

function updateDisplay() {{
  const zones = (ALL_DATA[currentDate] && ALL_DATA[currentDate][String(currentHour)]) || [];

  // Clear existing markers
  markers.forEach(m => map.removeLayer(m));
  markers = [];

  // Update table
  const tbody = document.getElementById('zone-tbody');
  tbody.innerHTML = '';

  zones.forEach(z => {{
    if (!z.lat || !z.lon) return;

    const color = tierColor(z.priority_tier);
    const radius = 14 + (10 - z.rank);

    // Leaflet circle marker
    const circle = L.circleMarker([z.lat, z.lon], {{
      radius: radius,
      fillColor: color, color: '#fff',
      weight: 2, opacity: 1, fillOpacity: 0.82
    }});

    circle.bindPopup(`
      <div style="font-family:Arial;font-size:13px;min-width:160px">
        <b>Rank #${{z.rank}} — Zone ${{z.zone_id}}</b><hr style="margin:4px 0">
        Priority Score: <b>${{z.priority_score.toFixed(4)}}</b><br>
        Tier: <b style="color:${{color}}">${{z.priority_tier}}</b><br>
        Predicted Count: ${{z.predicted_count.toFixed(1)}}<br>
        CIS Score: ${{z.cis_score.toFixed(4)}}<br>
        Junction: ${{z.has_junction ? '✓ Yes' : '—'}}
      </div>
    `);

    // Rank number label
    const label = L.marker([z.lat, z.lon], {{
      icon: L.divIcon({{
        html: `<div style="color:white;font-size:11px;font-weight:bold;text-align:center;line-height:22px">#${{z.rank}}</div>`,
        iconSize: [22, 22], iconAnchor: [11, 11], className: ''
      }})
    }});

    circle.addTo(map);
    label.addTo(map);
    markers.push(circle, label);

    // Table row
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{z.rank}}</td>
      <td>${{z.zone_id}}</td>
      <td class="tier-${{z.priority_tier}}">${{z.priority_tier}}</td>
      <td>${{z.priority_score.toFixed(4)}}</td>
      <td>${{z.predicted_count.toFixed(1)}}</td>
      <td>${{z.cis_score.toFixed(4)}}</td>
      <td>${{z.has_junction ? '✓' : '—'}}</td>
    `;
    tbody.appendChild(tr);
  }});

  // Update labels
  const hourLabel = String(currentHour).padStart(2, '0') + ':00';
  document.getElementById('hour-display').textContent = hourLabel;
  document.getElementById('table-hour-label').textContent = `Hour ${{hourLabel}}`;
  document.getElementById('header-date').textContent = currentDate;
}}

// ── Event Listeners ──────────────────────────────────────────────────────────
const slider = document.getElementById('hour-slider');
slider.addEventListener('input', () => {{
  currentHour = parseInt(slider.value);
  updateDisplay();
}});

const dateSelect = document.getElementById('date-select');
if (dateSelect) {{
  dateSelect.addEventListener('change', (e) => {{
    currentDate = e.target.value;
    updateDisplay();
  }});
}}

function setHour(h) {{
  slider.value = h;
  currentHour = h;
  updateDisplay();
}}

// ── Initial render ────────────────────────────────────────────────────────────
updateDisplay();
</script>

</body>
</html>"""

    out.write_text(html, encoding="utf-8")
    logger.info(f"✓ Time-slider output saved → '{out}' ({out.stat().st_size / 1e3:.1f} KB)")
    return out

