"""
src/dashboard/app.py
GridLock R2 — PS1: Parking-Induced Congestion

Phase 5: Interactive Streamlit Dashboard

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path so we can import 'src'
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.inference.ranker import load_ranker, rank_zones
from src.inference.static_output import build_zone_centroids

st.set_page_config(
    page_title="GridLock R2 | Bengaluru Traffic Police",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_app_data() -> tuple[dict, pd.DataFrame, dict]:
    """Load model, ranker, and geographic data."""
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # 1. Load Ranker
    ranker = load_ranker(project_root)
    
    # 2. Load Zone Centroids
    features_path = project_root / "data" / "processed" / "features_with_zones.parquet"
    centroids_df = build_zone_centroids(features_path)
    
    # 3. Load Eval Metrics
    eval_metrics = {}
    import glob
    eval_files = sorted(glob.glob(str(project_root / "data" / "outputs" / "eval_*.json")), reverse=True)
    if eval_files:
        with open(eval_files[0], "r", encoding="utf-8") as f:
            ev_data = json.load(f)
            model_key = f"{ranker['model_name']}_{ranker['time_resolution']}"
            eval_metrics = ev_data.get(model_key, {})

    return ranker, centroids_df, eval_metrics

ranker, centroids_df, eval_metrics = load_app_data()

# ─── UI Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4b/Bengaluru_City_Police_Logo.png", width=150)
st.sidebar.title("GridLock R2 Engine")

st.sidebar.markdown("### 🗓️ Parameters")
target_date = st.sidebar.date_input("Target Date", value=pd.to_datetime("2024-03-18").date())
target_hour = st.sidebar.slider("Time of Day", min_value=0, max_value=23, value=9, step=1, format="%02d:00")
top_k = st.sidebar.selectbox("Display Top-K Zones", options=[5, 10, 15, 20], index=1)

# ─── UI Main Layout ───────────────────────────────────────────────────────────

st.title("📍 Bengaluru Enforcement Priority Map")
st.markdown("PS1: Parking-Induced Congestion — AI Hotspot Detection")

# Metrics Banner
mae = eval_metrics.get("regression", {}).get("mae", "N/A")
ndcg = eval_metrics.get("ranking", {}).get("k10", {}).get("ndcg_at_k", "N/A")
pai = eval_metrics.get("spatial_pai", {}).get("pai", "N/A")

if isinstance(mae, float): mae = f"{mae:.2f}"
if isinstance(ndcg, float): ndcg = f"{ndcg:.4f}"
if isinstance(pai, float): pai = f"{pai:.2f}×"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", f"{ranker['model_name'].upper()} ({ranker['time_resolution']})")
col2.metric("NDCG@10", str(ndcg))
col3.metric("MAE", str(mae))
col4.metric("Spatial Accuracy (PAI)", str(pai))

st.divider()

# ─── Core Logic (Cached for Speed) ────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_ranked_zones(_ranker, target_date: str, target_hour: int, top_k: int) -> pd.DataFrame:
    """Wrapper to cache predictions so slider movement is instant."""
    return rank_zones(_ranker, target_date=target_date, target_hour=target_hour, top_k=top_k)

with st.spinner(f"Ranking zones for {target_date} {target_hour:02d}:00..."):
    top_k_df = get_ranked_zones(ranker, str(target_date), target_hour, top_k)

# ─── Map Builder ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_map(df: pd.DataFrame, centroids: pd.DataFrame, top_k: int) -> folium.Map:
    # Bengaluru city centre
    BENGALURU_LAT = 12.9716
    BENGALURU_LON = 77.5946

    m = folium.Map(location=[BENGALURU_LAT, BENGALURU_LON], zoom_start=12, tiles="CartoDB positron")
    colour_map = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#27ae60"}

    merged = df.reset_index().merge(centroids, on="zone_id", how="left")

    for _, row in merged.iterrows():
        if pd.isna(row.get("lat_centroid")) or pd.isna(row.get("lon_centroid")):
            continue

        tier = str(row.get("priority_tier", "LOW"))
        colour = colour_map.get(tier, "blue")
        rank = int(row.get("rank", 0))

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
            radius=15 + (top_k - rank),
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"#{rank} Zone {int(row['zone_id'])} ({tier})",
        ).add_to(m)

        folium.Marker(
            location=[row["lat_centroid"], row["lon_centroid"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;color:white;text-align:center;line-height:20px">#{rank}</div>',
                icon_size=(20, 20),
                icon_anchor=(10, 10),
            ),
        ).add_to(m)

    return m

# Display Map
m = build_map(top_k_df, centroids_df, top_k)
# Setting returned_objects=[] prevents Streamlit from doing expensive bidirectional JS syncing on click
st_data = st_folium(m, height=500, use_container_width=True, returned_objects=[])

# ─── Leaderboard Table ────────────────────────────────────────────────────────

st.markdown("### 🏆 Top Enforcement Priority Zones")

# Prepare dataframe for display
display_df = top_k_df.copy().reset_index()
display_df["Junction"] = display_df["has_junction"].map({True: "✓", False: "—"})
display_df = display_df[["rank", "zone_id", "priority_tier", "priority_score", "predicted_count", "cis_score", "Junction"]]
display_df.columns = ["Rank", "Zone ID", "Priority", "Priority Score", "Predicted Count", "CIS Score", "Junction"]

# Style it
def style_priority(val):
    if val == "HIGH":
        return "color: #c0392b; font-weight: bold;"
    elif val == "MEDIUM":
        return "color: #e67e22; font-weight: bold;"
    return "color: #27ae60; font-weight: bold;"

st.dataframe(
    display_df.style.map(style_priority, subset=["Priority"]).format({
        "Priority Score": "{:.4f}",
        "Predicted Count": "{:.1f}",
        "CIS Score": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.markdown("*(Powered by GridLock R2 — 6 Months of Bengaluru Police Violation Data)*")
