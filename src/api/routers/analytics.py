"""Analytics and metrics routes."""
from fastapi import APIRouter, Request, HTTPException, Query
from loguru import logger
import pandas as pd
import json
from src.api.schemas import DailyTrendResponse, MetricsResponse, ZoneTrendResponse
from src.inference.ranker import load_ranker, rank_zones

router = APIRouter()

# In-memory caches to prevent redundant model inferences
TREND_CACHE = {}
METRICS_CACHE = {}

@router.get("/daily-trend", response_model=DailyTrendResponse)
async def get_daily_trend(
    request: Request,
    date: str = Query(..., description="Target date YYYY-MM-DD")
):
    """Returns 24-hour total violation trend for the given date by summing predictions per hour."""
    if date in TREND_CACHE:
        return TREND_CACHE[date]

    try:
        ranker = request.app.state.ranker
        if ranker is None:
            try:
                ranker = load_ranker(project_root=request.app.state.project_root)
                request.app.state.ranker = ranker
            except Exception as e:
                raise HTTPException(status_code=503, detail="Model checkpoint not found. Train model first.")
        
        trend = []
        for h in range(24):
            # Rank all zones to get total estimated violations for that hour
            df = rank_zones(ranker, target_date=date, target_hour=h, top_k=9999)
            total = float(df["predicted_count"].sum())
            trend.append(total)
            
        response = DailyTrendResponse(
            date=date,
            trend=trend
        )
        TREND_CACHE[date] = response
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trend generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    request: Request,
    date: str = Query(..., description="Target date YYYY-MM-DD"),
    hour: int = Query(..., ge=0, le=23, description="Hour of day (0-23)")
):
    """Returns high-level KPIs for a given hour and date."""
    cache_key = f"{date}_{hour}"
    if cache_key in METRICS_CACHE:
        return METRICS_CACHE[cache_key]

    try:
        ranker = request.app.state.ranker
        if ranker is None:
            raise HTTPException(status_code=503, detail="Model checkpoint not found. Train model first.")
            
        df = rank_zones(ranker, target_date=date, target_hour=hour, top_k=9999)
        total_violations = float(df['predicted_count'].sum())
        
        # Determine critical junctions (High tier + has_junction)
        if 'has_junction' in df.columns:
            critical_df = df[(df['priority_tier'] == 'HIGH') & (df['has_junction'] == True)]
            critical_junctions = len(critical_df)
        else:
            critical_junctions = 0
            
        hotspots_count = len(df[df['priority_tier'].isin(['HIGH', 'MEDIUM'])])
        
        response = MetricsResponse(
            total_violations=total_violations,
            critical_junctions=critical_junctions,
            hotspots_count=hotspots_count
        )
        METRICS_CACHE[cache_key] = response
        return response
    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/feature-importance")
async def get_feature_importance(request: Request):
    """Reads and returns data/outputs/shap_report.json if it exists."""
    try:
        shap_path = request.app.state.project_root / "data" / "outputs" / "shap_report.json"
        if not shap_path.exists():
            raise FileNotFoundError()
            
        with open(shap_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="SHAP report not found.")
    except Exception as e:
        logger.error(f"Failed to read SHAP report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/zones/{zone_id}/trend", response_model=ZoneTrendResponse)
async def get_zone_trend(request: Request, zone_id: int):
    """Loads zone_day_grid.parquet and returns the 7-day violation counts for the specified zone."""
    try:
        parquet_path = request.app.state.project_root / "data" / "processed" / "zone_day_grid.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError()
            
        df = pd.read_parquet(parquet_path)
        zone_df = df[df['zone_id'] == zone_id].copy()
        
        if len(zone_df) == 0:
            raise HTTPException(status_code=404, detail="Zone ID not found in data.")
            
        # Sort by date and take the last 7 days
        zone_df = zone_df.sort_values('date').tail(7)
        
        trend = []
        for _, row in zone_df.iterrows():
            trend.append({
                "date": str(row['date'])[:10],  # Format as YYYY-MM-DD
                "count": float(row.get('zone_day_violation_count', 0.0))
            })
            
        return ZoneTrendResponse(
            zone_id=zone_id,
            trend=trend
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Zone day grid data not generated yet.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get zone trend: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


import threading

def _warm_single_date(ranker, date_str: str):
    try:
        if date_str in TREND_CACHE: return
        trend = []
        for h in range(24):
            df = rank_zones(ranker, target_date=date_str, target_hour=h, top_k=9999)
            trend.append(float(df["predicted_count"].sum()))
        TREND_CACHE[date_str] = DailyTrendResponse(date=date_str, trend=trend)
        logger.info(f"✓ Background cache warmed for {date_str}")
    except Exception as e:
        logger.error(f"Pre-warm failed for {date_str}: {e}")

def start_background_prewarm(ranker):
    def worker():
        import pandas as pd
        dates = [(pd.Timestamp("2024-02-23") + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        logger.info("Starting background cache pre-warming for 7 days...")
        for d in dates:
            _warm_single_date(ranker, d)
        logger.info("🎉 Cache pre-warming complete! All 7 days are instant.")
    
    threading.Thread(target=worker, daemon=True).start()
