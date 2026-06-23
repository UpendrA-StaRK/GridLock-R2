"""Prediction routes."""
from fastapi import APIRouter, Request, HTTPException, Query
from loguru import logger
from src.api.schemas import HotspotsResponse
from src.inference.ranker import load_ranker, rank_zones

router = APIRouter()

@router.get("/hotspots", response_model=HotspotsResponse)
async def get_hotspots(
    request: Request,
    date: str = Query(..., description="Target date YYYY-MM-DD"),
    hour: int = Query(..., ge=0, le=23, description="Hour of day (0-23)"),
    top_k: int = Query(10, ge=1, le=100)
):
    """Returns top K enforcement zones for a specific date and hour."""
    try:
        ranker = request.app.state.ranker
        if ranker is None:
            # Lazy load attempt
            try:
                ranker = load_ranker(project_root=request.app.state.project_root)
                request.app.state.ranker = ranker
            except Exception as e:
                logger.error(f"Lazy load failed: {e}")
                raise HTTPException(status_code=503, detail="Model checkpoint not found. Train model first.")
                
        # rank_zones returns a pandas DataFrame
        top_k_df = rank_zones(ranker, target_date=date, target_hour=hour, top_k=top_k)
        top_k_df['rank'] = top_k_df.index
        
        # Join centroids
        if getattr(request.app.state, "centroids_df", None) is not None:
            top_k_df = top_k_df.merge(request.app.state.centroids_df, on="zone_id", how="left")
            top_k_df = top_k_df.rename(columns={"lat_centroid": "lat", "lon_centroid": "lon"})
        else:
            top_k_df["lat"] = 0.0
            top_k_df["lon"] = 0.0
            top_k_df["area_name"] = "Unknown Area"
            
        top_k_df["lat"] = top_k_df["lat"].fillna(0.0)
        top_k_df["lon"] = top_k_df["lon"].fillna(0.0)
        top_k_df["area_name"] = top_k_df["area_name"].fillna("Unknown Area")
        
        records = top_k_df.to_dict(orient="records")
        # Cast boolean explicitly to match schema
        for r in records:
            r["has_junction"] = bool(r.get("has_junction", False))
            
        return HotspotsResponse(
            date=date,
            hour=hour,
            hotspots=records
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
