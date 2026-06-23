"""System and admin routes."""
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from datetime import datetime, timezone
import shutil
from loguru import logger
from src.api.schemas import HealthResponse, ModelStatusResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns API status."""
    try:
        return HealthResponse(
            status="ok",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status(request: Request):
    """Returns current active model information."""
    try:
        ranker = request.app.state.ranker
        if ranker is None:
            return ModelStatusResponse(status="Not Trained")
        
        return ModelStatusResponse(
            status="Loaded",
            model_name=ranker.get("model_name"),
            time_resolution=ranker.get("time_resolution"),
            features=ranker.get("features", [])
        )
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/upload")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    """Upload new raw CSV dataset. Overwrites existing raw data."""
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a CSV")
            
        raw_dir = request.app.state.project_root / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing CSVs to maintain single raw file assumption
        for existing_file in raw_dir.glob("*.csv"):
            existing_file.unlink()
            
        file_path = raw_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"status": "success", "filename": file.filename, "message": "Pipeline must be re-run manually to train on new data."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
