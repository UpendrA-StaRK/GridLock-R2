"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pathlib import Path

from src.inference.ranker import load_ranker
from src.inference.static_output import build_zone_centroids
from src.api.routers import system, predictions, analytics

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, handle missing model gracefully."""
    logger.info("Starting up FastAPI server...")
    project_root = Path(__file__).resolve().parent.parent.parent
    try:
        app.state.ranker = load_ranker(project_root=project_root)
        app.state.centroids_df = build_zone_centroids(project_root / "data" / "processed" / "features_with_zones.parquet")
        logger.info("Model and centroids loaded successfully.")
        analytics.start_background_prewarm(app.state.ranker)
    except Exception as e:
        logger.warning(f"Model could not be loaded: {e}. Starting in untrained state.")
        app.state.ranker = None
        app.state.centroids_df = None
    app.state.project_root = project_root
    yield
    logger.info("Shutting down FastAPI server...")

app = FastAPI(title="GridLock R2 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api/v1/system", tags=["System"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])

# Serve frontend statically
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent.parent / "docs", html=True), name="docs")
