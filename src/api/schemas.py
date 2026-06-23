"""Pydantic schemas for the API."""
from pydantic import BaseModel
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str
    timestamp: str

class ModelStatusResponse(BaseModel):
    status: str
    model_name: Optional[str] = None
    time_resolution: Optional[str] = None
    features: Optional[List[str]] = None

class Hotspot(BaseModel):
    zone_id: int
    rank: int
    priority_score: float
    predicted_count: float
    priority_tier: str
    cis_score: float
    has_junction: bool
    dominant_violation_type: str
    dispatch_strategy: str
    nlp_explanation: str
    area_name: str
    police_station: str
    lat: float
    lon: float

class HotspotsResponse(BaseModel):
    date: str
    hour: int
    hotspots: List[Hotspot]

class DailyTrendResponse(BaseModel):
    date: str
    trend: List[float]

class MetricsResponse(BaseModel):
    total_violations: float
    critical_junctions: int
    hotspots_count: int

class ZoneTrendItem(BaseModel):
    date: str
    count: float

class ZoneTrendResponse(BaseModel):
    zone_id: int
    trend: List[ZoneTrendItem]
