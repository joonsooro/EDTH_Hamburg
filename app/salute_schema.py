from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class Location(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    source: Literal["GPS","dead_reckon","N/A"] = "N/A"
    confidence_radius_m: Optional[float] = None

class Evidence(BaseModel):
    video_uri: Optional[str] = None
    frame_png_uri: Optional[str] = None
    sha256: Optional[str] = None

class SystemMeta(BaseModel):
    model: str
    version: str
    build: str

class Salute(BaseModel):
    size: str
    activity: str
    location: Location
    time_utc: str = Field(default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
    equipment: List[str]
    evidence: Evidence
    system: SystemMeta
    notes: Optional[str] = None

def default_salute(size: str, activity: str, loc: Location, equipment: List[str], notes: str) -> Salute:
    return Salute(
        size=size,
        activity=activity,
        location=loc,
        equipment=equipment,
        evidence=Evidence(),
        system=SystemMeta(model="yolov8n-int8-or-fp16", version="0.1.0", build="ham-edth"),
        notes=notes,
    )
