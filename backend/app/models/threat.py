from pydantic import BaseModel, Field


class FrequencyRange(BaseModel):
    min: float = Field(..., ge=0)
    max: float = Field(..., ge=0)


class ThreatEvent(BaseModel):
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk: str
    frequency: FrequencyRange
    duration: float = Field(..., ge=0)
    pattern: str