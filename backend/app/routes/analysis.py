from fastapi import APIRouter

from backend.app.models.threat import ThreatEvent


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


@router.post("/analyze")
def analyze_threat(event: ThreatEvent):
    return {
        "status": "received",
        "event": event,
    }