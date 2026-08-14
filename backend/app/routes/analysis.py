from fastapi import APIRouter

from backend.app.models.threat import ThreatEvent
from backend.app.services.threat_service import add_threat


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


@router.post("/analyze")
def analyze_threat(event: ThreatEvent):
    add_threat(event)
    return {
        "status": "received",
        "event": event,
    }