from fastapi import APIRouter

from backend.app.services.threat_service import get_latest_threat, get_threats


router = APIRouter(
    prefix="/api",
    tags=["Threats"],
)


@router.get("/threats")
def list_threats():
    return {"threats": get_threats()}


@router.get("/threats/current")
def current_threat():
    latest = get_latest_threat()
    if latest is None:
        return {
            "current": None,
            "message": "No threats have been recorded yet",
        }
    return {"current": latest}
