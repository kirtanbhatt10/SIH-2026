from fastapi import FastAPI

from backend.app.routes.analysis import router as analysis_router


app = FastAPI(
    title="SIH 2026 Acoustic Shield",
    description="Backend API for the Acoustic Cybersecurity System",
    version="0.1.0",
)


@app.get("/api/system-status")
def system_status():
    return {
        "status": "online",
        "service": "acoustic-shield-backend",
        "version": "0.1.0",
    }


app.include_router(analysis_router)