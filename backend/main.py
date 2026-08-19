import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import audio, simulator
from backend.api import system_status
from backend.api import analysis as analysis_api
from backend.api import threats as threats_api
from backend.core.config import APP_TITLE, APP_VERSION, STORAGE_DIR

os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(
    title=APP_TITLE,
    description="SIH 2026 Acoustic Cybersecurity System — Unified Backend",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=STORAGE_DIR), name="files")

# --- Backend 2 routers (Simulator & Audio Streaming) ---
app.include_router(simulator.router)
app.include_router(audio.router)

# --- Backend 1 routers (Threat Management API) ---
app.include_router(analysis_api.router)
app.include_router(threats_api.router)


# --- Backend 2 health check ---
@app.get("/health", tags=["System"])
def health():
    return system_status.health_check()


# --- Backend 1 system-status endpoint (preserves original contract) ---
@app.get("/api/system-status", tags=["System"])
def api_system_status():
    return {
        "status": "online",
        "service": "acoustic-shield-backend",
        "version": APP_VERSION,
    }
