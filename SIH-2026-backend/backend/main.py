import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import audio, simulator
from backend.api import system_status
from backend.api import analysis as analysis_api
from backend.api import threats as threats_api
from backend.core.config import APP_TITLE, APP_VERSION, MIC_DEVICE_ID, SPEAKER_DEVICE_ID, STORAGE_DIR
from backend.services.audio_service import list_input_devices
from backend.services.monitoring_pipeline import load_inference_service
from backend.services.payload_service import get_audio_device_info

logger = logging.getLogger(__name__)

os.makedirs(STORAGE_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        result = list_input_devices()
        count = len(result.get("input_devices", []))
        if result.get("error"):
            logger.warning("Startup device warmup failed: %s", result.get("error"))
        else:
            logger.info("Startup device warmup OK: %d input device(s)", count)
        mic = get_audio_device_info(MIC_DEVICE_ID, "input")
        spk = get_audio_device_info(SPEAKER_DEVICE_ID, "output")
        logger.info("[MIC DEVICE] index=%s name=%s sr=%s", mic.get("index"), mic.get("name"), mic.get("default_samplerate"))
        logger.info("[SPEAKER DEVICE] index=%s name=%s sr=%s", spk.get("index"), spk.get("name"), spk.get("default_samplerate"))
    except Exception as exc:
        logger.warning("Startup device warmup raised: %s", exc)
    try:
        await load_inference_service()
        logger.info("[D1.1] InferenceService warmup OK")
    except Exception as exc:
        logger.warning("[D1.1] InferenceService warmup failed: %s", exc)
    yield


app = FastAPI(
    title=APP_TITLE,
    description="SIH 2026 Acoustic Cybersecurity System — Unified Backend",
    version=APP_VERSION,
    lifespan=lifespan,
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
