import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import audio, simulator
from backend.api import system_status
from backend.core.config import APP_TITLE, APP_VERSION, STORAGE_DIR

os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(
    title=APP_TITLE,
    description="Ultrasonic Covert Channel & Air-Gapped Exfiltration Backend",
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

app.include_router(simulator.router)
app.include_router(audio.router)


@app.get("/health")
def health():
    return system_status.health_check()
