import os
from backend.models.data_schemas import StreamConfig

SAMPLE_RATE = 48000
FREQ_0 = 18500
FREQ_1 = 20500  # was 21000; AI DSP band ends at 21000 Hz (exclusive at bin edge)
BIT_DURATION = 0.05
PREAMBLE = "10101010"

STORAGE_DIR = os.environ.get("STORAGE_DIR", "generated_payloads")
FORENSIC_CAPTURE_DIR = os.environ.get("FORENSIC_CAPTURE_DIR", "forensic_captures")

# Verified hardware defaults (override via env only when needed).
MIC_DEVICE_ID = int(os.environ.get("MIC_DEVICE_ID", "1"))
SPEAKER_DEVICE_ID = int(os.environ.get("SPEAKER_DEVICE_ID", "3"))

APP_TITLE = "Silent Dog-Whistle - Ultrasonic C2 Backend"
APP_VERSION = "2.0.0"

STREAM_CONFIG = StreamConfig()
