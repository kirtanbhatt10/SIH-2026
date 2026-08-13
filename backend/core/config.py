import os
from backend.models.data_schemas import StreamConfig

SAMPLE_RATE = 48000
FREQ_0 = 18500
FREQ_1 = 21000
BIT_DURATION = 0.05
PREAMBLE = "10101010"

STORAGE_DIR = os.environ.get("STORAGE_DIR", "generated_payloads")
FORENSIC_CAPTURE_DIR = os.environ.get("FORENSIC_CAPTURE_DIR", "forensic_captures")

APP_TITLE = "Silent Dog-Whistle - Ultrasonic C2 Backend"
APP_VERSION = "2.0.0"

STREAM_CONFIG = StreamConfig()
