from backend.core.config import APP_TITLE, APP_VERSION, BIT_DURATION, FREQ_0, FREQ_1, SAMPLE_RATE, STORAGE_DIR
from backend.services.audio_service import AudioStreamer

_payloads: dict[str, dict] = {}
_active_streamer: AudioStreamer | None = None


def get_payloads() -> dict[str, dict]:
    return _payloads


def get_active_streamer() -> AudioStreamer | None:
    return _active_streamer


def set_active_streamer(streamer: AudioStreamer | None) -> None:
    global _active_streamer
    _active_streamer = streamer


def health_check() -> dict:
    return {
        "status": "healthy",
        "service": APP_TITLE,
        "version": APP_VERSION,
        "sample_rate": SAMPLE_RATE,
        "freq_0": FREQ_0,
        "freq_1": FREQ_1,
        "bit_duration": BIT_DURATION,
        "storage_dir": STORAGE_DIR,
        "total_payloads_generated": len(_payloads),
        "stream_active": _active_streamer is not None,
    }
