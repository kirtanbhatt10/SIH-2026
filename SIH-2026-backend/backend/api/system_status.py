from backend.core.config import APP_TITLE, APP_VERSION, BIT_DURATION, FREQ_0, FREQ_1, SAMPLE_RATE, STORAGE_DIR
from backend.services.stream_manager import get_active_streamer, set_active_streamer

_payloads: dict[str, dict] = {}
_last_simulator_diag: dict = {}


def get_payloads() -> dict[str, dict]:
    return _payloads


def get_last_simulator_diag() -> dict:
    return dict(_last_simulator_diag)


def set_last_simulator_diag(**kwargs) -> None:
    _last_simulator_diag.update(kwargs)


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
        "stream_active": get_active_streamer() is not None,
    }
