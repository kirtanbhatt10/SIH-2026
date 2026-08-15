"""
Tests for record_wav.select_input_device ranking.

Encodes the 15 Aug 2026 incident: the old name-only matcher selected
[2] Microphone Array (Senary Audio) on MME at native 44100 Hz while the
recorder requested 48000 Hz. The stream stalled 1-2 s in and wrote digital
silence. Device [18] (WDM-KS, native 48000) was measured working.

record_wav imports sounddevice, which is not available in CI, so we test the
ranking function in isolation rather than importing the module.
"""

import pytest


SAMPLE_RATE = 48000
API_RANK = {"Windows WASAPI": 3, "Windows WDM-KS": 2,
            "Windows DirectSound": 1, "MME": 0}


def rank_devices(devices, sample_rate=SAMPLE_RATE):
    """Mirror of the ranking in record_wav.select_input_device()."""
    candidates = []
    for idx, name, api, native, ch in devices:
        if ch <= 0:
            continue
        low = name.lower()
        if "iriun" in low or "virtual" in low or "sound mapper" in low:
            continue
        candidates.append({
            "idx": idx, "name": name, "api": api, "native": native,
            "rate_ok": abs(native - sample_rate) < 1.0,
            "api_rank": API_RANK.get(api, 0),
            "known_hw": any(k in low for k in ("senary", "realtek", "array")),
        })
    candidates.sort(key=lambda c: (c["rate_ok"], c["api_rank"], c["known_hw"]),
                    reverse=True)
    return candidates


# The real device table from the affected laptop.
LAPTOP = [
    (0,  "Microsoft Sound Mapper - Input",          "MME",                 44100, 2),
    (1,  "Microphone (Iriun Webcam)",               "MME",                 44100, 2),
    (2,  "Microphone Array (Senary Audio)",         "MME",                 44100, 2),
    (5,  "Primary Sound Capture Driver",            "Windows DirectSound", 44100, 2),
    (6,  "Microphone (Iriun Webcam)",               "Windows DirectSound", 44100, 2),
    (7,  "Microphone Array (Senary Audio)",         "Windows DirectSound", 44100, 2),
    (11, "Microphone Array (Senary Audio)",         "Windows WASAPI",      48000, 2),
    (12, "Microphone (Iriun Webcam)",               "Windows WASAPI",      48000, 2),
    (18, "Microphone Array (Senary Audio capture)", "Windows WDM-KS",      48000, 2),
    (22, "Microphone (Senary Audio capture)",       "Windows WDM-KS",      48000, 2),
    (23, "MIDI (Iriun Webcam Audio)",               "Windows WDM-KS",      48000, 2),
]


def test_does_not_pick_the_device_that_broke_recordings():
    """Regression: [2] MME @44100 destroyed the whole physical test set."""
    best = rank_devices(LAPTOP)[0]
    assert best["idx"] != 2
    assert best["api"] != "MME"


def test_picks_native_48k_device():
    best = rank_devices(LAPTOP)[0]
    assert best["rate_ok"], f"picked {best['name']} at {best['native']} Hz"
    assert best["native"] == 48000


def test_prefers_wasapi_when_rates_tie():
    best = rank_devices(LAPTOP)[0]
    assert best["api"] == "Windows WASAPI"
    assert best["idx"] == 11


def test_excludes_virtual_and_webcam_devices():
    for c in rank_devices(LAPTOP):
        low = c["name"].lower()
        assert "iriun" not in low
        assert "sound mapper" not in low


def test_native_rate_beats_host_api():
    """A 44.1k WASAPI device must lose to a 48k WDM-KS device."""
    devices = [
        (1, "Mic A (Senary Audio)", "Windows WASAPI", 44100, 2),
        (2, "Mic B (Senary Audio)", "Windows WDM-KS", 48000, 2),
    ]
    assert rank_devices(devices)[0]["idx"] == 2


def test_warns_when_no_native_match_exists():
    """All devices at 44.1k: still returns one, but flagged as not rate_ok."""
    devices = [
        (1, "Mic A (Realtek)", "Windows WASAPI", 44100, 2),
        (2, "Mic B (Realtek)", "MME", 44100, 2),
    ]
    best = rank_devices(devices)[0]
    assert not best["rate_ok"]
    assert best["idx"] == 1          # still prefers WASAPI


def test_output_only_devices_ignored():
    devices = [
        (1, "Speakers (Senary Audio)", "Windows WASAPI", 48000, 0),
        (2, "Mic (Senary Audio)",      "Windows WDM-KS", 48000, 2),
    ]
    ranked = rank_devices(devices)
    assert len(ranked) == 1
    assert ranked[0]["idx"] == 2


def test_empty_when_only_virtual_devices():
    devices = [(1, "Microphone (Iriun Webcam)", "MME", 44100, 2)]
    assert rank_devices(devices) == []
