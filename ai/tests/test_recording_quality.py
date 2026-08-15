"""
Tests for dsp.recording_quality — the gate that rejects dead captures.

These encode the failure we actually hit on 15 Aug 2026: an entire physical
test sweep saved at 1-2 LSB with no tone present, which was nearly presented
as measured hardware evidence.
"""

import numpy as np
import pytest

from dsp.recording_quality import check_recording, LSB


SR = 48000


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed the RNG so these tests never flake."""
    np.random.seed(1234)


def tone(freq, dur=1.0, amp=0.2, sr=SR, noise=1e-4):
    t = np.arange(int(sr * dur)) / sr
    return amp * np.sin(2 * np.pi * freq * t) + np.random.randn(len(t)) * noise


# ── The real-world failure ───────────────────────────────────────────────

def test_rejects_dead_capture_like_rx_18000():
    """Reproduce rx_18000.wav: 12 s, peak 2 LSB, ~5 unique values."""
    n = SR * 12
    audio = np.random.choice([0.0, LSB, -LSB], size=n, p=[0.77, 0.115, 0.115])
    r = check_recording(audio, SR)
    assert not r["ok"]
    assert any("DEAD CAPTURE" in p for p in r["problems"])
    assert any("UNIQUE SAMPLE VALUES" in p for p in r["problems"])


def test_rejects_one_lsb_baseline():
    """Reproduce baseline_quiet.wav: 3 unique values."""
    audio = np.random.choice([0.0, LSB, -LSB], size=SR * 12)
    assert not check_recording(audio, SR)["ok"]


def test_accepts_real_audio_like_clap():
    """clap.wav measured 8064 LSB and passed — a working mic must not be flagged."""
    audio = tone(1000, dur=5.0, amp=0.25, noise=1e-3)
    r = check_recording(audio, SR)
    assert r["ok"], r["summary"]
    assert r["peak_lsb"] > 1000


def test_accepts_quiet_but_valid_recording():
    """rx_1000.wav at 557 LSB is quiet but genuinely usable."""
    audio = tone(1000, dur=5.0, amp=0.017, noise=1e-4)
    r = check_recording(audio, SR)
    assert r["ok"], r["summary"]


# ── Tone verification ────────────────────────────────────────────────────

def test_detects_tone_when_present():
    r = check_recording(tone(18000, amp=0.2), SR, expected_hz=18000)
    assert r["ok"]
    assert r["tone"]["tone_present"]
    assert abs(r["tone"]["frequency_error_hz"]) < 50


def test_rejects_when_tone_absent():
    """Loud audio, but not the tone we claim to have transmitted."""
    r = check_recording(tone(1000, amp=0.2), SR, expected_hz=18000)
    assert not r["ok"]
    assert any("TONE ABSENT" in p for p in r["problems"])
    assert not r["tone"]["tone_present"]


# ── Other guards ─────────────────────────────────────────────────────────

def test_rejects_all_zeros():
    r = check_recording(np.zeros(SR), SR)
    assert not r["ok"]


def test_detects_clipping():
    audio = np.clip(tone(1000, amp=3.0), -1.0, 1.0)
    r = check_recording(audio, SR)
    assert any("CLIPPING" in p for p in r["problems"])


def test_warns_on_quiet_but_passes():
    audio = tone(1000, amp=100 * LSB, noise=1e-6)
    r = check_recording(audio, SR)
    assert r["ok"]
    assert r["warnings"]


def test_accepts_int16_input():
    """Integer PCM must be scaled, not treated as raw floats."""
    audio = (tone(1000, amp=0.5) * 32767).astype(np.int16)
    r = check_recording(audio, SR)
    assert r["ok"], r["summary"]


def test_report_fields_present():
    r = check_recording(tone(1000), SR)
    for key in ("ok", "summary", "peak_lsb", "unique_values", "rms_dbfs", "duration_sec"):
        assert key in r
