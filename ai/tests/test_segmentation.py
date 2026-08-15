"""
Tests for dsp.segmentation — onset/offset detection.

Segmentation is an AI 1 charter deliverable and the prerequisite for the
`duration` field in the AI -> Backend contract (a single 2048-sample frame is
only ~42.7 ms, so duration cannot come from one frame).
"""

import numpy as np
import pytest

from dsp.segmentation import (
    segment_signal,
    StreamingSegmenter,
    band_energy,
    DEFAULT_BAND,
)


SR = 48000


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(7)


def part(dur, freq=None, amp=0.3):
    t = np.arange(int(SR * dur)) / SR
    base = amp * np.sin(2 * np.pi * freq * t) if freq else np.zeros(len(t))
    return base + np.random.randn(int(SR * dur)) * 1e-4


# ── band_energy ──────────────────────────────────────────────────────────

def test_band_energy_responds_to_in_band_tone():
    inband = band_energy(part(0.05, 19000)[:2048], SR)
    silence = band_energy(part(0.05)[:2048], SR)
    assert inband > silence * 100


def test_band_energy_ignores_out_of_band_tone():
    """
    A 5 kHz tone must not register in the 18-21 kHz band.

    Compared against silence-with-noise rather than an absolute floor: both
    samples carry the same 1e-4 white noise, which itself contributes ~1e-9
    of in-band energy. The tone must add nothing on top of that.
    """
    out_of_band = band_energy(part(0.05, 5000)[:2048], SR)
    silence = band_energy(part(0.05)[:2048], SR)
    assert out_of_band < silence * 10

    in_band = band_energy(part(0.05, 19000)[:2048], SR)
    assert in_band > out_of_band * 1000


def test_band_energy_handles_empty():
    assert band_energy(np.array([]), SR) == 0.0


# ── segment_signal (offline) ─────────────────────────────────────────────

def test_finds_two_separate_events():
    audio = np.concatenate([part(0.5), part(2.0, 19000),
                            part(0.5), part(1.0, 19500), part(0.5)])
    events = segment_signal(audio, SR)
    assert len(events) == 2
    assert events[0]["duration_sec"] == pytest.approx(2.0, abs=0.25)
    assert events[1]["duration_sec"] == pytest.approx(1.0, abs=0.25)


def test_returns_nothing_on_silence():
    assert segment_signal(part(3.0), SR) == []


def test_handles_signal_covering_most_of_recording():
    """
    Regression: with a median-based noise floor this returned [] because the
    median landed inside the tone. Percentile split fixed it.
    """
    audio = np.concatenate([part(0.3), part(2.4, 19000), part(0.3)])
    events = segment_signal(audio, SR)
    assert len(events) == 1
    assert events[0]["duration_sec"] > 2.0


def test_finds_three_bursts():
    audio = np.concatenate([part(0.3), part(0.5, 19000),
                            part(0.3), part(0.5, 19000),
                            part(0.3), part(0.5, 19000), part(0.3)])
    assert len(segment_signal(audio, SR)) == 3


def test_short_blip_detected_but_transients_rejected():
    audio = np.concatenate([part(1.0), part(0.15, 19000), part(1.0)])
    assert len(segment_signal(audio, SR)) == 1
    # Same blip, but min_duration raised above its length
    assert segment_signal(audio, SR, min_duration_sec=0.5) == []


def test_merge_gap_joins_close_events():
    audio = np.concatenate([part(0.3), part(0.4, 19000),
                            part(0.05), part(0.4, 19000), part(0.3)])
    merged = segment_signal(audio, SR, merge_gap_sec=0.3)
    assert len(merged) == 1


def test_event_fields_present_and_consistent():
    audio = np.concatenate([part(0.5), part(1.0, 19000), part(0.5)])
    ev = segment_signal(audio, SR)[0]
    for key in ("start_sec", "end_sec", "duration_sec", "start_sample",
                "end_sample", "peak_energy_db", "snr_db", "noise_floor_db"):
        assert key in ev
    assert ev["end_sec"] > ev["start_sec"]
    assert ev["duration_sec"] == pytest.approx(ev["end_sec"] - ev["start_sec"], abs=1e-3)
    assert ev["snr_db"] > 0


def test_empty_input():
    assert segment_signal(np.array([]), SR) == []


# ── StreamingSegmenter ───────────────────────────────────────────────────

def test_streaming_matches_offline_total():
    audio = np.concatenate([part(0.5), part(2.0, 19000),
                            part(0.5), part(1.0, 19500), part(0.5)])
    seg = StreamingSegmenter(sample_rate=SR)
    for i in range(0, len(audio) - 2048, 2048):
        seg.push(audio[i:i + 2048])
    seg.flush()
    assert len(seg.events) == 2
    assert seg.total_active_sec() == pytest.approx(3.0, abs=0.4)


def test_streaming_reports_event_on_close():
    audio = np.concatenate([part(0.5), part(1.0, 19000), part(0.5)])
    seg = StreamingSegmenter(sample_rate=SR)
    closed = [seg.push(audio[i:i + 2048]) for i in range(0, len(audio) - 2048, 2048)]
    assert sum(1 for c in closed if c) == 1


def test_streaming_is_active_flag():
    seg = StreamingSegmenter(sample_rate=SR)
    for chunk in np.split(part(0.5)[:2048 * 10], 10):
        seg.push(chunk)
    assert not seg.is_active

    tone = part(1.0, 19000)
    for i in range(0, 2048 * 5, 2048):
        seg.push(tone[i:i + 2048])
    assert seg.is_active


def test_streaming_flush_closes_open_event():
    seg = StreamingSegmenter(sample_rate=SR)
    for chunk in np.split(part(0.5)[:2048 * 10], 10):
        seg.push(chunk)
    tone = part(1.0, 19000)
    for i in range(0, len(tone) - 2048, 2048):
        seg.push(tone[i:i + 2048])

    ev = seg.flush()
    assert ev is not None
    assert ev.get("truncated") is True


def test_streaming_reset():
    seg = StreamingSegmenter(sample_rate=SR)
    audio = np.concatenate([part(0.3), part(0.5, 19000), part(0.3)])
    for i in range(0, len(audio) - 2048, 2048):
        seg.push(audio[i:i + 2048])
    seg.flush()
    assert seg.events

    seg.reset()
    assert seg.events == []
    assert not seg.is_active
    assert seg.noise_floor_db is None
