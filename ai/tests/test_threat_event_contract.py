"""
Tests for the AI -> Backend detection contract (charter §14).

These pin the shape Backend 1 consumes. Changing any field name here is a
breaking change and must be coordinated with Backend 1 first.
"""

import numpy as np
import pytest

from dsp import DSPPipeline


SR = 48000
CHUNK = 2048


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(11)


def run(pipeline, audio):
    for i in range(0, len(audio) - CHUNK, CHUNK):
        pipeline.process(audio[i:i + CHUNK])
    return pipeline.to_threat_event()


# ── Documented API that used to raise ────────────────────────────────────

def test_documented_result_keys_exist():
    """
    INTEGRATION_GUIDE.md told Backend 1 to use these; they raised KeyError.
    """
    r = DSPPipeline().process(np.random.randn(CHUNK) * 0.01)
    assert "is_suspicious" in r
    assert "suspicion_score" in r
    assert r["is_suspicious"] == r["frame_is_suspicious"]
    assert r["suspicion_score"] == r["frame_suspicion_score"]


def test_class_level_config_call_works():
    """DSPPipeline.get_config() was documented as a classmethod and wasn't."""
    cfg = DSPPipeline.get_default_config()
    assert cfg["sample_rate"] == 48000
    assert cfg["num_features"] == 32


def test_snr_surfaced_at_top_level():
    """Charter §14 asks for `snr`; it was computed but never exposed."""
    r = DSPPipeline().process(np.random.randn(CHUNK) * 0.01)
    assert "snr_db" in r
    assert "peak_frequency_hz" in r


# ── Contract shape ───────────────────────────────────────────────────────

REQUIRED_FIELDS = {
    "schema_version", "detected", "confidence", "risk", "suspicion_score",
    "frequency_start", "frequency_end", "carrier_freqs", "duration",
    "pattern", "snr", "chunks_analyzed", "timestamp",
}


def test_event_has_all_contract_fields():
    p = DSPPipeline()
    event = run(p, p.generate_attack_signal("fsk", duration_sec=2.0))
    assert REQUIRED_FIELDS.issubset(event.keys())


def test_event_is_json_serializable():
    """Backend returns this over HTTP; numpy scalars would break json.dumps."""
    import json
    p = DSPPipeline()
    event = run(p, p.generate_attack_signal("fsk", duration_sec=2.0))
    json.dumps(event)


def test_detects_fsk_with_correct_carriers():
    p = DSPPipeline()
    event = run(p, p.generate_attack_signal(
        "fsk", duration_sec=3.0, freq_mark=19000, freq_space=20500, baud_rate=25))

    assert event["detected"] is True
    assert event["pattern"] == "fsk"
    assert len(event["carrier_freqs"]) >= 2
    assert event["frequency_start"] == pytest.approx(19000, abs=200)
    assert event["frequency_end"] == pytest.approx(20500, abs=200)


def test_duration_comes_from_segmentation_not_one_frame():
    """A single chunk is ~42.7 ms; a 3 s signal must report ~3 s."""
    p = DSPPipeline()
    event = run(p, p.generate_attack_signal("fsk", duration_sec=3.0))
    assert event["duration"] > 1.0, "duration collapsed to a single frame"
    assert event["duration"] == pytest.approx(3.0, abs=0.5)


def test_carrier_freqs_preserved_alongside_range():
    """
    Collapsing [19000, 20500] to a min/max range destroys the two-carrier
    evidence that IS the FSK signature. Both must be present.
    """
    p = DSPPipeline()
    event = run(p, p.generate_attack_signal(
        "fsk", duration_sec=2.0, freq_mark=19000, freq_space=20500))
    assert isinstance(event["carrier_freqs"], list)
    assert len(event["carrier_freqs"]) >= 2


def test_snr_is_measured():
    p = DSPPipeline()
    event = run(p, p.generate_attack_signal("fsk", duration_sec=2.0))
    assert event["snr"] > 0


# ── Negative cases ───────────────────────────────────────────────────────

def test_quiet_noise_is_not_a_threat():
    p = DSPPipeline()
    event = run(p, np.random.randn(SR) * 0.005)
    assert event["detected"] is False
    assert event["risk"] == "LOW"
    assert event["carrier_freqs"] == []


def test_no_chunks_processed_is_safe():
    """Backend may call this before any audio arrives."""
    event = DSPPipeline().to_threat_event()
    assert event["detected"] is False
    assert event["chunks_analyzed"] == 0
    assert event["duration"] == 0.0


def test_reset_clears_event_state():
    p = DSPPipeline()
    run(p, p.generate_attack_signal("fsk", duration_sec=2.0))
    p.reset()
    event = p.to_threat_event()
    assert event["chunks_analyzed"] == 0
    assert event["detected"] is False


# ── Risk mapping ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (0.00, "LOW"), (0.30, "LOW"), (0.44, "LOW"),
    (0.45, "MEDIUM"), (0.60, "MEDIUM"), (0.74, "MEDIUM"),
    (0.75, "HIGH"), (0.90, "HIGH"), (1.00, "HIGH"),
])
def test_risk_band_mapping(score, expected):
    assert DSPPipeline.score_to_risk(score) == expected


def test_risk_is_low_when_not_detected():
    p = DSPPipeline()
    event = run(p, np.random.randn(SR) * 0.005)
    assert event["risk"] == "LOW"


def test_two_confidences_are_distinct_fields():
    """
    `confidence` = modulation-classifier certainty.
    `suspicion_score` = threat heuristic.
    They must not be conflated into one number.
    """
    p = DSPPipeline()
    event = run(p, p.generate_attack_signal("fsk", duration_sec=2.0))
    assert "confidence" in event
    assert "suspicion_score" in event
    assert 0.0 <= event["confidence"] <= 1.0
    assert 0.0 <= event["suspicion_score"] <= 1.0
