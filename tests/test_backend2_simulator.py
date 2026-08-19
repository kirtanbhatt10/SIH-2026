"""Backend 2 simulator verification tests.

Covers encoding determinism, required payloads, reproducibility, and invalid input.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.config import BIT_DURATION, FREQ_0, FREQ_1, PREAMBLE, SAMPLE_RATE
from backend.main import app
from backend.services.payload_service import (
    decode_signal_to_text,
    encode_text_to_signal,
)

client = TestClient(app)


def _text_to_bits(text: str) -> str:
    return "".join(f"{ord(c):08b}" for c in text)


class TestPayloadEncoding:
    def test_hello_payload_roundtrip(self):
        text = "HELLO"
        signal = encode_text_to_signal(text)
        result = decode_signal_to_text(signal)
        assert result.success is True
        assert result.text == text

    def test_sih_backend2_payload_roundtrip(self):
        text = "SIH 2026 BACKEND 2"
        signal = encode_text_to_signal(text)
        result = decode_signal_to_text(signal)
        assert result.success is True
        assert result.text == text

    def test_encoding_is_deterministic(self):
        text = "HELLO"
        bits_expected = PREAMBLE + _text_to_bits(text)
        sig1 = encode_text_to_signal(text)
        sig2 = encode_text_to_signal(text)
        assert np.array_equal(sig1, sig2)
        assert bits_expected == PREAMBLE + _text_to_bits(text)

    def test_bit_mapping_documented_values(self):
        """bit 0 -> FREQ_0, bit 1 -> FREQ_1 per config defaults."""
        assert FREQ_0 == 18500
        assert FREQ_1 == 20500
        assert BIT_DURATION == 0.05
        assert PREAMBLE == "10101010"


class TestReproducibility:
    def test_wav_signal_identical_twice(self):
        text = "SIH 2026 BACKEND 2"
        a = encode_text_to_signal(text)
        b = encode_text_to_signal(text)
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=0.0)


class TestInvalidInput:
    def test_api_rejects_overlong_text(self):
        response = client.post(
            "/simulate/generate",
            json={
                "text": "X" * 257,
                "freq_0_hz": FREQ_0,
                "freq_1_hz": FREQ_1,
                "bit_duration_ms": 50.0,
            },
        )
        assert response.status_code == 422

    def test_api_rejects_unknown_payload_id_on_transmit(self):
        response = client.post(
            "/simulate/transmit",
            json={"payload_id": "nonexistent", "mode": "virtual"},
        )
        assert response.status_code == 404
        assert "Unknown payload_id" in response.json()["detail"]
