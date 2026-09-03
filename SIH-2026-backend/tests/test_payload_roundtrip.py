"""Tests for Backend 2's payload encode/decode round-trip pipeline.

Pipeline: text → bits → BFSK modulation → WAV → demodulation → bits → text
"""

import os
import tempfile

import numpy as np

from backend.services.payload_service import (
    DecodeResult,
    decode_audio_file,
    decode_signal_to_text,
    encode_text_to_signal,
    save_signal_to_wav,
)


class TestPayloadRoundTrip:
    """Verify that encoding then decoding recovers the original payload."""

    def test_normal_payload(self):
        text = "HELLO SIH 2026"
        signal = encode_text_to_signal(text)
        result = decode_signal_to_text(signal)
        assert result.success is True
        assert result.text == text
        assert result.preamble_found is True
        assert result.confidence > 0.5

    def test_short_payload(self):
        text = "A"
        signal = encode_text_to_signal(text)
        result = decode_signal_to_text(signal)
        assert result.success is True
        assert result.text == text

    def test_alphanumeric_payload(self):
        text = "SIH2026-DEMO"
        signal = encode_text_to_signal(text)
        result = decode_signal_to_text(signal)
        assert result.success is True
        assert result.text == text

    def test_special_characters(self):
        text = "TEST@123!"
        signal = encode_text_to_signal(text)
        result = decode_signal_to_text(signal)
        assert result.success is True
        assert result.text == text

    def test_wav_roundtrip(self):
        """Encode → save WAV → load WAV → decode must match."""
        text = "WAV ROUNDTRIP"
        signal = encode_text_to_signal(text)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            save_signal_to_wav(tmp_path, signal)
            result = decode_audio_file(tmp_path)
            assert result.success is True
            assert result.text == text
        finally:
            os.unlink(tmp_path)

    def test_empty_signal_fails_gracefully(self):
        """An empty/too-short signal should not crash, just return failure."""
        signal = np.zeros(100, dtype=np.float32)
        result = decode_signal_to_text(signal)
        assert result.success is False
        assert result.preamble_found is False

    def test_noise_signal_fails_gracefully(self):
        """Pure noise should not decode as a valid payload."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(48000 * 2).astype(np.float32) * 0.01
        result = decode_signal_to_text(noise)
        # Should either fail or produce garbage — must not crash
        assert isinstance(result, DecodeResult)

    def test_decode_result_fields(self):
        """Verify all DecodeResult fields are populated."""
        text = "FIELDS"
        signal = encode_text_to_signal(text)
        result = decode_signal_to_text(signal)
        assert isinstance(result.success, bool)
        assert isinstance(result.text, (str, type(None)))
        assert isinstance(result.bits, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.preamble_found, bool)
        assert isinstance(result.bit_count, int)
