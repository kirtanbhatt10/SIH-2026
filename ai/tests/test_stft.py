"""
Tests for dsp.stft — the explicit, configurable STFT.

STFT is an AI 1 charter deliverable. It previously existed only implicitly
inside SpectrogramGenerator (hop == frame_size, not configurable), and
INTEGRATION_GUIDE.md documented a HOP_LENGTH that existed nowhere in code.
"""

import numpy as np
import pytest

from dsp.stft import (
    stft, istft, spectrogram_db,
    band_power_over_time, dominant_frequency_over_time,
)


SR = 48000


def tone(freq, dur=1.0, amp=0.3, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


# ── Shape and axes ───────────────────────────────────────────────────────

def test_output_shapes_consistent():
    freqs, times, Z = stft(tone(19000), SR, n_fft=2048, hop=512)
    assert Z.shape[0] == len(freqs) == 2048 // 2 + 1
    assert Z.shape[1] == len(times)
    assert np.iscomplexobj(Z)


def test_frequency_resolution():
    freqs, _, _ = stft(tone(19000), SR, n_fft=2048, hop=512)
    assert freqs[1] - freqs[0] == pytest.approx(SR / 2048, abs=0.01)
    assert freqs[-1] == pytest.approx(SR / 2, abs=1.0)


def test_hop_controls_frame_count():
    """Smaller hop = more frames. This is the configurability that was missing."""
    _, _, z_dense = stft(tone(19000), SR, n_fft=2048, hop=512)
    _, _, z_sparse = stft(tone(19000), SR, n_fft=2048, hop=2048)
    assert z_dense.shape[1] > z_sparse.shape[1] * 3


def test_default_hop_is_75_percent_overlap():
    _, _, z_default = stft(tone(19000), SR, n_fft=2048)
    _, _, z_explicit = stft(tone(19000), SR, n_fft=2048, hop=512)
    assert z_default.shape == z_explicit.shape


def test_no_overlap_matches_spectrogram_generator_behaviour():
    """hop == n_fft reproduces the streaming generator's framing."""
    audio = tone(19000, dur=1.0)
    _, _, Z = stft(audio, SR, n_fft=2048, hop=2048)
    assert Z.shape[1] == len(audio) // 2048


# ── Correctness ──────────────────────────────────────────────────────────

def test_detects_correct_dominant_frequency():
    freqs, _, Z = stft(tone(19000), SR, n_fft=2048, hop=512)
    dom = dominant_frequency_over_time(Z, freqs)
    assert np.median(dom) == pytest.approx(19000, abs=30)


def test_tracks_frequency_change_over_time():
    audio = np.concatenate([tone(19000, 0.5), tone(20500, 0.5)])
    freqs, times, Z = stft(audio, SR, n_fft=2048, hop=512)
    dom = dominant_frequency_over_time(Z, freqs)
    first = dom[:len(dom) // 3]
    last = dom[-len(dom) // 3:]
    assert np.median(first) == pytest.approx(19000, abs=50)
    assert np.median(last) == pytest.approx(20500, abs=50)


def test_band_power_responds_to_in_band_signal():
    _, _, z_in = stft(tone(19000), SR, n_fft=2048, hop=512)
    freqs, _, z_out = stft(tone(5000), SR, n_fft=2048, hop=512)
    p_in = band_power_over_time(z_in, freqs).mean()
    p_out = band_power_over_time(z_out, freqs).mean()
    assert p_in > p_out * 1000


# ── Round trip ───────────────────────────────────────────────────────────

def test_istft_reconstructs_interior_exactly():
    """
    Interior reconstruction must be at machine precision. Edges are expected
    to be degraded (fewer overlapping frames), so they are excluded.
    """
    x = tone(19000, dur=1.0)
    _, _, Z = stft(x, SR, n_fft=2048, hop=512)
    r = istft(Z, hop=512, length=len(x))

    a, b = 2048, len(x) - 2048
    err = np.sqrt(np.mean((r[a:b] - x[a:b]) ** 2)) / np.sqrt(np.mean(x[a:b] ** 2))
    assert err < 1e-9, f"interior round-trip error {err:.2e}"


def test_istft_returns_requested_length():
    x = tone(19000, dur=0.5)
    _, _, Z = stft(x, SR, n_fft=2048, hop=512)
    assert len(istft(Z, hop=512, length=len(x))) == len(x)


# ── dB conversion ────────────────────────────────────────────────────────

def test_spectrogram_db_is_bounded():
    _, _, Z = stft(tone(19000), SR, n_fft=2048, hop=512)
    db = spectrogram_db(Z)
    assert db.max() == pytest.approx(0.0, abs=1e-6)
    assert db.min() >= -120.0
    assert np.all(np.isfinite(db))


def test_spectrogram_db_handles_silence():
    _, _, Z = stft(np.zeros(SR), SR, n_fft=2048, hop=512)
    db = spectrogram_db(Z)
    assert np.all(np.isfinite(db))


# ── Edge cases ───────────────────────────────────────────────────────────

def test_signal_shorter_than_frame_is_padded():
    freqs, times, Z = stft(np.random.randn(500), SR, n_fft=2048, hop=512)
    assert Z.shape[1] == 1


def test_windows_supported():
    for w in ("hann", "hamming", "blackman", "rect"):
        _, _, Z = stft(tone(19000, 0.2), SR, n_fft=1024, hop=256, window=w)
        assert np.all(np.isfinite(np.abs(Z)))


def test_unknown_window_raises():
    with pytest.raises(ValueError):
        stft(tone(19000, 0.1), SR, window="triangle")


def test_invalid_hop_raises():
    with pytest.raises(ValueError):
        stft(tone(19000, 0.1), SR, hop=0)


def test_center_mode_runs():
    _, _, Z = stft(tone(19000, 0.3), SR, n_fft=2048, hop=512, center=True)
    assert np.all(np.isfinite(np.abs(Z)))
