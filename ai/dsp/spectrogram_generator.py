"""
THING 3 — Spectrogram Generator (The Picture Maker)
=====================================================
Produces frequency-vs-time "pictures" in two modes:

1. Real-time slices  → JSON payloads for Frontend WebSocket visualizer
2. Full spectrograms → PNG/numpy images for demo, reports, and ML

Who uses this:
    Frontend 1 — Live dashboard spectrogram (real-time slices via WebSocket)
    Frontend 2 — Demo mode visualization & attack simulator display
    AI 2       — (optional) spectrogram images as CNN input
    PPT Lead   — Pretty pictures for the presentation slides

Design note:
    We keep a rolling buffer of recent FFT frames so the frontend can
    request a full spectrogram image at any time (e.g., after an alert).
"""

import time
import json
from collections import deque

import numpy as np


class SpectrogramGenerator:
    """
    Builds spectrograms from audio chunks in real time.

    Usage (Frontend 1 — call from your WebSocket handler):
    ──────────────────────────────────────────────────────
        from dsp import SpectrogramGenerator

        spec = SpectrogramGenerator(sample_rate=48000)

        # Every time a new audio chunk arrives:
        json_payload = spec.add_chunk(audio_chunk)
        # → send json_payload over WebSocket to the browser

        # When user wants a full spectrogram image:
        image_array = spec.get_full_spectrogram()
        # → shape (n_freq_bins, n_time_frames), values in dB
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        ultrasonic_low: float = 18000.0,
        ultrasonic_high: float = 22000.0,
        history_seconds: float = 10.0,
        ui_bins: int = 64,
    ):
        """
        Parameters
        ----------
        sample_rate : int
            Audio sample rate in Hz.
        n_fft : int
            FFT window size.
        ultrasonic_low, ultrasonic_high : float
            Frequency range to visualize (Hz).
        history_seconds : float
            How many seconds of spectrogram history to keep in the
            rolling buffer (for full spectrogram snapshots).
        ui_bins : int
            Number of frequency bins to send to the frontend per frame.
            More bins = higher resolution but more bandwidth.
            64 is a good balance for smooth real-time rendering.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.ui_bins = ui_bins

        # Frequency axis
        self.freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        # Ultrasonic band mask
        self.us_mask = (self.freq_bins >= ultrasonic_low) & (self.freq_bins <= ultrasonic_high)
        self.us_freqs = self.freq_bins[self.us_mask]

        # Evenly spaced target frequencies for UI interpolation
        self.ui_freq_axis = np.linspace(ultrasonic_low, ultrasonic_high, ui_bins)

        # Rolling history buffer
        chunk_duration = n_fft / sample_rate  # ~42.7 ms per frame
        max_frames = int(history_seconds / chunk_duration)
        self._history = deque(maxlen=max_frames)
        self._timestamps = deque(maxlen=max_frames)

        # Track peak for global normalization
        self._global_max_db = -120.0

    # ──────────────────────────────────────────────────────────
    # Real-time API (Frontend 1 & 2)
    # ──────────────────────────────────────────────────────────

    def add_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Process one audio chunk and return a JSON string ready for WebSocket.

        Parameters
        ----------
        audio_chunk : np.ndarray
            Audio samples (filtered or raw).

        Returns
        -------
        str — JSON payload:
            {
                "timestamp": 1720000000.123,
                "freq_axis": [18000, 18062, ..., 22000],    // ui_bins floats
                "magnitudes_db": [-80.2, -45.1, ..., -70.5],// ui_bins floats (dB)
                "magnitudes_norm": [0.0, 0.6, ..., 0.1],    // ui_bins floats [0–1]
                "peak_freq_hz": 19500.0,
                "peak_magnitude_db": -12.3,
                "ultrasonic_active": true
            }
        """
        frame = self._compute_frame(audio_chunk)
        self._history.append(frame["magnitudes_db"])
        self._timestamps.append(frame["timestamp"])
        return json.dumps(frame)

    def add_chunk_dict(self, audio_chunk: np.ndarray) -> dict:
        """Same as add_chunk() but returns a Python dict instead of JSON string."""
        frame = self._compute_frame(audio_chunk)
        self._history.append(frame["magnitudes_db"])
        self._timestamps.append(frame["timestamp"])
        return frame

    # ──────────────────────────────────────────────────────────
    # Full spectrogram snapshot
    # ──────────────────────────────────────────────────────────

    def get_full_spectrogram(self) -> dict:
        """
        Returns the full rolling spectrogram as a 2D array.

        Returns
        -------
        dict:
            "spectrogram_db"  : np.ndarray, shape (n_freq_bins, n_time_frames)
            "freq_axis"       : np.ndarray, shape (n_freq_bins,)  in Hz
            "time_axis"       : list of float timestamps
            "duration_sec"    : float
        """
        if len(self._history) == 0:
            return {
                "spectrogram_db": np.array([]).reshape(self.ui_bins, 0),
                "freq_axis": self.ui_freq_axis.tolist(),
                "time_axis": [],
                "duration_sec": 0.0,
            }

        # Stack columns: each frame is one column
        spec = np.column_stack(list(self._history))  # (ui_bins, n_frames)
        timestamps = list(self._timestamps)

        duration = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0

        return {
            "spectrogram_db": spec,
            "freq_axis": self.ui_freq_axis.tolist(),
            "time_axis": timestamps,
            "duration_sec": duration,
        }

    def save_spectrogram_image(self, filepath: str = "spectrogram.png"):
        """
        Save the current rolling spectrogram as a PNG image.
        Requires matplotlib (only used for demo/report generation,
        NOT in the real-time path).

        Parameters
        ----------
        filepath : str
            Output file path (PNG, JPG, SVG, PDF all supported).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for save_spectrogram_image(). "
                "Install with: pip install matplotlib"
            )

        data = self.get_full_spectrogram()
        spec_db = data["spectrogram_db"]

        if spec_db.size == 0:
            print("No spectrogram data to save.")
            return

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(
            spec_db,
            aspect="auto",
            origin="lower",
            cmap="inferno",
            extent=[
                0,
                data["duration_sec"],
                self.ui_freq_axis[0] / 1000,
                self.ui_freq_axis[-1] / 1000,
            ],
            vmin=-100,
            vmax=0,
        )
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Frequency (kHz)", fontsize=12)
        ax.set_title("Acoustic Cybersecurity System — Ultrasonic Spectrogram", fontsize=14, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, label="Magnitude (dB)")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"Spectrogram saved to {filepath}")

    # ──────────────────────────────────────────────────────────
    # Internal computation
    # ──────────────────────────────────────────────────────────

    def _compute_frame(self, audio_chunk: np.ndarray) -> dict:
        """Compute one spectrogram frame from an audio chunk."""
        audio = np.asarray(audio_chunk, dtype=np.float64)

        # Windowed FFT
        windowed = audio * np.hanning(len(audio))
        fft_mag = np.abs(np.fft.rfft(windowed, n=self.n_fft))

        # Extract ultrasonic band
        us_mag = fft_mag[self.us_mask]

        # Interpolate to fixed UI bin count
        if len(us_mag) >= 2:
            ui_mag = np.interp(self.ui_freq_axis, self.us_freqs, us_mag)
        else:
            ui_mag = np.zeros(self.ui_bins)

        # Convert to dB
        mag_db = 20.0 * np.log10(ui_mag + 1e-12)

        # Clamp to reasonable range
        mag_db = np.clip(mag_db, -120.0, 0.0)

        # Track global max for normalization
        frame_max = float(np.max(mag_db))
        if frame_max > self._global_max_db:
            self._global_max_db = frame_max

        # Normalize to [0, 1] for easy frontend rendering
        range_db = max(self._global_max_db - (-120.0), 1.0)
        mag_norm = (mag_db - (-120.0)) / range_db
        mag_norm = np.clip(mag_norm, 0.0, 1.0)

        # Peak detection
        peak_idx = int(np.argmax(ui_mag))
        peak_freq = float(self.ui_freq_axis[peak_idx])
        peak_db = float(mag_db[peak_idx])
        peak_snr_db = float(peak_db - float(np.mean(mag_db)))

        # Active flag — simple threshold
        ultrasonic_active = bool(peak_db > -40.0 and np.mean(mag_db) > -80.0)

        return {
            "timestamp": time.time(),
            "freq_axis": self.ui_freq_axis.tolist(),
            "magnitudes_db": mag_db.tolist(),
            "magnitudes_norm": mag_norm.tolist(),
            "peak_freq_hz": peak_freq,
            "peak_magnitude_db": peak_db,
            "peak_snr_db": peak_snr_db,
            "ultrasonic_active": ultrasonic_active,
        }

    def clear_history(self):
        """Clear the rolling spectrogram buffer."""
        self._history.clear()
        self._timestamps.clear()
        self._global_max_db = -120.0

    @property
    def frame_count(self) -> int:
        """Number of frames currently in the rolling buffer."""
        return len(self._history)


if __name__ == "__main__":
    print("==================================================")
    print("  Testing Spectrogram Generator...")
    print("==================================================")

    sr = 48000
    n_fft = 2048
    spec_gen = SpectrogramGenerator(sample_rate=sr, n_fft=n_fft, history_seconds=5.0, ui_bins=64)

    # Test 1: Single chunk payload
    t = np.arange(n_fft) / sr
    loud_us = np.sin(2 * np.pi * 19500 * t) * 0.9
    frame = spec_gen.add_chunk_dict(loud_us)

    assert len(frame["freq_axis"]) == 64
    assert len(frame["magnitudes_db"]) == 64
    assert len(frame["magnitudes_norm"]) == 64
    assert frame["peak_freq_hz"] > 18000 and frame["peak_freq_hz"] < 22000
    assert isinstance(frame["ultrasonic_active"], bool)
    assert np.isfinite(frame["peak_snr_db"])
    print("  [PASS] Single chunk frame generated with valid 64-bin UI payload")

    # Test 2: Full Spectrogram snapshot
    for _ in range(10):
        spec_gen.add_chunk(np.random.randn(n_fft))
    full = spec_gen.get_full_spectrogram()
    assert full["spectrogram_db"].shape[0] == 64
    print(f"  [PASS] Full spectrogram shape: {full['spectrogram_db'].shape}")

    # Test 3: Clear history
    spec_gen.clear_history()
    assert spec_gen.frame_count == 0
    print("  [PASS] History cleared successfully")

    print("\n[ALL TESTS PASSED] spectrogram_generator.py")
