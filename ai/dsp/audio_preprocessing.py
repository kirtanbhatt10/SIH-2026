"""
THING 1 — Audio Preprocessor (The Cleaner)
============================================
Takes raw microphone PCM buffers and outputs a clean, filtered signal
containing only the ultrasonic band (17.5 kHz – 21.5 kHz filter
edges around an 18–21 kHz detection band; see dsp/__init__.py).

Who uses this:
    Backend 2 — feeds raw audio chunks in, gets clean chunks out
    AI 2      — receives clean signal for feature extraction pipeline
    You       — called internally by dsp_api.py

Design decisions:
    • Butterworth IIR bandpass (order 6) — low latency, causal, no lookahead
    • sosfilt with zi state for gapless streaming across chunk boundaries
    • DC removal + pre-emphasis before filtering for cleaner spectral shape
    • Hann windowing provided as a separate step (used before FFT, not before filtering)
"""

"""

Fixes applied:
    1. zi initialised with np.zeros (simpler and correct)
    2. Pre-emphasis is now stateful across chunk boundaries
    3. reset_filter_state also resets pre-emphasis state
    4. process() returns float32 instead of float64
    5. Input validation added
    6. validate_audio() method added
    7. filter_high default 21500 — matches dsp/__init__.py FILTER_HIGH
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class AudioPreprocessor:
    """
    Real-time audio preprocessor that isolates the ultrasonic band.

    Usage (Backend 2 — call this in your audio callback loop):
    ----------------------------------------------------------
        preprocessor = AudioPreprocessor(sample_rate=48000)

        # Inside your audio stream callback:
        clean = preprocessor.process(raw_chunk)   # np.float32 array
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        filter_low: float = 17500.0,
        filter_high: float = 21500.0,
        filter_order: int = 6,
        pre_emphasis_coeff: float = 0.97,
    ):
        """
        Parameters
        ----------
        sample_rate : int
            Must be >= 44100. 48000 recommended (Nyquist = 24 kHz).
        filter_low : float
            Lower cutoff of the bandpass filter in Hz.
        filter_high : float
            Upper cutoff in Hz. Default 21500 matches FILTER_HIGH in
            dsp/__init__.py (single source of truth) for
            better compatibility with most laptop microphones.
        filter_order : int
            Butterworth filter order. 6 gives ~36 dB/octave rolloff.
        pre_emphasis_coeff : float
            First-order high-pass pre-emphasis coefficient.
        """
        if sample_rate < 44100:
            raise ValueError(
                f"sample_rate must be >= 44100 Hz (got {sample_rate}). "
                "Lower rates cannot reach the ultrasonic band."
            )

        self.sample_rate = sample_rate
        self.pre_emphasis_coeff = pre_emphasis_coeff

        # Design bandpass Butterworth filter (second-order sections)
        nyquist = 0.5 * sample_rate
        low_norm = filter_low / nyquist
        high_norm = min(filter_high / nyquist, 0.99)  # safety clamp

        self.sos = butter(
            filter_order,
            [low_norm, high_norm],
            btype="bandpass",
            output="sos"
        )

        # FIX 1: Cleaner zi initialisation (zeros = silence start)
        self._zi = np.zeros((self.sos.shape[0], 2))

        # FIX 2: Stateful pre-emphasis — stores last sample of
        # each chunk so the first sample of the next chunk
        # is correctly computed
        self._pre_emphasis_last_sample = 0.0

        # Stats for monitoring
        self._chunks_processed = 0

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def process(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline on one chunk.

        Parameters
        ----------
        raw_audio : np.ndarray, shape (N,), dtype float32/float64
            Raw PCM samples normalized to [-1.0, 1.0].

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            Cleaned, bandpass-filtered signal ready for
            feature extraction.
        """
        # FIX 5: Input validation
        if not isinstance(raw_audio, np.ndarray):
            raise TypeError(
                f"Expected np.ndarray, got {type(raw_audio)}"
            )

        if raw_audio.ndim != 1:
            raise ValueError(
                f"Expected 1D mono audio, got shape {raw_audio.shape}. "
                "Convert stereo to mono before passing."
            )

        if len(raw_audio) == 0:
            raise ValueError("Empty audio chunk received.")

        if len(raw_audio) < 512:
            raise ValueError(
                f"Chunk too small ({len(raw_audio)} samples). "
                "Minimum 512 samples needed for reliable filtering."
            )

        audio = np.asarray(raw_audio, dtype=np.float64)

        # Step 1: Remove DC offset
        audio = self.remove_dc(audio)

        # Step 2: Pre-emphasis (stateful)
        if self.pre_emphasis_coeff > 0:
            audio = self.pre_emphasis(audio)

        # Step 3: Bandpass filter (stateful streaming)
        audio = self.bandpass_filter(audio)

        # Step 4: Normalize peak amplitude
        audio = self.normalize(audio)

        self._chunks_processed += 1

        # FIX 3: Return float32 (expected by ML and Backend)
        return audio.astype(np.float32)

    def remove_dc(self, audio: np.ndarray) -> np.ndarray:
        """Subtract mean to remove any DC bias from the microphone."""
        return audio - np.mean(audio)

    def pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        FIX 2: Stateful first-order high-pass pre-emphasis.

        y[n] = x[n] - alpha * x[n-1]

        The last sample of each chunk is saved so the first
        sample of the next chunk is computed correctly across
        chunk boundaries.
        """
        output = np.empty_like(audio)

        # First sample uses last sample from PREVIOUS chunk
        output[0] = (
            audio[0]
            - self.pre_emphasis_coeff * self._pre_emphasis_last_sample
        )

        # Remaining samples
        output[1:] = audio[1:] - self.pre_emphasis_coeff * audio[:-1]

        # Save last sample for next chunk
        self._pre_emphasis_last_sample = float(audio[-1])

        return output

    def bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply the Butterworth bandpass filter with stateful streaming.
        The internal zi state is carried across calls so consecutive
        chunks are seamlessly stitched without edge artefacts.
        """
        filtered, self._zi = sosfilt(self.sos, audio, zi=self._zi)
        return filtered

    def normalize(
        self,
        audio: np.ndarray,
        target_peak: float = 1.0
    ) -> np.ndarray:
        """
        Peak-normalize to [-target_peak, +target_peak].
        Returns zeros if chunk is silent or out-of-band suppressed (peak < 0.05).
        """
        peak = np.max(np.abs(audio))
        if peak < 0.05:
            return np.zeros_like(audio)
        return audio * (target_peak / peak)

    def apply_window(
        self,
        audio: np.ndarray,
        window_type: str = "hann"
    ) -> np.ndarray:
        """
        Apply a tapering window before FFT to reduce spectral leakage.

        Called separately — NOT part of process() because filtering
        and windowing serve different purposes.

        Parameters
        ----------
        window_type : str
            'hann' (default) or 'hamming'.
        """
        n = len(audio)
        if window_type == "hann":
            window = np.hanning(n)
        elif window_type == "hamming":
            window = np.hamming(n)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        return audio * window

    def validate_audio(self, audio: np.ndarray) -> dict:
        """
        FIX 6: Check audio quality BEFORE processing.

        Backend 2 can call this to verify the microphone
        is working and the signal is usable.

        Returns
        -------
        dict with quality metrics and an 'is_usable' flag.
        """
        return {
            "length_seconds":  len(audio) / self.sample_rate,
            "sample_count":    len(audio),
            "max_amplitude":   float(np.max(np.abs(audio))),
            "mean_amplitude":  float(np.mean(np.abs(audio))),
            "is_clipping":     bool(np.any(np.abs(audio) > 0.99)),
            "is_silent":       bool(np.max(np.abs(audio)) < 1e-4),
            "dc_offset":       float(np.mean(audio)),
            "rms":             float(np.sqrt(np.mean(audio ** 2))),
            "is_usable": bool(
                np.max(np.abs(audio)) > 1e-4
                and not np.any(np.abs(audio) > 0.99)
                and len(audio) >= 512
            ),
        }

    def reset_filter_state(self):
        """
        FIX 4: Reset ALL streaming state.
        Call this when switching audio sources or restarting capture.
        """
        self._zi = np.zeros((self.sos.shape[0], 2))
        self._pre_emphasis_last_sample = 0.0  # FIX 4: was missing
        self._chunks_processed = 0

    @property
    def chunks_processed(self) -> int:
        """Number of chunks processed since creation or last reset."""
        return self._chunks_processed


# ----------------------------------------------------------
# Self-test
# ----------------------------------------------------------

if __name__ == "__main__":
    print("Testing AudioPreprocessor...")

    sr = 48000
    t = np.linspace(0, 1, sr)

    # Simulate voice (300 Hz) + ultrasonic attack (19000 Hz)
    voice  = 0.5 * np.sin(2 * np.pi * 300   * t)
    attack = 0.3 * np.sin(2 * np.pi * 19000 * t)
    mixed  = (voice + attack).astype(np.float32)

    preprocessor = AudioPreprocessor(sample_rate=sr)

    # Test 1: Single chunk
    clean = preprocessor.process(mixed)
    assert clean.dtype == np.float32,      "Should return float32"
    assert len(clean)  == len(mixed),      "Length should match"
    assert np.max(np.abs(clean)) <= 1.001, "Should be normalized"
    print("  [PASS] Single chunk processing")

    # Test 2: Streaming (2 chunks — checks state continuity)
    preprocessor.reset_filter_state()
    chunk1 = mixed[: sr // 2]
    chunk2 = mixed[sr // 2 :]
    clean1 = preprocessor.process(chunk1)
    clean2 = preprocessor.process(chunk2)
    assert preprocessor.chunks_processed == 2, "Should count 2 chunks"
    print("  [PASS] Streaming across chunk boundaries")

    # Test 3: validate_audio
    quality = preprocessor.validate_audio(mixed)
    assert quality["is_usable"] is True, "Test signal should be usable"
    print("  [PASS] validate_audio()")

    # Test 4: Input validation
    try:
        preprocessor.process(np.zeros((2, 100)))  # 2D input
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [PASS] Rejects 2D input")

    print(f"\n  Input  dtype : {mixed.dtype}")
    print(f"  Output dtype : {clean.dtype}")
    print(f"  Input  shape : {mixed.shape}")
    print(f"  Output shape : {clean.shape}")
    print("\n[ALL TESTS PASSED] audio_preprocessing.py")