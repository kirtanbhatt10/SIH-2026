"""
THING 2 — Feature Extractor (The Feature Calculator)
=====================================================
Extracts exactly 32 numeric features from each audio chunk.
These features are the ONLY input AI 2's ML classifier ever sees.

Who uses this:
    AI 2 — THIS IS YOUR MOST IMPORTANT FILE.
           Call extract() → get a dict of 32 floats → feed to your model.
    Backend 1 — features are included in the API response via dsp_api.py.

Feature Groups (32 total):
    Energy        (4)  — How much ultrasonic power is present
    Spectral Shape(6)  — Shape of the frequency distribution
    Peak          (6)  — Dominant tones and their arrangement
    Tonal         (4)  — Is it structured sound or random noise?
    Temporal      (6)  — How the signal changes within the chunk
    Statistical   (6)  — Basic distribution stats on magnitudes

Output Format:
    {
        "feature_names": ["ultrasonic_energy", "total_energy", ...],   # 32 strings
        "feature_values": [0.854, 1.120, ...],                         # 32 floats
        "feature_dict": {"ultrasonic_energy": 0.854, ...}              # keyed dict
    }
"""

import numpy as np
from scipy.signal import find_peaks, peak_prominences


class FeatureExtractor:
    """
    Extracts 32 spectral/temporal features from a preprocessed audio chunk.

    Usage (AI 2 — this is your main interface):
    ────────────────────────────────────────────
        from dsp import FeatureExtractor

        extractor = FeatureExtractor(sample_rate=48000, n_fft=2048)

        # raw_chunk or filtered_chunk — both work, but filtered is better
        result = extractor.extract(audio_chunk)

        feature_vector = result["feature_values"]   # list of 32 floats
        feature_names  = result["feature_names"]     # list of 32 strings
        feature_dict   = result["feature_dict"]      # {"name": value, ...}

    Training tip for AI 2:
        np.array(result["feature_values"])  →  shape (32,)  →  feed to sklearn / torch
    """

    # ── Canonical feature order (DO NOT CHANGE without telling AI 2) ──
    FEATURE_NAMES = [
        # Energy (4)
        "ultrasonic_energy",
        "total_energy",
        "energy_ratio",
        "energy_db",
        # Spectral Shape (6)
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_flatness",
        "spectral_rolloff",
        "spectral_skewness",
        "spectral_kurtosis",
        # Peak (6)
        "peak_frequency",
        "peak_magnitude",
        "peak_prominence",
        "num_peaks",
        "peak_spacing_mean",
        "peak_spacing_std",
        # Tonal (4)
        "tonal_prominence",
        "harmonic_ratio",
        "crest_factor",
        "zero_crossing_rate",
        # Temporal (6)
        "rms_amplitude",
        "amplitude_envelope_std",
        "onset_strength",
        "temporal_flatness",
        "duty_cycle",
        "bit_rate_estimate",
        # Statistical (6)
        "mean_magnitude",
        "std_magnitude",
        "max_magnitude",
        "min_magnitude",
        "magnitude_range",
        "coefficient_of_variation",
    ]

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        ultrasonic_low: float = 18000.0,
        ultrasonic_high: float = 22000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft

        # Pre-compute frequency bin array (only once)
        self.freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        # Boolean masks for the ultrasonic band
        self.us_mask = (self.freq_bins >= ultrasonic_low) & (self.freq_bins <= ultrasonic_high)
        self.us_indices = np.where(self.us_mask)[0]
        self.us_freqs = self.freq_bins[self.us_mask]

    # ──────────────────────────────────────────────────────────
    # Main public method
    # ──────────────────────────────────────────────────────────

    def extract(self, audio_chunk: np.ndarray) -> dict:
        """
        Extract all 32 features from one audio chunk.

        Parameters
        ----------
        audio_chunk : np.ndarray, shape (N,)
            Audio samples (filtered or raw).  Length should equal n_fft
            for best frequency resolution, but any length works.

        Returns
        -------
        dict with keys:
            "feature_names"  : list[str]   — 32 canonical names
            "feature_values" : list[float] — 32 values in the same order
            "feature_dict"   : dict        — {name: value} for convenience
        """
        audio = np.asarray(audio_chunk, dtype=np.float64)

        # ── FFT ──
        windowed = audio * np.hanning(len(audio))
        fft_complex = np.fft.rfft(windowed, n=self.n_fft)
        magnitude = np.abs(fft_complex)
        power = magnitude ** 2

        # Ultrasonic-band slices
        us_mag = magnitude[self.us_mask]
        us_pow = power[self.us_mask]

        # ── Compute each group ──
        energy_feats = self._energy_features(power, us_pow)
        shape_feats = self._spectral_shape_features(magnitude, power, us_mag, us_pow)
        peak_feats = self._peak_features(us_mag)
        tonal_feats = self._tonal_features(audio, us_mag, us_pow)
        temporal_feats = self._temporal_features(audio)
        stat_feats = self._statistical_features(us_mag)

        # ── Assemble in canonical order ──
        values = (
            energy_feats
            + shape_feats
            + peak_feats
            + tonal_feats
            + temporal_feats
            + stat_feats
        )

        # Safety: replace any NaN / Inf with 0.0
        values = [0.0 if not np.isfinite(v) else float(v) for v in values]

        assert len(values) == 32, f"Expected 32 features, got {len(values)}"

        return {
            "feature_names": list(self.FEATURE_NAMES),
            "feature_values": values,
            "feature_dict": dict(zip(self.FEATURE_NAMES, values)),
        }

    # ──────────────────────────────────────────────────────────
    # Feature group implementations
    # ──────────────────────────────────────────────────────────

    def _energy_features(self, power: np.ndarray, us_pow: np.ndarray) -> list:
        """4 features: ultrasonic_energy, total_energy, energy_ratio, energy_db."""
        total_energy = float(np.sum(power)) + 1e-12
        us_energy = float(np.sum(us_pow)) + 1e-12
        energy_ratio = us_energy / total_energy
        energy_db = 10.0 * np.log10(us_energy + 1e-12)

        return [us_energy, total_energy, energy_ratio, energy_db]

    def _spectral_shape_features(
        self,
        magnitude: np.ndarray,
        power: np.ndarray,
        us_mag: np.ndarray,
        us_pow: np.ndarray,
    ) -> list:
        """6 features describing the shape of the ultrasonic spectrum."""
        eps = 1e-12

        # Spectral centroid (center of mass in Hz)
        us_sum = np.sum(us_mag) + eps
        centroid = float(np.sum(self.us_freqs * us_mag) / us_sum)

        # Spectral bandwidth (std around centroid)
        bandwidth = float(np.sqrt(np.sum(us_mag * (self.us_freqs - centroid) ** 2) / us_sum))

        # Spectral flatness (Wiener entropy) — low = tonal, high = noise
        if len(us_pow) > 0 and np.all(us_pow > 0):
            log_mean = np.exp(np.mean(np.log(us_pow + eps)))
            arith_mean = np.mean(us_pow) + eps
            flatness = float(log_mean / arith_mean)
        else:
            flatness = 1.0

        # Spectral rolloff — frequency below which 85% of energy lies
        cumulative = np.cumsum(us_pow)
        total_us = cumulative[-1] if len(cumulative) > 0 else eps
        rolloff_idx = np.searchsorted(cumulative, 0.85 * total_us)
        rolloff_idx = min(rolloff_idx, len(self.us_freqs) - 1)
        rolloff = float(self.us_freqs[rolloff_idx]) if len(self.us_freqs) > 0 else 0.0

        # Spectral skewness
        if bandwidth > eps:
            skewness = float(
                np.sum(us_mag * ((self.us_freqs - centroid) / (bandwidth + eps)) ** 3) / (us_sum + eps)
            )
        else:
            skewness = 0.0

        # Spectral kurtosis
        if bandwidth > eps:
            kurtosis = float(
                np.sum(us_mag * ((self.us_freqs - centroid) / (bandwidth + eps)) ** 4) / (us_sum + eps)
            )
        else:
            kurtosis = 0.0

        return [centroid, bandwidth, flatness, rolloff, skewness, kurtosis]

    def _peak_features(self, us_mag: np.ndarray) -> list:
        """6 features about dominant frequency peaks (key for FSK detection)."""
        if len(us_mag) < 3:
            return [0.0, 0.0, 0.0, 0, 0.0, 0.0]

        # Find peaks with minimum prominence
        peak_indices, properties = find_peaks(
            us_mag,
            height=np.max(us_mag) * 0.1,  # at least 10% of max
            prominence=np.max(us_mag) * 0.05,
            distance=3,  # at least ~70 Hz apart
        )

        num_peaks = len(peak_indices)

        if num_peaks == 0:
            # No clear peaks — use global max
            peak_idx = int(np.argmax(us_mag))
            peak_freq = float(self.us_freqs[peak_idx]) if len(self.us_freqs) > 0 else 0.0
            peak_mag = float(us_mag[peak_idx])
            return [peak_freq, peak_mag, 0.0, 0, 0.0, 0.0]

        # Dominant peak
        sorted_by_height = sorted(peak_indices, key=lambda i: us_mag[i], reverse=True)
        top_idx = sorted_by_height[0]
        peak_freq = float(self.us_freqs[top_idx])
        peak_mag = float(us_mag[top_idx])

        # Peak prominence (how far above the local baseline)
        prominences = properties.get("prominences", np.array([0.0]))
        peak_prominence = float(np.max(prominences))

        # Peak spacing (important: regular spacing → FSK / data encoding)
        if num_peaks >= 2:
            peak_freqs_sorted = np.sort(self.us_freqs[peak_indices])
            spacings = np.diff(peak_freqs_sorted)
            spacing_mean = float(np.mean(spacings))
            spacing_std = float(np.std(spacings))
        else:
            spacing_mean = 0.0
            spacing_std = 0.0

        return [peak_freq, peak_mag, peak_prominence, num_peaks, spacing_mean, spacing_std]

    def _tonal_features(
        self, audio: np.ndarray, us_mag: np.ndarray, us_pow: np.ndarray
    ) -> list:
        """4 features measuring how 'structured' vs 'noisy' the signal is."""
        eps = 1e-12

        # Tonal prominence — ratio of peak to mean (high = single strong tone)
        if len(us_pow) > 1:
            peak_val = np.max(us_pow)
            mean_bg = (np.sum(us_pow) - peak_val) / max(1, len(us_pow) - 1)
            tonal_prom = float(peak_val / (mean_bg + eps))
        else:
            tonal_prom = 0.0

        # Harmonic ratio — energy in top 3 peaks / total ultrasonic energy
        if len(us_mag) >= 3:
            sorted_mags = np.sort(us_mag)[::-1]
            top3_energy = float(np.sum(sorted_mags[:3] ** 2))
            total_us_energy = float(np.sum(us_pow)) + eps
            harmonic_ratio = top3_energy / total_us_energy
        else:
            harmonic_ratio = 0.0

        # Crest factor — peak / RMS (high for impulsive / structured signals)
        rms = np.sqrt(np.mean(audio ** 2)) + eps
        peak_amp = np.max(np.abs(audio)) + eps
        crest_factor = float(peak_amp / rms)

        # Zero-crossing rate (higher for ultrasonic, useful as sanity check)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))) > 0)
        zcr = float(zero_crossings / (len(audio) + eps))

        return [tonal_prom, harmonic_ratio, crest_factor, zcr]

    def _temporal_features(self, audio: np.ndarray) -> list:
        """6 features about how the signal varies over time within the chunk."""
        eps = 1e-12

        # RMS amplitude
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Amplitude envelope (using short sub-frames)
        sub_frame_len = max(64, len(audio) // 16)
        n_sub = max(1, len(audio) // sub_frame_len)
        envelope = np.array([
            np.sqrt(np.mean(audio[i * sub_frame_len : (i + 1) * sub_frame_len] ** 2))
            for i in range(n_sub)
        ])
        envelope_std = float(np.std(envelope))

        # Onset strength — max frame-to-frame energy jump
        if len(envelope) >= 2:
            diffs = np.abs(np.diff(envelope))
            onset_strength = float(np.max(diffs))
        else:
            onset_strength = 0.0

        # Temporal flatness — how consistent is the energy over time
        if len(envelope) > 0 and np.all(envelope > 0):
            t_geo = np.exp(np.mean(np.log(envelope + eps)))
            t_arith = np.mean(envelope) + eps
            temporal_flat = float(t_geo / t_arith)
        else:
            temporal_flat = 1.0

        # Duty cycle — fraction of sub-frames where energy > 20% of max
        if len(envelope) > 0:
            threshold = 0.2 * np.max(envelope)
            duty_cycle = float(np.sum(envelope > threshold) / len(envelope))
        else:
            duty_cycle = 0.0

        # Bit rate estimate — count on/off transitions in envelope
        if len(envelope) >= 2:
            binary = (envelope > 0.2 * np.max(envelope)).astype(int)
            transitions = int(np.sum(np.abs(np.diff(binary))))
            chunk_duration = len(audio) / (self.sample_rate + eps)
            bit_rate_est = float(transitions / (2.0 * chunk_duration + eps))
        else:
            bit_rate_est = 0.0

        return [rms, envelope_std, onset_strength, temporal_flat, duty_cycle, bit_rate_est]

    def _statistical_features(self, us_mag: np.ndarray) -> list:
        """6 basic distribution statistics on ultrasonic magnitudes."""
        eps = 1e-12

        if len(us_mag) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        mean_mag = float(np.mean(us_mag))
        std_mag = float(np.std(us_mag))
        max_mag = float(np.max(us_mag))
        min_mag = float(np.min(us_mag))
        mag_range = max_mag - min_mag
        coeff_var = std_mag / (mean_mag + eps)

        return [mean_mag, std_mag, max_mag, min_mag, mag_range, coeff_var]

    # ──────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────

    @classmethod
    def get_feature_names(cls) -> list:
        """Return the canonical list of 32 feature names (for column headers, etc.)."""
        return list(cls.FEATURE_NAMES)

    @classmethod
    def num_features(cls) -> int:
        """Always returns 32."""
        return len(cls.FEATURE_NAMES)


if __name__ == "__main__":
    print("==================================================")
    print("  Testing Feature Extractor...")
    print("==================================================")

    sr = 48000
    n_fft = 2048
    extractor = FeatureExtractor(sample_rate=sr, n_fft=n_fft)

    # Test 1: Feature counts
    names = FeatureExtractor.get_feature_names()
    assert len(names) == 32, f"Expected 32 feature names, got {len(names)}"
    assert FeatureExtractor.num_features() == 32
    print("  [PASS] Canonical 32 feature names defined")

    # Test 2: Extraction on synthetic signal
    t = np.linspace(0, 1, sr)
    fsk = np.zeros(sr)
    for i in range(20):
        freq = 19500 if i % 2 else 18500
        start = i * 2400
        fsk[start:start+2400] = np.sin(2 * np.pi * freq * t[start:start+2400])

    result = extractor.extract(fsk[:n_fft])
    assert len(result["feature_names"]) == 32, "Feature names count mismatch"
    assert len(result["feature_values"]) == 32, "Feature values count mismatch"
    assert len(result["feature_dict"]) == 32, "Feature dict count mismatch"
    assert all(np.isfinite(v) for v in result["feature_values"]), "Non-finite feature value detected"
    print("  [PASS] Extracted 32 finite features successfully")

    print("\n[ALL TESTS PASSED] feature_extraction.py")
