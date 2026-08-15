"""
THING 5 — Utilities (Test Signal Generators & Helpers)
=======================================================
Generates synthetic ultrasonic signals for:

1. AI 2's training dataset    — thousands of labeled examples
2. Backend 2's attack simulator — live playback on stage
3. Your own testing            — verify DSP pipeline end-to-end

Signal types:
    • FSK  (Frequency Shift Keying)  — two alternating tones
    • OOK  (On-Off Keying)           — single tone toggled on/off
    • Chirp                          — frequency sweep
    • Steady tone                    — constant single frequency
    • White noise                    — random broadband
    • Ambient noise                  — realistic environmental hiss
    • Mixed                          — signal + noise at given SNR

Who uses this:
    AI 2      — call generate_dataset() to create labeled training data
    Backend 2 — use generate_*() to create attack simulator audio
    You       — test every DSP module with known signals
"""

# ─────────────────────────────────────────────────────────────────────────
# SCOPE (Backend 1 audit, point 3)
#
# SignalGenerator is a TEST FIXTURE. It is not the attack simulator —
# Backend 2 owns that (payload encoding, modulation, speaker transmission,
# over-the-air behaviour).
#
# Used for:
#   • deterministic fixtures for the automated test suite
#   • baseline labelled training data for AI 2 (generate_dataset)
#   • driving demo/live_demo.py without hardware
#
# Produces idealised in-memory arrays with NO channel effects: no room
# acoustics, no speaker/mic frequency response, no multipath, no distance
# attenuation. Any claim about real-world performance must come from
# Backend 2's WAV files or physical recordings, never from this module.
#
# Never called on the runtime detection path.
# ─────────────────────────────────────────────────────────────────────────

import os
import json
from typing import Optional

import numpy as np
from scipy.io import wavfile


class SignalGenerator:
    """
    Synthetic ultrasonic signal generator for testing and ML training.

    Quick start:
    ─────────────
        from dsp import SignalGenerator

        gen = SignalGenerator(sample_rate=48000)

        # Generate a single FSK signal
        audio = gen.generate_fsk(duration_sec=2.0, freq_mark=19000, freq_space=20500)

        # Generate a complete labeled dataset for AI 2
        gen.generate_dataset(output_dir="training_data", samples_per_class=500)
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    # ──────────────────────────────────────────────────────────
    # Individual signal generators
    # ──────────────────────────────────────────────────────────

    def generate_fsk(
        self,
        duration_sec: float = 2.0,
        freq_mark: float = 19000.0,
        freq_space: float = 20500.0,
        baud_rate: float = 20.0,
        payload_bits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate FSK (Frequency Shift Keying) ultrasonic signal.

        bit=1 → freq_mark,  bit=0 → freq_space

        Parameters
        ----------
        payload_bits : np.ndarray, optional
            Binary data to encode.  If None, random bits are used.
        """
        n_samples = int(self.sample_rate * duration_sec)
        samples_per_bit = int(self.sample_rate / baud_rate)
        n_bits = n_samples // samples_per_bit

        if payload_bits is None:
            bits = np.random.randint(0, 2, n_bits)
        else:
            bits = payload_bits[:n_bits]
            if len(bits) < n_bits:
                bits = np.tile(bits, (n_bits // len(bits)) + 1)[:n_bits]

        t = np.arange(n_samples) / self.sample_rate
        audio = np.zeros(n_samples, dtype=np.float64)

        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = min((i + 1) * samples_per_bit, n_samples)
            freq = freq_mark if bit == 1 else freq_space
            audio[start:end] = np.sin(2 * np.pi * freq * t[start:end])

        return self._apply_fade(audio)

    def generate_ook(
        self,
        duration_sec: float = 2.0,
        carrier_freq: float = 19000.0,
        baud_rate: float = 20.0,
        payload_bits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate OOK (On-Off Keying) signal.

        bit=1 → carrier tone,  bit=0 → silence
        """
        n_samples = int(self.sample_rate * duration_sec)
        samples_per_bit = int(self.sample_rate / baud_rate)
        n_bits = n_samples // samples_per_bit

        if payload_bits is None:
            bits = np.random.randint(0, 2, n_bits)
        else:
            bits = payload_bits[:n_bits]
            if len(bits) < n_bits:
                bits = np.tile(bits, (n_bits // len(bits)) + 1)[:n_bits]

        t = np.arange(n_samples) / self.sample_rate
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        audio = np.zeros(n_samples, dtype=np.float64)

        for i, bit in enumerate(bits):
            if bit == 1:
                start = i * samples_per_bit
                end = min((i + 1) * samples_per_bit, n_samples)
                audio[start:end] = carrier[start:end]

        return self._apply_fade(audio)

    def generate_chirp(
        self,
        duration_sec: float = 2.0,
        freq_start: float = 18500.0,
        freq_end: float = 21500.0,
        num_sweeps: int = 4,
    ) -> np.ndarray:
        """
        Generate a chirp / frequency sweep signal.

        Repeats `num_sweeps` up-sweeps across the duration.
        """
        n_samples = int(self.sample_rate * duration_sec)
        t = np.arange(n_samples) / self.sample_rate
        sweep_duration = duration_sec / num_sweeps

        audio = np.zeros(n_samples, dtype=np.float64)

        for s in range(num_sweeps):
            start_idx = int(s * self.sample_rate * sweep_duration)
            end_idx = min(int((s + 1) * self.sample_rate * sweep_duration), n_samples)
            t_sweep = t[start_idx:end_idx] - t[start_idx]
            T = sweep_duration

            # Linear chirp: instantaneous freq = f_start + (f_end - f_start) * t / T
            phase = 2 * np.pi * (freq_start * t_sweep + 0.5 * (freq_end - freq_start) * t_sweep**2 / T)
            audio[start_idx:end_idx] = np.sin(phase)

        return self._apply_fade(audio)

    def generate_tone(
        self,
        duration_sec: float = 2.0,
        frequency: float = 19000.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate a steady single-frequency tone."""
        n_samples = int(self.sample_rate * duration_sec)
        t = np.arange(n_samples) / self.sample_rate
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        return self._apply_fade(audio)

    def generate_white_noise(
        self,
        duration_sec: float = 2.0,
        amplitude: float = 0.5,
    ) -> np.ndarray:
        """Generate white noise (broadband, all frequencies)."""
        n_samples = int(self.sample_rate * duration_sec)
        return amplitude * np.random.randn(n_samples)

    def generate_ambient_noise(
        self,
        duration_sec: float = 2.0,
        amplitude: float = 0.3,
    ) -> np.ndarray:
        """
        Generate realistic ambient noise with characteristics typical
        of office / conference environments:
        - Low-frequency rumble (HVAC)
        - Mid-frequency hiss
        - Occasional ultrasonic artefacts from electronics
        """
        n_samples = int(self.sample_rate * duration_sec)
        t = np.arange(n_samples) / self.sample_rate

        # Base: pink-ish noise (1/f falloff)
        white = np.random.randn(n_samples)
        # Simple 1/f approximation via cumulative sum + normalization
        pink = np.cumsum(white)
        pink = pink - np.mean(pink)
        pink = pink / (np.max(np.abs(pink)) + 1e-12) * 0.3

        # Mains hum leakage (50/60 Hz harmonics)
        hum = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.03 * np.sin(2 * np.pi * 100 * t)

        # Random ultrasonic spikes (electronics, fluorescent lights)
        spikes = np.zeros(n_samples)
        n_spikes = np.random.randint(2, 8)
        for _ in range(n_spikes):
            spike_freq = np.random.uniform(17000, 22000)
            spike_start = np.random.randint(0, max(1, n_samples - 2000))
            spike_len = np.random.randint(500, 2000)
            spike_end = min(spike_start + spike_len, n_samples)
            t_spike = t[spike_start:spike_end]
            spike_amp = np.random.uniform(0.01, 0.08)
            spikes[spike_start:spike_end] = spike_amp * np.sin(2 * np.pi * spike_freq * t_spike)

        audio = amplitude * (pink + hum + spikes)
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * amplitude
        return audio

    # ──────────────────────────────────────────────────────────
    # Signal mixing
    # ──────────────────────────────────────────────────────────

    def mix_with_noise(
        self,
        signal: np.ndarray,
        snr_db: float = 10.0,
        noise_type: str = "ambient",
    ) -> np.ndarray:
        """
        Mix a clean signal with noise at a given SNR (dB).

        Parameters
        ----------
        snr_db : float
            Signal-to-Noise Ratio in decibels.
            20 dB = signal 10× louder than noise (easy detection)
            0 dB  = equal power (challenging detection)
           -5 dB  = noise louder (very challenging)
        noise_type : str
            "white" or "ambient"
        """
        duration = len(signal) / self.sample_rate

        if noise_type == "ambient":
            noise = self.generate_ambient_noise(duration_sec=duration, amplitude=1.0)
        else:
            noise = self.generate_white_noise(duration_sec=duration, amplitude=1.0)

        # Trim/pad noise to match signal length
        if len(noise) < len(signal):
            noise = np.pad(noise, (0, len(signal) - len(noise)))
        else:
            noise = noise[: len(signal)]

        # Scale noise to achieve target SNR
        signal_power = np.mean(signal**2) + 1e-12
        noise_power = np.mean(noise**2) + 1e-12
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

        mixed = signal + noise_scaled

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val

        return mixed

    # ──────────────────────────────────────────────────────────
    # Dataset generation (for AI 2)
    # ──────────────────────────────────────────────────────────

    def generate_dataset(
        self,
        output_dir: str = "training_data",
        samples_per_class: int = 500,
        chunk_size: int = 2048,
        snr_range: tuple = (-5, 25),
        chunk_duration_sec: Optional[float] = None,
        n_fft: Optional[int] = None,
        preprocess_fn: Optional[callable] = None,
    ):
        """
        Generate a complete labeled dataset for AI 2's ML classifier.

        Creates:
            output_dir/
            ├── features.npy     — shape (N, 32) float64
            ├── labels.npy       — shape (N,) int  {0=benign, 1=fsk, 2=ook, 3=chirp, 4=tone}
            ├── label_map.json   — {"0": "benign", "1": "fsk", ...}
            └── metadata.json    — generation parameters

        Parameters
        ----------
        samples_per_class : int
            Number of audio chunks per class.  Total samples = 5 × samples_per_class.
        chunk_size : int
            Samples per chunk (should match N_FFT).
        snr_range : tuple
            (min_snr_db, max_snr_db) — SNR is randomized within this range.
        preprocess_fn : callable, optional
            Optional preprocessing function (e.g. AudioPreprocessor.process) to apply
            to each chunk before feature extraction.
        """
        if n_fft is not None:
            chunk_size = n_fft
        elif chunk_duration_sec is not None:
            chunk_size = int(self.sample_rate * chunk_duration_sec)

        try:
            from .feature_extraction import FeatureExtractor
        except ImportError:
            from feature_extraction import FeatureExtractor

        os.makedirs(output_dir, exist_ok=True)

        extractor = FeatureExtractor(
            sample_rate=self.sample_rate,
            n_fft=chunk_size,
        )

        label_map = {
            0: "benign",
            1: "fsk",
            2: "ook",
            3: "chirp",
            4: "tone",
        }

        all_features = []
        all_labels = []

        chunk_duration = chunk_size / self.sample_rate

        for class_idx, class_name in label_map.items():
            print(f"  Generating {samples_per_class} samples for class '{class_name}'...")

            for i in range(samples_per_class):
                snr = np.random.uniform(*snr_range)

                # ── Generate signal based on class ──
                if class_name == "benign":
                    # Pure noise — no structured ultrasonic content
                    signal = self.generate_ambient_noise(
                        duration_sec=chunk_duration, amplitude=0.5
                    )
                elif class_name == "fsk":
                    f_mark = np.random.uniform(18500, 20000)
                    f_space = f_mark + np.random.uniform(500, 2500)
                    baud = np.random.uniform(10, 100)
                    signal = self.generate_fsk(
                        duration_sec=chunk_duration,
                        freq_mark=f_mark,
                        freq_space=min(f_space, 22000),
                        baud_rate=baud,
                    )
                    signal = self.mix_with_noise(signal, snr_db=snr)
                elif class_name == "ook":
                    carrier = np.random.uniform(18500, 21500)
                    baud = np.random.uniform(10, 80)
                    signal = self.generate_ook(
                        duration_sec=chunk_duration,
                        carrier_freq=carrier,
                        baud_rate=baud,
                    )
                    signal = self.mix_with_noise(signal, snr_db=snr)
                elif class_name == "chirp":
                    f_start = np.random.uniform(18000, 19500)
                    f_end = np.random.uniform(20500, 22000)
                    sweeps = np.random.randint(1, 6)
                    signal = self.generate_chirp(
                        duration_sec=chunk_duration,
                        freq_start=f_start,
                        freq_end=f_end,
                        num_sweeps=sweeps,
                    )
                    signal = self.mix_with_noise(signal, snr_db=snr)
                elif class_name == "tone":
                    freq = np.random.uniform(18500, 21500)
                    signal = self.generate_tone(
                        duration_sec=chunk_duration,
                        frequency=freq,
                    )
                    signal = self.mix_with_noise(signal, snr_db=snr)

                # Trim to exact chunk_size
                chunk = signal[:chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                # Apply optional preprocessing (matches runtime pipeline)
                if preprocess_fn is not None:
                    chunk = preprocess_fn(chunk)

                # Extract features
                result = extractor.extract(chunk)
                all_features.append(result["feature_values"])
                all_labels.append(class_idx)

        # ── Save dataset ──
        features_array = np.array(all_features, dtype=np.float64)
        labels_array = np.array(all_labels, dtype=np.int32)

        # Shuffle
        indices = np.random.permutation(len(features_array))
        features_array = features_array[indices]
        labels_array = labels_array[indices]

        np.save(os.path.join(output_dir, "features.npy"), features_array)
        np.save(os.path.join(output_dir, "labels.npy"), labels_array)

        with open(os.path.join(output_dir, "label_map.json"), "w") as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

        metadata = {
            "total_samples": len(features_array),
            "samples_per_class": samples_per_class,
            "num_classes": len(label_map),
            "num_features": 32,
            "feature_names": extractor.get_feature_names(),
            "sample_rate": self.sample_rate,
            "chunk_size": chunk_size,
            "snr_range_db": list(snr_range),
            "label_map": {str(k): v for k, v in label_map.items()},
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n  Dataset saved to '{output_dir}/'")
        print(f"    features.npy : {features_array.shape}")
        print(f"    labels.npy   : {labels_array.shape}")
        print(f"    Classes      : {label_map}")

        return features_array, labels_array

    # ──────────────────────────────────────────────────────────
    # WAV file I/O
    # ──────────────────────────────────────────────────────────

    def save_wav(self, audio: np.ndarray, filepath: str):
        """Save audio as a 16-bit WAV file."""
        # Normalize to int16 range
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio_norm = audio / peak
        else:
            audio_norm = audio
        audio_int16 = np.int16(audio_norm * 32767)
        wavfile.write(filepath, self.sample_rate, audio_int16)

    def load_wav(self, filepath: str) -> np.ndarray:
        """Load a WAV file and return float64 samples normalized to [-1, 1]."""
        sr, data = wavfile.read(filepath)
        if sr != self.sample_rate:
            print(f"WARNING: File sample rate {sr} != expected {self.sample_rate}")
        if data.dtype == np.int16:
            return data.astype(np.float64) / 32767.0
        elif data.dtype == np.float32:
            return data.astype(np.float64)
        return data.astype(np.float64)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _apply_fade(self, audio: np.ndarray, fade_ms: float = 5.0) -> np.ndarray:
        """Apply fade-in and fade-out to prevent clicks."""
        fade_samples = int(self.sample_rate * fade_ms / 1000.0)
        fade_samples = min(fade_samples, len(audio) // 4)

        if fade_samples > 0:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        return audio


if __name__ == "__main__":
    print("==================================================")
    print("  Testing Signal Generator Utilities...")
    print("==================================================")

    gen = SignalGenerator(sample_rate=48000)

    # Test 1: Signal generation
    fsk = gen.generate_fsk(duration_sec=1.0)
    assert len(fsk) == 48000 and fsk.ndim == 1 and np.all(np.isfinite(fsk))
    print("  [PASS] FSK signal generation")

    ook = gen.generate_ook(duration_sec=1.0)
    assert len(ook) == 48000 and np.all(np.isfinite(ook))
    print("  [PASS] OOK signal generation")

    chirp = gen.generate_chirp(duration_sec=1.0)
    assert len(chirp) == 48000 and np.all(np.isfinite(chirp))
    print("  [PASS] Chirp signal generation")

    tone = gen.generate_tone(duration_sec=1.0, frequency=19000)
    assert len(tone) == 48000 and np.all(np.isfinite(tone))
    print("  [PASS] Tone signal generation")

    print("\n[ALL TESTS PASSED] utils.py")
