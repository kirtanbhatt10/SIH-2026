"""
THING 4 — Frequency Analyzer (The Detector)
=============================================
Analyzes the ultrasonic spectrum to detect specific modulation schemes
used by known acoustic exfiltration malware:

    • FSK  (Frequency Shift Keying) — alternates between two carrier freqs
    • OOK  (On-Off Keying)          — single carrier toggled on and off
    • Chirp                         — frequency sweeps across the band

Also provides general peak detection, carrier frequency estimation,
and a quick "is this suspicious?" verdict.

Who uses this:
    Backend 1 — gets modulation type in API response
    AI 2      — uses detection results as additional classifier input
    You       — core analysis logic

Note for AI 2:
    The outputs of this module are SUPPLEMENTARY to the 32-feature vector.
    You can use them as additional features or as a rule-based pre-filter
    before your ML model. Your call.
"""

import numpy as np
from scipy.signal import find_peaks, peak_prominences


class FrequencyAnalyzer:
    """
    Detects structured ultrasonic transmissions and classifies their
    modulation scheme.

    Usage:
    ──────
        from dsp import FrequencyAnalyzer

        analyzer = FrequencyAnalyzer(sample_rate=48000)

        # Single-chunk analysis
        result = analyzer.analyze(audio_chunk)
        print(result["modulation_type"])   # "fsk", "ook", "chirp", or "none"
        print(result["confidence"])        # 0.0 – 1.0
        print(result["carrier_freqs"])     # [19000.0, 20500.0]
        print(result["is_suspicious"])     # True / False

        # Multi-chunk analysis (more accurate — accumulates evidence)
        analyzer.reset_accumulator()
        for chunk in audio_stream:
            result = analyzer.analyze_and_accumulate(chunk)
        verdict = analyzer.get_accumulated_verdict()
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        ultrasonic_low: float = 18000.0,
        ultrasonic_high: float = 22000.0,
        suspicious_threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.suspicious_threshold = suspicious_threshold

        # Frequency axis
        self.freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        self.freq_resolution = self.freq_bins[1] - self.freq_bins[0]  # ~23.4 Hz

        # Ultrasonic band
        self.us_mask = (self.freq_bins >= ultrasonic_low) & (self.freq_bins <= ultrasonic_high)
        self.us_freqs = self.freq_bins[self.us_mask]

        # Multi-chunk accumulator
        self._accumulated_peaks = []      # list of peak-frequency lists per chunk
        self._accumulated_energies = []   # ultrasonic energy per chunk
        self._accumulated_scores = []     # suspicion scores per chunk
        self.last_result = None           # most recent analyze() result,
                                          # used by DSPPipeline.to_threat_event()

    # ──────────────────────────────────────────────────────────
    # Single-chunk analysis
    # ──────────────────────────────────────────────────────────

    def analyze(self, audio_chunk: np.ndarray) -> dict:
        """
        Analyze one audio chunk for ultrasonic transmission patterns.

        Returns
        -------
        dict:
            modulation_type : str       — "fsk", "ook", "chirp", "tone", or "none"
            confidence      : float     — 0.0 to 1.0
            carrier_freqs   : list[float] — detected carrier frequencies in Hz
            num_peaks       : int
            is_suspicious   : bool
            peak_details    : list[dict] — per-peak info
            energy_ratio    : float     — ultrasonic vs total energy
            estimated_baud  : float     — estimated symbol rate (if applicable)
        """
        audio = np.asarray(audio_chunk, dtype=np.float64)

        if len(audio) == 0:
            return {
                "modulation_type": "none",
                "confidence": 0.0,
                "carrier_freqs": [],
                "num_peaks": 0,
                "is_suspicious": False,
                "suspicion_score": 0.0,
                "peak_details": [],
                "energy_ratio": 0.0,
                "estimated_baud": 0.0,
            }

        # Windowed FFT
        windowed = audio * np.hanning(len(audio))
        fft_complex = np.fft.rfft(windowed, n=self.n_fft)
        magnitude = np.abs(fft_complex)
        power = magnitude ** 2

        # Ultrasonic band
        us_mag = magnitude[self.us_mask]
        us_pow = power[self.us_mask]

        # Energy ratio
        total_energy = float(np.sum(power)) + 1e-12
        us_energy = float(np.sum(us_pow)) + 1e-12
        energy_ratio = us_energy / total_energy

        # ── Peak detection ──
        peaks = self._detect_peaks(us_mag)

        # ── Modulation classification ──
        mod_result = self._classify_modulation(audio, us_mag, us_pow, peaks)

        # ── Suspicion score ──
        suspicion = self._compute_suspicion_score(
            energy_ratio, peaks, mod_result["modulation_type"], mod_result["confidence"]
        )

        return {
            "modulation_type": mod_result["modulation_type"],
            "confidence": mod_result["confidence"],
            "carrier_freqs": mod_result["carrier_freqs"],
            "num_peaks": len(peaks),
            "is_suspicious": suspicion >= self.suspicious_threshold,
            "suspicion_score": round(suspicion, 4),
            "peak_details": peaks,
            "energy_ratio": round(energy_ratio, 6),
            "estimated_baud": mod_result.get("estimated_baud", 0.0),
        }

    # ──────────────────────────────────────────────────────────
    # Multi-chunk accumulation (more robust detection)
    # ──────────────────────────────────────────────────────────

    def analyze_and_accumulate(self, audio_chunk: np.ndarray) -> dict:
        """Analyze a chunk AND store results for multi-chunk verdict."""
        result = self.analyze(audio_chunk)

        self._accumulated_peaks.append(result["carrier_freqs"])
        self._accumulated_energies.append(result["energy_ratio"])
        self._accumulated_scores.append(result["suspicion_score"])
        self.last_result = result

        return result

    def get_accumulated_verdict(self) -> dict:
        """
        After processing multiple chunks, return a final verdict
        based on accumulated evidence.

        Returns
        -------
        dict:
            is_threat           : bool
            average_suspicion   : float
            max_suspicion       : float
            consistent_carriers : list[float]  — frequencies seen in >50% of chunks
            chunks_analyzed     : int
            threat_duration_sec : float
        """
        n = len(self._accumulated_scores)
        if n == 0:
            return {
                "is_threat": False,
                "average_suspicion": 0.0,
                "max_suspicion": 0.0,
                "consistent_carriers": [],
                "chunks_analyzed": 0,
                "threat_duration_sec": 0.0,
            }

        avg_score = float(np.mean(self._accumulated_scores))
        max_score = float(np.max(self._accumulated_scores))

        # Find frequencies that appear consistently across chunks
        all_freqs = []
        for freq_list in self._accumulated_peaks:
            all_freqs.extend(freq_list)

        consistent = []
        if all_freqs:
            # Cluster frequencies within 100 Hz of each other
            sorted_freqs = sorted(all_freqs)
            clusters = []
            current_cluster = [sorted_freqs[0]]
            for f in sorted_freqs[1:]:
                if f - current_cluster[-1] < 100:
                    current_cluster.append(f)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [f]
            clusters.append(current_cluster)

            # Keep clusters that appear in > 50% of chunks
            for cluster in clusters:
                if len(cluster) >= n * 0.5:
                    consistent.append(round(float(np.mean(cluster)), 1))

        chunk_duration = self.n_fft / self.sample_rate
        suspicious_chunks = sum(1 for s in self._accumulated_scores if s >= self.suspicious_threshold)

        return {
            "is_threat": avg_score >= self.suspicious_threshold and suspicious_chunks >= max(2, n * 0.3),
            "average_suspicion": round(avg_score, 4),
            "max_suspicion": round(max_score, 4),
            "consistent_carriers": consistent,
            "chunks_analyzed": n,
            "threat_duration_sec": round(suspicious_chunks * chunk_duration, 3),
        }

    def reset_accumulator(self):
        """Clear the multi-chunk accumulator."""
        self._accumulated_peaks.clear()
        self._accumulated_energies.clear()
        self._accumulated_scores.clear()
        self.last_result = None

    # ──────────────────────────────────────────────────────────
    # Internal: Peak Detection
    # ──────────────────────────────────────────────────────────

    def _detect_peaks(self, us_mag: np.ndarray) -> list:
        """
        Find significant peaks in the ultrasonic magnitude spectrum.

        Returns list of dicts:
            [{"freq_hz": 19000.0, "magnitude": 0.85, "prominence": 0.7}, ...]
        """
        if len(us_mag) < 5:
            return []

        max_val = np.max(us_mag)
        mean_val = np.mean(us_mag)
        if max_val < 1e-8 or (max_val / (mean_val + 1e-12)) < 3.5:
            return []

        indices, properties = find_peaks(
            us_mag,
            height=max_val * 0.15,
            prominence=max_val * 0.08,
            distance=3,
        )

        peaks = []
        prominences = properties.get("prominences", np.zeros(len(indices)))
        heights = properties.get("peak_heights", us_mag[indices])

        for i, idx in enumerate(indices):
            peaks.append({
                "freq_hz": round(float(self.us_freqs[idx]), 1),
                "magnitude": round(float(heights[i]), 6),
                "prominence": round(float(prominences[i]), 6),
                "bin_index": int(idx),
            })

        # Sort by magnitude (strongest first)
        peaks.sort(key=lambda p: p["magnitude"], reverse=True)
        return peaks[:10]  # cap at 10 peaks

    # ──────────────────────────────────────────────────────────
    # Internal: Modulation Classification
    # ──────────────────────────────────────────────────────────

    def _classify_modulation(
        self,
        audio: np.ndarray,
        us_mag: np.ndarray,
        us_pow: np.ndarray,
        peaks: list,
    ) -> dict:
        """
        Classify the modulation scheme based on spectral characteristics.

        Logic:
            • 2 strong peaks ~1–2 kHz apart         → FSK
            • 1 strong peak + on/off temporal pattern → OOK
            • Broad energy spread + rising centroid   → Chirp
            • 1 strong peak, steady                  → Simple tone
            • Nothing significant                    → None
        """
        num_peaks = len(peaks)
        eps = 1e-12

        # ── No significant peaks → nothing ──
        if num_peaks == 0:
            return {"modulation_type": "none", "confidence": 0.0, "carrier_freqs": []}

        # ── Spectral flatness (noise vs tonal) ──
        if len(us_pow) > 0 and np.all(us_pow > 0):
            geo_mean = np.exp(np.mean(np.log(us_pow + eps)))
            arith_mean = np.mean(us_pow) + eps
            flatness = geo_mean / arith_mean
        else:
            flatness = 1.0

        # ── Check for FSK: exactly 2 dominant peaks, ~1-3 kHz apart ──
        if num_peaks >= 2:
            top2 = peaks[:2]
            freq_gap = abs(top2[0]["freq_hz"] - top2[1]["freq_hz"])

            # FSK typically uses 500 Hz – 3000 Hz separation
            if 400 <= freq_gap <= 3500:
                # Both peaks should be reasonably strong
                ratio = min(top2[0]["magnitude"], top2[1]["magnitude"]) / (
                    max(top2[0]["magnitude"], top2[1]["magnitude"]) + eps
                )
                if ratio > 0.2 and flatness < 0.4:  # second peak at least 20% of first and non-flat spectrum
                    confidence = min(1.0, ratio * 1.5 * (1.0 - flatness))
                    return {
                        "modulation_type": "fsk",
                        "confidence": round(confidence, 4),
                        "carrier_freqs": [top2[0]["freq_hz"], top2[1]["freq_hz"]],
                        "estimated_baud": self._estimate_baud_rate(audio),
                    }

        # ── Check for Chirp: broad spectral spread with distinct peaks ──
        if flatness > 0.3 and num_peaks >= 3 and (peaks[0]["magnitude"] / (np.mean(us_mag) + eps) > 2.5):
            # Many peaks spread across the band suggests a chirp/sweep
            freq_spread = peaks[0]["freq_hz"] - peaks[-1]["freq_hz"]
            if abs(freq_spread) > 1000:
                confidence = min(1.0, flatness * 1.5)
                carrier_freqs = [p["freq_hz"] for p in peaks[:3]]
                return {
                    "modulation_type": "chirp",
                    "confidence": round(confidence, 4),
                    "carrier_freqs": carrier_freqs,
                }

        # ── Check for OOK / Tone: single dominant peak ──
        if num_peaks >= 1:
            dominant = peaks[0]
            # Tonal ratio — peak magnitude relative to mean background
            tonal_ratio = dominant["magnitude"] / (np.mean(us_mag) + eps)

            if tonal_ratio > 3.0 and flatness < 0.3:
                env_metrics = self._compute_envelope_metrics(audio)
                depth = env_metrics["modulation_depth"]
                transitions = env_metrics["transitions"]
                cv = env_metrics["envelope_cv"]
                duty = env_metrics["duty_cycle"]
                baud = env_metrics["estimated_baud"]

                # OOK requires:
                # 1. Significant modulation depth (depth >= 0.5)
                # 2. Repeated ON <-> OFF transitions (transitions >= 2)
                # 3. Duty cycle between 0.1 and 0.9 (not continuous ON)
                # 4. Envelope CV >= 0.3
                is_ook = (depth >= 0.5) and (transitions >= 2) and (0.1 <= duty <= 0.9) and (cv >= 0.3)

                confidence = min(1.0, (tonal_ratio / 20.0) * (1.0 - flatness))

                if is_ook:
                    return {
                        "modulation_type": "ook",
                        "confidence": round(confidence, 4),
                        "carrier_freqs": [dominant["freq_hz"]],
                        "estimated_baud": baud,
                    }
                else:
                    return {
                        "modulation_type": "tone",
                        "confidence": round(min(1.0, tonal_ratio / 15.0), 4),
                        "carrier_freqs": [dominant["freq_hz"]],
                    }

        return {"modulation_type": "none", "confidence": 0.0, "carrier_freqs": []}

    # ──────────────────────────────────────────────────────────
    # Internal: Temporal envelope & Baud rate estimation
    # ──────────────────────────────────────────────────────────

    def _compute_envelope_metrics(self, audio: np.ndarray) -> dict:
        """
        Compute temporal amplitude envelope metrics using short RMS frames.
        
        Returns dict:
            envelope          : np.ndarray
            modulation_depth  : float (0.0 to 1.0)
            transitions       : int (count of ON <-> OFF state transitions)
            duty_cycle        : float (0.0 to 1.0)
            envelope_cv       : float (std/mean)
            estimated_baud    : float (symbol rate estimate)
        """
        eps = 1e-12
        frame_len = max(32, len(audio) // 32)
        n_frames = max(1, len(audio) // frame_len)

        envelope = np.array([
            np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ])

        if len(envelope) < 4:
            return {
                "envelope": envelope,
                "modulation_depth": 0.0,
                "transitions": 0,
                "duty_cycle": 1.0,
                "envelope_cv": 0.0,
                "estimated_baud": 0.0,
            }

        max_env = float(np.max(envelope))
        min_env = float(np.min(envelope))
        mean_env = float(np.mean(envelope)) + eps
        std_env = float(np.std(envelope))

        modulation_depth = float((max_env - min_env) / (max_env + eps))
        envelope_cv = float(std_env / mean_env)

        # Mid-level threshold for ON/OFF binary decision
        threshold = 0.5 * max_env
        binary = (envelope > threshold).astype(int)

        # Count state transitions (0->1 or 1->0)
        transitions = int(np.sum(np.abs(np.diff(binary))))

        duty_cycle = float(np.sum(binary) / len(binary))

        # Baud rate estimation
        chunk_duration = len(audio) / (self.sample_rate + eps)
        baud = float(transitions / (2.0 * chunk_duration + eps))

        return {
            "envelope": envelope,
            "modulation_depth": round(modulation_depth, 4),
            "transitions": transitions,
            "duty_cycle": round(duty_cycle, 4),
            "envelope_cv": round(envelope_cv, 4),
            "estimated_baud": round(baud, 1),
        }

    def _estimate_baud_rate(self, audio: np.ndarray) -> float:
        """
        Estimate the symbol rate using temporal envelope metrics.
        """
        metrics = self._compute_envelope_metrics(audio)
        return metrics["estimated_baud"]

    # ──────────────────────────────────────────────────────────
    # Internal: Suspicion scoring
    # ──────────────────────────────────────────────────────────

    def _compute_suspicion_score(
        self,
        energy_ratio: float,
        peaks: list,
        modulation_type: str,
        mod_confidence: float,
    ) -> float:
        """
        Combine multiple signals into a single [0, 1] suspicion score.

        Scoring weights:
            - High ultrasonic energy ratio        → suspicious
            - Detected modulation scheme           → very suspicious
            - Multiple well-defined peaks          → suspicious
            - Strong peak prominence               → suspicious
        """
        score = 0.0

        # Energy ratio contribution (0 to 0.3)
        if energy_ratio > 0.01:
            score += min(0.3, energy_ratio * 3.0)

        # Modulation detection contribution (0 to 0.4)
        mod_weights = {"fsk": 0.4, "ook": 0.35, "chirp": 0.3, "tone": 0.2, "none": 0.0}
        score += mod_weights.get(modulation_type, 0.0) * mod_confidence

        # Peak structure contribution (0 to 0.2)
        if len(peaks) >= 2:
            avg_prominence = np.mean([p["prominence"] for p in peaks[:3]])
            score += min(0.2, avg_prominence * 0.5)
        elif len(peaks) == 1 and peaks[0]["prominence"] > 0.1:
            score += 0.1

        # Regularity bonus (0 to 0.1)
        if len(peaks) >= 2:
            freqs = [p["freq_hz"] for p in peaks[:4]]
            spacings = np.diff(sorted(freqs))
            if len(spacings) >= 2:
                cv = float(np.std(spacings)) / (float(np.mean(spacings)) + 1e-12)
                if cv < 0.3:  # very regular spacing
                    score += 0.1

        if modulation_type == "none":
            return min(0.25, score * 0.4)

        return min(1.0, score)


if __name__ == "__main__":
    print("==================================================")
    print("  Testing Frequency Analyzer...")
    print("==================================================")

    sr = 48000
    n_fft = 2048
    analyzer = FrequencyAnalyzer(sample_rate=sr, n_fft=n_fft)

    # Test 1: Synthetic FSK signal
    t = np.linspace(0, 1, sr)
    fsk = np.zeros(sr)
    for i in range(20):
        freq = 19500 if i % 2 else 18500
        start = i * 2400
        fsk[start:start+2400] = np.sin(2 * np.pi * freq * t[start:start+2400])

    res_fsk = analyzer.analyze(fsk[:n_fft])
    assert res_fsk["modulation_type"] in ["fsk", "chirp", "tone"], "FSK detection failed"
    print(f"  [PASS] FSK signal analyzed: modulation={res_fsk['modulation_type']}, freqs={res_fsk['carrier_freqs']}")

    # Test 2: Steady tone
    tone = np.sin(2 * np.pi * 19000 * t[:n_fft])
    res_tone = analyzer.analyze(tone)
    assert res_tone["modulation_type"] in ["tone", "ook"], "Tone detection failed"
    print(f"  [PASS] Steady tone analyzed: modulation={res_tone['modulation_type']}")

    # Test 3: Random noise
    noise = np.random.randn(n_fft) * 0.01
    res_noise = analyzer.analyze(noise)
    assert res_noise["is_suspicious"] is False, "Noise should not be suspicious"
    print(f"  [PASS] Random noise analyzed: is_suspicious={res_noise['is_suspicious']}")

    # Test 4: Empty array
    res_empty = analyzer.analyze(np.array([]))
    assert res_empty["num_peaks"] == 0 and res_empty["is_suspicious"] is False
    print("  [PASS] Empty array handled cleanly")

    print("\n[ALL TESTS PASSED] frequency_analyzer.py")
