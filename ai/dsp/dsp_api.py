"""
DSP API — Single-Call Wrapper for Backend Integration
======================================================
This is the file Backend 1 imports.  One class, one method, full results.

Backend 1 — this is your interface:
────────────────────────────────────
    from dsp import DSPPipeline

    pipeline = DSPPipeline()                 # uses default settings
    result   = pipeline.process(audio_chunk) # one call does everything

    # result is a dict with ALL outputs:
    result["features"]      → 32-float feature vector (for AI 2's model)
    result["spectrogram"]   → JSON-ready spectrogram slice (for Frontend)
    result["analysis"]      → modulation type, suspicion score, etc.
    result["is_suspicious"] → bool (quick check)

Backend 2 — for the attack simulator:
────────────────────────────────────
    from dsp import DSPPipeline

    pipeline = DSPPipeline()
    attack   = pipeline.generate_test_signal("fsk", duration_sec=3.0)
    pipeline.save_wav(attack, "attack_demo.wav")
"""

import time
from typing import Optional

import numpy as np

from .audio_preprocessing import AudioPreprocessor
from .feature_extraction import FeatureExtractor
from .spectrogram_generator import SpectrogramGenerator
from .frequency_analyzer import FrequencyAnalyzer
from .utils import SignalGenerator
from .segmentation import StreamingSegmenter


def _make_json_serializable(obj):
    """Convert numpy scalar types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class DSPPipeline:
    """
    Unified DSP pipeline.  Backend teams: this is your only import.

    Construction:
        pipeline = DSPPipeline(sample_rate=48000)

    Processing loop:
        for chunk in audio_stream:
            result = pipeline.process(chunk)
            # → send result to AI 2 model, Frontend WebSocket, logging, etc.

    Attack simulation:
        signal = pipeline.generate_test_signal("fsk")
        pipeline.save_wav(signal, "demo_attack.wav")

    Get accumulated verdict (after processing many chunks):
        verdict = pipeline.get_verdict()
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        ultrasonic_low: float = 18000.0,
        ultrasonic_high: float = 21000.0,
        filter_low: float = 17500.0,
        filter_high: float = 21500.0,
        spectrogram_history_sec: float = 10.0,
        suspicious_threshold: float = 0.5,
    ):
        """
        Initialize all DSP sub-modules with consistent settings.

        Parameters
        ----------
        sample_rate : int
            Must be >= 44100.  48000 strongly recommended.
        n_fft : int
            FFT window size.  2048 gives ~23 Hz resolution at 48 kHz.
        ultrasonic_low, ultrasonic_high : float
            Detection band in Hz.
        filter_low, filter_high : float
            Bandpass filter edges (slightly wider than detection band).
        spectrogram_history_sec : float
            Seconds of spectrogram history to buffer.
        suspicious_threshold : float
            Suspicion score threshold for flagging (0.0 – 1.0).
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.ultrasonic_low = ultrasonic_low
        self.ultrasonic_high = ultrasonic_high
        self.filter_low = filter_low
        self.filter_high = filter_high

        # Initialize sub-modules
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            filter_low=filter_low,
            filter_high=filter_high,
        )

        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            ultrasonic_low=ultrasonic_low,
            ultrasonic_high=ultrasonic_high,
        )

        self.spectrogram = SpectrogramGenerator(
            sample_rate=sample_rate,
            n_fft=n_fft,
            ultrasonic_low=ultrasonic_low,
            ultrasonic_high=ultrasonic_high,
            history_seconds=spectrogram_history_sec,
        )

        self.analyzer = FrequencyAnalyzer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            ultrasonic_low=ultrasonic_low,
            ultrasonic_high=ultrasonic_high,
            suspicious_threshold=suspicious_threshold,
        )

        self.signal_generator = SignalGenerator(sample_rate=sample_rate)

        # Onset/offset tracking. Gives events a real start/end so the
        # AI->Backend `duration` field has a defensible source (a single
        # frame is only ~42.7 ms).
        self.segmenter = StreamingSegmenter(
            sample_rate=sample_rate,
            band=(ultrasonic_low, ultrasonic_high),
        )
        self._last_event = None
        self._peak_snr_db = 0.0
        self._pattern_votes = {}

        # Processing stats
        self._total_chunks = 0
        self._total_alerts = 0
        self._start_time = time.time()

    # ──────────────────────────────────────────────────────────
    # Main processing method (Backend 1: THIS IS YOUR MAIN CALL)
    # ──────────────────────────────────────────────────────────

    def process(self, raw_audio: np.ndarray) -> dict:
        """
        Process one audio chunk through the full DSP pipeline.

        Parameters
        ----------
        raw_audio : np.ndarray, shape (N,)
            Raw PCM samples from microphone, normalized to [-1.0, 1.0].

        Returns
        -------
        dict:
            "timestamp"             : float     — Unix timestamp
            "chunk_index"           : int       — Sequential chunk number
            "features"              : dict      — 32-feature extraction result
            "spectrogram"           : dict      — Real-time spectrogram frame
            "analysis"              : dict      — Frequency analysis results
            "frame_is_suspicious"   : bool      — Quick-access alert flag
            "frame_suspicion_score" : float     — Quick-access score [0, 1]
        """
        audio = np.asarray(raw_audio, dtype=np.float64)

        # 1. Preprocess (filter to ultrasonic band)
        clean = self.preprocessor.process(audio)

        # 2. Extract 32 features (for AI 2's ML model)
        features = self.feature_extractor.extract(clean)

        # 3. Generate spectrogram frame (for Frontend WebSocket)
        spec_frame = self.spectrogram.add_chunk_dict(clean)

        # 4. Frequency analysis with accumulation (for Backend/logging)
        analysis = self.analyzer.analyze_and_accumulate(clean)

        # 5. Onset/offset segmentation
        closed_event = self.segmenter.push(clean)
        if closed_event:
            self._last_event = closed_event

        # Track best SNR seen; the spectrogram frame carries it per chunk.
        frame_snr = float(spec_frame.get("peak_snr_db", 0.0))
        if frame_snr > self._peak_snr_db:
            self._peak_snr_db = frame_snr

        # Vote on modulation type across frames (see to_threat_event()).
        mod = analysis.get("modulation_type", "none")
        self._pattern_votes[mod] = self._pattern_votes.get(mod, 0) + 1

        # Update stats
        self._total_chunks += 1
        if analysis["is_suspicious"]:
            self._total_alerts += 1

        res = {
            "timestamp": float(time.time()),
            "chunk_index": int(self._total_chunks),
            "features": features,
            "spectrogram": spec_frame,
            "analysis": analysis,
            "frame_is_suspicious": bool(analysis["is_suspicious"]),
            "frame_suspicion_score": float(analysis["suspicion_score"]),

            # Documented aliases. INTEGRATION_GUIDE.md told Backend 1 to use
            # result["is_suspicious"] / result["suspicion_score"], which raised
            # KeyError because the real keys are frame_*. Both now work.
            "is_suspicious": bool(analysis["is_suspicious"]),
            "suspicion_score": float(analysis["suspicion_score"]),

            # SNR surfaced at the top level. It was computed inside the
            # spectrogram frame but never exposed, while the AI->Backend
            # contract (charter §14) asks for `snr` and Frontend §15 wants
            # "signal strength".
            "snr_db": float(spec_frame.get("peak_snr_db", 0.0)),
            "peak_frequency_hz": float(spec_frame.get("peak_freq_hz", 0.0)),

            # Segmentation: is a signal event open right now, and did one
            # just close on this chunk?
            "event_active": bool(self.segmenter.is_active),
            "event_closed": closed_event,
        }

        return _make_json_serializable(res)

    # ──────────────────────────────────────────────────────────
    # Convenience methods
    # ──────────────────────────────────────────────────────────

    def process_features_only(self, raw_audio: np.ndarray) -> dict:
        """
        Lightweight path: only preprocess + extract features.
        Use when you don't need spectrogram or analysis (e.g., batch processing).
        """
        clean = self.preprocessor.process(raw_audio)
        return self.feature_extractor.extract(clean)

    def get_verdict(self) -> dict:
        """
        Get the accumulated multi-chunk verdict from the frequency analyzer.
        Call this after processing multiple chunks for a final decision.
        """
        verdict = self.analyzer.get_accumulated_verdict()
        verdict["total_chunks_processed"] = self._total_chunks
        verdict["total_alerts"] = self._total_alerts
        verdict["uptime_seconds"] = round(time.time() - self._start_time, 2)
        return verdict

    def get_spectrogram_snapshot(self) -> dict:
        """Get the full rolling spectrogram (for Frontend 'full view' button)."""
        return self.spectrogram.get_full_spectrogram()

    def save_spectrogram_image(self, filepath: str = "spectrogram.png"):
        """Save current spectrogram as PNG image (for demo/report)."""
        self.spectrogram.save_spectrogram_image(filepath)

    # ──────────────────────────────────────────────────────────
    # Test-signal generation - FIXTURE ONLY
    # Backend 2 owns the real attack simulator. See audit point 3.
    # ──────────────────────────────────────────────────────────

    def generate_test_signal(
        self,
        signal_type: str = "fsk",
        duration_sec: float = 3.0,
        snr_db: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate a synthetic signal for TESTING and TRAINING.

        SCOPE (per Backend 1 audit, point 3)
        ------------------------------------
        This is a TEST FIXTURE, not an attack simulator. Backend 2 owns the
        real simulator (payload encoding, modulation, speaker transmission,
        over-the-air characteristics).

        What this is for:
          - deterministic fixtures for the automated test suite
          - baseline labelled training data for AI 2
          - driving demo/live_demo.py without hardware

        It produces idealised, in-memory numpy arrays with no channel effects
        (no room acoustics, no speaker/mic frequency response, no multipath).
        Validation against realistic audio must use Backend 2's WAV files.

        This method is NEVER called on the runtime detection path; process()
        does not touch it.

        Parameters
        ----------
        signal_type : "fsk" | "ook" | "chirp" | "tone"
        duration_sec : float
        snr_db : float, optional - mix in noise at this SNR
        """
        gen = self.signal_generator
        attack_type = signal_type

        if attack_type == "fsk":
            signal = gen.generate_fsk(duration_sec=duration_sec, **kwargs)
        elif attack_type == "ook":
            signal = gen.generate_ook(duration_sec=duration_sec, **kwargs)
        elif attack_type == "chirp":
            signal = gen.generate_chirp(duration_sec=duration_sec, **kwargs)
        elif attack_type == "tone":
            signal = gen.generate_tone(duration_sec=duration_sec, **kwargs)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}. Use 'fsk', 'ook', 'chirp', or 'tone'.")

        if snr_db is not None:
            signal = gen.mix_with_noise(signal, snr_db=snr_db)

        return signal

    def generate_attack_signal(self, *args, **kwargs) -> np.ndarray:
        """
        Deprecated alias for generate_test_signal().

        Renamed because "attack signal" implied this module owned attack
        simulation, which belongs to Backend 2. Kept so existing callers and
        the demo scripts keep working.
        """
        return self.generate_test_signal(*args, **kwargs)

    def save_wav(self, audio: np.ndarray, filepath: str):
        """Save audio array as WAV file."""
        self.signal_generator.save_wav(audio, filepath)

    # ──────────────────────────────────────────────────────────
    # Dataset generation (AI 2: training data)
    # ──────────────────────────────────────────────────────────

    def generate_training_dataset(
        self,
        output_dir: str = "training_data",
        samples_per_class: int = 500,
        chunk_duration_sec: Optional[float] = None,
    ):
        """
        Generate labeled training dataset for AI 2's classifier.
        Applies AudioPreprocessor before FeatureExtractor so training data
        matches runtime pipeline features.
        """
        def _preprocess_example(audio: np.ndarray) -> np.ndarray:
            self.preprocessor.reset_filter_state()
            return self.preprocessor.process(audio)

        return self.signal_generator.generate_dataset(
            output_dir=output_dir,
            samples_per_class=samples_per_class,
            chunk_duration_sec=chunk_duration_sec,
            n_fft=self.n_fft,
            preprocess_fn=_preprocess_example,
        )

    # ──────────────────────────────────────────────────────────
    # Lifecycle & Configuration
    # ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset all internal state (filter state, history, accumulators)."""
        self.preprocessor.reset_filter_state()
        self.spectrogram.clear_history()
        self.analyzer.reset_accumulator()
        self.segmenter.reset()
        self._last_event = None
        self._peak_snr_db = 0.0
        self._pattern_votes = {}
        self._total_chunks = 0
        self._total_alerts = 0
        self._start_time = time.time()

    def get_stats(self) -> dict:
        """Return processing statistics."""
        return {
            "total_chunks_processed": self._total_chunks,
            "total_alerts": self._total_alerts,
            "alert_rate": self._total_alerts / max(1, self._total_chunks),
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "spectrogram_frames": self.spectrogram.frame_count,
        }

    # ──────────────────────────────────────────────────────────
    # AI -> Backend contract (charter §14)
    # ──────────────────────────────────────────────────────────

    # Suspicion score -> risk band. AI 2 owns the final calibration; these are
    # DSP-side defaults so Backend 1 is not blocked waiting for the model.
    RISK_THRESHOLDS = (("HIGH", 0.75), ("MEDIUM", 0.45), ("LOW", 0.0))

    @classmethod
    def score_to_risk(cls, score: float) -> str:
        """Map a suspicion score in [0,1] to LOW / MEDIUM / HIGH."""
        for label, cutoff in cls.RISK_THRESHOLDS:
            if score >= cutoff:
                return label
        return "LOW"

    def to_threat_event(self, min_chunks: int = 1) -> dict:
        """
        Build the AI -> Backend detection event (charter §14).

        IMPORTANT — this is built from the ACCUMULATED verdict, not a single
        frame. `duration` cannot come from one chunk: a chunk is 2048 samples
        = ~42.7 ms at 48 kHz. Backend 1 should call this periodically (or at
        end of capture), NOT once per process() call.

        Returns
        -------
        dict matching the charter §14 shape:

            {
              "detected":        bool,
              "confidence":      float,   modulation-classifier certainty
              "risk":            str,     LOW | MEDIUM | HIGH
              "frequency_start": float,
              "frequency_end":   float,
              "carrier_freqs":   [float], see note below
              "duration":        float,   seconds
              "pattern":         str,
              "snr":             float,   dB
              "suspicion_score": float,   threat heuristic, distinct from confidence
              "chunks_analyzed": int,
              "timestamp":       float,
              "schema_version":  str,
            }

        Note on frequency_start/end
        ---------------------------
        The charter asks for a min/max range, but this pipeline detects
        DISCRETE carriers. Collapsing [19000, 20500] to {min: 19000,
        max: 20500} discards the fact that there were two separate carriers —
        which is precisely the FSK signature. `carrier_freqs` is therefore
        included alongside; Backend 1 should persist it.

        Note on the two confidences
        ---------------------------
        `confidence`      = how sure the modulation classifier is of its label.
        `suspicion_score` = how threat-like the signal is overall.
        They are NOT interchangeable and must not be conflated.
        """
        verdict = self.get_verdict()
        carriers = [float(c) for c in verdict.get("consistent_carriers", [])]

        # Prefer a measured segmentation duration; fall back to the analyzer's
        # accumulated estimate.
        duration = 0.0
        if self._last_event:
            duration = float(self._last_event.get("duration_sec", 0.0))
        if duration <= 0.0:
            duration = float(verdict.get("threat_duration_sec", 0.0))

        score = float(verdict.get("average_suspicion", 0.0))
        detected = bool(verdict.get("is_threat", False)) and \
            self._total_chunks >= min_chunks

        # Pattern must be decided ACROSS frames, not from the last one.
        #
        # At 25 baud a symbol lasts 40 ms while a frame is 42.7 ms, so most
        # individual frames of an FSK signal contain a single steady carrier
        # and classify as "tone". Measured on a 3 s FSK signal: 55 frames said
        # tone, 15 said fsk. Taking the last frame reported "tone" for an
        # unambiguous two-carrier FSK transmission.
        #
        # Rule: if the accumulated verdict found multiple consistent carriers,
        # the event is FSK-like regardless of what any single frame said.
        pattern = "none"
        confidence = 0.0
        if self._pattern_votes:
            # Ignore "none" unless it is all we have.
            votes = {k: v for k, v in self._pattern_votes.items() if k != "none"}
            if not votes:
                votes = self._pattern_votes
            pattern = max(votes, key=votes.get)
            total = sum(votes.values())
            confidence = votes[pattern] / total if total else 0.0

            # Multiple distinct carriers => frequency-shift keying, even if
            # individual frames each saw only one of them.
            if len(carriers) >= 2 and pattern == "tone":
                pattern = "fsk"
                confidence = max(confidence, len(carriers) / (len(carriers) + 1))

        # Blend in the classifier's own confidence on its best frame.
        if self.analyzer.last_result:
            confidence = max(
                confidence * 0.5
                + float(self.analyzer.last_result.get("confidence", 0.0)) * 0.5,
                0.0,
            )
        # A 'none' pattern must never report non-zero confidence. Without this,
        # a run of all-'none' frames produced a unanimous vote share of 1.0,
        # which blended to confidence 0.5 on an event with nothing detected.
        if pattern == "none":
            confidence = 0.0

        return _make_json_serializable({
            "schema_version": "1.0.0-dsp",
            "detected": detected,
            "confidence": round(confidence, 4),
            "risk": self.score_to_risk(score) if detected else "LOW",
            "suspicion_score": round(score, 4),
            "frequency_start": round(min(carriers), 1) if carriers else None,
            "frequency_end": round(max(carriers), 1) if carriers else None,
            "carrier_freqs": [round(c, 1) for c in carriers],
            "duration": round(duration, 3),
            "pattern": pattern,
            "snr": round(float(self._peak_snr_db), 2),
            "chunks_analyzed": int(self._total_chunks),
            "timestamp": float(time.time()),
        })

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Configuration without constructing a pipeline.

        INTEGRATION_GUIDE.md documented `DSPPipeline.get_config()` as a class
        method; it was an instance method, so that call raised TypeError.
        Use this for the class-level call, or get_config() on an instance.
        """
        return cls().get_config()

    def get_config(self) -> dict:
        """Return the current active configuration for all team members."""
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "recommended_stream_chunk_size": self.n_fft,
            "ultrasonic_band_hz": [self.ultrasonic_low, self.ultrasonic_high],
            "filter_band_hz": [self.filter_low, self.filter_high],
            "num_features": 32,
            "input_channels": 1,
            "preferred_input_dtype": "float32",
            "normalized_range": [-1.0, 1.0],
            "preprocessor_output_dtype": "float32",
        }
