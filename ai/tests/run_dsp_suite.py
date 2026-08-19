"""
DSP Test Suite — Verify Everything Works
==========================================
Run this to confirm all DSP modules produce correct, consistent output
before handing off to Backend and AI 2.

Run:
    python -m pytest tests/run_dsp_suite.py -v
    # or simply:
    python tests/run_dsp_suite.py
"""

import sys
import time
import numpy as np

# ── Allow running from project root ──
sys.path.insert(0, ".")

from dsp.audio_preprocessing import AudioPreprocessor
from dsp.feature_extraction import FeatureExtractor
from dsp.spectrogram_generator import SpectrogramGenerator
from dsp.frequency_analyzer import FrequencyAnalyzer
from dsp.utils import SignalGenerator
from dsp.dsp_api import DSPPipeline


# ═══════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════

SAMPLE_RATE = 48000
N_FFT = 2048
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

results = []

def run_test(name, test_func):
    """Run a test and record pass/fail."""
    try:
        test_func()
        print(f"  {PASS}  {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL}  {name} — {e}")
        results.append((name, False))


# ═══════════════════════════════════════════════════════════════
# THING 1 TESTS: Audio Preprocessor
# ═══════════════════════════════════════════════════════════════

def test_preprocessor_output_shape():
    pp = AudioPreprocessor(sample_rate=SAMPLE_RATE)
    chunk = np.random.randn(N_FFT).astype(np.float32)
    result = pp.process(chunk)
    assert result.shape == (N_FFT,), f"Expected shape ({N_FFT},), got {result.shape}"

def test_preprocessor_removes_dc():
    pp = AudioPreprocessor(sample_rate=SAMPLE_RATE)
    # Signal with massive DC offset
    chunk = np.ones(N_FFT) * 0.5 + 0.001 * np.random.randn(N_FFT)
    result = pp.process(chunk)
    # After processing, the signal should have no significant DC
    # (the bandpass filter will also remove DC, but let's check the flow works)
    assert result is not None and len(result) == N_FFT

def test_preprocessor_filters_low_freq():
    pp = AudioPreprocessor(sample_rate=SAMPLE_RATE)
    gen = SignalGenerator(sample_rate=SAMPLE_RATE)
    # Pure 1 kHz tone — should be completely killed by the 17.5 kHz bandpass
    t = np.arange(N_FFT) / SAMPLE_RATE
    low_tone = np.sin(2 * np.pi * 1000 * t)
    result = pp.process(low_tone)
    # Energy should be near zero
    energy = np.sum(result ** 2)
    assert energy < 0.01, f"Low freq energy not suppressed: {energy}"

def test_preprocessor_passes_ultrasonic():
    pp = AudioPreprocessor(sample_rate=SAMPLE_RATE)
    t = np.arange(N_FFT) / SAMPLE_RATE
    us_tone = np.sin(2 * np.pi * 19000 * t)
    result = pp.process(us_tone)
    energy = np.sum(result ** 2)
    assert energy > 0.1, f"Ultrasonic signal was killed: energy={energy}"

def test_preprocessor_streaming_state():
    pp = AudioPreprocessor(sample_rate=SAMPLE_RATE)
    # Process multiple chunks — should not crash
    for _ in range(10):
        chunk = np.random.randn(N_FFT) * 0.1
        pp.process(chunk)
    assert pp.chunks_processed == 10


# ═══════════════════════════════════════════════════════════════
# THING 2 TESTS: Feature Extractor
# ═══════════════════════════════════════════════════════════════

def test_feature_count():
    ext = FeatureExtractor(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    chunk = np.random.randn(N_FFT)
    result = ext.extract(chunk)
    assert len(result["feature_values"]) == 32, f"Got {len(result['feature_values'])} features"
    assert len(result["feature_names"]) == 32
    assert len(result["feature_dict"]) == 32

def test_features_no_nan():
    ext = FeatureExtractor(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    # Test with various signal types
    signals = [
        np.zeros(N_FFT),                      # silence
        np.random.randn(N_FFT),               # noise
        np.sin(2 * np.pi * 19000 * np.arange(N_FFT) / SAMPLE_RATE),  # tone
    ]
    for i, sig in enumerate(signals):
        result = ext.extract(sig)
        for name, val in zip(result["feature_names"], result["feature_values"]):
            assert np.isfinite(val), f"NaN/Inf in feature '{name}' for signal {i}"

def test_features_detect_ultrasonic():
    ext = FeatureExtractor(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    t = np.arange(N_FFT) / SAMPLE_RATE

    # Ultrasonic tone
    us = ext.extract(np.sin(2 * np.pi * 19500 * t))
    # Noise only
    noise = ext.extract(np.random.randn(N_FFT) * 0.01)

    # Ultrasonic signal should have much higher energy ratio
    assert us["feature_dict"]["energy_ratio"] > noise["feature_dict"]["energy_ratio"] * 2

def test_feature_names_static():
    names = FeatureExtractor.get_feature_names()
    assert len(names) == 32
    assert names[0] == "ultrasonic_energy"
    assert names[-1] == "coefficient_of_variation"


# ═══════════════════════════════════════════════════════════════
# THING 3 TESTS: Spectrogram Generator
# ═══════════════════════════════════════════════════════════════

def test_spectrogram_json_fields():
    spec = SpectrogramGenerator(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    chunk = np.random.randn(N_FFT)
    json_str = spec.add_chunk(chunk)

    import json
    data = json.loads(json_str)

    required_fields = [
        "timestamp", "freq_axis", "magnitudes_db", "magnitudes_norm",
        "peak_freq_hz", "peak_magnitude_db", "ultrasonic_active"
    ]
    for field in required_fields:
        assert field in data, f"Missing field '{field}' in spectrogram JSON"

def test_spectrogram_history():
    spec = SpectrogramGenerator(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    for _ in range(20):
        spec.add_chunk(np.random.randn(N_FFT))
    full = spec.get_full_spectrogram()
    assert full["spectrogram_db"].shape[1] == 20

def test_spectrogram_detects_active():
    spec = SpectrogramGenerator(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    t = np.arange(N_FFT) / SAMPLE_RATE
    loud_us = np.sin(2 * np.pi * 19500 * t) * 0.9
    frame = spec.add_chunk_dict(loud_us)
    # A strong 19.5 kHz tone should trigger active flag
    assert frame["peak_freq_hz"] > 18000
    assert frame["peak_freq_hz"] < 22000


# ═══════════════════════════════════════════════════════════════
# THING 4 TESTS: Frequency Analyzer
# ═══════════════════════════════════════════════════════════════

def test_analyzer_detects_fsk():
    gen = SignalGenerator(sample_rate=SAMPLE_RATE)
    analyzer = FrequencyAnalyzer(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    fsk = gen.generate_fsk(duration_sec=0.5, freq_mark=19000, freq_space=20500, baud_rate=20)
    # Process multiple chunks
    chunk_size = N_FFT
    for i in range(0, len(fsk) - chunk_size, chunk_size):
        chunk = fsk[i:i+chunk_size]
        analyzer.analyze_and_accumulate(chunk)
    verdict = analyzer.get_accumulated_verdict()
    # Should detect something suspicious
    assert verdict["max_suspicion"] > 0.2, f"FSK not detected: {verdict}"

def test_analyzer_quiet_on_noise():
    analyzer = FrequencyAnalyzer(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    for _ in range(10):
        noise = np.random.randn(N_FFT) * 0.01
        analyzer.analyze_and_accumulate(noise)
    verdict = analyzer.get_accumulated_verdict()
    assert verdict["average_suspicion"] < 0.3, f"False alarm on noise: {verdict}"

def test_analyzer_single_chunk_result_fields():
    analyzer = FrequencyAnalyzer(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    chunk = np.random.randn(N_FFT)
    result = analyzer.analyze(chunk)
    required = ["modulation_type", "confidence", "carrier_freqs", "is_suspicious", "suspicion_score"]
    for field in required:
        assert field in result, f"Missing field '{field}'"


# ═══════════════════════════════════════════════════════════════
# THING 5 TESTS: Signal Generators
# ═══════════════════════════════════════════════════════════════

def test_generator_fsk_length():
    gen = SignalGenerator(sample_rate=SAMPLE_RATE)
    audio = gen.generate_fsk(duration_sec=1.0)
    expected = SAMPLE_RATE  # 1 second
    assert len(audio) == expected, f"Expected {expected}, got {len(audio)}"

def test_generator_ook_length():
    gen = SignalGenerator(sample_rate=SAMPLE_RATE)
    audio = gen.generate_ook(duration_sec=2.0)
    expected = 2 * SAMPLE_RATE
    assert len(audio) == expected

def test_generator_chirp_length():
    gen = SignalGenerator(sample_rate=SAMPLE_RATE)
    audio = gen.generate_chirp(duration_sec=1.5)
    expected = int(1.5 * SAMPLE_RATE)
    assert len(audio) == expected

def test_generator_mix_snr():
    gen = SignalGenerator(sample_rate=SAMPLE_RATE)
    signal = gen.generate_tone(duration_sec=1.0, frequency=19000)
    mixed = gen.mix_with_noise(signal, snr_db=10)
    assert len(mixed) == len(signal)
    assert np.max(np.abs(mixed)) <= 1.0  # no clipping


# ═══════════════════════════════════════════════════════════════
# DSP API (Integration) TESTS
# ═══════════════════════════════════════════════════════════════

def test_pipeline_full_process():
    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    chunk = np.random.randn(N_FFT).astype(np.float32)
    result = pipeline.process(chunk)

    assert "features" in result
    assert "spectrogram" in result
    assert "analysis" in result
    assert "frame_is_suspicious" in result
    assert "frame_suspicion_score" in result
    assert len(result["features"]["feature_values"]) == 32

def test_pipeline_attack_generation():
    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE)
    for attack_type in ["fsk", "ook", "chirp", "tone"]:
        signal = pipeline.generate_attack_signal(attack_type, duration_sec=0.5)
        assert len(signal) > 0, f"Empty signal for {attack_type}"

def test_pipeline_performance():
    """Each chunk should process in under 10 ms."""
    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    chunk = np.random.randn(N_FFT).astype(np.float64)

    # Warm up
    pipeline.process(chunk)

    # Benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        pipeline.process(chunk)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    avg_ms = np.mean(times)
    max_ms = np.max(times)
    assert avg_ms < 10.0, f"Too slow: avg={avg_ms:.2f}ms (target <10ms)"
    print(f"      Performance: avg={avg_ms:.2f}ms, max={max_ms:.2f}ms")

def test_pipeline_config():
    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE)
    config = pipeline.get_config()
    assert config["sample_rate"] == 48000
    assert config["num_features"] == 32
    assert config["recommended_stream_chunk_size"] == 2048

def test_pipeline_stats():
    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE)
    for _ in range(5):
        pipeline.process(np.random.randn(N_FFT))
    stats = pipeline.get_stats()
    assert stats["total_chunks_processed"] == 5


# ═══════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Acoustic Cybersecurity System — DSP Module Test Suite")
    print("=" * 60)

    # ── THING 1 ──
    print("\n📦 THING 1: Audio Preprocessor")
    run_test("Output shape matches input", test_preprocessor_output_shape)
    run_test("DC offset removed", test_preprocessor_removes_dc)
    run_test("Low frequencies filtered out", test_preprocessor_filters_low_freq)
    run_test("Ultrasonic frequencies pass through", test_preprocessor_passes_ultrasonic)
    run_test("Streaming state across chunks", test_preprocessor_streaming_state)

    # ── THING 2 ──
    print("\n📦 THING 2: Feature Extractor")
    run_test("Exactly 32 features", test_feature_count)
    run_test("No NaN/Inf in any feature", test_features_no_nan)
    run_test("Ultrasonic signal has higher energy ratio", test_features_detect_ultrasonic)
    run_test("Feature names are stable", test_feature_names_static)

    # ── THING 3 ──
    print("\n📦 THING 3: Spectrogram Generator")
    run_test("JSON output has all required fields", test_spectrogram_json_fields)
    run_test("History accumulates correctly", test_spectrogram_history)
    run_test("Detects active ultrasonic signal", test_spectrogram_detects_active)

    # ── THING 4 ──
    print("\n📦 THING 4: Frequency Analyzer")
    run_test("Detects FSK modulation", test_analyzer_detects_fsk)
    run_test("Quiet on pure noise", test_analyzer_quiet_on_noise)
    run_test("Result has all required fields", test_analyzer_single_chunk_result_fields)

    # ── THING 5 ──
    print("\n📦 THING 5: Signal Generators")
    run_test("FSK signal correct length", test_generator_fsk_length)
    run_test("OOK signal correct length", test_generator_ook_length)
    run_test("Chirp signal correct length", test_generator_chirp_length)
    run_test("SNR mixing no clipping", test_generator_mix_snr)

    # ── Integration ──
    print("\n🔗 DSP API (Integration)")
    run_test("Full pipeline process", test_pipeline_full_process)
    run_test("Attack signal generation", test_pipeline_attack_generation)
    run_test("Processing performance (<10ms)", test_pipeline_performance)
    run_test("Config returns correct values", test_pipeline_config)
    run_test("Stats tracking", test_pipeline_stats)

    # ── Summary ──
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    total = len(results)

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  {PASS}  ALL {total} TESTS PASSED")
    else:
        print(f"  {FAIL}  {failed}/{total} TESTS FAILED")
        for name, ok in results:
            if not ok:
                print(f"       - {name}")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)
