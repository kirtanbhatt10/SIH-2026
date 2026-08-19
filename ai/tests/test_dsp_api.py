"""
DSP API Integration & Interface Unit Tests
=============================================
Validates the public DSPPipeline API methods:
process(), process_features_only(), get_verdict(), reset(),
get_config(), generate_attack_signal(), generate_training_dataset(),
and JSON serializability.
"""

import os
import sys
import json
import shutil
import pprint
import numpy as np

# Ensure project root & dsp/ are in sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./dsp"))

try:
    from dsp.dsp_api import DSPPipeline
except ImportError:
    from dsp_api import DSPPipeline

SAMPLE_RATE = 48000
N_FFT = 2048


def test_1_configuration():
    print("\n==================================================")
    print("  TEST 1 — Configuration & Initialization")
    print("==================================================")

    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    config = pipeline.get_config()

    print("  DSPPipeline.get_config() Output:")
    pprint.pprint(config)

    assert config["sample_rate"] == 48000, "Sample rate mismatch"
    assert config["n_fft"] == 2048, "N_FFT mismatch"
    assert config["recommended_stream_chunk_size"] == 2048, "Recommended chunk size mismatch"
    assert config["ultrasonic_band_hz"] == [18000.0, 21000.0], "Ultrasonic band mismatch"
    assert config["filter_band_hz"] == [17500.0, 21500.0], "Filter band mismatch"
    assert config["num_features"] == 32, "Feature count mismatch"
    assert config["input_channels"] == 1, "Input channels mismatch"
    assert config["preferred_input_dtype"] == "float32", "Input dtype mismatch"
    assert config["normalized_range"] == [-1.0, 1.0], "Normalized range mismatch"
    assert config["preprocessor_output_dtype"] == "float32", "Preprocessor output dtype mismatch"

    print("  [PASS] TEST 1 — Configuration is internally consistent")


def test_2_and_3_fsk_processing_and_verdict():
    print("\n==================================================")
    print("  TEST 2 & 3 — FSK Chunk Processing & Multi-Chunk Verdict")
    print("==================================================")

    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE, n_fft=N_FFT)

    # Generate 2-second FSK signal (18500 Hz & 19500 Hz)
    fsk_signal = pipeline.generate_attack_signal(
        "fsk",
        duration_sec=2.0,
        freq_mark=18500.0,
        freq_space=19500.0,
        baud_rate=20.0
    )

    chunk_size = N_FFT
    num_chunks = len(fsk_signal) // chunk_size

    required_keys = [
        "timestamp",
        "chunk_index",
        "features",
        "spectrogram",
        "analysis",
        "frame_is_suspicious",
        "frame_suspicion_score",
    ]

    for i in range(num_chunks):
        chunk = fsk_signal[i * chunk_size : (i + 1) * chunk_size]
        result = pipeline.process(chunk)

        # Check required fields
        for k in required_keys:
            assert k in result, f"Missing key '{k}' in process() result"

        # Check features
        feat_vals = result["features"]["feature_values"]
        assert len(feat_vals) == 32, f"Expected 32 features, got {len(feat_vals)}"
        assert np.all(np.isfinite(feat_vals)), "Non-finite feature detected"

        # Check spectrogram
        spec = result["spectrogram"]
        assert len(spec["freq_axis"]) == 64, f"Expected 64 ui_bins, got {len(spec['freq_axis'])}"

        # Check analysis
        analysis = result["analysis"]
        assert "modulation_type" in analysis
        assert "carrier_freqs" in analysis
        assert "suspicion_score" in analysis

    print(f"  Processed {num_chunks} sequential chunks successfully.")
    print("  [PASS] TEST 2 — All chunks validated against required fields & payload shape")

    # TEST 3: Accumulated Verdict
    verdict = pipeline.get_verdict()
    print("\n  Accumulated Verdict Output:")
    print(f"    is_threat           : {verdict['is_threat']}")
    print(f"    average_suspicion   : {verdict['average_suspicion']}")
    print(f"    consistent_carriers : {verdict['consistent_carriers']}")
    print(f"    chunks_analyzed     : {verdict['chunks_analyzed']}")
    print(f"    threat_duration_sec : {verdict['threat_duration_sec']}")

    assert verdict["chunks_analyzed"] == num_chunks, "Verdict chunk count mismatch"
    assert verdict["is_threat"] is True, "Expected threat verdict for 2s FSK signal"
    print("  [PASS] TEST 3 — Accumulated verdict verified")


def test_4_pipeline_reset():
    print("\n==================================================")
    print("  TEST 4 — Pipeline Reset")
    print("==================================================")

    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    chunk = np.random.randn(N_FFT).astype(np.float32)

    for _ in range(5):
        pipeline.process(chunk)

    assert pipeline._total_chunks == 5
    assert pipeline.spectrogram.frame_count > 0

    pipeline.reset()

    stats = pipeline.get_stats()
    verdict = pipeline.get_verdict()

    assert stats["total_chunks_processed"] == 0, f"Expected 0 processed chunks, got {stats['total_chunks_processed']}"
    assert stats["total_alerts"] == 0, f"Expected 0 total alerts, got {stats['total_alerts']}"
    assert stats["spectrogram_frames"] == 0, f"Expected 0 spectrogram frames, got {stats['spectrogram_frames']}"
    assert verdict["chunks_analyzed"] == 0, f"Expected 0 analyzed chunks in verdict, got {verdict['chunks_analyzed']}"
    assert len(verdict["consistent_carriers"]) == 0, "Accumulator carrier list not empty"

    print("  [PASS] TEST 4 — Pipeline state reset verified")


def test_5_attack_signal_generation():
    print("\n==================================================")
    print("  TEST 5 — Attack Signal Generation")
    print("==================================================")

    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE)
    duration = 1.5
    expected_samples = int(duration * SAMPLE_RATE)

    attack_types = ["fsk", "ook", "chirp", "tone"]
    for atype in attack_types:
        signal = pipeline.generate_attack_signal(atype, duration_sec=duration)
        assert signal.ndim == 1, f"Signal for {atype} is not 1D"
        assert len(signal) == expected_samples, f"Expected {expected_samples} samples for {atype}, got {len(signal)}"
        assert np.all(np.isfinite(signal)), f"Non-finite values in {atype} signal"
        print(f"  [PASS] Attack type '{atype}': shape={signal.shape}, finite=True")

    print("  [PASS] TEST 5 — All attack signals generated correctly")


def test_6_training_dataset_generation():
    print("\n==================================================")
    print("  TEST 6 — Small Training Dataset Generation")
    print("==================================================")

    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    test_dir = "temp_dsp_api_dataset"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    X, y = pipeline.generate_training_dataset(
        output_dir=test_dir,
        samples_per_class=3
    )

    print(f"  Generated dataset X shape: {X.shape}, y shape: {y.shape}")

    expected_shape = (15, 32)
    assert X.shape == expected_shape, f"Expected shape {expected_shape}, got {X.shape}"
    assert len(y) == 15, f"Expected 15 labels, got {len(y)}"
    assert np.all(np.isfinite(X)), "Non-finite values in training dataset"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("  [PASS] TEST 6 — Training dataset generated with shape (15, 32)")


def test_7_json_serialization():
    print("\n==================================================")
    print("  TEST 7 — Real-Time Response JSON Serialization")
    print("==================================================")

    pipeline = DSPPipeline(sample_rate=SAMPLE_RATE, n_fft=N_FFT)
    chunk = np.random.randn(N_FFT).astype(np.float32)

    result = pipeline.process(chunk)

    try:
        json_str = json.dumps(result)
        assert len(json_str) > 0
        deserialized = json.loads(json_str)
        assert deserialized["chunk_index"] == result["chunk_index"]
        print(f"  Successfully serialized result payload to JSON ({len(json_str)} bytes).")
        print("  [PASS] TEST 7 — Real-time result dictionary is 100% JSON serializable")
    except Exception as e:
        print(f"  [FAIL] JSON serialization failed: {e}")
        raise e


if __name__ == "__main__":
    print("\n============================================================")
    print("  Acoustic Cybersecurity System — DSP API Public Interface Test Suite")
    print("============================================================")

    test_1_configuration()
    test_2_and_3_fsk_processing_and_verdict()
    test_4_pipeline_reset()
    test_5_attack_signal_generation()
    test_6_training_dataset_generation()
    test_7_json_serialization()

    print("\n============================================================")
    print("  [ALL DSP API TESTS PASSED SUCCESSFULLY]")
    print("============================================================")
