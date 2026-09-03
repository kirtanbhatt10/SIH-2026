"""
Dataset V2 sample generator.

Pipeline for every sample (matches the frozen Phase 1 contract exactly):

    dsp.SignalGenerator   -> synthetic audio waveform (varied parameters)
          |
    dsp.AudioPreprocessor -> real bandpass/pre-emphasis pipeline
          |
    dsp.FeatureExtractor  -> real, frozen 32-feature extraction
          |
    (features, label, rich metadata)

This module never fabricates a 32-dim feature vector, never imports the
tier-2 fallback in ai/ml/data_generation.py, and never modifies
dsp/utils.py, dsp/audio_preprocessing.py, or dsp/feature_extraction.py.
It only calls their existing public APIs with logged, varied arguments.

Reproducibility
----------------
SignalGenerator internally draws from the *global* numpy RNG (np.random.*)
for payload bits and noise. To keep that reproducible without touching
dsp/utils.py, each sample seeds the global RNG from a value derived
(via np.random.SeedSequence) from (dataset_seed, class_index, sample_index)
immediately before calling into SignalGenerator. All of Dataset V2's own
parameter choices (frequency, SNR, amplitude, crop offset, ...) are drawn
from a *separate*, independently seeded np.random.Generator, so adding or
reordering Dataset V2's own random draws can never change what
SignalGenerator produces for a given seed, and vice versa.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# --- make `import dsp` work regardless of current working directory -------
_HERE = Path(__file__).resolve().parent            # ai/ml/dataset_v2
_AI_DIR = _HERE.parent.parent                       # ai/
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from dsp import (  # noqa: E402
    AudioPreprocessor,
    FeatureExtractor,
    SignalGenerator,
    SAMPLE_RATE,
    CHUNK_SIZE,
    ULTRASONIC_LOW,
    ULTRASONIC_HIGH,
    FILTER_LOW,
    FILTER_HIGH,
)

try:
    from . import config_v2 as cfg  # noqa: E402
except ImportError:  # pragma: no cover - allows `python generator.py` directly
    import config_v2 as cfg  # type: ignore  # noqa: E402


CLASS_NAMES = cfg.CLASS_NAMES
CHUNK_DURATION_SEC = CHUNK_SIZE / SAMPLE_RATE


def _derive_seeds(dataset_seed: int, class_idx: int, sample_idx: int):
    """Deterministically derive two independent seeds for one sample.

    local_seed  -> feeds this module's own np.random.Generator
    global_seed -> seeds np.random (global) right before calling
                   SignalGenerator, which reads global state internally.
    """
    ss = np.random.SeedSequence([int(dataset_seed), int(class_idx), int(sample_idx)])
    local_seed, global_seed = (int(x) for x in ss.generate_state(2))
    return local_seed, global_seed


def _crop_window(local_rng: np.random.Generator, waveform: np.ndarray, chunk_size: int):
    """Pick a chunk_size-length window from `waveform`, giving timing/phase
    diversity. Pads with zeros if the source is somehow shorter than one
    chunk (should not normally happen given SOURCE_FACTOR_RANGE >= 3).
    """
    if len(waveform) <= chunk_size:
        pad = chunk_size - len(waveform)
        return np.pad(waveform, (0, pad)), 0.0
    max_start = len(waveform) - chunk_size
    start = int(local_rng.integers(0, max_start + 1))
    offset_sec = start / SAMPLE_RATE
    return waveform[start:start + chunk_size], offset_sec


def _crop_until_audible(
    local_rng: np.random.Generator,
    waveform: np.ndarray,
    chunk_size: int,
    preprocessor: AudioPreprocessor,
    max_attempts: int = cfg.MAX_CROP_RESAMPLE_ATTEMPTS,
    squelch_floor: float = cfg.AUDIO_PREPROCESSOR_SQUELCH_FLOOR,
):
    """Crop a chunk_size window and preprocess it; if AudioPreprocessor's
    real squelch (see config_v2.AUDIO_PREPROCESSOR_SQUELCH_FLOOR) zeroes it
    out, re-crop a different window from the SAME already-generated audio
    and try again -- still fully deterministic, since it only continues
    drawing from local_rng.

    Only used for the four communication classes. For benign, a squelched
    (near-silent) chunk is a legitimate example -- "no ultrasonic content"
    is exactly what benign means -- so benign never retries (see
    generate_sample()).

    Returns (chunk, crop_offset_sec, processed, attempts_used). If every
    attempt is squelched, returns the loudest attempt seen and
    attempts_used == max_attempts, so it can be reported rather than hidden.
    """
    best = None
    for attempt in range(max_attempts):
        chunk, offset = _crop_window(local_rng, waveform, chunk_size)
        preprocessor.reset_filter_state()
        processed = preprocessor.process(chunk)
        peak = float(np.max(np.abs(processed)))
        if best is None or peak > best[3]:
            best = (chunk, offset, processed, peak)
        if peak >= squelch_floor:
            return chunk, offset, processed, attempt
    chunk, offset, processed, _peak = best
    return chunk, offset, processed, max_attempts


def _pick_snr(local_rng: np.random.Generator):
    tier = local_rng.choice(cfg.SNR_TIER_NAMES, p=cfg.SNR_TIER_WEIGHTS)
    lo, hi = cfg.SNR_TIERS[tier]
    snr_db = float(local_rng.uniform(lo, hi))
    return tier, snr_db


def _pick_noise_type(local_rng: np.random.Generator):
    return str(local_rng.choice(cfg.NOISE_TYPES, p=cfg.NOISE_TYPE_WEIGHTS))


def _empty_mod_params():
    return dict(
        frequency=None,
        frequency_deviation=None,
        start_frequency=None,
        end_frequency=None,
        duty_cycle=None,
        bit_rate=None,
        chirp_sweeps=None,
        benign_variant=None,
        amplitude=None,
    )


def _gen_benign(sg: SignalGenerator, local_rng: np.random.Generator, duration_sec: float):
    variant = str(local_rng.choice(cfg.BENIGN_VARIANTS, p=cfg.BENIGN_VARIANT_WEIGHTS))
    amp = float(local_rng.uniform(*cfg.BENIGN_AMPLITUDE_RANGE))

    params = _empty_mod_params()
    params["benign_variant"] = variant
    params["amplitude"] = amp

    if variant == "ambient":
        audio = sg.generate_ambient_noise(duration_sec=duration_sec, amplitude=amp)
        noise_type, snr_db = "ambient", None
    elif variant == "white":
        audio = sg.generate_white_noise(duration_sec=duration_sec, amplitude=amp)
        noise_type, snr_db = "white", None
    else:  # ambient_hf_boost
        base = sg.generate_ambient_noise(duration_sec=duration_sec, amplitude=amp)
        lo, hi = cfg.BENIGN_HF_BOOST_SNR_RANGE
        snr_db = float(local_rng.uniform(lo, hi))
        audio = sg.mix_with_noise(base, snr_db=snr_db, noise_type="white")
        noise_type = "white"

    return audio, noise_type, snr_db, params


def _gen_tone(sg: SignalGenerator, local_rng: np.random.Generator, duration_sec: float):
    freq = float(local_rng.uniform(*cfg.TONE_FREQ_RANGE))
    amp = float(local_rng.uniform(*cfg.AMPLITUDE_RANGE))
    audio = sg.generate_tone(duration_sec=duration_sec, frequency=freq, amplitude=amp)
    params = _empty_mod_params()
    params["frequency"] = freq
    params["amplitude"] = amp
    return audio, params


def _gen_fsk(sg: SignalGenerator, local_rng: np.random.Generator, duration_sec: float):
    f_mark = float(local_rng.uniform(*cfg.FSK_MARK_RANGE))
    deviation = float(local_rng.uniform(*cfg.FSK_DEVIATION_RANGE))
    f_space = min(f_mark + deviation, cfg.FSK_MAX_SPACE_FREQ)
    baud = float(local_rng.uniform(*cfg.FSK_BAUD_RANGE))
    amp = float(local_rng.uniform(*cfg.AMPLITUDE_RANGE))

    audio = sg.generate_fsk(
        duration_sec=duration_sec, freq_mark=f_mark, freq_space=f_space, baud_rate=baud
    )
    audio = audio * amp

    params = _empty_mod_params()
    params["frequency"] = f_mark
    params["frequency_deviation"] = f_space - f_mark
    params["bit_rate"] = baud
    params["amplitude"] = amp
    return audio, params


def _gen_ook(sg: SignalGenerator, local_rng: np.random.Generator, duration_sec: float):
    carrier = float(local_rng.uniform(*cfg.OOK_CARRIER_RANGE))
    baud = float(local_rng.uniform(*cfg.OOK_BAUD_RANGE))
    amp = float(local_rng.uniform(*cfg.AMPLITUDE_RANGE))

    audio = sg.generate_ook(duration_sec=duration_sec, carrier_freq=carrier, baud_rate=baud)
    audio = audio * amp

    params = _empty_mod_params()
    params["frequency"] = carrier
    params["bit_rate"] = baud
    # NOTE: duty_cycle is NOT independently controllable by the existing
    # generate_ook() -- bits are drawn uniformly random (~50% expected duty
    # cycle), not user-configurable. Recorded as null rather than fabricated.
    # See README.md "Known Limitations".
    params["duty_cycle"] = None
    params["amplitude"] = amp
    return audio, params


def _gen_chirp(sg: SignalGenerator, local_rng: np.random.Generator, duration_sec: float):
    f_start = float(local_rng.uniform(*cfg.CHIRP_START_RANGE))
    f_end = float(local_rng.uniform(*cfg.CHIRP_END_RANGE))
    sweeps = int(local_rng.integers(cfg.CHIRP_SWEEPS_RANGE[0], cfg.CHIRP_SWEEPS_RANGE[1] + 1))
    amp = float(local_rng.uniform(*cfg.AMPLITUDE_RANGE))

    audio = sg.generate_chirp(
        duration_sec=duration_sec, freq_start=f_start, freq_end=f_end, num_sweeps=sweeps
    )
    audio = audio * amp

    params = _empty_mod_params()
    params["start_frequency"] = f_start
    params["end_frequency"] = f_end
    params["chirp_sweeps"] = sweeps
    params["amplitude"] = amp
    return audio, params


_GENERATORS = {
    "benign": None,   # handled specially (owns its own noise/snr logic)
    "tone": _gen_tone,
    "fsk": _gen_fsk,
    "ook": _gen_ook,
    "chirp": _gen_chirp,
}


def _freq_bucket(frequency: Optional[float]) -> str:
    if frequency is None:
        return "na"
    return f"{int(round(frequency / 500.0) * 500)}"


def _snr_tier_from_db(snr_db: Optional[float]) -> Optional[str]:
    if snr_db is None:
        return None
    for tier, (lo, hi) in cfg.SNR_TIERS.items():
        if lo - 1e-6 <= snr_db <= hi + 1e-6:
            return tier
    return "other"


def _amplitude_bucket(amplitude: Optional[float]) -> str:
    if amplitude is None:
        return "na"
    # 0.1-wide buckets: 0.15->a10, 0.55->a50, 0.6->a60
    return f"a{int(round(amplitude * 10) * 10)}"


def _build_generation_group(
    class_name: str,
    snr_tier: Optional[str],
    snr_db: Optional[float],
    params: dict,
) -> str:
    """Composite key for grouped train/test splits.

    Groups related synthetic scenarios without creating one group per sample.
    """
    if class_name == "benign":
        variant = params.get("benign_variant") or "na"
        if variant == "ambient_hf_boost":
            tier = _snr_tier_from_db(snr_db) or "na"
            return f"benign|{variant}|{tier}"
        return f"benign|{variant}|{_amplitude_bucket(params.get('amplitude'))}"

    if class_name == "chirp":
        start_b = _freq_bucket(params.get("start_frequency"))
        end_b = _freq_bucket(params.get("end_frequency"))
        return f"chirp|{snr_tier or 'na'}|{start_b}_{end_b}"

    return f"{class_name}|{snr_tier or 'na'}|{_freq_bucket(params.get('frequency'))}"


def _crop_prefer_loudest(
    local_rng: np.random.Generator,
    waveform: np.ndarray,
    chunk_size: int,
    preprocessor: AudioPreprocessor,
    max_attempts: int,
):
    """Try several random crop offsets; keep the loudest post-preprocess window.

    Used for benign samples to reduce identical all-zero feature rows when a
    longer source waveform contains intermittent high-frequency energy.
    """
    best = None
    for attempt in range(max_attempts):
        chunk, offset = _crop_window(local_rng, waveform, chunk_size)
        preprocessor.reset_filter_state()
        processed = preprocessor.process(chunk)
        peak = float(np.max(np.abs(processed)))
        if best is None or peak > best[3]:
            best = (chunk, offset, processed, peak, attempt)
    chunk, offset, processed, _peak, attempts_used = best
    return chunk, offset, processed, attempts_used


class DatasetV2Sample:
    """Container for one generated sample: raw audio + features + metadata."""

    __slots__ = ("audio", "features", "metadata")

    def __init__(self, audio: np.ndarray, features: np.ndarray, metadata: dict):
        self.audio = audio
        self.features = features
        self.metadata = metadata


def generate_sample(
    class_idx: int,
    sample_idx: int,
    dataset_seed: int,
    preprocessor: AudioPreprocessor,
    extractor: FeatureExtractor,
) -> DatasetV2Sample:
    """Generate exactly one Dataset V2 sample, end to end."""
    class_name = CLASS_NAMES[class_idx]
    local_seed, global_seed = _derive_seeds(dataset_seed, class_idx, sample_idx)
    local_rng = np.random.default_rng(local_seed)

    source_factor = float(local_rng.uniform(*cfg.SOURCE_FACTOR_RANGE))
    source_duration_sec = CHUNK_DURATION_SEC * source_factor

    # Seed the GLOBAL numpy RNG right before touching SignalGenerator, so
    # its internal random draws (payload bits, noise) are reproducible for
    # this (dataset_seed, class_idx, sample_idx) tuple, independent of the
    # order in which Dataset V2's own local_rng calls happen above/below.
    np.random.seed(global_seed)

    if class_name == "benign":
        audio, noise_type, snr_db, params = _gen_benign(
            SignalGenerator(sample_rate=SAMPLE_RATE), local_rng, source_duration_sec
        )
        if params.get("benign_variant") == "ambient_hf_boost" and snr_db is not None:
            snr_tier = _snr_tier_from_db(snr_db)
        else:
            snr_tier = None
    else:
        sg = SignalGenerator(sample_rate=SAMPLE_RATE)
        audio, params = _GENERATORS[class_name](sg, local_rng, source_duration_sec)
        snr_tier, snr_db = _pick_snr(local_rng)
        noise_type = _pick_noise_type(local_rng)
        audio = sg.mix_with_noise(audio, snr_db=snr_db, noise_type=noise_type)

    # "amplitude" in metadata = the generation-time amplitude parameter Dataset
    # V2 actually chose (what was varied per Task 6), not a post-hoc peak
    # reading -- mix_with_noise() renormalizes on clipping, which would make
    # a post-hoc peak measurement saturate at ~1.0 for most noisy samples and
    # silently erase the intended amplitude variation from the record.
    amplitude = params.get("amplitude")
    peak_amplitude = float(np.max(np.abs(audio))) if len(audio) else 0.0

    # For communication classes (not benign), use the retry-aware crop that
    # avoids degenerate all-zero rows caused by AudioPreprocessor's squelch.
    # For benign, a squelched chunk is legitimate (no ultrasonic content).
    if class_name != "benign":
        chunk, crop_offset_sec, processed, crop_attempts = _crop_until_audible(
            local_rng, audio, CHUNK_SIZE, preprocessor,
        )
    else:
        chunk, crop_offset_sec, processed, crop_attempts = _crop_prefer_loudest(
            local_rng, audio, CHUNK_SIZE, preprocessor, cfg.BENIGN_CROP_ATTEMPTS,
        )

    result = extractor.extract(processed)
    feature_values = np.asarray(result["feature_values"], dtype=np.float64)

    sample_id = f"{cfg.DATASET_VERSION}_{class_name}_{sample_idx:06d}"
    generation_group = _build_generation_group(class_name, snr_tier, snr_db, params)

    metadata = {
        "sample_id": sample_id,
        "label": class_idx,
        "signal_type": class_name,
        "sample_rate": SAMPLE_RATE,
        "duration": CHUNK_DURATION_SEC,
        "frequency": params.get("frequency"),
        "snr": snr_db,
        "noise_level": snr_tier,
        "amplitude": amplitude,
        "peak_amplitude": peak_amplitude,
        "noise_type": noise_type,
        "seed": dataset_seed,
        "local_seed": local_seed,
        "global_seed": global_seed,
        "generator_version": cfg.GENERATOR_VERSION,
        "dataset_version": cfg.DATASET_VERSION,
        "bit_rate": params.get("bit_rate"),
        "frequency_deviation": params.get("frequency_deviation"),
        "start_frequency": params.get("start_frequency"),
        "end_frequency": params.get("end_frequency"),
        "duty_cycle": params.get("duty_cycle"),
        "chirp_sweeps": params.get("chirp_sweeps"),
        "benign_variant": params.get("benign_variant"),
        "source_duration_sec": source_duration_sec,
        "crop_offset_sec": crop_offset_sec,
        "crop_attempts": crop_attempts,
        "generation_group": generation_group,
    }

    return DatasetV2Sample(audio=chunk.astype(np.float32), features=feature_values, metadata=metadata)


def make_shared_dsp_components() -> Tuple[AudioPreprocessor, FeatureExtractor]:
    """One AudioPreprocessor + FeatureExtractor, reused across all samples
    (state is explicitly reset per sample in generate_sample()).

    Constructed with the SAME band/filter edges DSPPipeline uses (dsp/__init__
    single source of truth), not FeatureExtractor's own defaults -- those
    differ (ultrasonic_high defaults to 22000 in feature_extraction.py's
    signature but DSPPipeline overrides it to 21000). Using DSPPipeline's
    values here is what keeps Dataset V2 features comparable to what the
    live pipeline actually computes.
    """
    preprocessor = AudioPreprocessor(
        sample_rate=SAMPLE_RATE, filter_low=FILTER_LOW, filter_high=FILTER_HIGH
    )
    extractor = FeatureExtractor(
        sample_rate=SAMPLE_RATE,
        n_fft=CHUNK_SIZE,
        ultrasonic_low=ULTRASONIC_LOW,
        ultrasonic_high=ULTRASONIC_HIGH,
    )
    return preprocessor, extractor
