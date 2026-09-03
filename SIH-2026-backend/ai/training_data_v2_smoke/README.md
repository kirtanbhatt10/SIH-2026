# Synthetic Dataset V2

**SOURCE = SYNTHETIC.**

This dataset is generated synthetically through the project's
signal-generation and DSP pipeline (`dsp.SignalGenerator` ->
`dsp.AudioPreprocessor` -> `dsp.FeatureExtractor`). It is intended for
reproducible ML development and does not establish real-world / OTA
detection performance.

It is not a real recording, not an over-the-air capture, and not a
microphone recording.

Regenerate with:

    cd ai/ml/dataset_v2
    python build_dataset.py --samples-per-class 500 --seed 42

See `ai/ml/dataset_v2/README.md` (committed to git) for the full dataset
card: parameter ranges, metadata schema, and known limitations. This copy
exists for local convenience only -- this whole directory is gitignored,
same as `ai/training_data/`.
