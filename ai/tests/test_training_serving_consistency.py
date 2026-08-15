import os
import shutil
import numpy as np
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./dsp"))

try:
    from dsp.dsp_api import DSPPipeline
except ImportError:
    from dsp_api import DSPPipeline

print("==================================================")
print("  Training-Serving Consistency Verification")
print("==================================================")

pipeline = DSPPipeline(sample_rate=48000, n_fft=2048)
test_dir = "temp_consistency_dataset"

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

print("\n--- Generating Small Preprocessed Dataset (samples_per_class=3) ---")
X, y = pipeline.generate_training_dataset(
    output_dir=test_dir,
    samples_per_class=3
)

# Checks
five_classes_ok = len(np.unique(y)) == 5
cols_ok = X.shape[1] == 32
len_ok = len(X) == len(y) == 15
finite_ok = np.all(np.isfinite(X))

feat_file = os.path.join(test_dir, "features.npy")
label_file = os.path.join(test_dir, "labels.npy")
meta_file = os.path.join(test_dir, "metadata.json")
map_file = os.path.join(test_dir, "label_map.json")

files_exist = (
    os.path.exists(feat_file) and
    os.path.exists(label_file) and
    os.path.exists(meta_file) and
    os.path.exists(map_file)
)

print(f"  X.shape           : {X.shape}")
print(f"  y.shape           : {y.shape}")
print(f"  Unique classes    : {np.unique(y)} (count: {len(np.unique(y))})")
print(f"  All values finite : {finite_ok}")
print(f"  Files created     : {files_exist}")

assert five_classes_ok, "Expected 5 classes"
assert cols_ok, "Expected 32 columns in feature matrix"
assert len_ok, "Mismatch between X and y length"
assert finite_ok, "Non-finite feature value in preprocessed dataset"
assert files_exist, "Missing expected dataset artifact files"

# Clean up
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

print("\n[ALL TRAINING-SERVING CONSISTENCY CHECKS PASSED!]")
