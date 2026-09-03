import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt

TX = r"generated_payloads\7fffa953.wav"
CAP = r"forensic_captures\pc2_capture_20260816_232524.wav"

BS = 2400  # 50 ms at 48 kHz

tx, fs = sf.read(TX)
cap, fs2 = sf.read(CAP)

tx = tx[:, 0] if tx.ndim > 1 else tx
cap = cap[:, 0] if cap.ndim > 1 else cap

assert fs == fs2 == 48000

f18 = butter(4, [18000, 19000], btype="bandpass", fs=fs, output="sos")
f20 = butter(4, [20000, 21000], btype="bandpass", fs=fs, output="sos")

tx18 = sosfilt(f18, tx)
tx20 = sosfilt(f20, tx)

cap18 = sosfilt(f18, cap)
cap20 = sosfilt(f20, cap)


def scores(x18, x20):
    n = (len(x18) // BS) * BS

    x18 = x18[:n].reshape(-1, BS)
    x20 = x20[:n].reshape(-1, BS)

    e18 = np.mean(x18 * x18, axis=1)
    e20 = np.mean(x20 * x20, axis=1)

    return (e20 - e18) / (e20 + e18 + 1e-30)


txs = scores(tx18, tx20)
caps = scores(cap18, cap20)

N = len(txs)

print("TX symbols:", N)
print("CAP windows:", len(caps))
print()

best_corr = -2
best_start = None

# Search every 5 ms alignment.
STEP = 240

for start in range(0, len(cap) - N * BS + 1, STEP):

    end = start + N * BS

    c18 = cap18[start:end]
    c20 = cap20[start:end]

    cs = scores(c18, c20)

    if len(cs) != N:
        continue

    if np.std(cs) < 1e-12:
        continue

    corr = np.corrcoef(txs, cs)[0, 1]

    if np.isfinite(corr) and corr > best_corr:
        best_corr = corr
        best_start = start


print("BEST CORRELATION:", best_corr)

if best_start is not None:
    print("START SAMPLE:", best_start)
    print("START TIME:", best_start / fs, "sec")
    print("END TIME:", (best_start + N * BS) / fs, "sec")

    c18 = cap18[best_start:best_start + N * BS]
    c20 = cap20[best_start:best_start + N * BS]

    cs = scores(c18, c20)

    tx_bits = (txs > 0).astype(int)
    cap_bits = (cs > 0).astype(int)

    print()
    print("TX bits:")
    print("".join(map(str, tx_bits)))

    print()
    print("CAP bits:")
    print("".join(map(str, cap_bits)))

    print()
    print("BIT MATCH:",
          np.mean(tx_bits == cap_bits) * 100, "%")

else:
    print("No valid alignment found.")