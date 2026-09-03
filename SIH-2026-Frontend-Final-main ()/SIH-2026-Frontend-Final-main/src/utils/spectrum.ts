/** In-place radix-2 Cooley–Tukey FFT (real input → complex interleaved re,im). */
function fftInPlace(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  let j = 0;
  for (let i = 1; i < n; i += 1) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(ang);
    const wlenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let k = 0; k < len / 2; k += 1) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + len / 2] * wRe - im[i + k + len / 2] * wIm;
        const vIm = re[i + k + len / 2] * wIm + im[i + k + len / 2] * wRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nextWRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextWRe;
      }
    }
  }
}

function nextPowerOfTwo(value: number): number {
  let n = 1;
  while (n < value) n <<= 1;
  return n;
}

function applyHannWindow(samples: Float32Array): Float32Array {
  const out = new Float32Array(samples.length);
  const denom = samples.length - 1 || 1;
  for (let i = 0; i < samples.length; i += 1) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / denom));
    out[i] = samples[i] * w;
  }
  return out;
}

export interface SpectrumResult {
  /** Positive-frequency bin center (Hz). */
  frequencies: Float32Array;
  /** Linear magnitude per bin. */
  magnitudes: Float32Array;
  sampleRate: number;
}

/**
 * Compute magnitude spectrum from time-domain samples.
 * Source: live microphone samples (client-side FFT; backend does not stream FFT).
 */
export function computeMagnitudeSpectrum(
  samples: Float32Array,
  sampleRate: number,
): SpectrumResult | null {
  if (samples.length < 64) return null;

  const n = nextPowerOfTwo(Math.min(samples.length, 4096));
  const windowed = applyHannWindow(samples.subarray(samples.length - n));
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  re.set(windowed);

  fftInPlace(re, im);

  const half = n / 2;
  const frequencies = new Float32Array(half);
  const magnitudes = new Float32Array(half);
  for (let i = 0; i < half; i += 1) {
    frequencies[i] = (i * sampleRate) / n;
    magnitudes[i] = Math.hypot(re[i], im[i]) / n;
  }

  return { frequencies, magnitudes, sampleRate };
}
