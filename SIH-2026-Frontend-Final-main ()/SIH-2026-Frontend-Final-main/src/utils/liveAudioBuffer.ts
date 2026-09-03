/** Bounded ring buffer for live microphone samples (source: WS /stream/audio). */
export class LiveAudioBuffer {
  private readonly storage: Float32Array;
  private writeIndex = 0;
  private filled = 0;
  readonly capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.storage = new Float32Array(capacity);
  }

  push(samples: Float32Array): void {
    for (let i = 0; i < samples.length; i += 1) {
      this.storage[this.writeIndex] = samples[i];
      this.writeIndex = (this.writeIndex + 1) % this.capacity;
      this.filled = Math.min(this.filled + 1, this.capacity);
    }
  }

  clear(): void {
    this.writeIndex = 0;
    this.filled = 0;
    this.storage.fill(0);
  }

  get filledCount(): number {
    return this.filled;
  }

  /** Copy the most recent `count` samples into `out` (oldest → newest). Returns samples copied. */
  copyRecent(out: Float32Array): number {
    const count = Math.min(out.length, this.filled);
    if (count === 0) {
      out.fill(0);
      return 0;
    }
    const start = (this.writeIndex - count + this.capacity) % this.capacity;
    if (start + count <= this.capacity) {
      out.set(this.storage.subarray(start, start + count));
    } else {
      const firstPart = this.capacity - start;
      out.set(this.storage.subarray(start), 0);
      out.set(this.storage.subarray(0, count - firstPart), firstPart);
    }
    return count;
  }
}

/** Decode backend AudioChunkMessage.samples_b64 (float32 PCM, little-endian). */
export function decodeBase64Float32(samplesB64: string): Float32Array {
  const binary = atob(samplesB64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
