import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { LiveAudioBuffer } from "../../utils/liveAudioBuffer";
import { computeMagnitudeSpectrum } from "../../utils/spectrum";

export interface ThreatBandOverlay {
  startHz: number;
  endHz: number;
  carrierFreqs: number[];
}

interface LiveSpectrumProps {
  bufferRef: RefObject<LiveAudioBuffer>;
  sampleRateRef: RefObject<number>;
  active: boolean;
  height?: number;
  className?: string;
  minHz?: number;
  maxHz?: number;
  threatBand?: ThreatBandOverlay | null;
}

const FFT_SAMPLES = 4096;

function formatKhz(hz: number): string {
  return `${(hz / 1000).toFixed(1)} kHz`;
}

export default function LiveSpectrum({
  bufferRef,
  sampleRateRef,
  active,
  height = 190,
  className = "",
  minHz = 16000,
  maxHz = 24000,
  threatBand = null,
}: LiveSpectrumProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const scratchRef = useRef(new Float32Array(FFT_SAMPLES));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      context.clearRect(0, 0, w, h);

      const labelY = h - 6;
      context.fillStyle = "rgba(160,160,160,0.7)";
      context.font = "10px ui-monospace, monospace";
      context.textAlign = "left";
      context.fillText(formatKhz(minHz), 4, labelY);
      context.textAlign = "right";
      context.fillText(formatKhz(maxHz), w - 4, labelY);

      const buffer = bufferRef.current;
      const sampleRate = sampleRateRef.current || 48000;
      if (!active || !buffer || buffer.filledCount < 256) {
        context.fillStyle = "rgba(160,160,160,0.55)";
        context.font = "11px ui-monospace, monospace";
        context.textAlign = "center";
        context.fillText(
          active ? "Waiting for spectrum data…" : "NO LIVE SPECTRUM",
          w / 2,
          h / 2,
        );
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const scratch = scratchRef.current;
      buffer.copyRecent(scratch);
      const spectrum = computeMagnitudeSpectrum(scratch, sampleRate);
      if (!spectrum) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const bars = 96;
      const gap = 2;
      const barW = w / bars - gap;
      let maxMag = 1e-9;
      const barMagnitudes = new Float32Array(bars);

      for (let i = 0; i < bars; i += 1) {
        const t0 = i / bars;
        const t1 = (i + 1) / bars;
        const f0 = minHz + t0 * (maxHz - minHz);
        const f1 = minHz + t1 * (maxHz - minHz);
        let sum = 0;
        let count = 0;
        for (let b = 0; b < spectrum.frequencies.length; b += 1) {
          const f = spectrum.frequencies[b];
          if (f >= f0 && f < f1) {
            sum += spectrum.magnitudes[b];
            count += 1;
          }
        }
        const avg = count > 0 ? sum / count : 0;
        barMagnitudes[i] = avg;
        maxMag = Math.max(maxMag, avg);
      }

      const hzToX = (hz: number) => ((hz - minHz) / (maxHz - minHz)) * w;

      if (threatBand) {
        const x0 = hzToX(threatBand.startHz);
        const x1 = hzToX(threatBand.endHz);
        context.fillStyle = "rgba(216, 29, 41, 0.12)";
        context.fillRect(x0, 0, Math.max(1, x1 - x0), h - 14);
        context.fillStyle = "rgba(216, 29, 41, 0.85)";
        context.font = "10px ui-monospace, monospace";
        context.textAlign = "center";
        context.fillText("DETECTED REGION", (x0 + x1) / 2, 12);
      }

      for (let i = 0; i < bars; i += 1) {
        const norm = barMagnitudes[i] / maxMag;
        const barH = Math.max(2, norm * (h - 20));
        const x = i * (barW + gap);
        const y = h - 14 - barH;

        let color = "#8a8a8a";
        const fCenter = minHz + ((i + 0.5) / bars) * (maxHz - minHz);
        if (threatBand && fCenter >= threatBand.startHz && fCenter <= threatBand.endHz) {
          color = "#ff1e2d";
        } else if (threatBand?.carrierFreqs.some((cf) => Math.abs(cf - fCenter) < (maxHz - minHz) / bars)) {
          color = "#ffb000";
        }

        context.fillStyle = color;
        context.globalAlpha = color === "#8a8a8a" ? 0.55 : 0.95;
        context.fillRect(x, y, barW, barH);
        context.globalAlpha = 1;
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [active, bufferRef, maxHz, minHz, sampleRateRef, threatBand]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: "100%", height }}
      role="img"
      aria-label="Live ultrasonic frequency spectrum"
    />
  );
}
