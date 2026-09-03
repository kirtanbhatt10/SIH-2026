import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { LiveAudioBuffer } from "../../utils/liveAudioBuffer";

interface LiveWaveformProps {
  bufferRef: RefObject<LiveAudioBuffer>;
  active: boolean;
  height?: number;
  className?: string;
}

const DISPLAY_SAMPLES = 2048;

export default function LiveWaveform({ bufferRef, active, height = 190, className = "" }: LiveWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const scratchRef = useRef(new Float32Array(DISPLAY_SAMPLES));

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

      context.beginPath();
      context.moveTo(0, h / 2);
      context.lineTo(w, h / 2);
      context.strokeStyle = "rgba(245,245,245,0.08)";
      context.lineWidth = 1;
      context.stroke();

      const buffer = bufferRef.current;
      if (!active || !buffer || buffer.filledCount === 0) {
        context.fillStyle = "rgba(160,160,160,0.55)";
        context.font = "11px ui-monospace, monospace";
        context.textAlign = "center";
        context.fillText(
          active ? "Waiting for microphone data…" : "NO LIVE SIGNAL",
          w / 2,
          h / 2,
        );
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const scratch = scratchRef.current;
      const copied = buffer.copyRecent(scratch);
      const points = Math.min(copied, w);
      const step = copied / points;
      let peak = 0;
      for (let i = 0; i < copied; i += 1) {
        peak = Math.max(peak, Math.abs(scratch[i]));
      }
      const scale = peak > 0 ? (h * 0.42) / peak : h * 0.05;

      context.beginPath();
      for (let i = 0; i < points; i += 1) {
        const sampleIndex = Math.min(copied - 1, Math.floor(i * step));
        const x = (i / (points - 1 || 1)) * w;
        const y = h / 2 - scratch[sampleIndex] * scale;
        if (i === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.strokeStyle = "#39d98a";
      context.lineWidth = 1.5;
      context.shadowColor = "#39d98a";
      context.shadowBlur = 6;
      context.stroke();
      context.shadowBlur = 0;

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [active, bufferRef]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: "100%", height }}
      role="img"
      aria-label="Live microphone waveform"
    />
  );
}
