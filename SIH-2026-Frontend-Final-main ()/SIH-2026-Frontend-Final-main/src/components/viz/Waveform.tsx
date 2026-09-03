import { useEffect, useRef } from "react";

type Mode = "normal" | "anomaly" | "threat" | "mixed";

interface WaveformProps { mode?: Mode; height?: number; active?: boolean; className?: string; }

const COLORS: Record<Exclude<Mode, "mixed">, string> = { normal: "#cfcfcf", anomaly: "#d29628", threat: "#d81d29" };

function signalSample(mode: Exclude<Mode, "mixed">, nx: number, time: number) {
  if (mode === "normal") {
    // Broadband ambient field: low, irregular pressure variations with no stable period.
    return (Math.sin(nx * 93 + time * 0.021) * 0.34 + Math.sin(nx * 157 - time * 0.013) * 0.2 + Math.sin(nx * 41 + time * 0.008) * 0.26) * 7;
  }
  if (mode === "anomaly") {
    // Beacon-like signal: a stable carrier presented in short, repeating bursts.
    const burst = Math.pow(Math.max(0, Math.sin(nx * 13.2 + time * 0.022)), 5);
    return Math.sin(nx * 58 + time * 0.052) * (3 + burst * 19) + Math.sin(nx * 116 + time * 0.052) * burst * 3;
  }
  // Rotating machinery: a fundamental with integer harmonics and slow load modulation.
  const load = 0.78 + 0.22 * Math.sin(nx * 3.8 + time * 0.01);
  return (Math.sin(nx * 30 + time * 0.026) * 13 + Math.sin(nx * 60 + time * 0.026) * 7 + Math.sin(nx * 90 + time * 0.026) * 3.5) * load;
}

export default function Waveform({ mode = "normal", height = 220, active = true, className = "" }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize(); window.addEventListener("resize", resize);

    const drawLine = (kind: Exclude<Mode, "mixed">, opacity: number) => {
      const rect = canvas.getBoundingClientRect();
      const points = 260; const step = rect.width / points; const middle = rect.height / 2;
      context.beginPath();
      for (let index = 0; index <= points; index += 1) {
        const nx = index / points;
        const x = index * step;
        const y = middle - signalSample(kind, nx, timeRef.current);
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      }
      context.globalAlpha = opacity;
      context.strokeStyle = COLORS[kind]; context.lineWidth = kind === "threat" ? 1.7 : 1.45;
      context.shadowColor = COLORS[kind]; context.shadowBlur = kind === "normal" ? 0 : 9;
      context.stroke(); context.shadowBlur = 0; context.globalAlpha = 1;
    };
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      context.clearRect(0, 0, rect.width, rect.height);
      context.beginPath(); context.moveTo(0, rect.height / 2); context.lineTo(rect.width, rect.height / 2);
      context.strokeStyle = "rgba(245,245,245,0.06)"; context.lineWidth = 1; context.stroke();
      if (mode === "mixed") { drawLine("normal", 0.45); drawLine("anomaly", 0.68); drawLine("threat", 0.82); }
      else drawLine(mode, 1);
      if (active) timeRef.current += 1;
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener("resize", resize); };
  }, [active, mode]);

  return <canvas ref={canvasRef} className={className} style={{ width: "100%", height }} role="img" aria-label={`${mode} signal waveform`} />;
}
