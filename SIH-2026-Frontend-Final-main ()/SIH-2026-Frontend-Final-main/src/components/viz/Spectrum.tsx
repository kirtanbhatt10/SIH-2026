import { useEffect, useRef } from "react";

interface SpectrumProps {
  height?: number;
  bars?: number;
  className?: string;
  threatIndex?: number; // index of bar to mark as threat
  anomalyIndex?: number;
}

export default function Spectrum({ height = 180, bars = 64, className = "", threatIndex, anomalyIndex }: SpectrumProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tRef = useRef(0);
  const seedsRef = useRef<number[]>(Array.from({ length: bars }, () => Math.random() * 1000));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      const gap = 3;
      const barW = w / bars - gap;
      const t = tRef.current;

      for (let i = 0; i < bars; i++) {
        const seed = seedsRef.current[i];
        const base = (Math.sin(i * 0.4) + 1) / 2;
        const wobble = (Math.sin(t * 0.03 + seed) + 1) / 2;
        let magnitude = 0.15 + base * 0.5 + wobble * 0.3;

        let color = "#8a8a8a";
        if (i === threatIndex) {
          magnitude = 0.85 + Math.sin(t * 0.15) * 0.1;
          color = "#ff1e2d";
        } else if (i === anomalyIndex) {
          magnitude = 0.6 + Math.sin(t * 0.1) * 0.1;
          color = "#ffb000";
        }

        const barH = Math.max(2, magnitude * h);
        const x = i * (barW + gap);
        const y = h - barH;

        ctx.fillStyle = color;
        ctx.globalAlpha = i === threatIndex || i === anomalyIndex ? 1 : 0.55;
        ctx.fillRect(x, y, barW, barH);
        ctx.globalAlpha = 1;
      }

      tRef.current += 1;
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [bars, threatIndex, anomalyIndex]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: "100%", height }}
      role="img"
      aria-label="Frequency spectrum"
    />
  );
}
