import { useEffect, useRef } from "react";

interface Marker {
  angle: number; // degrees
  distance: number; // 0-1 of radius
  severity: "anomaly" | "threat";
  label?: string;
}

interface RadarSweepProps {
  size?: number;
  className?: string;
  markers?: Marker[];
}

const DEFAULT_MARKERS: Marker[] = [
  { angle: 40, distance: 0.55, severity: "anomaly", label: "SIG-04" },
  { angle: 210, distance: 0.75, severity: "threat", label: "SIG-11" },
  { angle: 290, distance: 0.35, severity: "anomaly", label: "SIG-02" },
];

export default function RadarSweep({ size = 480, className = "", markers = DEFAULT_MARKERS }: RadarSweepProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const angleRef = useRef(0);

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
      const cx = w / 2;
      const cy = h / 2;
      const r = Math.min(w, h) / 2 - 8;

      ctx.clearRect(0, 0, w, h);

      // range rings
      for (let i = 1; i <= 4; i++) {
        ctx.beginPath();
        ctx.arc(cx, cy, (r * i) / 4, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(245,245,245,0.08)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // cross lines
      ctx.beginPath();
      ctx.moveTo(cx - r, cy);
      ctx.lineTo(cx + r, cy);
      ctx.moveTo(cx, cy - r);
      ctx.lineTo(cx, cy + r);
      ctx.strokeStyle = "rgba(245,245,245,0.06)";
      ctx.stroke();

      // degree ticks
      for (let deg = 0; deg < 360; deg += 30) {
        const rad = (deg * Math.PI) / 180;
        const x1 = cx + Math.cos(rad) * (r - 6);
        const y1 = cy + Math.sin(rad) * (r - 6);
        const x2 = cx + Math.cos(rad) * r;
        const y2 = cy + Math.sin(rad) * r;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = "rgba(245,245,245,0.15)";
        ctx.stroke();
      }

      // sweep gradient
      const angle = angleRef.current;
      const sweepWidth = 0.55; // radians of trailing fade
      const grad = ctx.createConicGradient
        ? ctx.createConicGradient(angle - sweepWidth, cx, cy)
        : null;
      if (grad) {
        grad.addColorStop(0, "rgba(245,245,245,0)");
        grad.addColorStop(0.85, "rgba(245,245,245,0)");
        grad.addColorStop(1, "rgba(245,245,245,0.22)");
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, angle - sweepWidth, angle);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();
      }

      // sweep line
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r);
      ctx.strokeStyle = "rgba(245,245,245,0.7)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // markers
      markers.forEach((m) => {
        const rad = (m.angle * Math.PI) / 180;
        const mx = cx + Math.cos(rad) * r * m.distance;
        const my = cy + Math.sin(rad) * r * m.distance;
        const color = m.severity === "threat" ? "#ff1e2d" : "#ffb000";

        // has the sweep passed recently? create a fade based on angular distance
        let diff = angle - rad;
        while (diff < 0) diff += Math.PI * 2;
        while (diff > Math.PI * 2) diff -= Math.PI * 2;
        const glow = Math.max(0, 1 - diff / 1.2);

        ctx.beginPath();
        ctx.arc(mx, my, 4 + glow * 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.55 + glow * 0.45;
        ctx.shadowColor = color;
        ctx.shadowBlur = 8 + glow * 12;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;

        if (m.label) {
          ctx.font = "10px 'IBM Plex Mono', monospace";
          ctx.fillStyle = "rgba(245,245,245,0.55)";
          ctx.fillText(m.label, mx + 8, my + 3);
        }
      });

      // center hub
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(245,245,245,0.6)";
      ctx.fill();

      angleRef.current += 0.006;
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [markers]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: "100%", height: size, aspectRatio: "1 / 1" }}
      role="img"
      aria-label="Radar sweep display"
    />
  );
}
