import { useEffect, useRef } from "react";

// Ambient, subtle canvas layer: radial rings + drifting nodes.
// Sits behind page content as pure atmosphere — never competes with foreground data.
export default function FieldBackground({ className = "" }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const nodes = Array.from({ length: 14 }, () => ({
      x: Math.random(),
      y: Math.random(),
      r: Math.random() * 1.4 + 0.6,
      phase: Math.random() * Math.PI * 2,
    }));

    let t = 0;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      // radial acoustic rings, centered upper-right-ish for asymmetry
      const cx = w * 0.72;
      const cy = h * 0.38;
      for (let i = 1; i <= 6; i++) {
        const rad = i * (Math.max(w, h) * 0.09) + Math.sin(t * 0.004 + i) * 4;
        ctx.beginPath();
        ctx.arc(cx, cy, rad, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(245,245,245,${0.025 - i * 0.002})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // drifting sensor nodes
      nodes.forEach((n) => {
        const x = n.x * w;
        const y = n.y * h + Math.sin(t * 0.002 + n.phase) * 6;
        ctx.beginPath();
        ctx.arc(x, y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(245,245,245,0.14)";
        ctx.fill();
      });

      t += 1;
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`}>
      <div className="absolute inset-0 bg-grid-fine opacity-60" />
      <div className="absolute inset-0 bg-grid-major" />
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 60% 50% at 72% 30%, rgba(255,30,45,0.05), transparent 60%), radial-gradient(ellipse 50% 40% at 15% 80%, rgba(255,176,0,0.04), transparent 60%)",
        }}
      />
    </div>
  );
}
