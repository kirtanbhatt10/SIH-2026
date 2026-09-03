import { useEffect, useRef } from "react";

interface SpectrogramProps {
  height?: number;
  className?: string;
}

// Maps 0-1 intensity to a black -> gray -> amber -> crimson ramp (no blue/purple)
function rampColor(v: number): [number, number, number] {
  if (v < 0.35) {
    const k = v / 0.35;
    return [13 + k * 20, 13 + k * 20, 13 + k * 20];
  }
  if (v < 0.7) {
    const k = (v - 0.35) / 0.35;
    return [33 + k * (255 - 33), 33 + k * (176 - 33), 33 + k * (0 - 33)];
  }
  const k = (v - 0.7) / 0.3;
  return [255 - k * (255 - 255), 176 - k * 176, 0 + k * 45];
}

export default function Spectrogram({ height = 220, className = "" }: SpectrogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rows = 80; // frequency bins
    const cols = 220; // time steps
    const offscreen = document.createElement("canvas");
    offscreen.width = cols;
    offscreen.height = rows;
    offscreenRef.current = offscreen;
    const octx = offscreen.getContext("2d")!;
    const img = octx.createImageData(cols, rows);

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    let col = 0;
    const columnBuffer: number[][] = [];

    const genColumn = (c: number) => {
      const arr: number[] = [];
      for (let r = 0; r < rows; r++) {
        const base = Math.sin(r * 0.15 + c * 0.05) * 0.5 + 0.5;
        const noise = Math.random() * 0.25;
        let v = base * 0.35 + noise;
        // occasional anomaly/threat band
        if (Math.abs(r - (rows * 0.3 + Math.sin(c * 0.02) * 6)) < 2 && Math.random() > 0.4) {
          v = 0.65 + Math.random() * 0.2;
        }
        if (Math.abs(r - rows * 0.62) < 1.4 && (c % 90) > 70 && (c % 90) < 85) {
          v = 0.9 + Math.random() * 0.1;
        }
        arr.push(Math.min(1, v));
      }
      return arr;
    };

    for (let c = 0; c < cols; c++) columnBuffer.push(genColumn(c));

    const drawOffscreen = () => {
      for (let c = 0; c < cols; c++) {
        const arr = columnBuffer[c];
        for (let r = 0; r < rows; r++) {
          const v = arr[r];
          const [rr, gg, bb] = rampColor(v);
          const idx = (r * cols + c) * 4;
          img.data[idx] = rr;
          img.data[idx + 1] = gg;
          img.data[idx + 2] = bb;
          img.data[idx + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
    };
    drawOffscreen();

    let frame = 0;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;

      if (frame % 6 === 0) {
        columnBuffer.shift();
        columnBuffer.push(genColumn(col));
        col++;
        drawOffscreen();
      }
      frame++;

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(offscreen, 0, 0, cols, rows, 0, 0, w, h);

      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: "100%", height, imageRendering: "pixelated" }}
      role="img"
      aria-label="Signal spectrogram history"
    />
  );
}
