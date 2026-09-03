import { useEffect, useRef, useState } from "react";

interface RadialScoreProps {
  score: number; // 0-100
  size?: number;
  label?: string;
  sublabel?: string;
}

export default function RadialScore({ score, size = 220, label, sublabel }: RadialScoreProps) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          let start: number | null = null;
          const duration = 1400;
          const step = (ts: number) => {
            if (start === null) start = ts;
            const progress = Math.min(1, (ts - start) / duration);
            setDisplay(Math.round(progress * score));
            if (progress < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
          obs.disconnect();
        }
      },
      { threshold: 0.4 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [score]);

  const radius = size / 2 - 14;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - display / 100);
  const color = score >= 70 ? "#ff1e2d" : score >= 40 ? "#ffb000" : "#f5f5f5";

  return (
    <div ref={ref} className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#242424" strokeWidth={2} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={2.5}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 8px ${color}88)`, transition: "stroke 0.3s" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-5xl font-semibold" style={{ color }}>
          {display}
        </span>
        <span className="mono-label">/ 100</span>
        {label && <span className="mono-label mt-2" style={{ color }}>{label}</span>}
        {sublabel && <span className="mono-label mt-0.5 text-ink-3">{sublabel}</span>}
      </div>
    </div>
  );
}
