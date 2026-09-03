import Reveal from "../layout/Reveal";

interface PipelineProps {
  stages: string[];
  direction?: "horizontal" | "vertical";
  activeIndex?: number;
}

export default function Pipeline({ stages, direction = "horizontal", activeIndex }: PipelineProps) {
  if (direction === "vertical") {
    return (
      <div className="flex flex-col">
        {stages.map((s, i) => (
          <Reveal key={s} delay={i * 80}>
            <div className="flex items-center gap-4">
              <div
                className={`flex h-10 w-10 shrink-0 items-center justify-center border font-mono text-xs ${
                  activeIndex !== undefined && i <= activeIndex
                    ? "border-crimson text-crimson"
                    : "border-line-2 text-ink-2"
                }`}
              >
                {String(i + 1).padStart(2, "0")}
              </div>
              <div className="font-display py-3 text-lg tracking-tight">{s}</div>
            </div>
            {i < stages.length - 1 && <div className="ml-5 h-8 w-px bg-line-2" />}
          </Reveal>
        ))}
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-6">
      {stages.map((s, i) => (
        <Reveal key={s} delay={i * 70} className="flex items-center gap-3">
          <div
            className={`border px-4 py-3 font-mono text-xs tracking-[0.14em] ${
              activeIndex !== undefined && i <= activeIndex
                ? "border-crimson text-crimson"
                : "border-line-2 text-ink-1"
            }`}
          >
            {s}
          </div>
          {i < stages.length - 1 && <span className="text-ink-3">→</span>}
        </Reveal>
      ))}
    </div>
  );
}
