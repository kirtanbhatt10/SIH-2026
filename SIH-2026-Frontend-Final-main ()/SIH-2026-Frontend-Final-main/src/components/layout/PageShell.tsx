import Reveal from "./Reveal";

interface PageShellProps {
  eyebrow: string;
  title: string;
  subtitle: string;
}

/** Minimal route shell for final-product areas that will receive functionality in later steps. */
export default function PageShell({ eyebrow, title, subtitle }: PageShellProps) {
  return (
    <div className="px-5 pb-24 pt-28 md:px-10">
      <div className="mx-auto max-w-[1600px]">
        <Reveal>
          <p className="mono-label mb-4">{eyebrow}</p>
          <h1 className="font-display text-5xl tracking-tight sm:text-6xl">{title}</h1>
          <p className="mt-4 max-w-xl text-ink-1">{subtitle}</p>
        </Reveal>
        <Reveal delay={100} className="mt-12 border border-line-2 bg-surface p-6">
          <p className="mono-label text-ink-3">FRONTEND ROUTE READY · FUNCTIONAL MODULE PENDING</p>
        </Reveal>
      </div>
    </div>
  );
}
