import Reveal from "../components/layout/Reveal";
import Spectrum from "../components/viz/Spectrum";
import Spectrogram from "../components/viz/Spectrogram";
import RadialScore from "../components/viz/RadialScore";
import Pipeline from "../components/viz/Pipeline";
import { aiReasoning, liveEvents, signalCharacteristics, threatScore } from "../data/mock";

function StatCell({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="border-l border-line-2 pl-4">
      <div className="mono-label mb-1">{label}</div>
      <div className={`font-display text-2xl tracking-tight ${accent ?? "text-ink-0"}`}>{value}</div>
    </div>
  );
}

export default function Intelligence() {
  return (
    <div className="px-5 pt-28 pb-24 md:px-10">
      <div className="mx-auto max-w-[1600px]">
        {/* HEADER */}
        <Reveal>
          <div className="mono-label mb-3">WORKSTATION</div>
          <div className="flex flex-wrap items-end justify-between gap-4">
            <h1 className="font-display text-5xl tracking-tight sm:text-6xl">INTELLIGENCE</h1>
            <div className="mono-label flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-crimson animate-pulse-slow" />
              SYSTEM · ACTIVE
            </div>
          </div>
          <p className="mt-3 text-ink-1">AI threat analysis for the current signal under review.</p>
        </Reveal>

        {/* TOP INTELLIGENCE BAR */}
        <Reveal delay={100} className="mt-12 grid grid-cols-2 gap-6 border-y border-line py-8 sm:grid-cols-4">
          <StatCell label="CURRENT THREAT" value="HIGH" accent="text-crimson" />
          <StatCell label="SIGNAL" value="2.41 GHz" />
          <StatCell label="CONFIDENCE" value="94%" accent="text-amber" />
          <StatCell label="RISK" value="87 / 100" accent="text-crimson" />
        </Reveal>

        {/* MAIN GRID */}
        <div className="mt-14 grid gap-8 lg:grid-cols-[1.4fr_1fr]">
          <div className="space-y-8">
            <Reveal className="border border-line-2 bg-surface p-6">
              <div className="mono-label mb-4 flex items-center justify-between">
                LIVE SPECTRUM
                <span className="text-crimson">● THREAT MARKER</span>
              </div>
              <Spectrum height={320} threatIndex={40} anomalyIndex={20} bars={56} />
            </Reveal>

            <Reveal delay={100} className="border border-line-2 bg-surface p-6">
              <div className="mono-label mb-4">SIGNAL HISTORY — SPECTROGRAM</div>
              <Spectrogram height={220} />
            </Reveal>

            <Reveal delay={150} className="border border-line-2 bg-surface p-6">
              <div className="mono-label mb-4">AI REASONING</div>
              <ul className="space-y-4">
                {aiReasoning.map((r, i) => (
                  <li key={i} className="flex gap-4 text-sm text-ink-1">
                    <span className="font-mono text-ink-3">{String(i + 1).padStart(2, "0")}</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </Reveal>
          </div>

          <div className="space-y-8">
            <Reveal className="border border-crimson-3/50 bg-surface p-6">
              <div className="mono-label mb-4">AI CLASSIFICATION</div>
              <div className="font-display text-2xl tracking-tight text-crimson">SUSPICIOUS SIGNAL</div>
              <div className="mt-6 grid grid-cols-2 gap-4">
                <StatCell label="CONFIDENCE" value="94%" />
                <StatCell label="RISK" value="HIGH" accent="text-crimson" />
                <StatCell label="PATTERN MATCH" value="ELEVATED" accent="text-amber" />
                <StatCell label="MODEL" value="ACTIVE" accent="text-system" />
              </div>
            </Reveal>

            <Reveal delay={100} className="flex flex-col items-center border border-line-2 bg-surface p-8">
              <div className="mono-label mb-6 self-start">THREAT SCORE</div>
              <RadialScore score={threatScore} label="HIGH RISK" />
            </Reveal>

            <Reveal delay={150} className="border border-line-2 bg-surface p-6">
              <div className="mono-label mb-4">SIGNAL CHARACTERISTICS</div>
              <div className="divide-y divide-line">
                {Object.entries(signalCharacteristics).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between py-2.5 text-sm">
                    <span className="capitalize text-ink-2">{k.replace(/([A-Z])/g, " $1")}</span>
                    <span className="font-mono text-ink-0">{v}</span>
                  </div>
                ))}
              </div>
            </Reveal>
          </div>
        </div>

        {/* THREAT TIMELINE */}
        <Reveal delay={100} className="mt-16 border-t border-line pt-14">
          <div className="mono-label mb-8">THREAT TIMELINE</div>
          <Pipeline
            direction="vertical"
            stages={["SIGNAL ACQUIRED", "PREPROCESSING", "ANOMALY", "FEATURE EXTRACTION", "CLASSIFICATION", "THREAT"]}
            activeIndex={5}
          />
        </Reveal>

        {/* LIVE EVENT STREAM */}
        <Reveal delay={100} className="mt-16 border-t border-line pt-14">
          <div className="mono-label mb-8">LIVE EVENT STREAM</div>
          <div className="hidden grid-cols-[100px_1fr_120px_100px] gap-4 border-b border-line pb-3 sm:grid">
            <span className="mono-label">TIME</span>
            <span className="mono-label">EVENT</span>
            <span className="mono-label">FREQUENCY</span>
            <span className="mono-label">STATUS</span>
          </div>
          <div className="divide-y divide-line">
            {liveEvents.map((e) => {
              const color =
                e.severity === "threat" ? "text-crimson" : e.severity === "anomaly" ? "text-amber" : "text-ink-1";
              return (
                <div key={e.id} className="grid grid-cols-2 gap-2 py-4 text-sm sm:grid-cols-[100px_1fr_120px_100px] sm:gap-4">
                  <span className="font-mono text-ink-3">{e.time}</span>
                  <span className="col-span-2 text-ink-0 sm:col-span-1">{e.label}</span>
                  <span className="font-mono text-ink-2">{e.frequency}</span>
                  <span className={`font-mono uppercase ${color}`}>{e.severity}</span>
                </div>
              );
            })}
          </div>
        </Reveal>
      </div>
    </div>
  );
}
