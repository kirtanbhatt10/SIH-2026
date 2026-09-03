import Reveal from "../components/layout/Reveal";
import Waveform from "../components/viz/Waveform";
import Pipeline from "../components/viz/Pipeline";
import TeamSection from "../components/team/TeamSection";

export default function About() {
  return (
    <div className="pt-28">
      <section className="px-5 pb-20 md:px-10">
        <div className="mx-auto max-w-[1600px]">
          <Reveal>
            <div className="mono-label mb-4">ABOUT THE PLATFORM</div>
            <h1 className="font-display max-w-4xl text-5xl leading-[1.02] tracking-tight sm:text-7xl">
              BUILT FOR THE SIGNALS <span className="text-crimson">YOU CAN'T AFFORD TO MISS.</span>
            </h1>
          </Reveal>
        </div>
      </section>

      {/* THE PROBLEM */}
      <section className="border-t border-line bg-void-2 px-5 py-24 md:px-10">
        <div className="mx-auto grid max-w-[1600px] gap-10 lg:grid-cols-[1fr_1.2fr]">
          <Reveal>
            <div className="mono-label mb-3">01</div>
            <h2 className="font-display text-4xl tracking-tight">THE PROBLEM</h2>
          </Reveal>
          <Reveal delay={100}>
            <p className="text-lg text-ink-1">
              Acoustic and RF environments are dense with activity — most of it harmless. Analysts
              watching raw spectrum data manually can't keep pace with the volume, and threats that hide
              in plain noise routinely go unnoticed until it's too late.
            </p>
          </Reveal>
        </div>
      </section>

      {/* THE SIGNAL */}
      <section className="border-t border-line px-5 py-24 md:px-10">
        <div className="mx-auto max-w-[1600px]">
          <Reveal>
            <div className="mono-label mb-3">02</div>
            <h2 className="font-display text-4xl tracking-tight">THE SIGNAL</h2>
            <p className="mt-4 max-w-2xl text-ink-1">
              Every signal carries structure — frequency, amplitude, duration, and repetition. Acoustic
              Shield captures that structure continuously and turns it into something an analyst can
              actually reason about.
            </p>
          </Reveal>
          <Reveal delay={150} className="mt-12">
            <Waveform mode="mixed" height={180} />
          </Reveal>
        </div>
      </section>

      {/* THE INTELLIGENCE */}
      <section className="border-t border-line bg-void-2 px-5 py-24 md:px-10">
        <div className="mx-auto grid max-w-[1600px] gap-10 lg:grid-cols-[1fr_1.2fr]">
          <Reveal>
            <div className="mono-label mb-3">03</div>
            <h2 className="font-display text-4xl tracking-tight">THE INTELLIGENCE</h2>
          </Reveal>
          <Reveal delay={100}>
            <p className="text-lg text-ink-1">
              Anomaly detection flags deviations from an environment's established baseline. Classification
              models then assess those deviations against known threat patterns, producing a confidence
              score and a plain-language explanation an analyst can act on — not just a red light.
            </p>
          </Reveal>
        </div>
      </section>

      {/* THE TECHNOLOGY */}
      <section className="border-t border-line px-5 py-24 md:px-10">
        <div className="mx-auto max-w-[1600px]">
          <Reveal>
            <div className="mono-label mb-3">04</div>
            <h2 className="font-display text-4xl tracking-tight">THE TECHNOLOGY</h2>
            <p className="mt-4 max-w-2xl text-ink-1">
              Signal data moves through a processing pipeline before it ever reaches a classifier —
              cleaned, normalized, and reduced to the features that actually matter.
            </p>
          </Reveal>
          <Reveal delay={150} className="mt-14">
            <Pipeline stages={["SIGNAL", "PROCESSING", "FEATURE EXTRACTION", "AI", "THREAT CLASSIFICATION"]} />
          </Reveal>
        </div>
      </section>

      {/* THE MISSION */}
      <section className="border-t border-line bg-void-2 px-5 py-28 text-center md:px-10">
        <div className="mx-auto max-w-3xl">
          <Reveal>
            <div className="mono-label mb-4">05 — THE MISSION</div>
            <h2 className="font-display text-4xl leading-[1.05] tracking-tight sm:text-5xl">
              GIVE ANALYSTS THE SIGNAL, <span className="text-amber">NOT THE NOISE.</span>
            </h2>
          </Reveal>
          <Reveal delay={150}>
            <p className="mt-6 text-ink-1">
              THE SILENT DOG'S WHISTLE exists to close the gap between raw spectrum data and a decision an operator
              can trust — validated continuously through controlled simulation, not assumption.
            </p>
          </Reveal>
        </div>
      </section>

      {/* THE TEAM */}
      <TeamSection />
    </div>
  );
}
