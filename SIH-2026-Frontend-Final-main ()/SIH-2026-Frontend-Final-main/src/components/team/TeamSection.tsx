import Reveal from "../layout/Reveal";
import TeamMemberCard from "./TeamMemberCard";
import { teamMembers } from "../../data/team";

export default function TeamSection() {
  const featured = teamMembers.find((m) => m.featured);
  const rest = teamMembers.filter((m) => !m.featured);

  return (
    <section className="relative overflow-hidden border-t border-line px-5 py-24 md:px-10">
      {/* faint continuation of the page's technical environment */}
      <div className="pointer-events-none absolute inset-0 opacity-[0.05]">
        <div className="absolute inset-0 bg-grid-fine" />
      </div>
      <div className="pointer-events-none absolute left-0 right-0 top-1/3 h-px bg-gradient-to-r from-transparent via-ink-2/20 to-transparent" />

      <div className="relative mx-auto max-w-[1600px]">
        <Reveal>
          <div className="mono-label mb-3 flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-system animate-pulse-slow" />
            AS // PERSONNEL &middot; PERSONNEL DATABASE ONLINE
          </div>
          <h2 className="font-display max-w-3xl text-4xl leading-[1.05] tracking-tight sm:text-5xl">
            THE PEOPLE <span className="text-crimson">BEHIND THE SIGNAL</span>
          </h2>
          <p className="mt-4 max-w-xl text-ink-1">
            THE SILENT DOG'S WHISTLE is built by a multidisciplinary team focused on acoustic intelligence,
            artificial intelligence, security, and intelligent systems.
          </p>
        </Reveal>

        <div className="mt-14 space-y-6">
          {featured && (
            <Reveal delay={100}>
              <TeamMemberCard member={featured} featured />
            </Reveal>
          )}

          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {rest.map((member, i) => (
              <Reveal key={member.id} delay={200 + i * 100}>
                <TeamMemberCard member={member} />
              </Reveal>
            ))}
          </div>
        </div>

        <Reveal delay={150} className="mt-16 border-t border-line pt-10 text-center">
          <p className="font-display text-2xl tracking-tight sm:text-3xl">
            DIFFERENT EXPERTISE. <span className="text-amber">ONE SIGNAL.</span>
          </p>
          <p className="mt-3 text-sm text-ink-2">
            Built by people who believe the signals we miss can matter the most.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
