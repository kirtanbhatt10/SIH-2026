import { useState } from "react";
import type { TeamMember } from "../../data/team";

function SocialLink({ href, label }: { href?: string; label: string }) {
  if (!href) return null;
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="mono-label border border-line-2 px-3 py-2 transition-colors hover:border-crimson hover:text-crimson"
    >
      {label} ↗
    </a>
  );
}

function initials(name: string) {
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0])
    .join("")
    .toUpperCase();
}

export default function TeamMemberCard({ member, featured = false }: { member: TeamMember; featured?: boolean }) {
  const [hover, setHover] = useState(false);
  const hasLinks = member.linkedin || member.github || member.portfolio || member.email;
  const baseScale = member.imageScale ?? 1;
  const hoverScale = baseScale * 1.02;

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className={`group relative flex flex-col border bg-surface transition-all duration-300 ${
        hover ? "-translate-y-1 border-crimson-3/60" : "border-line-2"
      } ${featured ? "lg:flex-row lg:items-stretch" : ""}`}
    >
      {/* corner detail */}
      <div className="mono-label flex items-center justify-between border-b border-line px-4 py-3 sm:px-5">
        <span>AS // PERSONNEL {member.node.replace("AS-", "")}</span>
        <span
          className={`h-1.5 w-1.5 rounded-full ${member.status === "active" ? "bg-system" : "bg-ink-3"} ${
            member.status === "active" ? "animate-pulse-slow" : ""
          }`}
        />
      </div>

      <div className={`flex flex-1 flex-col ${featured ? "lg:flex-row" : ""}`}>
        {/* portrait */}
        <div
          className={`relative shrink-0 overflow-hidden border-b border-line bg-void-2 ${
            featured
              ? "aspect-[4/5] sm:aspect-[3/4] lg:aspect-auto lg:w-[40%] lg:min-h-[320px] lg:self-stretch lg:border-b-0 lg:border-r"
              : "aspect-[4/5]"
          }`}
        >
          <div className="bg-grid-fine pointer-events-none absolute inset-0 z-[1] opacity-30" />
          {member.image ? (
            <>
              <img
                src={member.image}
                alt={member.name}
                className="absolute inset-0 h-full w-full object-cover transition-[filter,transform] duration-300"
                style={{
                  objectPosition: member.imagePosition ?? "center 30%",
                  transform: hover ? `scale(${hoverScale})` : `scale(${baseScale})`,
                  filter: hover ? "brightness(1.02) contrast(1.05)" : "brightness(0.92) contrast(1.03)",
                }}
              />
              <div className="pointer-events-none absolute inset-0 z-[2] bg-gradient-to-t from-void/55 via-void/15 to-transparent" />
            </>
          ) : (
            <span
              className={`font-display relative z-[3] flex h-full min-h-[200px] items-center justify-center text-5xl tracking-tight text-ink-2 ${featured ? "sm:text-7xl" : ""}`}
            >
              {initials(member.name)}
            </span>
          )}
          {/* scan line on hover */}
          <div
            className={`pointer-events-none absolute inset-x-0 z-[3] h-px bg-crimson transition-all duration-500 ${
              hover ? "top-full opacity-70" : "top-0 opacity-0"
            }`}
            style={{ boxShadow: "0 0 12px rgba(255,30,45,0.6)" }}
          />
          <div
            className={`pointer-events-none absolute inset-0 z-[3] border transition-colors duration-300 ${hover ? "border-crimson-3/40" : "border-transparent"}`}
          />
        </div>

        {/* details */}
        <div className="flex flex-1 flex-col px-5 py-6 sm:px-6">
          <h3 className={`font-display tracking-tight ${featured ? "text-3xl sm:text-4xl" : "text-2xl"}`}>{member.name}</h3>
          <div className={`mono-label mt-2 ${hover ? "text-crimson" : "text-ink-2"} transition-colors`}>{member.role}</div>

          <p className="mt-4 flex-1 text-sm text-ink-1">{member.description}</p>

          <div className="divider my-5" />

          <div className="mono-label mb-2">SPECIALIZATION</div>
          <div className="font-mono text-xs tracking-[0.08em] text-ink-1">{member.specialization.join(" • ")}</div>

          {hasLinks && (
            <div className="mt-5 flex flex-wrap gap-2">
              <SocialLink href={member.linkedin} label="LINKEDIN" />
              <SocialLink href={member.github} label="GITHUB" />
              <SocialLink href={member.portfolio} label="PORTFOLIO" />
              {member.email && (
                <a href={`mailto:${member.email}`} className="mono-label border border-line-2 px-3 py-2 transition-colors hover:border-crimson hover:text-crimson">
                  EMAIL ↗
                </a>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
