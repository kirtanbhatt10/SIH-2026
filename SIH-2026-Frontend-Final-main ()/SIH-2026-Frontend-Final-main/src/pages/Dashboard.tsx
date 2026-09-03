import { Link } from "react-router-dom";
import Reveal from "../components/layout/Reveal";
import RadialScore from "../components/viz/RadialScore";
import Spectrogram from "../components/viz/Spectrogram";
import Spectrum from "../components/viz/Spectrum";
import { dashboardStatusCopy, severityLabels, type Severity } from "../data/mock";
import { useThreatEvents } from "../hooks/useThreatEvents";
import { severityToDashboardStatus } from "../services/adapter";

const riskStyles: Record<Severity, string> = {
  normal: "border-system/40 text-system",
  anomaly: "border-amber/50 text-amber",
  threat: "border-crimson-3/60 text-crimson",
};

const statusStyles = {
  safe: "border-system/50 bg-system/5 text-system",
  suspicious: "border-amber/50 bg-amber/5 text-amber",
  "high-risk": "border-crimson-3/60 bg-crimson-3/10 text-crimson",
};

export default function Dashboard() {
  const { events, current, status, error } = useThreatEvents();

  const dashboardStatus = current ? severityToDashboardStatus(current.severity) : "safe";
  const posture = dashboardStatusCopy[dashboardStatus];
  const recentEvents = [...events].reverse().slice(0, 5);

  const metrics = current
    ? [
        ["THREAT LEVEL", severityLabels[current.severity], current.severity === "threat" ? "text-crimson" : current.severity === "anomaly" ? "text-amber" : "text-system"],
        ["CONFIDENCE", current.confidence, "text-amber"],
        ["DOMINANT FREQUENCY", current.frequency, "text-ink-0"],
        ["SIGNAL STRENGTH", current.signalStrength, "text-ink-0"],
        ["DURATION", current.duration, "text-ink-0"],
        ["CLASSIFICATION", current.classification, "text-ink-0"],
      ] as const
    : [
        ["THREAT LEVEL", "—", "text-ink-3"],
        ["CONFIDENCE", "—", "text-ink-3"],
        ["DOMINANT FREQUENCY", "—", "text-ink-3"],
        ["SIGNAL STRENGTH", "—", "text-ink-3"],
        ["DURATION", "—", "text-ink-3"],
        ["CLASSIFICATION", "—", "text-ink-3"],
      ] as const;

  const spectrumProps = current
    ? {
        threatIndex: Math.round(current.threatScore * 0.45),
        anomalyIndex: Math.round(current.source.confidence * 50),
        bars: 56,
      }
    : { bars: 56 };

  const dataBadge =
    status === "loading"
      ? "LOADING BACKEND DATA"
      : status === "error"
        ? "BACKEND ERROR"
        : "LIVE BACKEND DATA";

  return (
    <div className="px-5 pb-24 pt-28 md:px-10">
      <div className="mx-auto max-w-[1600px]">
        <Reveal>
          <div className="flex flex-wrap items-end justify-between gap-5">
            <div><p className="mono-label mb-3">ACOUSTIC SHIELD / SECURITY OVERVIEW</p><h1 className="font-display text-5xl tracking-tight sm:text-6xl">DASHBOARD</h1></div>
            <span className={`border px-3 py-2 font-mono text-[10px] tracking-[0.14em] ${status === "error" ? "border-crimson-3/60 text-crimson" : "border-line-2 text-ink-2"}`}>{dataBadge}</span>
          </div>
        </Reveal>

        {status === "loading" && (
          <Reveal delay={60} className="mt-10 border border-line-2 bg-surface p-6 text-sm text-ink-2">
            Loading threat data from backend…
          </Reveal>
        )}

        {status === "error" && (
          <Reveal delay={60} className="mt-10 border border-crimson-3/60 bg-crimson-3/5 p-6">
            <p className="text-sm text-crimson" role="alert">{error ?? "Unable to reach backend at the configured API URL."}</p>
          </Reveal>
        )}

        {status === "success" && (
          <>
            <Reveal delay={80} className={`mt-10 border p-6 sm:p-8 ${statusStyles[dashboardStatus]}`}>
              <div className="flex flex-wrap items-start justify-between gap-6">
                <div><p className="mono-label mb-4">CURRENT SECURITY POSTURE</p><h2 className="font-display max-w-4xl text-3xl tracking-tight sm:text-5xl">{posture.label}</h2><p className="mt-4 max-w-2xl text-sm text-ink-1 sm:text-base">{posture.description}</p></div>
                <span className="border border-current px-3 py-2 font-mono text-[10px] tracking-[0.14em]">{current ? severityLabels[current.severity] : "SAFE"}</span>
              </div>
            </Reveal>

            <Reveal delay={120} className="mt-8 grid grid-cols-2 gap-px overflow-hidden border border-line bg-line sm:grid-cols-3 lg:grid-cols-6">
              {metrics.map(([label, value, accent]) => <article key={label} className="min-h-32 bg-surface p-4"><p className="mono-label mb-4">{label}</p><p className={`font-display text-xl leading-tight tracking-tight ${accent}`}>{value}</p></article>)}
            </Reveal>

            <div className="mt-8 grid gap-8 xl:grid-cols-[1.45fr_.85fr]">
              <div className="space-y-8">
                <Reveal className="border border-line-2 bg-surface p-5 sm:p-6"><header className="mb-5 flex flex-wrap items-center justify-between gap-3"><p className="mono-label">LIVE FREQUENCY SPECTRUM</p><span className="mono-label text-ink-3">{current ? "BACKEND-DRIVEN VISUALIZATION" : "AWAITING SIGNAL"}</span></header><Spectrum height={300} {...spectrumProps} /></Reveal>
                <Reveal delay={80} className="border border-line-2 bg-surface p-5 sm:p-6"><header className="mb-5 flex flex-wrap items-center justify-between gap-3"><p className="mono-label">SPECTROGRAM ANALYSIS</p><span className="mono-label text-ink-3">{current ? "CURRENT THREAT CONTEXT" : "NO ACTIVE THREAT"}</span></header><Spectrogram height={220} /></Reveal>
              </div>
              <div className="space-y-8">
                <Reveal delay={80} className="flex min-h-80 flex-col items-center justify-center border border-crimson-3/45 bg-surface p-6">
                  <p className="mono-label mb-6 self-start">THREAT SCORE</p>
                  {current ? (
                    <>
                      <RadialScore score={current.threatScore} label={severityLabels[current.severity]} />
                      <p className="mt-6 text-center text-sm text-ink-2">Score from backend suspicion_score (0–1 scaled to 0–100).</p>
                    </>
                  ) : (
                    <p className="text-sm text-ink-3">No current threat recorded.</p>
                  )}
                </Reveal>
                <Reveal delay={120} className="border border-line-2 bg-surface p-6">
                  <p className="mono-label mb-5">CURRENT ANALYSIS</p>
                  {current ? (
                    <dl className="divide-y divide-line">
                      <div className="flex justify-between gap-4 py-3 text-sm"><dt className="text-ink-2">Classification</dt><dd className="text-right text-ink-0">{current.classification}</dd></div>
                      <div className="flex justify-between gap-4 py-3 text-sm"><dt className="text-ink-2">Dominant frequency</dt><dd className="font-mono text-ink-0">{current.frequency}</dd></div>
                      <div className="flex justify-between gap-4 py-3 text-sm"><dt className="text-ink-2">Confidence</dt><dd className="font-mono text-amber">{current.confidence}</dd></div>
                    </dl>
                  ) : (
                    <p className="text-sm text-ink-3">No threats have been recorded yet.</p>
                  )}
                </Reveal>
              </div>
            </div>

            <Reveal delay={100} className="mt-12 border-t border-line pt-10">
              <div className="mb-6 flex flex-wrap items-end justify-between gap-4"><div><p className="mono-label mb-2">THREAT EVENT STREAM</p><h2 className="font-display text-3xl tracking-tight">RECENT THREAT EVENTS</h2></div><Link to="/events" className="border border-line-2 px-4 py-3 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">VIEW ALL EVENTS →</Link></div>
              {recentEvents.length === 0 ? (
                <div className="border border-line-2 bg-surface px-6 py-12 text-center text-sm text-ink-2">No threat events in backend history.</div>
              ) : (
                <div className="divide-y divide-line border-y border-line">
                  {recentEvents.map((event) => <Link key={event.id} to={`/events/${event.id}`} className="grid grid-cols-2 gap-x-4 gap-y-2 px-1 py-4 transition-colors hover:bg-surface sm:grid-cols-[120px_1fr_130px_120px_110px] sm:items-center sm:px-4"><span className="font-mono text-sm text-ink-3">{event.time}</span><span className="col-span-1 text-sm text-ink-0">{event.label}</span><span className="font-mono text-sm text-ink-2">{event.frequency}</span><span className={`w-fit border px-2 py-1 font-mono text-[10px] tracking-[0.1em] ${riskStyles[event.severity]}`}>{severityLabels[event.severity]}</span><span className="font-mono text-sm text-ink-1">{event.confidence}</span></Link>)}
                </div>
              )}
            </Reveal>
          </>
        )}

        <Reveal delay={120} className="mt-12 border-t border-line pt-10"><p className="mono-label mb-5">OPERATOR ROUTING</p><div className="flex flex-wrap gap-4"><Link to="/monitor" className="command-button">OPEN LIVE MONITOR <span>→</span></Link><Link to="/events" className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">VIEW THREAT EVENTS →</Link><Link to="/simulator" className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">OPEN ATTACK SIMULATOR →</Link></div></Reveal>
      </div>
    </div>
  );
}
