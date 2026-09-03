import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import Reveal from "../components/layout/Reveal";
import { severityLabels, type Severity } from "../data/mock";
import { useFrontendSettings } from "../hooks/useFrontendSettings";
import { useThreatEvents } from "../hooks/useThreatEvents";

type Filter = "all" | Severity;

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "ALL" }, { key: "threat", label: "HIGH RISK" }, { key: "anomaly", label: "SUSPICIOUS" }, { key: "normal", label: "SAFE" },
];
const RISK_STYLE: Record<Severity, string> = { normal: "border-system/50 text-system", anomaly: "border-amber/55 text-amber", threat: "border-crimson-3/60 text-crimson" };

export default function Events() {
  const { settings } = useFrontendSettings();
  const { events, status, error } = useThreatEvents();
  const [filter, setFilter] = useState<Filter>("all");

  const filteredEvents = useMemo(
    () => (filter === "all" ? events : events.filter((event) => event.severity === filter)),
    [events, filter],
  );

  const summary = useMemo(
    () => ({
      total: events.length,
      threat: events.filter((event) => event.severity === "threat").length,
      anomaly: events.filter((event) => event.severity === "anomaly").length,
      normal: events.filter((event) => event.severity === "normal").length,
    }),
    [events],
  );

  const summaryCards = [["TOTAL EVENTS", summary.total, "text-ink-0"], ["HIGH RISK", summary.threat, "text-crimson"], ["SUSPICIOUS", summary.anomaly, "text-amber"], ["SAFE / NORMAL", summary.normal, "text-system"]] as const;

  const dataBadge =
    status === "loading" ? "LOADING BACKEND DATA" : status === "error" ? "BACKEND ERROR" : "LIVE BACKEND DATA";

  return <div className="px-5 pb-24 pt-28 md:px-10"><div className="mx-auto max-w-[1600px]">
    <Reveal><div className="flex flex-wrap items-end justify-between gap-5"><div><p className="mono-label mb-3">ACOUSTIC DETECTION HISTORY</p><h1 className="font-display text-5xl tracking-tight sm:text-6xl">THREAT EVENTS</h1><p className="mt-3 text-ink-1">Acoustic detection history</p></div><span className={`border px-3 py-2 font-mono text-[10px] tracking-[0.14em] ${status === "error" ? "border-crimson-3/60 text-crimson" : "border-line-2 text-ink-2"}`}>{dataBadge}</span></div></Reveal>

    {status === "loading" && (
      <Reveal delay={60} className="mt-10 border border-line-2 bg-surface p-6 text-sm text-ink-2">
        Loading threat events from backend…
      </Reveal>
    )}

    {status === "error" && (
      <Reveal delay={60} className="mt-10 border border-crimson-3/60 bg-crimson-3/5 p-6">
        <p className="text-sm text-crimson" role="alert">{error ?? "Unable to reach backend at the configured API URL."}</p>
      </Reveal>
    )}

    {status === "success" && (
      <>
        <Reveal delay={80} className="mt-10 grid grid-cols-2 gap-px overflow-hidden border border-line bg-line sm:grid-cols-4">{summaryCards.map(([label, value, accent]) => <article key={label} className="bg-surface p-5"><p className="mono-label mb-4">{label}</p><p className={`font-display text-3xl tracking-tight ${accent}`}>{value}</p></article>)}</Reveal>

        <Reveal delay={110} className="mt-10"><div className="flex flex-wrap items-center justify-between gap-4"><div><p className="mono-label mb-3">EVENT FILTER</p><div className="flex flex-wrap gap-2" role="group" aria-label="Filter detection events">{FILTERS.map((item) => <button type="button" key={item.key} onClick={() => setFilter(item.key)} aria-pressed={filter === item.key} className={`border px-3 py-2 font-mono text-[10px] tracking-[0.12em] transition-colors ${filter === item.key ? "border-crimson-3 bg-crimson-3/10 text-crimson" : "border-line-2 text-ink-2 hover:border-crimson hover:text-crimson"}`}>{item.label}</button>)}</div></div><p className="mono-label text-ink-3">{filteredEvents.length} RECORD{filteredEvents.length === 1 ? "" : "S"} SHOWN</p></div></Reveal>

        <Reveal delay={130} className="mt-8">{events.length === 0 ? <div className="border border-line-2 bg-surface px-6 py-16 text-center"><p className="font-display text-2xl tracking-tight">NO THREAT EVENTS RECORDED</p><p className="mt-3 text-sm text-ink-2">The backend has not stored any threat events yet.</p></div> : filteredEvents.length === 0 ? <div className="border border-line-2 bg-surface px-6 py-16 text-center"><p className="font-display text-2xl tracking-tight">NO EVENTS FOUND</p><p className="mt-3 text-sm text-ink-2">No detections match the selected filter.</p><button type="button" onClick={() => setFilter("all")} className="mt-7 border border-line-2 px-4 py-3 font-mono text-[11px] tracking-[0.12em] transition-colors hover:border-crimson hover:text-crimson">CLEAR FILTER</button></div> : <><div className="hidden border border-line-2 bg-surface md:block"><div className={`grid grid-cols-[130px_175px_125px_110px_1fr_120px_120px] gap-4 border-b border-line px-5 ${settings.compactEvents ? "py-2" : "py-4"}`}><span className="mono-label">EVENT ID</span><span className="mono-label">TIMESTAMP</span><span className="mono-label">FREQUENCY</span><span className="mono-label">CONFIDENCE</span><span className="mono-label">CLASSIFICATION</span><span className="mono-label">RISK</span><span className="mono-label">ACTION</span></div><div className="divide-y divide-line">{filteredEvents.map((event) => <div key={event.id} className={`grid grid-cols-[130px_175px_125px_110px_1fr_120px_120px] items-center gap-4 px-5 text-sm transition-colors hover:bg-void-2 ${settings.compactEvents ? "py-2" : "py-4"}`}><span className="font-mono text-ink-0">{event.id}</span><time className="font-mono text-ink-3">{event.timestamp}</time><span className="font-mono text-ink-2">{event.frequency}</span><span className="font-mono text-ink-1">{event.confidence}</span><span className="text-ink-1">{event.classification}</span><span className={`w-fit border px-2 py-1 font-mono text-[10px] tracking-[0.1em] ${RISK_STYLE[event.severity]}`}>{severityLabels[event.severity]}</span><Link to={`/events/${event.id}`} className="font-mono text-[10px] tracking-[0.1em] text-ink-1 transition-colors hover:text-crimson">VIEW DETAILS →</Link></div>)}</div></div><div className="grid gap-4 md:hidden">{filteredEvents.map((event) => <article key={event.id} className={`border border-line-2 bg-surface ${settings.compactEvents ? "p-4" : "p-5"}`}><div className="flex items-start justify-between gap-4"><div><p className="font-mono text-sm text-ink-0">{event.id}</p><time className="mono-label mt-2 block text-ink-3">{event.timestamp}</time></div><span className={`border px-2 py-1 font-mono text-[10px] tracking-[0.1em] ${RISK_STYLE[event.severity]}`}>{severityLabels[event.severity]}</span></div><p className="mt-5 text-sm text-ink-1">{event.classification}</p><dl className="mt-5 grid grid-cols-2 gap-4 border-t border-line pt-4"><div><dt className="mono-label mb-1">FREQUENCY</dt><dd className="font-mono text-sm text-ink-0">{event.frequency}</dd></div><div><dt className="mono-label mb-1">CONFIDENCE</dt><dd className="font-mono text-sm text-ink-0">{event.confidence}</dd></div></dl><Link to={`/events/${event.id}`} className="mt-6 inline-block font-mono text-[11px] tracking-[0.12em] text-ink-1 transition-colors hover:text-crimson">VIEW DETAILS →</Link></article>)}</div></>}</Reveal>
      </>
    )}

    <Reveal delay={150} className="mt-12 border-t border-line pt-10"><p className="mono-label mb-5">OPERATOR ROUTING</p><div className="flex flex-wrap gap-4"><Link to="/monitor" className="command-button">OPEN LIVE MONITOR <span>→</span></Link><Link to="/simulator" className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">OPEN ATTACK SIMULATOR →</Link></div></Reveal>
  </div></div>;
}
