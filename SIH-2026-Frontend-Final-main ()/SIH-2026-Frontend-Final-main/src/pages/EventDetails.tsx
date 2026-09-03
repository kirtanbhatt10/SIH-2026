import { Link, useParams } from "react-router-dom";
import Reveal from "../components/layout/Reveal";
import RadialScore from "../components/viz/RadialScore";
import Spectrogram from "../components/viz/Spectrogram";
import Spectrum from "../components/viz/Spectrum";
import Waveform from "../components/viz/Waveform";
import { severityLabels, type Severity } from "../data/mock";
import { useFrontendSettings } from "../hooks/useFrontendSettings";
import { useThreatEvents } from "../hooks/useThreatEvents";

const riskMessages: Record<Severity, string> = {
  threat: "Potential covert acoustic communication detected",
  anomaly: "Suspicious acoustic activity detected",
  normal: "No communication-like threat pattern identified",
};
const riskStyles: Record<Severity, string> = {
  threat: "border-crimson-3/60 bg-crimson-3/10 text-crimson",
  anomaly: "border-amber/55 bg-amber/5 text-amber",
  normal: "border-system/50 bg-system/5 text-system",
};
const timeline = ["SIGNAL DETECTED", "FREQUENCY ANOMALY IDENTIFIED", "SIGNAL ANALYSIS", "CLASSIFICATION GENERATED", "THREAT ASSESSMENT", "EVENT LOGGED"];

export default function EventDetails() {
  const { settings } = useFrontendSettings();
  const { id } = useParams();
  const { events, status, error } = useThreatEvents();

  if (status === "loading") {
    return <div className="px-5 pb-24 pt-28 md:px-10"><div className="mx-auto max-w-[1600px]"><Reveal><div className="border border-line-2 bg-surface px-6 py-16 text-center text-sm text-ink-2">Loading event from backend…</div></Reveal></div></div>;
  }

  if (status === "error") {
    return <div className="px-5 pb-24 pt-28 md:px-10"><div className="mx-auto max-w-[1600px]"><Reveal><div className="border border-crimson-3/60 bg-crimson-3/5 px-6 py-16 text-center text-sm text-crimson" role="alert">{error ?? "Unable to load threat events from backend."}</div></Reveal></div></div>;
  }

  const index = events.findIndex((event) => event.id === id);
  const event = index >= 0 ? events[index] : undefined;

  if (!event) {
    return <div className="px-5 pb-24 pt-28 md:px-10"><div className="mx-auto max-w-[1600px]"><Reveal><div className="border border-line-2 bg-surface px-6 py-16 text-center"><p className="mono-label mb-4">BACKEND DATA</p><h1 className="font-display text-3xl tracking-tight sm:text-5xl">DETECTION EVENT NOT FOUND</h1><p className="mx-auto mt-4 max-w-lg text-ink-2">The requested acoustic event does not exist in the stored threat history.</p><Link to="/events" className="command-button mt-8 inline-block">BACK TO THREAT EVENTS <span>→</span></Link></div></Reveal></div></div>;
  }

  const previous = events[index - 1];
  const next = events[index + 1];
  const spectrum = event.severity === "threat" ? { threatIndex: Math.round(event.threatScore * 0.45), anomalyIndex: Math.round(event.source.confidence * 50) } : event.severity === "anomaly" ? { anomalyIndex: Math.round(event.source.confidence * 50) } : {};
  const metadata = [["Event ID", event.id], ["Timestamp", event.timestamp], ["Risk Level", severityLabels[event.severity]], ["Confidence", event.confidence], ["Classification", event.classification], ["Dominant Frequency", event.frequency], ["Frequency Range", `${event.frequencyStart} – ${event.frequencyEnd}`], ["Signal Strength", event.signalStrength], ["Duration", event.duration], ["Threat Score", `${event.threatScore} / 100`], ["Pattern", event.source.pattern], ["SNR", `${event.source.snr.toFixed(1)} dB`], ["Chunks Analyzed", String(event.source.chunks_analyzed)]];

  return <div className="px-5 pb-24 pt-28 md:px-10"><div className="mx-auto max-w-[1600px]">
    <Reveal><div className="flex flex-wrap items-end justify-between gap-5"><div><p className="mono-label mb-3">DETECTION DETAILS</p><h1 className="font-display text-5xl tracking-tight sm:text-6xl">EVENT {event.id}</h1></div><span className="border border-line-2 px-3 py-2 font-mono text-[10px] tracking-[0.14em] text-ink-2">LIVE BACKEND DATA</span></div></Reveal>

    <Reveal delay={80} className={`mt-10 border p-6 sm:p-8 ${riskStyles[event.severity]}`}><p className="mono-label mb-4">RISK ASSESSMENT · {severityLabels[event.severity]}</p><h2 className="font-display text-3xl tracking-tight sm:text-5xl">{riskMessages[event.severity]}</h2><p className="mt-4 max-w-2xl text-sm text-ink-1">This assessment is derived from the backend ThreatEvent record.</p></Reveal>

    <div className="mt-8 grid gap-8 xl:grid-cols-[1.2fr_.8fr]"><Reveal className="border border-line-2 bg-surface p-5 sm:p-6"><p className="mono-label mb-6">EVENT METADATA</p><dl className="grid gap-x-8 sm:grid-cols-2">{metadata.map(([label, value]) => <div key={label} className="flex justify-between gap-4 border-b border-line py-3 text-sm"><dt className="text-ink-2">{label}</dt><dd className="text-right text-ink-0">{value}</dd></div>)}</dl></Reveal><Reveal delay={100} className="flex min-h-80 flex-col items-center justify-center border border-line-2 bg-surface p-6"><p className="mono-label mb-6 self-start">EVENT THREAT SCORE</p><RadialScore score={event.threatScore} label={severityLabels[event.severity]} /><p className="mt-5 text-center text-sm text-ink-2">Score from backend suspicion_score (0–1 scaled to 0–100).</p></Reveal></div>

    {(settings.showSpectrum || settings.showSpectrogram) && <div className={`mt-8 grid xl:grid-cols-2 ${settings.visualizationDensity === "compact" ? "gap-4" : "gap-8"}`}>{settings.showSpectrum && <Reveal className="border border-line-2 bg-surface p-5 sm:p-6"><header className="mb-5 flex flex-wrap items-center justify-between gap-3"><p className="mono-label">FREQUENCY SPECTRUM</p><span className="mono-label text-ink-3">BACKEND-DRIVEN VISUALIZATION</span></header><Spectrum height={settings.visualizationDensity === "compact" ? 170 : 230} bars={56} {...spectrum} /></Reveal>}{settings.showSpectrogram && <Reveal delay={80} className="border border-line-2 bg-surface p-5 sm:p-6"><header className="mb-5 flex flex-wrap items-center justify-between gap-3"><p className="mono-label">SPECTROGRAM ANALYSIS</p><span className="mono-label text-ink-3">EVENT TIME-FREQUENCY CONTEXT</span></header><Spectrogram height={settings.visualizationDensity === "compact" ? 170 : 230} /></Reveal>}</div>}
    {settings.showWaveform && <Reveal delay={100} className="mt-8 border border-line-2 bg-surface p-5 sm:p-6"><header className="mb-5 flex flex-wrap items-center justify-between gap-3"><p className="mono-label">SIGNAL WAVEFORM</p><span className="mono-label text-ink-3">PROCEDURAL VISUALIZATION</span></header><Waveform mode={event.severity === "threat" ? "threat" : event.severity === "anomaly" ? "anomaly" : "normal"} active height={settings.visualizationDensity === "compact" ? 120 : 160} /></Reveal>}

    <div className="mt-12 grid gap-8 xl:grid-cols-[1fr_.85fr]"><Reveal className="border border-line-2 bg-surface p-5 sm:p-6"><p className="mono-label mb-6">SIGNAL CHARACTERISTICS</p><dl className="divide-y divide-line">{[["Frequency span", `${event.frequencyStart} – ${event.frequencyEnd}`], ["Signal profile", event.classification], ["Detection confidence", event.confidence], ["Risk assessment", severityLabels[event.severity]]].map(([label, value]) => <div key={label} className="flex justify-between gap-5 py-3 text-sm"><dt className="text-ink-2">{label}</dt><dd className="text-right text-ink-0">{value}</dd></div>)}</dl></Reveal><Reveal delay={100} className="border border-line-2 bg-surface p-5 sm:p-6"><p className="mono-label mb-6">ANALYSIS TIMELINE</p><ol className="space-y-3">{timeline.map((stage, stageIndex) => <li key={stage} className="flex items-center gap-4"><span className={`flex h-8 w-8 shrink-0 items-center justify-center border font-mono text-[10px] ${stageIndex === timeline.length - 1 ? riskStyles[event.severity] : "border-system/50 text-system"}`}>{String(stageIndex + 1).padStart(2, "0")}</span><span className="text-sm text-ink-1">{stage}</span></li>)}</ol></Reveal></div>

    <Reveal delay={120} className="mt-12 border-t border-line pt-10"><div className="flex flex-wrap gap-4"><Link to="/events" className="command-button">BACK TO THREAT EVENTS <span>→</span></Link><Link to="/monitor" className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">OPEN LIVE MONITOR →</Link><Link to="/simulator" className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">OPEN ATTACK SIMULATOR →</Link></div><div className="mt-8 flex flex-wrap gap-3">{previous ? <Link to={`/events/${previous.id}`} className="border border-line-2 px-4 py-3 font-mono text-[10px] tracking-[0.12em] text-ink-1 transition-colors hover:border-crimson hover:text-crimson">← PREVIOUS EVENT</Link> : <button type="button" disabled className="cursor-not-allowed border border-line px-4 py-3 font-mono text-[10px] tracking-[0.12em] text-ink-3">← PREVIOUS EVENT</button>}{next ? <Link to={`/events/${next.id}`} className="border border-line-2 px-4 py-3 font-mono text-[10px] tracking-[0.12em] text-ink-1 transition-colors hover:border-crimson hover:text-crimson">NEXT EVENT →</Link> : <button type="button" disabled className="cursor-not-allowed border border-line px-4 py-3 font-mono text-[10px] tracking-[0.12em] text-ink-3">NEXT EVENT →</button>}</div></Reveal>
  </div></div>;
}
