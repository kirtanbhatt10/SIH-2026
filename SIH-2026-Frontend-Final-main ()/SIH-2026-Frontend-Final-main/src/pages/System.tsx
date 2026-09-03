import Reveal from "../components/layout/Reveal";
import { systemDemoStatus, type DemoReadiness } from "../data/mock";
import { useSystemStatus } from "../hooks/useSystemStatus";

const STATUS_STYLE: Record<DemoReadiness, string> = { READY: "text-system border-system/50", ACTIVE: "text-ink-0 border-ink-2", AVAILABLE: "text-amber border-amber/50", LOADED: "text-system border-system/50", PENDING: "text-ink-3 border-line-2", "NOT CONNECTED": "text-ink-2 border-line-2" };

export default function System() {
  const { data, status, error } = useSystemStatus();

  const backendReadiness: DemoReadiness = status === "online" ? "ACTIVE" : status === "loading" ? "PENDING" : "NOT CONNECTED";

  const readinessRows = systemDemoStatus.readiness.map(([feature, rowStatus]) =>
    feature === "Backend Integration" ? [feature, backendReadiness] as const : [feature, rowStatus] as const,
  );

  return <div className="px-5 pb-24 pt-28 md:px-10"><div className="mx-auto max-w-[1600px]">
    <Reveal><div className="flex flex-wrap items-end justify-between gap-5"><div><p className="mono-label mb-3">PLATFORM DIAGNOSTICS AND MODULE READINESS</p><h1 className="font-display text-5xl tracking-tight sm:text-6xl">SYSTEM STATUS</h1></div><span className={`border px-3 py-2 font-mono text-[10px] tracking-[0.14em] ${status === "error" ? "border-crimson-3/60 text-crimson" : "border-line-2 text-ink-2"}`}>{status === "loading" ? "CHECKING BACKEND…" : status === "online" ? "BACKEND ONLINE" : "BACKEND OFFLINE"}</span></div></Reveal>

    <Reveal delay={70} className={`mt-10 border p-6 sm:p-8 ${status === "online" ? "border-system/40 bg-system/5" : status === "error" ? "border-crimson-3/60 bg-crimson-3/5" : "border-line-2 bg-surface"}`}>
      <p className={`mono-label mb-4 ${status === "online" ? "text-system" : status === "error" ? "text-crimson" : "text-ink-2"}`}>BACKEND API STATUS</p>
      {status === "loading" && <p className="text-sm text-ink-2">Contacting GET /api/system-status…</p>}
      {status === "error" && <p className="text-sm text-crimson" role="alert">{error ?? "Backend is offline or unreachable."}</p>}
      {status === "online" && data && (
        <>
          <h2 className="font-display text-3xl tracking-tight sm:text-5xl">{data.status.toUpperCase()}</h2>
          <dl className="mt-6 grid gap-3 sm:grid-cols-3">
            <div><dt className="mono-label mb-1">SERVICE</dt><dd className="text-sm text-ink-0">{data.service}</dd></div>
            <div><dt className="mono-label mb-1">VERSION</dt><dd className="font-mono text-sm text-ink-0">{data.version}</dd></div>
            <div><dt className="mono-label mb-1">ENDPOINT</dt><dd className="font-mono text-sm text-ink-0">GET /api/system-status</dd></div>
          </dl>
        </>
      )}
    </Reveal>

    <Reveal delay={80} className="mt-10 border border-system/40 bg-system/5 p-6 sm:p-8"><p className="mono-label mb-4 text-system">FRONTEND PRESENTATION STATE</p><h2 className="font-display text-3xl tracking-tight sm:text-5xl">{systemDemoStatus.overall.label}</h2><p className="mt-4 max-w-2xl text-ink-1">{systemDemoStatus.overall.description}</p></Reveal>

    <Reveal delay={100} className="mt-12"><p className="mono-label mb-5">FRONTEND MODULE STATUS</p><div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{systemDemoStatus.modules.map((module) => <article key={module.id} className="border border-line-2 bg-surface p-5"><div className="flex items-start justify-between gap-4"><h2 className="font-display text-xl tracking-tight">{module.name}</h2><span className={`shrink-0 border px-2 py-1 font-mono text-[9px] tracking-[0.1em] ${STATUS_STYLE[module.status]}`}>{module.status}</span></div><p className="mt-4 text-sm leading-relaxed text-ink-2">{module.description}</p></article>)}</div></Reveal>

    <div className="mt-12 grid gap-8 xl:grid-cols-[.9fr_1.1fr]"><Reveal className="border border-line-2 bg-surface p-5 sm:p-6"><p className="mono-label mb-6">FEATURE READINESS</p><dl className="divide-y divide-line">{readinessRows.map(([feature, rowStatus]) => <div key={feature} className="flex items-center justify-between gap-4 py-3 text-sm"><dt className="text-ink-1">{feature}</dt><dd className={`border px-2 py-1 font-mono text-[10px] tracking-[0.1em] ${STATUS_STYLE[rowStatus as DemoReadiness]}`}>{rowStatus}</dd></div>)}</dl><p className="mt-6 text-sm text-ink-3">Backend Integration status reflects live GET /api/system-status.</p></Reveal><Reveal delay={100} className="border border-line-2 bg-surface p-5 sm:p-6"><p className="mono-label mb-6">PLATFORM OVERVIEW</p><dl className="divide-y divide-line">{systemDemoStatus.platform.map(([label, value]) => <div key={label} className="flex justify-between gap-5 py-3 text-sm"><dt className="text-ink-2">{label}</dt><dd className="text-right text-ink-0">{label === "Application Mode" && status === "online" ? "Live backend integration" : value}</dd></div>)}</dl></Reveal></div>

    <Reveal delay={120} className="mt-12 border-t border-line pt-10"><p className="mono-label mb-5">CURRENT DEMO CAPABILITIES</p><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{systemDemoStatus.capabilities.map((capability, index) => <div key={capability} className="border border-line-2 bg-surface px-4 py-4"><span className="mono-label text-ink-3">{String(index + 1).padStart(2, "0")}</span><p className="mt-2 text-sm text-ink-1">{capability}</p></div>)}</div></Reveal>
  </div></div>;
}
