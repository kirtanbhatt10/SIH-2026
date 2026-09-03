import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Reveal from "../components/layout/Reveal";
import RadarSweep from "../components/viz/RadarSweep";
import LiveSpectrum from "../components/viz/LiveSpectrum";
import LiveWaveform from "../components/viz/LiveWaveform";
import Spectrogram from "../components/viz/Spectrogram";
import { useFrontendSettings } from "../hooks/useFrontendSettings";
import { useLiveAudioStream } from "../hooks/useLiveAudioStream";
import { useMonitoringSession } from "../hooks/useMonitoringSession";
import { severityLabels, type ComponentStatus } from "../data/mock";

const LIFECYCLE: { label: string }[] = [
  { label: "LISTENING" },
  { label: "SIGNAL DETECTED" },
  { label: "ANALYZING" },
  { label: "CLASSIFYING" },
  { label: "RESULT" },
];

const STATUS_STYLE: Record<ComponentStatus, string> = {
  IDLE: "text-ink-2",
  READY: "text-system",
  ACTIVE: "text-ink-0",
  ANALYZING: "text-amber",
  COMPLETE: "text-system",
  STOPPED: "text-ink-3",
};

interface LogEntry {
  time: string;
  message: string;
}

function deriveComponentStatuses(
  streamActive: boolean,
  hasThreat: boolean,
  phase: string,
): Record<"Acoustic Input" | "DSP Engine" | "AI Detector" | "Monitoring Engine", ComponentStatus> {
  if (phase === "stopped" || phase === "idle") {
    return {
      "Acoustic Input": phase === "stopped" ? "STOPPED" : "IDLE",
      "DSP Engine": phase === "stopped" ? "STOPPED" : "READY",
      "AI Detector": phase === "stopped" ? "STOPPED" : "READY",
      "Monitoring Engine": phase === "stopped" ? "STOPPED" : "IDLE",
    };
  }
  if (hasThreat) {
    return {
      "Acoustic Input": "ACTIVE",
      "DSP Engine": "COMPLETE",
      "AI Detector": "COMPLETE",
      "Monitoring Engine": "COMPLETE",
    };
  }
  if (streamActive) {
    return {
      "Acoustic Input": "ACTIVE",
      "DSP Engine": "ACTIVE",
      "AI Detector": "ANALYZING",
      "Monitoring Engine": "ACTIVE",
    };
  }
  return {
    "Acoustic Input": "IDLE",
    "DSP Engine": "READY",
    "AI Detector": "READY",
    "Monitoring Engine": "IDLE",
  };
}

function lifecycleLabel(
  phase: string,
  streamActive: boolean,
  hasThreat: boolean,
  error: string | null,
): string {
  if (error) return "ERROR";
  if (phase === "starting") return "STARTING";
  if (phase === "stopping") return "STOPPING";
  if (hasThreat) return "THREAT DETECTED";
  if (streamActive) return "MONITORING";
  if (phase === "stopped") return "STOPPED";
  return "IDLE";
}

export default function Monitoring() {
  const { settings } = useFrontendSettings();
  const session = useMonitoringSession();
  const liveAudio = useLiveAudioStream();
  const {
    bufferRef: liveBufferRef,
    sampleRateRef: liveSampleRateRef,
    connectionStatus: liveConnectionStatus,
    snapshot: liveSnapshot,
    connect: connectLiveAudio,
    disconnect: disconnectLiveAudio,
  } = liveAudio;
  const [log, setLog] = useState<LogEntry[]>([]);
  const prevThreatIdRef = useRef<string | null>(null);
  const prevForensicKeyRef = useRef<string | null>(null);
  const prevAudioStatusRef = useRef<string>("idle");
  const startLockRef = useRef(false);

  const addLog = useCallback((message: string) => {
    setLog((current) => [
      ...current,
      { time: new Date().toLocaleTimeString("en-US", { hour12: false }), message },
    ]);
  }, []);

  const hasThreat = session.hasSessionThreat;
  const isRunning = session.streamActive || session.isBusy;
  const activeIndex = hasThreat
    ? LIFECYCLE.length - 1
    : session.streamActive
      ? 0
      : -1;
  const componentStatuses = deriveComponentStatuses(session.streamActive, hasThreat, session.phase);

  useEffect(() => {
    const threat = session.currentThreat;
    if (!threat || !session.hasSessionThreat) return;
    if (prevThreatIdRef.current === threat.id) return;
    prevThreatIdRef.current = threat.id;
    addLog(`ThreatEvent received · ${threat.classification} · risk ${threat.source.risk}`);
  }, [session.currentThreat, session.hasSessionThreat, addLog]);

  useEffect(() => {
    const forensic = session.latestForensics.find((event) => event.event_type === "payload_decoded");
    if (!forensic) return;
    const key = `${forensic.stream_id}-${forensic.timestamp}-${forensic.recovered_text ?? ""}`;
    if (prevForensicKeyRef.current === key) return;
    prevForensicKeyRef.current = key;
    addLog(
      `Payload decoded · text="${forensic.recovered_text ?? "—"}" · confidence ${Math.round(forensic.confidence * 100)}% · bits ${forensic.bit_count}`,
    );
  }, [session.latestForensics, addLog]);

  useEffect(() => {
    if (session.streamActive) {
      connectLiveAudio();
    } else {
      disconnectLiveAudio();
    }
  }, [session.streamActive, session.streamId, connectLiveAudio, disconnectLiveAudio]);

  useEffect(() => {
    if (prevAudioStatusRef.current === liveConnectionStatus) return;
    prevAudioStatusRef.current = liveConnectionStatus;
    if (liveConnectionStatus === "connected") {
      addLog("Live audio stream connected (WS /stream/audio)");
    } else if (liveConnectionStatus === "error" && session.streamActive) {
      addLog("Live audio stream connection failed");
    }
  }, [liveConnectionStatus, session.streamActive, addLog]);

  const startMonitoring = async () => {
    if (startLockRef.current) return;
    startLockRef.current = true;
    setLog([]);
    prevThreatIdRef.current = null;
    addLog("Starting backend stream (mic device 1)…");
    try {
      const result = await session.start();
      if (result.success) {
        addLog(`Backend stream active · stream_id=${result.streamId ?? "—"}`);
      } else {
        addLog(`START FAILED · ${session.error ?? "unknown error"}`);
      }
    } finally {
      startLockRef.current = false;
    }
  };

  const stopMonitoring = async () => {
    addLog("Stopping backend stream…");
    const result = await session.stop();
    if (result.success) {
      addLog("Backend stream stopped");
    } else {
      addLog(`STOP FAILED · ${session.error ?? "unknown error"}`);
    }
  };

  const resetSession = async () => {
    if (session.streamActive) {
      await session.stop();
    }
    session.reset();
    setLog([]);
    prevThreatIdRef.current = null;
    prevForensicKeyRef.current = null;
    addLog("Session reset");
  };

  const display = hasThreat ? session.currentThreat : null;
  const analysisRows: [string, string][] = display
    ? [
        ["Dominant Frequency", display.frequency],
        ["Frequency Range", `${display.frequencyStart} – ${display.frequencyEnd}`],
        ["Signal Strength", display.signalStrength],
        ["Duration", display.duration],
        ["Confidence", display.confidence],
        ["Classification", display.classification],
        ["Threat Level", severityLabels[display.severity]],
        ["Threat Score", `${display.threatScore} / 100`],
        ["Pattern", display.source.pattern],
        ["Chunks Analyzed", String(display.source.chunks_analyzed)],
      ]
    : [];

  const threatBand =
    display &&
    display.source.frequency_start !== null &&
    display.source.frequency_end !== null
      ? {
          startHz: display.source.frequency_start,
          endHz: display.source.frequency_end,
          carrierFreqs: display.source.carrier_freqs,
        }
      : null;

  const vizHeight = settings.visualizationDensity === "compact" ? 140 : 190;

  return (
    <div className="px-5 pb-24 pt-28 md:px-10">
      <div className="mx-auto max-w-[1600px]">
        <Reveal>
          <div className="flex flex-wrap items-end justify-between gap-5">
            <div>
              <p className="mono-label mb-3">ACOUSTIC ENVIRONMENT ANALYSIS</p>
              <h1 className="font-display text-5xl tracking-tight sm:text-6xl">LIVE MONITOR</h1>
            </div>
            <span className="border border-system/50 px-3 py-2 font-mono text-[10px] tracking-[0.14em] text-system">
              LIVE BACKEND · PORT 8021
            </span>
          </div>
        </Reveal>

        <Reveal delay={80} className="mt-10 border border-line-2 bg-surface p-5 sm:p-6">
          <div className="flex flex-wrap items-center justify-between gap-5">
            <div>
              <p className="mono-label mb-2">MONITORING LIFECYCLE</p>
              <p className="font-display text-2xl tracking-tight">
                {lifecycleLabel(session.phase, session.streamActive, hasThreat, session.error)}
              </p>
              {session.streamId && (
                <p className="mt-2 font-mono text-xs text-ink-3">stream_id: {session.streamId}</p>
              )}
              {session.error && (
                <p className="mt-2 text-sm text-crimson" role="alert">{session.error}</p>
              )}
              <dl className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                <div className="text-xs">
                  <dt className="mono-label text-ink-3">Microphone</dt>
                  <dd className="font-mono text-ink-1">Device 1</dd>
                </div>
                <div className="text-xs">
                  <dt className="mono-label text-ink-3">Sample rate</dt>
                  <dd className="font-mono text-ink-1">{liveSnapshot.sampleRate / 1000} kHz</dd>
                </div>
                <div className="text-xs">
                  <dt className="mono-label text-ink-3">Live audio</dt>
                  <dd className="font-mono text-ink-1">{liveConnectionStatus.toUpperCase()}</dd>
                </div>
                <div className="text-xs">
                  <dt className="mono-label text-ink-3">Chunks received</dt>
                  <dd className="font-mono text-ink-1">{liveSnapshot.chunksReceived}</dd>
                </div>
              </dl>
            </div>
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={startMonitoring}
                disabled={isRunning || session.phase === "monitoring"}
                className="command-button disabled:cursor-not-allowed disabled:opacity-40"
              >
                START MONITORING <span>→</span>
              </button>
              <button
                type="button"
                onClick={stopMonitoring}
                disabled={!session.streamActive && !session.isBusy}
                className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson disabled:cursor-not-allowed disabled:opacity-40"
              >
                STOP MONITORING
              </button>
              <button
                type="button"
                onClick={resetSession}
                disabled={session.phase === "idle" && log.length === 0}
                className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson disabled:cursor-not-allowed disabled:opacity-40"
              >
                RESET SESSION
              </button>
            </div>
          </div>
        </Reveal>

        <Reveal delay={100} className="mt-8 grid grid-cols-2 gap-px overflow-hidden border border-line bg-line sm:grid-cols-4">
          {Object.entries(componentStatuses).map(([name, value]) => (
            <article key={name} className="bg-surface p-4">
              <p className="mono-label mb-3">{name}</p>
              <p className={`font-mono text-sm tracking-[0.1em] ${STATUS_STYLE[value]}`}>
                <span aria-hidden="true">● </span>
                {value}
              </p>
            </article>
          ))}
        </Reveal>

        <Reveal delay={110} className="mt-8 border-y border-line py-7">
          <p className="mono-label mb-5">DETECTION LIFECYCLE</p>
          <ol className="flex flex-wrap gap-2" aria-label="Detection lifecycle">
            {LIFECYCLE.map((stage, index) => {
              const completed = activeIndex > index || hasThreat;
              const active = activeIndex === index && !hasThreat;
              return (
                <li
                  key={stage.label}
                  className={`flex items-center gap-2 border px-3 py-2 font-mono text-[10px] tracking-[0.1em] transition-colors ${
                    completed ? "border-system/60 text-system" : active ? "border-crimson-3 text-crimson" : "border-line-2 text-ink-3"
                  }`}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  {stage.label}
                </li>
              );
            })}
          </ol>
        </Reveal>

        <div className={`mt-8 grid gap-8 xl:grid-cols-2 ${settings.visualizationDensity === "compact" ? "gap-4" : ""}`}>
          {settings.showWaveform && (
            <Reveal className="border border-line-2 bg-surface p-5 sm:p-6">
              <header className="mb-5 flex flex-wrap items-center justify-between gap-3">
                <p className="mono-label">RAW LIVE SIGNAL</p>
                <span className="mono-label text-system">LIVE MICROPHONE</span>
              </header>
              <LiveWaveform
                bufferRef={liveBufferRef}
                active={session.streamActive}
                height={vizHeight}
              />
              <p className="mt-3 font-mono text-[10px] tracking-[0.1em] text-ink-3">
                Source: WS /stream/audio · samples buffered: {liveSnapshot.samplesBuffered}
              </p>
            </Reveal>
          )}
          {settings.showSpectrum && (
            <Reveal delay={80} className="border border-line-2 bg-surface p-5 sm:p-6">
              <header className="mb-5 flex flex-wrap items-center justify-between gap-3">
                <p className="mono-label">ULTRASONIC SPECTRUM</p>
                <span className="mono-label text-system">LIVE FFT · 16–24 kHz</span>
              </header>
              <LiveSpectrum
                bufferRef={liveBufferRef}
                sampleRateRef={liveSampleRateRef}
                active={session.streamActive}
                height={vizHeight}
                minHz={16000}
                maxHz={24000}
                threatBand={threatBand}
              />
              <p className="mt-3 font-mono text-[10px] tracking-[0.1em] text-ink-3">
                Source: client FFT on live mic samples · overlay from backend DSP threat band
              </p>
            </Reveal>
          )}
          {settings.showSpectrogram && (
            <Reveal delay={100} className="border border-line-2 bg-surface p-5 sm:p-6">
              <header className="mb-5 flex flex-wrap items-center justify-between gap-3">
                <p className="mono-label">SPECTROGRAM</p>
                <span className="mono-label text-ink-3">VISUALIZATION ONLY</span>
              </header>
              <Spectrogram height={settings.visualizationDensity === "compact" ? 160 : 210} />
            </Reveal>
          )}
          <Reveal delay={120} className="border border-line-2 bg-surface p-5 sm:p-6">
            <header className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <p className="mono-label">NODE FIELD</p>
              <span className="mono-label text-ink-3">VISUAL CONTEXT ONLY</span>
            </header>
            <div className="flex justify-center">
              <RadarSweep size={270} />
            </div>
          </Reveal>
        </div>

        <div className="mt-12 grid gap-8 xl:grid-cols-[1.15fr_.85fr]">
          <Reveal className="border border-line-2 bg-surface p-5 sm:p-6">
            <p className="mono-label mb-6">THREAT ANALYSIS · BACKEND DSP</p>
            <dl className="grid gap-x-8 sm:grid-cols-2">
              {hasThreat
                ? analysisRows.map(([label, value]) => (
                    <div key={label} className="flex justify-between gap-4 border-b border-line py-3 text-sm">
                      <dt className="text-ink-2">{label}</dt>
                      <dd className="text-right text-ink-0">{value}</dd>
                    </div>
                  ))
                : (
                  <p className="text-sm text-ink-3">
                    {session.streamActive
                      ? "NO THREAT — monitoring active, awaiting backend detection"
                      : "— Waiting for monitoring session"}
                  </p>
                )}
            </dl>
            {session.latestForensics.length > 0 && (
              <div className="mt-8 border-t border-line pt-6">
                <p className="mono-label mb-4">PAYLOAD FORENSICS</p>
                <dl className="grid gap-x-8 sm:grid-cols-2">
                  {session.latestForensics.map((forensic, index) => (
                    <div key={`${forensic.stream_id}-${forensic.timestamp}-${index}`} className="col-span-full border-b border-line py-3 text-sm">
                      <div className="flex justify-between gap-4">
                        <dt className="text-ink-2">Event</dt>
                        <dd className="text-right text-ink-0">{forensic.event_type}</dd>
                      </div>
                      {forensic.recovered_text !== null && (
                        <div className="mt-2 flex justify-between gap-4">
                          <dt className="text-ink-2">Recovered text</dt>
                          <dd className="text-right font-mono text-ink-0">{forensic.recovered_text}</dd>
                        </div>
                      )}
                      <div className="mt-2 flex justify-between gap-4">
                        <dt className="text-ink-2">Confidence</dt>
                        <dd className="text-right text-ink-0">{Math.round(forensic.confidence * 100)}%</dd>
                      </div>
                      <div className="mt-2 flex justify-between gap-4">
                        <dt className="text-ink-2">Bits / preamble</dt>
                        <dd className="text-right text-ink-0">
                          {forensic.bit_count} · {forensic.preamble_found ? "found" : "not found"}
                        </dd>
                      </div>
                    </div>
                  ))}
                </dl>
              </div>
            )}
          </Reveal>
          <Reveal delay={100} className="border border-line-2 bg-surface p-5 sm:p-6">
            <p className="mono-label mb-6">SESSION EVENT LOG</p>
            {log.length === 0 ? (
              <p className="text-sm text-ink-3">Waiting for monitoring session to start.</p>
            ) : (
              <ol className="divide-y divide-line">
                {log.map((entry, index) => (
                  <li key={`${entry.time}-${index}`} className="grid grid-cols-[90px_1fr] gap-4 py-3 text-sm">
                    <time className="font-mono text-ink-3">{entry.time}</time>
                    <span className="text-ink-1">{entry.message}</span>
                  </li>
                ))}
              </ol>
            )}
          </Reveal>
        </div>

        {hasThreat && display && (
          <Reveal className="mt-12 border border-crimson-3/60 bg-crimson-3/5 p-6 sm:p-8">
            <p className="mono-label mb-3 text-crimson">BACKEND DSP DETECTION · {display.source.schema_version}</p>
            <h2 className="font-display text-3xl tracking-tight sm:text-4xl">
              {display.label}
            </h2>
            <div className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-5">
              {[
                ["Risk", severityLabels[display.severity]],
                ["Confidence", display.confidence],
                ["Classification", display.classification],
                ["Dominant Frequency", display.frequency],
                ["Threat Score", `${display.threatScore} / 100`],
              ].map(([label, value]) => (
                <div key={label}>
                  <p className="mono-label mb-2">{label}</p>
                  <p className="text-sm text-ink-0">{value}</p>
                </div>
              ))}
            </div>
            <div className="mt-8 flex flex-wrap gap-4">
              <Link to={`/events/${display.id}`} className="command-button">
                VIEW DETECTION DETAILS <span>→</span>
              </Link>
              <Link
                to="/events"
                className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson"
              >
                VIEW ALL THREAT EVENTS →
              </Link>
              <button type="button" onClick={resetSession} className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">
                RESET SESSION →
              </button>
            </div>
          </Reveal>
        )}
      </div>
    </div>
  );
}
