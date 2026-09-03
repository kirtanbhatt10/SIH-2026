import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Reveal from "../components/layout/Reveal";
import Pipeline from "../components/viz/Pipeline";
import Spectrum from "../components/viz/Spectrum";
import Waveform from "../components/viz/Waveform";
import { useFrontendSettings } from "../hooks/useFrontendSettings";
import { simulatorDefaults } from "../data/mock";
import { adaptThreatEvent, type ThreatEventDisplay } from "../services/adapter";
import { ApiError } from "../services/api";
import { generatePayload, getSimulationStatus, transmitPayload } from "../services/simulator";
import { getThreats } from "../services/threats";

type SimulatorPhase =
  | "ready"
  | "generating"
  | "generated"
  | "transmitting"
  | "polling"
  | "transmission-complete"
  | "detection-available"
  | "detection-unavailable";

interface LogEntry {
  time: string;
  message: string;
}

const PIPELINE = ["PAYLOAD", "ENCODING", "MODULATION", "SIGNAL GENERATION", "TRANSMISSION", "DETECTION", "ANALYSIS", "CLASSIFICATION"];

const PHASE_INFO: Record<SimulatorPhase, { label: string; message: string; pipelineIndex: number }> = {
  ready: { label: "READY", message: "Simulation configured and ready.", pipelineIndex: -1 },
  generating: { label: "GENERATING PAYLOAD", message: "Encoding payload via backend BFSK simulator.", pipelineIndex: 3 },
  generated: { label: "PAYLOAD GENERATED", message: "Backend returned payload_id and WAV artifact.", pipelineIndex: 3 },
  transmitting: { label: "TRANSMITTING", message: "Requesting backend transmission.", pipelineIndex: 4 },
  polling: { label: "CHECKING STATUS", message: "Polling backend transmission status.", pipelineIndex: 5 },
  "transmission-complete": { label: "TRANSMISSION COMPLETE", message: "Backend reports transmission complete.", pipelineIndex: 4 },
  "detection-available": { label: "DETECTION AVAILABLE", message: "A ThreatEvent exists in backend history.", pipelineIndex: 7 },
  "detection-unavailable": { label: "DETECTION NOT AVAILABLE", message: "Transmission complete. No new ThreatEvent was returned by the backend.", pipelineIndex: 6 },
};

const TERMINAL_PHASES: SimulatorPhase[] = [
  "transmission-complete",
  "detection-available",
  "detection-unavailable",
];

const POLL_INTERVAL_MS = 500;
const POLL_MAX_ATTEMPTS = 24;
const DETECTION_POLL_INTERVAL_MS = 1500;
const DETECTION_POLL_MAX_ATTEMPTS = 20;

function parseCarrierKhz(label: string): number {
  const match = label.match(/([\d.]+)/);
  return match ? Number.parseFloat(match[1]) : 20;
}

function buildFskFrequencies(carrierLabel: string): { freq_0_hz: number; freq_1_hz: number } {
  const carrierHz = Math.round(parseCarrierKhz(carrierLabel) * 1000);
  // Match backend DSP defaults (FREQ_0/FREQ_1 around 18.5/20.5 kHz).
  return { freq_0_hz: carrierHz - 1500, freq_1_hz: carrierHz + 500 };
}

function transmissionModeToBackend(mode: string): string {
  return mode === "Ultrasonic Simulation" ? "audio" : "virtual";
}

export default function AttackLab() {
  const { settings } = useFrontendSettings();
  const [payload, setPayload] = useState(simulatorDefaults.payload);
  const [carrierFrequency, setCarrierFrequency] = useState(simulatorDefaults.carrierFrequency);
  const [duration, setDuration] = useState(simulatorDefaults.duration);
  const [modulation, setModulation] = useState(simulatorDefaults.modulation);
  const [transmissionMode, setTransmissionMode] = useState(simulatorDefaults.transmissionMode);
  const [phase, setPhase] = useState<SimulatorPhase>("ready");
  const [log, setLog] = useState<LogEntry[]>([]);
  const [error, setError] = useState("");
  const [payloadId, setPayloadId] = useState<string | null>(null);
  const [progressPct, setProgressPct] = useState(0);
  const [backendState, setBackendState] = useState<string | null>(null);
  const [detectionEvent, setDetectionEvent] = useState<ThreatEventDisplay | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const runTokenRef = useRef(0);

  const addLog = useCallback((message: string) => {
    setLog((entries) => [...entries, { time: new Date().toLocaleTimeString("en-US", { hour12: false }), message }]);
  }, []);

  const clearPollTimer = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  useEffect(() => clearPollTimer, [clearPollTimer]);

  const isRunning = !TERMINAL_PHASES.includes(phase) && phase !== "ready";
  const info = PHASE_INFO[phase];
  const progress = progressPct > 0 ? progressPct : Math.max(0, Math.round(((info.pipelineIndex + 1) / PIPELINE.length) * 100));

  const resolveDetection = async (threatCountBefore: number, runToken: number) => {
    setPhase("polling");
    addLog("Waiting for backend threat history after transmission...");

    for (let attempt = 1; attempt <= DETECTION_POLL_MAX_ATTEMPTS; attempt += 1) {
      if (runToken !== runTokenRef.current) return;

      try {
        const threatsResponse = await getThreats();
        if (runToken !== runTokenRef.current) return;

        console.info("[ATTACKLAB] polling threats attempt", attempt, threatsResponse.threats.length);
        const adapted = threatsResponse.threats.map(adaptThreatEvent);
        if (adapted.length > threatCountBefore) {
          const newest = adapted[adapted.length - 1];
          console.info("[ATTACKLAB] detection found", newest.id);
          setDetectionEvent(newest);
          setPhase("detection-available");
          addLog(`Backend threat history increased (${adapted.length} total). Latest: ${newest.id}`);
          return;
        }

        if (attempt < DETECTION_POLL_MAX_ATTEMPTS) {
          addLog(`Detection poll ${attempt}/${DETECTION_POLL_MAX_ATTEMPTS}: no new ThreatEvent yet`);
          await new Promise((resolve) => window.setTimeout(resolve, DETECTION_POLL_INTERVAL_MS));
        }
      } catch (err) {
        if (runToken !== runTokenRef.current) return;
        setDetectionEvent(null);
        setPhase("detection-unavailable");
        const message = err instanceof ApiError ? err.message : "Unable to verify backend threat history.";
        addLog(message);
        return;
      }
    }

    if (runToken !== runTokenRef.current) return;
    setDetectionEvent(null);
    setPhase("detection-unavailable");
    addLog("No new ThreatEvent returned by backend after transmission.");
  };

  const handleTransmitComplete = async (
    transmitted: Awaited<ReturnType<typeof transmitPayload>>,
    threatCountBefore: number,
    runToken: number,
  ) => {
    if (runToken !== runTokenRef.current) return;

    setBackendState(transmitted.state);
    setProgressPct(transmitted.progress_pct);
    addLog(
      `Transmit: state=${transmitted.state}, threats ${transmitted.threats_before ?? "?"}→${transmitted.threats_after ?? "?"}, dsp=${transmitted.dsp_detected}, ml=${transmitted.ml_class}/${transmitted.ml_risk}`,
    );

    if (
      transmitted.threats_after != null &&
      transmitted.threats_before != null &&
      transmitted.threats_after > transmitted.threats_before
    ) {
      setPhase("transmission-complete");
      addLog("Backend reports detection during transmit.");
      await resolveDetection(threatCountBefore, runToken);
      return;
    }

    if (transmitted.state === "completed_no_detection") {
      setPhase("detection-unavailable");
      setError(transmitted.reason ?? "Processing completed without new ThreatEvent.");
      addLog(transmitted.reason ?? "completed_no_detection — no new ThreatEvent.");
      return;
    }

    if (transmitted.state === "complete" || transmitted.progress_pct >= 100) {
      setPhase("transmission-complete");
      addLog("Transmission and processing complete.");
      await resolveDetection(threatCountBefore, runToken);
      return;
    }

    pollStatus(transmitted.payload_id, threatCountBefore, runToken);
  };

  const pollStatus = (id: string, threatCountBefore: number, runToken: number) => {
    let attempts = 0;
    clearPollTimer();
    setPhase("polling");

    pollTimerRef.current = window.setInterval(async () => {
      attempts += 1;
      try {
        const status = await getSimulationStatus(id);
        if (runToken !== runTokenRef.current) {
          clearPollTimer();
          return;
        }

        setBackendState(status.state);
        setProgressPct(status.progress_pct);
        addLog(`Status poll: state=${status.state}, progress=${status.progress_pct}%`);

        if (
          status.state === "complete" ||
          status.state === "completed_no_detection" ||
          status.state === "failed" ||
          status.progress_pct >= 100
        ) {
          clearPollTimer();
          if (status.state === "failed") {
            setError("Backend reported transmission failed.");
            setPhase("ready");
            return;
          }
          if (status.state === "completed_no_detection") {
            setPhase("detection-unavailable");
            setError(status.reason ?? "Processing completed without detection.");
            addLog(status.reason ?? "completed_no_detection");
            return;
          }
          setPhase("transmission-complete");
          addLog("Transmission complete (backend).");
          await resolveDetection(threatCountBefore, runToken);
        } else if (attempts >= POLL_MAX_ATTEMPTS) {
          clearPollTimer();
          setError("Transmission status polling timed out.");
          setPhase("ready");
        }
      } catch (err) {
        if (runToken !== runTokenRef.current) return;
        clearPollTimer();
        setError(err instanceof ApiError ? err.message : "Failed to poll simulator status.");
        setPhase("ready");
      }
    }, POLL_INTERVAL_MS);
  };

  const runSimulation = async () => {
    if (!payload.trim()) {
      setError("Enter a test payload before running the simulation.");
      return;
    }

    if (modulation !== "FSK") {
      setError("Backend simulator currently supports BFSK only. Select FSK modulation.");
      return;
    }

    const runToken = runTokenRef.current + 1;
    runTokenRef.current = runToken;

    clearPollTimer();
    setError("");
    setLog([]);
    setPayloadId(null);
    setProgressPct(0);
    setBackendState(null);
    setDetectionEvent(null);
    setPhase("generating");
    addLog("Simulation initialized");
    addLog(`Payload accepted (${payload.length} chars)`);

    let threatCountBefore = 0;
    try {
      const existingThreats = await getThreats();
      threatCountBefore = existingThreats.threats.length;
    } catch {
      addLog("Could not read threat count before run; detection check may be inconclusive.");
    }

    const frequencies = buildFskFrequencies(carrierFrequency);
    addLog(`Requesting generate: freq_0=${frequencies.freq_0_hz} Hz, freq_1=${frequencies.freq_1_hz} Hz`);

    try {
      const generated = await generatePayload({
        text: payload.trim(),
        freq_0_hz: frequencies.freq_0_hz,
        freq_1_hz: frequencies.freq_1_hz,
        bit_duration_ms: 50,
      });

      if (runToken !== runTokenRef.current) return;

      console.info("[ATTACKLAB] generate response", generated);

      setPayloadId(generated.payload_id);
      setBackendState("ready");
      setProgressPct(0);
      setPhase("generated");
      addLog(`Generate OK: payload_id=${generated.payload_id}, duration=${generated.duration_sec.toFixed(2)}s, bits=${generated.bit_count}`);
      addLog(`WAV: ${generated.wav_url}`);

      setPhase("transmitting");
      const mode = transmissionModeToBackend(transmissionMode);
      addLog(`Transmitting via backend mode='${mode}'`);
      console.info("[ATTACKLAB] transmit request", { payload_id: generated.payload_id, mode });

      const transmitted = await transmitPayload({ payload_id: generated.payload_id, mode });
      if (runToken !== runTokenRef.current) return;

      console.info("[ATTACKLAB] transmit response", transmitted);

      await handleTransmitComplete(transmitted, threatCountBefore, runToken);
    } catch (err) {
      if (runToken !== runTokenRef.current) return;
      setPhase("ready");
      setError(err instanceof ApiError ? err.message : "Unable to communicate with simulator API.");
      addLog("Simulator API error.");
    }
  };

  const stop = () => {
    runTokenRef.current += 1;
    clearPollTimer();
    setPhase("ready");
    addLog("Simulation stopped · returned to ready state");
  };

  const reset = () => {
    runTokenRef.current += 1;
    clearPollTimer();
    setPhase("ready");
    setLog([]);
    setError("");
    setPayloadId(null);
    setProgressPct(0);
    setBackendState(null);
    setDetectionEvent(null);
  };

  const selectClass = "w-full border border-line-2 bg-void-2 px-3 py-3 text-sm text-ink-0 outline-none transition-colors focus:border-crimson disabled:opacity-50";
  const showVizActive = isRunning || TERMINAL_PHASES.includes(phase);
  const canRunAgain = TERMINAL_PHASES.includes(phase);

  return <div className="px-5 pb-24 pt-28 md:px-10"><div className="mx-auto max-w-[1600px]">
    <Reveal><div className="flex flex-wrap items-end justify-between gap-5"><div><p className="mono-label mb-3">CONTROLLED SECURITY TESTING ENVIRONMENT</p><h1 className="font-display text-5xl tracking-tight sm:text-6xl">ACOUSTIC ATTACK LAB</h1></div><div className="flex flex-wrap gap-2"><span className="border border-line-2 px-3 py-2 font-mono text-[10px] tracking-[0.14em] text-ink-2">BACKEND SIMULATOR</span><span className="border border-system/50 px-3 py-2 font-mono text-[10px] tracking-[0.14em] text-system">PORT 8021</span></div></div></Reveal>

    <Reveal delay={80} className="mt-10 border border-line-2 bg-surface p-5 sm:p-6"><div className="flex flex-wrap items-center justify-between gap-5"><div><p className="mono-label mb-2">TRANSMISSION STATUS</p><h2 className="font-display text-2xl tracking-tight">{info.label}</h2><p className="mt-2 text-sm text-ink-2">{info.message}</p>{payloadId && <p className="mt-2 font-mono text-xs text-ink-3">payload_id: {payloadId}{backendState ? ` · state: ${backendState}` : ""}</p>}</div><div className="min-w-44"><div className="mb-2 flex justify-between font-mono text-[10px] text-ink-3"><span>PIPELINE PROGRESS</span><span>{progress}%</span></div><div className="h-1 bg-surface-3"><div className="h-full bg-crimson transition-[width] duration-500" style={{ width: `${progress}%` }} /></div></div></div></Reveal>

    <div className="mt-8 grid gap-8 xl:grid-cols-[.9fr_1.1fr]"><Reveal className="border border-line-2 bg-surface p-5 sm:p-6"><p className="mono-label mb-5">TEST PAYLOAD</p><label htmlFor="payload" className="sr-only">Controlled simulation payload</label><textarea id="payload" value={payload} maxLength={simulatorDefaults.maxPayloadLength} disabled={isRunning} onChange={(event) => { setPayload(event.target.value); setError(""); }} className="min-h-36 w-full resize-y border border-line-2 bg-void-2 p-4 font-mono text-sm text-ink-0 outline-none transition-colors focus:border-crimson disabled:opacity-50" placeholder="SESSION_TOKEN=SIH2026" /><div className="mt-3 flex justify-between gap-4"><span className="text-sm text-crimson" role="alert">{error}</span><span className="mono-label shrink-0 text-ink-3">{payload.length} / {simulatorDefaults.maxPayloadLength}</span></div></Reveal><Reveal delay={80} className="border border-line-2 bg-surface p-5 sm:p-6"><p className="mono-label mb-5">SIMULATION CONFIGURATION</p><div className="grid gap-5 sm:grid-cols-2"><label className="mono-label">CARRIER FREQUENCY<select value={carrierFrequency} disabled={isRunning} onChange={(event) => setCarrierFrequency(event.target.value)} className={`${selectClass} mt-2`}>{["18.5 kHz", "19.0 kHz", "19.5 kHz", "20.0 kHz", "20.5 kHz", "21.0 kHz"].map((value) => <option key={value}>{value}</option>)}</select></label><label className="mono-label">DURATION<select value={duration} disabled={isRunning} onChange={(event) => setDuration(event.target.value)} className={`${selectClass} mt-2`}>{["2 sec", "3 sec", "5 sec"].map((value) => <option key={value}>{value}</option>)}</select><span className="mt-1 block text-[10px] normal-case tracking-normal text-ink-3">UI only — backend duration follows encoded payload length.</span></label><label className="mono-label">MODULATION<select value={modulation} disabled={isRunning} onChange={(event) => setModulation(event.target.value)} className={`${selectClass} mt-2`}>{["FSK", "ASK", "Demo Pattern"].map((value) => <option key={value}>{value}</option>)}</select></label><label className="mono-label">TRANSMISSION MODE<select value={transmissionMode} disabled={isRunning} onChange={(event) => setTransmissionMode(event.target.value)} className={`${selectClass} mt-2`}>{["Ultrasonic Simulation", "Acoustic Simulation"].map((value) => <option key={value}>{value}</option>)}</select><span className="mt-1 block text-[10px] normal-case tracking-normal text-ink-3">Ultrasonic → mode=audio · Acoustic → mode=virtual</span></label></div></Reveal></div>

    <Reveal delay={100} className="mt-8 border border-line-2 bg-surface p-5 sm:p-6"><div className="flex flex-wrap gap-3"><button type="button" onClick={runSimulation} disabled={isRunning} className="command-button disabled:cursor-not-allowed disabled:opacity-40">RUN SIMULATION <span>→</span></button><button type="button" onClick={stop} disabled={!isRunning} className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson disabled:cursor-not-allowed disabled:opacity-40">STOP</button><button type="button" onClick={reset} disabled={phase === "ready" && log.length === 0} className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson disabled:cursor-not-allowed disabled:opacity-40">RESET</button>{canRunAgain && <button type="button" onClick={runSimulation} className="border border-crimson-3 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-crimson transition-colors hover:bg-crimson-3/10">RUN AGAIN</button>}</div></Reveal>

    <Reveal delay={110} className="mt-10 border-t border-line pt-10"><p className="mono-label mb-6">TRANSMISSION PIPELINE</p><Pipeline stages={PIPELINE} activeIndex={info.pipelineIndex} /></Reveal>

    {(settings.showWaveform || settings.showSpectrum) && <div className={`mt-10 grid xl:grid-cols-2 ${settings.visualizationDensity === "compact" ? "gap-4" : "gap-8"}`}>{settings.showWaveform && <Reveal className="border border-line-2 bg-surface p-5 sm:p-6"><header className="mb-5 flex flex-wrap justify-between gap-3"><p className="mono-label">SIGNAL WAVEFORM</p><span className="mono-label text-ink-3">{showVizActive ? "BACKEND RUN ACTIVE" : "IDLE"}</span></header><Waveform mode={phase === "detection-available" ? "threat" : showVizActive ? "anomaly" : "normal"} active={showVizActive} height={settings.visualizationDensity === "compact" ? 130 : 180} /></Reveal>}{settings.showSpectrum && <Reveal delay={80} className="border border-line-2 bg-surface p-5 sm:p-6"><header className="mb-5 flex flex-wrap justify-between gap-3"><p className="mono-label">FREQUENCY SPECTRUM</p><span className="mono-label text-ink-3">PROCEDURAL VIZ</span></header><Spectrum height={settings.visualizationDensity === "compact" ? 130 : 180} threatIndex={phase === "detection-available" ? 30 : undefined} anomalyIndex={showVizActive ? 30 : undefined} /></Reveal>}</div>}

    <div className="mt-12 grid gap-8 xl:grid-cols-[1.15fr_.85fr]"><Reveal className="border border-line-2 bg-surface p-5 sm:p-6"><p className="mono-label mb-6">SIMULATION EVENT LOG</p>{log.length === 0 ? <p className="text-sm text-ink-3">Simulation activity will be recorded here.</p> : <ol className="divide-y divide-line">{log.map((entry, index) => <li key={`${entry.time}-${index}`} className="grid grid-cols-[90px_1fr] gap-4 py-3 text-sm"><time className="font-mono text-ink-3">{entry.time}</time><span className="text-ink-1">{entry.message}</span></li>)}</ol>}</Reveal></div>

    {phase === "detection-unavailable" && (
      <Reveal className="mt-12 border border-line-2 bg-surface p-6 sm:p-8">
        <p className="mono-label mb-3 text-ink-2">DETECTION STATUS</p>
        <h2 className="font-display text-3xl tracking-tight sm:text-4xl">TRANSMISSION COMPLETE</h2>
        <p className="mt-4 max-w-2xl text-sm text-ink-1">No detection event has been returned by the backend yet. Ensure the monitoring stream is active on the Monitor page, then retry. Ultrasonic mode plays through the configured speaker while the live microphone stream feeds the DSP/ML pipeline.</p>
        <div className="mt-8 flex flex-wrap gap-4"><Link to="/events" className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">VIEW THREAT EVENTS →</Link></div>
      </Reveal>
    )}

    {phase === "detection-available" && detectionEvent && (
      <Reveal className="mt-12 border border-crimson-3/60 bg-crimson-3/5 p-6 sm:p-8">
        <p className="mono-label mb-3 text-crimson">BACKEND THREAT HISTORY</p>
        <h2 className="font-display text-3xl tracking-tight sm:text-4xl">THREAT EVENT AVAILABLE IN BACKEND</h2>
        <p className="mt-4 max-w-2xl text-sm text-ink-1">A new entry appeared in GET /api/threats after this run. This is not guaranteed to be caused solely by this transmission unless the full E2E pipeline is active.</p>
        <div className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-5">{[["Event ID", detectionEvent.id], ["Classification", detectionEvent.classification], ["Confidence", detectionEvent.confidence], ["Risk Level", detectionEvent.source.risk], ["Dominant Frequency", detectionEvent.frequency], ["Threat Score", `${detectionEvent.threatScore} / 100`]].map(([label, value]) => <div key={label}><p className="mono-label mb-2">{label}</p><p className="text-sm text-ink-0">{value}</p></div>)}</div>
        <div className="mt-8 flex flex-wrap gap-4"><Link to={`/events/${detectionEvent.id}`} className="command-button">VIEW DETECTION DETAILS <span>→</span></Link><Link to="/events" className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">VIEW ALL THREAT EVENTS →</Link></div>
      </Reveal>
    )}
  </div></div>;
}
