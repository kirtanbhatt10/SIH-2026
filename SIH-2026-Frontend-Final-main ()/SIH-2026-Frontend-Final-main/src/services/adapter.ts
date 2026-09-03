import type { DashboardStatus, Severity } from "../data/mock";
import type { PatternType, RiskLevel, ThreatEvent } from "./types";

/**
 * Frontend display model derived from backend ThreatEvent.
 * Kept separate from mock SignalEvent — pages will adopt this in later phases.
 */
export interface ThreatEventDisplay {
  /** Client-derived stable key from backend timestamp (no server event ID exists). */
  id: string;
  time: string;
  timestamp: string;
  /** Derived from detected + pattern — not a backend field. */
  label: string;
  /** Dominant carrier formatted from carrier_freqs or frequency range (Hz → kHz). */
  frequency: string;
  frequencyStart: string;
  frequencyEnd: string;
  /** Maps backend snr (dB) — not RF dBm. */
  signalStrength: string;
  /** Backend duration in seconds, formatted for display. */
  duration: string;
  /** Mapped from backend risk level. */
  severity: Severity;
  /** confidence 0–1 → percentage string. */
  confidence: string;
  /** Human-readable label from backend pattern enum. */
  classification: string;
  /** suspicion_score 0–1 → integer 0–100. */
  threatScore: number;
  /** Original backend fields preserved for future integration. */
  source: ThreatEvent;
}

const PATTERN_LABELS: Record<PatternType, string> = {
  none: "No modulation pattern",
  tone: "Ultrasonic tone",
  fsk: "Frequency-shift keying (FSK)",
  ook: "On-off keying (OOK)",
  chirp: "Frequency chirp",
};

const RISK_TO_SEVERITY: Record<RiskLevel, Severity> = {
  LOW: "normal",
  MEDIUM: "anomaly",
  HIGH: "threat",
};

function formatHzToKhz(hz: number): string {
  const khz = hz / 1000;
  return `${khz.toFixed(khz >= 10 ? 1 : 2)} kHz`;
}

function formatFrequencyRange(start: number | null, end: number | null): { start: string; end: string } {
  if (start === null && end === null) {
    return { start: "—", end: "—" };
  }
  return {
    start: start === null ? "—" : formatHzToKhz(start),
    end: end === null ? "—" : formatHzToKhz(end),
  };
}

function dominantFrequency(event: ThreatEvent): string {
  if (event.carrier_freqs.length > 0) {
    const hz = event.carrier_freqs[0];
    return formatHzToKhz(hz);
  }
  if (event.frequency_start !== null) {
    return formatHzToKhz(event.frequency_start);
  }
  return "—";
}

function buildLabel(event: ThreatEvent): string {
  if (!event.detected) {
    return "No threat detected";
  }
  return `${PATTERN_LABELS[event.pattern]} detected`;
}

function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

/** suspicion_score [0,1] → display score [0,100] (rounded). */
export function suspicionScoreToDisplayScore(suspicionScore: number): number {
  return Math.round(Math.min(1, Math.max(0, suspicionScore)) * 100);
}

/** Map adapted severity to dashboard posture band. */
export function severityToDashboardStatus(severity: Severity): DashboardStatus {
  if (severity === "threat") return "high-risk";
  if (severity === "anomaly") return "suspicious";
  return "safe";
}

function formatDurationSeconds(durationSec: number): string {
  if (durationSec <= 0) {
    return "—";
  }
  return `${durationSec.toFixed(1)} s`;
}

function formatTimestamp(unixSeconds: number): { iso: string; time: string } {
  const date = new Date(unixSeconds * 1000);
  return {
    iso: date.toISOString().replace("T", " ").slice(0, 19),
    time: date.toLocaleTimeString("en-US", { hour12: false }),
  };
}

/** Derive a client-side ID from backend timestamp (no backend event_id field). */
export function threatEventToClientId(event: ThreatEvent): string {
  return `TE-${Math.floor(event.timestamp * 1000)}`;
}

/**
 * Convert a backend ThreatEvent into a frontend display model.
 * Conversions are explicit; fields without backend equivalents are labeled in JSDoc above.
 */
export function adaptThreatEvent(event: ThreatEvent): ThreatEventDisplay {
  const range = formatFrequencyRange(event.frequency_start, event.frequency_end);
  const { iso, time } = formatTimestamp(event.timestamp);

  return {
    id: threatEventToClientId(event),
    time,
    timestamp: iso,
    label: buildLabel(event),
    frequency: dominantFrequency(event),
    frequencyStart: range.start,
    frequencyEnd: range.end,
    signalStrength: `${event.snr.toFixed(1)} dB SNR`,
    duration: formatDurationSeconds(event.duration),
    severity: RISK_TO_SEVERITY[event.risk],
    confidence: formatConfidence(event.confidence),
    classification: PATTERN_LABELS[event.pattern],
    threatScore: suspicionScoreToDisplayScore(event.suspicion_score),
    source: event,
  };
}
