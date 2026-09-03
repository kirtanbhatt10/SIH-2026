/** Backend 1 — GET /api/system-status */
export interface SystemStatusResponse {
  status: string;
  service: string;
  version: string;
}

/** Backend 2 — GET /health */
export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  sample_rate: number;
  freq_0: number;
  freq_1: number;
  bit_duration: number;
  storage_dir: string;
  total_payloads_generated: number;
  stream_active: boolean;
}

/** Backend 1 — ThreatEvent (schema 1.0.0-dsp) */
export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";
export type PatternType = "none" | "tone" | "fsk" | "ook" | "chirp";

export interface ThreatEvent {
  schema_version: string;
  detected: boolean;
  confidence: number;
  risk: RiskLevel;
  suspicion_score: number;
  frequency_start: number | null;
  frequency_end: number | null;
  carrier_freqs: number[];
  duration: number;
  pattern: PatternType;
  snr: number;
  chunks_analyzed: number;
  timestamp: number;
}

/** Backend 1 — GET /api/threats */
export interface ThreatsListResponse {
  threats: ThreatEvent[];
}

/** Backend 1 — GET /api/threats/current */
export interface CurrentThreatResponse {
  current: ThreatEvent | null;
  message?: string;
}

/** Backend 1 — POST /api/analyze */
export interface AnalyzeThreatResponse {
  status: string;
  event: ThreatEvent;
}

/** Backend 2 — POST /simulate/generate */
export interface GeneratePayloadRequest {
  text: string;
  freq_0_hz?: number;
  freq_1_hz?: number;
  bit_duration_ms?: number;
}

export interface GeneratePayloadResponse {
  payload_id: string;
  duration_sec: number;
  bit_count: number;
  wav_url: string;
}

/** Backend 2 — POST /simulate/transmit */
export interface TransmitRequest {
  payload_id: string;
  mode?: string;
}

export interface TransmitStatus {
  payload_id: string;
  state: string;
  progress_pct: number;
  chunks_processed?: number | null;
  dsp_detected?: boolean | null;
  ml_class?: string | null;
  ml_risk?: string | null;
  threats_before?: number | null;
  threats_after?: number | null;
  pipeline_object_id?: number | null;
  dsp_object_id?: number | null;
  reason?: string | null;
}

/** Backend 2 — POST /simulate/acoustic-exfiltrate */
export interface AcousticExfiltrateRequest {
  source_type?: string;
  custom_text?: string | null;
  freq_0_hz?: number;
  freq_1_hz?: number;
  bit_duration_ms?: number;
  emit_audio?: boolean;
}

export interface AcousticExfiltrateResponse {
  status: string;
  text_length: number;
  duration_sec: number;
  bit_count: number;
  payload_id: string;
  wav_url?: string | null;
  preview_text: string;
}

/** Backend 2 — GET /simulate/list item */
export interface SimulatorPayloadListItem {
  payload_id: string;
  text: string;
  state: string;
  wav_url: string;
}

/** Backend 2 — audio stream */
export interface AudioChunkMessage {
  stream_id: string;
  seq: number;
  sample_rate: number;
  samples_b64: string;
  timestamp: number;
  duration_sec: number;
}

export interface DecodedPayloadEvent {
  event_type: string;
  stream_id: string;
  recovered_text: string | null;
  confidence: number;
  bit_count: number;
  preamble_found: boolean;
  timestamp: number;
}

export interface StreamStopResponse {
  status: string;
}

export interface StreamStartResponse {
  status: string;
  stream_id: string;
  source: string;
  sample_rate: number;
  window_sec: number;
}

export interface StreamStatusResponse {
  active: boolean;
  stream_id?: string;
  subscribers?: number;
  forensic_subscribers?: number;
  latest_forensics?: DecodedPayloadEvent[];
}
