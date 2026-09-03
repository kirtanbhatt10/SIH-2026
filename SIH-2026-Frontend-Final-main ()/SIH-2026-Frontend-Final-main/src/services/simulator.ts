import { apiGet, apiPost } from "./api";
import type {
  AcousticExfiltrateRequest,
  AcousticExfiltrateResponse,
  GeneratePayloadRequest,
  GeneratePayloadResponse,
  SimulatorPayloadListItem,
  TransmitRequest,
  TransmitStatus,
} from "./types";

/** POST /simulate/generate — encode payload to BFSK WAV (in-memory + file). */
export function generatePayload(request: GeneratePayloadRequest): Promise<GeneratePayloadResponse> {
  return apiPost<GeneratePayloadResponse>("/simulate/generate", request);
}

/** POST /simulate/transmit — play through speaker (audio) or mark virtual complete. */
export function transmitPayload(request: TransmitRequest): Promise<TransmitStatus> {
  return apiPost<TransmitStatus>("/simulate/transmit", request);
}

/** GET /simulate/status/{payload_id} — current payload transmission state. */
export function getSimulationStatus(payloadId: string): Promise<TransmitStatus> {
  return apiGet<TransmitStatus>(`/simulate/status/${payloadId}`);
}

/** GET /simulate/list — all payloads held in backend memory. */
export function listSimulationPayloads(): Promise<SimulatorPayloadListItem[]> {
  return apiGet<SimulatorPayloadListItem[]>("/simulate/list");
}

/** POST /simulate/acoustic-exfiltrate — one-shot encode + optional emit. */
export function acousticExfiltrate(request: AcousticExfiltrateRequest): Promise<AcousticExfiltrateResponse> {
  return apiPost<AcousticExfiltrateResponse>("/simulate/acoustic-exfiltrate", request);
}

/** GET /simulate/diagnostics — runtime simulator/stream diagnostics. */
export function getSimulatorDiagnostics(): Promise<Record<string, unknown>> {
  return apiGet<Record<string, unknown>>("/simulate/diagnostics");
}
