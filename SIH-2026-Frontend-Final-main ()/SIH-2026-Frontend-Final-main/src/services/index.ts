export { getApiBaseUrl, getStreamWebSocketUrl } from "./config";
export { apiGet, apiPost, ApiError, type ApiErrorKind, type ApiRequestOptions } from "./api";
export {
  adaptThreatEvent,
  severityToDashboardStatus,
  suspicionScoreToDisplayScore,
  threatEventToClientId,
  type ThreatEventDisplay,
} from "./adapter";
export { getSystemStatus } from "./systemStatus";
export { getCurrentThreat, getThreats } from "./threats";
export { getStreamStatus, startMicStream, stopStream } from "./stream";
export {
  acousticExfiltrate,
  generatePayload,
  getSimulationStatus,
  listSimulationPayloads,
  transmitPayload,
} from "./simulator";
export type {
  AcousticExfiltrateRequest,
  AcousticExfiltrateResponse,
  AnalyzeThreatResponse,
  AudioChunkMessage,
  CurrentThreatResponse,
  DecodedPayloadEvent,
  GeneratePayloadRequest,
  GeneratePayloadResponse,
  HealthResponse,
  PatternType,
  RiskLevel,
  SimulatorPayloadListItem,
  StreamStartResponse,
  StreamStatusResponse,
  StreamStopResponse,
  SystemStatusResponse,
  ThreatEvent,
  ThreatsListResponse,
  TransmitRequest,
  TransmitStatus,
} from "./types";
