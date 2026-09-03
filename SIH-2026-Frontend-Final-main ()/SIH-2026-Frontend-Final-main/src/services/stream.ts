import { apiGet, apiPost } from "./api";
import type { StreamStartResponse, StreamStatusResponse, StreamStopResponse } from "./types";

export type { StreamStopResponse };

/** POST /stream/start — begin live microphone capture (query params, empty JSON body). */
export function startMicStream(deviceId = 1): Promise<StreamStartResponse> {
  return apiPost<StreamStartResponse>(`/stream/start?source=mic&device_id=${deviceId}`, {});
}

/** POST /stream/stop — end active stream. */
export function stopStream(): Promise<StreamStopResponse> {
  return apiPost<StreamStopResponse>("/stream/stop", {});
}

/** GET /stream/status — whether a stream task is active on the backend. */
export function getStreamStatus(): Promise<StreamStatusResponse> {
  return apiGet<StreamStatusResponse>("/stream/status");
}
