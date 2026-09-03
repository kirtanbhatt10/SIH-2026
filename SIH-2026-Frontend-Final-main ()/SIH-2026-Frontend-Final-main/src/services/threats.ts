import { apiGet } from "./api";
import type { CurrentThreatResponse, ThreatsListResponse } from "./types";

/** GET /api/threats — full threat history in arrival order. */
export function getThreats(): Promise<ThreatsListResponse> {
  return apiGet<ThreatsListResponse>("/api/threats");
}

/** GET /api/threats/current — latest stored threat or null. */
export function getCurrentThreat(): Promise<CurrentThreatResponse> {
  return apiGet<CurrentThreatResponse>("/api/threats/current");
}
