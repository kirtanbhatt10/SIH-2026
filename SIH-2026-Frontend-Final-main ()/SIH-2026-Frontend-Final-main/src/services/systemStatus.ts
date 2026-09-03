import { apiGet } from "./api";
import type { SystemStatusResponse } from "./types";

/**
 * Backend connectivity check — GET /api/system-status.
 * Throws ApiError when the backend is offline or returns an error.
 */
export function getSystemStatus(): Promise<SystemStatusResponse> {
  return apiGet<SystemStatusResponse>("/api/system-status");
}
