const DEFAULT_API_BASE_URL = "http://127.0.0.1:8021";

/**
 * Backend REST base URL (no trailing slash).
 * Configured via VITE_API_BASE_URL; falls back to the team integration default.
 */
export function getApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  const base = configured && configured.length > 0 ? configured : DEFAULT_API_BASE_URL;
  return base.replace(/\/+$/, "");
}

/** WebSocket URL for backend stream endpoints (e.g. /stream/audio). */
export function getStreamWebSocketUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const httpBase = getApiBaseUrl();
  const wsBase = httpBase.replace(/^http/i, "ws");
  return `${wsBase}${normalizedPath}`;
}
