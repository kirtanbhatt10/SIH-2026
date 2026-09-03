import { getApiBaseUrl } from "./config";

export type ApiErrorKind = "http" | "network" | "malformed";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly body?: unknown;

  constructor(message: string, kind: ApiErrorKind, status?: number, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.body = body;
  }
}

export interface ApiRequestOptions {
  signal?: AbortSignal;
  headers?: Record<string, string>;
}

function buildUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalizedPath}`;
}

async function parseJsonBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (text.length === 0) {
    return null;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new ApiError(
      `Malformed JSON response from ${response.url}`,
      "malformed",
      response.status,
      text,
    );
  }
}

async function request<T>(
  method: "GET" | "POST",
  path: string,
  body?: unknown,
  options: ApiRequestOptions = {},
): Promise<T> {
  const url = buildUrl(path);
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...options.headers,
  };

  let response: Response;

  try {
    response = await fetch(url, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json", ...headers } : headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: options.signal,
    });
  } catch (error) {
    const message =
      error instanceof Error
        ? `Network error reaching backend at ${url}: ${error.message}`
        : `Network error reaching backend at ${url}`;
    throw new ApiError(message, "network");
  }

  const parsed = await parseJsonBody(response);

  if (!response.ok) {
    throw new ApiError(
      `HTTP ${response.status} from ${path}`,
      "http",
      response.status,
      parsed,
    );
  }

  return parsed as T;
}

/** GET JSON from the backend API. Throws ApiError on failure — never returns mock data. */
export function apiGet<T>(path: string, options?: ApiRequestOptions): Promise<T> {
  return request<T>("GET", path, undefined, options);
}

/** POST JSON to the backend API. Throws ApiError on failure — never returns mock data. */
export function apiPost<T>(
  path: string,
  body: unknown,
  options?: ApiRequestOptions,
): Promise<T> {
  return request<T>("POST", path, body, options);
}
