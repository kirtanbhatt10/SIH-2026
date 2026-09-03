import { useEffect, useState } from "react";
import { ApiError } from "../services/api";
import { getSystemStatus } from "../services/systemStatus";
import type { SystemStatusResponse } from "../services/types";

export type SystemStatusLoadState = "loading" | "online" | "error";

export function useSystemStatus() {
  const [data, setData] = useState<SystemStatusResponse | null>(null);
  const [status, setStatus] = useState<SystemStatusLoadState>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    setStatus("loading");
    setError(null);

    getSystemStatus()
      .then((response) => {
        if (cancelled) return;
        setData(response);
        setStatus("online");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setData(null);
        setStatus("error");
        setError(err instanceof ApiError ? err.message : "Backend is offline or unreachable");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { data, status, error };
}
