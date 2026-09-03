import { useEffect, useState } from "react";
import { adaptThreatEvent, type ThreatEventDisplay } from "../services/adapter";
import { ApiError } from "../services/api";
import { getCurrentThreat, getThreats } from "../services/threats";

export type ThreatLoadState = "loading" | "success" | "error";

export function useThreatEvents() {
  const [events, setEvents] = useState<ThreatEventDisplay[]>([]);
  const [current, setCurrent] = useState<ThreatEventDisplay | null>(null);
  const [status, setStatus] = useState<ThreatLoadState>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    setStatus("loading");
    setError(null);

    Promise.all([getThreats(), getCurrentThreat()])
      .then(([listResponse, currentResponse]) => {
        if (cancelled) return;
        setEvents(listResponse.threats.map(adaptThreatEvent));
        setCurrent(currentResponse.current ? adaptThreatEvent(currentResponse.current) : null);
        setStatus("success");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setStatus("error");
        setError(err instanceof ApiError ? err.message : "Failed to load threat events from backend");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { events, current, status, error };
}
