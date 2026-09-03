import { useCallback, useEffect, useRef, useState } from "react";
import { adaptThreatEvent, type ThreatEventDisplay } from "../services/adapter";
import { ApiError } from "../services/api";
import type { DecodedPayloadEvent } from "../services/types";
import { getCurrentThreat, getThreats } from "../services/threats";
import { getStreamStatus, startMicStream, stopStream } from "../services/stream";

const THREAT_POLL_MS = 1500;

export type MonitoringSessionPhase =
  | "idle"
  | "starting"
  | "monitoring"
  | "stopping"
  | "stopped"
  | "error";

function applyInactiveStreamState(
  setPhase: (value: MonitoringSessionPhase | ((prev: MonitoringSessionPhase) => MonitoringSessionPhase)) => void,
  setStreamActive: (value: boolean) => void,
  setStreamId: (value: string | null) => void,
  clearPoll: () => void,
) {
  setStreamActive(false);
  setStreamId(null);
  setPhase((prev) => (prev === "monitoring" || prev === "starting" ? "stopped" : prev));
  clearPoll();
}

export function useMonitoringSession() {
  const [phase, setPhase] = useState<MonitoringSessionPhase>("idle");
  const [streamActive, setStreamActive] = useState(false);
  const [streamId, setStreamId] = useState<string | null>(null);
  const [latestForensics, setLatestForensics] = useState<DecodedPayloadEvent[]>([]);
  const [currentThreat, setCurrentThreat] = useState<ThreatEventDisplay | null>(null);
  const [threatCount, setThreatCount] = useState(0);
  const [threatCountAtStart, setThreatCountAtStart] = useState(0);
  const [sessionStartedAt, setSessionStartedAt] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastThreatKeyRef = useRef<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const threatFetchRef = useRef(0);
  const streamStatusFetchRef = useRef(0);
  const startInFlightRef = useRef(false);

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const refreshThreats = useCallback(async (): Promise<ThreatEventDisplay | null> => {
    const fetchId = ++threatFetchRef.current;
    try {
      const [listResponse, currentResponse] = await Promise.all([getThreats(), getCurrentThreat()]);
      if (fetchId !== threatFetchRef.current) {
        return null;
      }
      setThreatCount(listResponse.threats.length);
      const adapted = currentResponse.current ? adaptThreatEvent(currentResponse.current) : null;
      setCurrentThreat(adapted);
      setError(null);
      return adapted;
    } catch (err: unknown) {
      if (fetchId === threatFetchRef.current) {
        const message = err instanceof ApiError ? err.message : "Threat poll failed";
        setError(message);
      }
      throw err;
    }
  }, []);

  const syncStreamStatus = useCallback(async () => {
    const fetchId = ++streamStatusFetchRef.current;
    const status = await getStreamStatus();
    if (fetchId !== streamStatusFetchRef.current) {
      return status;
    }
    setLatestForensics(status.latest_forensics ?? []);
    if (!status.active) {
      applyInactiveStreamState(setPhase, setStreamActive, setStreamId, clearPoll);
      return status;
    }
    setStreamActive(true);
    setStreamId(status.stream_id ?? null);
    return status;
  }, [clearPoll]);

  const startPolling = useCallback(() => {
    clearPoll();
    pollRef.current = window.setInterval(() => {
      refreshThreats().catch(() => {
        // Error state updated inside refreshThreats for the latest fetch only.
      });
      syncStreamStatus().catch((err: unknown) => {
        const message = err instanceof ApiError ? err.message : "Stream status poll failed";
        setError(message);
      });
    }, THREAT_POLL_MS);
  }, [clearPoll, refreshThreats, syncStreamStatus]);

  const start = useCallback(async () => {
    if (
      startInFlightRef.current ||
      phase === "starting" ||
      phase === "monitoring" ||
      streamActive
    ) {
      return { success: false, streamId: streamId };
    }
    startInFlightRef.current = true;
    setError(null);
    setPhase("starting");
    lastThreatKeyRef.current = null;
    try {
      const startResponse = await startMicStream(1);
      const status = await syncStreamStatus();
      if (!status.active) {
        throw new Error("Backend did not report an active stream after start");
      }
      const activeStreamId = startResponse.stream_id ?? status.stream_id ?? null;
      setStreamId(activeStreamId);
      setStreamActive(true);
      setSessionStartedAt(Date.now() / 1000);
      const listResponse = await getThreats();
      setThreatCountAtStart(listResponse.threats.length);
      setPhase("monitoring");
      refreshThreats().catch((err: unknown) => {
        setError(err instanceof ApiError ? err.message : "Threat poll failed");
      });
      startPolling();
      return {
        success: true,
        streamId: activeStreamId,
      };
    } catch (err: unknown) {
      clearPoll();
      setStreamActive(false);
      setStreamId(null);
      setPhase("error");
      setError(err instanceof ApiError ? err.message : "Failed to start backend monitoring");
      return { success: false, streamId: null };
    } finally {
      startInFlightRef.current = false;
    }
  }, [clearPoll, phase, refreshThreats, startPolling, streamActive, streamId, syncStreamStatus]);

  const stop = useCallback(async () => {
    setPhase("stopping");
    clearPoll();
    try {
      await stopStream();
      const status = await syncStreamStatus();
      if (status.active) {
        throw new Error("Backend still reports an active stream after stop");
      }
      setPhase("stopped");
      setStreamActive(false);
      setStreamId(null);
      setError(null);
      return { success: true };
    } catch (err: unknown) {
      setPhase("error");
      setError(err instanceof ApiError ? err.message : "Failed to stop backend monitoring");
      return { success: false };
    }
  }, [clearPoll, syncStreamStatus]);

  const reset = useCallback(() => {
    clearPoll();
    setPhase("idle");
    setStreamActive(false);
    setStreamId(null);
    setLatestForensics([]);
    setCurrentThreat(null);
    setThreatCount(0);
    setThreatCountAtStart(0);
    setSessionStartedAt(null);
    setError(null);
    lastThreatKeyRef.current = null;
  }, [clearPoll]);

  useEffect(() => {
    return () => clearPoll();
  }, [clearPoll]);

  const isNewThreat = useCallback((threat: ThreatEventDisplay | null) => {
    if (!threat) return false;
    const key = threat.id;
    if (lastThreatKeyRef.current === key) return false;
    lastThreatKeyRef.current = key;
    return true;
  }, []);

  const hasSessionThreat =
    currentThreat?.source.detected === true &&
    sessionStartedAt !== null &&
    (threatCount > threatCountAtStart ||
      currentThreat.source.timestamp >= sessionStartedAt - 2);

  return {
    phase,
    streamActive,
    streamId,
    latestForensics,
    currentThreat,
    threatCount,
    threatCountAtStart,
    sessionStartedAt,
    hasSessionThreat,
    error,
    start,
    stop,
    reset,
    refreshThreats,
    isNewThreat,
    isBusy: phase === "starting" || phase === "stopping",
  };
}
