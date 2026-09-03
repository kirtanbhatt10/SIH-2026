import { useCallback, useEffect, useRef, useState } from "react";
import { getStreamWebSocketUrl } from "../services/config";
import type { AudioChunkMessage } from "../services/types";
import { decodeBase64Float32, LiveAudioBuffer } from "../utils/liveAudioBuffer";

const BUFFER_CAPACITY = 48000 * 2;

export type LiveAudioConnectionStatus = "idle" | "connecting" | "connected" | "disconnected" | "error";

export interface LiveAudioStreamSnapshot {
  sampleRate: number;
  chunksReceived: number;
  samplesBuffered: number;
  connectionStatus: LiveAudioConnectionStatus;
}

export function useLiveAudioStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const bufferRef = useRef(new LiveAudioBuffer(BUFFER_CAPACITY));
  const sampleRateRef = useRef(48000);
  const chunksReceivedRef = useRef(0);
  const [connectionStatus, setConnectionStatus] = useState<LiveAudioConnectionStatus>("idle");
  const [snapshot, setSnapshot] = useState<LiveAudioStreamSnapshot>({
    sampleRate: 48000,
    chunksReceived: 0,
    samplesBuffered: 0,
    connectionStatus: "idle",
  });

  const updateSnapshot = useCallback((status: LiveAudioConnectionStatus) => {
    setSnapshot({
      sampleRate: sampleRateRef.current,
      chunksReceived: chunksReceivedRef.current,
      samplesBuffered: bufferRef.current.filledCount,
      connectionStatus: status,
    });
  }, []);

  const disconnect = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    }
    bufferRef.current.clear();
    chunksReceivedRef.current = 0;
    sampleRateRef.current = 48000;
    setConnectionStatus("idle");
    updateSnapshot("idle");
  }, [updateSnapshot]);

  const connect = useCallback(() => {
    disconnect();
    setConnectionStatus("connecting");
    updateSnapshot("connecting");

    const ws = new WebSocket(getStreamWebSocketUrl("/stream/audio"));
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus("connected");
      updateSnapshot("connected");
    };

    ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const message = JSON.parse(event.data) as AudioChunkMessage;
        if (!message.samples_b64) return;
        const samples = decodeBase64Float32(message.samples_b64);
        bufferRef.current.push(samples);
        sampleRateRef.current = message.sample_rate || 48000;
        chunksReceivedRef.current += 1;
        if (chunksReceivedRef.current % 4 === 0) {
          updateSnapshot("connected");
        }
      } catch {
        // Ignore malformed packets; REST lifecycle handles hard failures.
      }
    };

    ws.onerror = () => {
      setConnectionStatus("error");
      updateSnapshot("error");
    };

    ws.onclose = () => {
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
      setConnectionStatus((prev) => {
        const next = prev === "error" ? "error" : "disconnected";
        updateSnapshot(next);
        return next;
      });
    };
  }, [disconnect, updateSnapshot]);

  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  return {
    bufferRef,
    sampleRateRef,
    connectionStatus,
    snapshot,
    connect,
    disconnect,
  };
}
