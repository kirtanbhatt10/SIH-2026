import { useEffect, useState } from "react";

/** Detects WebGL once so the visual layer can choose a safe CSS fallback. */
export function useWebGLSupport() {
  const [supported, setSupported] = useState(true);

  useEffect(() => {
    try {
      const canvas = document.createElement("canvas");
      setSupported(Boolean(window.WebGLRenderingContext && (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))));
    } catch {
      setSupported(false);
    }
  }, []);

  return supported;
}
