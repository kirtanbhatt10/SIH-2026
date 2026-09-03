import { Canvas } from "@react-three/fiber";
import { Component, Suspense, useEffect, useState, type ReactNode } from "react";
import { WidowmakerSpider } from "./WidowmakerSpider";

class SpiderBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error: unknown) { console.warn("THE SILENT DOG'S WHISTLE spider disabled: the Widowmaker model could not load.", error); }
  render() { return this.state.failed ? null : this.props.children; }
}

function useSpiderEligibility(introPlaying: boolean, showSite: boolean) {
  const [eligible, setEligible] = useState(false);

  useEffect(() => {
    setEligible(false);
    if (introPlaying || !showSite || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const eligibilityTimer = window.setTimeout(() => setEligible(true), 2000);
    return () => window.clearTimeout(eligibilityTimer);
  }, [introPlaying, showSite]);

  return eligible;
}

/**
 * Safety-first mount point. It is absent during the intro/replay and while the
 * approved asset is missing, so it can never display a placeholder creature.
 */
export default function GlobalSpiderSystem({ introPlaying, showSite }: { introPlaying: boolean; showSite: boolean }) {
  const enabled = useSpiderEligibility(introPlaying, showSite);
  if (!enabled) return null;

  return <SpiderBoundary>
    <div className="widowmaker-layer" aria-hidden="true">
      <Canvas dpr={[1, 1.5]} shadows gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }} camera={{ position: [0, 0, 6], fov: 42, near: .1, far: 30 }}>
        <hemisphereLight args={["#efe6dc", "#060505", 1.45]} />
        <directionalLight position={[3, 4, 5]} intensity={4} color="#fff5e7" castShadow />
        <pointLight position={[-2.5, 1.8, 2]} intensity={1.45} color="#b70e17" />
        <directionalLight position={[-4, 2, -2]} intensity={.9} color="#8f1018" />
        <pointLight position={[1.5, -2, 1]} intensity={.22} color="#7a8a71" />
        <Suspense fallback={null}><WidowmakerSpider /></Suspense>
      </Canvas>
    </div>
  </SpiderBoundary>;
}
