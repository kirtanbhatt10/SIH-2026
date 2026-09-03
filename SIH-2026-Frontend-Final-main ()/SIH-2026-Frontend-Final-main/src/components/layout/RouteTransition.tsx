import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

// A brief crimson scan-line sweep on route change — the same visual language
// as the intro's activation scan, kept subtle and non-blocking.
export default function RouteTransition() {
  const location = useLocation();
  const [key, setKey] = useState(0);
  const [show, setShow] = useState(false);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) return;
    setShow(true);
    setKey((k) => k + 1);
    const t = window.setTimeout(() => setShow(false), 600);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);

  if (!show) return null;

  return (
    <div
      key={key}
      className="route-scan-line pointer-events-none fixed left-0 right-0 top-0 z-[90] h-px"
      style={{
        background: "linear-gradient(to right, transparent, #ff1e2d, transparent)",
        boxShadow: "0 0 20px rgba(255,30,45,0.5)",
      }}
    />
  );
}
