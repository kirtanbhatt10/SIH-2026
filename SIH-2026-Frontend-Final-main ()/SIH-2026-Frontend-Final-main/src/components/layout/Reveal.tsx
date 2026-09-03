import { useEffect, useRef, useState, type ReactNode } from "react";
import { useFrontendSettings } from "../../hooks/useFrontendSettings";

export default function Reveal({
  children,
  className = "",
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  const { settings } = useFrontendSettings();
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { threshold: 0.15 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`${settings.reduceMotion ? "" : "transition-all duration-700 ease-out"} ${visible ? "translate-y-0 opacity-100" : settings.reduceMotion ? "opacity-0" : "translate-y-6 opacity-0"} ${className}`}
      style={{ transitionDelay: settings.reduceMotion ? "0ms" : `${delay}ms` }}
    >
      {children}
    </div>
  );
}
