import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useIntro } from "../../hooks/useIntro";
import { useWebGLSupport } from "../../hooks/useWebGLSupport";

type Stage = "dark" | "pulse-one" | "pause" | "pulse-two" | "waveform" | "brand" | "transition";
const schedule: Array<[Stage, number]> = [["dark", 0], ["pulse-one", 900], ["pause", 1550], ["pulse-two", 1810], ["waveform", 2580], ["brand", 3400], ["transition", 6200]];
const phrases = ["LISTEN. DETECT. DEFEND.", "SILENCE IS THE FIRST DEFENSE.", "WE SEE WHAT OTHERS MISS."];

export default function IntroSequence() {
  const { finishIntro, revealSite, skipIntro } = useIntro();
  const [stage, setStage] = useState<Stage>("dark");
  const [reduced, setReduced] = useState(false);
  const [phrase, setPhrase] = useState(0);
  const webglSupported = useWebGLSupport();
  const timers = useRef<number[]>([]);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const change = () => setReduced(media.matches);
    change(); media.addEventListener("change", change);
    return () => media.removeEventListener("change", change);
  }, []);
  useEffect(() => {
    if (reduced) { const id = window.setTimeout(finishIntro, 180); return () => clearTimeout(id); }
    const activeTimers = timers.current;
    schedule.forEach(([next, delay]) => activeTimers.push(window.setTimeout(() => { setStage(next); if (next === "transition") revealSite(); }, delay)));
    activeTimers.push(window.setTimeout(finishIntro, 7050));
    const phraseTimer = window.setInterval(() => setPhrase((value) => (value + 1) % phrases.length), 1300);
    return () => { activeTimers.forEach(clearTimeout); clearInterval(phraseTimer); };
  }, [finishIntro, reduced, revealSite]);
  useEffect(() => { document.body.style.overflow = "hidden"; return () => { document.body.style.overflow = ""; }; }, []);
  if (reduced) return <div className="intro-overlay" aria-hidden="true"><div className="intro-static-brand">THE SILENT DOG'S WHISTLE</div></div>;
  const active = stage !== "dark";
  const waveform = ["waveform", "brand", "transition"].includes(stage);
  const brand = ["brand", "transition"].includes(stage);
  const moving = stage === "transition";
  return <div className={`intro-overlay intro-${stage} ${webglSupported ? "" : "webgl-fallback"}`} aria-hidden={moving}>
    <div className="intro-stage-frame" />
    <div className="intro-grid" /><div className="signal-core" />
    <div className={`pulse pulse-one ${active ? "is-active" : ""}`} /><div className={`pulse pulse-two ${stage === "pulse-two" || waveform ? "is-active" : ""}`} />
    {waveform && <div className="intro-wave" />}
    {brand && <motion.div layoutId="silent-dog-brand" className="intro-brand" animate={moving ? { opacity: 0, y: -18 } : { opacity: 1, y: 0 }} transition={{ duration: 0.65, ease: [0.65, 0, 0.35, 1] }}><span>THE</span><strong>SILENT DOG'S WHISTLE</strong></motion.div>}
    {stage !== "transition" && <p className="intro-phrase">{phrases[phrase]}</p>}
    {stage !== "transition" && <button className="intro-skip" onClick={skipIntro}>SKIP INTRO <span>→</span></button>}
  </div>;
}
