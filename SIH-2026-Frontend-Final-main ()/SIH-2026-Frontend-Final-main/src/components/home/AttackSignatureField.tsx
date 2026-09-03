import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import Waveform from "../viz/Waveform";

const signatures = [
  { id: "SIG-04", title: "ENVIRONMENTAL BASELINE", mode: "normal" as const, signal: "Known ventilation / ambient field", response: "MONITOR", detail: "Retain as trusted baseline." },
  { id: "SIG-21", title: "SPOOFED BEACON", mode: "anomaly" as const, signal: "Repeating acoustic marker", response: "VALIDATE", detail: "Cross-check node origin and cadence." },
  { id: "SIG-11", title: "MECHANICAL INTRUSION", mode: "threat" as const, signal: "Low-frequency rotary signature", response: "CONTAIN", detail: "Lock, correlate, and route to defense." },
];

export default function AttackSignatureField() {
  const fieldRef = useRef<HTMLElement>(null);
  const active = useInView(fieldRef, { amount: 0.3, once: false });
  return <section ref={fieldRef} className={`attack-signature-field ${active ? "is-active" : ""}`}>
    <div className="attack-signature-heading"><div><p className="mono-label">03 / CLASSIFICATION FIELD</p><h2>NOISE HAS<br />A SIGNATURE.</h2></div><p>Signals enter one by one. THE SILENT DOG'S WHISTLE compares each against the active field, assigns its risk, and arms the matching defense response.</p></div>
    <div className="signature-manifest"><span>INCOMING SIGNAL QUEUE</span><span>CLASSIFICATION ROUTE</span><span>DEFENSE POSTURE</span></div>
    <div className="signature-targets">
      {signatures.map((signature, index) => <motion.article key={signature.id} className={`signature-target ${signature.mode}`} initial={{ opacity: 0, x: -130, y: 8 }} animate={active ? { opacity: 1, x: 0, y: 0 } : { opacity: 0, x: -130, y: 8 }} transition={{ duration: 0.72, delay: active ? 0.35 + index * 0.65 : 0, ease: [0.16, 1, 0.3, 1] }}>
        <div className="signature-route"><span>{signature.id}</span><i /></div>
        <header><span>{index + 1 < 10 ? `0${index + 1}` : index + 1}</span><b>{signature.mode}</b></header>
        <Waveform mode={signature.mode} height={90} active={active} />
        <h3>{signature.title}</h3><p>{signature.signal}</p>
        <footer><span>DEFENSE</span><b>{signature.response}</b><small>{signature.detail}</small></footer>
      </motion.article>)}
    </div>
    <p className="attack-field-note">SIMULATION DATA ONLY — classifications demonstrate the interface flow and do not represent live defensive actions.</p>
  </section>;
}
