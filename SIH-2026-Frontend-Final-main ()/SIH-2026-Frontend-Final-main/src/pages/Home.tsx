import { Link } from "react-router-dom";
import type { CSSProperties } from "react";
import { useRef } from "react";
import { useInView } from "framer-motion";
import IntroSequence from "../components/intro/IntroSequence";
import Waveform from "../components/viz/Waveform";
import Spectrum from "../components/viz/Spectrum";
import RadarSweep from "../components/viz/RadarSweep";
import AttackSignatureField from "../components/home/AttackSignatureField";
import { useIntro } from "../hooks/useIntro";
import AcousticBackground from "../components/three/AcousticBackground";

function WaveWord({ children }: { children: string }) {
  return <strong className="wave-word" aria-label={children}>{[...children].map((letter, index) => <span key={`${letter}-${index}`} style={{ "--letter": index } as CSSProperties}>{letter}</span>)}</strong>;
}

export default function Home() {
  const { introPlaying, showSite, replayIntro } = useIntro();
  const pipelineRef = useRef<HTMLElement>(null);
  const pipelineInView = useInView(pipelineRef, { amount: 0.35, once: false });
  return <div>
    {introPlaying && <IntroSequence />}
    <AcousticBackground introPlaying={introPlaying} showSite={showSite} />
    <section className="silent-hero" style={{ opacity: showSite ? 1 : 0 }}>
      <div className="hero-grid" />
      <div className="hero-content">
        <p className="mono-label hero-status"><i /> SIGNAL INTELLIGENCE · SYSTEM NOMINAL</p>
        <p className="hero-eyebrow">THE SILENT DOG'S WHISTLE / DEFENSE NETWORK</p>
        <h1>HEAR THE THREAT<br />BEFORE IT STRIKES.</h1>
        <p className="hero-copy">Signal intelligence, anomaly detection and autonomous acoustic defense built for environments where silence carries information.</p>
        <div className="hero-actions"><Link className="command-button" to="/monitoring">ENTER COMMAND CENTER <span>→</span></Link><Link className="text-button" to="/intelligence">EXPLORE INTELLIGENCE</Link></div>
      </div>
      <div className="hero-telemetry"><span>NODE 01</span><span>CHANNEL SECURE</span><span>ACTIVE SENSORS 06</span></div>
      <button className="replay-button" onClick={replayIntro}>REPLAY INTRO</button>
    </section>
    <section ref={pipelineRef} className={`home-intelligence ${pipelineInView ? "pipeline-active" : ""}`}>
      <div className="pipeline-copy"><p className="mono-label">01 / SIGNAL INTELLIGENCE</p><h2>FROM A WHISPER<br />TO A DECISION.</h2><p>A single waveform is carried through the defense chain. Each stop gives the signal more context before it becomes an action.</p></div>
      <ol className="wave-pipeline" aria-label="Signal intelligence process">
        <li><b>01</b><div><span className="wave-tail" /><WaveWord>CAPTURE</WaveWord><p>Acquire the raw signal from active sensors.</p></div></li>
        <li><b>02</b><div><span className="wave-tail" /><WaveWord>FILTER</WaveWord><p>Remove known baseline and environmental noise.</p></div></li>
        <li><b>03</b><div><span className="wave-tail" /><WaveWord>CLASSIFY</WaveWord><p>Identify pattern type, risk, and confidence.</p></div></li>
        <li><b>04</b><div><span className="wave-tail" /><WaveWord>CORRELATE</WaveWord><p>Compare across nodes, time, and context.</p></div></li>
        <li><b>05</b><div><span className="wave-tail" /><WaveWord>RESPOND</WaveWord><p>Route confirmed intelligence to defense.</p></div></li>
      </ol>
    </section>
    <section className="home-telemetry">
      <div className="telemetry-heading"><p className="mono-label">02 / SIMULATED INTELLIGENCE</p><h2>OBSERVE THE<br />INVISIBLE.</h2><p>Sample frontend telemetry demonstrates the Acoustic Shield interface. Future API adapters can replace this simulated data stream without changing the visualization layer.</p></div>
      <div className="telemetry-grid">
        <article className="telemetry-card telemetry-wave"><header><span>CHANNEL 04 / WAVEFORM</span><b>DEMO</b></header><Waveform mode="mixed" height={150} /></article>
        <article className="telemetry-card"><header><span>SPECTRUM / 2.4 GHz</span><b className="threat-label">THREAT LOCK</b></header><Spectrum height={180} threatIndex={38} anomalyIndex={19} /></article>
        <article className="telemetry-card telemetry-radar"><header><span>NODE TOPOLOGY</span><b>DEMO READY</b></header><RadarSweep size={240} /></article>
        <article className="telemetry-card event-card"><header><span>EVENT STREAM</span><b>NOW</b></header><div className="event-stream"><p><time>12:41:02</time> NODE-04 <strong>SIGNAL DETECTED</strong></p><p><time>12:41:04</time> CORE <span>ANALYSIS STARTED</span></p><p><time>12:41:05</time> NODE-02 <span>CORRELATION FOUND</span></p><p><time>12:41:07</time> CORE <strong>THREAT SCORE 0.82</strong></p></div></article>
        <article className="telemetry-card attack-card"><header><span>SIMULATED ATTACK VECTOR</span><b className="threat-label">SIG-11</b></header><h3>MECHANICAL INTRUSION</h3><p>Low-frequency rotary cadence injected at the perimeter to imitate a concealed drive or unmanned platform approaching NODE-04.</p><dl><div><dt>ENTRY</dt><dd>NODE-04 / EAST CORRIDOR</dd></div><div><dt>SIGNATURE</dt><dd>78–132 Hz / REPEATING ROTOR HARMONICS</dd></div><div><dt>OBJECTIVE</dt><dd>TEST EARLY DETECTION BEFORE VISUAL CONFIRMATION</dd></div></dl></article>
        <article className="telemetry-card defense-card"><header><span>DEFENSE PLAYBOOK</span><b>ROUTE-04</b></header><ol><li><span>01</span><p><b>ISOLATE BASELINE</b>Discard trusted ambient sources from the active channel.</p></li><li><span>02</span><p><b>CORRELATE NODES</b>Require matching cadence across NODE-04 and CORE relay.</p></li><li><span>03</span><p><b>ESCALATE RESPONSE</b>Lock the signature and create an operator review event.</p></li></ol><div className="defense-ready"><i /> SIMULATED CONTAINMENT ROUTE READY</div></article>
      </div>
    </section>
    <AttackSignatureField />
    <section className="capability-links">
      <p className="mono-label">04 / COMMAND CAPABILITIES</p>
      <Link to="/monitoring"><span>01</span><b>LIVE MONITORING</b><em>Continuous sensor coverage and signal visibility.</em><i>→</i></Link>
      <Link to="/intelligence"><span>02</span><b>THREAT INTELLIGENCE</b><em>Classify, correlate, and explain anomalies.</em><i>→</i></Link>
      <Link to="/attack-lab"><span>03</span><b>ATTACK LAB</b><em>Test defensive response against controlled simulations.</em><i>→</i></Link>
    </section>
    <section className="home-sentinel">
      <p className="mono-label">CONTINUOUS MONITORING / ACTIVE SENSORS 06</p>
      <h2>THE DOG<br /><span>NEVER SLEEPS.</span></h2>
      <p>Track subtle anomalies through a single operational field designed to keep the signal clear and the operator focused.</p>
      <Link className="text-button monitor-link" to="/monitoring"><span>VIEW LIVE MONITORING</span><i>→</i></Link>
    </section>
  </div>;
}
