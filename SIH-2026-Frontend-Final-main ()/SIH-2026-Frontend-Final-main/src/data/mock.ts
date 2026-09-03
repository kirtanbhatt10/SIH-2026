// -----------------------------------------------------------------------
// DEMO / VISUALIZATION DATA ONLY.
// This file exists because THE SILENT DOG'S WHISTLE has no backend in this build.
// Every export here is structured so a real API layer can replace it
// without touching component code — see src/data/README below.
// -----------------------------------------------------------------------

export type Severity = "normal" | "anomaly" | "threat";
export type DashboardStatus = "safe" | "suspicious" | "high-risk";
export type MonitoringState = "idle" | "listening" | "signal-detected" | "analyzing" | "classifying" | "result" | "stopped";
export type ComponentStatus = "IDLE" | "READY" | "ACTIVE" | "ANALYZING" | "COMPLETE" | "STOPPED";

export interface SignalEvent {
  id: string;
  time: string;
  timestamp: string;
  label: string;
  frequency: string;
  frequencyStart: string;
  frequencyEnd: string;
  signalStrength: string;
  duration: string;
  severity: Severity;
  confidence: string;
  classification: string;
  threatScore: number;
}

export const dashboardStatusCopy: Record<DashboardStatus, { label: string; description: string }> = {
  safe: { label: "SYSTEM SAFE", description: "No suspicious acoustic communication detected" },
  suspicious: { label: "SUSPICIOUS ACOUSTIC ACTIVITY", description: "Signal analysis required" },
  "high-risk": { label: "POTENTIAL COVERT ACOUSTIC COMMUNICATION", description: "Simulated signal pattern requires analyst review" },
};

export interface SensorNode {
  id: string;
  label: string;
  x: number; // 0-100, percentage position
  y: number;
  status: "active" | "degraded";
}

export const liveEvents: SignalEvent[] = [
  { id: "EVT-1041", time: "14:52:03", timestamp: "2026-08-19 14:52:03", label: "Signal acquired", frequency: "2.41 GHz", frequencyStart: "2.39 GHz", frequencyEnd: "2.43 GHz", signalStrength: "-58 dBm", duration: "1.1 s", severity: "normal", confidence: "38%", classification: "Normal environmental signal", threatScore: 12 },
  { id: "EVT-1042", time: "14:52:11", timestamp: "2026-08-19 14:52:11", label: "Baseline drift detected", frequency: "2.43 GHz", frequencyStart: "2.40 GHz", frequencyEnd: "2.46 GHz", signalStrength: "-49 dBm", duration: "2.2 s", severity: "anomaly", confidence: "66%", classification: "Acoustic anomaly", threatScore: 46 },
  { id: "EVT-1043", time: "14:52:19", timestamp: "2026-08-19 14:52:19", label: "Pattern match — elevated", frequency: "2.41 GHz", frequencyStart: "2.38 GHz", frequencyEnd: "2.44 GHz", signalStrength: "-45 dBm", duration: "3.1 s", severity: "anomaly", confidence: "78%", classification: "Communication-like pattern", threatScore: 63 },
  { id: "EVT-1044", time: "14:52:26", timestamp: "2026-08-19 14:52:26", label: "Classification: suspicious signal", frequency: "2.41 GHz", frequencyStart: "2.38 GHz", frequencyEnd: "2.44 GHz", signalStrength: "-42 dBm", duration: "4.2 s", severity: "threat", confidence: "94%", classification: "Suspicious pulsed signal", threatScore: 87 },
  { id: "EVT-1045", time: "14:52:34", timestamp: "2026-08-19 14:52:34", label: "Node 04 signal loss, recovered", frequency: "—", frequencyStart: "—", frequencyEnd: "—", signalStrength: "-71 dBm", duration: "0.4 s", severity: "normal", confidence: "91%", classification: "Normal environmental signal", threatScore: 8 },
  { id: "EVT-1046", time: "14:52:41", timestamp: "2026-08-19 14:52:41", label: "Harmonic anomaly, sector 3", frequency: "5.80 GHz", frequencyStart: "5.76 GHz", frequencyEnd: "5.84 GHz", signalStrength: "-47 dBm", duration: "2.8 s", severity: "anomaly", confidence: "72%", classification: "Acoustic anomaly", threatScore: 54 },
  { id: "EVT-1047", time: "14:52:49", timestamp: "2026-08-19 14:52:49", label: "Potential covert acoustic pattern logged", frequency: "2.41 GHz", frequencyStart: "2.38 GHz", frequencyEnd: "2.44 GHz", signalStrength: "-42 dBm", duration: "4.2 s", severity: "threat", confidence: "94%", classification: "Communication-like pattern", threatScore: 87 },
  { id: "EVT-1048", time: "14:52:57", timestamp: "2026-08-19 14:52:57", label: "Sensor calibration complete", frequency: "—", frequencyStart: "—", frequencyEnd: "—", signalStrength: "-68 dBm", duration: "0.8 s", severity: "normal", confidence: "98%", classification: "Normal environmental signal", threatScore: 4 },
];

export const sensorNodes: SensorNode[] = [
  { id: "A", label: "NODE A", x: 18, y: 28, status: "active" },
  { id: "B", label: "NODE B", x: 72, y: 20, status: "active" },
  { id: "C", label: "NODE C", x: 82, y: 68, status: "active" },
  { id: "D", label: "NODE D", x: 28, y: 76, status: "degraded" },
];

export const signalCharacteristics = {
  frequency: "2.41 GHz",
  amplitude: "-42 dBm",
  bandwidth: "18 MHz",
  duration: "4.2 s",
  peak: "-31 dBm",
  signalType: "Pulsed / intermittent",
  status: "Under analysis",
};

export const threatScore = 87;

/** Frontend-only dashboard state. Replace with real detector data in a later integration step. */
export const dashboardDemo = {
  status: "high-risk" as DashboardStatus,
  confidence: "94%",
  dominantFrequency: "2.41 GHz",
  signalStrength: "-42 dBm",
  duration: "4.2 s",
  classification: "Suspicious pulsed signal",
  threatScore,
  spectrum: { threatIndex: 40, anomalyIndex: 20, bars: 56 },
  recentEventIds: ["EVT-1047", "EVT-1046", "EVT-1044", "EVT-1043", "EVT-1045"],
};

export const severityLabels: Record<Severity, "SAFE" | "SUSPICIOUS" | "HIGH RISK"> = {
  normal: "SAFE",
  anomaly: "SUSPICIOUS",
  threat: "HIGH RISK",
};

/** Frontend-only monitoring values; no microphone, DSP, AI, or backend is connected. */
export const monitoringDemoAnalysis = {
  dominantFrequency: "2.41 GHz",
  frequencyRange: "2.38–2.44 GHz",
  signalStrength: "-42 dBm",
  duration: "4.2 s",
  confidence: "94%",
  classification: "Suspicious pulsed signal",
  threatLevel: "HIGH RISK",
  threatScore,
  spectrum: { threatIndex: 44, anomalyIndex: 22, bars: 48 },
};

export const monitoringComponentStatuses: Record<MonitoringState, Record<"Acoustic Input" | "DSP Engine" | "AI Detector" | "Monitoring Engine", ComponentStatus>> = {
  idle: { "Acoustic Input": "IDLE", "DSP Engine": "READY", "AI Detector": "READY", "Monitoring Engine": "IDLE" },
  listening: { "Acoustic Input": "ACTIVE", "DSP Engine": "ACTIVE", "AI Detector": "READY", "Monitoring Engine": "ACTIVE" },
  "signal-detected": { "Acoustic Input": "ACTIVE", "DSP Engine": "ANALYZING", "AI Detector": "READY", "Monitoring Engine": "ACTIVE" },
  analyzing: { "Acoustic Input": "ACTIVE", "DSP Engine": "ANALYZING", "AI Detector": "READY", "Monitoring Engine": "ACTIVE" },
  classifying: { "Acoustic Input": "ACTIVE", "DSP Engine": "COMPLETE", "AI Detector": "ANALYZING", "Monitoring Engine": "ACTIVE" },
  result: { "Acoustic Input": "ACTIVE", "DSP Engine": "COMPLETE", "AI Detector": "COMPLETE", "Monitoring Engine": "COMPLETE" },
  stopped: { "Acoustic Input": "STOPPED", "DSP Engine": "STOPPED", "AI Detector": "STOPPED", "Monitoring Engine": "STOPPED" },
};

/** Default controls for the frontend-only controlled simulation. */
export const simulatorDefaults = {
  payload: "SESSION_TOKEN=SIH2026",
  maxPayloadLength: 160,
  carrierFrequency: "20.0 kHz",
  duration: "5 sec",
  modulation: "FSK",
  transmissionMode: "Acoustic Simulation",
  resultEventId: "EVT-1047",
};

export type DemoReadiness = "READY" | "ACTIVE" | "AVAILABLE" | "LOADED" | "PENDING" | "NOT CONNECTED";

/** Frontend-only diagnostics. These values do not represent live backend, device, or service checks. */
export const systemDemoStatus = {
  overall: { label: "DEMO ENVIRONMENT READY", description: "Frontend simulation modules are available for presentation." },
  modules: [
    { id: "monitor", name: "Monitoring Interface", status: "READY" as DemoReadiness, description: "Interactive local monitoring lifecycle." },
    { id: "dsp", name: "DSP Visualization Module", status: "AVAILABLE" as DemoReadiness, description: "Simulated waveform, spectrum, and spectrogram views." },
    { id: "ai", name: "AI Detection Interface", status: "READY" as DemoReadiness, description: "Frontend classification presentation states." },
    { id: "events", name: "Threat Event Engine", status: "READY" as DemoReadiness, description: "Centralized simulated event-history presentation." },
    { id: "simulator", name: "Attack Simulator", status: "READY" as DemoReadiness, description: "Controlled local acoustic-test demonstration." },
    { id: "visual", name: "Visualization Engine", status: "LOADED" as DemoReadiness, description: "Canvas and 3D presentation layers available." },
    { id: "routing", name: "Routing / Navigation", status: "ACTIVE" as DemoReadiness, description: "Frontend routes and responsive navigation enabled." },
    { id: "dataset", name: "Demo Dataset", status: "LOADED" as DemoReadiness, description: "Centralized frontend simulation records." },
  ],
  platform: [
    ["Application Mode", "Simulation"], ["Frontend Framework", "React"], ["Language", "TypeScript"], ["Build Tool", "Vite"],
    ["Routing", "React Router"], ["Visualization Layer", "Canvas / Three.js available"], ["Demo Dataset", "Loaded"], ["Current Demo State", "Presentation ready"],
  ],
  readiness: [
    ["Dashboard", "READY" as DemoReadiness], ["Live Monitor", "READY" as DemoReadiness], ["Threat Events", "READY" as DemoReadiness], ["Detection Details", "READY" as DemoReadiness],
    ["Attack Simulator", "READY" as DemoReadiness], ["System Status", "ACTIVE" as DemoReadiness], ["Settings", "PENDING" as DemoReadiness], ["Backend Integration", "NOT CONNECTED" as DemoReadiness],
  ],
  capabilities: ["Security Dashboard", "Monitoring Lifecycle", "Signal Visualizations", "Threat Event History", "Detection Analysis", "Controlled Attack Simulation", "Responsive Navigation", "Demo State Transitions"],
};

export const aiReasoning = [
  "Signal exhibits a repeating pulse interval inconsistent with known ambient sources in this band.",
  "Amplitude profile matches 3 of 5 markers associated with directional transmission.",
  "No corresponding entry in the trusted-emitter registry for this frequency and location.",
];
