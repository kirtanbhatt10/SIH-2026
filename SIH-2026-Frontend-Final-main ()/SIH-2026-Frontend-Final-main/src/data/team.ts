import teamMember01 from "../assets/team/team-member-01.jpg";
import teamMember02 from "../assets/team/team-member-02.jpg";
import teamMember03 from "../assets/team/team-member-03.jpg";
import teamMember04 from "../assets/team/team-member-04.jpg";
import teamMember05 from "../assets/team/team-member-05.jpg";
import teamMember06 from "../assets/team/team-member-06.jpg";

export interface TeamMember {
  id: string;
  node: string; // e.g. "AS-01"
  name: string;
  role: string;
  description: string;
  specialization: string[];
  status: "active" | "standby";
  featured?: boolean;
  image: string;
  /** CSS object-position for portrait crop (e.g. "center 28%") */
  imagePosition?: string;
  /** Base zoom so full-body shots frame face/torso inside the card */
  imageScale?: number;
  linkedin?: string;
  github?: string;
  portfolio?: string;
  email?: string;
}

export const teamMembers: TeamMember[] = [
  {
    id: "member-1",
    node: "AS-01",
    name: "KIRTAN ISHANKUMAR",
    role: "FOUNDER / TEAM LEAD / BACKEND 1",
    description:
      "Owns Backend 1 and the integration pipeline — threat APIs, monitoring pipeline wiring, DSP/ML handoff, and cross-team contracts that connect simulator output, live mic streams, and frontend operator surfaces.",
    specialization: ["INTEGRATION", "BACKEND", "SYSTEMS"],
    status: "active",
    featured: true,
    image: teamMember01,
    imagePosition: "center 22%",
    imageScale: 1.18,
  },
  {
    id: "member-2",
    node: "AS-02",
    name: "TEJAS PADIA",
    role: "BACKEND 2 / SIMULATOR ENGINEER",
    description:
      "Owns Backend 2 — BFSK payload encode/decode, attack simulator endpoints, audio streaming, and controlled transmission modes (virtual and ultrasonic) that feed the real detection pipeline.",
    specialization: ["SIMULATOR", "SIGNAL TRANSPORT", "BACKEND"],
    status: "active",
    image: teamMember02,
    imagePosition: "center 32%",
    imageScale: 1.14,
  },
  {
    id: "member-3",
    node: "AS-03",
    name: "ANSH RAJAN",
    role: "FRONTEND ENGINEER",
    description:
      "Builds the React operator interface — dashboard, monitor, threat events, attack simulator, and live audio visualization — wired to Backend 1 and Backend 2 APIs across the tactical HUD experience.",
    specialization: ["FRONTEND", "UI", "VISUALIZATION"],
    status: "active",
    image: teamMember05,
    imagePosition: "center 28%",
    imageScale: 1.06,
  },
  {
    id: "member-4",
    node: "AS-04",
    name: "KASHISH",
    role: "AI / ML LEAD — DSP",
    description:
      "Leads the DSP layer of the acoustic intelligence stack — frequency analysis, ultrasonic band detection, feature extraction, and the pipeline that produces structured threat signals for downstream classification.",
    specialization: ["DSP", "ACOUSTICS", "FEATURE EXTRACTION"],
    status: "active",
    image: teamMember04,
    imagePosition: "center 30%",
    imageScale: 1.05,
  },
  {
    id: "member-5",
    node: "AS-05",
    name: "DENISH",
    role: "AI / ML LEAD — MACHINE LEARNING",
    description:
      "Leads the machine-learning path that interprets DSP outputs — modulation classification, risk scoring, and suspicion assessment before a ThreatEvent is stored and surfaced to operators.",
    specialization: ["MACHINE LEARNING", "CLASSIFICATION", "RISK SCORING"],
    status: "active",
    image: teamMember03,
    imagePosition: "center 38%",
    imageScale: 1.08,
  },
  {
    id: "member-6",
    node: "AS-06",
    name: "MOSAM",
    role: "RESEARCH",
    description:
      "Supports the research layer of the project — acoustic threat literature, experimental validation, and analysis that grounds the system's detection approach in documented covert-channel and ultrasonic communication research.",
    specialization: ["RESEARCH", "SIGNAL ANALYSIS", "VALIDATION"],
    status: "active",
    image: teamMember06,
    imagePosition: "center 35%",
    imageScale: 1.1,
  },
];
