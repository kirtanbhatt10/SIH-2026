import { createContext } from "react";

export interface FrontendSettings { reduceMotion: boolean; compactEvents: boolean; visualizationDensity: "standard" | "compact"; showWaveform: boolean; showSpectrum: boolean; showSpectrogram: boolean; showDemoLabels: boolean; autoRunDemo: boolean; }
export const defaultFrontendSettings: FrontendSettings = { reduceMotion: false, compactEvents: false, visualizationDensity: "standard", showWaveform: true, showSpectrum: true, showSpectrogram: true, showDemoLabels: true, autoRunDemo: false };
export const frontendSettingsContext = createContext<{ settings: FrontendSettings; updateSettings: (patch: Partial<FrontendSettings>) => void; resetSettings: () => void } | null>(null);
