import { useEffect, useMemo, useState, type ReactNode } from "react";
import { defaultFrontendSettings, frontendSettingsContext, type FrontendSettings } from "./frontendSettingsShared";

const STORAGE_KEY = "acoustic-shield-settings";
function readSettings(): FrontendSettings { try { const stored = window.localStorage.getItem(STORAGE_KEY); return stored ? { ...defaultFrontendSettings, ...(JSON.parse(stored) as Partial<FrontendSettings>) } : defaultFrontendSettings; } catch { return defaultFrontendSettings; } }

export function FrontendSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<FrontendSettings>(readSettings);
  useEffect(() => { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings)); document.documentElement.dataset.reduceMotion = String(settings.reduceMotion); }, [settings]);
  const value = useMemo(() => ({ settings, updateSettings: (patch: Partial<FrontendSettings>) => setSettings((current) => ({ ...current, ...patch })), resetSettings: () => setSettings(defaultFrontendSettings) }), [settings]);
  return <frontendSettingsContext.Provider value={value}>{children}</frontendSettingsContext.Provider>;
}
