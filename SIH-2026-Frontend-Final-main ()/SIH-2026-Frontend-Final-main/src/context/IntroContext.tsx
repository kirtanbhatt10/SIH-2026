import { useCallback, useMemo, useState, type ReactNode } from "react";
import { introContext } from "./introShared";

const STORAGE_KEY = "silentDogIntroSeen";

function readSeen() {
  try { return window.sessionStorage.getItem(STORAGE_KEY) === "1"; } catch { return true; }
}

export function IntroProvider({ children }: { children: ReactNode }) {
  const startsOnHome = typeof window === "undefined" || window.location.pathname === "/";
  const hasSeen = useMemo(readSeen, []);
  const [introPlaying, setIntroPlaying] = useState(startsOnHome && !hasSeen);
  const [showSite, setShowSite] = useState(!startsOnHome || hasSeen);
  const revealSite = useCallback(() => setShowSite(true), []);
  const finishIntro = useCallback(() => { try { window.sessionStorage.setItem(STORAGE_KEY, "1"); } catch { /* non-fatal */ } setShowSite(true); setIntroPlaying(false); }, []);
  const skipIntro = useCallback(() => finishIntro(), [finishIntro]);
  const replayIntro = useCallback(() => { setShowSite(false); setIntroPlaying(true); }, []);
  return <introContext.Provider value={{ introPlaying, showSite, revealSite, finishIntro, skipIntro, replayIntro }}>{children}</introContext.Provider>;
}
