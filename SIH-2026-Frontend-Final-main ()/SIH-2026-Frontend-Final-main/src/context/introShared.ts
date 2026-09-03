import { createContext } from "react";

export interface IntroContextValue { introPlaying: boolean; showSite: boolean; revealSite: () => void; finishIntro: () => void; skipIntro: () => void; replayIntro: () => void; }
export const introContext = createContext<IntroContextValue | null>(null);
