import { useContext } from "react";
import { introContext } from "../context/introShared";

export function useIntro() {
  const context = useContext(introContext);
  if (!context) throw new Error("useIntro must be used within IntroProvider");
  return context;
}
