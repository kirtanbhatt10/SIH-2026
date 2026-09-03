import { useContext } from "react";
import { frontendSettingsContext } from "../context/frontendSettingsShared";

export function useFrontendSettings() {
  const context = useContext(frontendSettingsContext);
  if (!context) throw new Error("useFrontendSettings must be used within FrontendSettingsProvider");
  return context;
}
