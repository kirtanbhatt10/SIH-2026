import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import App from "./App.tsx";
import { IntroProvider } from "./context/IntroContext";
import { FrontendSettingsProvider } from "./context/FrontendSettingsContext";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <FrontendSettingsProvider>
      <BrowserRouter>
        <IntroProvider>
          <App />
        </IntroProvider>
      </BrowserRouter>
    </FrontendSettingsProvider>
  </StrictMode>
);
