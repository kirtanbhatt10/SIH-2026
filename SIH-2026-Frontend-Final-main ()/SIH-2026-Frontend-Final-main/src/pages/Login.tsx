import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Waveform from "../components/viz/Waveform";
import RadarSweep from "../components/viz/RadarSweep";
import { useSystemStatus } from "../hooks/useSystemStatus";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const { status: backendStatus } = useSystemStatus();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) { setError("Enter a demo username and password to continue."); return; }
    // Frontend demo access only — no authentication request is made.
    navigate("/dashboard");
  };

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      {/* LEFT — identity */}
      <div className="relative hidden flex-col justify-between overflow-hidden border-r border-line bg-void-2 p-10 lg:flex">
        <div className="absolute inset-0 opacity-30">
          <RadarSweep size={700} />
        </div>
        <div className="bg-grid-fine absolute inset-0 opacity-40" />

        <Link to="/" className="font-display relative z-10 text-sm font-semibold tracking-tight">
          THE <span className="text-crimson">SILENT DOG'S WHISTLE</span>
        </Link>

        <div className="relative z-10">
          <h1 className="font-display text-6xl leading-[0.9] tracking-tighter">
            SECURE
            <br />
            ACCESS
          </h1>
          <p className="mono-label mt-6 max-w-xs normal-case tracking-normal text-ink-1">
            Demo access for the Acoustic Shield frontend presentation.
          </p>
        </div>

        <div className="relative z-10">
          <Waveform mode="normal" height={90} />
          <div className="mono-label mt-4 flex gap-8">
            <span>AS-01</span>
            <span>DEMO ENVIRONMENT</span>
          </div>
        </div>
      </div>

      {/* RIGHT — form */}
      <div className="relative flex flex-col justify-center bg-void px-6 py-16 sm:px-16">
        <Link to="/" className="font-display mb-16 text-sm font-semibold tracking-tight lg:hidden">
          THE <span className="text-crimson">SILENT DOG'S WHISTLE</span>
        </Link>

        <div className="mono-label mb-3">DEMO ACCESS</div>
        <h2 className="font-display mb-3 text-3xl tracking-tight">PRESENTATION ENTRY</h2>
        <p className="mb-7 max-w-sm text-sm text-ink-2">No credentials are sent or verified in this frontend-only demo.</p>

        <form onSubmit={handleSubmit} className="max-w-sm space-y-6">
          <div>
            <label htmlFor="username" className="mono-label mb-2 block">
              Username
            </label>
            <input
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              type="text"
              autoComplete="username"
              className="w-full border border-line-2 bg-surface px-4 py-3 text-sm text-ink-0 outline-none transition-colors focus:border-crimson"
              placeholder="analyst.id"
            />
          </div>
          <div>
            <label htmlFor="password" className="mono-label mb-2 block">
              Password
            </label>
            <input
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              type="password"
              autoComplete="current-password"
              className="w-full border border-line-2 bg-surface px-4 py-3 text-sm text-ink-0 outline-none transition-colors focus:border-crimson"
              placeholder="••••••••••"
            />
          </div>

          <button
            type="submit"
            className="w-full border border-crimson-3 bg-crimson-3/10 px-6 py-4 font-mono text-xs tracking-[0.14em] text-crimson transition-colors hover:bg-crimson-3/20"
          >
            ENTER DEMO
          </button>
          {error && <p className="text-sm text-crimson" role="alert">{error}</p>}
        </form>

        <div className="mt-12 flex max-w-sm flex-col gap-2">
          <div className="mono-label flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-system" />
            DEMO ACCESS · ENABLED
          </div>
          <div className="mono-label flex items-center gap-2">
            <span className={`h-1.5 w-1.5 rounded-full ${backendStatus === "online" ? "bg-system" : backendStatus === "loading" ? "bg-amber" : "bg-crimson"}`} />
            BACKEND · {backendStatus === "loading" ? "CHECKING…" : backendStatus === "online" ? "CONNECTED" : "NOT CONNECTED"}
          </div>
        </div>
      </div>
    </div>
  );
}
