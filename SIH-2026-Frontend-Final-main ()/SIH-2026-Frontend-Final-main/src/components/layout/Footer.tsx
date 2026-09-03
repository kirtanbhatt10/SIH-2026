import { useNavigate } from "react-router-dom";
import { Link } from "react-router-dom";
import { useIntro } from "../../hooks/useIntro";

export default function Footer() {
  const navigate = useNavigate();
  const { replayIntro } = useIntro();

  const handleReplay = () => {
    navigate("/");
    replayIntro();
  };

  return (
    <footer className="relative border-t border-line bg-void-2 px-5 py-14 md:px-10">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-10 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="font-display text-lg font-semibold leading-none">
            THE <span className="text-crimson">SILENT DOG'S WHISTLE</span>
          </div>
          <p className="mono-label mt-3 max-w-xs normal-case tracking-normal text-ink-2">
            Intelligent spectrum defense for teams who monitor what others can't hear.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-10 text-sm sm:grid-cols-3">
          <div>
            <div className="mono-label mb-3">Platform</div>
            <ul className="space-y-2 text-ink-1">
              <li><Link className="footer-route-link" to="/intelligence">Intelligence <i>→</i></Link></li>
              <li><Link className="footer-route-link" to="/monitoring">Monitoring <i>→</i></Link></li>
              <li><Link className="footer-route-link" to="/attack-lab">Attack Lab <i>→</i></Link></li>
            </ul>
          </div>
          <div>
            <div className="mono-label mb-3">System</div>
            <ul className="space-y-2 text-ink-1">
              <li><Link className="footer-system-link" to="/system"><span /> Demo ready <i>→</i></Link></li>
              <li><Link className="footer-system-link" to="/system"><span /> Simulation data <i>→</i></Link></li>
            </ul>
          </div>
        </div>
      </div>
      <div className="divider mx-auto mt-10 max-w-[1600px]" />
      <div className="mx-auto mt-6 flex max-w-[1600px] flex-col gap-2 text-xs text-ink-3 sm:flex-row sm:items-center sm:justify-between">
        <span className="font-mono">© {new Date().getFullYear()} THE SILENT DOG'S WHISTLE — SIMULATION ENVIRONMENT</span>
        <div className="flex items-center gap-5">
          <button
            onClick={handleReplay}
            className="font-mono text-ink-3 underline decoration-line-2 underline-offset-4 transition-colors hover:text-crimson"
          >
            REPLAY INTRO
          </button>
          <span className="font-mono">BUILD AS-2026.08</span>
        </div>
      </div>
    </footer>
  );
}
