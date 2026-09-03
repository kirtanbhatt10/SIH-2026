import { Link } from "react-router-dom";
import Reveal from "../components/layout/Reveal";

export default function NotFound() {
  return <div className="px-5 pb-24 pt-28 md:px-10"><div className="mx-auto max-w-[1100px]"><Reveal><section className="border border-line-2 bg-surface px-6 py-20 text-center sm:px-10"><p className="font-mono text-7xl tracking-tight text-crimson sm:text-9xl">404</p><h1 className="font-display mt-5 text-3xl tracking-tight sm:text-5xl">PAGE NOT FOUND</h1><p className="mx-auto mt-4 max-w-lg text-ink-2">The requested Acoustic Shield interface does not exist.</p><div className="mt-9 flex flex-wrap justify-center gap-4"><Link to="/dashboard" className="command-button">RETURN TO DASHBOARD <span>→</span></Link><Link to="/" className="border border-line-2 px-5 py-4 font-mono text-[11px] tracking-[0.12em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson">RETURN HOME</Link></div></section></Reveal></div></div>;
}
