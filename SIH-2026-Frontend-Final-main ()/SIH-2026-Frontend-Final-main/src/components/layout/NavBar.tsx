import { useEffect, useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useIntro } from "../../hooks/useIntro";

const NAV_GROUPS = [
  { label: "MONITORING", links: [{ to: "/dashboard", label: "DASHBOARD" }, { to: "/monitor", label: "LIVE MONITOR" }, { to: "/events", label: "THREAT EVENTS" }] },
  { label: "SECURITY LAB", links: [{ to: "/simulator", label: "ATTACK SIMULATOR" }] },
  { label: "SYSTEM", links: [{ to: "/system", label: "SYSTEM STATUS" }, { to: "/settings", label: "SETTINGS" }, { to: "/about", label: "ABOUT" }] },
];

export default function NavBar() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const navigate = useNavigate();
  const { showSite } = useIntro();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <>
      <header
        className={`fixed top-0 left-0 right-0 z-50 border-b transition-[opacity,background-color,border-color] duration-700 ${
          scrolled ? "border-line bg-void-2/90 backdrop-blur-md" : "border-transparent bg-transparent"
        } ${showSite ? "opacity-100" : "pointer-events-none opacity-0"}`}
      >
        <div className="mx-auto flex h-16 w-full min-w-0 max-w-[1600px] items-center justify-between gap-4 px-5 md:px-10">
          <Link
            to="/"
            onClick={() => setOpen(false)}
            className="shrink-0 font-display leading-[0.95] text-sm font-semibold tracking-tight"
          >
            <motion.span layoutId="silent-dog-brand">THE SILENT DOG'S WHISTLE</motion.span>
          </Link>

          <div className="desktop-navigation hidden min-w-0 flex-1 items-center justify-end gap-6 xl:flex">
          <span className="menu-frequency" aria-hidden="true"><i /><i /><i /><i /></span>
          <nav className="flex min-w-0 items-center gap-5" aria-label="Primary navigation">
            {NAV_GROUPS.map((group) => (
              <div key={group.label} className="flex shrink-0 items-center gap-4 border-l border-line pl-5 first:border-l-0 first:pl-0">
                {group.links.map((link) => <NavLink key={link.to} to={link.to} className={({ isActive }) => `nav-route-link mono-label ${isActive ? "text-amber" : "text-ink-2"}`}>{link.label}</NavLink>)}
              </div>
            ))}
          </nav>

          <button
            onClick={() => navigate("/login")}
            className="shrink-0 whitespace-nowrap border border-line-2 px-3 py-2 font-mono text-[10px] tracking-[0.1em] text-ink-0 transition-colors hover:border-crimson hover:text-crimson"
          >
            SECURE ACCESS
          </button>
          </div>

          <button
            aria-label={open ? "Close navigation" : "Open navigation"}
            aria-expanded={open}
            onClick={() => setOpen((o) => !o)}
            className="flex h-9 w-9 flex-col items-center justify-center gap-1.5 xl:hidden"
          >
            <span className={`h-px w-6 bg-ink-0 transition-transform ${open ? "translate-y-[3.5px] rotate-45" : ""}`} />
            <span className={`h-px w-6 bg-ink-0 transition-transform ${open ? "-translate-y-[3.5px] -rotate-45" : ""}`} />
          </button>
        </div>
      </header>

      {/* mobile full-screen panel */}
      <div
        className={`fixed inset-0 z-40 flex flex-col justify-center bg-void-2/98 backdrop-blur-md transition-opacity duration-300 xl:hidden ${
          open ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0"
        }`}
      >
        <div className="bg-grid-fine absolute inset-0 opacity-40" />
        <nav className="relative max-h-full overflow-y-auto px-8 py-24" aria-label="Mobile primary navigation">
          <Link to="/" onClick={() => setOpen(false)} className="mono-label mb-8 inline-block text-ink-2">HOME</Link>
          {NAV_GROUPS.map((group) => (
            <div key={group.label} className="mb-8">
              <p className="mono-label mb-2 text-crimson">{group.label}</p>
              {group.links.map((link, index) => <NavLink key={link.to} to={link.to} onClick={() => setOpen(false)} className={({ isActive }) => `block font-display border-b border-line py-3 text-2xl tracking-tight transition-colors ${isActive ? "text-amber" : "text-ink-0"}`}><span className="mono-label mr-3 text-ink-3">0{index + 1}</span>{link.label}</NavLink>)}
            </div>
          ))}
          <button
            onClick={() => {
              setOpen(false);
              navigate("/login");
            }}
            className="mt-8 border border-crimson-3 px-5 py-4 text-left font-mono text-sm tracking-[0.12em] text-crimson"
          >
            ENTER COMMAND →
          </button>
        </nav>
      </div>
    </>
  );
}
