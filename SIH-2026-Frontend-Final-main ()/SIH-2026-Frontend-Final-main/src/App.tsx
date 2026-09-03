import { Navigate, Routes, Route, useLocation } from "react-router-dom";
import { useEffect } from "react";
import NavBar from "./components/layout/NavBar";
import Footer from "./components/layout/Footer";
import RouteTransition from "./components/layout/RouteTransition";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Intelligence from "./pages/Intelligence";
import Monitoring from "./pages/Monitoring";
import About from "./pages/About";
import AttackLab from "./pages/AttackLab";
import Dashboard from "./pages/Dashboard";
import Events from "./pages/Events";
import EventDetails from "./pages/EventDetails";
import System from "./pages/System";
import Settings from "./pages/Settings";
import NotFound from "./pages/NotFound";

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }, [pathname]);
  return null;
}

export default function App() {
  const location = useLocation();
  const isLogin = location.pathname === "/login";

  return (
    <div className="min-h-screen bg-void">
      <ScrollToTop />
      <RouteTransition />
      {!isLogin && <NavBar />}
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/intelligence" element={<Intelligence />} />
          <Route path="/monitor" element={<Monitoring />} />
          <Route path="/monitoring" element={<Navigate to="/monitor" replace />} />
          <Route path="/events" element={<Events />} />
          <Route path="/events/:id" element={<EventDetails />} />
          <Route path="/about" element={<About />} />
          <Route path="/simulator" element={<AttackLab />} />
          <Route path="/attack-lab" element={<Navigate to="/simulator" replace />} />
          <Route path="/system" element={<System />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      {!isLogin && <Footer />}
    </div>
  );
}
