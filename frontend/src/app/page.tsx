"use client";

import { useEffect, useState, useCallback, useRef } from "react";

/* ═══════════════════ TYPES ═══════════════════ */
type ScoredDriver = {
  driverId: string;
  score_rank: number;
  p_dnf_raw: number;
  p_dnf: number;
  score_adj: number;
};
type PredictionResponse = {
  raceId: string;
  order: ScoredDriver[];
  alpha: number;
  p_dnf_cap: number;
  mode: string;
  sources: { ergast: boolean; fastf1: boolean; dataset_form: boolean };
  warnings: string[];
  form_cutoff_raceId: string;
};

// The API runs on port 8000 by default in the backend
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://f1-predictor-api-qpym.onrender.com";

/* ═══════════════ RACE CALENDAR ═══════════════ */
const CALENDAR: Record<number, { round: number; name: string; flag: string; circuit: string; date: string }[]> = {
  2026: [
    { round: 1, name: "Australian Grand Prix", flag: "🇦🇺", circuit: "Albert Park, Melbourne", date: "Mar 8" },
    { round: 2, name: "Chinese Grand Prix", flag: "🇨🇳", circuit: "Shanghai International", date: "Mar 22" },
    { round: 3, name: "Japanese Grand Prix", flag: "🇯🇵", circuit: "Suzuka", date: "Apr 5" },
    { round: 4, name: "Bahrain Grand Prix", flag: "🇧🇭", circuit: "Sakhir", date: "Apr 12" },
    { round: 5, name: "Saudi Arabian Grand Prix", flag: "🇸🇦", circuit: "Jeddah Corniche", date: "Apr 19" },
    { round: 6, name: "Miami Grand Prix", flag: "🇺🇸", circuit: "Miami International", date: "May 3" },
    { round: 7, name: "Emilia Romagna Grand Prix", flag: "🇮🇹", circuit: "Imola", date: "May 17" },
    { round: 8, name: "Monaco Grand Prix", flag: "🇲🇨", circuit: "Circuit de Monaco", date: "May 24" },
    { round: 9, name: "Spanish Grand Prix", flag: "🇪🇸", circuit: "Circuit de Barcelona", date: "Jun 1" },
    { round: 10, name: "Canadian Grand Prix", flag: "🇨🇦", circuit: "Circuit Gilles Villeneuve", date: "Jun 15" },
    { round: 11, name: "Austrian Grand Prix", flag: "🇦🇹", circuit: "Red Bull Ring", date: "Jun 29" },
    { round: 12, name: "British Grand Prix", flag: "🇬🇧", circuit: "Silverstone", date: "Jul 6" },
    { round: 13, name: "Belgian Grand Prix", flag: "🇧🇪", circuit: "Spa-Francorchamps", date: "Jul 27" },
    { round: 14, name: "Hungarian Grand Prix", flag: "🇭🇺", circuit: "Hungaroring", date: "Aug 3" },
    { round: 15, name: "Dutch Grand Prix", flag: "🇳🇱", circuit: "Zandvoort", date: "Aug 31" },
    { round: 16, name: "Italian Grand Prix", flag: "🇮🇹", circuit: "Monza", date: "Sep 7" },
    { round: 17, name: "Azerbaijan Grand Prix", flag: "🇦🇿", circuit: "Baku City", date: "Sep 21" },
    { round: 18, name: "Singapore Grand Prix", flag: "🇸🇬", circuit: "Marina Bay", date: "Oct 5" },
    { round: 19, name: "United States Grand Prix", flag: "🇺🇸", circuit: "COTA, Austin", date: "Oct 19" },
    { round: 20, name: "Mexico City Grand Prix", flag: "🇲🇽", circuit: "Hermanos Rodriguez", date: "Oct 26" },
    { round: 21, name: "Brazilian Grand Prix", flag: "🇧🇷", circuit: "Interlagos", date: "Nov 9" },
    { round: 22, name: "Las Vegas Grand Prix", flag: "🇺🇸", circuit: "Las Vegas Strip", date: "Nov 22" },
    { round: 23, name: "Qatar Grand Prix", flag: "🇶🇦", circuit: "Losail International", date: "Nov 30" },
    { round: 24, name: "Abu Dhabi Grand Prix", flag: "🇦🇪", circuit: "Yas Marina", date: "Dec 7" },
  ],
  2025: [
    { round: 1, name: "Australian Grand Prix", flag: "🇦🇺", circuit: "Albert Park", date: "Mar 16" },
    { round: 2, name: "Chinese Grand Prix", flag: "🇨🇳", circuit: "Shanghai", date: "Mar 23" },
    { round: 3, name: "Japanese Grand Prix", flag: "🇯🇵", circuit: "Suzuka", date: "Apr 6" },
    { round: 4, name: "Bahrain Grand Prix", flag: "🇧🇭", circuit: "Sakhir", date: "Apr 13" },
    { round: 5, name: "Saudi Arabian Grand Prix", flag: "🇸🇦", circuit: "Jeddah", date: "Apr 20" },
    { round: 6, name: "Miami Grand Prix", flag: "🇺🇸", circuit: "Miami", date: "May 4" },
    { round: 7, name: "Emilia Romagna Grand Prix", flag: "🇮🇹", circuit: "Imola", date: "May 18" },
    { round: 8, name: "Monaco Grand Prix", flag: "🇲🇨", circuit: "Monte Carlo", date: "May 25" },
    { round: 9, name: "Spanish Grand Prix", flag: "🇪🇸", circuit: "Barcelona", date: "Jun 1" },
    { round: 10, name: "Canadian Grand Prix", flag: "🇨🇦", circuit: "Montreal", date: "Jun 15" },
    { round: 11, name: "Austrian Grand Prix", flag: "🇦🇹", circuit: "Spielberg", date: "Jun 29" },
    { round: 12, name: "British Grand Prix", flag: "🇬🇧", circuit: "Silverstone", date: "Jul 6" },
  ],
  2024: [
    { round: 1, name: "Bahrain Grand Prix", flag: "🇧🇭", circuit: "Sakhir", date: "Mar 2" },
    { round: 2, name: "Saudi Arabian Grand Prix", flag: "🇸🇦", circuit: "Jeddah", date: "Mar 9" },
    { round: 3, name: "Australian Grand Prix", flag: "🇦🇺", circuit: "Albert Park", date: "Mar 24" },
    { round: 4, name: "Japanese Grand Prix", flag: "🇯🇵", circuit: "Suzuka", date: "Apr 7" },
    { round: 5, name: "Chinese Grand Prix", flag: "🇨🇳", circuit: "Shanghai", date: "Apr 21" },
    { round: 6, name: "Miami Grand Prix", flag: "🇺🇸", circuit: "Miami", date: "May 5" },
    { round: 7, name: "Emilia Romagna Grand Prix", flag: "🇮🇹", circuit: "Imola", date: "May 19" },
    { round: 8, name: "Monaco Grand Prix", flag: "🇲🇨", circuit: "Monte Carlo", date: "May 26" },
    { round: 9, name: "Canadian Grand Prix", flag: "🇨🇦", circuit: "Montreal", date: "Jun 9" },
    { round: 10, name: "Spanish Grand Prix", flag: "🇪🇸", circuit: "Barcelona", date: "Jun 23" },
    { round: 11, name: "Austrian Grand Prix", flag: "🇦🇹", circuit: "Spielberg", date: "Jun 30" },
    { round: 12, name: "British Grand Prix", flag: "🇬🇧", circuit: "Silverstone", date: "Jul 7" },
    { round: 13, name: "Hungarian Grand Prix", flag: "🇭🇺", circuit: "Hungaroring", date: "Jul 21" },
    { round: 14, name: "Belgian Grand Prix", flag: "🇧🇪", circuit: "Spa", date: "Jul 28" },
    { round: 15, name: "Dutch Grand Prix", flag: "🇳🇱", circuit: "Zandvoort", date: "Aug 25" },
    { round: 16, name: "Italian Grand Prix", flag: "🇮🇹", circuit: "Monza", date: "Sep 1" },
    { round: 17, name: "Azerbaijan Grand Prix", flag: "🇦🇿", circuit: "Baku", date: "Sep 15" },
    { round: 18, name: "Singapore Grand Prix", flag: "🇸🇬", circuit: "Marina Bay", date: "Sep 22" },
    { round: 19, name: "United States Grand Prix", flag: "🇺🇸", circuit: "Austin", date: "Oct 20" },
    { round: 20, name: "Mexico City Grand Prix", flag: "🇲🇽", circuit: "Mexico City", date: "Oct 27" },
    { round: 21, name: "Brazilian Grand Prix", flag: "🇧🇷", circuit: "Interlagos", date: "Nov 3" },
    { round: 22, name: "Las Vegas Grand Prix", flag: "🇺🇸", circuit: "Las Vegas", date: "Nov 23" },
    { round: 23, name: "Qatar Grand Prix", flag: "🇶🇦", circuit: "Losail", date: "Dec 1" },
    { round: 24, name: "Abu Dhabi Grand Prix", flag: "🇦🇪", circuit: "Yas Marina", date: "Dec 8" },
  ],
};

/* ═══════════════ DRIVER DATA (Season-Aware) ═══════════════ */
type DriverInfo = { team: string; color: string; accent: string; num: number };
const TEAMS_BY_SEASON: Record<number, Record<string, DriverInfo>> = {
  2024: {
    max_verstappen: { team: "Red Bull Racing", color: "#1E3A8A", accent: "#3B82F6", num: 1 },
    perez: { team: "Red Bull Racing", color: "#1E3A8A", accent: "#3B82F6", num: 11 },
    hamilton: { team: "Mercedes", color: "#0D9488", accent: "#2DD4BF", num: 44 },
    russell: { team: "Mercedes", color: "#0D9488", accent: "#2DD4BF", num: 63 },
    leclerc: { team: "Ferrari", color: "#DC2626", accent: "#F87171", num: 16 },
    sainz: { team: "Ferrari", color: "#DC2626", accent: "#F87171", num: 55 },
    norris: { team: "McLaren", color: "#EA580C", accent: "#FB923C", num: 4 },
    piastri: { team: "McLaren", color: "#EA580C", accent: "#FB923C", num: 81 },
    alonso: { team: "Aston Martin", color: "#166534", accent: "#4ADE80", num: 14 },
    stroll: { team: "Aston Martin", color: "#166534", accent: "#4ADE80", num: 18 },
    gasly: { team: "Alpine", color: "#0369A1", accent: "#38BDF8", num: 10 },
    ocon: { team: "Alpine", color: "#0369A1", accent: "#38BDF8", num: 31 },
    ricciardo: { team: "RB", color: "#1E40AF", accent: "#818CF8", num: 3 },
    tsunoda: { team: "RB", color: "#1E40AF", accent: "#818CF8", num: 22 },
    bottas: { team: "Kick Sauber", color: "#10B981", accent: "#34D399", num: 77 },
    zhou: { team: "Kick Sauber", color: "#10B981", accent: "#34D399", num: 24 },
    kevin_magnussen: { team: "Haas", color: "#B91C1C", accent: "#F87171", num: 20 },
    magnussen: { team: "Haas", color: "#B91C1C", accent: "#F87171", num: 20 },
    hulkenberg: { team: "Haas", color: "#B91C1C", accent: "#F87171", num: 27 },
    albon: { team: "Williams", color: "#1D4ED8", accent: "#60A5FA", num: 23 },
    sargeant: { team: "Williams", color: "#1D4ED8", accent: "#60A5FA", num: 2 },
    colapinto: { team: "Williams", color: "#1D4ED8", accent: "#60A5FA", num: 43 },
    lawson: { team: "RB", color: "#1E40AF", accent: "#818CF8", num: 40 },
    bearman: { team: "Ferrari", color: "#DC2626", accent: "#F87171", num: 38 },
    de_vries: { team: "RB", color: "#1E40AF", accent: "#818CF8", num: 21 },
  },
  2025: {
    max_verstappen: { team: "Red Bull Racing", color: "#1E3A8A", accent: "#3B82F6", num: 1 },
    lawson: { team: "Red Bull Racing", color: "#1E3A8A", accent: "#3B82F6", num: 30 },
    hamilton: { team: "Ferrari", color: "#DC2626", accent: "#F87171", num: 44 },
    leclerc: { team: "Ferrari", color: "#DC2626", accent: "#F87171", num: 16 },
    russell: { team: "Mercedes", color: "#0D9488", accent: "#2DD4BF", num: 63 },
    antonelli: { team: "Mercedes", color: "#0D9488", accent: "#2DD4BF", num: 12 },
    norris: { team: "McLaren", color: "#EA580C", accent: "#FB923C", num: 4 },
    piastri: { team: "McLaren", color: "#EA580C", accent: "#FB923C", num: 81 },
    alonso: { team: "Aston Martin", color: "#166534", accent: "#4ADE80", num: 14 },
    stroll: { team: "Aston Martin", color: "#166534", accent: "#4ADE80", num: 18 },
    gasly: { team: "Alpine", color: "#0369A1", accent: "#38BDF8", num: 10 },
    doohan: { team: "Alpine", color: "#0369A1", accent: "#38BDF8", num: 7 },
    tsunoda: { team: "Racing Bulls", color: "#1E40AF", accent: "#818CF8", num: 22 },
    hadjar: { team: "Racing Bulls", color: "#1E40AF", accent: "#818CF8", num: 6 },
    hulkenberg: { team: "Kick Sauber", color: "#10B981", accent: "#34D399", num: 27 },
    bortoleto: { team: "Kick Sauber", color: "#10B981", accent: "#34D399", num: 5 },
    ocon: { team: "Haas", color: "#B91C1C", accent: "#F87171", num: 31 },
    bearman: { team: "Haas", color: "#B91C1C", accent: "#F87171", num: 87 },
    albon: { team: "Williams", color: "#1D4ED8", accent: "#60A5FA", num: 23 },
    sainz: { team: "Williams", color: "#1D4ED8", accent: "#60A5FA", num: 55 },
    perez: { team: "Red Bull Racing", color: "#1E3A8A", accent: "#3B82F6", num: 11 },
    colapinto: { team: "Alpine", color: "#0369A1", accent: "#38BDF8", num: 43 },
  },
  2026: {
    russell: { team: "Mercedes", color: "#0D9488", accent: "#2DD4BF", num: 63 },
    antonelli: { team: "Mercedes", color: "#0D9488", accent: "#2DD4BF", num: 12 },
    leclerc: { team: "Ferrari", color: "#DC2626", accent: "#F87171", num: 16 },
    hamilton: { team: "Ferrari", color: "#DC2626", accent: "#F87171", num: 44 },
    norris: { team: "McLaren", color: "#EA580C", accent: "#FB923C", num: 1 },
    piastri: { team: "McLaren", color: "#EA580C", accent: "#FB923C", num: 81 },
    max_verstappen: { team: "Red Bull Racing", color: "#1E3A8A", accent: "#3B82F6", num: 3 },
    hadjar: { team: "Red Bull Racing", color: "#1E3A8A", accent: "#3B82F6", num: 6 },
    bearman: { team: "Haas", color: "#B91C1C", accent: "#F87171", num: 87 },
    ocon: { team: "Haas", color: "#B91C1C", accent: "#F87171", num: 31 },
    arvid_lindblad: { team: "Racing Bulls", color: "#1E40AF", accent: "#818CF8", num: 41 },
    lindblad: { team: "Racing Bulls", color: "#1E40AF", accent: "#818CF8", num: 41 },
    lawson: { team: "Racing Bulls", color: "#1E40AF", accent: "#818CF8", num: 30 },
    bortoleto: { team: "Audi", color: "#10B981", accent: "#34D399", num: 5 },
    hulkenberg: { team: "Audi", color: "#10B981", accent: "#34D399", num: 27 },
    gasly: { team: "Alpine", color: "#0369A1", accent: "#38BDF8", num: 10 },
    colapinto: { team: "Alpine", color: "#0369A1", accent: "#38BDF8", num: 43 },
    albon: { team: "Williams", color: "#1D4ED8", accent: "#60A5FA", num: 23 },
    sainz: { team: "Williams", color: "#1D4ED8", accent: "#60A5FA", num: 55 },
    perez: { team: "Cadillac", color: "#CA8A04", accent: "#FACC15", num: 11 },
    bottas: { team: "Cadillac", color: "#CA8A04", accent: "#FACC15", num: 77 },
    stroll: { team: "Aston Martin", color: "#166534", accent: "#4ADE80", num: 18 },
    alonso: { team: "Aston Martin", color: "#166534", accent: "#4ADE80", num: 14 },
  },
};

const td = (id: string, yr: number): DriverInfo => {
  const seasonMap = TEAMS_BY_SEASON[yr] || TEAMS_BY_SEASON[2026];
  return seasonMap[id] || { team: "Unknown", color: "#3F3F46", accent: "#71717A", num: 0 };
};
const fmt = (id: string) => id.split("_").map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");

/* ═══════════════ COMPONENTS ═══════════════ */

/* Tooltip wrapper */
function Tip({ text, children }: { text: string; children: React.ReactNode }) {
  return (
    <span className="tooltip-container cursor-help">
      {children}
      <span className="tooltip-text">{text}</span>
    </span>
  );
}

/* Circular DNF gauge */
function Gauge({ value, size = 56 }: { value: number; size?: number }) {
  const pct = Math.min(value * 100, 100);
  const r = (size - 6) / 2;
  const c = 2 * Math.PI * r;
  const off = c - (pct / 100) * c;
  const col = pct > 20 ? "#EF4444" : pct > 10 ? "#F59E0B" : "#22C55E";
  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#27272A" strokeWidth={3} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={col} strokeWidth={3}
          strokeDasharray={c} strokeDashoffset={off} strokeLinecap="round"
          className="transition-all duration-1000 ease-out" />
      </svg>
      <span className="absolute text-[10px] font-bold" style={{ color: col }}>{pct.toFixed(0)}%</span>
    </div>
  );
}

/* Source status pill */
function Source({ label, on }: { label: string; on: boolean }) {
  return (
    <Tip text={on ? `${label} data was successfully fetched and used in this prediction.` : `${label} data was unavailable. The model fell back to other features.`}>
      <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold ${on ? "bg-emerald-950/50 text-emerald-400 border border-emerald-800/50" : "bg-red-950/50 text-red-400 border border-red-800/50"}`}>
        <div className={`w-1.5 h-1.5 rounded-full ${on ? "bg-emerald-400 animate-pulse" : "bg-red-500"}`} />
        {label}
      </div>
    </Tip>
  );
}

/* ═══════════════ MAIN ═══════════════ */
export default function Home() {
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [season, setSeason] = useState(2026);
  const [round, setRound] = useState(1);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [tab, setTab] = useState<"grid" | "h2h">("grid");
  const [cmp, setCmp] = useState<string[]>([]);
  const [calOpen, setCalOpen] = useState(false);
  const calRef = useRef<HTMLDivElement>(null);

  const races = CALENDAR[season] || [];
  const race = races.find((r) => r.round === round);

  // Close calendar on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (calRef.current && !calRef.current.contains(e.target as Node)) setCalOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const predict = useCallback(async (s: number, r: number) => {
    setLoading(true); setError(null); setExpanded(null);
    try {
      const res = await fetch(`${API_BASE}/predict/live?season=${s}&round=${r}`);
      if (!res.ok) { const e = await res.json().catch(() => ({ detail: "API Error" })); throw new Error(e.detail || `HTTP ${res.status}`); }
      setData(await res.json());
    } catch (e: any) { setError(e.message || "Connection failed"); }
    finally { setLoading(false); }
  }, []);

  const toggleCmp = (id: string) => setCmp((p) => p.includes(id) ? p.filter((x) => x !== id) : p.length < 2 ? [...p, id] : [p[1], id]);
  const cmpData = data ? cmp.map((id) => data.order.find((d) => d.driverId === id)).filter(Boolean) as ScoredDriver[] : [];

  /* Best / worst scores for normalization */
  const best = data ? Math.max(...data.order.map((d) => d.score_adj)) : 0;
  const worst = data ? Math.min(...data.order.map((d) => d.score_adj)) : 0;
  const range = best - worst || 1;

  return (
    <div className="min-h-screen bg-[#09090B] text-white">
      {/* ═══ HEADER ═══ */}
      <header className="sticky top-0 z-50 border-b border-white/5 bg-[#09090B]/90 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex gap-0.5">
              <div className="w-1 h-6 bg-red-600 rounded-full" />
              <div className="w-1 h-6 bg-red-600/60 rounded-full" />
              <div className="w-1 h-6 bg-red-600/30 rounded-full" />
            </div>
            <h1 className="text-lg font-black tracking-tight">F1 <span className="text-red-500">ORACLE</span></h1>
          </div>
          {data && (
            <div className="hidden md:flex items-center gap-1.5">
              <Source label="Ergast" on={data.sources.ergast} />
              <Source label="FastF1 Pace" on={data.sources.fastf1} />
              <Source label="Form Data" on={data.sources.dataset_form} />
            </div>
          )}
        </div>
        {/* Racing stripe accent */}
        <div className="h-[2px] w-full bg-gradient-to-r from-transparent via-red-600 to-transparent opacity-60" />
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {/* ═══ RACE SELECTOR ═══ */}
        <section className="mb-6" style={{ position: "relative", zIndex: 40 }}>
          <div className="flex flex-col sm:flex-row gap-3 items-stretch">
            {/* Season Tabs */}
            <div className="flex gap-1 bg-zinc-900/60 p-1 rounded-xl border border-white/5">
              {[2024, 2025, 2026].map((y) => (
                <button key={y} onClick={() => { setSeason(y); setRound(1); setData(null); }}
                  className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${season === y ? "bg-red-600 text-white shadow-lg shadow-red-900/30" : "text-zinc-500 hover:text-white hover:bg-zinc-800"}`}>
                  {y}
                </button>
              ))}
            </div>

            {/* Race Dropdown */}
            <div className="flex-1 relative" ref={calRef} style={{ zIndex: 60 }}>
              <button onClick={(e) => { e.stopPropagation(); setCalOpen(!calOpen); }}
                className="w-full flex items-center justify-between px-4 py-2.5 rounded-xl border border-white/10 bg-zinc-900/60 hover:bg-zinc-800/60 transition-all text-left relative z-10">
                <div className="flex items-center gap-3 min-w-0">
                  {race && <span className="text-2xl animate-flag-wave flex-shrink-0">{race.flag}</span>}
                  <div className="min-w-0">
                    <div className="font-bold text-sm text-white truncate">{race ? race.name : "Select a Race"}</div>
                    <div className="text-[10px] text-zinc-500">{race ? `${race.circuit} · ${race.date}` : "Choose from the calendar"}</div>
                  </div>
                </div>
                <svg className={`w-4 h-4 text-zinc-500 transition-transform flex-shrink-0 ${calOpen ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {/* Dropdown Calendar */}
              {calOpen && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-[#18181B] border border-white/10 rounded-2xl overflow-hidden animate-fade-in" style={{ zIndex: 9999, boxShadow: "0 25px 60px rgba(0,0,0,0.8)" }}>
                  <div className="max-h-80 overflow-y-auto">
                    {races.map((r) => (
                      <button key={r.round} onClick={(e) => { e.stopPropagation(); setRound(r.round); setCalOpen(false); }}
                        className={`w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/5 transition-colors border-b border-white/5 last:border-0 ${round === r.round ? "bg-red-600/10" : ""}`}>
                        <span className="text-xl flex-shrink-0">{r.flag}</span>
                        <div className="flex-1 min-w-0">
                          <div className={`text-sm font-semibold truncate ${round === r.round ? "text-red-400" : "text-white"}`}>{r.name}</div>
                          <div className="text-[10px] text-zinc-500">{r.circuit}</div>
                        </div>
                        <div className="text-[10px] text-zinc-600 flex-shrink-0">{r.date}</div>
                        {round === r.round && <div className="w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" />}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Predict Button */}
            <button onClick={() => predict(season, round)} disabled={loading}
              className="px-6 py-2.5 rounded-xl bg-red-600 hover:bg-red-500 active:scale-95 text-white font-black text-sm uppercase tracking-wider transition-all disabled:opacity-40 animate-pulse-glow flex-shrink-0 flex items-center gap-2">
              {loading ? <><div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" /> Computing</> : <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                Predict
              </>}
            </button>
          </div>
        </section>

        {/* ═══ RACE HERO BANNER ═══ */}
        {race && data && !loading && (
          <section className="mb-8 relative overflow-hidden rounded-2xl border border-white/5 checkered-bg animate-fade-in-up stagger-1">
            <div className="absolute inset-0 bg-gradient-to-r from-[#09090B] via-transparent to-[#09090B]" />
            <div className="relative z-10 flex items-center justify-between px-6 py-5">
              <div className="flex items-center gap-4">
                <span className="text-5xl animate-flag-wave">{race.flag}</span>
                <div>
                  <h2 className="text-2xl sm:text-3xl font-black tracking-tight">{race.name}</h2>
                  <p className="text-zinc-500 text-xs mt-0.5">{race.circuit} &middot; Round {round} &middot; {season}</p>
                </div>
              </div>
              <div className="hidden sm:flex items-center gap-6 text-right">
                <Tip text="The model only uses race data from before this race, preventing any cheating or data leakage.">
                  <div>
                    <div className="text-[9px] uppercase tracking-widest text-zinc-600 font-bold">No-Leak Cutoff</div>
                    <div className="text-sm font-mono text-zinc-300">{data.form_cutoff_raceId}</div>
                  </div>
                </Tip>
                <Tip text="The combined score blending factor. Higher alpha means DNF risk has more weight in the final ranking.">
                  <div>
                    <div className="text-[9px] uppercase tracking-widest text-zinc-600 font-bold">Alpha</div>
                    <div className="text-sm font-mono text-zinc-300">{data.alpha}</div>
                  </div>
                </Tip>
              </div>
            </div>
            {/* Speed lines decoration */}
            <div className="speed-lines absolute inset-0 pointer-events-none" />
          </section>
        )}

        {/* ═══ LOADING — F1 Tire ═══ */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 animate-fade-in">
            {/* F1 Tire SVG */}
            <div className="relative mb-8">
              <svg width="120" height="120" viewBox="0 0 120 120" className="animate-spin" style={{ animationDuration: "1.2s" }}>
                {/* Outer tire rubber */}
                <circle cx="60" cy="60" r="55" fill="none" stroke="#27272A" strokeWidth="10" />
                {/* Tire tread marks */}
                {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((angle) => (
                  <line key={angle} x1="60" y1="5" x2="60" y2="15" stroke="#3F3F46" strokeWidth="3" strokeLinecap="round"
                    transform={`rotate(${angle} 60 60)`} />
                ))}
                {/* Red compound band */}
                <circle cx="60" cy="60" r="50" fill="none" stroke="#DC2626" strokeWidth="3" />
                {/* Inner sidewall */}
                <circle cx="60" cy="60" r="38" fill="none" stroke="#3F3F46" strokeWidth="6" />
                {/* Rim */}
                <circle cx="60" cy="60" r="30" fill="#18181B" stroke="#52525B" strokeWidth="2" />
                {/* Rim spokes */}
                {[0, 72, 144, 216, 288].map((angle) => (
                  <line key={angle} x1="60" y1="35" x2="60" y2="55" stroke="#71717A" strokeWidth="3" strokeLinecap="round"
                    transform={`rotate(${angle} 60 60)`} />
                ))}
                {/* Center hub */}
                <circle cx="60" cy="60" r="8" fill="#52525B" />
                <circle cx="60" cy="60" r="4" fill="#DC2626" />
              </svg>
              {/* Glow effect */}
              <div className="absolute inset-0 rounded-full" style={{ boxShadow: "0 0 40px rgba(220,38,38,0.2)" }} />
            </div>
            <p className="text-white font-black text-xl tracking-tight mb-1">LIGHTS OUT...</p>
            <p className="text-zinc-500 text-sm mb-5">Crunching telemetry data</p>
            {/* Progress steps */}
            <div className="flex gap-2">
              {["Qualifying Grid", "FastF1 Telemetry", "Historical Form"].map((s, i) => (
                <div key={i} className="flex items-center gap-1.5 text-[10px] px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 animate-fade-in" style={{ animationDelay: `${i * 0.4}s`, opacity: 0 }}>
                  <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" style={{ animationDelay: `${i * 0.4}s` }} />
                  <span className="text-zinc-400">{s}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ═══ ERROR ═══ */}
        {error && !loading && (
          <div className="p-6 rounded-2xl bg-red-950/20 border border-red-900/30 animate-fade-in-up">
            <h3 className="font-bold text-red-400 mb-1">Prediction Failed</h3>
            <p className="text-sm text-red-300/70">{error}</p>
            <p className="text-xs text-zinc-600 mt-3">Make sure the Python API is running: <code className="bg-zinc-800 px-2 py-0.5 rounded text-zinc-400">python -m uvicorn f1outcome.api.app:app --reload --app-dir src</code></p>
          </div>
        )}

        {/* ═══ RESULTS ═══ */}
        {data && !loading && (
          <div className="space-y-6">
            {/* Warnings */}
            {data.warnings.length > 0 && (
              <div className="p-3 rounded-xl bg-amber-950/20 border border-amber-900/30 text-amber-400 text-xs flex items-center gap-2">
                <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M12 3l9.66 16.5H2.34L12 3z" /></svg>
                {data.warnings.join(" · ")}
              </div>
            )}

            {/* Tab Bar */}
            <div className="flex items-center gap-4">
              <div className="flex gap-1 bg-zinc-900/60 p-1 rounded-xl border border-white/5">
                {([["grid", "Race Prediction"], ["h2h", "Head-to-Head"]] as const).map(([key, label]) => (
                  <button key={key} onClick={() => setTab(key)}
                    className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${tab === key ? "bg-zinc-700 text-white" : "text-zinc-500 hover:text-zinc-300"}`}>{label}</button>
                ))}
              </div>
              {tab === "grid" && (
                <div className="hidden sm:flex items-center gap-1.5 text-[10px] text-zinc-600">
                  <Tip text="Click any driver row to expand their full stats breakdown"><span className="underline decoration-dotted cursor-help">Click rows to expand</span></Tip>
                </div>
              )}
            </div>

            {/* ───── GRID TAB ───── */}
            {tab === "grid" && (
              <div className="space-y-6">
                {/* Podium */}
                <div className="grid grid-cols-3 gap-3 items-end animate-fade-in-up">
                  {[1, 0, 2].map((pos) => {
                    const driver = data.order[pos];
                    if (!driver) return null;
                    const t = td(driver.driverId, season);
                    const h = pos === 0 ? "h-52" : pos === 1 ? "h-44" : "h-36";
                    const medal = ["🥇", "🥈", "🥉"][pos];
                    const order = pos === 0 ? "order-2" : pos === 1 ? "order-1" : "order-3";
                    return (
                      <div key={driver.driverId} className={`${order} flex flex-col items-center animate-fade-in-up stagger-${pos + 1}`}>
                        <div className="mb-2 text-center">
                          <div className={`text-3xl ${pos === 0 ? "animate-flag-wave" : ""}`}>{medal}</div>
                          <div className="text-sm sm:text-base font-black text-white mt-1">{fmt(driver.driverId)}</div>
                          <div className="text-[10px] text-zinc-500">{t.team}</div>
                        </div>
                        <div className={`w-full ${h} rounded-t-xl racing-shimmer relative overflow-hidden transition-all duration-500 hover:brightness-125 cursor-pointer`}
                          onClick={() => setExpanded(expanded === driver.driverId ? null : driver.driverId)}
                          style={{ background: `linear-gradient(180deg, ${t.color}DD, ${t.color}40)`, border: `1px solid ${t.accent}40`, borderBottom: "none" }}>
                          <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="text-4xl font-black text-white/15">P{pos + 1}</span>
                            <Tip text="The final predicted score combining pace, form, and DNF risk. Higher is better.">
                              <div className="text-base font-bold text-white mt-1">{driver.score_adj.toFixed(2)}</div>
                            </Tip>
                            <Tip text="Probability this driver does not finish the race, based on historical reliability data.">
                              <div className={`text-xs font-semibold mt-0.5 ${driver.p_dnf > 0.15 ? "text-red-300" : "text-emerald-300"}`}>
                                {(driver.p_dnf * 100).toFixed(1)}% DNF
                              </div>
                            </Tip>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Full Grid */}
                <div className="rounded-2xl border border-white/5 overflow-hidden bg-zinc-900/20">
                  {/* Legend */}
                  <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between text-[10px] text-zinc-600">
                    <div className="flex items-center gap-6">
                      <span className="w-8">POS</span>
                      <span>DRIVER</span>
                    </div>
                    <div className="hidden sm:flex items-center gap-8">
                      <Tip text="How likely the AI believes this driver will perform relative to others. Green bar = above midfield, Red = below."><span className="underline decoration-dotted cursor-help w-36">PACE SCORE</span></Tip>
                      <Tip text="Circular gauge showing the calibrated probability of a Did Not Finish based on historical reliability."><span className="underline decoration-dotted cursor-help w-14 text-right">DNF</span></Tip>
                    </div>
                  </div>

                  {data.order.map((driver, idx) => {
                    const t = td(driver.driverId, season);
                    const isOpen = expanded === driver.driverId;
                    const barPct = ((driver.score_adj - worst) / range) * 100;
                    return (
                      <div key={driver.driverId} className={`border-b border-white/5 last:border-0 transition-colors ${isOpen ? "bg-white/[0.03]" : "hover:bg-white/[0.02]"}`}>
                        <div className="flex items-center px-5 py-3 cursor-pointer gap-4" onClick={() => setExpanded(isOpen ? null : driver.driverId)}>
                          {/* Position */}
                          <div className="w-8 h-8 flex items-center justify-center rounded-lg text-xs font-black"
                            style={{ background: `${t.color}30`, color: t.accent }}>
                            {idx + 1}
                          </div>
                          {/* Driver */}
                          <div className="flex items-center gap-2.5 flex-1 min-w-0">
                            <span className="text-base font-black tabular-nums" style={{ color: t.accent }}>{t.num}</span>
                            <div className="min-w-0">
                              <div className="text-sm font-bold text-white truncate">{fmt(driver.driverId)}</div>
                              <div className="text-[10px] text-zinc-600">{t.team}</div>
                            </div>
                          </div>
                          {/* Pace Bar */}
                          <div className="hidden sm:block flex-1 max-w-[180px]">
                            <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                              <div className="h-full rounded-full animate-bar-grow"
                                style={{ width: `${barPct}%`, background: `linear-gradient(90deg, ${t.color}, ${t.accent})`, animationDelay: `${idx * 50}ms` }} />
                            </div>
                            <div className="text-[9px] text-zinc-600 mt-0.5 font-mono tabular-nums">{driver.score_adj > 0 ? "+" : ""}{driver.score_adj.toFixed(2)}</div>
                          </div>
                          {/* DNF Gauge */}
                          <div className="hidden sm:block"><Gauge value={driver.p_dnf} size={40} /></div>
                          {/* Expand */}
                          <svg className={`w-3.5 h-3.5 text-zinc-600 transition-transform flex-shrink-0 ${isOpen ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>

                        {/* Expanded Stats */}
                        {isOpen && (
                          <div className="px-5 pb-4 pt-0 animate-fade-in">
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
                              <MetricCard label="Final Score" value={driver.score_adj.toFixed(3)} tip="The combined prediction score after pace and DNF risk are blended. This is the number that determines the predicted finishing order." />
                              <MetricCard label="Raw Pace" value={driver.score_rank.toFixed(3)} tip="The pure pace ranking score from the LightGBM ranker model, based on qualifying, form, and telemetry — before any DNF penalty is applied." />
                              <MetricCard label="DNF Risk (Display)" value={`${(driver.p_dnf * 100).toFixed(1)}%`} warn={driver.p_dnf > 0.15} tip="The calibrated probability of not finishing. Uses isotonic regression to ensure if the model says 10%, approximately 10% of those predictions actually DNF." />
                              <MetricCard label="DNF Risk (Raw)" value={`${(driver.p_dnf_raw * 100).toFixed(1)}%`} warn={driver.p_dnf_raw > 0.15} tip="The uncalibrated DNF signal from the raw LightGBM classifier. Used internally for score adjustments. May not correspond to real-world probabilities." />
                            </div>
                            {/* Full-width pace bar */}
                            <div className="p-3 rounded-xl bg-black/30 border border-white/5">
                              <div className="flex justify-between text-[10px] text-zinc-500 mb-1.5">
                                <Tip text="Visual representation of this driver's predicted strength relative to the field. The bar spans from worst to best predicted driver.">
                                  <span className="underline decoration-dotted">Relative Pace Position</span>
                                </Tip>
                                <span className="font-mono">{driver.score_adj > 0 ? "+" : ""}{driver.score_adj.toFixed(3)}</span>
                              </div>
                              <div className="h-3 bg-zinc-800 rounded-full overflow-hidden relative">
                                <div className="h-full rounded-full animate-rev"
                                  style={{ width: `${barPct}%`, background: `linear-gradient(90deg, ${t.color}, ${t.accent})` }} />
                              </div>
                              <div className="flex justify-between text-[9px] text-zinc-700 mt-1">
                                <span>Slowest</span><span>Fastest</span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* ───── HEAD-TO-HEAD TAB ───── */}
            {tab === "h2h" && (
              <div className="space-y-6 animate-fade-in-up">
                <p className="text-xs text-zinc-500">Tap two drivers below to compare them head-to-head.</p>

                {/* Driver Selector */}
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 gap-2">
                  {data.order.map((d) => {
                    const t = td(d.driverId, season);
                    const on = cmp.includes(d.driverId);
                    return (
                      <button key={d.driverId} onClick={() => toggleCmp(d.driverId)}
                        className={`p-3 rounded-xl text-left transition-all border ${on ? "border-white/30 bg-white/10 scale-[1.02]" : "border-white/5 bg-zinc-900/30 hover:bg-zinc-800/50"}`}>
                        <div className="text-lg font-black tabular-nums leading-none" style={{ color: t.accent }}>{t.num}</div>
                        <div className="text-xs font-bold text-white mt-1">{fmt(d.driverId)}</div>
                        <div className="text-[10px] text-zinc-600">{t.team}</div>
                      </button>
                    );
                  })}
                </div>

                {/* Comparison Panel */}
                {cmpData.length === 2 && (
                  <div className="animate-fade-in-up">
                    <div className="text-center text-zinc-600 text-xs font-bold uppercase tracking-widest mb-4">vs</div>
                    <div className="grid grid-cols-2 gap-4">
                      {cmpData.map((driver) => {
                        const t = td(driver.driverId, season);
                        const pos = data!.order.findIndex((o) => o.driverId === driver.driverId) + 1;
                        return (
                          <div key={driver.driverId} className="p-5 rounded-2xl border bg-zinc-900/40 racing-shimmer"
                            style={{ borderColor: `${t.accent}30` }}>
                            <div className="flex items-center gap-3 mb-5">
                              <div className="w-12 h-12 rounded-xl flex items-center justify-center text-xl font-black" style={{ background: `${t.color}40`, color: t.accent }}>{t.num}</div>
                              <div>
                                <div className="text-lg font-bold">{fmt(driver.driverId)}</div>
                                <div className="text-[10px] text-zinc-500">{t.team}</div>
                              </div>
                            </div>

                            <div className="space-y-3">
                              <CmpRow label="Predicted Finish" value={`P${pos}`} big tip="The model's predicted finishing position for this race." />
                              <CmpRow label="Final Score" value={driver.score_adj.toFixed(3)} tip="Combined score determining predicted position." />
                              <CmpRow label="Raw Pace" value={driver.score_rank.toFixed(3)} tip="Pure pace score before DNF penalty." />
                              <CmpRow label="DNF Probability" value={`${(driver.p_dnf * 100).toFixed(1)}%`} warn={driver.p_dnf > 0.15} tip="Calibrated chance of not finishing." />
                            </div>
                            <div className="mt-5 flex justify-center"><Gauge value={driver.p_dnf} size={72} /></div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Delta Bar */}
                    {(() => {
                      const delta = cmpData[0].score_adj - cmpData[1].score_adj;
                      const t0 = td(cmpData[0].driverId, season);
                      const t1 = td(cmpData[1].driverId, season);
                      return (
                        <div className="mt-4 p-4 rounded-xl bg-zinc-900/40 border border-white/5">
                          <div className="text-[10px] text-zinc-500 text-center mb-2 font-bold uppercase tracking-widest">Score Advantage</div>
                          <div className="flex items-center gap-3">
                            <span className="text-xs font-bold w-20 text-right truncate" style={{ color: t0.accent }}>{fmt(cmpData[0].driverId)}</span>
                            <div className="flex-1 h-4 bg-zinc-800 rounded-full overflow-hidden relative">
                              <div className="absolute inset-y-0 left-0 rounded-l-full animate-bar-grow"
                                style={{ width: `${50 + (delta / range) * 40}%`, background: `linear-gradient(90deg, ${t0.color}, ${t0.accent})` }} />
                              <div className="absolute inset-y-0 right-0 rounded-r-full animate-bar-grow"
                                style={{ width: `${50 - (delta / range) * 40}%`, background: `linear-gradient(270deg, ${t1.color}, ${t1.accent})` }} />
                            </div>
                            <span className="text-xs font-bold w-20 truncate" style={{ color: t1.accent }}>{fmt(cmpData[1].driverId)}</span>
                          </div>
                          <div className="text-center text-xs font-mono mt-1 text-zinc-400">
                            {delta > 0 ? fmt(cmpData[0].driverId) : fmt(cmpData[1].driverId)} leads by {Math.abs(delta).toFixed(3)}
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Empty State — Photorealistic F1 Car */}
        {!data && !loading && !error && (
          <div className="flex flex-col items-center justify-center py-10 text-center animate-fade-in relative">
            {/* High-Tech Telemetry HUD Animation */}
            <div className="relative w-full max-w-4xl h-64 mb-10 overflow-hidden rounded-3xl bg-black border border-red-900/30 shadow-[0_0_50px_rgba(225,6,0,0.15)] group">
              {/* Scanline Overlay */}
              <div className="hud-scanline z-20" />
              {/* Grid Background */}
              <div className="absolute inset-0 z-0 opacity-20"
                style={{ backgroundImage: 'linear-gradient(rgba(220,38,38,0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(220,38,38,0.2) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />

              <div className="absolute inset-0 z-10 flex items-center justify-between px-10">
                {/* ══ LEFT: Equalizers & Crosshairs ══ */}
                <div className="flex flex-col items-center gap-6 opacity-80">
                  {/* Crosshair radar */}
                  <div className="relative w-16 h-16 pointer-events-none">
                    <svg className="hud-crosshair text-red-600/80" viewBox="0 0 100 100" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="50" cy="50" r="40" strokeDasharray="4 8" />
                      <circle cx="50" cy="50" r="30" />
                      <circle cx="50" cy="50" r="10" fill="currentColor" />
                      <line x1="50" y1="0" x2="50" y2="100" />
                      <line x1="0" y1="50" x2="100" y2="50" />
                    </svg>
                    <div className="absolute inset-0 rounded-full border border-red-500/30 hud-radar" />
                  </div>
                  {/* Equalizer Bars */}
                  <div className="flex items-end gap-2 h-20 w-16 border-b border-red-800/50 pb-1">
                    <div className="w-4 bg-red-600/80 hud-bar-1" />
                    <div className="w-4 bg-red-500/80 hud-bar-2" />
                    <div className="w-4 bg-red-700/80 hud-bar-3" />
                  </div>
                  <div className="text-[9px] font-mono text-red-500 font-bold tracking-widest hud-data-flicker">INP // THROTTLE</div>
                </div>

                {/* ══ CENTER: Main RPM / Speed Dial ══ */}
                <div className="relative w-56 h-56 flex items-center justify-center pointer-events-none">
                  <svg className="absolute inset-0 w-full h-full -rotate-90 drop-shadow-[0_0_15px_rgba(220,38,38,0.6)]" viewBox="0 0 100 100" fill="none">
                    {/* Background track */}
                    <circle cx="50" cy="50" r="45" stroke="#450a0a" strokeWidth="8" />
                    {/* Tick marks */}
                    <circle cx="50" cy="50" r="45" stroke="#7f1d1d" strokeWidth="8" strokeDasharray="2 6" />
                    {/* Sweeping red arc */}
                    <circle cx="50" cy="50" r="45" stroke="#ef4444" strokeWidth="4" strokeLinecap="round" className="hud-rpm-sweep" />
                    {/* Inner decorative ring */}
                    <circle cx="50" cy="50" r="34" stroke="#dc2626" strokeWidth="1" strokeDasharray="4 4" className="hud-crosshair" style={{ animationDuration: '20s', animationDirection: 'reverse' }} />
                  </svg>
                  {/* Center Text */}
                  <div className="text-center">
                    <div className="text-4xl font-black italic text-white tracking-tighter hud-data-flicker" style={{ textShadow: '0 0 10px rgba(255,255,255,0.5)' }}>315<span className="text-xl">kph</span></div>
                    <div className="text-[10px] font-mono text-red-400 font-bold uppercase tracking-widest mt-1">Telemetry Live</div>
                  </div>
                </div>

                {/* ══ RIGHT: Track Map & Data Stream ══ */}
                <div className="flex flex-col items-center gap-6 opacity-80">
                  {/* Abstract Track SVG */}
                  <div className="relative w-32 h-24 mb-2 pointer-events-none drop-shadow-[0_0_10px_rgba(220,38,38,0.4)]">
                    <svg viewBox="0 0 100 60" fill="none" className="w-full h-full">
                      {/* Track background outline */}
                      <path d="M 20 50 L 80 50 C 90 50 90 30 80 30 L 60 30 C 55 30 50 20 60 10 C 70 0 90 10 90 20 C 90 40 80 60 20 60 C 10 60 10 40 20 40 Z" stroke="#7f1d1d" strokeWidth="4" strokeLinejoin="round" />
                      {/* Glowing dot tracer */}
                      <path d="M 20 50 L 80 50 C 90 50 90 30 80 30 L 60 30 C 55 30 50 20 60 10 C 70 0 90 10 90 20 C 90 40 80 60 20 60 C 10 60 10 40 20 40 Z" stroke="#ef4444" strokeWidth="2" strokeLinejoin="round" className="hud-track-line" />
                      <circle cx="20" cy="50" r="3" fill="#fff" className="hud-data-flicker" />
                    </svg>
                    <div className="absolute -bottom-4 text-center w-full text-[9px] font-mono text-red-400 tracking-widest uppercase">SECTOR // 3</div>
                  </div>

                  {/* Scrolling hex data */}
                  <div className="h-16 overflow-hidden text-[8px] font-mono text-red-500/70 leading-tight w-24 text-right">
                    <div className="hud-data-scroll">
                      {Array.from({ length: 15 }).map((_, i) => (
                        <div key={i}>{Math.random().toString(16).slice(2, 8).toUpperCase()} // {Math.floor(Math.random() * 99)}.{Math.floor(Math.random() * 9)}</div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <h3 className="text-4xl font-black mb-3 tracking-tight">
              Welcome to <span className="text-red-500">F1 Oracle</span>
            </h3>
            <p className="text-zinc-400 text-sm max-w-lg mb-8">
              AI-powered race predictions using LightGBM, FastF1 telemetry,
              and 6 years of historical form data with isotonic DNF calibration.
            </p>

            {/* Feature badges */}
            <div className="flex flex-wrap justify-center gap-3 text-[11px] mb-8">
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/60 border border-white/5 hover:border-red-900/50 transition-colors">
                <div className="w-2 h-2 rounded-full bg-red-600 animate-pulse" />
                <span className="text-zinc-400">Live Telemetry</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/60 border border-white/5 hover:border-orange-900/50 transition-colors">
                <div className="w-2 h-2 rounded-full bg-orange-500" />
                <span className="text-zinc-400">2019-2024 Training Data</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/60 border border-white/5 hover:border-emerald-900/50 transition-colors">
                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                <span className="text-zinc-400">Isotonic DNF Calibration</span>
              </div>
            </div>

            {/* CTA arrow */}
            <div className="flex flex-col items-center gap-2 text-zinc-600 text-xs">
              <span>Select a Grand Prix above to begin</span>
              <svg className="w-5 h-5 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
              </svg>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 mt-16">
        <div className="max-w-7xl mx-auto px-6 py-5 flex flex-col sm:flex-row items-center justify-between text-[10px] text-zinc-700 gap-2">
          <span>F1 Oracle Lab &middot; Powered by LightGBM + FastF1 Telemetry + Isotonic Calibration</span>
          <Tip text="The model was trained on 2019-2024 F1 data with strict time-series splits to prevent data leakage.">
            <span className="underline decoration-dotted cursor-help">How does this work?</span>
          </Tip>
        </div>
      </footer>
    </div>
  );
}

/* ═══ Metric Card with Tooltip ═══ */
function MetricCard({ label, value, warn, tip }: { label: string; value: string; warn?: boolean; tip: string }) {
  return (
    <Tip text={tip}>
      <div className="p-3 rounded-xl bg-black/30 border border-white/5 cursor-help">
        <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-bold mb-1 flex items-center gap-1">
          {label}
          <svg className="w-3 h-3 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="10" strokeWidth="2" /><path strokeLinecap="round" strokeWidth="2" d="M12 16v-4m0-4h.01" /></svg>
        </div>
        <div className={`text-sm font-mono font-bold ${warn ? "text-red-400" : "text-white"}`}>{value}</div>
      </div>
    </Tip>
  );
}

/* ═══ Comparison Row ═══ */
function CmpRow({ label, value, warn, big, tip }: { label: string; value: string; warn?: boolean; big?: boolean; tip: string }) {
  return (
    <Tip text={tip}>
      <div className="flex justify-between items-center cursor-help">
        <span className="text-[10px] text-zinc-500 flex items-center gap-1">
          {label}
          <svg className="w-2.5 h-2.5 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="10" strokeWidth="2" /><path strokeLinecap="round" strokeWidth="2" d="M12 16v-4m0-4h.01" /></svg>
        </span>
        <span className={`font-mono font-bold ${big ? "text-lg text-white" : warn ? "text-sm text-red-400" : "text-sm text-zinc-300"}`}>{value}</span>
      </div>
    </Tip>
  );
}
