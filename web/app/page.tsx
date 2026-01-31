"use client";

import { useEffect, useMemo, useState } from "react";

type RaceInfo = {
  season: number;
  round: number;
  raceId: string;
  raceName?: string | null;
};

type ScoredDriver = {
  driverId: string;
  score_rank: number;
  p_dnf_raw: number;
  p_dnf: number;
  score_adj: number;
};

type PredictResponse = {
  raceId: string;
  order: ScoredDriver[];
  alpha: number;
  p_dnf_cap: number;
  mode: "subtract" | "subtract_cap";
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

function riskLabel(p: number) {
  if (p >= 0.25) return "High";
  if (p >= 0.15) return "Med";
  return "Low";
}

export default function Home() {
  const [season, setSeason] = useState<number>(2024);
  const [races, setRaces] = useState<RaceInfo[]>([]);
  const [round, setRound] = useState<number>(1);
  const [mode, setMode] = useState<"subtract" | "subtract_cap">("subtract_cap");

  const [loadingRaces, setLoadingRaces] = useState(false);
  const [loadingPred, setLoadingPred] = useState(false);
  const [pred, setPred] = useState<PredictResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // load races when season changes
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoadingRaces(true);
      setErr(null);
      try {
        const res = await fetch(`${API_BASE}/races?season=${season}`);
        if (!res.ok) throw new Error(`Failed to load races (${res.status})`);
        const data: RaceInfo[] = await res.json();
        if (cancelled) return;
        setRaces(data);
        setRound(data.length ? data[0].round : 1);
      } catch (e: any) {
        if (!cancelled) setErr(e.message || "Failed to load races");
      } finally {
        if (!cancelled) setLoadingRaces(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [season]);

  const roundOptions = useMemo(() => races.map((r) => r.round), [races]);

  async function predict() {
    setLoadingPred(true);
    setErr(null);
    setPred(null);
    try {
      const url = `${API_BASE}/predict/from_parquet?season=${season}&round=${round}&mode=${mode}`;
      const res = await fetch(url);
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j?.detail || `Predict failed (${res.status})`);
      }
      const data: PredictResponse = await res.json();
      setPred(data);
    } catch (e: any) {
      setErr(e.message || "Prediction failed");
    } finally {
      setLoadingPred(false);
    }
  }

  return (
    <main style={{ maxWidth: 1000, margin: "0 auto", padding: 24, fontFamily: "system-ui" }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>F1 Outcome Lab</h1>
      <p style={{ marginTop: 0, opacity: 0.8 }}>
        Pre-race ranking with DNF risk (raw for scoring, calibrated for display).
      </p>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 16 }}>
        <label>
          <div style={{ fontSize: 12, opacity: 0.8 }}>Season</div>
          <select value={season} onChange={(e) => setSeason(Number(e.target.value))} style={{ padding: 8 }}>
            {[2019, 2020, 2021, 2022, 2023, 2024].map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>

        <label>
          <div style={{ fontSize: 12, opacity: 0.8 }}>Round</div>
          <select
            value={round}
            onChange={(e) => setRound(Number(e.target.value))}
            disabled={loadingRaces || roundOptions.length === 0}
            style={{ padding: 8 }}
          >
            {roundOptions.map((r) => {
              const race = races.find((x) => x.round === r);
              return (
                <option key={r} value={r}>
                  {r}{race?.raceName ? ` — ${race.raceName}` : ""}
                </option>
              );
            })}
          </select>
        </label>

        <label>
          <div style={{ fontSize: 12, opacity: 0.8 }}>Mode</div>
          <select value={mode} onChange={(e) => setMode(e.target.value as any)} style={{ padding: 8 }}>
            <option value="subtract_cap">Conservative (cap)</option>
            <option value="subtract">Accuracy (no cap)</option>
          </select>
        </label>

        <button
          onClick={predict}
          disabled={loadingPred || loadingRaces}
          style={{ padding: "10px 14px", fontWeight: 600, cursor: "pointer", alignSelf: "end" }}
        >
          {loadingPred ? "Predicting..." : "Predict"}
        </button>
      </div>

      {err && (
        <div style={{ marginTop: 16, padding: 12, border: "1px solid #ffb3b3", borderRadius: 8 }}>
          <b>Error:</b> {err}
        </div>
      )}

      {pred && (
        <div style={{ marginTop: 18 }}>
          <div style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
            <h2 style={{ margin: 0 }}>Race {pred.raceId}</h2>
            <span style={{ opacity: 0.75 }}>
              α={pred.alpha}, cap={pred.p_dnf_cap}, mode={pred.mode}
            </span>
          </div>

          <div style={{ overflowX: "auto", marginTop: 10 }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", padding: 8, borderBottom: "1px solid #333" }}>Pos</th>
                  <th style={{ textAlign: "left", padding: 8, borderBottom: "1px solid #333" }}>Driver</th>
                  <th style={{ textAlign: "left", padding: 8, borderBottom: "1px solid #333" }}>Risk</th>
                  <th style={{ textAlign: "right", padding: 8, borderBottom: "1px solid #333" }}>p(non-finish)</th>
                  <th style={{ textAlign: "right", padding: 8, borderBottom: "1px solid #333" }}>Score</th>
                </tr>
              </thead>
              <tbody>
                {pred.order.map((d, i) => (
                  <tr key={d.driverId}>
                    <td style={{ padding: 8, borderBottom: "1px solid #222" }}>{i + 1}</td>
                    <td style={{ padding: 8, borderBottom: "1px solid #222" }}>{d.driverId}</td>
                    <td style={{ padding: 8, borderBottom: "1px solid #222" }}>{riskLabel(d.p_dnf)}</td>
                    <td style={{ padding: 8, borderBottom: "1px solid #222", textAlign: "right" }}>
                      {d.p_dnf.toFixed(3)}
                    </td>
                    <td style={{ padding: 8, borderBottom: "1px solid #222", textAlign: "right" }}>
                      {d.score_adj.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p style={{ marginTop: 10, opacity: 0.8 }}>
            Note: <code>p(non-finish)</code> is calibrated for display; scoring uses raw risk (capped in conservative mode).
          </p>
        </div>
      )}
    </main>
  );
}
