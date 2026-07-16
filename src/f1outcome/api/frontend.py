PRODUCT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>F1 Outcome Lab</title>
  <meta name="description" content="Formula 1 race outcome prediction dashboard" />
  <style>
    :root {
      color-scheme: dark;
      --bg: #090a0f;
      --panel: #11131a;
      --panel-2: #171923;
      --line: #282b36;
      --text: #f7f7f8;
      --muted: #a3a8b8;
      --soft: #747b91;
      --red: #e10600;
      --cyan: #21d4c4;
      --green: #50d985;
      --yellow: #f6c453;
      --blue: #5b8cff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 20% 0%, rgba(225, 6, 0, .22), transparent 30rem),
        radial-gradient(circle at 80% 10%, rgba(33, 212, 196, .14), transparent 28rem),
        linear-gradient(180deg, #090a0f, #0d0f16 42%, #090a0f);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, select { font: inherit; }
    button { cursor: pointer; }
    .shell { max-width: 1180px; margin: 0 auto; padding: 22px; }
    .topbar {
      position: sticky; top: 0; z-index: 20;
      border-bottom: 1px solid rgba(255,255,255,.08);
      background: rgba(9,10,15,.86); backdrop-filter: blur(16px);
    }
    .topbar-inner {
      max-width: 1180px; height: 64px; margin: 0 auto; padding: 0 22px;
      display: flex; align-items: center; justify-content: space-between; gap: 18px;
    }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .mark { display: grid; grid-template-columns: repeat(3, 5px); gap: 3px; height: 28px; }
    .mark span { display: block; border-radius: 8px; background: var(--red); }
    .mark span:nth-child(2) { opacity: .62; }
    .mark span:nth-child(3) { opacity: .28; }
    .brand h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    .brand p { margin: 1px 0 0; color: var(--muted); font-size: 12px; }
    .status { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: end; }
    .pill {
      display: inline-flex; align-items: center; gap: 7px;
      min-height: 30px; padding: 6px 10px;
      border: 1px solid rgba(255,255,255,.09);
      border-radius: 999px; background: rgba(255,255,255,.045);
      color: var(--muted); font-size: 12px; white-space: nowrap;
    }
    .dot { width: 7px; height: 7px; border-radius: 99px; background: var(--soft); }
    .dot.ok { background: var(--green); box-shadow: 0 0 16px rgba(80,217,133,.5); }
    .hero {
      display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(300px, .92fr);
      gap: 18px; align-items: stretch; margin: 22px 0 18px;
    }
    .hero-main, .control-panel, .panel {
      border: 1px solid rgba(255,255,255,.08);
      background: linear-gradient(180deg, rgba(23,25,35,.88), rgba(17,19,26,.92));
      box-shadow: 0 20px 70px rgba(0,0,0,.22);
    }
    .hero-main {
      min-height: 360px; position: relative; overflow: hidden; padding: 28px;
      display: flex; flex-direction: column; justify-content: space-between;
    }
    .hero-main:before {
      content: ""; position: absolute; inset: 0; opacity: .25;
      background-image:
        linear-gradient(45deg, rgba(255,255,255,.08) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(255,255,255,.08) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, rgba(255,255,255,.08) 75%),
        linear-gradient(-45deg, transparent 75%, rgba(255,255,255,.08) 75%);
      background-size: 24px 24px; background-position: 0 0, 0 12px, 12px -12px, -12px 0;
      mask-image: linear-gradient(90deg, transparent, #000 55%, transparent);
    }
    .hero-copy { position: relative; z-index: 1; max-width: 640px; }
    .eyebrow { color: var(--cyan); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .14em; }
    h2 { margin: 10px 0 10px; font-size: clamp(36px, 7vw, 78px); line-height: .88; letter-spacing: 0; }
    .hero-copy p { margin: 0; max-width: 570px; color: var(--muted); font-size: 15px; }
    .car-wrap { position: relative; z-index: 1; margin-top: 24px; min-height: 112px; }
    .car-wrap img { width: min(520px, 92%); max-height: 190px; object-fit: contain; filter: drop-shadow(0 24px 35px rgba(0,0,0,.55)); }
    .speedline { position: absolute; left: 0; right: 12%; height: 2px; background: linear-gradient(90deg, transparent, rgba(225,6,0,.85), transparent); opacity: .55; }
    .speedline.a { bottom: 36px; }
    .speedline.b { bottom: 62px; right: 28%; background: linear-gradient(90deg, transparent, rgba(33,212,196,.8), transparent); }
    .control-panel { padding: 18px; display: flex; flex-direction: column; gap: 14px; }
    .field { display: grid; gap: 7px; }
    .field label { color: var(--soft); font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .1em; }
    select {
      width: 100%; min-height: 44px; padding: 0 12px; color: var(--text);
      border: 1px solid rgba(255,255,255,.12); background: #0c0e14; border-radius: 8px;
    }
    .mode { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .mode button, .primary, .secondary {
      min-height: 44px; border: 1px solid rgba(255,255,255,.1); border-radius: 8px;
      background: rgba(255,255,255,.045); color: var(--text); font-weight: 800;
    }
    .mode button.active { background: var(--red); border-color: var(--red); }
    .primary { background: var(--red); border-color: var(--red); text-transform: uppercase; letter-spacing: .08em; }
    .secondary { color: var(--muted); }
    .primary:disabled, .secondary:disabled { opacity: .55; cursor: wait; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
    .metric, .panel { border-radius: 8px; }
    .metric {
      border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.04);
      padding: 14px; min-height: 88px;
    }
    .metric .label { color: var(--soft); font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
    .metric .value { margin-top: 8px; font-size: 27px; font-weight: 900; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(310px, .75fr); gap: 18px; align-items: start; }
    .panel { padding: 0; overflow: hidden; }
    .panel-head { padding: 16px 18px; border-bottom: 1px solid rgba(255,255,255,.08); display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .panel-head h3 { margin: 0; font-size: 15px; }
    .panel-head span { color: var(--soft); font-size: 12px; }
    .table { width: 100%; border-collapse: collapse; }
    .table th, .table td { padding: 12px 14px; border-bottom: 1px solid rgba(255,255,255,.06); text-align: left; }
    .table th { color: var(--soft); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    .table td.num, .table th.num { text-align: right; font-variant-numeric: tabular-nums; }
    .pos {
      display: inline-grid; place-items: center; width: 30px; height: 30px; border-radius: 8px;
      background: rgba(225,6,0,.16); color: #ff746f; font-weight: 900;
    }
    .driver { font-weight: 850; }
    .team { display: block; color: var(--soft); font-size: 12px; margin-top: 2px; }
    .risk { display: inline-flex; min-width: 64px; justify-content: center; padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 800; }
    .risk.low { color: var(--green); background: rgba(80,217,133,.12); }
    .risk.med { color: var(--yellow); background: rgba(246,196,83,.12); }
    .risk.high { color: #ff7771; background: rgba(225,6,0,.15); }
    .bar { height: 8px; border-radius: 99px; background: #2b2f3b; overflow: hidden; min-width: 90px; }
    .bar span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--red), var(--cyan)); }
    .side-body { padding: 16px 18px; display: grid; gap: 14px; }
    .message { border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.04); border-radius: 8px; padding: 14px; color: var(--muted); }
    .message.error { border-color: rgba(225,6,0,.35); color: #ffaaa6; background: rgba(225,6,0,.1); }
    .calendar { display: grid; gap: 8px; max-height: 330px; overflow: auto; padding-right: 4px; }
    .race-row { display: grid; grid-template-columns: 44px 1fr; gap: 10px; align-items: center; padding: 9px; border: 1px solid rgba(255,255,255,.07); border-radius: 8px; background: rgba(255,255,255,.035); }
    .race-row strong { display: block; }
    .race-row span { color: var(--soft); font-size: 12px; }
    .footer { margin: 18px 0 8px; color: var(--soft); font-size: 12px; display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap; }
    .empty { padding: 28px; color: var(--muted); }
    @media (max-width: 860px) {
      .hero, .grid { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .topbar-inner { height: auto; padding-top: 12px; padding-bottom: 12px; align-items: flex-start; flex-direction: column; }
      .status { justify-content: start; }
      .table th:nth-child(4), .table td:nth-child(4) { display: none; }
    }
    @media (max-width: 560px) {
      .shell { padding: 14px; }
      .hero-main { padding: 20px; min-height: 320px; }
      .metrics { grid-template-columns: 1fr; }
      .table th, .table td { padding: 10px 8px; }
      .table th:nth-child(5), .table td:nth-child(5) { display: none; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="mark" aria-hidden="true"><span></span><span></span><span></span></div>
        <div>
          <h1>F1 Outcome Lab</h1>
          <p>Race prediction dashboard powered by the deployed FastAPI model.</p>
        </div>
      </div>
      <div class="status">
        <span class="pill"><span id="apiDot" class="dot"></span><span id="apiStatus">Checking API</span></span>
        <span class="pill"><span id="dataStatus">Data window loading</span></span>
      </div>
    </div>
  </header>

  <main class="shell">
    <section class="hero">
      <div class="hero-main">
        <div class="hero-copy">
          <div class="eyebrow">Predictive race intelligence</div>
          <h2>From grid to likely finish.</h2>
          <p>Pick a season and race, then run the model to rank the field with pace, form, and non-finish risk blended into one race-order view.</p>
        </div>
        <div class="car-wrap" aria-hidden="true">
          <img src="/assets/f1-car.png" alt="" onerror="this.style.display='none'" />
          <div class="speedline a"></div>
          <div class="speedline b"></div>
        </div>
      </div>

      <aside class="control-panel" aria-label="Prediction controls">
        <div class="field">
          <label for="season">Season</label>
          <select id="season"></select>
        </div>
        <div class="field">
          <label for="race">Race</label>
          <select id="race"></select>
        </div>
        <div class="field">
          <label>Scoring mode</label>
          <div class="mode">
            <button id="modeCap" class="active" type="button">Conservative</button>
            <button id="modeRaw" type="button">Raw risk</button>
          </div>
        </div>
        <button id="predictBtn" class="primary" type="button">Run prediction</button>
        <button id="liveBtn" class="secondary" type="button">Check live qualifying</button>
        <div id="notice" class="message">Historical model data is available for completed dataset seasons. Live mode requires qualifying data for the selected round.</div>
      </aside>
    </section>

    <section class="metrics" aria-label="Model summary">
      <div class="metric"><div class="label">Race</div><div id="metricRace" class="value">--</div></div>
      <div class="metric"><div class="label">Top driver</div><div id="metricWinner" class="value">--</div></div>
      <div class="metric"><div class="label">Backtest top-10</div><div id="metricTop10" class="value">--</div></div>
      <div class="metric"><div class="label">Kendall tau</div><div id="metricTau" class="value">--</div></div>
    </section>

    <section class="grid">
      <div class="panel">
        <div class="panel-head">
          <div><h3>Predicted order</h3><span id="resultMeta">Run a prediction to populate the grid.</span></div>
          <span id="modeLabel">subtract_cap</span>
        </div>
        <div id="results" class="empty">No prediction loaded yet.</div>
      </div>
      <aside class="panel">
        <div class="panel-head"><h3>Race calendar</h3><span id="raceCount">-- races</span></div>
        <div class="side-body">
          <div id="calendar" class="calendar"></div>
          <div id="backtest" class="message">Backtest metrics will load for the selected season when available.</div>
        </div>
      </aside>
    </section>

    <footer class="footer">
      <span>Uses the deployed model artifacts and parquet dataset in this repository.</span>
      <span><a href="/docs" style="color:var(--cyan)">API docs</a> · <a href="/meta" style="color:var(--cyan)">Model metadata</a></span>
    </footer>
  </main>

  <script>
    const state = { seasons: [], races: [], mode: "subtract_cap", prediction: null };
    const teams = {
      red_bull: "Red Bull", mercedes: "Mercedes", ferrari: "Ferrari", mclaren: "McLaren",
      aston_martin: "Aston Martin", alpine: "Alpine", rb: "RB", alphatauri: "AlphaTauri",
      sauber: "Sauber", alfa: "Alfa Romeo", haas: "Haas", williams: "Williams"
    };
    const $ = (id) => document.getElementById(id);

    function fmtDriver(id) {
      return String(id).split("_").map((part) => part ? part[0].toUpperCase() + part.slice(1) : part).join(" ");
    }
    function riskClass(p) { return p >= .20 ? "high" : p >= .12 ? "med" : "low"; }
    function riskLabel(p) { return p >= .20 ? "High" : p >= .12 ? "Medium" : "Low"; }
    function setNotice(text, isError = false) {
      $("notice").textContent = text;
      $("notice").className = isError ? "message error" : "message";
    }
    function selectedSeason() { return Number($("season").value); }
    function selectedRound() { return Number($("race").value); }
    function selectedRace() { return state.races.find((race) => race.round === selectedRound()); }

    async function fetchJson(path) {
      const res = await fetch(path, { headers: { accept: "application/json" } });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
      return data;
    }

    async function checkApi() {
      try {
        await fetchJson("/meta");
        $("apiDot").className = "dot ok";
        $("apiStatus").textContent = "API online";
      } catch (err) {
        $("apiStatus").textContent = "API unavailable";
      }
    }

    async function loadSeasons() {
      const all = await fetchJson("/races");
      const seasons = [...new Set(all.map((race) => race.season))].sort((a, b) => b - a);
      state.seasons = seasons;
      $("season").innerHTML = seasons.map((season) => `<option value="${season}">${season}</option>`).join("");
      $("dataStatus").textContent = seasons.length ? `Dataset through ${Math.max(...seasons)}` : "No dataset seasons";
      await loadRaces();
    }

    async function loadRaces() {
      const season = selectedSeason();
      state.races = await fetchJson(`/races?season=${season}`);
      $("race").innerHTML = state.races.map((race) => `<option value="${race.round}">Round ${race.round} - ${race.raceName || race.raceId}</option>`).join("");
      $("raceCount").textContent = `${state.races.length} races`;
      renderCalendar();
      loadBacktest();
      $("metricRace").textContent = season;
      $("metricWinner").textContent = "--";
      $("resultMeta").textContent = "Run a prediction to populate the grid.";
      $("results").className = "empty";
      $("results").textContent = "No prediction loaded yet.";
    }

    function renderCalendar() {
      $("calendar").innerHTML = state.races.map((race) => `
        <div class="race-row">
          <strong>R${race.round}</strong>
          <div><strong>${race.raceName || race.raceId}</strong><span>${race.season} · ${race.raceId}</span></div>
        </div>
      `).join("");
    }

    async function loadBacktest() {
      const season = selectedSeason();
      $("backtest").textContent = "Loading backtest metrics...";
      $("metricTop10").textContent = "--";
      $("metricTau").textContent = "--";
      try {
        const bt = await fetchJson(`/predict/backtest?season=${season}`);
        $("metricTop10").textContent = `${Math.round(bt.avg_top10_precision * 100)}%`;
        $("metricTau").textContent = bt.avg_kendall_tau.toFixed(3);
        $("backtest").innerHTML = `Backtest ${season}: top-3 hit rate ${Math.round(bt.avg_top3_rate * 100)}%, top-10 precision ${Math.round(bt.avg_top10_precision * 100)}%, Kendall tau ${bt.avg_kendall_tau.toFixed(3)}.`;
      } catch (err) {
        $("backtest").textContent = `Backtest unavailable for ${season}: ${err.message}`;
      }
    }

    async function predict(kind = "historical") {
      const btn = kind === "live" ? $("liveBtn") : $("predictBtn");
      btn.disabled = true;
      btn.textContent = kind === "live" ? "Checking..." : "Running...";
      setNotice(kind === "live" ? "Checking live qualifying data..." : "Running deployed model...");
      try {
        const season = selectedSeason();
        const round = selectedRound();
        const path = kind === "live"
          ? `/predict/live?season=${season}&round=${round}&mode=${state.mode}`
          : `/predict/from_parquet?season=${season}&round=${round}&mode=${state.mode}`;
        const data = await fetchJson(path);
        state.prediction = data;
        renderPrediction(data, kind);
        setNotice(kind === "live" ? "Live qualifying prediction loaded." : "Historical prediction loaded from deployed model artifacts.");
      } catch (err) {
        setNotice(err.message, true);
      } finally {
        btn.disabled = false;
        btn.textContent = kind === "live" ? "Check live qualifying" : "Run prediction";
      }
    }

    function renderPrediction(data, kind) {
      const race = selectedRace();
      const scores = data.order.map((driver) => driver.score_adj);
      const min = Math.min(...scores);
      const max = Math.max(...scores);
      const span = max - min || 1;
      $("metricRace").textContent = race ? `R${race.round}` : data.raceId;
      $("metricWinner").textContent = fmtDriver(data.order[0].driverId).split(" ").slice(-1)[0];
      $("modeLabel").textContent = data.mode;
      $("resultMeta").textContent = `${race ? race.raceName : data.raceId} · ${kind === "live" ? "live qualifying" : "dataset prediction"} · alpha ${data.alpha}`;
      $("results").className = "";
      $("results").innerHTML = `
        <table class="table">
          <thead>
            <tr><th>Pos</th><th>Driver</th><th>Risk</th><th class="num">DNF</th><th class="num">Score</th><th>Pace</th></tr>
          </thead>
          <tbody>
            ${data.order.map((driver, index) => {
              const pct = ((driver.score_adj - min) / span) * 100;
              const risk = riskClass(driver.p_dnf);
              return `<tr>
                <td><span class="pos">${index + 1}</span></td>
                <td><span class="driver">${fmtDriver(driver.driverId)}</span><span class="team">Model driver id: ${driver.driverId}</span></td>
                <td><span class="risk ${risk}">${riskLabel(driver.p_dnf)}</span></td>
                <td class="num">${(driver.p_dnf * 100).toFixed(1)}%</td>
                <td class="num">${driver.score_adj.toFixed(3)}</td>
                <td><div class="bar"><span style="width:${pct.toFixed(1)}%"></span></div></td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>`;
    }

    $("season").addEventListener("change", loadRaces);
    $("predictBtn").addEventListener("click", () => predict("historical"));
    $("liveBtn").addEventListener("click", () => predict("live"));
    $("modeCap").addEventListener("click", () => {
      state.mode = "subtract_cap";
      $("modeCap").classList.add("active");
      $("modeRaw").classList.remove("active");
    });
    $("modeRaw").addEventListener("click", () => {
      state.mode = "subtract";
      $("modeRaw").classList.add("active");
      $("modeCap").classList.remove("active");
    });

    checkApi();
    loadSeasons().catch((err) => setNotice(err.message, true));
  </script>
</body>
</html>"""
