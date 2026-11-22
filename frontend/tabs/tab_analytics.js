/**
 * MODULE 2: RACE ENGINEERING, TYRES, WEATHER, CONSTRUCTORS & PIT SIMULATOR
 * File: frontend/tabs/tab_analytics.js
 * Mirrors all 6 engineering analytics views + Pit-Stop Crossover Simulator from app.py
 */

window.ActiveAnalyticsModule = 'pit_sim';

// Helper: Fits Ordinary Least Squares (y = m*x + b) trendline points in JavaScript
function fitOlsLine(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - meanX) * (ys[i] - meanY);
    den += (xs[i] - meanX) ** 2;
  }
  if (den === 0) return null;
  const slope = num / den;
  const intercept = meanY - slope * meanX;
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  return { x: [minX, maxX], y: [slope * minX + intercept, slope * maxX + intercept], slope };
}

window.loadAnalyticsWorkspace = async function () {
  const container = document.getElementById('workspace-analytics');
  const ov = PitWallState.overview;
  if (!ov) return;

  // Compact single-row dropdown selector (spacious layout with zero clutter)
  container.innerHTML = `
    <div class="pit-card p-4 flex flex-wrap items-center justify-between gap-4">
      <div>
        <h2 class="text-sm font-extrabold text-white">Engineering &amp; Strategy Analytics — ${ov.year} ${ov.event_name} (${ov.session_type})</h2>
        <p class="text-xs text-slate-400">Select any engineering discipline below. All charts &amp; tables stack at 100% full screen width with Graph &amp; Table Summaries.</p>
      </div>
      <div class="min-w-[360px]">
        <label class="block text-[10px] uppercase text-slate-400 font-bold mb-1">Select Engineering Analysis View</label>
        <select id="analytics-module-select" onchange="onAnalyticsModuleChanged()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-bold text-white">
          <option value="pit_sim" ${ActiveAnalyticsModule === 'pit_sim' ? 'selected' : ''}>🛠️ Interactive Pit-Wall Strategy Crossover Simulator (1-Stop vs. 2-Stop)</option>
          <option value="race_flow" ${ActiveAnalyticsModule === 'race_flow' ? 'selected' : ''}>📈 Pit-Wall Race Trace ('Worm Chart'), Position Flow &amp; Overtakes</option>
          <option value="tyres" ${ActiveAnalyticsModule === 'tyres' ? 'selected' : ''}>🛞 Tyre Stint Strategy Timeline, Fuel-Corrected OLS Deg &amp; Pace</option>
          <option value="weather" ${ActiveAnalyticsModule === 'weather' ? 'selected' : ''}>🌧️ Weather Crossovers ('Was It Worth It?'), SC/VSC &amp; Reliability</option>
          <option value="constructors" ${ActiveAnalyticsModule === 'constructors' ? 'selected' : ''}>🏗️ Constructor Aero Philosophy Quadrant &amp; Pit-Wall Strategy Audit</option>
          <option value="driver_dna" ${ActiveAnalyticsModule === 'driver_dna' ? 'selected' : ''}>🧬 Multi-Axis Driver DNA Radar, Optimal Sectors &amp; Teammates</option>
        </select>
      </div>
    </div>
    <div id="analytics-content" class="space-y-6">
      <div class="pit-card p-6 text-xs text-slate-400 animate-pulse">Loading DuckDB engineering analytics...</div>
    </div>
  `;

  await renderActiveAnalyticsModule();
};

window.onAnalyticsModuleChanged = async function () {
  window.ActiveAnalyticsModule = document.getElementById('analytics-module-select').value;
  await renderActiveAnalyticsModule();
};

async function renderActiveAnalyticsModule() {
  const mod = window.ActiveAnalyticsModule;
  if (mod === 'pit_sim') await renderPitSimulatorModule();
  else if (mod === 'race_flow') await renderRaceFlowModule();
  else if (mod === 'tyres') await renderTyreAnalyticsModule();
  else if (mod === 'weather') await renderWeatherIncidentsModule();
  else if (mod === 'constructors') await renderConstructorsModule();
  else if (mod === 'driver_dna') await renderDriverDnaModule();
}

// ============================================================================
// SUB-MODULE 1: PIT-WALL STRATEGY CROSSOVER SIMULATOR (1-STOP vs. 2-STOP)
// ============================================================================
async function renderPitSimulatorModule() {
  const { year, round, overview } = PitWallState;
  const host = document.getElementById('analytics-content');

  const t0 = performance.now();
  const profRes = await fetch(`/api/analytics/track-profile/${year}/${round}`);
  const prof = await profRes.json();
  recordLatency(performance.now() - t0);

  host.innerHTML = `
    ${buildConceptHeader(
      `Evaluates the exact crossover between pit-lane time loss (~22s) and cumulative thermal tyre wear across <strong>${prof.total_laps} laps</strong> at <strong>${prof.event_name} (${prof.location})</strong> to select the winning 1-stop vs. 2-stop race plan.`,
      `<strong>Virtual Reference Car Detrending:</strong> Subtracts constant-speed winner pace from cumulative strategy time so pit stops appear as vertical drops and fresh-tyre undercuts appear as steep upward recovery slopes.`,
      `Parameters below are auto-calibrated from <strong>${year} ${prof.event_name}</strong> DuckDB clean stints (Base Pace: <strong>${prof.base_pace_sec}s</strong>, Soft Deg: <strong>${prof.soft_deg} s/lap</strong>, Med Deg: <strong>${prof.medium_deg} s/lap</strong>).`
    )}

    <div class="pit-card p-5 grid grid-cols-2 md:grid-cols-6 gap-4 items-end">
      <div>
        <label class="block text-xs text-slate-300 mb-1">Race Laps: <span id="ps-v-laps" class="text-red-400 font-bold">${prof.total_laps}</span></label>
        <input type="range" id="ps-laps" min="35" max="78" value="${prof.total_laps}" oninput="document.getElementById('ps-v-laps').textContent=this.value" onchange="runAnalyticsPitSim()" class="w-full accent-red-500" />
      </div>
      <div>
        <label class="block text-xs text-slate-300 mb-1">Base Pace (s): <span id="ps-v-pace" class="text-red-400 font-bold">${prof.base_pace_sec}</span></label>
        <input type="range" id="ps-pace" min="65" max="115" step="0.5" value="${prof.base_pace_sec}" oninput="document.getElementById('ps-v-pace').textContent=this.value" onchange="runAnalyticsPitSim()" class="w-full accent-red-500" />
      </div>
      <div>
        <label class="block text-xs text-slate-300 mb-1">Pit Loss (s): <span id="ps-v-loss" class="text-red-400 font-bold">22.0</span></label>
        <input type="range" id="ps-loss" min="15" max="30" step="0.5" value="22.0" oninput="document.getElementById('ps-v-loss').textContent=this.value" onchange="runAnalyticsPitSim()" class="w-full accent-red-500" />
      </div>
      <div>
        <label class="block text-xs text-red-400 font-semibold mb-1">SOFT Deg (s/lap)</label>
        <input type="number" id="ps-soft" value="${prof.soft_deg}" step="0.01" onchange="runAnalyticsPitSim()" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-white" />
      </div>
      <div>
        <label class="block text-xs text-amber-400 font-semibold mb-1">MED Deg (s/lap)</label>
        <input type="number" id="ps-med" value="${prof.medium_deg}" step="0.01" onchange="runAnalyticsPitSim()" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-white" />
      </div>
      <div>
        <label class="block text-xs text-slate-300 font-semibold mb-1">HARD Deg (s/lap)</label>
        <input type="number" id="ps-hard" value="${prof.hard_deg}" step="0.01" onchange="runAnalyticsPitSim()" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs font-mono text-white" />
      </div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <div class="flex flex-wrap items-center justify-between gap-2">
        <h3 class="text-base font-bold text-white">📈 Lap-by-Lap Strategy Crossover Trace vs. Virtual Constant-Pace Winner</h3>
        <span id="ps-best-badge" class="text-xs px-3 py-1 rounded bg-emerald-950 border border-emerald-700 text-emerald-300 font-bold">Simulating...</span>
      </div>
      <div id="ps-crossover-chart" class="w-full h-[440px]"></div>
      <div id="ps-crossover-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📊 Total Projected Race Time Comparison (1-Stop vs. 2-Stop)</h3>
      <div id="ps-bar-chart" class="w-full h-[360px]"></div>
      <div id="ps-bar-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📋 Simulated Pit-Stop Strategy Benchmark Table</h3>
      <div id="ps-table"></div>
      <div id="ps-table-summary" class="table-summary"></div>
    </div>
  `;

  await runAnalyticsPitSim();
}

window.runAnalyticsPitSim = async function () {
  const payload = {
    total_laps: parseInt(document.getElementById('ps-laps').value),
    base_pace_sec: parseFloat(document.getElementById('ps-pace').value),
    pit_loss_sec: parseFloat(document.getElementById('ps-loss').value),
    soft_deg: parseFloat(document.getElementById('ps-soft').value),
    medium_deg: parseFloat(document.getElementById('ps-med').value),
    hard_deg: parseFloat(document.getElementById('ps-hard').value),
  };

  const t0 = performance.now();
  const res = await fetch('/api/analytics/pit-simulator', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const data = await res.json();
  recordLatency(performance.now() - t0);

  const tracesRaw = data.lap_traces || [];
  const summary = data.summary || [];
  if (!summary.length) return;

  const best = summary[0];
  const runnerUp = summary[1] || best;
  const gapSec = (runnerUp.CumulativeTime_Sec - best.CumulativeTime_Sec).toFixed(2);
  document.getElementById('ps-best-badge').textContent = `Optimal Call: ${best.Strategy} (${best.CumulativeTime_Sec.toFixed(1)}s)`;

  // Chart 1: Lap-by-Lap Crossover Line Chart (Inverted Y-axis: Higher = Ahead on Track)
  const byStrat = {};
  tracesRaw.forEach(r => { if (!byStrat[r.Strategy]) byStrat[r.Strategy] = []; byStrat[r.Strategy].push(r); });
  const colors = ['#10b981', '#38bdf8', '#f59e0b', '#ef4444'];
  const lineTraces = Object.keys(byStrat).map((strat, i) => ({
    x: byStrat[strat].map(r => r.LapNumber),
    y: byStrat[strat].map(r => r.GapToVirtualWinner_Sec),
    name: strat,
    type: 'scatter', mode: 'lines',
    line: { color: colors[i % colors.length], width: strat === best.Strategy ? 3 : 2 }
  }));

  Plotly.newPlot('ps-crossover-chart', lineTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 },
    margin: { t: 20, r: 20, b: 45, l: 65 },
    xaxis: { title: 'Race Lap Number', gridcolor: '#1e293b' },
    yaxis: { title: 'Gap to Virtual Winner (s) — Higher = Ahead', autorange: 'reversed', gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: 1.08, x: 0 }
  }, { responsive: true });

  document.getElementById('ps-crossover-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Vertical drops represent the <strong>${payload.pit_loss_sec}s</strong> pit-lane time loss; steep upward slopes show fresh tyres hunting down worn rubber. <strong>${best.Strategy}</strong> crosses the checkered flag ahead of <strong>${runnerUp.Strategy}</strong> by <strong>${gapSec}s</strong>.`;

  // Chart 2: Total Time Comparison Bar Chart
  Plotly.newPlot('ps-bar-chart', [{
    x: summary.map(r => r.Strategy),
    y: summary.map(r => r.CumulativeTime_Sec),
    type: 'bar',
    text: summary.map(r => r.GapToFastest_Sec === 0 ? 'FASTEST' : `+${r.GapToFastest_Sec}s`),
    textposition: 'outside',
    marker: { color: summary.map((_, i) => i === 0 ? '#10b981' : '#ef4444') }
  }], {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 },
    margin: { t: 25, r: 20, b: 65, l: 65 },
    yaxis: { title: 'Total Projected Race Time (s)', range: [best.CumulativeTime_Sec - 30, summary[summary.length - 1].CumulativeTime_Sec + 30], gridcolor: '#1e293b' }
  }, { responsive: true });

  document.getElementById('ps-bar-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Zoomed Y-axis exposes exact multi-second gaps between 1-stop and 2-stop plans. Adjust the Pit Loss or Soft/Med degradation sliders above to locate the exact strategy crossover threshold.`;

  // Table 1
  document.getElementById('ps-table').innerHTML = renderDataTable(
    summary,
    ['Strategy', 'LapNumber', 'Compound', 'CumulativeTime_Sec', 'GapToFastest_Sec'],
    { Strategy: 'Simulated Pit Strategy', LapNumber: 'Race Laps', Compound: 'Final Compound', CumulativeTime_Sec: 'Total Time (s)', GapToFastest_Sec: 'Deficit vs Fastest (+s)' }
  );
  document.getElementById('ps-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Ranks all ${summary.length} candidate pit-wall strategies from fastest to slowest across ${payload.total_laps} laps, with a total spread of <strong>${(summary[summary.length - 1].CumulativeTime_Sec - best.CumulativeTime_Sec).toFixed(2)}s</strong>.`;
};

// ============================================================================
// SUB-MODULE 2: PIT-WALL RACE TRACE ('WORM CHART'), POSITION FLOW & OVERTAKES
// ============================================================================
async function renderRaceFlowModule() {
  const { year, round, session, overview } = PitWallState;
  const host = document.getElementById('analytics-content');

  const t0 = performance.now();
  const res = await fetch(`/api/analytics/race-flow/${year}/${round}?session_type=${session}`);
  const data = await res.json();
  recordLatency(performance.now() - t0);

  const trace = data.trace || [];
  const posFlow = data.pos_flow || [];
  const overtakes = data.overtakes || [];
  const notes = data.notes || {};

  if (!trace.length) {
    host.innerHTML = `<div class="pit-card p-6 text-sm text-amber-300">Race Trace &amp; Overtake Flow applies to Race (R) or Sprint (S) sessions. Switch Session to <strong>Race (R)</strong> and click <strong>Load Session</strong>.</div>`;
    return;
  }

  // Highlight top 6 finishers by default so the worm chart stays crisp and readable
  const topDrivers = overtakes.slice(0, 6).map(r => r.Driver);
  const colors = ['#00F5D4', '#FF2A85', '#FFE600', '#38BDF8', '#A855F7', '#FB923C'];

  host.innerHTML = `
    ${buildConceptHeader(notes.f1_context, notes.ds_methodology, notes.key_insight)}

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📈 Pit-Wall Race Trace ('Worm Chart'): Cumulative Gap vs. Virtual Reference Winner Car (Top 6)</h3>
      <div id="rf-worm-chart" class="w-full h-[460px]"></div>
      <div id="rf-worm-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🔀 Lap-by-Lap Track Position Flow (Grid → Checkered Flag, Top 6)</h3>
      <div id="rf-bump-chart" class="w-full h-[440px]"></div>
      <div id="rf-bump-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">⚡ Net Positions Gained Decomposition: Lap 1 Launch vs. Post-Lap 1 Overtakes &amp; Strategy</h3>
      <div id="rf-gain-chart" class="w-full h-[480px]"></div>
      <div id="rf-gain-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📋 Complete Grid-to-Flag Position Gain &amp; Overtake Table</h3>
      <div id="rf-table"></div>
      <div id="rf-table-summary" class="table-summary"></div>
    </div>
  `;

  // Chart 1: Worm Chart
  const wormTraces = topDrivers.map((drv, i) => {
    const pts = trace.filter(r => r.Driver === drv);
    return {
      x: pts.map(r => r.LapNumber),
      y: pts.map(r => r.GapToVirtualCar_Sec),
      name: drv, type: 'scatter', mode: 'lines',
      line: { color: colors[i % colors.length], width: 2.2 }
    };
  });
  Plotly.newPlot('rf-worm-chart', wormTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 }, margin: { t: 20, r: 20, b: 45, l: 65 },
    xaxis: { title: 'Lap Number', gridcolor: '#1e293b' },
    yaxis: { title: 'Gap to Virtual Car (s) — Higher = Ahead', autorange: 'reversed', gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: 1.07, x: 0 }
  }, { responsive: true });
  document.getElementById('rf-worm-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Detrends cumulative race time against a constant-speed Virtual Winner Car for the top 6 finishers (${topDrivers.join(', ')}). Vertical ~22s drops pinpoint pit stops; sudden vertical compression across all lines marks Safety Car periods.`;

  // Chart 2: Bump Chart
  const bumpTraces = topDrivers.map((drv, i) => {
    const pts = posFlow.filter(r => r.Driver === drv);
    return {
      x: pts.map(r => r.LapNumber),
      y: pts.map(r => r.Position),
      name: drv, type: 'scatter', mode: 'lines+markers',
      marker: { size: 4 }, line: { color: colors[i % colors.length], width: 2 }
    };
  });
  Plotly.newPlot('rf-bump-chart', bumpTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 }, margin: { t: 20, r: 20, b: 45, l: 55 },
    xaxis: { title: 'Lap Number (Lap 0 = Starting Grid)', gridcolor: '#1e293b' },
    yaxis: { title: 'Track Position (P1 at Top)', autorange: 'reversed', dtick: 1, gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: 1.07, x: 0 }
  }, { responsive: true });
  document.getElementById('rf-bump-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Tracks exact running order from starting grid (Lap 0) to the checkered flag. Every line crossing highlights an on-track overtake or pit-cycle undercut/overcut.`;

  // Chart 3: Lap 1 vs Post-Lap 1 Overtakes
  const sortedGain = [...overtakes].sort((a, b) => (a.Total_PositionsGained || 0) - (b.Total_PositionsGained || 0));
  Plotly.newPlot('rf-gain-chart', [
    { y: sortedGain.map(r => r.Driver), x: sortedGain.map(r => r.Lap1_PositionsGained), name: 'Lap 1 Launch Gain', type: 'bar', orientation: 'h', marker: { color: '#38bdf8' } },
    { y: sortedGain.map(r => r.Driver), x: sortedGain.map(r => r.PostLap1_PositionsGained), name: 'Post-Lap 1 Craft & Strategy', type: 'bar', orientation: 'h', marker: { color: '#10b981' } }
  ], {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 }, barmode: 'relative', margin: { t: 20, r: 20, b: 45, l: 55 },
    xaxis: { title: 'Net Positions Gained (+) / Lost (-)', gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: 1.07, x: 0 }
  }, { responsive: true });

  const bestClimber = sortedGain[sortedGain.length - 1];
  document.getElementById('rf-gain-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Decomposes total race climb into standing-start Lap 1 launches vs. Lap 2+ overtakes. Biggest climber: <strong>${bestClimber.Driver}</strong> (${bestClimber.Total_PositionsGained >= 0 ? '+' : ''}${bestClimber.Total_PositionsGained} total spots: Lap 1 ${bestClimber.Lap1_PositionsGained}, Post-L1 ${bestClimber.PostLap1_PositionsGained}).`;

  // Table 1
  document.getElementById('rf-table').innerHTML = renderDataTable(
    overtakes,
    ['FinishPosition', 'Driver', 'Team', 'Total_PositionsGained', 'GridPosition', 'Lap1Position', 'Lap1_PositionsGained', 'PostLap1_PositionsGained', 'PositionChangeEvents', 'Status'],
    { FinishPosition: 'Finish', Total_PositionsGained: 'Net Pos Δ', GridPosition: 'Grid', Lap1Position: 'Lap 1 Pos', Lap1_PositionsGained: 'Lap 1 Gain', PostLap1_PositionsGained: 'Post-L1 Gain', PositionChangeEvents: 'Pos Changes' }
  );
  document.getElementById('rf-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Complete ${overtakes.length}-car Grid-to-Flag classification for ${overview.year} ${overview.event_name}, placing Finish Position and Total Net Positions Gained on the left.`;
}

// ============================================================================
// SUB-MODULE 3: TYRE STINTS, FUEL-CORRECTED OLS DEGRADATION & PACE
// ============================================================================
async function renderTyreAnalyticsModule() {
  const { year, round, session } = PitWallState;
  const host = document.getElementById('analytics-content');

  const t0 = performance.now();
  const res = await fetch(`/api/analytics/tyres/${year}/${round}?session_type=${session}`);
  const data = await res.json();
  recordLatency(performance.now() - t0);

  const stints = data.stints || [];
  const paceDist = data.pace_dist || [];
  const compSum = data.compound_summary || [];
  const cleanLaps = data.clean_laps || [];
  const notes = data.notes || {};

  host.innerHTML = `
    ${buildConceptHeader(notes.f1_context, notes.ds_methodology, notes.key_insight)}

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🛞 Driver Tyre Compound Stint Timeline (Exact StartLap → EndLap Strategy History)</h3>
      <div id="ty-gantt-chart" class="w-full h-[520px]"></div>
      <div id="ty-gantt-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📦 Compound-Aware Clean Pace Distribution (Sorted Left-to-Right by Median Pace Leader)</h3>
      <div id="ty-box-chart" class="w-full h-[460px]"></div>
      <div id="ty-box-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📉 Fuel-Corrected Lap Time vs. Tyre Age with Ordinary Least Squares (OLS) Degradation Slopes (Top 5 Pace Leaders)</h3>
      <div id="ty-ols-chart" class="w-full h-[460px]"></div>
      <div id="ty-ols-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📊 Pirelli Compound Benchmark Summary Table</h3>
      <div id="ty-comp-table"></div>
      <div id="ty-comp-table-summary" class="table-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📋 Driver Pace &amp; Consistency Table: Non-Parametric (Median/IQR) vs. Parametric (Mean/StdDev)</h3>
      <div id="ty-pace-table"></div>
      <div id="ty-pace-table-summary" class="table-summary"></div>
    </div>
  `;

  // Chart 1: Horizontal Stint Gantt Chart
  const shownComp = new Set();
  const ganttTraces = stints.map(r => {
    const comp = String(r.Compound || 'UNKNOWN').toUpperCase();
    const showLeg = !shownComp.has(comp);
    shownComp.add(comp);
    return {
      y: [r.Driver], x: [r.StintSpanLaps], base: [r.StartLap - 1],
      orientation: 'h', type: 'bar', name: comp, legendgroup: comp, showlegend: showLeg,
      marker: { color: TYRE_COLORS[comp] || '#94a3b8', line: { color: '#0b0f19', width: 1.2 } },
      hovertemplate: `<b>${r.Driver}</b> Stint ${r.Stint} (${comp})<br>Laps ${r.StartLap}–${r.EndLap} (${r.StintSpanLaps} laps)<br>Median Pace: ${r.MedianRawPace_Sec}s | Deg: ${r.TyreDegSlope_SecPerLap} s/lap<extra></extra>`
    };
  });
  Plotly.newPlot('ty-gantt-chart', ganttTraces, {
    barmode: 'overlay', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 }, margin: { t: 25, r: 20, b: 45, l: 55 },
    xaxis: { title: 'Lap Number', gridcolor: '#1e293b' },
    yaxis: { autorange: 'reversed' },
    legend: { orientation: 'h', y: 1.07, x: 0 }
  }, { responsive: true });
  document.getElementById('ty-gantt-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Maps every driver's exact tyre compound stints across the race distance. Vertical alignments of color switches expose pit-stop undercut windows and weather crossover waves.`;

  // Chart 2: Compound-Aware Pace Boxplot
  const drvOrder = paceDist.map(r => r.Driver);
  const compoundsUsed = [...new Set(cleanLaps.map(r => String(r.Compound || 'UNKNOWN').toUpperCase()))];
  const boxTraces = compoundsUsed.map(comp => {
    const sub = cleanLaps.filter(r => String(r.Compound || 'UNKNOWN').toUpperCase() === comp);
    return {
      x: sub.map(r => r.Driver), y: sub.map(r => r.LapTime_Sec),
      name: comp, type: 'box', boxpoints: 'outliers',
      marker: { color: TYRE_COLORS[comp] || '#94a3b8' }
    };
  });
  Plotly.newPlot('ty-box-chart', boxTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 }, boxmode: 'group', margin: { t: 20, r: 20, b: 45, l: 60 },
    xaxis: { categoryorder: 'array', categoryarray: drvOrder },
    yaxis: { title: 'Clean Lap Time (s)', gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: 1.08, x: 0 }
  }, { responsive: true });
  const leaderPace = paceDist[0];
  document.getElementById('ty-box-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Non-parametric Median &amp; IQR box distributions sorted fastest-to-slowest. <strong>${leaderPace ? leaderPace.Driver : 'Leader'}</strong> led clean median pace (${leaderPace ? leaderPace.MedianPace_Sec : '--'}s, IQR ±${leaderPace ? leaderPace.IQR_Consistency_Sec : '--'}s).`;

  // Chart 3: Fuel-Corrected OLS Degradation Scatter + Trendlines (Top 5 Pace Leaders)
  const top5Pace = drvOrder.slice(0, 5);
  const olsPalette = ['#00F5D4', '#FF2A85', '#FFE600', '#38BDF8', '#A855F7'];
  const olsTraces = [];
  top5Pace.forEach((drv, idx) => {
    const pts = cleanLaps.filter(r => r.Driver === drv && r.TyreLife != null && r.FuelCorrectedLap_Sec != null);
    if (!pts.length) return;
    const col = olsPalette[idx % olsPalette.length];
    olsTraces.push({
      x: pts.map(r => r.TyreLife), y: pts.map(r => r.FuelCorrectedLap_Sec),
      name: drv, mode: 'markers', type: 'scatter', marker: { size: 6, color: col, opacity: 0.65 }
    });
    const fit = fitOlsLine(pts.map(r => r.TyreLife), pts.map(r => r.FuelCorrectedLap_Sec));
    if (fit) {
      olsTraces.push({
        x: fit.x, y: fit.y, name: `${drv} OLS (${fit.slope >= 0 ? '+' : ''}${fit.slope.toFixed(3)} s/lap)`,
        mode: 'lines', type: 'scatter', line: { color: col, width: 2.5 }
      });
    }
  });
  Plotly.newPlot('ty-ols-chart', olsTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 }, margin: { t: 20, r: 20, b: 45, l: 65 },
    xaxis: { title: 'Tyre Age (Laps on Current Set)', gridcolor: '#1e293b' },
    yaxis: { title: 'Fuel-Corrected Lap Time (s)', gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: 1.12, x: 0 }
  }, { responsive: true });
  document.getElementById('ty-ols-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Removes ~0.055 s/lap fuel burn-off to isolate pure rubber degradation. Steeper upward OLS lines show thermal drop-off; flat/negative lines show superior tyre conservation or a drying track.`;

  // Tables 1 & 2
  document.getElementById('ty-comp-table').innerHTML = renderDataTable(compSum, null, null);
  document.getElementById('ty-comp-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Aggregates stint count, median clean pace, and OLS degradation slope (s/lap) across all ${compSum.length} Pirelli compounds raced.`;

  document.getElementById('ty-pace-table').innerHTML = renderDataTable(
    paceDist,
    ['Driver', 'Team', 'MedianPace_Sec', 'GapToPaceLeader_Sec', 'IQR_Consistency_Sec', 'BestLap_Sec', 'FuelCorrectedMedian_Sec', 'MeanPace_Sec', 'StdDev_Sec', 'CleanLaps', 'CompoundsUsed'],
    { MedianPace_Sec: 'Median Pace (s)', GapToPaceLeader_Sec: 'Gap (s)', IQR_Consistency_Sec: 'IQR (±s)', BestLap_Sec: 'Best Lap (s)', FuelCorrectedMedian_Sec: 'Fuel-Corr Med', MeanPace_Sec: 'Mean (s)', StdDev_Sec: 'StdDev (s)' }
  );
  document.getElementById('ty-pace-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Ranks all ${paceDist.length} drivers by non-parametric Median Clean Pace &amp; IQR Consistency alongside parametric Mean &amp; StdDev.`;
}

// ============================================================================
// SUB-MODULE 4: WEATHER CROSSOVERS, SC/VSC IMPACT & DNF/DNS RELIABILITY
// ============================================================================
async function renderWeatherIncidentsModule() {
  const { year, round, session } = PitWallState;
  const host = document.getElementById('analytics-content');

  const t0 = performance.now();
  const res = await fetch(`/api/analytics/weather-incidents/${year}/${round}?session_type=${session}`);
  const data = await res.json();
  recordLatency(performance.now() - t0);

  const weather = data.weather_timeline || [];
  const attr = data.attrition || [];
  const pitDec = data.pit_decisions || [];
  const neuts = data.neutralizations || [];
  const dnf = data.dnf || [];
  const notes = data.notes || {};

  host.innerHTML = `
    ${buildConceptHeader(notes.f1_context, notes.ds_methodology, notes.key_insight)}

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🌡️ Live Session Weather Evolution (Track Temp °C, Air Temp °C &amp; Relative Humidity %)</h3>
      <div id="wi-weather-chart" class="w-full h-[380px]"></div>
      <div id="wi-weather-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🛠️ Session Classification &amp; Attrition Breakdown (Lead Lap vs. Lapped vs. DNF / DNS)</h3>
      <div id="wi-attr-chart" class="w-full h-[380px]"></div>
      <div id="wi-attr-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🔄 Weather Crossover &amp; Tyre Pit Stop Evaluator: <em>'Was the Pit Decision Worth It?'</em></h3>
      <div id="wi-pit-table"></div>
      <div id="wi-pit-table-summary" class="table-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🚩 Safety Car (SC), Virtual Safety Car (VSC) &amp; Red Flag Episode Tracker</h3>
      <div id="wi-neut-table"></div>
      <div id="wi-neut-table-summary" class="table-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📋 Complete Driver Classification, Laps Completed &amp; DNF/DNS Attribution Table</h3>
      <div id="wi-dnf-table"></div>
      <div id="wi-dnf-table-summary" class="table-summary"></div>
    </div>
  `;

  // Chart 1: Dual-Axis Weather Timeline
  if (weather.length) {
    Plotly.newPlot('wi-weather-chart', [
      { x: weather.map(r => r.SessionMinute), y: weather.map(r => r.TrackTemp), name: 'Track Temp (°C)', line: { color: '#ef4444', width: 2.5 } },
      { x: weather.map(r => r.SessionMinute), y: weather.map(r => r.AirTemp), name: 'Air Temp (°C)', line: { color: '#f59e0b', width: 2 } },
      { x: weather.map(r => r.SessionMinute), y: weather.map(r => r.Humidity), name: 'Humidity (%)', yaxis: 'y2', line: { color: '#38bdf8', width: 1.8, dash: 'dot' } }
    ], {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#cbd5e1', size: 11 }, margin: { t: 20, r: 60, b: 45, l: 55 },
      xaxis: { title: 'Session Elapsed Minutes', gridcolor: '#1e293b' },
      yaxis: { title: 'Temperature (°C)', gridcolor: '#1e293b' },
      yaxis2: { title: 'Humidity (%)', overlaying: 'y', side: 'right' },
      legend: { orientation: 'h', y: 1.1, x: 0 }
    }, { responsive: true });
    document.getElementById('wi-weather-summary').innerHTML =
      `📊 <strong>Graph Summary:</strong> Tracks minute-by-minute asphalt Track Temp (red), Air Temp (yellow), and Relative Humidity (blue dotted). Dropping Track Temp paired with rising Humidity triggers wet-tyre crossover windows.`;
  } else {
    document.getElementById('wi-weather-chart').innerHTML = '<div class="text-xs text-slate-400 p-4">Minute-by-minute weather telemetry not recorded for this session.</div>';
  }

  // Chart 2: Attrition Donut Chart
  if (attr.length) {
    Plotly.newPlot('wi-attr-chart', [{
      labels: attr.map(r => r.ReliabilityCategory),
      values: attr.map(r => r.DriverCount),
      type: 'pie', hole: 0.45,
      marker: { colors: ['#10b981', '#38bdf8', '#f59e0b', '#ef4444', '#a855f7'] }
    }], {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#cbd5e1', size: 12 }, margin: { t: 20, r: 20, b: 20, l: 20 }
    }, { responsive: true });
    document.getElementById('wi-attr-summary').innerHTML =
      `📊 <strong>Graph Summary:</strong> Breakdown of the starting grid across Lead-Lap Finishers, Lapped Finishers, Mechanical/Crash DNFs, and DNS/DSQ exclusions.`;
  }

  // Tables 1, 2, 3
  document.getElementById('wi-pit-table').innerHTML = renderDataTable(
    pitDec,
    ['Driver', 'Team', 'PitLap', 'DecisionVerdict', 'NetPositions_5Laps', 'PaceDelta_AfterPit_Sec', 'Transition', 'PittedUnder_SC_VSC', 'Pos_BeforePit', 'Pos_5LapsAfter'],
    { PitLap: 'Pit Lap', DecisionVerdict: 'Engineering Verdict', NetPositions_5Laps: 'Net Pos Δ (5L)', PaceDelta_AfterPit_Sec: 'Pace Δ (s/lap)', Transition: 'Tyre Switch', PittedUnder_SC_VSC: 'SC/VSC Stop?', Pos_BeforePit: 'Pos Before', Pos_5LapsAfter: 'Pos L+5' }
  );
  document.getElementById('wi-pit-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Counterfactual [t-2 vs. t+5] audit across ${pitDec.length} pit-stop transitions, highlighting Masterstroke undercuts vs. Costly stops on the left.`;

  document.getElementById('wi-neut-table').innerHTML = neuts.length
    ? renderDataTable(neuts, ['Episode', 'IncidentType', 'StartLap', 'EndLap', 'Duration_Laps', 'BiggestPositionGainers', 'BiggestPositionLosers', 'DriversWhoPitted_CheapStop'], { IncidentType: 'Type', StartLap: 'Start', EndLap: 'End', Duration_Laps: 'Laps', BiggestPositionGainers: 'Top Gainers', BiggestPositionLosers: 'Top Losers', DriversWhoPitted_CheapStop: 'Cheap Stop Takers' })
    : '<div class="p-3 rounded bg-emerald-950/60 border border-emerald-800 text-xs text-emerald-300">Green-flag session: 0 Safety Car, VSC, or Red Flag episodes recorded.</div>';
  document.getElementById('wi-neut-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Deduplicates consecutive FIA neutralization laps into distinct SC, VSC, and Red Flag episodes, pinpointing who gained ~11s via a 'Cheap Pit Stop'.`;

  document.getElementById('wi-dnf-table').innerHTML = renderDataTable(
    dnf,
    ['FinishPosition', 'Driver', 'Team', 'ReliabilityCategory', 'OfficialStatus', 'GridPosition', 'LapsCompleted', 'RaceCompletion_Pct', 'Points'],
    { FinishPosition: 'Finish', ReliabilityCategory: 'Category', OfficialStatus: 'FIA Status', GridPosition: 'Grid', LapsCompleted: 'Laps', RaceCompletion_Pct: 'Distance %' }
  );
  document.getElementById('wi-dnf-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Complete ${dnf.length}-car classification detailing mechanical retirements, collisions, lapped finishers, and exact race distance completion %.`;
}

// ============================================================================
// SUB-MODULE 5: CONSTRUCTOR AERO QUADRANT & PIT-WALL STRATEGY AUDIT
// ============================================================================
async function renderConstructorsModule() {
  const { year, round, session } = PitWallState;
  const host = document.getElementById('analytics-content');

  const t0 = performance.now();
  const res = await fetch(`/api/analytics/constructors/${year}/${round}?session_type=${session}`);
  const data = await res.json();
  recordLatency(performance.now() - t0);

  const teams = data.event_teams || [];
  const audit = data.strategy_audit || [];
  const wcc = data.season_wcc || [];
  const notes = data.notes || {};

  host.innerHTML = `
    ${buildConceptHeader(notes.f1_context, notes.ds_methodology, notes.key_insight)}

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">✈️ Constructor Aero Quadrant: Peak Speed Trap (km/h) vs. Median Clean Pace (s)</h3>
      <div id="con-aero-chart" class="w-full h-[500px]"></div>
      <div id="con-aero-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🧠 Constructor Pit-Wall Strategy Execution Success Rate (%)</h3>
      <div id="con-strat-chart" class="w-full h-[420px]"></div>
      <div id="con-strat-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🏎️ Event Constructor Pace, Tyre Degradation &amp; Aero Setup Table</h3>
      <div id="con-ev-table"></div>
      <div id="con-ev-table-summary" class="table-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📋 Constructor Pit-Wall Strategy Execution Audit Table</h3>
      <div id="con-audit-table"></div>
      <div id="con-audit-table-summary" class="table-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🏆 ${year} World Constructors' Championship (WCC), Reliability % &amp; Tyre Wear Table</h3>
      <div id="con-wcc-table"></div>
      <div id="con-wcc-table-summary" class="table-summary"></div>
    </div>
  `;

  // Chart 1: Constructor Aero Quadrant Scatter Plot
  const validTeams = teams.filter(r => r.Peak_Speed_kmh != null && r.Median_Pace_Sec != null);
  if (validTeams.length) {
    Plotly.newPlot('con-aero-chart', [{
      x: validTeams.map(r => r.Peak_Speed_kmh),
      y: validTeams.map(r => r.Median_Pace_Sec),
      text: validTeams.map(r => r.Constructor),
      customdata: validTeams.map(r => r.Aero_Setup_Profile),
      mode: 'markers+text', textposition: 'top center',
      marker: {
        size: 16,
        color: validTeams.map(r => TEAM_COLORS[r.Constructor] || '#38bdf8'),
        line: { color: '#ffffff', width: 1 }
      },
      hovertemplate: '<b>%{text}</b><br>Peak Speed: %{x} km/h<br>Median Pace: %{y}s<br>Profile: %{customdata}<extra></extra>'
    }], {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#cbd5e1', size: 11 }, margin: { t: 25, r: 30, b: 50, l: 65 },
      xaxis: { title: 'Peak Straight-Line Speed Trap (km/h — Further Right = Lower Drag)', gridcolor: '#1e293b' },
      yaxis: { title: 'Median Clean Pace (s — Higher on Chart = Faster)', autorange: 'reversed', gridcolor: '#1e293b' }
    }, { responsive: true });
    document.getElementById('con-aero-summary').innerHTML =
      `📊 <strong>Graph Summary:</strong> Partitions constructors into 4 aero regimes: Top-Right = High Efficiency (fast on straights &amp; corners), Top-Left = High Downforce, Bottom-Right = Low-Drag Trim.`;
  }

  // Chart 2: Pit-Wall Strategy Execution Bar Chart
  if (audit.length) {
    const sortedAudit = [...audit].sort((a, b) => (a.Strategy_Success_Rate_Pct || 0) - (b.Strategy_Success_Rate_Pct || 0));
    Plotly.newPlot('con-strat-chart', [{
      y: sortedAudit.map(r => r.Team),
      x: sortedAudit.map(r => r.Strategy_Success_Rate_Pct),
      type: 'bar', orientation: 'h',
      text: sortedAudit.map(r => `${r.Strategy_Success_Rate_Pct}%`), textposition: 'outside',
      marker: { color: sortedAudit.map(r => TEAM_COLORS[r.Team] || '#10b981') }
    }], {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#cbd5e1', size: 11 }, margin: { t: 20, r: 45, b: 45, l: 120 },
      xaxis: { title: 'Strategy Success Rate (%)', range: [0, 110], gridcolor: '#1e293b' }
    }, { responsive: true });
    document.getElementById('con-strat-summary').innerHTML =
      `📊 <strong>Graph Summary:</strong> Ranks each team's pit wall by the percentage of stops that gained position or unlocked faster pace without being undercut.`;
  } else {
    document.getElementById('con-strat-chart').innerHTML = '<div class="text-xs text-slate-400 p-4">No mid-session pit-wall strategy stops recorded in this session.</div>';
  }

  // Tables 1, 2, 3
  document.getElementById('con-ev-table').innerHTML = renderDataTable(
    teams,
    ['Constructor', 'Drivers_Classification', 'Points_Scored', 'Median_Pace_Sec', 'Gap_To_Fastest_Team_Sec', 'Aero_Setup_Profile', 'Peak_Speed_kmh', 'Tyre_Deg_Slope', 'Consistency_IQR_Sec', 'Net_Pos_Gained'],
    { Drivers_Classification: 'Drivers & Finish', Points_Scored: 'Pts', Median_Pace_Sec: 'Median Pace (s)', Gap_To_Fastest_Team_Sec: 'Gap (s)', Aero_Setup_Profile: 'Aero Profile', Peak_Speed_kmh: 'Peak Speed', Tyre_Deg_Slope: 'Deg (s/lap)', Consistency_IQR_Sec: 'IQR (±s)', Net_Pos_Gained: 'Net Pos Δ' }
  );
  document.getElementById('con-ev-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Event constructor benchmarks ordered by Median Clean Pace, pairing Aero Setup Profile and Tyre Degradation Slope by chassis.`;

  document.getElementById('con-audit-table').innerHTML = renderDataTable(
    audit,
    ['Team', 'Strategy_Success_Rate_Pct', 'Masterstroke_Calls', 'WorthIt_Calls', 'Costly_Undercut_Calls', 'Cheap_SC_VSC_Stops', 'Net_PitCycle_Positions', 'Total_Pit_Calls'],
    { Team: 'Constructor', Strategy_Success_Rate_Pct: 'Success (%)', Masterstroke_Calls: 'Masterstrokes', WorthIt_Calls: 'Positive Stops', Costly_Undercut_Calls: 'Costly Calls', Cheap_SC_VSC_Stops: 'Cheap SC/VSC', Net_PitCycle_Positions: 'Net Pit Pos Δ', Total_Pit_Calls: 'Total Stops' }
  );
  document.getElementById('con-audit-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Audits constructor pit-wall execution across Masterstroke calls, cheap SC/VSC stops, and net pit-cycle positions gained.`;

  document.getElementById('con-wcc-table').innerHTML = renderDataTable(wcc, null, null);
  document.getElementById('con-wcc-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Full ${year} World Constructors' Championship standings combined with Finish Reliability (%) and season tyre degradation rates.`;
}

// ============================================================================
// SUB-MODULE 6: 6-AXIS DRIVER DNA RADAR, OPTIMAL SECTORS & TEAMMATES
// ============================================================================
async function renderDriverDnaModule() {
  const { year, round, session } = PitWallState;
  const host = document.getElementById('analytics-content');

  const t0 = performance.now();
  const res = await fetch(`/api/analytics/driver-dna/${year}/${round}?session_type=${session}`);
  const data = await res.json();
  recordLatency(performance.now() - t0);

  const dna = data.dna || [];
  const sectors = data.sectors || [];
  const h2h = data.h2h || [];
  const notes = data.notes || {};

  host.innerHTML = `
    ${buildConceptHeader(notes.f1_context, notes.ds_methodology, notes.key_insight)}

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🕸️ 6-Axis Driver DNA Radar Polygon (Top 5 Overall DNA Leaders on 45–99 Percentile Scale)</h3>
      <div id="dna-radar-chart" class="w-full h-[500px]"></div>
      <div id="dna-radar-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🧩 Sector Execution Gap (s): Actual Best Lap minus Theoretical Optimal Lap (Best S1 + S2 + S3)</h3>
      <div id="dna-sector-chart" class="w-full h-[480px]"></div>
      <div id="dna-sector-summary" class="graph-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📋 Complete 6-Axis Driver DNA Ratings Table</h3>
      <div id="dna-table"></div>
      <div id="dna-table-summary" class="table-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">⚡ Theoretical Optimal Lap (Best S1 + Best S2 + Best S3) vs. Actual Best Lap Table</h3>
      <div id="dna-sec-table"></div>
      <div id="dna-sec-table-summary" class="table-summary"></div>
    </div>

    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🥊 ${year} Season Teammate Head-to-Head Comparison Table (Identical Car Benchmark)</h3>
      <div id="dna-h2h-table"></div>
      <div id="dna-h2h-table-summary" class="table-summary"></div>
    </div>
  `;

  // Chart 1: 6-Axis Radar Chart (Top 5 DNA Leaders)
  const top5Dna = dna.slice(0, 5);
  const categories = ['Quali Pace', 'Race Pace', 'Tyre Mgmt', 'Overtake Craft', 'Top Speed', 'Consistency', 'Quali Pace'];
  const radarTraces = top5Dna.map(r => ({
    type: 'scatterpolar',
    r: [r.Quali_Pace_Score, r.Race_Pace_Score, r.Tyre_Mgmt_Score, r.Overtake_Craft_Score, r.Top_Speed_Score, r.Consistency_Score, r.Quali_Pace_Score],
    theta: categories,
    fill: 'toself',
    opacity: 0.55,
    name: `${r.Driver} (DNA: ${r.Overall_DNA_Rating})`
  }));

  Plotly.newPlot('dna-radar-chart', radarTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 12 },
    polar: {
      bgcolor: 'rgba(15,23,42,0.7)',
      radialaxis: { visible: true, range: [40, 100], gridcolor: '#334155' },
      angularaxis: { gridcolor: '#334155' }
    },
    margin: { t: 40, r: 90, b: 40, l: 90 },
    legend: { orientation: 'h', y: 1.09, x: 0 }
  }, { responsive: true });
  document.getElementById('dna-radar-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Overlays 6 percentile-normalized traits on a uniform [45–99] scale. <strong>${top5Dna[0] ? top5Dna[0].Driver : 'Leader'}</strong> leads the field with an Overall DNA Rating of <strong>${top5Dna[0] ? top5Dna[0].Overall_DNA_Rating : '--'}/100</strong>.`;

  // Chart 2: Sector Execution Gap Horizontal Bar Chart
  const sortedSec = [...sectors].sort((a, b) => (a.Execution_Gap_Sec || 0) - (b.Execution_Gap_Sec || 0));
  Plotly.newPlot('dna-sector-chart', [{
    y: sortedSec.map(r => r.Driver),
    x: sortedSec.map(r => r.Execution_Gap_Sec),
    type: 'bar', orientation: 'h',
    text: sortedSec.map(r => `+${r.Execution_Gap_Sec}s`), textposition: 'outside',
    marker: { color: sortedSec.map(r => TEAM_COLORS[r.Team] || '#38bdf8') }
  }], {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 }, margin: { t: 20, r: 50, b: 45, l: 55 },
    xaxis: { title: 'Untapped Sector Execution Gap (s — Closer to 0.000s = Cleaner Lap Hook-Up)', gridcolor: '#1e293b' },
    yaxis: { autorange: 'reversed' }
  }, { responsive: true });
  const cleanest = sortedSec[0];
  document.getElementById('dna-sector-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Measures how close each driver came to linking their personal best S1, S2, and S3 on one lap. Cleanest execution: <strong>${cleanest ? cleanest.Driver : '--'}</strong> (only <strong>+${cleanest ? cleanest.Execution_Gap_Sec : '--'}s</strong> off theoretical optimal).`;

  // Tables 1, 2, 3
  document.getElementById('dna-table').innerHTML = renderDataTable(
    dna,
    ['Driver', 'Team', 'Overall_DNA_Rating', 'Race_Pace_Score', 'Quali_Pace_Score', 'Tyre_Mgmt_Score', 'Overtake_Craft_Score', 'Consistency_Score', 'Top_Speed_Score', 'TotalPoints'],
    { Overall_DNA_Rating: 'Overall DNA', Race_Pace_Score: 'Race Pace', Quali_Pace_Score: 'Quali', Tyre_Mgmt_Score: 'Tyre Mgmt', Overtake_Craft_Score: 'Overtake', Consistency_Score: 'Consistency', Top_Speed_Score: 'Top Speed', TotalPoints: 'Pts' }
  );
  document.getElementById('dna-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Composite 6-axis percentile ratings across ${dna.length} active drivers, distinguishing Saturday qualifying pace from Sunday tyre management and overtaking.`;

  document.getElementById('dna-sec-table').innerHTML = renderDataTable(
    sectors,
    ['Driver', 'Team', 'Actual_Best_Lap', 'Theoretical_Best_Lap', 'Execution_Gap_Sec', 'Best_S1', 'Best_S2', 'Best_S3', 'Peak_SpeedTrap_kmh'],
    { Actual_Best_Lap: 'Actual Best (s)', Theoretical_Best_Lap: 'Theoretical Opt (s)', Execution_Gap_Sec: 'Execution Gap (s)', Best_S1: 'Best S1', Best_S2: 'Best S2', Best_S3: 'Best S3', Peak_SpeedTrap_kmh: 'Speed Trap (km/h)' }
  );
  document.getElementById('dna-sec-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Compares Actual Best Lap against Theoretical Optimal Lap (Best S1 + S2 + S3) with untapped time highlighted in column 5.`;

  document.getElementById('dna-h2h-table').innerHTML = renderDataTable(h2h, null, null);
  document.getElementById('dna-h2h-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Season ${year} teammate head-to-head control benchmark comparing drivers inside identical constructor machinery.`;
}