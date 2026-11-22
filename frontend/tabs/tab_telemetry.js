/**
 * MODULE 1: UNIVERSAL 2024-2026 TELEMETRY LAB (frontend/tabs/tab_telemetry.js)
 * Connects to: GET /api/telemetry/{year}/{round_num}?drivers=...&session_type=...
 */

window.loadTelemetryWorkspace = async function () {
  const container = document.getElementById('workspace-telemetry');
  const ov = PitWallState.overview;
  if (!ov) return;

  // Build interactive Driver Selector Checkboxes/Pills from the race's official entry list
  const rosterHtml = (ov.roster || []).map(r => {
    const isSelected = PitWallState.selectedDrivers.includes(r.Driver);
    const statusBadge = !r.IsClassified
      ? `<span class="text-[10px] text-amber-400 font-mono">(${r.Status})</span>`
      : `<span class="text-[10px] text-slate-400 font-mono">P${r.FinishPosition ?? '-'}</span>`;
    const activeStyle = isSelected
      ? 'bg-red-600/90 border-red-400 text-white font-bold'
      : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800';

    return `
      <button onclick="toggleTelemetryDriver('${r.Driver}')"
        class="drv-pill px-2.5 py-1 rounded-lg border text-xs flex items-center gap-1.5 ${activeStyle}"
        title="${r.FullName} (${r.Team}) — Status: ${r.Status}">
        <span>${r.Driver}</span>
        ${statusBadge}
      </button>
    `;
  }).join('');

  // Render workspace skeleton once if not yet initialized
  container.innerHTML = `
    <div id="tel-concept-banner"></div>

    <!-- Driver Selection Bar (Click any driver from this race to compare) -->
    <div class="pit-card p-4 space-y-3">
      <div class="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 class="text-sm font-bold text-white">Select Drivers to Compare (${ov.year} ${ov.event_name} — ${ov.location})</h2>
          <p class="text-xs text-slate-400">Click any driver below to add/remove them. DNF/DNS statuses from official FIA classification are shown next to each driver code.</p>
        </div>
        <span id="tel-tier-badge" class="text-xs font-mono px-3 py-1 rounded bg-slate-900 border border-slate-700 text-sky-300">Loading telemetry...</span>
      </div>
      <div class="flex flex-wrap gap-1.5">${rosterHtml}</div>
      <!-- DNF / DNS / Non-Participation Alert Box -->
      <div id="tel-diagnostics-box" class="space-y-1.5"></div>
    </div>

    <!-- Chart 1: Full-Width 5-Panel Synchronized Telemetry Stack -->
    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📡 Synchronized 5-Channel Telemetry Stack (Speed, Continuous Δt, Throttle %, Brake % &amp; Gear)</h3>
      <div id="tel-stack-chart" class="w-full h-[780px]"></div>
      <div id="tel-stack-summary" class="graph-summary"></div>
    </div>

    <!-- Chart 2: Full-Width Corner-by-Corner Minimum Apex Speeds -->
    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📐 Corner-by-Corner Minimum Apex Speed Comparison (km/h)</h3>
      <div id="tel-corner-chart" class="w-full h-[380px]"></div>
      <div id="tel-corner-summary" class="graph-summary"></div>
    </div>

    <!-- Chart 3: Full-Width 2D GPS Circuit Map & Track Dominance + Explicit Track Verification Footer -->
    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">🗺️ 2D GPS Circuit Map &amp; Mini-Sector Speed Dominance</h3>
      <div id="tel-gps-chart" class="w-full h-[500px]"></div>
      <!-- Explicit Track & Race Verification Banner Directly Under GPS Graph -->
      <div id="tel-gps-track-footer" class="p-3 rounded-lg bg-slate-900 border border-slate-700 text-center text-xs font-semibold text-emerald-300"></div>
      <div id="tel-gps-summary" class="graph-summary"></div>
    </div>

    <!-- Table 1: Full-Width Telemetry DNA Comparison Table + Table Summary -->
    <div class="pit-card p-5 space-y-3">
      <h3 class="text-base font-bold text-white">📋 Full-Lap Telemetry DNA Metrics &amp; Classification Status Table</h3>
      <div id="tel-dna-table" class="overflow-x-auto text-xs"></div>
      <div id="tel-dna-table-summary" class="table-summary"></div>
    </div>
  `;

  await fetchAndRenderTelemetry();
};

window.toggleTelemetryDriver = async function (drv) {
  const idx = PitWallState.selectedDrivers.indexOf(drv);
  if (idx >= 0) {
    if (PitWallState.selectedDrivers.length > 1) {
      PitWallState.selectedDrivers.splice(idx, 1);
    }
  } else {
    if (PitWallState.selectedDrivers.length >= 5) {
      PitWallState.selectedDrivers.shift();
    }
    PitWallState.selectedDrivers.push(drv);
  }
  await window.loadTelemetryWorkspace();
};

async function fetchAndRenderTelemetry() {
  const { year, round, session, selectedDrivers } = PitWallState;
  const driversCsv = selectedDrivers.join(',');

  const t0 = performance.now();
  const res = await fetch(`/api/telemetry/${year}/${round}?drivers=${encodeURIComponent(driversCsv)}&session_type=${session}`);
  const data = await res.json();
  recordLatency(performance.now() - t0);

  document.getElementById('tel-tier-badge').textContent = data.data_tier || 'Telemetry Ready';

  // 1. Render DNF / DNS / Retirement Diagnostic Alerts
  const diagBox = document.getElementById('tel-diagnostics-box');
  const diags = data.diagnostics || [];
  if (diags.length > 0) {
    diagBox.innerHTML = diags.map(d => {
      const style = d.Severity === 'error'
        ? 'bg-red-950/80 border-red-700 text-red-200'
        : 'bg-amber-950/80 border-amber-700 text-amber-200';
      const icon = d.Severity === 'error' ? '🚫 DNS / DNF Telemetry Exclusion:' : '⚠️ Mid-Session Retirement Notice:';
      return `<div class="p-2.5 rounded-lg border text-xs ${style}"><strong>${icon}</strong> ${d.Message}</div>`;
    }).join('');
  } else {
    diagBox.innerHTML = '';
  }

  const rows = data.telemetry || [];
  const dnaRows = data.driver_dna || [];
  const corners = data.corners || [];
  const cornerSpeeds = data.corner_speeds || [];
  const dominance = data.dominance || [];

  if (!rows.length || !dnaRows.length) {
    document.getElementById('tel-stack-chart').innerHTML =
      `<div class="flex items-center justify-center h-full text-sm text-amber-300">No timed flying laps available for the selected driver(s). Select another driver pill above.</div>`;
    return;
  }

  const refDrv = dnaRows[0];
  const slowestDrv = dnaRows[dnaRows.length - 1];
  const topSpdDrv = [...dnaRows].sort((a, b) => b.TopSpeed_kmh - a.TopSpeed_kmh)[0];

  // 2. Render 3-Column Concept & Terminology Header
  document.getElementById('tel-concept-banner').innerHTML = buildConceptHeader(
    `Aligns high-frequency car telemetry by physical <strong>Track Distance (0m to ${data.track_length_m}m)</strong> rather than elapsed time so braking zones, corner apex speeds, and straight-line speed traps line up meter-for-meter across drivers.`,
    `<strong>Continuous Time Delta (Δt):</strong> 1D spatial interpolation (<code>np.interp</code>) subtracting Reference Driver (${refDrv.Driver}) time from comparison drivers. Upward slope = losing time; downward slope = gaining time. <strong>Apex Min Speed:</strong> Lowest speed within ±45m of each turn.`,
    `In <strong>${data.year} ${data.event_name} (${data.location})</strong>, <strong>${refDrv.FullName} (${refDrv.Driver})</strong> set the reference lap of <strong>${refDrv.LapTime_Sec}s</strong> (${refDrv.Compound}), while <strong>${topSpdDrv.Driver}</strong> achieved the highest speed trap at <strong>${topSpdDrv.TopSpeed_kmh} km/h</strong>.`
  );

  // Group telemetry rows by driver
  const byDriver = {};
  rows.forEach(r => {
    if (!byDriver[r.Driver]) byDriver[r.Driver] = [];
    byDriver[r.Driver].push(r);
  });

  const palette = ['#ef4444', '#38bdf8', '#10b981', '#f59e0b', '#a855f7'];
  const drvList = data.ordered_drivers || Object.keys(byDriver);
  const drvColorMap = {};
  drvList.forEach((d, idx) => { drvColorMap[d] = palette[idx % palette.length]; });

  // 3. Render 5-Panel Synchronized Telemetry Stack
  const stackTraces = [];
  drvList.forEach(drv => {
    const dRows = byDriver[drv] || [];
    const xDist = dRows.map(r => r.Distance);
    const col = drvColorMap[drv];
    const label = `${drv} (L${dRows[0].LapNumber} • ${dRows[0].LapTime_Sec}s)`;

    stackTraces.push({ x: xDist, y: dRows.map(r => r.Speed), name: label, legendgroup: drv, line: { color: col, width: 2 }, xaxis: 'x', yaxis: 'y1' });
    stackTraces.push({ x: xDist, y: dRows.map(r => r.Delta_Sec), name: `${drv} Δt (s)`, legendgroup: drv, showlegend: false, line: { color: col, width: 2 }, xaxis: 'x', yaxis: 'y2' });
    stackTraces.push({ x: xDist, y: dRows.map(r => r.Throttle), name: `${drv} Throttle %`, legendgroup: drv, showlegend: false, line: { color: col, width: 1.8 }, xaxis: 'x', yaxis: 'y3' });
    stackTraces.push({ x: xDist, y: dRows.map(r => r.Brake_Pct), name: `${drv} Brake %`, legendgroup: drv, showlegend: false, line: { color: col, width: 1.8 }, xaxis: 'x', yaxis: 'y4' });
    stackTraces.push({ x: xDist, y: dRows.map(r => r.nGear), name: `${drv} Gear`, legendgroup: drv, showlegend: false, line: { color: col, width: 1.8 }, xaxis: 'x', yaxis: 'y5' });
  });

  const cornerLines = corners.map(c => ({
    type: 'line', x0: c.Distance, x1: c.Distance, y0: 0, y1: 1, yref: 'paper',
    line: { color: 'rgba(148,163,184,0.22)', width: 1, dash: 'dot' }
  }));

  Plotly.newPlot('tel-stack-chart', stackTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 },
    margin: { t: 25, r: 20, b: 45, l: 60 },
    grid: { rows: 5, columns: 1, pattern: 'coupled', roworder: 'top to bottom' },
    xaxis: { title: `Circuit Distance (m) — ${data.event_name} (${data.location})`, gridcolor: '#1e293b' },
    yaxis: { title: 'Speed (km/h)', domain: [0.76, 1.0], gridcolor: '#1e293b' },
    yaxis2: { title: `Δt vs ${refDrv.Driver} (s)`, domain: [0.57, 0.73], gridcolor: '#1e293b' },
    yaxis3: { title: 'Throttle (%)', domain: [0.38, 0.54], gridcolor: '#1e293b' },
    yaxis4: { title: 'Brake (%)', domain: [0.19, 0.35], gridcolor: '#1e293b' },
    yaxis5: { title: 'Gear (1-8)', domain: [0.0, 0.16], gridcolor: '#1e293b' },
    shapes: cornerLines,
    legend: { orientation: 'h', y: 1.05, x: 0 }
  }, { responsive: true });

  document.getElementById('tel-stack-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Synchronized across <strong>${data.track_length_m}m</strong> and <strong>${data.corner_count} official corners</strong> at ${data.location}. <strong>${refDrv.Driver}</strong> serves as the 0.000s baseline in Panel 2; by the checkered line, <strong>${slowestDrv.Driver}</strong> trails by <strong>+${slowestDrv.GapToRef_Sec}s</strong>.`;

  // 4. Render Corner-by-Corner Minimum Apex Speed Chart
  const cornerTraces = drvList.map(drv => ({
    x: cornerSpeeds.map(c => c.Corner),
    y: cornerSpeeds.map(c => c[drv]),
    name: `${drv} Min Apex Speed`,
    type: 'bar',
    marker: { color: drvColorMap[drv] }
  }));

  Plotly.newPlot('tel-corner-chart', cornerTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 },
    barmode: 'group',
    margin: { t: 20, r: 20, b: 45, l: 60 },
    xaxis: { title: 'Official Circuit Corner Number (T1–Tn)', gridcolor: '#1e293b' },
    yaxis: { title: 'Minimum Apex Speed (km/h)', gridcolor: '#1e293b' },
    legend: { orientation: 'h', y: 1.08, x: 0 }
  }, { responsive: true });

  const slowestCorner = [...cornerSpeeds].sort((a, b) => a[refDrv.Driver] - b[refDrv.Driver])[0];
  const fastestCorner = [...cornerSpeeds].sort((a, b) => b[refDrv.Driver] - a[refDrv.Driver])[0];
  document.getElementById('tel-corner-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Compares mechanical grip in slow hairpins vs. aerodynamic downforce in high-speed sweeps. At ${data.location}, <strong>${slowestCorner.Corner}</strong> is the slowest hairpin (${slowestCorner[refDrv.Driver]} km/h for ${refDrv.Driver}), while <strong>${fastestCorner.Corner}</strong> is the fastest high-speed turn (${fastestCorner[refDrv.Driver]} km/h).`;

  // 5. Render 2D GPS Circuit Dominance Map + Track Identity Footer Under Map
  const gpsTraces = [{
  x: dominance.map(p => p.X),
  y: dominance.map(p => p.Y),
  mode: 'lines',
  showlegend: false,
  hoverinfo: 'skip',
  line: { color: '#334155', width: 9 }
}];

drvList.forEach(drv => {
  const pts = dominance.filter(p => p.FastestDriver === drv);
  gpsTraces.push({
    x: pts.map(p => p.X),
    y: pts.map(p => p.Y),
    mode: 'markers',
    name: `${drv} Fastest Sector`,
    marker: { size: 6, color: drvColorMap[drv] }
  });
});

if (corners.length) {
  gpsTraces.push({
    x: corners.map(c => c.LabelX ?? c.X),
    y: corners.map(c => c.LabelY ?? c.Y),
    text: corners.map(c => c.CornerLabel),
    mode: 'text',
    name: 'Official Corners',
    textfont: { color: '#f8fafc', size: 11 }
  });
}

  Plotly.newPlot('tel-gps-chart', gpsTraces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 },
    margin: { t: 20, r: 20, b: 20, l: 20 },
    xaxis: { visible: false },
    yaxis: { visible: false, scaleanchor: 'x' },
    legend: { orientation: 'h', y: 1.05, x: 0 }
  }, { responsive: true });

  document.getElementById('tel-gps-track-footer').innerHTML =
    `📍 <strong>Active GPS Circuit Verification:</strong> ${data.year} ${data.event_name} — <strong>${data.location}, ${data.country}</strong> • Track Length: <strong>${data.track_length_m} m</strong> • Official Corners: <strong>${data.corner_count} Turns (T1–T${data.corner_count})</strong> • Engine: ${data.data_tier}`;

  // Calculate dominance percentage per driver
  const domCounts = {};
  dominance.forEach(p => { domCounts[p.FastestDriver] = (domCounts[p.FastestDriver] || 0) + 1; });
  const domSummaryStr = drvList.map(d => `${d}: ${(((domCounts[d] || 0) / dominance.length) * 100).toFixed(1)}% of track`).join(' | ');

  document.getElementById('tel-gps-summary').innerHTML =
    `📊 <strong>Graph Summary:</strong> Colors every mini-sector of the <strong>${data.location}</strong> circuit by the driver carrying the highest instantaneous speed. Track dominance split: <strong>${domSummaryStr}</strong>.`;

  // 6. Render Full-Lap Telemetry DNA Table + Table Summary
  const cols = Object.keys(dnaRows[0]);
  const thHtml = cols.map(c => `<th class="text-left py-2.5 px-3 text-slate-400 border-b border-slate-800">${c}</th>`).join('');
  const trHtml = dnaRows.map(r => `<tr>${cols.map(c => `<td class="py-2 px-3 border-b border-slate-800/60 font-mono">${r[c] ?? '-'}</td>`).join('')}</tr>`).join('');
  document.getElementById('tel-dna-table').innerHTML = `<table class="w-full"><thead><tr>${thHtml}</tr></thead><tbody>${trHtml}</tbody></table>`;

  document.getElementById('tel-dna-table-summary').innerHTML =
    `📋 <strong>Table Summary:</strong> Displays full-lap sensor DNA and official FIA race classification status across <strong>${dnaRows.length} drivers</strong> for the <strong>${data.year} ${data.event_name}</strong>. Full-throttle percentage ranged from <strong>${Math.min(...dnaRows.map(d => d.FullThrottle_Pct))}%</strong> to <strong>${Math.max(...dnaRows.map(d => d.FullThrottle_Pct))}%</strong>.`;
}