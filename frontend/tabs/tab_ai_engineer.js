/**
 * MODULE 4: 2024–2026 MULTI-FACTOR AI RACE ENGINEER STRATEGIST
 * File: frontend/tabs/tab_ai_engineer.js
 * Grounded on live DuckDB Weather, Engine Speed Traps, Sector Splits, Corners & Tyre Slopes
 */

window.loadEngineerWorkspace = function () {
  const container = document.getElementById('workspace-engineer');
  const ov = PitWallState.overview;
  if (!ov) return;

  container.innerHTML = `
    ${buildConceptHeader(
      `Acts as an on-demand Chief Race Strategist covering all <strong>2024, 2025, and 2026</strong> race weekends — answering corner-specific performance, engine speed-trap behavior in hot vs. rainy conditions, and pit-stop crossover calls.`,
      `<strong>Single-Pass DuckDB Grounded Inference:</strong> Queries verified DuckDB tables (Weather, Pirelli Stints, Engine Speed Traps, Sector Splits, and Corner Apexes) in <strong>&lt;25 ms</strong> and attaches exact numerical evidence to Gemini 3.8 Flash.`,
      `Every strategic answer is paired with a <strong>Grounded Statistical Evidence Panel</strong> directly above the briefing so race engineers can verify every speed trap, temperature, and degradation slope.`
    )}

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <!-- Left: Pit-Wall Radio Console -->
      <div class="lg:col-span-4 pit-card p-5 space-y-4">
        <h2 class="text-base font-bold text-white">🎙️ Pit-Wall Strategy Radio (2024–2026)</h2>

        <div class="space-y-2">
          <p class="text-xs font-semibold text-slate-300">Quick Multi-Factor Engineering Scenarios:</p>
          <button onclick="setAiPrompt('Analyze tyre degradation, corner apex speeds, and optimal pit strategy for the ${ov.year} ${ov.event_name} (${ov.location}).')" class="w-full text-left text-xs p-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200">
            "Active Race: ${ov.year} ${ov.event_name} tyres, corners &amp; pit call"
          </button>
          <button onclick="setAiPrompt('How did engine speed traps and constructor pace react across hot tracks (>=38C) vs rainy wet sessions from 2024 to 2026? Give setup strategy takeaways.')" class="w-full text-left text-xs p-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200">
            "2024–2026 Engine &amp; Pace: Hot Tracks (&gt;=38°C) vs. Rainy Days"
          </button>
          <button onclick="setAiPrompt('Compare driver sector splits, engine top speed traps, and corner performance for ${ov.year} ${ov.event_name}. Where is time won or lost?')" class="w-full text-left text-xs p-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200">
            "Driver, Engine Speed-Trap &amp; Corner Sector Breakdown (${ov.year} ${ov.location})"
          </button>
        </div>

        <div>
          <label class="block text-xs text-slate-400 mb-1">Custom Engineering / Strategy Query</label>
          <textarea id="ai-question-box" rows="4" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-white">Analyze tyre degradation, corner apex speeds, and optimal pit strategy for the ${ov.year} ${ov.event_name} (${ov.location}).</textarea>
        </div>

        <button onclick="transmitToEngineer()" id="ai-submit-btn" class="w-full py-2.5 rounded-lg bg-red-600 hover:bg-red-500 text-sm font-extrabold text-white cursor-pointer">
          Transmit to AI Race Engineer
        </button>
        <p id="ai-elapsed-timer" class="text-xs text-center text-slate-400 font-mono"></p>
      </div>

      <!-- Right: Grounded Evidence + Strategy Brief -->
      <div class="lg:col-span-8 space-y-5">
        <div class="pit-card p-6">
          <div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3 mb-4">
            <div>
              <h2 class="text-base font-bold text-white">📋 Chief Strategist Pit-Wall Brief</h2>
              <p id="ai-resolved-label" class="text-xs text-slate-400">Ready — click Transmit to ground on 2024–2026 DuckDB evidence</p>
            </div>
            <div class="flex items-center gap-2">
              <span id="ai-db-badge" class="text-xs px-2.5 py-0.5 rounded bg-emerald-950 border border-emerald-700 text-emerald-300 font-mono hidden"></span>
              <span id="ai-model-badge" class="text-xs px-2.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">Standby</span>
            </div>
          </div>
          <div id="ai-brief-body" class="text-sm text-slate-300 leading-relaxed space-y-2">
            Ask any question across <strong>2024, 2025, or 2026</strong> — including track-specific tyre wear, corner speeds, engine speed traps, or hot vs. rainy session performance.
          </div>
        </div>

        <!-- Live Grounded Statistical Evidence Panel -->
        <div id="ai-evidence-panel" class="hidden space-y-5">
          <div class="pit-card p-5 space-y-3">
            <h3 class="text-sm font-bold text-white">🏎️ Grounded Evidence 1: Engine Speed Traps &amp; Sector Benchmarks</h3>
            <div id="ai-ev-engine"></div>
            <div id="ai-ev-engine-sum" class="table-summary"></div>
          </div>

          <div class="pit-card p-5 space-y-3">
            <h3 class="text-sm font-bold text-white">🛞 Grounded Evidence 2: Pirelli Compound Slopes &amp; Pit-Stop Simulation</h3>
            <div id="ai-ev-tyres"></div>
            <div id="ai-ev-tyres-sum" class="table-summary"></div>
          </div>

          <div class="pit-card p-5 space-y-3">
            <h3 class="text-sm font-bold text-white">🌡️ Grounded Evidence 3: 2024–2026 Engine &amp; Pace Behavior Across Hot / Cool / Wet Regimes</h3>
            <div id="ai-ev-regimes"></div>
            <div class="table-summary">📋 <strong>Table Summary:</strong> Cross-season 2024–2026 power unit speed-trap and pace delta sensitivity across Hot (&ge;38°C), Moderate, Cool (&le;26°C), and Rainy regimes.</div>
          </div>
        </div>
      </div>
    </div>
  `;
};

window.setAiPrompt = function (txt) {
  document.getElementById('ai-question-box').value = txt;
};

window.transmitToEngineer = async function () {
  const { year, round } = PitWallState;
  const question = document.getElementById('ai-question-box').value.trim();
  if (!question) return;

  const btn = document.getElementById('ai-submit-btn');
  const timerEl = document.getElementById('ai-elapsed-timer');
  const briefEl = document.getElementById('ai-brief-body');
  btn.disabled = true;
  btn.textContent = 'Querying 2024–2026 DuckDB & Gemini...';
  briefEl.innerHTML = '<p class="text-slate-400 animate-pulse">Pulling multi-season telemetry, weather regimes, and pit simulations...</p>';

  const start = performance.now();
  const interval = setInterval(() => {
    timerEl.textContent = `Elapsed: ${((performance.now() - start) / 1000).toFixed(1)}s`;
  }, 100);

  try {
    const res = await fetch('/api/race-engineer/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, year, round_num: round })
    });
    const data = await res.json();
    clearInterval(interval);
    timerEl.textContent = `Completed in ${((performance.now() - start) / 1000).toFixed(2)}s`;

    if (!res.ok) {
      briefEl.innerHTML = `<p class="text-red-400">Error: ${data.detail || 'Request failed'}</p>`;
      return;
    }

    document.getElementById('ai-model-badge').textContent = data.model_used;
    const dbBadge = document.getElementById('ai-db-badge');
    dbBadge.classList.remove('hidden');
    dbBadge.textContent = `DuckDB: ${data.db_query_ms} ms`;
    document.getElementById('ai-resolved-label').textContent = `Grounded on ${data.resolved_event} + 2024–2026 Regimes`;
    briefEl.innerHTML = marked.parse(data.strategy_brief || '');

    // Render Live Grounded Evidence Tables
    const ev = data.evidence || {};
    const panel = document.getElementById('ai-evidence-panel');
    panel.classList.remove('hidden');

    const engRows = ev.engine_sectors || [];
    document.getElementById('ai-ev-engine').innerHTML = renderDataTable(
      engRows,
      ['Driver', 'Team', 'BestLap_Sec', 'Best_S1', 'Best_S2', 'Best_S3', 'PeakSpeedST_kmh', 'SpeedFL_kmh'],
      { BestLap_Sec: 'Best Lap (s)', PeakSpeedST_kmh: 'Speed Trap (km/h)', SpeedFL_kmh: 'Finish Line (km/h)' }
    );
    document.getElementById('ai-ev-engine-sum').innerHTML = engRows.length
      ? `📋 <strong>Table Summary:</strong> Top ${engRows.length} representative drivers for ${ev.resolved_event}. Benchmark lap: <strong>${engRows[0].Driver} (${engRows[0].BestLap_Sec}s, ${engRows[0].PeakSpeedST_kmh} km/h trap)</strong>.`
      : '';

    const compRows = ev.compound_summary || [];
    document.getElementById('ai-ev-tyres').innerHTML = renderDataTable(
      compRows,
      ['Compound', 'Stints', 'MedianRawPace_Sec', 'FuelCorrPace_Sec', 'MedianDeg_SecPerLap'],
      { MedianRawPace_Sec: 'Raw Median (s)', FuelCorrPace_Sec: 'Fuel-Corr Pace (s)', MedianDeg_SecPerLap: 'Deg Slope (s/lap)' }
    );
    const bestSim = (ev.pit_simulation || [])[0];
    document.getElementById('ai-ev-tyres-sum').innerHTML =
      `📋 <strong>Table Summary:</strong> Real DuckDB tyre degradation slopes for ${ev.resolved_event}` +
      (bestSim ? ` — fastest simulated strategy: <strong>${bestSim.Strategy} (${bestSim.CumulativeTime_Sec}s)</strong>.` : '.');

    document.getElementById('ai-ev-regimes').innerHTML = renderDataTable(
      ev.weather_engine_regimes || [],
      ['Year', 'WeatherRegime', 'Team', 'Races', 'AvgEngineSpeedST_kmh', 'AvgPaceDelta_Pct'],
      { WeatherRegime: 'Thermal / Wet Regime', AvgEngineSpeedST_kmh: 'Avg Speed Trap (km/h)', AvgPaceDelta_Pct: 'Pace Δ vs Best (%)' }
    );
  } catch (err) {
    clearInterval(interval);
    briefEl.innerHTML = `<p class="text-red-400">Network error: ${err}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Transmit to AI Race Engineer';
  }
};