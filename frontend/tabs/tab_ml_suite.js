/**
 * MODULE 3: 10-MODEL AI / MACHINE LEARNING & SIMULATION SUITE
 * File: frontend/tabs/tab_ml_suite.js
 * Renders 10 individual, separated models with bespoke Plotly visualizations (no generic bar fallback)
 */

window.ActiveMLModelKey = 'model2_ranker';

window.loadMLWorkspace = async function () {
  const container = document.getElementById('workspace-ml');
  const ov = PitWallState.overview;
  if (!ov) return;

  container.innerHTML = `
    <div class="pit-card p-4 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h2 class="text-sm font-extrabold text-white">10-Model AI / Machine Learning Suite — ${ov.year} ${ov.event_name}</h2>
        <p class="text-xs text-slate-400">Select any of the 10 individual supervised, deep learning, unsupervised, Bayesian, or Monte Carlo models below.</p>
      </div>

      <div class="flex flex-wrap items-end gap-3">
        <div class="min-w-[340px]">
          <label class="block text-[10px] uppercase text-slate-400 font-bold mb-1">Select Individual AI / ML Model</label>
          <select id="ml-model-dropdown" onchange="onMLDropdownChange()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-bold text-white">
            <option value="model1_quali" ${ActiveMLModelKey === 'model1_quali' ? 'selected' : ''}>⚡ Practice-to-Qualifying (FP → Q) Pace Regressor</option>
            <option value="model2_ranker" ${ActiveMLModelKey === 'model2_ranker' ? 'selected' : ''}>🏁 Walk-Forward Pre-Race Classification Ranker (XGBoost)</option>
            <option value="model3_probabilities" ${ActiveMLModelKey === 'model3_probabilities' ? 'selected' : ''}>🎯 Calibrated Podium (Top 3) &amp; Points (Top 10) Classifier</option>
            <option value="model4_nn_tyre" ${ActiveMLModelKey === 'model4_nn_tyre' ? 'selected' : ''}>🧠 Deep Learning MLP (64→32 ReLU) Thermal Tyre Degradation</option>
            <option value="model5_xai" ${ActiveMLModelKey === 'model5_xai' ? 'selected' : ''}>🔎 Explainable AI (XAI): Global &amp; Local Feature Attribution</option>
            <option value="model6_clustering" ${ActiveMLModelKey === 'model6_clustering' ? 'selected' : ''}>🧩 Unsupervised PCA (2D) + K-Means Driver DNA Archetypes</option>
            <option value="model7_anomalies" ${ActiveMLModelKey === 'model7_anomalies' ? 'selected' : ''}>🚨 Unsupervised Isolation Forest Green-Flag Lap Anomalies</option>
            <option value="model8_evolution" ${ActiveMLModelKey === 'model8_evolution' ? 'selected' : ''}>📉 Time-Series Weekend Track Evolution &amp; Grip Gain</option>
            <option value="model9_bayesian" ${ActiveMLModelKey === 'model9_bayesian' ? 'selected' : ''}>⚖️ Sequential Bayesian TrueSkill / Elo (Driver vs. Car)</option>
            <option value="model10_montecarlo" ${ActiveMLModelKey === 'model10_montecarlo' ? 'selected' : ''}>🔮 1,000-Run Monte Carlo Stochastic Race Simulator</option>
          </select>
        </div>

        <button onclick="executeSelectedMLModel()" id="ml-exec-btn" class="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-xs font-extrabold text-white cursor-pointer">
          Run Model
        </button>
      </div>
    </div>

    <!-- Monte Carlo Controls (Shown Only for Model 10) -->
    <div id="ml-mc-bar" class="${ActiveMLModelKey === 'model10_montecarlo' ? '' : 'hidden'} pit-card p-4 grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
      <div>
        <label class="block text-xs text-slate-300 font-semibold mb-1">Target Grand Prix to Simulate (Active or Upcoming)</label>
        <select id="ml-mc-circuit" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-bold text-white">
          <option value="${ov.event_name}">Active Track: ${ov.year} ${ov.event_name}</option>
          <option value="Las Vegas Grand Prix">Upcoming: 2026 Las Vegas Grand Prix</option>
          <option value="Qatar Grand Prix">Upcoming: 2026 Qatar Grand Prix</option>
          <option value="Abu Dhabi Grand Prix">Upcoming: 2026 Abu Dhabi Grand Prix</option>
          <option value="2027 Season Opener">Future: 2027 Season Opener</option>
        </select>
      </div>
      <div>
        <div class="flex justify-between text-xs mb-1"><span class="text-slate-300">Rain Probability</span><span id="ml-v-rain" class="font-mono text-sky-400 font-bold">20%</span></div>
        <input type="range" id="ml-mc-rain" min="0" max="100" value="20" oninput="document.getElementById('ml-v-rain').textContent=this.value+'%'" class="w-full accent-sky-500" />
      </div>
      <div>
        <div class="flex justify-between text-xs mb-1"><span class="text-slate-300">Safety Car Probability</span><span id="ml-v-sc" class="font-mono text-amber-400 font-bold">40%</span></div>
        <input type="range" id="ml-mc-sc" min="0" max="100" value="40" oninput="document.getElementById('ml-v-sc').textContent=this.value+'%'" class="w-full accent-amber-500" />
      </div>
    </div>

    <div id="ml-output-host" class="space-y-6"></div>
  `;

  await executeSelectedMLModel();
};

window.onMLDropdownChange = function () {
  window.ActiveMLModelKey = document.getElementById('ml-model-dropdown').value;
  document.getElementById('ml-mc-bar').classList.toggle('hidden', window.ActiveMLModelKey !== 'model10_montecarlo');
  document.getElementById('ml-output-host').innerHTML =
    `<div class="pit-card p-6 text-sm text-slate-300">Model selected. Adjust any simulation sliders above (if applicable) and click <strong class="text-red-400">Run Model</strong> to execute.</div>`;
};

window.executeSelectedMLModel = async function () {
  const { year, round, session, overview } = PitWallState;
  const modelKey = document.getElementById('ml-model-dropdown').value;
  const host = document.getElementById('ml-output-host');
  const btn = document.getElementById('ml-exec-btn');

  btn.disabled = true;
  btn.textContent = 'Running...';
  host.innerHTML = `<div class="pit-card p-6 text-xs text-slate-400 animate-pulse">Running ${modelKey} on DuckDB Lakehouse...</div>`;

  const circuitName = modelKey === 'model10_montecarlo'
    ? document.getElementById('ml-mc-circuit').value
    : overview.event_name;

  const payload = {
    model_key: modelKey,
    year,
    round_num: round,
    session_type: session,
    circuit_name: circuitName,
    rain_prob: parseInt(document.getElementById('ml-mc-rain')?.value || '20') / 100.0,
    sc_prob: parseInt(document.getElementById('ml-mc-sc')?.value || '40') / 100.0,
  };

  const t0 = performance.now();
  try {
    const res = await fetch('/api/ml/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    const data = await res.json();
    recordLatency(performance.now() - t0);

    if (!res.ok) {
      host.innerHTML = `<div class="pit-card p-6 text-sm text-red-400">Model Error: ${data.detail}</div>`;
      return;
    }
    renderBespokeMLModel(modelKey, data, overview, circuitName);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Model';
  }
};

function renderBespokeMLModel(modelKey, data, ov, circuitName) {
  const host = document.getElementById('ml-output-host');
  const ctx = data.context || {};
  const primary = data.primary || [];
  const secondary = data.secondary || [];

  const conceptHtml = buildConceptHeader(ctx.f1_context, ctx.ds_methodology, ctx.key_insight);
  const darkBase = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', size: 11 }, margin: { t: 25, r: 30, b: 55, l: 65 }
  };

  if (!primary.length) {
    host.innerHTML = `${conceptHtml}<div class="pit-card p-6 text-sm text-amber-300">No training/evaluation records available for this specific round or session.</div>`;
    return;
  }

  // ---- MODEL 1: FP -> Q PARITY SCATTER + ALGORITHM METRICS ----
  if (modelKey === 'model1_quali') {
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">⚡ Predicted Saturday Grid vs. Actual Qualifying Result (Ideal Parity Line y = x)</h3>
        <div id="ml-viz-1" class="w-full h-[480px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> Points sitting directly on the dashed diagonal represent exact Saturday grid predictions from Friday Practice pace; points below the line outperformed their Friday practice showing.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📊 Regression Algorithm Evaluation Metrics (MAE, RMSE &amp; R²)</h3>
        ${renderDataTable(secondary, ['Algorithm', 'Position_MAE', 'Position_RMSE', 'Score_R2'], { Position_MAE: 'Position MAE', Position_RMSE: 'Position RMSE', Score_R2: 'Pace R²' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Benchmarks L2 Ridge, L1 Lasso, Random Forest, and XGBoost on out-of-sample Qualifying grid accuracy.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📋 Predicted Saturday Qualifying Classification vs. Actual Result</h3>
        ${renderDataTable(primary, ['Predicted_Quali_Pos', 'QualiPosition', 'Driver', 'OfficialTeamName', 'FP_BestDelta_Pct', 'DriverRollingQuali_5R'], { Predicted_Quali_Pos: 'Pred Grid', QualiPosition: 'Actual Grid', OfficialTeamName: 'Constructor', FP_BestDelta_Pct: 'FP Best Δ (%)', DriverRollingQuali_5R: '5R Avg Quali' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Full grid comparison between Friday practice predictions and official Saturday Q1–Q3 results for ${ov.year} ${ov.event_name}.</div>
      </div>`;

    const valid = primary.filter(r => r.Predicted_Quali_Pos != null && r.QualiPosition != null);
    Plotly.newPlot('ml-viz-1', [
      { x: [1, 20], y: [1, 20], mode: 'lines', name: 'Perfect Prediction (y = x)', line: { color: '#94a3b8', dash: 'dash' } },
      {
        x: valid.map(r => r.Predicted_Quali_Pos), y: valid.map(r => r.QualiPosition),
        text: valid.map(r => r.Driver), mode: 'markers+text', textposition: 'top center', name: 'Drivers',
        marker: { size: 12, color: valid.map(r => TEAM_COLORS[r.OfficialTeamName] || '#38bdf8') }
      }
    ], { ...darkBase, xaxis: { title: 'Predicted Qualifying Position', dtick: 2, gridcolor: '#1e293b' }, yaxis: { title: 'Actual Qualifying Position', dtick: 2, gridcolor: '#1e293b' } }, { responsive: true });
    return;
  }

  // ---- MODEL 2: PRE-RACE POSITION RANKER (GRID -> PREDICTED DUMBBELL DELTA CHART) ----
  if (modelKey === 'model2_ranker') {
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">🏁 Walk-Forward Pre-Race Ranker: Starting Grid vs. Predicted Finish (Dumbbell Movement Chart)</h3>
        <div id="ml-viz-1" class="w-full h-[540px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> Compares each driver's starting grid slot (grey diamond) against XGBoost's predicted finish (red circle) and actual race finish (green square) — lines moving left indicate forecasted race overtakes.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📋 Walk-Forward Pre-Race Classification Prediction Table</h3>
        ${renderDataTable(primary, ['PredictedPosition', 'FinishPosition', 'Driver', 'OfficialTeamName', 'GridPosition', 'ExpectedFinishScore', 'DriverRollingFinish_5R', 'TeamRollingFinish_5R', 'Status'], { PredictedPosition: 'AI Pred Pos', FinishPosition: 'Actual Pos', OfficialTeamName: 'Constructor', GridPosition: 'Grid', ExpectedFinishScore: 'XGB Score', DriverRollingFinish_5R: '5R Driver Avg', TeamRollingFinish_5R: '5R Team Avg' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Walk-forward pre-race ranker output for ${ov.year} ${ov.event_name} sorted from Predicted P1 to P${primary.length}.</div>
      </div>`;

    const ordered = [...primary].reverse();
    const shapes = ordered.map(r => ({
      type: 'line', x0: r.GridPosition, x1: r.PredictedPosition, y0: r.Driver, y1: r.Driver,
      line: { color: (r.PredictedPosition < r.GridPosition) ? '#10b981' : '#f59e0b', width: 2.5 }
    }));
    Plotly.newPlot('ml-viz-1', [
      { y: ordered.map(r => r.Driver), x: ordered.map(r => r.GridPosition), name: 'Starting Grid', mode: 'markers', marker: { size: 10, symbol: 'diamond', color: '#94a3b8' } },
      { y: ordered.map(r => r.Driver), x: ordered.map(r => r.PredictedPosition), name: 'AI Predicted Finish', mode: 'markers', marker: { size: 11, color: '#ef4444' } },
      { y: ordered.map(r => r.Driver), x: ordered.map(r => r.FinishPosition), name: 'Actual Finish', mode: 'markers', marker: { size: 9, symbol: 'square-open', color: '#10b981', line: { width: 2 } } }
    ], { ...darkBase, shapes, xaxis: { title: 'Race Position (P1 on Left = Better)', dtick: 1, gridcolor: '#1e293b' }, legend: { orientation: 'h', y: 1.07, x: 0 } }, { responsive: true });
    return;
  }

  // ---- MODEL 3: CALIBRATED PODIUM & POINTS PROBABILITY CLASSIFIER ----
  if (modelKey === 'model3_probabilities') {
    const top12 = primary.slice(0, 12).reverse();
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">🎯 Platt-Calibrated Podium (Top 3) &amp; Points (Top 10) Probabilities — Top 12 Contenders</h3>
        <div id="ml-viz-1" class="w-full h-[500px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> <strong>${primary[0].Driver}</strong> (${primary[0].OfficialTeamName}) leads calibrated podium probability at <strong>${primary[0].Podium_Prob_Pct}%</strong> (Points probability: <strong>${primary[0].Points_Prob_Pct}%</strong>).</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📊 Classification Calibration Metrics (ROC-AUC &amp; Brier Score Loss)</h3>
        ${renderDataTable(secondary, ['Target', 'Model', 'ROC_AUC', 'Brier_Score_Loss'], { ROC_AUC: 'ROC-AUC (Higher=Better)', Brier_Score_Loss: 'Brier Loss (Lower=Better)' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Verifies probability sharpness and ranking discrimination across Logistic Regression and Platt-Calibrated Gradient Boosting.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📋 Complete Driver Podium &amp; Points Probability Table</h3>
        ${renderDataTable(primary, ['Driver', 'OfficialTeamName', 'Podium_Prob_Pct', 'Points_Prob_Pct', 'GridPosition', 'PredictedPosition', 'FinishPosition'], { OfficialTeamName: 'Constructor', Podium_Prob_Pct: 'Podium %', Points_Prob_Pct: 'Points %', GridPosition: 'Grid', PredictedPosition: 'Pred Pos', FinishPosition: 'Actual Pos' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Sorted from highest Podium likelihood downward across all ${primary.length} starters.</div>
      </div>`;

    Plotly.newPlot('ml-viz-1', [
      { y: top12.map(r => r.Driver), x: top12.map(r => r.Podium_Prob_Pct), name: 'Podium Prob (Top 3 %)', type: 'bar', orientation: 'h', marker: { color: '#10b981' } },
      { y: top12.map(r => r.Driver), x: top12.map(r => r.Points_Prob_Pct), name: 'Points Prob (Top 10 %)', type: 'bar', orientation: 'h', marker: { color: '#38bdf8' } }
    ], { ...darkBase, barmode: 'group', xaxis: { title: 'Calibrated Probability (%)', range: [0, 105], gridcolor: '#1e293b' }, legend: { orientation: 'h', y: 1.07, x: 0 } }, { responsive: true });
    return;
  }

  // ---- MODEL 4: DEEP LEARNING MLP THERMAL TYRE DEGRADATION ----
  if (modelKey === 'model4_nn_tyre') {
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">🧠 Deep Learning MLP (64→32 ReLU): Non-Linear Thermal Tyre Degradation Curves (20°C–55°C)</h3>
        <div id="ml-viz-1" class="w-full h-[460px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> Plots neural network predicted pace drop-off (s/lap) as asphalt temperature rises across SOFT (red), MEDIUM (yellow), and HARD (white) compounds — steeper curves reveal thermal blistering thresholds.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">🌡️ Neural Network Thermal Degradation Lookup Matrix</h3>
        ${renderDataTable(primary, ['Compound', 'TrackTemp_C', 'NN_Predicted_DegSlope_SecPerLap'], { TrackTemp_C: 'Track Temp (°C)', NN_Predicted_DegSlope_SecPerLap: 'Predicted Deg Slope (s/lap)' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Feedforward Neural Network lookup matrix calibrated on ${ov.year} clean stint telemetry across all asphalt temperature regimes.</div>
      </div>`;

    const byComp = {};
    primary.forEach(r => { const c = String(r.Compound).toUpperCase(); if (!byComp[c]) byComp[c] = []; byComp[c].push(r); });
    const traces = Object.keys(byComp).map(c => ({
      x: byComp[c].map(r => r.TrackTemp_C),
      y: byComp[c].map(r => r.NN_Predicted_DegSlope_SecPerLap),
      name: c, mode: 'lines+markers', type: 'scatter',
      line: { color: TYRE_COLORS[c] || '#38bdf8', width: 3 }, marker: { size: 7 }
    }));
    Plotly.newPlot('ml-viz-1', traces, { ...darkBase, xaxis: { title: 'Asphalt Track Temperature (°C)', gridcolor: '#1e293b' }, yaxis: { title: 'Predicted Tyre Deg Slope (s/lap)', gridcolor: '#1e293b' }, legend: { orientation: 'h', y: 1.08, x: 0 } }, { responsive: true });
    return;
  }

  // ---- MODEL 5: EXPLAINABLE AI (XAI) FEATURE ATTRIBUTION ----
  if (modelKey === 'model5_xai') {
    const revXai = [...primary].reverse();
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">🔎 Explainable AI (XAI): Global XGBoost Gain vs. Permutation Importance</h3>
        <div id="ml-viz-1" class="w-full h-[460px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> <strong>${primary[0].Feature}</strong> is the #1 global driver of race outcome predictions (Gain: <strong>${primary[0].XGB_Gain_Importance}</strong>), cross-validated against out-of-sample Permutation Importance.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📊 Global Feature Importance Rankings Table</h3>
        ${renderDataTable(primary, ['Feature', 'XGB_Gain_Importance', 'Permutation_Importance'], { XGB_Gain_Importance: 'XGBoost Gain', Permutation_Importance: 'Permutation Importance' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Quantifies every engineered feature's global influence across the pre-race XGBoost ensemble.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📋 Local Per-Driver Feature Attribution Table (<em>'Why Did AI Rank Each Driver Here?'</em>)</h3>
        ${renderDataTable(secondary, ['PredictedPosition', 'Driver', 'OfficialTeamName', 'GridPosition', 'Podium_Prob_Pct', 'XAI_KeyDrivers'], { PredictedPosition: 'Pred Pos', OfficialTeamName: 'Constructor', GridPosition: 'Grid', Podium_Prob_Pct: 'Podium %', XAI_KeyDrivers: 'Local XAI Attribution Drivers' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Explains individual driver predictions in plain English so strategists can audit every ranking before lights out.</div>
      </div>`;

    Plotly.newPlot('ml-viz-1', [
      { y: revXai.map(r => r.Feature), x: revXai.map(r => r.XGB_Gain_Importance), name: 'XGBoost Gain Importance', type: 'bar', orientation: 'h', marker: { color: '#ef4444' } },
      { y: revXai.map(r => r.Feature), x: revXai.map(r => r.Permutation_Importance), name: 'Permutation Importance', type: 'bar', orientation: 'h', marker: { color: '#38bdf8' } }
    ], { ...darkBase, margin: { t: 25, r: 30, b: 45, l: 190 }, barmode: 'group', xaxis: { title: 'Feature Importance Score', gridcolor: '#1e293b' }, legend: { orientation: 'h', y: 1.08, x: 0 } }, { responsive: true });
    return;
  }

  // ---- MODEL 6: UNSUPERVISED PCA + K-MEANS DRIVER ARCHETYPES ----
  if (modelKey === 'model6_clustering') {
    const byArch = {};
    primary.forEach(r => { const a = r.Driver_Archetype || `Cluster ${r.Cluster_ID}`; if (!byArch[a]) byArch[a] = []; byArch[a].push(r); });
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">🧩 Unsupervised 2D PCA Projection + K-Means Driver Performance Archetypes (${ov.year})</h3>
        <div id="ml-viz-1" class="w-full h-[500px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> Compresses 6-axis driver telemetry DNA into 2 Principal Components, automatically separating the ${ov.year} grid into <strong>${Object.keys(byArch).length} behavioral archetypes</strong> without human labels.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📋 Unsupervised Driver Archetype &amp; Principal Component Coordinates Table</h3>
        ${renderDataTable(primary, ['Driver', 'Team', 'Driver_Archetype', 'Overall_DNA_Rating', 'PCA_1', 'PCA_2', 'Quali_Pace_Score', 'Race_Pace_Score', 'Tyre_Mgmt_Score', 'Overtake_Craft_Score'], { Driver_Archetype: 'K-Means Archetype', Overall_DNA_Rating: 'DNA Rating' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Lists every driver's discovered K-Means Archetype alongside exact 2D PCA coordinates and underlying telemetry scores.</div>
      </div>`;

    const cPal = ['#00F5D4', '#FF2A85', '#FFE600', '#38BDF8', '#A855F7'];
    const traces = Object.keys(byArch).map((arch, i) => ({
      x: byArch[arch].map(r => r.PCA_1), y: byArch[arch].map(r => r.PCA_2),
      text: byArch[arch].map(r => r.Driver), name: arch,
      mode: 'markers+text', textposition: 'top center',
      marker: { size: 13, color: cPal[i % cPal.length], line: { color: '#ffffff', width: 1 } }
    }));
    Plotly.newPlot('ml-viz-1', traces, { ...darkBase, xaxis: { title: 'Principal Component 1 (Overall Pace Axis)', gridcolor: '#1e293b' }, yaxis: { title: 'Principal Component 2 (Racecraft & Tyre Mgmt Axis)', gridcolor: '#1e293b' }, legend: { orientation: 'h', y: 1.08, x: 0 } }, { responsive: true });
    return;
  }

  // ---- MODEL 7: ISOLATION FOREST GREEN-FLAG LAP ANOMALIES ----
  if (modelKey === 'model7_anomalies') {
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">🚨 Unsupervised Isolation Forest: Green-Flag Multivariate Lap Anomalies (Lockups, Damage &amp; Traffic)</h3>
        <div id="ml-viz-1" class="w-full h-[460px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> Isolates the top <strong>${primary.length} anomalous green-flag laps</strong> deviating from each driver's compound-specific sector envelope. Most extreme outlier: <strong>${primary[0].Driver} Lap ${primary[0].LapNumber}</strong> (${primary[0].LapTime_Sec}s, worst loss in <strong>${primary[0].Worst_Sector}</strong>).</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📋 Isolated Green-Flag Lap Anomalies Table (Top Multivariate Outliers)</h3>
        ${renderDataTable(primary, ['Driver', 'Team', 'LapNumber', 'Compound', 'LapTime_Sec', 'LapTime_Sec_Dev', 'Worst_Sector', 'Anomaly_Score', 'SpeedST', 'TyreLife'], { LapNumber: 'Lap', LapTime_Sec: 'Lap Time (s)', LapTime_Sec_Dev: 'Dev vs Med (+s)', Worst_Sector: 'Worst Sector', Anomaly_Score: 'IsoForest Score', SpeedST: 'Speed Trap' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Flags mechanical drops, off-track excursions, or slow punctures without relying on FIA yellow-flag flags.</div>
      </div>`;

    Plotly.newPlot('ml-viz-1', [{
      x: primary.map(r => r.LapNumber), y: primary.map(r => r.LapTime_Sec),
      text: primary.map(r => `${r.Driver} L${r.LapNumber} (${r.Worst_Sector})`),
      mode: 'markers+text', textposition: 'top center',
      marker: { size: 11, color: primary.map(r => r.Anomaly_Score), colorscale: 'Hot', showscale: true, colorbar: { title: 'Score' } }
    }], { ...darkBase, xaxis: { title: 'Race Lap Number', gridcolor: '#1e293b' }, yaxis: { title: 'Anomalous Green-Flag Lap Time (s)', gridcolor: '#1e293b' } }, { responsive: true });
    return;
  }

  // ---- MODEL 8: TIME-SERIES TRACK EVOLUTION & GRIP GAIN ----
  if (modelKey === 'model8_evolution') {
    const lastEv = primary[primary.length - 1];
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📉 Chronological Weekend Track Evolution: Best Lap Drop vs. Cumulative Rubber Grip Gain</h3>
        <div id="ml-viz-1" class="w-full h-[450px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> Tracks circuit rubbering-in from <strong>${primary[0].SessionType}</strong> (${primary[0].BestLap_Sec}s) through <strong>${lastEv.SessionType}</strong> (${lastEv.BestLap_Sec}s), quantifying a cumulative grip gain of <strong>${lastEv.CumulativeGripGain_Sec}s</strong>.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📋 Chronological Session Track Evolution &amp; Weather Regime Table</h3>
        ${renderDataTable(primary, ['SessionType', 'BestLap_Sec', 'Top10PctLap_Sec', 'CumulativeGripGain_Sec', 'TrackTemp_C', 'IsWet'], { SessionType: 'Session', BestLap_Sec: 'Best Lap (s)', Top10PctLap_Sec: 'Top 10% Pace (s)', CumulativeGripGain_Sec: 'Cumulative Gain (s)', TrackTemp_C: 'Track Temp (°C)', IsWet: 'Wet?' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Chronological session progression pairing lap-time evolution against asphalt temperature and wet regimes.</div>
      </div>`;

    Plotly.newPlot('ml-viz-1', [
      { x: primary.map(r => r.SessionType), y: primary.map(r => r.BestLap_Sec), name: 'Session Best Lap (s)', type: 'scatter', mode: 'lines+markers', line: { color: '#00F5D4', width: 3 }, marker: { size: 9 } },
      { x: primary.map(r => r.SessionType), y: primary.map(r => r.Top10PctLap_Sec), name: 'Top 10% Pace (s)', type: 'scatter', mode: 'lines+markers', line: { color: '#38bdf8', width: 2, dash: 'dot' } },
      { x: primary.map(r => r.SessionType), y: primary.map(r => r.CumulativeGripGain_Sec), name: 'Cumulative Grip Gain (s)', type: 'bar', yaxis: 'y2', opacity: 0.35, marker: { color: '#f59e0b' } }
    ], { ...darkBase, margin: { t: 25, r: 65, b: 45, l: 65 }, xaxis: { title: 'Weekend Session Chronology', gridcolor: '#1e293b' }, yaxis: { title: 'Lap Time (s)', gridcolor: '#1e293b' }, yaxis2: { title: 'Cumulative Grip Gain (s)', overlaying: 'y', side: 'right' }, legend: { orientation: 'h', y: 1.09, x: 0 } }, { responsive: true });
    return;
  }

  // ---- MODEL 9: SEQUENTIAL BAYESIAN TRUESKILL / ELO (DRIVER VS. CAR) ----
  if (modelKey === 'model9_bayesian') {
    const top12 = primary.slice(0, 12);
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">⚖️ Sequential Bayesian Skill Decomposition: Pure Driver Elo (±σ Uncertainty) vs. Constructor Car Elo</h3>
        <div id="ml-viz-1" class="w-full h-[480px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> Decouples pure driver talent (with ±σ Bayesian uncertainty error bars) from constructor car speed. <strong>${top12[0].Driver}</strong> ranks #1 in Pure Driver Skill at <strong>${top12[0].PureDriverSkill_Elo} Elo</strong> (Car Elo: ${top12[0].ConstructorCar_Elo}).</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📋 Bayesian Pure Driver Skill vs. Constructor Car Elo Rankings Table</h3>
        ${renderDataTable(primary, ['DriverSkill_Rank', 'Driver', 'Team', 'PureDriverSkill_Elo', 'SkillUncertainty_Sigma', 'ConstructorCar_Elo', 'Combined_WinPower_Index'], { DriverSkill_Rank: 'Skill Rank', PureDriverSkill_Elo: 'Pure Driver Elo', SkillUncertainty_Sigma: 'Uncertainty (±σ)', ConstructorCar_Elo: 'Car Elo', Combined_WinPower_Index: 'Combined Win Power' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Identifies drivers outperforming midfield machinery versus drivers elevated by dominant chassis performance up to ${ov.year}.</div>
      </div>`;

    Plotly.newPlot('ml-viz-1', [
      {
        x: top12.map(r => r.Driver), y: top12.map(r => r.PureDriverSkill_Elo),
        error_y: { type: 'data', array: top12.map(r => r.SkillUncertainty_Sigma), visible: true, color: '#e2e8f0' },
        name: 'Pure Driver Skill Elo (±σ)', type: 'bar', marker: { color: '#38bdf8' }
      },
      { x: top12.map(r => r.Driver), y: top12.map(r => r.ConstructorCar_Elo), name: 'Constructor Car Elo', type: 'bar', marker: { color: '#ef4444' } }
    ], { ...darkBase, barmode: 'group', yaxis: { title: 'Bayesian Elo Rating', range: [900, Math.max(...top12.map(r => Math.max(r.PureDriverSkill_Elo, r.ConstructorCar_Elo))) + 150], gridcolor: '#1e293b' }, legend: { orientation: 'h', y: 1.08, x: 0 } }, { responsive: true });
    return;
  }

  // ---- MODEL 10: 1,000-RUN MONTE CARLO STOCHASTIC RACE SIMULATOR ----
  if (modelKey === 'model10_montecarlo') {
    const top12 = primary.slice(0, 12).reverse();
    host.innerHTML = `
      ${conceptHtml}
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">🔮 1,000-Run Monte Carlo Win &amp; Podium Probabilities — ${circuitName}</h3>
        <div id="ml-viz-1" class="w-full h-[460px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> Across 1,000 stochastic race simulations for <strong>${circuitName}</strong>, <strong>${primary[0].Driver}</strong> (${primary[0].Team}) holds a <strong>${primary[0].Win_Prob_Pct}% Win</strong> and <strong>${primary[0].Podium_Prob_Pct}% Podium</strong> probability.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📏 Stochastic Finishing Position Confidence Intervals (Best-Case P10 Ceiling → Median P50 → Worst-Case P90 Floor)</h3>
        <div id="ml-viz-2" class="w-full h-[480px]"></div>
        <div class="graph-summary">📊 <strong>Graph Summary:</strong> Horizontal bars span each driver's 80% probability band from P10 Ceiling (green dot) to P90 Floor (red dot) under weather and Safety Car volatility.</div>
      </div>
      <div class="pit-card p-5 space-y-3">
        <h3 class="text-base font-bold text-white">📋 Monte Carlo Probability Distribution &amp; Confidence Intervals Table — ${circuitName}</h3>
        ${renderDataTable(primary, ['Predicted_Rank', 'Driver', 'Team', 'Win_Prob_Pct', 'Podium_Prob_Pct', 'Points_Prob_Pct', 'Expected_Finish_Mean', 'Median_Finish_P50', 'Ceiling_P10', 'Floor_P90', 'Simulated_DNF_Risk_Pct'], { Predicted_Rank: 'Rank', Win_Prob_Pct: 'Win %', Podium_Prob_Pct: 'Podium %', Points_Prob_Pct: 'Points %', Expected_Finish_Mean: 'Mean Pos', Median_Finish_P50: 'Median P50', Ceiling_P10: 'Ceiling P10', Floor_P90: 'Floor P90', Simulated_DNF_Risk_Pct: 'DNF Risk %' })}
        <div class="table-summary">📋 <strong>Table Summary:</strong> Complete 1,000-run Monte Carlo grid distribution including DNF attrition risk and best/worst percentile finishes.</div>
      </div>`;

    Plotly.newPlot('ml-viz-1', [
      { y: top12.map(r => r.Driver), x: top12.map(r => r.Win_Prob_Pct), name: 'Win Prob (%)', type: 'bar', orientation: 'h', marker: { color: '#10b981' } },
      { y: top12.map(r => r.Driver), x: top12.map(r => r.Podium_Prob_Pct), name: 'Podium Prob (%)', type: 'bar', orientation: 'h', marker: { color: '#f59e0b' } }
    ], { ...darkBase, barmode: 'group', xaxis: { title: 'Simulated Probability (%)', gridcolor: '#1e293b' }, legend: { orientation: 'h', y: 1.08, x: 0 } }, { responsive: true });

    const rangeShapes = top12.map(r => ({
      type: 'line', x0: r.Ceiling_P10, x1: r.Floor_P90, y0: r.Driver, y1: r.Driver,
      line: { color: '#475569', width: 4 }
    }));
    Plotly.newPlot('ml-viz-2', [
      { y: top12.map(r => r.Driver), x: top12.map(r => r.Ceiling_P10), name: 'Best-Case Ceiling (P10)', mode: 'markers', marker: { size: 11, color: '#10b981' } },
      { y: top12.map(r => r.Driver), x: top12.map(r => r.Median_Finish_P50), name: 'Median Finish (P50)', mode: 'markers', marker: { size: 11, symbol: 'diamond', color: '#38bdf8' } },
      { y: top12.map(r => r.Driver), x: top12.map(r => r.Floor_P90), name: 'Worst-Case Floor (P90)', mode: 'markers', marker: { size: 11, color: '#ef4444' } }
    ], { ...darkBase, shapes: rangeShapes, xaxis: { title: 'Simulated Finishing Position (P1 on Left = Better)', dtick: 1, gridcolor: '#1e293b' }, legend: { orientation: 'h', y: 1.08, x: 0 } }, { responsive: true });
  }
}