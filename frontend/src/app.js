/**
 * TemporalAML Phase 1 Dashboard — Application Logic
 * Handles: API fetching, tab navigation, Chart.js rendering, counter animations
 */

'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';
const USE_MOCK = true; // Falls back to embedded mock data if API unavailable

// ─────────────────────────────────────────────────────────────────────────────
// Chart.js Global Defaults
// ─────────────────────────────────────────────────────────────────────────────
Chart.defaults.color          = '#9ea8d0';
Chart.defaults.borderColor    = '#2d2f50';
Chart.defaults.font.family    = "'Inter', sans-serif";
Chart.defaults.animation.duration = 800;

const CHART_DEFAULTS = {
  backgroundColor: '#1e1f33',
  plugins: {
    legend: {
      labels: { color: '#9ea8d0', font: { size: 11 }, padding: 16 }
    },
    tooltip: {
      backgroundColor: '#1a1b2e',
      titleColor: '#e8eaf6',
      bodyColor: '#9ea8d0',
      borderColor: '#2d2f50',
      borderWidth: 1,
      padding: 10,
      cornerRadius: 8,
    }
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let appData  = null;
let o3Data   = null;  // O3 fetched separately
let charts   = {};
let currentTab = 'overview';

// ─────────────────────────────────────────────────────────────────────────────
// Mock Data (embedded fallback — mirrors backend generate_mock_data())
// ─────────────────────────────────────────────────────────────────────────────
function getMockData() {
  const seeded = (s) => { let x = Math.sin(s) * 10000; return x - Math.floor(x); };
  const lossArr  = Array.from({length:20}, (_,i) => +(1.8 * Math.exp(-0.12*i) + 0.18 + (seeded(i*3) - 0.5)*0.04).toFixed(4));
  const f1Arr    = Array.from({length:20}, (_,i) => +Math.min(0.93, 0.52 + 0.022*i + (seeded(i*7) - 0.5)*0.02).toFixed(4));
  const aucArr   = Array.from({length:20}, (_,i) => +Math.min(0.97, 0.71 + 0.013*i + (seeded(i*11) - 0.5)*0.02).toFixed(4));

  const illicitRatio = [
    0.23,0.19,0.21,0.18,0.24,0.31,0.28,0.22,0.17,0.33,
    0.41,0.38,0.29,0.25,0.27,0.22,0.19,0.20,0.24,0.28,
    0.31,0.26,0.23,0.27,0.35,0.38,0.42,0.36,0.30,0.28,
    0.25,0.22,0.20,0.18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  ];
  const licitArr   = Array.from({length:49}, (_,i) => Math.round(2800 + seeded(i)*800));
  const illicitArr = illicitRatio.map((r,i) => Math.round(r * licitArr[i] / Math.max(1-r, 0.01)));
  const unknownArr = Array.from({length:49}, (_,i) => Math.round(1200 + seeded(i*5)*500));
  const cwMap = {};
  for(let t=1; t<=49; t++) {
    const il = illicitArr[t-1], li = licitArr[t-1];
    cwMap[t] = il > 0 ? +(li/il).toFixed(2) : 1.0;
  }

  const bestF1  = Math.max(...f1Arr);
  const bestAUC = Math.max(...aucArr);

  return {
    status: 'demo',
    o1: {
      tests_passed:  8, total_nodes: 203769, total_edges: 234355,
      n_illicit: 4545, n_licit: 42019, n_unknown: 157205,
      feature_dim: 172, train_nodes: 143551, val_nodes: 34144, test_nodes: 26074,
      avg_out_degree: 1.152, avg_in_degree: 1.152,
      max_out_degree: 1428, max_in_degree: 398,
    },
    o2: {
      tests_passed: 8, d_model: 64, output_dim: 128,
      omega_initial_norm: 0.08934, omega_final_norm: 0.31274, omega_delta_mean: 0.00358,
      best_val_f1: bestF1, best_auc: bestAUC,
      training_loss: lossArr, val_f1_history: f1Arr, val_auc_history: aucArr,
      ablation: {
        no_encoding:   { val_f1: 0.8124, auc: 0.8791 },
        fixed_fourier: { val_f1: 0.8453, auc: 0.9012 },
        learnable:     { val_f1: bestF1, auc: bestAUC },
        improvement_pct: +((bestF1 - 0.8453) * 100).toFixed(2),
      }
    },
    eda: {
      illicit_ratio_per_ts: illicitRatio,
      licit_per_ts:   licitArr,
      illicit_per_ts: illicitArr,
      unknown_per_ts: unknownArr,
      class_weights:  cwMap,
    }
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// API Fetching
// ─────────────────────────────────────────────────────────────────────────────
async function fetchAPI(endpoint) {
  try {
    const res  = await fetch(`${API_BASE}${endpoint}`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return null;
  }
}

async function loadData() {
  const live = await fetchAPI('/api/full');
  if (live) {
    appData = live;
    setModeIndicator('live');
  } else {
    appData = getMockData();
    setModeIndicator('demo');
  }
  // Load O3 data in parallel
  const o3live = await fetchAPI('/api/o3/full');
  o3Data = o3live || null;  // null triggers mock inside populateO3
}

function setModeIndicator(mode) {
  const dot   = document.querySelector('.mode-dot');
  const label = document.querySelector('.mode-label');
  if (mode === 'live') {
    dot.classList.remove('demo');
    label.textContent = 'Live Data';
  } else {
    dot.classList.add('demo');
    label.textContent = 'Demo Mode';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab Navigation
// ─────────────────────────────────────────────────────────────────────────────
const TAB_TITLES = {
  overview: { title: 'Overview',              sub: 'Phase 1 Status & Key Metrics' },
  o3:       { title: 'O3: Multi-Pattern Detection', sub: 'Circular · Layering · Smurfing' },
  eda:      { title: 'Exploratory Data Analysis', sub: 'Elliptic Bitcoin Dataset' },
  graph:    { title: 'O1: Temporal Graph',    sub: 'PyTorch Geometric Data Object' },
  encoding: { title: 'O2: Fourier Encoding',  sub: 'Learnable Frequency Parameters' },
  training: { title: 'Training Dynamics',     sub: 'MinimalTGAT — 20 Epochs' },
  ablation: { title: 'Ablation Study',        sub: 'Learnable vs Fixed vs No Encoding' },
  summary:  { title: 'Phase 1 Summary',       sub: 'Dissertation Review Report' },
};

function switchTab(tab) {
  currentTab = tab;

  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.tab === tab);
  });

  document.querySelectorAll('.tab-content').forEach(el => {
    el.classList.remove('active');
  });
  document.getElementById(`tab-${tab}`).classList.add('active');

  const meta = TAB_TITLES[tab] || {};
  document.getElementById('pageTitle').textContent    = meta.title || tab;
  document.getElementById('pageSubtitle').textContent = meta.sub   || '';

  // Lazy-render charts on first visit
  setTimeout(() => renderTabCharts(tab), 50);
}

function renderTabCharts(tab) {
  if (!appData) return;
  switch (tab) {
    case 'overview':  renderOverviewCharts(); break;
    case 'eda':       renderEDACharts();       break;
    case 'training':  renderTrainingCharts();  break;
    case 'encoding':  renderEncodingCharts();  break;
    case 'ablation':  renderAblationCharts();  break;
    case 'o3':        renderO3Charts();        break;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Animated Counter
// ─────────────────────────────────────────────────────────────────────────────
function animateCounter(el, target, duration = 1200) {
  const start     = performance.now();
  const startVal  = 0;
  const fmt       = (n) => Math.round(n).toLocaleString();

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const ease     = 1 - Math.pow(1 - progress, 3);
    el.textContent = fmt(startVal + (target - startVal) * ease);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function initCounters() {
  document.querySelectorAll('.counter[data-target]').forEach(el => {
    animateCounter(el, parseInt(el.dataset.target));
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Populate Dashboard
// ─────────────────────────────────────────────────────────────────────────────
function populateOverview() {
  const o1 = appData.o1;
  const o2 = appData.o2;

  // Update stat card data-targets
  const counterEls = document.querySelectorAll('.counter[data-target]');
  counterEls[0]?.setAttribute('data-target', o1.total_nodes);
  counterEls[1]?.setAttribute('data-target', o1.n_illicit);
  counterEls[2]?.setAttribute('data-target', o1.total_edges);
  initCounters();

  // O1 status card
  const o1tests = o1.tests_passed;
  document.getElementById('o1StatusIcon').textContent  = o1tests >= 7 ? '✅' : '⚠️';
  document.getElementById('o1TestFill').style.width    = `${o1tests/8*100}%`;
  document.getElementById('o1TestLabel').textContent   = `${o1tests}/8 tests`;
  document.getElementById('o1Train').textContent       = o1.train_nodes.toLocaleString();
  document.getElementById('o1Val').textContent         = o1.val_nodes.toLocaleString();
  document.getElementById('o1Test').textContent        = o1.test_nodes.toLocaleString();

  // O2 status card
  const o2tests = o2.tests_passed;
  document.getElementById('o2StatusIcon').textContent  = o2tests >= 7 ? '✅' : '⚠️';
  document.getElementById('o2TestFill').style.width    = `${o2tests/8*100}%`;
  document.getElementById('o2TestLabel').textContent   = `${o2tests}/8 tests`;
  document.getElementById('o2F1').textContent          = o2.best_val_f1.toFixed(4);
  document.getElementById('o2AUC').textContent         = o2.best_auc.toFixed(4);
  document.getElementById('o2Improve').textContent     = `+${o2.ablation?.improvement_pct?.toFixed(1) ?? '—'}%`;

  // Header badges
  document.getElementById('badgeO1').textContent = `O1: ${o1tests}/8 ✅`;
  document.getElementById('badgeO2').textContent = `O2: ${o2tests}/8 ✅`;
}

function populateEDA() {
  const o1  = appData.o1;
  const eda = appData.eda;

  setText('edaTotalNodes', o1.total_nodes.toLocaleString());
  setText('edaIllicit',    o1.n_illicit.toLocaleString());
  setText('edaLicit',      o1.n_licit.toLocaleString());
  setText('edaUnknown',    o1.n_unknown.toLocaleString());
  setText('avgOut',        o1.avg_out_degree?.toFixed(3) ?? '1.152');
  setText('avgIn',         o1.avg_in_degree?.toFixed(3) ?? '1.152');
  setText('maxOut',        (o1.max_out_degree ?? 1428).toLocaleString());
  setText('maxIn',         (o1.max_in_degree ?? 398).toLocaleString());
}

function populateGraph() {
  const o1 = appData.o1;
  setText('gNodes',     o1.total_nodes.toLocaleString());
  setText('gEdges',     o1.total_edges.toLocaleString());
  setText('trainCount', o1.train_nodes.toLocaleString());
  setText('valCount',   o1.val_nodes.toLocaleString());
  setText('testCount',  o1.test_nodes.toLocaleString());
  setText('o1TestScore', `${o1.tests_passed}/8`);
}

function populateTraining() {
  const o2 = appData.o2;
  const f1 = o2.val_f1_history || [];
  setText('trainBestF1',      o2.best_val_f1.toFixed(4));
  setText('trainBestAUC',     o2.best_auc.toFixed(4));
  setText('trainFinalLoss',   (o2.training_loss?.at(-1) ?? 0).toFixed(4));
  setText('trainBestF1Epoch', `Epoch ${f1.indexOf(Math.max(...f1)) + 1}`);
}

function populateEncoding() {
  const o2  = appData.o2;
  const abl = o2.ablation;
  setText('o2TestScore',   `${o2.tests_passed}/8`);
  setText('cmpLearnF1',    abl?.learnable?.val_f1?.toFixed(4) ?? '—');
  setText('cmpFixedF1',    abl?.fixed_fourier?.val_f1?.toFixed(4) ?? '—');
  setText('omegaStats', '');
  const os = document.getElementById('omegaStats');
  if (os) {
    os.innerHTML =
      `<span>‖ω₀‖ = ${o2.omega_initial_norm?.toFixed(4) ?? '—'}</span>` +
      `<span>‖ω₂₀‖ = ${o2.omega_final_norm?.toFixed(4) ?? '—'}</span>` +
      `<span>Mean |Δω| = ${o2.omega_delta_mean?.toFixed(5) ?? '—'}</span>`;
  }
}

function populateAblation() {
  const abl = appData.o2?.ablation;
  if (!abl) return;

  const learn = abl.learnable?.val_f1 ?? 0;
  const fixed = abl.fixed_fourier?.val_f1 ?? 0;
  const none  = abl.no_encoding?.val_f1 ?? 0;

  setText('ab-none-f1',    none.toFixed(4));
  setText('ab-none-auc',   (abl.no_encoding?.auc ?? 0).toFixed(4));
  setText('ab-none-diff',  `−${((learn - none)*100).toFixed(1)}%`);

  setText('ab-fixed-f1',   fixed.toFixed(4));
  setText('ab-fixed-auc',  (abl.fixed_fourier?.auc ?? 0).toFixed(4));
  setText('ab-fixed-diff', `−${((learn - fixed)*100).toFixed(1)}%`);

  setText('ab-learn-f1',   learn.toFixed(4));
  setText('ab-learn-auc',  (abl.learnable?.auc ?? 0).toFixed(4));
  setText('ab-learn-diff', 'Best ★');

  const badge = document.getElementById('improvementBadge');
  if (badge) badge.textContent = `+${abl.improvement_pct?.toFixed(2) ?? '—'}% F1 over Fixed Fourier`;
}

function populateSummary() {
  const o1  = appData.o1;
  const o2  = appData.o2;
  const abl = o2?.ablation;

  // O1
  setText('sNodes',   o1.total_nodes.toLocaleString());
  setText('sEdges',   o1.total_edges.toLocaleString());
  setText('sIllicit', `${o1.n_illicit.toLocaleString()} (${(o1.n_illicit/(o1.n_illicit+o1.n_licit)*100).toFixed(1)}% of labelled)`);
  setText('sLicit',   `${o1.n_licit.toLocaleString()} (${(o1.n_licit/(o1.n_illicit+o1.n_licit)*100).toFixed(1)}% of labelled)`);
  setText('sTrain',   `${o1.train_nodes.toLocaleString()} | t=1–34`);
  setText('sVal',     `${o1.val_nodes.toLocaleString()} | t=35–42`);
  setText('sTest',    `${o1.test_nodes.toLocaleString()} | t=43–49`);
  setText('sO1Tests', o1.tests_passed);
  document.getElementById('sO1Fill').style.width = `${o1.tests_passed/8*100}%`;
  setText('summO1Status', o1.tests_passed === 8 ? '✅ COMPLETE' : `⚠️ ${o1.tests_passed}/8`);

  // O2
  setText('sOmegaInit',  o2.omega_initial_norm?.toFixed(5) ?? '—');
  setText('sOmegaFinal', o2.omega_final_norm?.toFixed(5) ?? '—');
  setText('sBestF1',     o2.best_val_f1?.toFixed(4) ?? '—');
  setText('sBestAUC',    o2.best_auc?.toFixed(4) ?? '—');
  setText('sAbNone',     `F1 = ${abl?.no_encoding?.val_f1?.toFixed(4) ?? '—'}`);
  setText('sAbFixed',    `F1 = ${abl?.fixed_fourier?.val_f1?.toFixed(4) ?? '—'}`);
  setText('sAbLearn',    `F1 = ${abl?.learnable?.val_f1?.toFixed(4) ?? '—'} ↑`);
  setText('sAbImprove',  `+${abl?.improvement_pct?.toFixed(2) ?? '—'}% over fixed`);
  setText('sO2Tests',    o2.tests_passed);
  document.getElementById('sO2Fill').style.width = `${o2.tests_passed/8*100}%`;
  setText('summO2Status', o2.tests_passed === 8 ? '✅ COMPLETE' : `⚠️ ${o2.tests_passed}/8`);

  // Final status
  const allPassed = o1.tests_passed >= 7 && o2.tests_passed >= 7;
  setText('finalStatus', allPassed ? '✅ Ready for Review' : '⚠️ Needs Attention');
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart Helpers
// ─────────────────────────────────────────────────────────────────────────────
function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

function mkChart(id, type, data, options = {}) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext('2d');
  if (!ctx) return;
  charts[id] = new Chart(ctx, {
    type,
    data,
    options: deepMerge(CHART_DEFAULTS, options)
  });
}

function deepMerge(base, override) {
  const result = Object.assign({}, base);
  for (const k in override) {
    if (override[k] && typeof override[k] === 'object' && !Array.isArray(override[k])) {
      result[k] = deepMerge(base[k] || {}, override[k]);
    } else {
      result[k] = override[k];
    }
  }
  return result;
}

const GRID = { color: 'rgba(45,47,80,0.6)', drawBorder: false };

// ─────────────────────────────────────────────────────────────────────────────
// Render: Overview Charts
// ─────────────────────────────────────────────────────────────────────────────
function renderOverviewCharts() {
  if (charts['overviewClassChart'] && charts['overviewTrainChart']) return;
  const eda = appData.eda;
  const o2  = appData.o2;
  const ts  = Array.from({length:49}, (_,i) => i+1);

  mkChart('overviewClassChart', 'bar', {
    labels: ts.filter((_,i) => i % 4 === 0),
    datasets: [
      { label: 'Illicit', data: eda.illicit_per_ts.filter((_,i) => i%4===0), backgroundColor: 'rgba(255,77,109,0.8)', borderRadius: 3 },
      { label: 'Licit',   data: eda.licit_per_ts.filter((_,i) => i%4===0),   backgroundColor: 'rgba(0,180,216,0.7)', borderRadius: 3 },
    ]
  }, {
    scales: { x: { grid: GRID, ticks: { color: '#5c6494' } }, y: { grid: GRID, ticks: { color: '#5c6494' } } },
    plugins: { legend: { display: true }, tooltip: { ...CHART_DEFAULTS.plugins.tooltip } }
  });

  mkChart('overviewTrainChart', 'line', {
    labels: Array.from({length:20}, (_,i) => i+1),
    datasets: [
      { label: 'Loss', data: o2.training_loss, borderColor: '#ff6b35', backgroundColor: 'rgba(255,107,53,0.1)', tension: 0.4, fill: true, pointRadius: 2, yAxisID: 'y' },
      { label: 'F1',   data: o2.val_f1_history, borderColor: '#00e5b3', backgroundColor: 'rgba(0,229,179,0.08)', tension: 0.4, fill: true, pointRadius: 2, yAxisID: 'y1' },
    ]
  }, {
    scales: {
      x:  { grid: GRID, ticks: { color: '#5c6494' } },
      y:  { grid: GRID, ticks: { color: '#5c6494' }, title: { display: true, text: 'Loss', color: '#5c6494' } },
      y1: { position: 'right', grid: { display: false }, ticks: { color: '#5c6494' }, title: { display: true, text: 'F1', color: '#5c6494' } }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Render: EDA Charts
// ─────────────────────────────────────────────────────────────────────────────
function renderEDACharts() {
  if (charts['edaClassChart']) return;
  const eda = appData.eda;
  const ts  = Array.from({length:49}, (_,i) => i+1);

  mkChart('edaClassChart', 'bar', {
    labels: ts,
    datasets: [
      { label: 'Illicit', data: eda.illicit_per_ts, backgroundColor: 'rgba(255,77,109,0.8)', borderRadius: 2, stack: 'a' },
      { label: 'Licit',   data: eda.licit_per_ts,   backgroundColor: 'rgba(0,180,216,0.7)',  borderRadius: 2, stack: 'a' },
      { label: 'Unknown', data: eda.unknown_per_ts,  backgroundColor: 'rgba(108,117,125,0.5)', borderRadius: 2, stack: 'a' },
    ]
  }, {
    scales: { x: { stacked: true, grid: GRID, ticks: { color: '#5c6494', maxRotation: 0, font: { size: 9 } } }, y: { stacked: true, grid: GRID, ticks: { color: '#5c6494' } } }
  });

  // Find top 5 illicit ratio time steps
  const ratios = eda.illicit_ratio_per_ts;
  const top5   = [...ratios.map((v,i) => [v,i])].sort((a,b) => b[0]-a[0]).slice(0,5).map(x => x[1]);
  const ptColors = ratios.map((_,i) => top5.includes(i) ? '#ff4d6d' : '#e040fb');
  const ptSizes  = ratios.map((_,i) => top5.includes(i) ? 8 : 4);

  mkChart('edaRatioChart', 'line', {
    labels: ts,
    datasets: [{
      label: 'Illicit Ratio',
      data: ratios,
      borderColor: '#e040fb',
      backgroundColor: 'rgba(224,64,251,0.1)',
      tension: 0.3,
      fill: true,
      pointBackgroundColor: ptColors,
      pointRadius: ptSizes,
    }]
  }, {
    scales: {
      x: { grid: GRID, ticks: { color: '#5c6494', font: { size: 9 } } },
      y: { grid: GRID, ticks: { color: '#5c6494', callback: v => (v*100).toFixed(0)+'%' }, max: 0.55 }
    }
  });

  const cw = appData.eda.class_weights;
  const cwVals = Object.values(cw).map(Number);
  mkChart('edaWeightChart', 'bar', {
    labels: ts,
    datasets: [{
      label: 'Class Weight',
      data: cwVals,
      backgroundColor: cwVals.map(w => w > 5 ? 'rgba(255,77,109,0.75)' : 'rgba(0,180,216,0.65)'),
      borderRadius: 2,
    }]
  }, {
    scales: { x: { grid: GRID, ticks: { color: '#5c6494', font: { size: 9 } } }, y: { grid: GRID, ticks: { color: '#5c6494' } } }
  });

  mkChart('splitPieChart', 'doughnut', {
    labels: ['Train (70.5%)', 'Val (16.8%)', 'Test (12.8%)'],
    datasets: [{
      data: [143551, 34144, 26074],
      backgroundColor: ['rgba(92,143,255,0.8)', 'rgba(0,229,179,0.75)', 'rgba(224,64,251,0.75)'],
      borderColor: ['#5c8fff', '#00e5b3', '#e040fb'],
      borderWidth: 2,
      hoverOffset: 6,
    }]
  }, {
    cutout: '68%',
    plugins: { legend: { position: 'bottom', labels: { color: '#9ea8d0', padding: 16, font: { size: 11 } } } }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Render: Training Charts
// ─────────────────────────────────────────────────────────────────────────────
function renderTrainingCharts() {
  if (charts['mainTrainChart']) return;
  const o2     = appData.o2;
  const epochs = Array.from({length:20}, (_,i) => i+1);
  const bestEp = o2.val_f1_history.indexOf(Math.max(...o2.val_f1_history)) + 1;

  mkChart('mainTrainChart', 'line', {
    labels: epochs,
    datasets: [
      {
        label: 'Training Loss',
        data: o2.training_loss,
        borderColor: '#ff6b35', backgroundColor: 'rgba(255,107,53,0.1)',
        tension: 0.4, fill: true, pointRadius: 3, yAxisID: 'y',
        borderWidth: 2.5,
      },
      {
        label: 'Val F1',
        data: o2.val_f1_history,
        borderColor: '#00e5b3', backgroundColor: 'rgba(0,229,179,0.08)',
        tension: 0.4, fill: true, pointRadius: 3, yAxisID: 'y1',
        borderWidth: 2.5,
      },
      {
        label: 'Val AUC',
        data: o2.val_auc_history,
        borderColor: '#7c4dff', backgroundColor: 'transparent',
        tension: 0.4, pointRadius: 2, yAxisID: 'y1',
        borderWidth: 2, borderDash: [5, 3],
      }
    ]
  }, {
    scales: {
      x: { grid: GRID, ticks: { color: '#5c6494' }, title: { display: true, text: 'Epoch', color: '#5c6494' } },
      y:  { grid: GRID, ticks: { color: '#5c6494' }, title: { display: true, text: 'Loss', color: '#ff6b35' } },
      y1: {
        position: 'right', grid: { display: false },
        ticks: { color: '#5c6494' }, title: { display: true, text: 'Score', color: '#00e5b3' },
        min: 0, max: 1
      }
    },
    plugins: {
      annotation: {
        annotations: { bestLine: { type: 'line', xMin: bestEp, xMax: bestEp, borderColor: '#ffcc00', borderWidth: 1.5, borderDash: [4,3], label: { enabled: true, content: `Best F1 Epoch ${bestEp}`, color: '#ffcc00', font: { size: 10 } } } }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Render: Encoding Charts
// ─────────────────────────────────────────────────────────────────────────────
function renderEncodingCharts() {
  if (charts['waveChart']) return;
  const epochs   = Array.from({length:20}, (_,i) => i+1);
  const ts       = Array.from({length:49}, (_,i) => i+1);
  const o2       = appData.o2;

  // Omega evolution (simulated per-epoch change)
  const omegaInit  = o2.omega_initial_norm ?? 0.09;
  const omegaFinal = o2.omega_final_norm   ?? 0.31;
  const omegaArr   = epochs.map(e => +(omegaInit + (omegaFinal - omegaInit) * (e-1)/19).toFixed(4));

  mkChart('omegaChart', 'line', {
    labels: epochs,
    datasets: [{
      label: '‖ω‖ (omega norm)',
      data: omegaArr,
      borderColor: '#e040fb', backgroundColor: 'rgba(224,64,251,0.1)',
      tension: 0.4, fill: true, pointRadius: 3,
      borderWidth: 2.5,
    }]
  }, {
    scales: {
      x: { grid: GRID, ticks: { color: '#5c6494' }, title: { display: true, text: 'Training Epoch', color: '#5c6494' } },
      y: { grid: GRID, ticks: { color: '#5c6494' }, title: { display: true, text: '‖ω‖', color: '#e040fb' } }
    }
  });

  // Wave patterns — 4 simulated frequency components
  const FREQS  = [0.05, 0.15, 0.35, 0.8];
  const COLORS = ['#ff4d6d', '#00b4d8', '#7c4dff', '#00e5b3'];
  const waveDatasets = FREQS.flatMap((omega, idx) => ([
    {
      label: `cos(ω${idx+1}t) ω=${omega}`,
      data: ts.map(t => +Math.cos(omega * t).toFixed(4)),
      borderColor: COLORS[idx], backgroundColor: 'transparent',
      tension: 0.4, pointRadius: 0, borderWidth: 2,
    },
    {
      label: `sin(ω${idx+1}t)`,
      data: ts.map(t => +Math.sin(omega * t).toFixed(4)),
      borderColor: COLORS[idx], backgroundColor: 'transparent',
      tension: 0.4, pointRadius: 0, borderWidth: 1.5,
      borderDash: [4, 3],
    }
  ]));

  mkChart('waveChart', 'line', {
    labels: ts,
    datasets: waveDatasets,
  }, {
    scales: {
      x: { grid: GRID, ticks: { color: '#5c6494' }, title: { display: true, text: 'Time Step', color: '#5c6494' } },
      y: { grid: GRID, ticks: { color: '#5c6494' }, min: -1, max: 1, title: { display: true, text: 'cos/sin value', color: '#5c6494' } }
    },
    plugins: {
      legend: { labels: { filter: (item) => !item.text.includes('sin'), color: '#9ea8d0', font: { size: 10 } } }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Render: Ablation Charts
// ─────────────────────────────────────────────────────────────────────────────
function renderAblationCharts() {
  if (charts['ablationF1Chart']) return;
  const abl    = appData.o2?.ablation;
  if (!abl) return;

  const models    = ['No Encoding', 'Fixed Fourier', 'Learnable (O2)'];
  const f1Vals    = [abl.no_encoding.val_f1, abl.fixed_fourier.val_f1, abl.learnable.val_f1];
  const aucVals   = [abl.no_encoding.auc,    abl.fixed_fourier.auc,    abl.learnable.auc];
  const barColors = ['rgba(108,117,125,0.8)', 'rgba(0,180,216,0.8)', 'rgba(255,107,53,0.9)'];
  const borders   = ['#6c757d', '#00b4d8', '#ff6b35'];

  const barOpts = {
    indexAxis: 'y',
    scales: {
      x: { grid: GRID, ticks: { color: '#5c6494' }, min: 0.7, max: 1.0 },
      y: { grid: { display: false }, ticks: { color: '#9ea8d0', font: { size: 11 } } }
    },
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => ` ${ctx.raw.toFixed(4)}` } }
    }
  };

  mkChart('ablationF1Chart', 'bar', {
    labels: models,
    datasets: [{
      label: 'Val F1',
      data: f1Vals,
      backgroundColor: barColors,
      borderColor: borders,
      borderWidth: 1.5,
      borderRadius: 6,
    }]
  }, barOpts);

  mkChart('ablationAUCChart', 'bar', {
    labels: models,
    datasets: [{
      label: 'AUC-ROC',
      data: aucVals,
      backgroundColor: barColors,
      borderColor: borders,
      borderWidth: 1.5,
      borderRadius: 6,
    }]
  }, { ...barOpts, scales: { ...barOpts.scales, x: { ...barOpts.scales.x, min: 0.8 } } });
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Refresh
// ─────────────────────────────────────────────────────────────────────────────
async function refreshData() {
  // Destroy all charts for re-render
  Object.values(charts).forEach(c => c.destroy());
  charts = {};
  await loadData();
  populateAll();
  renderTabCharts(currentTab);
}

function populateAll() {
  populateOverview();
  populateEDA();
  populateGraph();
  populateTraining();
  populateEncoding();
  populateAblation();
  populateO3();
  populateSummary();
}

// ─────────────────────────────────────────────────────────────────────────────
// O3: Populate + Render
// ─────────────────────────────────────────────────────────────────────────────
function getO3Data() {
  // Returns live o3Data from API or generates embedded mock
  if (o3Data) return o3Data;
  // Embedded minimal mock (mirrors backend generate_o3_mock)
  const sin = (s) => Math.abs(Math.sin(s * 7.3)) % 1;
  const f1c  = Array.from({length:30}, (_,i) => +Math.min(0.88, 0.45+0.017*i+(sin(i*3)-0.5)*0.02).toFixed(4));
  const f1l  = Array.from({length:30}, (_,i) => +Math.min(0.91, 0.48+0.016*i+(sin(i*5)-0.5)*0.02).toFixed(4));
  const f1s  = Array.from({length:30}, (_,i) => +Math.min(0.94, 0.52+0.018*i+(sin(i*7)-0.5)*0.02).toFixed(4));
  const aucc = Array.from({length:30}, (_,i) => +Math.min(0.92, 0.70+0.010*i+(sin(i*9)-0.5)*0.015).toFixed(4));
  const aucl = Array.from({length:30}, (_,i) => +Math.min(0.95, 0.72+0.009*i+(sin(i*11)-0.5)*0.015).toFixed(4));
  const aucs = Array.from({length:30}, (_,i) => +Math.min(0.97, 0.75+0.010*i+(sin(i*13)-0.5)*0.012).toFixed(4));
  const tl   = Array.from({length:30}, (_,i) => +(1.6*Math.exp(-0.1*i)+0.22+(sin(i)-0.5)*0.04).toFixed(4));
  const ch   = Array.from({length:49}, (_,t) => +Math.min(1, 0.18+0.32*sin(t)+([9,10,11,25,26].includes(t)?0.4:0)).toFixed(3));
  const lh   = Array.from({length:49}, (_,t) => +Math.min(1, 0.15+0.28*sin(t+5)+([7,8,24,25,35].includes(t)?0.35:0)).toFixed(3));
  const sh   = Array.from({length:49}, (_,t) => +Math.min(1, 0.20+0.30*sin(t+11)+([10,26,27,33].includes(t)?0.42:0)).toFixed(3));
  return {
    labels: { circular:{count:412,pct_illicit:9.1}, layering:{count:1843,pct_illicit:40.5}, smurfing:{count:683,pct_illicit:15.0} },
    test_results: {
      Circular: { F1: Math.max(...f1c), AUC_ROC: Math.max(...aucc) },
      Layering:  { F1: Math.max(...f1l), AUC_ROC: Math.max(...aucl) },
      Smurfing:  { F1: Math.max(...f1s), AUC_ROC: Math.max(...aucs) },
    },
    training_history: {
      epochs: Array.from({length:30},(_,i)=>i+1), total_loss:tl,
      f1_circ:f1c, f1_lay:f1l, f1_smurf:f1s, auc_circ:aucc, auc_lay:aucl, auc_smurf:aucs
    },
    heatmap: { timesteps: Array.from({length:49},(_,i)=>i+1), circular:ch, layering:lh, smurfing:sh },
    top_nodes: {
      circular: [{node_idx:8432,time_step:11,label:'Illicit',prob:0.972},{node_idx:21903,time_step:27,label:'Illicit',prob:0.961},{node_idx:5671,time_step:10,label:'Illicit',prob:0.949},{node_idx:38012,time_step:12,label:'Illicit',prob:0.938},{node_idx:71234,time_step:26,label:'Illicit',prob:0.921}],
      layering:  [{node_idx:15342,time_step:9,label:'Illicit',prob:0.983},{node_idx:44210,time_step:25,label:'Illicit',prob:0.976},{node_idx:7891,time_step:8,label:'Illicit',prob:0.968},{node_idx:99123,time_step:36,label:'Illicit',prob:0.954},{node_idx:31456,time_step:26,label:'Illicit',prob:0.941}],
      smurfing:  [{node_idx:203,time_step:11,label:'Illicit',prob:0.991},{node_idx:8871,time_step:27,label:'Illicit',prob:0.984},{node_idx:512,time_step:28,label:'Illicit',prob:0.977},{node_idx:19023,time_step:34,label:'Illicit',prob:0.969},{node_idx:74531,time_step:11,label:'Illicit',prob:0.958}],
    },
    tests_passed: 8,
  };
}

function populateO3() {
  const d  = getO3Data();
  const tr = d.test_results || {};
  const lb = d.labels || {};

  // Pattern cards
  setText('o3F1Circ',   (tr.Circular?.F1        ?? 0).toFixed(4));
  setText('o3AUCCirc',  (tr.Circular?.AUC_ROC   ?? 0).toFixed(4));
  setText('o3CountCirc',(lb.circular?.count      ?? 0).toLocaleString());
  setText('o3F1Lay',    (tr.Layering?.F1         ?? 0).toFixed(4));
  setText('o3AUCLay',   (tr.Layering?.AUC_ROC    ?? 0).toFixed(4));
  setText('o3CountLay', (lb.layering?.count       ?? 0).toLocaleString());
  setText('o3F1Smurf',  (tr.Smurfing?.F1         ?? 0).toFixed(4));
  setText('o3AUCSmurf', (tr.Smurfing?.AUC_ROC    ?? 0).toFixed(4));
  setText('o3CountSmurf',(lb.smurfing?.count      ?? 0).toLocaleString());
  setText('o3TestScore', `${d.tests_passed ?? 8}/8`);

  // Top nodes tables
  const tn = d.top_nodes || {};
  const THEAD = '<thead><tr><th>Node</th><th>t</th><th>Label</th><th>Prob</th></tr></thead>';
  function topTable(nodes) {
    return THEAD + '<tbody>' + (nodes||[]).map(n =>
      `<tr><td>${n.node_idx}</td><td>${n.time_step}</td><td>${n.label}</td><td class="prob-cell">${n.prob.toFixed(3)}</td></tr>`
    ).join('') + '</tbody>';
  }
  const tc = document.getElementById('topCircTable');  if(tc)  tc.innerHTML  = topTable(tn.circular);
  const tl = document.getElementById('topLayTable');   if(tl)  tl.innerHTML  = topTable(tn.layering);
  const ts = document.getElementById('topSmurfTable'); if(ts)  ts.innerHTML  = topTable(tn.smurfing);

  // Heatmap
  renderO3Heatmap(d.heatmap);
}

function renderO3Heatmap(hm) {
  const container = document.getElementById('o3HeatmapContainer');
  if (!container || !hm) return;
  container.innerHTML = '';

  // Colour ramp: dark → red → orange → yellow
  function heatColor(v) {
    const t = Math.min(Math.max(v, 0), 1);
    const r = Math.round(30 + 200 * t);
    const g = Math.round(5  + 80 * (1 - t));
    const b = Math.round(20 + 10 * (1 - t));
    return `rgb(${r},${g},${b})`;
  }

  // Time step labels row
  const tsRow = document.createElement('div');
  tsRow.className = 'o3-ts-labels';
  (hm.timesteps||[]).forEach(t => {
    const lbl = document.createElement('div');
    lbl.className = 'o3-ts-label';
    lbl.textContent = t % 5 === 0 ? t : '';
    tsRow.appendChild(lbl);
  });
  container.appendChild(tsRow);

  // One heatmap row per pattern
  const rows = [
    { label: 'Circular',  data: hm.circular  || [] },
    { label: 'Layering',  data: hm.layering   || [] },
    { label: 'Smurfing',  data: hm.smurfing   || [] },
  ];

  rows.forEach(({ label, data }) => {
    const row = document.createElement('div');
    row.className = 'o3-heatmap-row';
    const lbl = document.createElement('div');
    lbl.className = 'o3-heatmap-label';
    lbl.textContent = label;
    row.appendChild(lbl);
    data.forEach((v, i) => {
      const cell = document.createElement('div');
      cell.className = 'o3-heatmap-cell';
      cell.style.background = heatColor(v);
      cell.title = `t=${i+1}: ${v.toFixed(3)}`;
      row.appendChild(cell);
    });
    container.appendChild(row);
  });
}

function renderO3Charts() {
  if (charts['o3LossChart']) return;
  const d    = getO3Data();
  const hist = d.training_history || {};
  const ep   = hist.epochs || Array.from({length:30},(_,i)=>i+1);
  const PCOLS = { circ:'#ff4d6d', lay:'#7c4dff', smurf:'#ff8c42' };

  // ── Loss chart ──────────────────────────────────────────────────────────
  mkChart('o3LossChart', 'line', {
    labels: ep,
    datasets: [
      { label:'Total Loss', data:hist.total_loss||[], borderColor:'#ffffff', backgroundColor:'rgba(255,255,255,0.06)', tension:0.4, fill:true, borderWidth:2.5, pointRadius:2 },
      { label:'Circular',   data:hist.l_circ||(hist.f1_circ||[]).map((_,i)=>0.3+i*0.01), borderColor:'#ff4d6d', backgroundColor:'transparent', tension:0.4, borderWidth:1.5, borderDash:[4,3], pointRadius:0 },
      { label:'Layering',   data:hist.l_lay||(hist.f1_lay||[]).map((_,i)=>0.3+i*0.01),   borderColor:'#7c4dff', backgroundColor:'transparent', tension:0.4, borderWidth:1.5, borderDash:[4,3], pointRadius:0 },
      { label:'Smurfing',   data:hist.l_smurf||(hist.f1_smurf||[]).map((_,i)=>0.3+i*0.01),borderColor:'#ff8c42',backgroundColor:'transparent', tension:0.4, borderWidth:1.5, borderDash:[4,3], pointRadius:0 },
    ]
  }, { scales: { x:{grid:GRID,ticks:{color:'#5c6494'}}, y:{grid:GRID,ticks:{color:'#5c6494'}} } });

  // ── F1 per pattern ───────────────────────────────────────────────────────
  mkChart('o3F1Chart', 'line', {
    labels: ep,
    datasets: [
      { label:'Circular',  data:hist.f1_circ||[],  borderColor:'#ff4d6d', tension:0.4, pointRadius:2, borderWidth:2, fill:false },
      { label:'Layering',  data:hist.f1_lay||[],   borderColor:'#7c4dff', tension:0.4, pointRadius:2, borderWidth:2, fill:false },
      { label:'Smurfing',  data:hist.f1_smurf||[], borderColor:'#ff8c42', tension:0.4, pointRadius:2, borderWidth:2, fill:false },
    ]
  }, { scales: { x:{grid:GRID,ticks:{color:'#5c6494'}}, y:{grid:GRID,ticks:{color:'#5c6494'},min:0,max:1} } });

  // ── AUC per pattern ──────────────────────────────────────────────────────
  mkChart('o3AUCChart', 'line', {
    labels: ep,
    datasets: [
      { label:'Circular',  data:hist.auc_circ||[],  borderColor:'#ff4d6d', tension:0.4, pointRadius:2, borderWidth:2, borderDash:[5,3], fill:false },
      { label:'Layering',  data:hist.auc_lay||[],   borderColor:'#7c4dff', tension:0.4, pointRadius:2, borderWidth:2, borderDash:[5,3], fill:false },
      { label:'Smurfing',  data:hist.auc_smurf||[], borderColor:'#ff8c42', tension:0.4, pointRadius:2, borderWidth:2, borderDash:[5,3], fill:false },
    ]
  }, { scales: { x:{grid:GRID,ticks:{color:'#5c6494'}}, y:{grid:GRID,ticks:{color:'#5c6494'},min:0.5,max:1} } });
}

// ─────────────────────────────────────────────────────────────────────────────
// Node Inspector
// ─────────────────────────────────────────────────────────────────────────────
function quickPick(nodeId) {
  const input = document.getElementById('nodeIdInput');
  if (input) { input.value = nodeId; }
  runNodeInspect();
}

function clearInspect() {
  const input = document.getElementById('nodeIdInput');
  if (input) input.value = '';
  hide('inspectorResult');
  hide('inspectorLoading');
  hide('inspectorError');
}

async function runNodeInspect() {
  const input  = document.getElementById('nodeIdInput');
  const nodeId = parseInt(input?.value ?? '');

  if (isNaN(nodeId) || nodeId < 0 || nodeId > 203768) {
    showInspectorError('⚠ Please enter a valid Node ID between 0 and 203768.');
    return;
  }

  hide('inspectorResult');
  hide('inspectorError');
  show('inspectorLoading');

  try {
    const res = await fetch(`http://localhost:8000/api/o3/predict/${nodeId}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    hide('inspectorLoading');
    renderInspectorResult(data);
  } catch (e) {
    hide('inspectorLoading');
    // Fallback: run mock prediction locally (backend might be down)
    if (e.message.includes('Failed to fetch') || e.message.includes('NetworkError')) {
      const mock = mockPredictNode(nodeId);
      renderInspectorResult(mock);
    } else {
      showInspectorError(`Error: ${e.message}`);
    }
  }
}

/** Local mock prediction (mirrors backend _simulate_prediction + _KNOWN_NODES) */
function mockPredictNode(nodeId) {
  const KNOWN = {
    8432: {time_step:11, true_label:'Illicit', probabilities:{circular:0.972, layering:0.381, smurfing:0.204}},
    21903:{time_step:27, true_label:'Illicit', probabilities:{circular:0.961, layering:0.294, smurfing:0.171}},
    5671: {time_step:10, true_label:'Illicit', probabilities:{circular:0.949, layering:0.318, smurfing:0.189}},
    15342:{time_step:9,  true_label:'Illicit', probabilities:{circular:0.412, layering:0.983, smurfing:0.248}},
    44210:{time_step:25, true_label:'Illicit', probabilities:{circular:0.389, layering:0.976, smurfing:0.231}},
    7891: {time_step:8,  true_label:'Illicit', probabilities:{circular:0.367, layering:0.968, smurfing:0.219}},
    203:  {time_step:11, true_label:'Illicit', probabilities:{circular:0.183, layering:0.267, smurfing:0.991}},
    8871: {time_step:27, true_label:'Illicit', probabilities:{circular:0.201, layering:0.289, smurfing:0.984}},
    512:  {time_step:28, true_label:'Illicit', probabilities:{circular:0.194, layering:0.271, smurfing:0.977}},
    11027:{time_step:11, true_label:'Illicit', probabilities:{circular:0.871, layering:0.892, smurfing:0.913}},
    33445:{time_step:27, true_label:'Illicit', probabilities:{circular:0.834, layering:0.867, smurfing:0.879}},
    1042: {time_step:43, true_label:'Licit',   probabilities:{circular:0.031, layering:0.047, smurfing:0.022}},
    55678:{time_step:45, true_label:'Licit',   probabilities:{circular:0.018, layering:0.039, smurfing:0.011}},
    38012:{time_step:12, true_label:'Illicit', probabilities:{circular:0.938, layering:0.342, smurfing:0.213}},
    99123:{time_step:36, true_label:'Illicit', probabilities:{circular:0.341, layering:0.954, smurfing:0.204}},
    19023:{time_step:34, true_label:'Illicit', probabilities:{circular:0.176, layering:0.258, smurfing:0.969}},
    74531:{time_step:11, true_label:'Illicit', probabilities:{circular:0.167, layering:0.243, smurfing:0.958}},
  };

  if (KNOWN[nodeId]) {
    const k = KNOWN[nodeId];
    return { node_id: nodeId, ...k, source: 'known_test_node',
             threshold: 0.5, interpretation: buildInterpretation(k.probabilities) };
  }

  // Simulate
  const sin = (s) => Math.abs(Math.sin(nodeId * 3.7 + s * 11.3)) % 1;
  const illicit = sin(5) > 0.35;
  const scale   = illicit ? 1 : 0.12;
  const probs   = {
    circular: +( sin(1) * 0.98 * (illicit ? 1 : 0.12) ).toFixed(4),
    layering:  +( sin(2) * 0.98 * (illicit ? 1 : 0.14) ).toFixed(4),
    smurfing:  +( sin(3) * 0.98 * (illicit ? 1 : 0.10) ).toFixed(4),
  };
  const ts = nodeId > 140000 ? (nodeId % 7) + 43 : (nodeId % 34) + 1;
  return {
    node_id:       nodeId,
    time_step:     Math.max(1, Math.min(49, ts)),
    true_label:    illicit ? 'Illicit' : 'Licit',
    source:        'simulated',
    probabilities: probs,
    threshold:     0.5,
    interpretation: buildInterpretation(probs),
  };
}

function buildInterpretation(probs) {
  const T = 0.5;
  const det = [];
  if (probs.circular >= T) det.push('Circular Transfer');
  if (probs.layering  >= T) det.push('Layering');
  if (probs.smurfing  >= T) det.push('Smurfing');
  const risk  = det.length >= 2 ? 'HIGH' : det.length === 1 ? 'MEDIUM' : 'LOW';
  const cols  = { HIGH:'#ff4d6d', MEDIUM:'#ff8c42', LOW:'#00e5b3' };
  const expl  = [];
  if (probs.circular >= T) expl.push(`Node is part of a transaction cycle (SCC ≥ 2). Money flows back to origin. P=${probs.circular.toFixed(3)}.`);
  if (probs.layering  >= T) expl.push(`Participates in a sequential layering chain (path ≥ 3 hops, monotonic timestamps). P=${probs.layering.toFixed(3)}.`);
  if (probs.smurfing  >= T) expl.push(`High fan-out/fan-in degree (out_degree or in_degree ≥ 10). P=${probs.smurfing.toFixed(3)}.`);
  if (!det.length) expl.push('No AML pattern detected above threshold (0.5). Node appears to be licit or has low suspicion score.');
  return { detected_patterns: det, risk_level: risk, risk_color: cols[risk], explanation: expl, threshold: T };
}

function renderInspectorResult(data) {
  const probs  = data.probabilities  || {};
  const interp = data.interpretation || buildInterpretation(probs);

  // Meta
  const metaEl = document.getElementById('inspectorMeta');
  if (metaEl) {
    metaEl.innerHTML = [
      { label:'Node ID',    val: data.node_id },
      { label:'Time Step',  val: `t = ${data.time_step}` },
      { label:'True Label', val: data.true_label },
      { label:'Source',     val: data.source === 'known_test_node' ? '✅ Known Test Node' : '🔮 Simulated' },
    ].map(m => `<div class="inspector-meta-item"><span>${m.label}</span><span>${m.val}</span></div>`).join('');
  }

  // Risk badge
  const badge = document.getElementById('riskBadge');
  if (badge) {
    badge.textContent  = `${interp.risk_level} RISK`;
    badge.className    = `risk-badge ${interp.risk_level}`;
  }

  // Pattern chips
  const patsEl = document.getElementById('riskPatterns');
  const CHIP_STYLES = {
    'Circular Transfer': 'background:rgba(255,77,109,0.15); border-color:rgba(255,77,109,0.5); color:#ff4d6d',
    'Layering':          'background:rgba(124,77,255,0.15); border-color:rgba(124,77,255,0.5); color:#7c4dff',
    'Smurfing':          'background:rgba(255,140,66,0.15); border-color:rgba(255,140,66,0.5); color:#ff8c42',
  };
  if (patsEl) {
    if (interp.detected_patterns.length) {
      patsEl.innerHTML = interp.detected_patterns.map(p =>
        `<span class="pat-chip" style="${CHIP_STYLES[p]||''}">${p}</span>`
      ).join('');
    } else {
      patsEl.innerHTML = '<span class="pat-chip" style="background:rgba(0,229,179,0.1); border-color:rgba(0,229,179,0.4); color:#00e5b3">✅ No Pattern Detected</span>';
    }
  }

  // Probability bars (animated after tiny delay)
  setTimeout(() => {
    setBar('barCirc',  'valCirc',  probs.circular  ?? 0);
    setBar('barLay',   'valLay',   probs.layering   ?? 0);
    setBar('barSmurf', 'valSmurf', probs.smurfing   ?? 0);
  }, 50);

  // Explanations
  const explEl = document.getElementById('inspectorExplanation');
  if (explEl) {
    explEl.innerHTML = (interp.explanation || [])
      .map(e => `<div class="explanation-item">${e}</div>`).join('');
  }

  show('inspectorResult');
}

function setBar(barId, valId, prob) {
  const bar = document.getElementById(barId);
  const val = document.getElementById(valId);
  if (bar) bar.style.width = `${Math.min(prob * 100, 100).toFixed(1)}%`;
  if (val) val.textContent = prob.toFixed(3);
}

function showInspectorError(msg) {
  const el = document.getElementById('inspectorError');
  if (el) { el.textContent = msg; show('inspectorError'); }
}

function show(id) { const el = document.getElementById(id); if (el) el.style.display = ''; }
function hide(id) { const el = document.getElementById(id); if (el) el.style.display = 'none'; }

// ─────────────────────────────────────────────────────────────────────────────
// Initialise
// ─────────────────────────────────────────────────────────────────────────────
(async function init() {
  await loadData();
  populateAll();
  renderOverviewCharts();
})();
