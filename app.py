"""
app.py — AE-PD Research Command Centre

Full-stack research dashboard: Flask backend + inline frontend.
Monitors training, analyses overfit/underfit, visualises signals,
manages model registry, and provides resume control.

Run:
    python app.py
    Open http://localhost:1001
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from flask import Flask, Response, jsonify, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))

from flask_cors import CORS

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    _mlflow_client = MlflowClient()
    _MLFLOW = True
except Exception:
    _MLFLOW = False
    _mlflow_client = None


# ═══════════════════════════════════════════════════════════════════════════
# FRONTEND — Single-page HTML served at /
# ═══════════════════════════════════════════════════════════════════════════

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AE-PD Research Command Centre</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg: #0a0e1a;
  --bg-card: #111827;
  --bg-card-alt: #0f172a;
  --border: #1e293b;
  --border-bright: #334155;
  --cyan: #00d4ff;
  --cyan-dim: #0891b2;
  --green: #22c55e;
  --red: #ef4444;
  --amber: #f59e0b;
  --text: #e2e8f0;
  --text-dim: #94a3b8;
  --text-bright: #f8fafc;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'Inter', -apple-system, sans-serif;
}

* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:var(--sans); font-size:14px; line-height:1.5; }

.header {
  display:flex; align-items:center; justify-content:space-between;
  padding:12px 24px; border-bottom:1px solid var(--border);
  background:linear-gradient(180deg, #111827 0%, #0a0e1a 100%);
}
.header h1 { font-size:16px; font-weight:600; letter-spacing:0.5px; }
.header h1 span { color:var(--cyan); }
.header .gpu-pill {
  font-family:var(--mono); font-size:11px; padding:4px 12px;
  border:1px solid var(--border-bright); background:var(--bg-card);
}
.tabs {
  display:flex; border-bottom:1px solid var(--border); padding:0 24px;
  background:var(--bg-card-alt);
}
.tab {
  padding:10px 20px; cursor:pointer; font-size:13px; font-weight:500;
  color:var(--text-dim); border-bottom:2px solid transparent; transition:all .15s;
}
.tab:hover { color:var(--text); }
.tab.active { color:var(--cyan); border-bottom-color:var(--cyan); }
.page { display:none; padding:20px 24px; }
.page.active { display:block; }

.grid { display:grid; gap:16px; }
.grid-2 { grid-template-columns:1fr 1fr; }
.grid-3 { grid-template-columns:1fr 1fr 1fr; }
.grid-4 { grid-template-columns:1fr 1fr 1fr 1fr; }
@media(max-width:900px) { .grid-2,.grid-3,.grid-4 { grid-template-columns:1fr; } }

.card {
  background:var(--bg-card); border:1px solid var(--border); padding:16px;
}
.card h3 {
  font-size:11px; text-transform:uppercase; letter-spacing:1px;
  color:var(--text-dim); margin-bottom:12px;
}
.metric-value { font-family:var(--mono); font-size:28px; font-weight:700; color:var(--text-bright); }
.metric-label { font-size:11px; color:var(--text-dim); margin-top:2px; }
.metric-value.cyan { color:var(--cyan); }
.metric-value.green { color:var(--green); }
.metric-value.red { color:var(--red); }
.metric-value.amber { color:var(--amber); }

table { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:12px; }
th { text-align:left; padding:8px; color:var(--text-dim); font-weight:500; border-bottom:1px solid var(--border-bright); text-transform:uppercase; letter-spacing:0.5px; font-size:10px; }
td { padding:8px; border-bottom:1px solid var(--border); }
tr:hover td { background:rgba(0,212,255,0.03); }
.status-done { color:var(--green); } .status-running { color:var(--cyan); }
.status-queued { color:var(--text-dim); } .status-skipped { color:var(--amber); }

.btn {
  font-family:var(--mono); font-size:12px; padding:8px 18px; cursor:pointer;
  border:1px solid var(--cyan); background:transparent; color:var(--cyan);
  transition:all .15s; text-transform:uppercase; letter-spacing:0.5px;
}
.btn:hover { background:var(--cyan); color:var(--bg); }
.btn.danger { border-color:var(--red); color:var(--red); }
.btn.danger:hover { background:var(--red); color:#fff; }
.btn.green { border-color:var(--green); color:var(--green); }
.btn.green:hover { background:var(--green); color:var(--bg); }

.log-panel {
  background:#000; border:1px solid var(--border); font-family:var(--mono);
  font-size:11px; padding:8px; height:300px; overflow-y:auto;
  white-space:pre-wrap; word-break:break-all; color:#8892b0;
}
.log-panel .log-epoch { color:var(--cyan); }
.log-panel .log-warn { color:var(--amber); }
.log-panel .log-err { color:var(--red); }

.chart-container { position:relative; height:200px; background:var(--bg-card-alt); border:1px solid var(--border); padding:8px; }
canvas { width:100%!important; height:100%!important; }

.resume-box {
  background:#000; border:1px solid var(--border); padding:12px;
  font-family:var(--mono); font-size:11px; white-space:pre-wrap;
  max-height:300px; overflow-y:auto; color:var(--text-dim);
}

select, input[type="text"] {
  font-family:var(--mono); font-size:12px; padding:6px 10px;
  background:var(--bg); border:1px solid var(--border-bright);
  color:var(--text); outline:none;
}
select:focus, input:focus { border-color:var(--cyan); }

.badge { display:inline-block; padding:2px 8px; font-size:10px; font-family:var(--mono); }
.badge-ok { background:rgba(34,197,94,0.15); color:var(--green); border:1px solid rgba(34,197,94,0.3); }
.badge-warn { background:rgba(245,158,11,0.15); color:var(--amber); border:1px solid rgba(245,158,11,0.3); }
.badge-err { background:rgba(239,68,68,0.15); color:var(--red); border:1px solid rgba(239,68,68,0.3); }

.fit-bar { height:8px; background:var(--bg); border:1px solid var(--border); position:relative; margin-top:6px; }
.fit-bar-fill { height:100%; transition:width .3s; }
.fit-ok .fit-bar-fill { background:var(--green); }
.fit-over .fit-bar-fill { background:var(--red); }
.fit-under .fit-bar-fill { background:var(--amber); }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
</head>
<body>

<div class="header">
  <h1><span>AE-PD</span> Research Command Centre</h1>
  <div class="gpu-pill" id="gpuPill">GPU: loading...</div>
</div>

<div class="tabs">
  <div class="tab active" data-page="overview">Overview</div>
  <div class="tab" data-page="training">Training</div>
  <div class="tab" data-page="analysis">Analysis</div>
  <div class="tab" data-page="signals">Signals</div>
  <div class="tab" data-page="registry">Registry</div>
  <div class="tab" data-page="logs">Live Log</div>
</div>

<!-- ═══ OVERVIEW PAGE ═══ -->
<div class="page active" id="page-overview">
  <div class="grid grid-4" style="margin-bottom:16px;">
    <div class="card"><h3>Models Done</h3><div class="metric-value green" id="mDone">-</div></div>
    <div class="card"><h3>Models Running</h3><div class="metric-value cyan" id="mRunning">-</div></div>
    <div class="card"><h3>Models Queued</h3><div class="metric-value" id="mQueued">-</div></div>
    <div class="card"><h3>Best NCC</h3><div class="metric-value cyan" id="mBestNCC">-</div></div>
  </div>
  <div class="grid grid-2">
    <div class="card">
      <h3>Session Queue</h3>
      <table><thead><tr><th>Model</th><th>Status</th><th>Best NCC</th><th>Epoch</th></tr></thead>
      <tbody id="sessionTable"></tbody></table>
    </div>
    <div class="card">
      <h3>Resume Prompt <button class="btn" style="float:right;padding:4px 10px;font-size:10px;" onclick="copyResume()">Copy</button></h3>
      <div class="resume-box" id="resumeBox">Loading...</div>
    </div>
  </div>
  <div style="margin-top:16px;">
    <button class="btn green" onclick="resumeTraining()">Resume Training</button>
    <button class="btn danger" style="margin-left:8px;" onclick="stopTraining()">Stop Training</button>
  </div>
</div>

<!-- ═══ TRAINING PAGE ═══ -->
<div class="page" id="page-training">
  <div class="grid grid-2" style="margin-bottom:16px;">
    <div class="card">
      <h3>Training Loss Curve</h3>
      <div class="chart-container"><canvas id="lossChart"></canvas></div>
    </div>
    <div class="card">
      <h3>Validation NCC Curve</h3>
      <div class="chart-container"><canvas id="nccChart"></canvas></div>
    </div>
  </div>
  <div class="grid grid-2">
    <div class="card">
      <h3>Start Model Training</h3>
      <div style="display:flex;gap:8px;align-items:center;margin-top:8px;">
        <select id="modelSelect"></select>
        <button class="btn green" onclick="startModel()">Train</button>
      </div>
    </div>
    <div class="card">
      <h3>Loss Config</h3>
      <div style="display:flex;gap:12px;margin-top:8px;">
        <label><input type="radio" name="lossConfig" value="A" checked> A (Charb+Focal+Dice)</label>
        <label><input type="radio" name="lossConfig" value="B"> B (Charb+Dice)</label>
        <label><input type="radio" name="lossConfig" value="C"> C (Charb+SSIM)</label>
        <label><input type="radio" name="lossConfig" value="D"> D (SmoothL1+CE+Dice)</label>
      </div>
    </div>
  </div>
</div>

<!-- ═══ ANALYSIS PAGE ═══ -->
<div class="page" id="page-analysis">
  <div class="grid grid-3" style="margin-bottom:16px;">
    <div class="card"><h3>Test Set Size</h3><div class="metric-value" id="aTestSize">-</div><div class="metric-label">Synthetic windows</div></div>
    <div class="card"><h3>Real Q.Lin Size</h3><div class="metric-value" id="aRealSize">-</div><div class="metric-label">Real AE windows</div></div>
    <div class="card"><h3>SNR Levels</h3><div class="metric-value">5</div><div class="metric-label">-15, -10, -5, 0, +5 dB</div></div>
  </div>
  <div class="card" style="margin-bottom:16px;">
    <h3>Overfit / Underfit Analysis</h3>
    <table>
      <thead><tr><th>Model</th><th>Train Loss</th><th>Val Loss</th><th>Gap</th><th>Status</th><th>Diagnosis</th></tr></thead>
      <tbody id="fitTable"></tbody>
    </table>
  </div>
  <div class="grid grid-2">
    <div class="card">
      <h3>Benchmark Comparison</h3>
      <table>
        <thead><tr><th>Model</th><th>Params(M)</th><th>NCC-Synth</th><th>NCC-Real</th><th>Latency(ms)</th></tr></thead>
        <tbody id="benchTable"></tbody>
      </table>
    </div>
    <div class="card">
      <h3>Dataset Integrity</h3>
      <div id="datasetInfo" style="font-family:var(--mono);font-size:12px;">Loading...</div>
    </div>
  </div>
</div>

<!-- ═══ SIGNALS PAGE ═══ -->
<div class="page" id="page-signals">
  <div class="card" style="margin-bottom:16px;">
    <h3>Signal Visualization — Latest Snapshot</h3>
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px;">
      <select id="snapModel"></select>
      <button class="btn" onclick="loadSnapshot()">Load</button>
    </div>
    <img id="snapImg" style="max-width:100%;border:1px solid var(--border);" />
    <div id="snapStatus" style="font-family:var(--mono);font-size:11px;color:var(--text-dim);margin-top:6px;"></div>
  </div>
  <div class="card">
    <h3>Test Signal Preview</h3>
    <div class="chart-container" style="height:250px;"><canvas id="signalChart"></canvas></div>
  </div>
</div>

<!-- ═══ REGISTRY PAGE ═══ -->
<div class="page" id="page-registry">
  <div class="card">
    <h3>Model Registry (config/model_registry.yaml)</h3>
    <table>
      <thead><tr><th>Model ID</th><th>Wavelet</th><th>BiGRU</th><th>Swin</th><th>Attn</th><th>MTL</th><th>Loss</th></tr></thead>
      <tbody id="registryTable"></tbody>
    </table>
  </div>
</div>

<!-- ═══ LOGS PAGE ═══ -->
<div class="page" id="page-logs">
  <div class="card">
    <h3>Live Training Log <button class="btn" style="float:right;padding:4px 10px;font-size:10px;" onclick="clearLog()">Clear</button></h3>
    <div class="log-panel" id="logPanel"></div>
  </div>
</div>

<script>
const API = '';

// ── Tab switching ──
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('page-' + tab.dataset.page).classList.add('active');
  });
});

// ── Utilities ──
function $(id) { return document.getElementById(id); }
function fmt(v, d=4) { return v != null ? Number(v).toFixed(d) : '-'; }
function flag(v) { return v ? '<span style="color:var(--green)">ON</span>' : '<span style="color:var(--text-dim)">off</span>'; }

// ── GPU polling ──
async function pollGPU() {
  try {
    const r = await fetch(API + '/api/gpu');
    const d = await r.json();
    if (d.error) { $('gpuPill').textContent = 'GPU: error'; return; }
    $('gpuPill').innerHTML = `${d.name} &nbsp;|&nbsp; ${d.temp_c}&deg;C &nbsp;|&nbsp; ${d.utilization_pct}% &nbsp;|&nbsp; ${d.memory_used_mb}/${d.memory_total_mb} MB`;
  } catch(e) { $('gpuPill').textContent = 'GPU: offline'; }
}

// ── Session / Overview ──
async function loadSession() {
  try {
    const r = await fetch(API + '/api/session');
    const d = await r.json();
    if (d.status === 'no_session') {
      $('mDone').textContent = '0'; $('mRunning').textContent = '0';
      $('mQueued').textContent = '0'; $('mBestNCC').textContent = '-';
      $('sessionTable').innerHTML = '<tr><td colspan="4" style="color:var(--text-dim)">No session yet</td></tr>';
      return;
    }
    const models = d.models || {};
    const queue = d.queue || Object.keys(models);
    let done=0, running=0, queued=0, bestNcc=0;
    let rows = '';
    queue.forEach(mid => {
      const m = models[mid] || {};
      const st = m.status || 'queued';
      if (st==='done') done++; else if (st==='running') running++; else queued++;
      const ncc = m.best_ncc || 0;
      if (ncc > bestNcc) bestNcc = ncc;
      rows += `<tr><td>${mid}</td><td class="status-${st}">${st}</td><td>${fmt(ncc)}</td><td>${m.current_epoch||0}</td></tr>`;
    });
    $('mDone').textContent = done; $('mRunning').textContent = running;
    $('mQueued').textContent = queued; $('mBestNCC').textContent = fmt(bestNcc);
    $('sessionTable').innerHTML = rows;
  } catch(e) { console.error('Session error', e); }
}

// ── Resume prompt ──
async function loadResume() {
  try {
    const r = await fetch(API + '/api/resume-prompt');
    const d = await r.json();
    $('resumeBox').textContent = d.content || 'No resume prompt yet.';
  } catch(e) {}
}
function copyResume() {
  navigator.clipboard.writeText($('resumeBox').textContent);
  alert('Copied to clipboard');
}

// ── Training control ──
async function resumeTraining() {
  try {
    const r = await fetch(API + '/api/training/resume', {method:'POST'});
    const d = await r.json();
    alert(d.status || d.error || 'Training started');
  } catch(e) { alert('Error: ' + e); }
}
async function stopTraining() {
  if (!confirm('Stop training? (checkpoint will be saved)')) return;
  try {
    const r = await fetch(API + '/api/training/stop', {method:'POST'});
    const d = await r.json();
    alert(d.status || d.error);
  } catch(e) { alert('Error: ' + e); }
}
async function startModel() {
  const mid = $('modelSelect').value;
  if (!mid) return;
  const r = await fetch(API + '/api/training/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({model_id:mid})});
  const d = await r.json();
  alert(d.status + ' — PID ' + d.pid);
}

// ── Registry ──
async function loadRegistry() {
  try {
    const r = await fetch(API + '/api/registry');
    const d = await r.json();
    let rows = ''; let opts = '';
    Object.entries(d).forEach(([mid, cfg]) => {
      rows += `<tr><td>${mid}</td><td>${flag(cfg.use_wavelet_pooling)}</td><td>${flag(cfg.use_bigru)}</td><td>${flag(cfg.use_swin)}</td><td>${flag(cfg.use_attention_gates)}</td><td>${flag(cfg.use_mtl)}</td><td style="font-family:var(--mono)">${cfg.loss_config}</td></tr>`;
      opts += `<option value="${mid}">${mid}</option>`;
    });
    $('registryTable').innerHTML = rows;
    $('modelSelect').innerHTML = opts;
    $('snapModel').innerHTML = opts;
  } catch(e) {}
}

// ── Analysis: overfit/underfit ──
async function loadAnalysis() {
  try {
    const r = await fetch(API + '/api/analysis/fit');
    const d = await r.json();
    let rows = '';
    d.forEach(m => {
      let cls = 'badge-ok', label = 'Good fit';
      if (m.diagnosis === 'overfit') { cls = 'badge-err'; label = 'OVERFIT'; }
      else if (m.diagnosis === 'underfit') { cls = 'badge-warn'; label = 'UNDERFIT'; }
      else if (m.diagnosis === 'early') { cls = 'badge-warn'; label = 'Too early'; }
      rows += `<tr><td>${m.model_id}</td><td style="font-family:var(--mono)">${fmt(m.train_loss)}</td><td style="font-family:var(--mono)">${fmt(m.val_loss)}</td><td style="font-family:var(--mono)">${fmt(m.gap)}</td><td><span class="badge ${cls}">${label}</span></td><td style="font-size:11px;color:var(--text-dim)">${m.detail}</td></tr>`;
    });
    $('fitTable').innerHTML = rows || '<tr><td colspan="6" style="color:var(--text-dim)">No trained models yet</td></tr>';
  } catch(e) {}

  try {
    const r = await fetch(API + '/api/analysis/dataset');
    const d = await r.json();
    let html = '';
    if (d.synthetic) html += `<div style="margin-bottom:8px;"><span class="badge badge-ok">Synthetic</span> ${d.synthetic.count} windows, ${d.synthetic.size_mb} MB<br>SHA256: <span style="color:var(--cyan)">${d.synthetic.sha256?.slice(0,16)}...</span></div>`;
    if (d.real) html += `<div><span class="badge badge-ok">Real Q.Lin</span> ${d.real.count} windows, ${d.real.size_mb} MB<br>SHA256: <span style="color:var(--cyan)">${d.real.sha256?.slice(0,16)}...</span></div>`;
    if (!d.synthetic && !d.real) html = '<span class="badge badge-err">No test sets found</span>';
    $('datasetInfo').innerHTML = html;
    $('aTestSize').textContent = d.synthetic ? d.synthetic.count.toLocaleString() : '-';
    $('aRealSize').textContent = d.real ? d.real.count.toLocaleString() : '-';
  } catch(e) {}

  try {
    const r = await fetch(API + '/api/benchmark/all');
    const d = await r.json();
    let rows = '';
    d.forEach(b => {
      rows += `<tr><td>${b.model_id}</td><td>${b.params||'-'}</td><td>${fmt(b.ncc_synth)}</td><td>${fmt(b.ncc_real)}</td><td>${b.latency||'-'}</td></tr>`;
    });
    $('benchTable').innerHTML = rows || '<tr><td colspan="5" style="color:var(--text-dim)">Run benchmark first</td></tr>';
  } catch(e) {}
}

// ── Training charts ──
let lossChartObj = null, nccChartObj = null;
async function loadCharts() {
  try {
    const r = await fetch(API + '/api/analysis/curves');
    const d = await r.json();
    if (!d || !d.length) return;
    const latest = d[0];
    const epochs = Array.from({length: latest.train_loss.length}, (_, i) => i+1);

    if (lossChartObj) lossChartObj.destroy();
    lossChartObj = new Chart($('lossChart'), {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          {label:'Train Loss', data:latest.train_loss, borderColor:'#00d4ff', borderWidth:1.5, pointRadius:0, tension:0.3},
          {label:'Val Loss', data:latest.val_loss, borderColor:'#f59e0b', borderWidth:1.5, pointRadius:0, tension:0.3},
        ]
      },
      options: {responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}}, scales:{x:{ticks:{color:'#475569',font:{size:9}},grid:{color:'#1e293b'}},y:{ticks:{color:'#475569',font:{size:9}},grid:{color:'#1e293b'}}}}
    });

    if (nccChartObj) nccChartObj.destroy();
    nccChartObj = new Chart($('nccChart'), {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          {label:'Train NCC', data:latest.train_ncc, borderColor:'#22c55e', borderWidth:1.5, pointRadius:0, tension:0.3},
          {label:'Val NCC', data:latest.val_ncc, borderColor:'#00d4ff', borderWidth:1.5, pointRadius:0, tension:0.3},
        ]
      },
      options: {responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}}, scales:{x:{ticks:{color:'#475569',font:{size:9}},grid:{color:'#1e293b'}},y:{ticks:{color:'#475569',font:{size:9}},grid:{color:'#1e293b'}}}}
    });
  } catch(e) { console.error('Charts error', e); }
}

// ── Snapshot ──
async function loadSnapshot() {
  const mid = $('snapModel').value;
  if (!mid) return;
  $('snapStatus').textContent = 'Loading...';
  try {
    const r = await fetch(API + `/api/snapshot/${mid}`);
    if (r.ok) {
      const blob = await r.blob();
      $('snapImg').src = URL.createObjectURL(blob);
      $('snapStatus').textContent = `Snapshot for ${mid}`;
    } else { $('snapStatus').textContent = 'No snapshot available'; $('snapImg').src = ''; }
  } catch(e) { $('snapStatus').textContent = 'Error loading snapshot'; }
}

// ── Live log SSE ──
function startLogStream() {
  const es = new EventSource(API + '/api/live-log');
  es.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      const panel = $('logPanel');
      let cls = '';
      if (d.line.includes('[EPOCH]')) cls = 'log-epoch';
      else if (d.line.includes('[WARN]') || d.line.includes('[SKIP]')) cls = 'log-warn';
      else if (d.line.includes('[ERROR]') || d.line.includes('FAILED')) cls = 'log-err';
      panel.innerHTML += `<span class="${cls}">${d.line}</span>\n`;
      panel.scrollTop = panel.scrollHeight;
    } catch(ex) {}
  };
}
function clearLog() { $('logPanel').innerHTML = ''; }

// ── Signal preview ──
let sigChartObj = null;
async function loadSignalPreview() {
  try {
    const r = await fetch(API + '/api/analysis/signal-preview');
    const d = await r.json();
    if (!d.clean) return;
    const labels = Array.from({length: d.clean.length}, (_, i) => i);
    if (sigChartObj) sigChartObj.destroy();
    sigChartObj = new Chart($('signalChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {label:'Noisy', data:d.noisy, borderColor:'#ef4444', borderWidth:0.8, pointRadius:0},
          {label:'Clean', data:d.clean, borderColor:'#22c55e', borderWidth:1, pointRadius:0},
        ]
      },
      options: {responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}}, scales:{x:{display:false},y:{ticks:{color:'#475569',font:{size:9}},grid:{color:'#1e293b'}}}}
    });
  } catch(e) {}
}

// ── Boot ──
async function boot() {
  pollGPU(); setInterval(pollGPU, 5000);
  loadSession(); setInterval(loadSession, 8000);
  loadResume(); setInterval(loadResume, 15000);
  loadRegistry();
  loadAnalysis(); setInterval(loadAnalysis, 30000);
  loadCharts(); setInterval(loadCharts, 20000);
  loadSignalPreview();
  startLogStream();
}
boot();
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return Response(FRONTEND_HTML, mimetype="text/html")


# ── GPU ──
@app.route("/api/gpu")
def gpu_status():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        f = [x.strip() for x in out.split(",")]
        return jsonify({
            "name": f[0], "temp_c": int(f[1]), "utilization_pct": int(f[2]),
            "memory_used_mb": int(f[3]), "memory_total_mb": int(f[4]),
            "power_draw_w": float(f[5]) if f[5] != "[N/A]" else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Session ──
@app.route("/api/session")
def get_session():
    sf = PROJECT_ROOT / "session_state.json"
    if not sf.exists():
        return jsonify({"status": "no_session"})
    with open(sf) as f:
        return jsonify(json.load(f))


@app.route("/api/resume-prompt")
def get_resume_prompt():
    pf = PROJECT_ROOT / "RESUME_PROMPT.md"
    if not pf.exists():
        return jsonify({"content": "No resume prompt yet."})
    return jsonify({"content": pf.read_text(encoding="utf-8", errors="replace")})


# ── Registry ──
@app.route("/api/registry")
def get_registry():
    reg_path = CONFIG_DIR / "model_registry.yaml"
    if not reg_path.exists():
        return jsonify({})
    with open(reg_path) as f:
        return jsonify(yaml.safe_load(f).get("models", {}))


# ── Training control ──
_training_proc = None

@app.route("/api/training/start", methods=["POST"])
def start_training():
    global _training_proc
    if _training_proc and _training_proc.poll() is None:
        return jsonify({"status": "already_running", "pid": _training_proc.pid}), 409

    body = request.get_json() or {}
    mid = body.get("model_id", "MR-TAE-FULL")
    venv_python = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else str(PROJECT_ROOT / ".venv" / "bin" / "python")
    
    log_path = PROJECT_ROOT / "logs" / "training_live.log"
    log_path.parent.mkdir(exist_ok=True)
    log_fh = open(log_path, "a", buffering=1)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    cmd = [venv_python, "pipeline/mlflow_orchestrator.py", "--model", mid]
    _training_proc = subprocess.Popen(
        cmd, 
        cwd=str(PROJECT_ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT,
        creationflags=creationflags
    )
    return jsonify({"status": "started", "pid": _training_proc.pid, "model_id": mid, "log": str(log_path)})


@app.route("/api/training/resume", methods=["POST"])
def resume_training():
    global _training_proc
    if _training_proc and _training_proc.poll() is None:
        return jsonify({"status": "already_running", "pid": _training_proc.pid}), 409

    venv_python = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else str(PROJECT_ROOT / ".venv" / "bin" / "python")
    
    log_path = PROJECT_ROOT / "logs" / "training_live.log"
    log_path.parent.mkdir(exist_ok=True)
    log_fh = open(log_path, "a", buffering=1)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    entry = PROJECT_ROOT / "pipeline" / "autonomous_loop.py"
    if entry.exists():
        cmd = [venv_python, str(entry), "--resume"]
    else:
        cmd = [venv_python, "pipeline/mlflow_orchestrator.py", "--resume"]
    _training_proc = subprocess.Popen(
        cmd, 
        cwd=str(PROJECT_ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT,
        creationflags=creationflags
    )
    return jsonify({"status": "resumed", "pid": _training_proc.pid, "log": str(log_path)})


@app.route("/api/training/stop", methods=["POST"])
def stop_training():
    global _training_proc
    if _training_proc and _training_proc.poll() is None:
        _training_proc.terminate()
        return jsonify({"status": "stopped", "pid": _training_proc.pid})
    return jsonify({"error": "No training running"}), 400


# ── Config update ──
@app.route("/api/config/update", methods=["POST"])
def update_config():
    body = request.get_json() or {}
    tc_path = CONFIG_DIR / "training_config.yaml"
    if tc_path.exists():
        with open(tc_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    cfg.update(body)
    with open(tc_path, "w") as f:
        yaml.dump(cfg, f)
    return jsonify({"status": "ok", "config": cfg})


# ── Analysis: overfit/underfit detection ──
@app.route("/api/analysis/fit")
def analysis_fit():
    results = []
    for hist_path in sorted(RESULTS_DIR.glob("*/history.json")):
        mid = hist_path.parent.name
        try:
            with open(hist_path) as f:
                h = json.load(f)
            tl = h.get("train_loss", [])
            vl = h.get("val_loss", [])
            tn = h.get("train_ncc", [])
            vn = h.get("val_ncc", [])
            if len(tl) < 5:
                results.append({"model_id": mid, "train_loss": tl[-1] if tl else None,
                                "val_loss": vl[-1] if vl else None, "gap": 0,
                                "diagnosis": "early", "detail": f"Only {len(tl)} epochs"})
                continue

            last_tl = np.mean(tl[-10:])
            last_vl = np.mean(vl[-10:])
            gap = last_vl - last_tl

            last_tn = np.mean(tn[-10:]) if tn else 0
            last_vn = np.mean(vn[-10:]) if vn else 0
            ncc_gap = last_tn - last_vn

            if gap > 0.3 and ncc_gap > 0.15:
                diag = "overfit"
                detail = f"Val-Train loss gap={gap:.3f}, NCC gap={ncc_gap:.3f}. Consider dropout/augmentation."
            elif last_vn < 0.3 and len(tl) > 50:
                diag = "underfit"
                detail = f"Val NCC={last_vn:.3f} after {len(tl)} epochs. Model may need more capacity or LR tuning."
            else:
                diag = "good"
                detail = f"Val NCC={last_vn:.3f}, gap={gap:.3f}. Healthy training."

            results.append({
                "model_id": mid, "train_loss": round(last_tl, 4),
                "val_loss": round(last_vl, 4), "gap": round(gap, 4),
                "diagnosis": diag, "detail": detail,
            })
        except Exception:
            pass
    return jsonify(results)


# ── Analysis: training curves ──
@app.route("/api/analysis/curves")
def analysis_curves():
    curves = []
    for hist_path in sorted(RESULTS_DIR.glob("*/history.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(hist_path) as f:
                h = json.load(f)
            curves.append({
                "model_id": hist_path.parent.name,
                "train_loss": h.get("train_loss", []),
                "val_loss": h.get("val_loss", []),
                "train_ncc": h.get("train_ncc", []),
                "val_ncc": h.get("val_ncc", []),
            })
        except Exception:
            pass
    return jsonify(curves)


# ── Analysis: dataset integrity ──
@app.route("/api/analysis/dataset")
def analysis_dataset():
    result = {}
    synth = DATA_DIR / "test_synthetic_ae.pt"
    if synth.exists():
        data = torch.load(synth, map_location="cpu", weights_only=True)
        result["synthetic"] = {
            "count": int(data["noisy"].shape[0]),
            "size_mb": round(synth.stat().st_size / 1e6, 1),
            "sha256": hashlib.sha256(synth.read_bytes()).hexdigest(),
            "snr_levels": sorted(set(data["snr_labels"].numpy().round().astype(int).tolist())),
        }

    real = DATA_DIR / "test_real_qlin.pt"
    if real.exists():
        data = torch.load(real, map_location="cpu", weights_only=True)
        result["real"] = {
            "count": int(data["windows"].shape[0]),
            "size_mb": round(real.stat().st_size / 1e6, 1),
            "sha256": hashlib.sha256(real.read_bytes()).hexdigest(),
        }
    return jsonify(result)


# ── Analysis: signal preview ──
@app.route("/api/analysis/signal-preview")
def signal_preview():
    synth = DATA_DIR / "test_synthetic_ae.pt"
    if not synth.exists():
        return jsonify({})
    data = torch.load(synth, map_location="cpu", weights_only=True)
    idx = 42
    return jsonify({
        "noisy": data["noisy"][idx].tolist(),
        "clean": data["clean"][idx].tolist(),
        "snr": float(data["snr_labels"][idx]),
    })


# ── Benchmark results ──
@app.route("/api/benchmark/all")
def benchmark_all():
    results = []
    for jf in sorted(RESULTS_DIR.glob("*/benchmark_results.json")):
        try:
            with open(jf) as f:
                d = json.load(f)
            results.append({
                "model_id": jf.parent.name,
                "params": d.get("efficiency", {}).get("param_count_M"),
                "ncc_synth": d.get("synthetic", {}).get("ncc_mean"),
                "ncc_real": d.get("real_qlin", {}).get("ncc_mean"),
                "latency": d.get("efficiency", {}).get("inference_latency_ms_b1"),
            })
        except Exception:
            pass
    return jsonify(results)


# ── Snapshot images ──
@app.route("/api/snapshot/<model_id>")
def get_snapshot(model_id):
    plot_dir = RESULTS_DIR / model_id / "plots"
    if not plot_dir.exists():
        return jsonify({"error": "no plots"}), 404
    pngs = sorted(plot_dir.glob("snapshot_epoch*.png"))
    if not pngs:
        return jsonify({"error": "no snapshots"}), 404
    return send_file(str(pngs[-1]), mimetype="image/png")


# ── Live log SSE ──
@app.route("/api/live-log")
def live_log_stream():
    log_file = LOGS_DIR / "training_live.log"
    log_file.touch(exist_ok=True)

    def generate():
        if log_file.exists():
            lines = log_file.read_text(errors="replace").splitlines()
            for line in lines[-80:]:
                yield f"data: {json.dumps({'line': line})}\n\n"
        with open(log_file, errors="replace") as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
                else:
                    time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Session note ──
@app.route("/api/session/note", methods=["POST"])
def add_note():
    body = request.get_json() or {}
    note = body.get("note", "").strip()
    if not note:
        return jsonify({"error": "note required"}), 400
    sf = PROJECT_ROOT / "session_state.json"
    if sf.exists():
        with open(sf) as f:
            s = json.load(f)
        s.setdefault("notes", []).append({"time": datetime.now().isoformat(), "note": note})
        with open(sf, "w") as f:
            json.dump(s, f, indent=2)
    return jsonify({"status": "ok"})


# ── Files API ──
@app.route("/api/files/<path:filepath>")
def serve_file(filepath):
    full = PROJECT_ROOT / filepath
    if not full.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(str(full))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  AE-PD Research Command Centre")
    print("  http://localhost:1001")
    print("=" * 60)
    app.run(host="0.0.0.0", port=1001, debug=False, threaded=True)
