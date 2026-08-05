"""Interactive AWR reward/weight explorer.

Extract per-frame percent/stock traces from a folder of .slp games and write one
self-contained HTML page. The page computes rewards, discounted returns, and AWR
weights in the browser, so the knobs (gamma, beta, stock value, damage weight,
w_max, baseline) respond instantly. Each chart links into a local slippilab
viewer, so you can click a weight spike and watch that moment as gameplay.

Run:
    uv run notebooks/awr_explorer.py
Then:
    cd ~/src/slippilab && npm run dev      # serves the viewer and the .slp files
    xdg-open ~/data/scratch/awr-explorer/explorer.html

Page units are percent: damage counts 1 per percent, one stock is worth the
"stock value" slider (Eric's 200:1 idea = default 200). The footer shows the
exact TrainConfig equivalents (code units use stock = +-1).
"""

# %%
import json
from pathlib import Path

import melee
import numpy as np

from hal.data.extract import extract_replay

try:
    from pyo3_runtime import PanicException  # peppi surfaces torn files as Rust panics
except ImportError:  # older peppi builds raise plain Exceptions instead

    class PanicException(Exception):  # type: ignore[no-redef]
        pass


SLPS_DIR = Path.home() / "data/scratch/awr-explorer/slps"
OUT_HTML = Path.home() / "data/scratch/awr-explorer/explorer.html"
SLIPPILAB_PUBLIC = Path.home() / "src/slippilab/public/hal-awr"
SLIPPILAB_BASE = "http://localhost:5173"

# %%
games: list[dict] = []
for slp in sorted(SLPS_DIR.glob("*.slp")):
    try:
        sample = extract_replay(str(slp))
    except PanicException as e:
        print(f"skip {slp.name}: peppi panic {e}")
        continue
    if sample is None:
        print(f"skip {slp.name}: peppi could not parse")
        continue
    frame = sample["frame"]
    if int(frame[-1]) - int(frame[0]) + 1 != len(frame):
        print(f"skip {slp.name}: frame ids not contiguous after dedup")
        continue
    games.append(
        {
            "file": slp.name,
            "source": "model-vs-model" if slp.name.startswith("h2h-") else "human (ranked)",
            "tier": slp.stem.rsplit("-", 1)[0] if not slp.name.startswith("h2h-") else slp.stem,
            "p1": melee.Character(int(sample["p1_character"][0])).name,
            "p2": melee.Character(int(sample["p2_character"][0])).name,
            "stage": melee.Stage(int(sample["stage"][0])).name,
            "frame0": int(frame[0]),  # -123: array index 0 is the first slp frame, slippilab's index origin
            "note": f"final {int(sample['p1_stock'][-1])}–{int(sample['p2_stock'][-1])}",
            "p1Percent": [round(float(v), 1) for v in sample["p1_percent"]],
            "p2Percent": [round(float(v), 1) for v in sample["p2_percent"]],
            "p1Stock": [int(v) for v in sample["p1_stock"]],
            "p2Stock": [int(v) for v in sample["p2_stock"]],
        }
    )
    print(f"ok   {slp.name}: {len(frame)} frames, {games[-1]['p1']} vs {games[-1]['p2']} on {games[-1]['stage']}")

if not games:
    raise SystemExit(f"no usable .slp files under {SLPS_DIR}")

# %%
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AWR reward/weight explorer</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; margin: 0; }
body { background:#111417; color:#d7dde3; font:14px/1.45 system-ui, sans-serif; padding-bottom:4rem; }
a { color:#4ea1ff; text-decoration:none; } a:hover { text-decoration:underline; }
code { background:#1a1f24; padding:1px 5px; border-radius:4px; font-size:13px; }
#knobs { position:sticky; top:0; z-index:5; background:#161a1e; border-bottom:1px solid #2a3138;
         padding:10px 18px; display:flex; flex-wrap:wrap; gap:6px 26px; align-items:center; }
#knobs .k { display:flex; flex-direction:column; min-width:150px; }
#knobs label { font-size:12px; color:#8b97a1; }
#knobs output { font-variant-numeric: tabular-nums; font-size:13px; }
#knobs input[type=range] { width:150px; accent-color:#4ea1ff; }
#stats { padding:10px 18px; display:flex; flex-wrap:wrap; gap:6px 30px; background:#13171b;
         border-bottom:1px solid #2a3138; font-variant-numeric: tabular-nums; }
#stats b { color:#fff; font-weight:600; }
#stats .lbl { color:#8b97a1; font-size:12px; display:block; }
h1 { font-size:18px; padding:14px 18px 4px; }
.sub { color:#8b97a1; padding:0 18px 8px; font-size:13px; }
.game { margin:14px 18px; background:#1a1f24; border:1px solid #262d33; border-radius:8px; padding:10px 14px; }
.game h2 { font-size:14px; font-weight:600; display:flex; flex-wrap:wrap; gap:4px 14px; align-items:baseline; }
.game h2 .meta { color:#8b97a1; font-weight:400; font-size:12px; }
.game canvas { width:100%; height:190px; display:block; margin-top:6px; cursor:crosshair; }
.egobtn { background:#242b32; color:#d7dde3; border:1px solid #38424b; border-radius:5px;
          padding:1px 8px; font-size:12px; cursor:pointer; }
.egobtn:hover { background:#2c343c; }
#tip { position:fixed; pointer-events:none; background:#0c0f11ee; border:1px solid #38424b; border-radius:6px;
       padding:6px 9px; font-size:12px; display:none; z-index:10; font-variant-numeric:tabular-nums; }
.legend { display:flex; gap:16px; font-size:12px; color:#8b97a1; padding:0 18px; flex-wrap:wrap; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; }
footer { margin:20px 18px; color:#8b97a1; font-size:13px; }
</style>
</head>
<body>
<h1>AWR reward/weight explorer</h1>
<div class="sub">Units are percent. One stock = the stock-value slider. w = clip(exp(A/&beta;), w_max),
 A = G &minus; baseline, G = discounted return of (damage dealt &minus; damage taken + stock events + match win).
 The match-win value adds ON TOP of the last stock's value, and only when a stock count reaches 0
 (a game that ends by quit-out has no win event). Click a chart to open that moment in slippilab.</div>
<div id="knobs">
  <div class="k"><label>discount half-life (s)</label>
    <input id="hl" type="range" min="-0.52" max="1.48" step="0.01" value="0.06">
    <output id="hlv"></output></div>
  <div class="k"><label>stock value (percent)</label>
    <input id="sv" type="range" min="0" max="500" step="10" value="200">
    <output id="svv"></output></div>
  <div class="k"><label>match-win value (percent)</label>
    <input id="wv" type="range" min="0" max="2000" step="50" value="600">
    <output id="wvv"></output></div>
  <div class="k"><label>damage weight</label>
    <input id="dw" type="range" min="0" max="2" step="0.05" value="1">
    <output id="dwv"></output></div>
  <div class="k"><label>&beta; (percent)</label>
    <input id="beta" type="range" min="0" max="3" step="0.02" value="1.7">
    <output id="betav"></output></div>
  <div class="k"><label>w_max</label>
    <input id="wmax" type="range" min="0.3" max="2.3" step="0.05" value="1.3">
    <output id="wmaxv"></output></div>
  <div class="k"><label>baseline</label>
    <select id="base">
      <option value="none">none (raw G)</option>
      <option value="global" selected>global mean of G</option>
      <option value="ema">causal EMA of G, 10 s (V-proxy)</option>
    </select>
    <output>&nbsp;</output></div>
</div>
<div id="stats"></div>
<div class="legend">
  <span><i style="background:#4caf7a"></i>damage dealt / opp death &#9650;</span>
  <span><i style="background:#e05555"></i>damage taken / ego death &#9660;</span>
  <span><i style="background:#4ea1ff"></i>G (return)</span>
  <span><i style="background:#8b97a1"></i>baseline</span>
  <span><i style="background:#ffb347"></i>w (log scale, line at w=1)</span>
  <span>| faint verticals = 256-frame training windows</span>
</div>
<div id="games"></div>
<footer id="foot"></footer>
<div id="tip"></div>
<script>
"use strict";
const DATA = __DATA__;
const BASE = "__SLIPPILAB_BASE__";
const FPS = 60, WIN = 256;

for (const g of DATA) { g.ego = 1; g.n = g.p1Percent.length; }

// --- reward math (mirrors hal: stock_loss_events / damage_taken semantics) ---
function deaths(stock) {
  const out = new Float32Array(stock.length);
  for (let t = 1; t < stock.length; t++) out[t] = stock[t] < stock[t-1] ? 1 : 0;
  return out;
}
function dmgTaken(pct) {
  const out = new Float32Array(pct.length);
  for (let t = 1; t < pct.length; t++) { const d = pct[t] - pct[t-1]; out[t] = d > 0 ? d : 0; }
  return out;
}
function returns(r, gamma) {
  const G = new Float32Array(r.length); let acc = 0;
  for (let t = r.length - 1; t >= 0; t--) { acc = r[t] + gamma * acc; G[t] = acc; }
  return G;
}
function emaBaseline(G, halfLifeS) {
  const a = 1 - Math.pow(0.5, 1 / (halfLifeS * FPS));
  const out = new Float32Array(G.length); let m = G[0];
  for (let t = 0; t < G.length; t++) { out[t] = m; m += a * (G[t] - m); }  // causal: uses past G only
  return out;
}

function knobs() {
  return {
    hl: Math.pow(10, +el.hl.value),
    S: +el.sv.value, W: +el.wv.value, d: +el.dw.value,
    beta: Math.pow(10, +el.beta.value),
    wmax: Math.pow(10, +el.wmax.value),
    base: el.base.value,
  };
}

function computeAll() {
  const k = knobs();
  const gamma = Math.pow(0.5, 1 / (k.hl * FPS));
  // pass 1: rewards + returns
  for (const g of DATA) {
    const egoP = g.ego === 1 ? g.p1Percent : g.p2Percent, oppP = g.ego === 1 ? g.p2Percent : g.p1Percent;
    const egoS = g.ego === 1 ? g.p1Stock : g.p2Stock, oppS = g.ego === 1 ? g.p2Stock : g.p1Stock;
    g.dealt = dmgTaken(oppP); g.taken = dmgTaken(egoP);
    g.oppDeath = deaths(oppS); g.egoDeath = deaths(egoS);
    // Match point: the stock event that empties a player's stock count. The win value W rides on top
    // of the ordinary stock value S, so the last stock is worth S + W and every other stock S.
    g.oppWin = new Float32Array(g.n); g.egoWin = new Float32Array(g.n);
    for (let t = 0; t < g.n; t++) {
      if (g.oppDeath[t] && oppS[t] === 0) g.oppWin[t] = 1;
      if (g.egoDeath[t] && egoS[t] === 0) g.egoWin[t] = 1;
    }
    const r = new Float32Array(g.n);
    for (let t = 0; t < g.n; t++)
      r[t] = k.d * (g.dealt[t] - g.taken[t]) + k.S * (g.oppDeath[t] - g.egoDeath[t])
           + k.W * (g.oppWin[t] - g.egoWin[t]);
    g.G = returns(r, gamma);
  }
  // pass 2: baseline (global mean over HUMAN games only, so stats match the training data)
  const human = DATA.filter(g => g.source.startsWith("human"));
  let mean = 0, n = 0;
  for (const g of human) { for (let t = 0; t < g.n; t++) mean += g.G[t]; n += g.n; }
  mean /= Math.max(n, 1);
  for (const g of DATA) {
    g.B = k.base === "none" ? new Float32Array(g.n)
        : k.base === "global" ? new Float32Array(g.n).fill(mean)
        : emaBaseline(g.G, 10);
    g.w = new Float32Array(g.n); g.clipped = 0;
    for (let t = 0; t < g.n; t++) {
      const w = Math.exp((g.G[t] - g.B[t]) / k.beta);
      g.w[t] = Math.min(w, k.wmax);
      if (w >= k.wmax) g.clipped++;
    }
  }
  // per-game ESS for every game; pooled stats over human games only (ESS is scale-free)
  let sw = 0, sw2 = 0, clip = 0, N = 0;
  const winMeans = []; let withinAcc = 0, withinN = 0;
  for (const g of DATA) {
    let gs = 0, gs2 = 0;
    for (let t = 0; t < g.n; t++) { gs += g.w[t]; gs2 += g.w[t] * g.w[t]; }
    g.ess = (gs * gs) / (g.n * gs2);
    if (!g.source.startsWith("human")) continue;
    sw += gs; sw2 += gs2; clip += g.clipped; N += g.n;
    for (let s = 0; s + WIN <= g.n; s += WIN) {
      let m = 0; for (let t = s; t < s + WIN; t++) m += g.w[t]; m /= WIN;
      let v = 0; for (let t = s; t < s + WIN; t++) v += (g.w[t] - m) ** 2; v /= WIN;
      winMeans.push(m); withinAcc += v; withinN++;
    }
  }
  const mb = winMeans.reduce((a, b) => a + b, 0) / Math.max(winMeans.length, 1);
  const between = winMeans.reduce((a, b) => a + (b - mb) ** 2, 0) / Math.max(winMeans.length, 1);
  const within = withinAcc / Math.max(withinN, 1);
  const cross = k.S > 20 ? Math.log(k.S / 20) / Math.log(1 / gamma) / FPS : 0;
  const crossW = k.S + k.W > 20 ? Math.log((k.S + k.W) / 20) / Math.log(1 / gamma) / FPS : 0;
  el.stats.innerHTML =
    stat("ESS (human pool)", ((sw * sw) / (N * sw2)).toFixed(3)) +
    stat("frames at w_max", (100 * clip / N).toFixed(2) + "%") +
    stat("weight variance between 256-frame windows", (100 * between / (between + within)).toFixed(1) + "%") +
    stat("a kill outweighs a 20% hit up to", cross ? cross.toFixed(1) + " s before it" : "never") +
    stat("match point outweighs it up to", crossW ? crossW.toFixed(1) + " s before it" : "never") +
    stat("gamma / frame", gamma.toFixed(5));
  el.hlv.textContent = k.hl.toFixed(2) + " s  (horizon ~" + (1 / (1 - gamma) / FPS).toFixed(1) + " s)";
  el.svv.textContent = k.S + "%"; el.wvv.textContent = k.W + "%"; el.dwv.textContent = k.d.toFixed(2);
  el.betav.textContent = k.beta.toFixed(1) + "%"; el.wmaxv.textContent = k.wmax.toFixed(1);
  syncEss();
  el.foot.innerHTML = "TrainConfig equivalents (code stock = &plusmn;1): <code>awr_gamma=" + gamma.toFixed(5) +
    "</code> <code>awr_damage_shaping=" + (k.S ? (k.d / k.S).toFixed(5) : "n/a (stock value 0)") +
    "</code> <code>awr_beta=" + (k.S ? (k.beta / k.S).toFixed(4) : "n/a") +
    "</code> <code>awr_weight_max=" + k.wmax.toFixed(1) +
    "</code> <code>awr_win_reward=" + (k.S ? (k.W / k.S).toFixed(2) : "n/a") + "</code> (win knob is NEW" +
    " — not in 020_awr.py yet)" +
    "<br>Baseline note: the training run learns V(s) from the trunk; this page uses analytic baselines" +
    " because V depends on the knobs. &beta; here is directly comparable only at matched baseline quality.";
  for (const g of DATA) draw(g);
}
function stat(lbl, val) { return '<span><span class="lbl">' + lbl + '</span><b>' + val + "</b></span>"; }

// --- drawing ---
const DPR = window.devicePixelRatio || 1;
function draw(g) {
  const c = g.canvas, W = c.clientWidth, H = 190;
  if (c.width !== W * DPR) { c.width = W * DPR; c.height = H * DPR; }
  const x = c.getContext("2d"); x.setTransform(DPR, 0, 0, DPR, 0, 0);
  x.clearRect(0, 0, W, H);
  const px = t => t / g.n * W;
  const B0 = 38, B1 = 110, B2 = 186;            // band bottoms: damage/events, G, w
  // window + time grid
  x.strokeStyle = "#20262c"; x.beginPath();
  for (let s = WIN; s < g.n; s += WIN) { const X = px(s); x.moveTo(X, 0); x.lineTo(X, H); }
  x.stroke();
  x.fillStyle = "#5c6870"; x.font = "10px system-ui";
  for (let s = 0; s < g.n; s += 30 * FPS) {
    const mm = Math.floor(s / FPS / 60), ss = Math.floor(s / FPS) % 60;
    x.fillText(mm + ":" + String(ss).padStart(2, "0"), px(s) + 2, H - 2);
  }
  // damage bars, binned per pixel: dealt grows up from the midline, taken grows down
  const mid = B0 - 16;
  let dmax = 1e-6;
  const up = new Float32Array(W + 1), dn = new Float32Array(W + 1);
  for (let t = 0; t < g.n; t++) {
    const X = Math.floor(px(t));
    up[X] += g.dealt[t]; dn[X] += g.taken[t];
  }
  for (let X = 0; X <= W; X++) dmax = Math.max(dmax, up[X], dn[X]);
  for (let X = 0; X <= W; X++) {
    if (up[X] > 0) { x.fillStyle = "#4caf7a"; const h = up[X] / dmax * 14; x.fillRect(X, mid - h, 1, h); }
    if (dn[X] > 0) { x.fillStyle = "#e05555"; const h = dn[X] / dmax * 14; x.fillRect(X, mid, 1, h); }
  }
  // stock events as triangles; the match point gets a ring around its triangle
  for (let t = 0; t < g.n; t++) {
    if (g.oppDeath[t]) tri(x, px(t), 6, 1, "#4caf7a");
    if (g.egoDeath[t]) tri(x, px(t), B0 - 4, -1, "#e05555");
    if (g.oppWin[t] || g.egoWin[t]) {
      x.strokeStyle = g.oppWin[t] ? "#4caf7a" : "#e05555"; x.lineWidth = 1.5;
      x.beginPath(); x.arc(px(t), g.oppWin[t] ? 6 : B0 - 4, 8, 0, 2 * Math.PI); x.stroke(); x.lineWidth = 1;
    }
  }
  // G band
  let gmin = Infinity, gmax = -Infinity;
  for (let t = 0; t < g.n; t++) { if (g.G[t] < gmin) gmin = g.G[t]; if (g.G[t] > gmax) gmax = g.G[t]; }
  for (let t = 0; t < g.n; t++) { if (g.B[t] < gmin) gmin = g.B[t]; if (g.B[t] > gmax) gmax = g.B[t]; }
  if (gmax - gmin < 1e-6) { gmax += 1; gmin -= 1; }
  const gy = v => B1 - (v - gmin) / (gmax - gmin) * (B1 - B0 - 8);
  if (gmin < 0 && gmax > 0) { x.strokeStyle = "#2a3138"; x.beginPath(); x.moveTo(0, gy(0)); x.lineTo(W, gy(0)); x.stroke(); }
  line(x, g.B, gy, g.n, W, "#8b97a1", true);
  line(x, g.G, gy, g.n, W, "#4ea1ff", false);
  // w band, log scale, filled from the w=1 line
  const k = knobs();
  const lmax = Math.log(k.wmax), lmin = -lmax;
  const wy = v => { const l = Math.max(lmin, Math.min(lmax, Math.log(Math.max(v, 1e-9)))); return B2 - 6 - (l - lmin) / (lmax - lmin) * (B2 - B1 - 16); };
  const y1 = wy(1);
  x.strokeStyle = "#3a4149"; x.beginPath(); x.moveTo(0, y1); x.lineTo(W, y1); x.stroke();
  x.fillStyle = "#ffb34733"; x.strokeStyle = "#ffb347"; x.beginPath();
  x.moveTo(0, y1);
  const step = Math.max(1, Math.floor(g.n / (W * 2)));
  for (let t = 0; t < g.n; t += step) x.lineTo(px(t), wy(g.w[t]));
  x.lineTo(W, y1); x.closePath(); x.fill(); x.stroke();
  g.geom = { gy, wy, gmin, gmax };
}
function tri(x, X, Y, dir, color) {
  x.fillStyle = color; x.beginPath();
  x.moveTo(X, Y + 5 * dir); x.lineTo(X - 4, Y - 4 * dir); x.lineTo(X + 4, Y - 4 * dir); x.closePath(); x.fill();
}
function line(x, arr, ymap, n, W, color, dashed) {
  x.strokeStyle = color; x.setLineDash(dashed ? [4, 4] : []);
  x.beginPath();
  const step = Math.max(1, Math.floor(n / (W * 2)));
  for (let t = 0; t < n; t += step) { const X = t / n * W; t === 0 ? x.moveTo(X, ymap(arr[t])) : x.lineTo(X, ymap(arr[t])); }
  x.stroke(); x.setLineDash([]);
}

// --- page assembly ---
const el = {};
for (const id of ["hl", "sv", "wv", "dw", "beta", "wmax", "base", "hlv", "svv", "wvv", "dwv", "betav", "wmaxv", "stats", "foot"])
  el[id] = document.getElementById(id);
const tip = document.getElementById("tip");
const holder = document.getElementById("games");
for (const g of DATA) {
  const div = document.createElement("div"); div.className = "game";
  const url = f => BASE + "/?replayUrl=" + encodeURIComponent(BASE + "/hal-awr/" + g.file) + "#" + f;
  div.innerHTML = '<h2><span class="who"></span>' +
    '<button class="egobtn"></button>' +
    '<span class="meta">' + g.stage + " &middot; " + g.source + " &middot; " + g.note +
    " &middot; ESS <span class='ess'></span></span>" +
    '<a target="_blank" href="' + url(0) + '">watch in slippilab</a></h2>';
  const canvas = document.createElement("canvas"); div.appendChild(canvas);
  holder.appendChild(div); g.canvas = canvas; g.div = div;
  const btn = div.querySelector(".egobtn");
  const sync = () => {
    div.querySelector(".who").textContent =
      (g.ego === 1 ? g.p1 + " vs " + g.p2 : g.p2 + " vs " + g.p1);
    btn.textContent = "ego: " + (g.ego === 1 ? g.p1 + " (P1)" : g.p2 + " (P2)");
  };
  btn.onclick = () => { g.ego = g.ego === 1 ? 2 : 1; sync(); computeAll(); };
  sync();
  canvas.addEventListener("mousemove", ev => {
    const r = canvas.getBoundingClientRect();
    const t = Math.max(0, Math.min(g.n - 1, Math.floor((ev.clientX - r.left) / r.width * g.n)));
    const s = t / FPS;
    tip.style.display = "block";
    tip.style.left = (ev.clientX + 14) + "px"; tip.style.top = (ev.clientY + 14) + "px";
    tip.innerHTML = Math.floor(s / 60) + ":" + String(Math.floor(s) % 60).padStart(2, "0") +
      " (frame " + t + ")<br>G = " + g.G[t].toFixed(1) + " &middot; w = " + g.w[t].toFixed(2) +
      "<br>ego " + (g.ego === 1 ? g.p1Percent[t] : g.p2Percent[t]) + "% (" + (g.ego === 1 ? g.p1Stock[t] : g.p2Stock[t]) +
      " st) &middot; opp " + (g.ego === 1 ? g.p2Percent[t] : g.p1Percent[t]) + "% (" +
      (g.ego === 1 ? g.p2Stock[t] : g.p1Stock[t]) + " st)<br><i>click to open in slippilab</i>";
  });
  canvas.addEventListener("mouseleave", () => tip.style.display = "none");
  canvas.addEventListener("click", ev => {
    const r = canvas.getBoundingClientRect();
    const t = Math.max(0, Math.floor((ev.clientX - r.left) / r.width * g.n));
    window.open(url(t), "_blank");
  });
}
function syncEss() { for (const g of DATA) g.div.querySelector(".ess").textContent = g.ess.toFixed(3); }
for (const id of ["hl", "sv", "wv", "dw", "beta", "wmax"]) el[id].addEventListener("input", computeAll);
el.base.addEventListener("change", computeAll);
window.addEventListener("resize", () => { for (const g of DATA) draw(g); });
computeAll();
</script>
</body>
</html>
"""

OUT_HTML.write_text(
    HTML.replace("__DATA__", json.dumps(games, separators=(",", ":"))).replace("__SLIPPILAB_BASE__", SLIPPILAB_BASE)
)
print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB, {len(games)} games)")

# %%
if not SLIPPILAB_PUBLIC.exists():
    SLIPPILAB_PUBLIC.symlink_to(SLPS_DIR)
print(f"slippilab mount: {SLIPPILAB_PUBLIC} -> {SLPS_DIR}")
print(f"start the viewer:  cd ~/src/slippilab && npm run dev   (then open {OUT_HTML})")
