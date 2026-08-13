"""Generate an interactive replay-based explorer for chunk-IQL stability.

The page combines real reward/return traces from Slippi replays with a deliberately
small local critic model.  For a selected replay prefix it visualizes

    Q_target = R_4 + gamma**4 * V

and decomposes the value residual into the real four-frame reward, Bellman
contraction, expectile uplift from noisy Q estimates, and categorical-support
clipping.  The replay labels are real; the repeated-prefix critic trajectory is
an explanatory model, not a reconstruction of a trained checkpoint.

Run:
    uv run notebooks/iql_bellman_explorer.py

Then start Slippilab if desired and open the generated page:
    cd ~/src/slippilab && npm run dev
    xdg-open ~/data/scratch/iql-bellman-explorer/explorer.html
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import melee
import numpy as np

from hal.data.extract import extract_replay
from hal.wire import MASK_INT32

try:
    from pyo3_runtime import PanicException  # peppi surfaces torn files as Rust panics
except ImportError:  # older peppi builds raise plain Exceptions instead

    class PanicException(Exception):  # type: ignore[no-redef]
        pass


DEFAULT_ROOT = Path.home() / "data/scratch/iql-bellman-explorer"
DEFAULT_SLPS_DIR = Path.home() / "data/scratch/awr-explorer/slps"
DEFAULT_OUT_HTML = DEFAULT_ROOT / "explorer.html"
DEFAULT_SLIPPILAB_PUBLIC = Path.home() / "src/slippilab/public/hal-iql"
DEFAULT_SLIPPILAB_BASE = "http://localhost:5173"
EXECUTED_CHUNK = 4


def stock_loss_events(stock: np.ndarray) -> np.ndarray:
    ids = np.asarray(stock).astype(np.int64)
    known = ids != MASK_INT32
    out = np.zeros(ids.shape, dtype=np.float32)
    out[1:] = ((ids[1:] < ids[:-1]) & known[1:] & known[:-1]).astype(np.float32)
    return out


def damage_taken(percent: np.ndarray) -> np.ndarray:
    values = np.asarray(percent, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.float32)
    out[1:] = np.maximum(values[1:] - values[:-1], 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def frame_reward(
    sample: dict,
    *,
    ego: str,
    opp: str,
    damage_shaping: float = 0.01,
    win_reward: float = 0.5,
) -> np.ndarray:
    """Mirror experiment 027's shaped stock reward exactly."""
    opp_loss = stock_loss_events(sample[f"{opp}_stock"])
    ego_loss = stock_loss_events(sample[f"{ego}_stock"])
    reward = opp_loss - ego_loss
    if win_reward:
        opp_final = opp_loss * (np.asarray(sample[f"{opp}_stock"]) == 0)
        ego_final = ego_loss * (np.asarray(sample[f"{ego}_stock"]) == 0)
        reward = reward + win_reward * (opp_final - ego_final)
    if damage_shaping:
        damage = damage_taken(sample[f"{opp}_percent"]) - damage_taken(sample[f"{ego}_percent"])
        reward = reward + damage_shaping * damage
    return reward.astype(np.float32)


def discounted_returns(reward: np.ndarray, gamma: float) -> np.ndarray:
    values = np.asarray(reward, dtype=np.float64)
    out = np.empty(values.shape, dtype=np.float64)
    acc = 0.0
    for index in range(len(values) - 1, -1, -1):
        acc = float(values[index]) + gamma * acc
        out[index] = acc
    return out.astype(np.float32)


def four_frame_rewards(reward: np.ndarray, gamma: float) -> np.ndarray:
    """Return the 027-aligned reward sum r[t+1] through r[t+4]."""
    values = np.asarray(reward, dtype=np.float64)
    count = max(0, len(values) - EXECUTED_CHUNK)
    out = np.zeros(count, dtype=np.float64)
    discounts = gamma ** np.arange(EXECUTED_CHUNK)
    for prefix in range(count):
        out[prefix] = float(values[prefix + 1 : prefix + 1 + EXECUTED_CHUNK] @ discounts)
    return out.astype(np.float32)


def four_frame_contraction(gamma: float) -> float:
    return 1.0 - gamma**EXECUTED_CHUNK


def _rounded(values: np.ndarray, digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def replay_record(path: Path, sample: dict) -> dict:
    frame = np.asarray(sample["frame"])
    if len(frame) < EXECUTED_CHUNK + 2:
        raise ValueError("replay is too short for a four-frame transition")
    if int(frame[-1]) - int(frame[0]) + 1 != len(frame):
        raise ValueError("frame ids are not contiguous after deduplication")
    return {
        "file": path.name,
        "source": "model-vs-model" if path.name.startswith("h2h-") else "human (ranked)",
        "tier": path.stem.rsplit("-", 1)[0] if not path.name.startswith("h2h-") else path.stem,
        "p1": melee.Character(int(sample["p1_character"][0])).name,
        "p2": melee.Character(int(sample["p2_character"][0])).name,
        "stage": melee.Stage(int(sample["stage"][0])).name,
        "frame0": int(frame[0]),
        "note": f"final {int(sample['p1_stock'][-1])}–{int(sample['p2_stock'][-1])}",
        "p1Percent": _rounded(np.asarray(sample["p1_percent"]), 1),
        "p2Percent": _rounded(np.asarray(sample["p2_percent"]), 1),
        "p1Stock": [int(value) for value in sample["p1_stock"]],
        "p2Stock": [int(value) for value in sample["p2_stock"]],
    }


def extract_games(slps_dir: Path) -> list[dict]:
    games: list[dict] = []
    for slp in sorted(slps_dir.glob("*.slp")):
        try:
            sample = extract_replay(str(slp))
        except PanicException as error:
            print(f"skip {slp.name}: peppi panic {error}")
            continue
        if sample is None:
            print(f"skip {slp.name}: peppi could not parse")
            continue
        try:
            record = replay_record(slp, sample)
        except ValueError as error:
            print(f"skip {slp.name}: {error}")
            continue
        games.append(record)
        print(f"ok   {slp.name}: {len(sample['frame'])} frames, {record['p1']} vs {record['p2']} on {record['stage']}")
    return games


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IQL Bellman push/pull explorer</title>
<style>
:root { color-scheme: dark; --bg:#111417; --panel:#1a1f24; --line:#303840; --muted:#8b97a1;
  --text:#d7dde3; --blue:#4ea1ff; --orange:#ffb347; --green:#4caf7a; --red:#e05555; --purple:#b783ff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; padding-bottom:3rem; }
button,select,input { font:inherit; }
button,select { color:var(--text); background:#242b32; border:1px solid #38424b; border-radius:5px; padding:4px 8px; }
button { cursor:pointer; } button:hover { background:#2c343c; }
a { color:var(--blue); text-decoration:none; } a:hover { text-decoration:underline; }
code { background:#1a1f24; padding:1px 5px; border-radius:4px; }
h1 { margin:0; font-size:20px; font-weight:600; }
h2 { margin:0 0 6px; font-size:15px; font-weight:600; }
.top { position:sticky; top:0; z-index:5; background:#161a1ef5; border-bottom:1px solid #2a3138; }
.title { padding:12px 18px 4px; display:flex; gap:14px; flex-wrap:wrap; align-items:baseline; }
.subtitle { color:var(--muted); font-size:12px; }
.controls { padding:8px 18px 12px; display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:8px 20px; }
.control { min-width:0; display:flex; flex-direction:column; }
.control label { color:var(--muted); font-size:12px; display:flex; justify-content:space-between; gap:6px; }
.control output { color:var(--text); font-variant-numeric:tabular-nums; white-space:nowrap; }
.control input[type=range] { width:100%; accent-color:var(--blue); }
.actions { display:flex; align-items:end; gap:8px; }
.content { padding:14px 18px; max-width:1500px; margin:auto; }
.replay-row { display:grid; grid-template-columns:minmax(250px,1fr) auto auto; gap:10px; align-items:center; margin-bottom:8px; }
.replay-row select { width:100%; min-width:0; }
.meta { color:var(--muted); font-size:12px; margin-bottom:8px; }
.plot { width:100%; display:block; background:var(--panel); border:1px solid #262d33; border-radius:7px; }
#timeline { height:260px; cursor:crosshair; }
.prefix-control { display:flex; align-items:center; gap:10px; margin:7px 0 18px; }
.prefix-control label { color:var(--muted); white-space:nowrap; font-size:12px; }
.prefix-control input { width:100%; accent-color:var(--blue); }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1px; margin:10px 0 18px;
  background:#262d33; border:1px solid #262d33; border-radius:7px; overflow:hidden; }
.stat { background:var(--panel); padding:8px 10px; min-height:58px; }
.stat .label { color:var(--muted); font-size:11px; display:block; }
.stat b { color:#fff; font-weight:600; font-variant-numeric:tabular-nums; display:block; }
.stat small { color:var(--muted); font-size:11px; }
.panels { display:grid; grid-template-columns:minmax(280px,.8fr) minmax(420px,1.4fr); gap:16px; }
.panel canvas { height:290px; }
.legend { display:flex; flex-wrap:wrap; gap:5px 16px; color:var(--muted); font-size:12px; margin:5px 0 8px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; }
.explain { margin-top:16px; color:var(--muted); font-size:12px; }
.explain strong { color:var(--text); font-weight:600; }
#tip { position:fixed; pointer-events:none; display:none; z-index:10; background:#0c0f11ee; border:1px solid #38424b;
  border-radius:6px; padding:6px 9px; font-size:12px; font-variant-numeric:tabular-nums; }
@media (max-width:850px) { .panels { grid-template-columns:1fr; } .replay-row { grid-template-columns:1fr 1fr; }
  .replay-row select { grid-column:1/-1; } }
@media (max-width:520px) { .content,.controls { padding-left:10px; padding-right:10px; } .replay-row { grid-template-columns:1fr; }
  .replay-row select { grid-column:auto; } #timeline { height:230px; } .panel canvas { height:260px; } }
</style>
</head>
<body>
<div class="top">
  <div class="title"><h1>IQL Bellman push/pull explorer</h1><span class="subtitle">real replay rewards + local critic dynamics</span></div>
  <div class="controls">
    <div class="control"><label for="gamma">gamma / frame <output id="gammav"></output></label>
      <input id="gamma" type="range" min="0.98000" max="0.99990" step="0.00001" value="0.99827"></div>
    <div class="control"><label for="tau">expectile tau <output id="tauv"></output></label>
      <input id="tau" type="range" min="0.50" max="0.99" step="0.01" value="0.90"></div>
    <div class="control"><label for="sigma">Q noise sigma <output id="sigmav"></output></label>
      <input id="sigma" type="range" min="0" max="0.50" step="0.005" value="0.10"></div>
    <div class="control"><label for="support">categorical support <output id="supportv"></output></label>
      <input id="support" type="range" min="1" max="20" step="0.25" value="4"></div>
    <div class="control"><label for="alpha">critic update rate <output id="alphav"></output></label>
      <input id="alpha" type="range" min="-3" max="-0.30" step="0.02" value="-1.52"></div>
    <div class="control"><label for="updates">training update <output id="updatesv"></output></label>
      <input id="updates" type="range" min="0" max="16384" step="64" value="16384"></div>
    <div class="actions"><button id="reset" type="button">Reset to 027</button></div>
  </div>
</div>

<main class="content">
  <section aria-labelledby="replay-heading">
    <h2 id="replay-heading">Replay prefix</h2>
    <div class="replay-row">
      <select id="game" aria-label="Replay"></select>
      <button id="ego" type="button"></button>
      <a id="watch" target="_blank">Watch selected frame</a>
    </div>
    <div id="gameMeta" class="meta"></div>
    <canvas id="timeline" class="plot" role="img" aria-label="Replay reward, four-frame reward, and Monte Carlo return timeline"></canvas>
    <div class="prefix-control"><label for="prefix">selected prefix <output id="prefixv"></output></label>
      <input id="prefix" type="range" min="0" max="1" step="1" value="0"></div>
  </section>

  <div id="stats" class="stats"></div>

  <div class="panels">
    <section class="panel" aria-labelledby="forces-heading">
      <h2 id="forces-heading">Push / pull at the selected update</h2>
      <div class="legend">
        <span><i style="background:var(--green)"></i>pushes V up</span>
        <span><i style="background:var(--red)"></i>pulls V down</span>
      </div>
      <canvas id="forces" class="plot" role="img" aria-label="Additive decomposition of the value target residual"></canvas>
    </section>
    <section class="panel" aria-labelledby="trajectory-heading">
      <h2 id="trajectory-heading">Repeated-prefix critic trajectory</h2>
      <div class="legend">
        <span><i style="background:var(--blue)"></i>V with support clipping</span>
        <span><i style="background:var(--orange)"></i>Bellman Q target</span>
        <span><i style="background:var(--purple)"></i>V without support clipping</span>
        <span><i style="background:var(--muted)"></i>actual MC return</span>
      </div>
      <canvas id="trajectory" class="plot" role="img" aria-label="Simulated value and Q-target trajectory over training updates"></canvas>
    </section>
  </div>

  <p class="explain"><strong>What is real:</strong> damage, stock events, the selected four-frame reward, and Monte Carlo return.
    <strong>What is modeled:</strong> Q-estimation noise, a locally repeated transition, V(t+4) approximately equal to V(t), and a relaxed critic update.
    The unclipped line answers “where would the feedback go without the Q bins?”; it is not a checkpoint prediction.</p>
</main>
<div id="tip"></div>

<script>
"use strict";
const DATA = __DATA__;
const SLIPPILAB_BASE = __SLIPPILAB_BASE__;
const SLIPPILAB_MOUNT = __SLIPPILAB_MOUNT__;
const FPS = 60, CHUNK = 4, MAX_UPDATES = 16384, DPR = window.devicePixelRatio || 1;
const colors = {blue:"#4ea1ff", orange:"#ffb347", green:"#4caf7a", red:"#e05555", purple:"#b783ff",
  muted:"#8b97a1", grid:"#303840", text:"#d7dde3", fill:"#1a1f24"};
const el = {};
for (const id of ["gamma","tau","sigma","support","alpha","updates","gammav","tauv","sigmav","supportv",
  "alphav","updatesv","reset","game","ego","watch","gameMeta","timeline","prefix","prefixv","stats","forces","trajectory"])
  el[id] = document.getElementById(id);
const tip = document.getElementById("tip");
let gameIndex = 0, ego = 1, prefix = 0, state = null;

function deaths(stock) {
  const out = new Float64Array(stock.length);
  for (let t=1;t<stock.length;t++) out[t] = stock[t] < stock[t-1] ? 1 : 0;
  return out;
}
function damageTaken(percent) {
  const out = new Float64Array(percent.length);
  for (let t=1;t<percent.length;t++) { const d=percent[t]-percent[t-1]; out[t]=Number.isFinite(d)&&d>0?d:0; }
  return out;
}
function returns(reward,gamma) {
  const out=new Float64Array(reward.length); let acc=0;
  for (let t=reward.length-1;t>=0;t--) { acc=reward[t]+gamma*acc; out[t]=acc; }
  return out;
}
function chunkRewards(reward,gamma) {
  const out=new Float64Array(Math.max(0,reward.length-CHUNK));
  for (let t=0;t<out.length;t++) for (let j=0;j<CHUNK;j++) out[t]+=Math.pow(gamma,j)*reward[t+1+j];
  return out;
}
function settings() {
  return {gamma:+el.gamma.value,tau:+el.tau.value,sigma:+el.sigma.value,support:+el.support.value,
    alpha:Math.pow(10,+el.alpha.value),updates:+el.updates.value};
}
function rewardData(g,k) {
  const egoP=ego===1?g.p1Percent:g.p2Percent, oppP=ego===1?g.p2Percent:g.p1Percent;
  const egoS=ego===1?g.p1Stock:g.p2Stock, oppS=ego===1?g.p2Stock:g.p1Stock;
  const dealt=damageTaken(oppP), taken=damageTaken(egoP), oppDeath=deaths(oppS), egoDeath=deaths(egoS);
  const reward=new Float64Array(g.n);
  for (let t=0;t<g.n;t++) {
    const oppFinal=oppDeath[t]&&oppS[t]===0?1:0, egoFinal=egoDeath[t]&&egoS[t]===0?1:0;
    reward[t]=(oppDeath[t]-egoDeath[t])+0.5*(oppFinal-egoFinal)+0.01*(dealt[t]-taken[t]);
  }
  return {reward,R4:chunkRewards(reward,k.gamma),G:returns(reward,k.gamma),dealt,taken,oppDeath,egoDeath};
}

const normalGrid=[];
for (let i=0;i<=160;i++) { const z=-5+10*i/160; normalGrid.push({z,w:Math.exp(-.5*z*z)}); }
function expectile(values,weights,tau) {
  let lo=Math.min(...values),hi=Math.max(...values);
  if (!(hi>lo)) return lo;
  for (let iter=0;iter<28;iter++) {
    const m=(lo+hi)/2; let score=0;
    for (let i=0;i<values.length;i++) { const d=values[i]-m; score+=weights[i]*(d>=0?tau:1-tau)*d; }
    if (score>0) lo=m; else hi=m;
  }
  return (lo+hi)/2;
}
function noiseUplift(k) {
  const values=normalGrid.map(p=>k.sigma*p.z),weights=normalGrid.map(p=>p.w);
  return expectile(values,weights,k.tau);
}
function clippedTarget(center,k) {
  const values=normalGrid.map(p=>Math.max(-k.support,Math.min(k.support,center+k.sigma*p.z)));
  return expectile(values,normalGrid.map(p=>p.w),k.tau);
}
function simulate(R4,k) {
  const gamma4=Math.pow(k.gamma,4),uplift=noiseUplift(k),stride=32;
  const blockAlpha=1-Math.pow(1-k.alpha,stride), rows=[];
  let V=0,U=0,firstCross=null;
  for (let step=0;step<=MAX_UPDATES;step+=stride) {
    const qCenter=R4+gamma4*V,unboundedQ=R4+gamma4*U;
    if (firstCross===null && Math.abs(unboundedQ)>k.support) firstCross=step;
    rows.push({step,V,U,qCenter,decodedQ:Math.max(-k.support,Math.min(k.support,qCenter))});
    const desired=clippedTarget(qCenter,k);
    V+=blockAlpha*(desired-V);
    U+=blockAlpha*(unboundedQ+uplift-U);
  }
  const fixed=Math.abs(1-gamma4)<1e-12?Math.sign(R4+uplift)*Infinity:(R4+uplift)/(1-gamma4);
  return {rows,gamma4,uplift,fixed,firstCross};
}
function selectedRow(sim,updates) {
  let best=sim.rows[0];
  for (const row of sim.rows) if (Math.abs(row.step-updates)<Math.abs(best.step-updates)) best=row;
  return best;
}
function forceParts(R4,k,sim,row) {
  const center=R4+sim.gamma4*row.V,unclipped=center+sim.uplift,clipped=clippedTarget(center,k);
  return [
    {name:"real R₄ reward",value:R4},
    {name:"expectile / noise push",value:sim.uplift},
    {name:"Bellman contraction",value:-(1-sim.gamma4)*row.V},
    {name:"support clipping",value:clipped-unclipped},
  ];
}

function fitCanvas(canvas,height) {
  const W=Math.max(300,Math.round(canvas.clientWidth)),H=height;
  if (canvas.width!==W*DPR||canvas.height!==H*DPR) { canvas.width=W*DPR; canvas.height=H*DPR; }
  const x=canvas.getContext("2d"); x.setTransform(DPR,0,0,DPR,0,0); x.clearRect(0,0,W,H); return {x,W,H};
}
function extent(arrays,extra=[]) {
  let lo=Infinity,hi=-Infinity;
  for (const arr of arrays) for (const v of arr) if (Number.isFinite(v)) { lo=Math.min(lo,v);hi=Math.max(hi,v); }
  for (const v of extra) if (Number.isFinite(v)) { lo=Math.min(lo,v);hi=Math.max(hi,v); }
  if (!Number.isFinite(lo)||!Number.isFinite(hi)) return [-1,1];
  if (hi-lo<1e-9) { lo-=1;hi+=1; } const pad=.08*(hi-lo); return [lo-pad,hi+pad];
}
function grid(x,W,H,left,top,right,bottom,xTicks,yTicks,xLabel,yLabel) {
  x.strokeStyle=colors.grid;x.fillStyle=colors.muted;x.font="11px system-ui";x.lineWidth=1;
  for (const tick of xTicks) { const X=left+tick.p*(W-left-right);x.beginPath();x.moveTo(X,top);x.lineTo(X,H-bottom);x.stroke();
    x.textAlign=tick.p===0?"left":tick.p===1?"right":"center";x.fillText(tick.label,X,H-20); }
  for (const tick of yTicks) { const Y=top+(1-tick.p)*(H-top-bottom);x.beginPath();x.moveTo(left,Y);x.lineTo(W-right,Y);x.stroke();
    x.textAlign="right";x.fillText(tick.label,left-6,Y+4); }
  x.textAlign="center";x.fillText(xLabel,left+(W-left-right)/2,H-8);
  x.save();x.translate(12,top+(H-top-bottom)/2);x.rotate(-Math.PI/2);x.fillText(yLabel,0,0);x.restore();x.textAlign="left";
}
function drawTimeline(g,d) {
  const {x,W,H}=fitCanvas(el.timeline,260),L=50,R=12,T=12,B=22,band=(H-T-B)/3;
  const px=t=>L+t/(g.n-1)*(W-L-R),selectedX=px(prefix);
  x.fillStyle=colors.fill;x.fillRect(0,0,W,H);
  for (let b=0;b<3;b++) { x.strokeStyle=colors.grid;x.strokeRect(L,T+b*band,W-L-R,band); }
  x.fillStyle=colors.muted;x.font="11px system-ui";x.fillText("reward",5,T+15);x.fillText("R₄",24,T+band+15);x.fillText("G",31,T+2*band+15);
  const eventMax=Math.max(.01,...Array.from(d.reward,v=>Math.abs(v))),eventMid=T+band/2;
  for (let t=0;t<g.n;t++) if (d.reward[t]!==0) { const Y=eventMid-d.reward[t]/eventMax*(band*.42);x.strokeStyle=d.reward[t]>=0?colors.green:colors.red;
    x.beginPath();x.moveTo(px(t),eventMid);x.lineTo(px(t),Y);x.stroke(); }
  const [rlo,rhi]=extent([d.R4],[0]),ry=v=>T+2*band-6-(v-rlo)/(rhi-rlo)*(band-12);
  drawLine(x,d.R4,px,ry,colors.orange);
  const [glo,ghi]=extent([d.G],[0]),gy=v=>T+3*band-6-(v-glo)/(ghi-glo)*(band-12);
  drawLine(x,d.G,px,gy,colors.blue);
  x.fillStyle="#4ea1ff22";x.fillRect(px(prefix+1),T,Math.max(2,px(prefix+CHUNK)-px(prefix+1)),3*band);
  x.strokeStyle="#ffffff";x.beginPath();x.moveTo(selectedX,T);x.lineTo(selectedX,T+3*band);x.stroke();
  x.fillStyle=colors.text;x.font="11px system-ui";x.fillText("t",Math.min(W-12,selectedX+4),T+12);
  x.fillStyle=colors.muted;x.textAlign="center";
  for (let s=0;s<=g.n;s+=Math.max(FPS*30,Math.ceil(g.n/(FPS*6))*FPS)) { const sec=s/FPS;x.fillText(Math.floor(sec/60)+":"+String(Math.floor(sec)%60).padStart(2,"0"),px(Math.min(s,g.n-1)),H-6); }
  x.textAlign="left";
}
function drawLine(x,arr,xmap,ymap,color) {
  x.strokeStyle=color;x.lineWidth=1.5;x.beginPath();const step=Math.max(1,Math.floor(arr.length/1600));
  for (let i=0;i<arr.length;i+=step) i===0?x.moveTo(xmap(i),ymap(arr[i])):x.lineTo(xmap(i),ymap(arr[i]));x.stroke();x.lineWidth=1;
}
function drawForces(parts,k) {
  const {x,W,H}=fitCanvas(el.forces,290),L=Math.min(150,W*.39),R=46,T=18,B=28;
  const residual=parts.reduce((sum,p)=>sum+p.value,0),all=[...parts,{name:"net target residual",value:residual}];
  let max=Math.max(.02,...all.map(p=>Math.abs(p.value)))*1.12,zero=L+(W-L-R)/2,scale=(W-L-R)/2/max,rowH=(H-T-B)/all.length;
  x.fillStyle=colors.fill;x.fillRect(0,0,W,H);x.strokeStyle=colors.grid;x.beginPath();x.moveTo(zero,T);x.lineTo(zero,H-B);x.stroke();
  x.fillStyle=colors.muted;x.font="11px system-ui";x.textAlign="center";x.fillText("−"+fmt(max),L,H-8);x.fillText("0",zero,H-8);x.fillText("+"+fmt(max),W-R,H-8);
  all.forEach((p,i)=>{ const y=T+i*rowH+rowH*.2,h=rowH*.55,w=p.value*scale;x.fillStyle=p.value>=0?colors.green:colors.red;
    x.fillRect(Math.min(zero,zero+w),y,Math.abs(w),h);x.fillStyle=colors.text;x.textAlign="right";x.fillText(p.name,L-7,y+h*.72);
    x.textAlign=p.value>=0?"left":"right";x.fillText((p.value>=0?"+":"")+p.value.toFixed(4),zero+w+(p.value>=0?5:-5),y+h*.72); });
  x.textAlign="left";
}
function drawTrajectory(sim,k,mc) {
  const {x,W,H}=fitCanvas(el.trajectory,290),L=62,R=16,T=15,B=34,rows=sim.rows;
  let arrays=[rows.map(r=>r.V),rows.map(r=>r.U),rows.map(r=>r.qCenter)];
  let [lo,hi]=extent(arrays,[-k.support,k.support,mc]);
  const cap=Math.max(k.support*4,Math.abs(mc)*2,8);lo=Math.max(lo,-cap);hi=Math.min(hi,cap);
  if (hi-lo<1) {lo-=1;hi+=1;}
  const px=s=>L+s/MAX_UPDATES*(W-L-R),py=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  x.fillStyle=colors.fill;x.fillRect(0,0,W,H);
  const xt=[0,.25,.5,.75,1].map(p=>({p,label:String(Math.round(p*MAX_UPDATES/1024))+"k"}));
  const yt=[0,.25,.5,.75,1].map(p=>({p,label:(lo+p*(hi-lo)).toFixed(1)}));
  grid(x,W,H,L,T,R,B,xt,yt,"training updates","value units");
  for (const v of [-k.support,k.support]) if (v>=lo&&v<=hi) {x.strokeStyle=colors.red;x.setLineDash([5,4]);x.beginPath();x.moveTo(L,py(v));x.lineTo(W-R,py(v));x.stroke();x.setLineDash([]);}
  if (mc>=lo&&mc<=hi) {x.strokeStyle=colors.muted;x.setLineDash([2,4]);x.beginPath();x.moveTo(L,py(mc));x.lineTo(W-R,py(mc));x.stroke();x.setLineDash([]);}
  const series=[["V",r=>r.V,colors.blue],["q",r=>r.qCenter,colors.orange],["U",r=>r.U,colors.purple]];
  for (const [,get,color] of series) {x.strokeStyle=color;x.lineWidth=1.8;x.beginPath();rows.forEach((r,i)=>i?x.lineTo(px(r.step),py(Math.max(lo,Math.min(hi,get(r))))):x.moveTo(px(r.step),py(Math.max(lo,Math.min(hi,get(r))))));x.stroke();}
  const selected=selectedRow(sim,k.updates);x.strokeStyle="#fff";x.lineWidth=1;x.beginPath();x.moveTo(px(selected.step),T);x.lineTo(px(selected.step),H-B);x.stroke();
}
function fmt(v) { const a=Math.abs(v);return a>=100?Math.round(v).toString():a>=10?v.toFixed(1):v.toFixed(3); }
function stat(label,value,small="") { return '<div class="stat"><span class="label">'+label+'</span><b>'+value+'</b>'+(small?'<small>'+small+'</small>':"")+'</div>'; }
function slippiUrl(g,t) { return SLIPPILAB_BASE+"/?replayUrl="+encodeURIComponent(SLIPPILAB_BASE+"/"+SLIPPILAB_MOUNT+"/"+g.file)+"#"+t; }
function timelineFrame(clientX,bounds,g) {
  const left=50,right=12,plotWidth=Math.max(1,bounds.width-left-right);
  return Math.round(Math.max(0,Math.min(1,(clientX-bounds.left-left)/plotWidth))*(g.n-1));
}

function render() {
  const g=DATA[gameIndex],k=settings(),d=rewardData(g,k);
  prefix=Math.max(0,Math.min(prefix,d.R4.length-1));el.prefix.max=String(d.R4.length-1);el.prefix.value=String(prefix);
  const R4=d.R4[prefix],mc=d.G[Math.min(prefix+1,d.G.length-1)],sim=simulate(R4,k),row=selectedRow(sim,k.updates),parts=forceParts(R4,k,sim,row);
  const clipTarget=clippedTarget(R4+sim.gamma4*row.V,k),net=parts.reduce((sum,p)=>sum+p.value,0);
  state={g,k,d,sim,row,parts};
  el.gammav.textContent=k.gamma.toFixed(5);el.tauv.textContent=k.tau.toFixed(2);el.sigmav.textContent=k.sigma.toFixed(3);
  el.supportv.textContent="±"+k.support.toFixed(2);el.alphav.textContent=k.alpha.toFixed(4);el.updatesv.textContent=k.updates.toLocaleString();
  el.prefixv.textContent=prefix+" ("+(prefix/FPS).toFixed(2)+" s)";
  el.ego.textContent="ego: "+(ego===1?g.p1+" (P1)":g.p2+" (P2)");el.watch.href=slippiUrl(g,prefix);
  el.gameMeta.textContent=g.stage+" · "+g.source+" · "+g.note+" · "+g.n.toLocaleString()+" frames";
  const contraction=1-sim.gamma4,ratio=contraction>0?Math.abs(sim.uplift/contraction):Infinity;
  el.stats.innerHTML=
    stat("four-frame contraction",(100*contraction).toFixed(2)+"%","γ⁴ = "+sim.gamma4.toFixed(5))+
    stat("real selected R₄",(R4>=0?"+":"")+R4.toFixed(4),"rewards t+1…t+4")+
    stat("real MC return Gₜ₊₁",(mc>=0?"+":"")+mc.toFixed(4),"same replay and γ")+
    stat("expectile uplift","+"+sim.uplift.toFixed(4),"σ × standard-normal expectile")+
    stat("discount pull",(-(1-sim.gamma4)*row.V).toFixed(4),"at V = "+row.V.toFixed(3))+
    stat("push / contraction",Number.isFinite(ratio)?ratio.toFixed(2):"∞","uplift needed per value unit")+
    stat("unclipped fixed point",Number.isFinite(sim.fixed)?fmt(sim.fixed):"±∞","local transition repeated")+
    stat("support crossing",sim.firstCross===null?"none by 16k":sim.firstCross.toLocaleString()+" updates","unclipped Bellman target")+
    stat("clipped V at selected update",row.V.toFixed(4),"next target = "+clipTarget.toFixed(4))+
    stat("net residual",(net>=0?"+":"")+net.toFixed(4),"actual update = "+(k.alpha*net).toFixed(5));
  drawTimeline(g,d);drawForces(parts,k);drawTrajectory(sim,k,mc);
}

DATA.forEach(g=>g.n=g.p1Percent.length);
DATA.forEach((g,i)=>{ const option=document.createElement("option");option.value=String(i);option.textContent=g.p1+" vs "+g.p2+" · "+g.stage+" · "+g.file;el.game.appendChild(option); });
el.game.addEventListener("change",()=>{gameIndex=+el.game.value;prefix=0;render();});
el.ego.addEventListener("click",()=>{ego=ego===1?2:1;render();});
for (const id of ["gamma","tau","sigma","support","alpha","updates"]) el[id].addEventListener("input",render);
el.prefix.addEventListener("input",()=>{prefix=+el.prefix.value;render();});
el.reset.addEventListener("click",()=>{el.gamma.value="0.99827";el.tau.value="0.90";el.sigma.value="0.10";el.support.value="4";
  el.alpha.value="-1.52";el.updates.value="16384";render();});
el.timeline.addEventListener("click",ev=>{const r=el.timeline.getBoundingClientRect(),g=DATA[gameIndex];prefix=timelineFrame(ev.clientX,r,g);render();});
el.timeline.addEventListener("mousemove",ev=>{const r=el.timeline.getBoundingClientRect(),g=DATA[gameIndex],d=state.d;
  const t=timelineFrame(ev.clientX,r,g);tip.style.display="block";
  tip.style.left=(ev.clientX+12)+"px";tip.style.top=(ev.clientY+12)+"px";const r4=t<d.R4.length?d.R4[t]:NaN;
  tip.innerHTML="frame "+t+" · "+(t/FPS).toFixed(2)+" s<br>r = "+d.reward[t].toFixed(4)+(Number.isFinite(r4)?" · R₄ = "+r4.toFixed(4):"")+"<br>G = "+d.G[t].toFixed(4);});
el.timeline.addEventListener("mouseleave",()=>tip.style.display="none");
window.addEventListener("resize",()=>{if(state){drawTimeline(state.g,state.d);drawForces(state.parts,state.k);drawTrajectory(state.sim,state.k,state.d.G[Math.min(prefix+1,state.d.G.length-1)]);}});
render();
</script>
</body>
</html>
"""


def render_html(games: Sequence[dict], *, slippilab_base: str, slippilab_mount: str) -> str:
    if not games:
        raise ValueError("at least one replay is required")
    return (
        HTML.replace("__DATA__", json.dumps(list(games), separators=(",", ":")))
        .replace("__SLIPPILAB_BASE__", json.dumps(slippilab_base.rstrip("/")))
        .replace("__SLIPPILAB_MOUNT__", json.dumps(slippilab_mount.strip("/")))
    )


def write_explorer(
    games: Sequence[dict],
    out_html: Path,
    *,
    slippilab_base: str = DEFAULT_SLIPPILAB_BASE,
    slippilab_mount: str = "hal-iql",
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        render_html(games, slippilab_base=slippilab_base, slippilab_mount=slippilab_mount),
        encoding="utf-8",
    )
    print(f"wrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB, {len(games)} games)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slps-dir", type=Path, default=DEFAULT_SLPS_DIR)
    parser.add_argument("--out-html", type=Path, default=DEFAULT_OUT_HTML)
    parser.add_argument("--slippilab-public", type=Path, default=DEFAULT_SLIPPILAB_PUBLIC)
    parser.add_argument("--slippilab-base", default=DEFAULT_SLIPPILAB_BASE)
    parser.add_argument("--no-slippilab-link", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    games = extract_games(args.slps_dir)
    if not games:
        raise SystemExit(f"no usable .slp files under {args.slps_dir}")
    mount = args.slippilab_public.name
    write_explorer(games, args.out_html, slippilab_base=args.slippilab_base, slippilab_mount=mount)
    if not args.no_slippilab_link:
        args.slippilab_public.parent.mkdir(parents=True, exist_ok=True)
        if not args.slippilab_public.exists():
            args.slippilab_public.symlink_to(args.slps_dir.resolve())
        print(f"slippilab mount: {args.slippilab_public} -> {args.slps_dir.resolve()}")
        print(f"start the viewer: cd {args.slippilab_public.parents[1]} && npm run dev")


if __name__ == "__main__":
    main()
