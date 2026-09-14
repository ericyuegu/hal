(function () {
"use strict";
const WB = id => `https://wandb.ai/ericyuegu/hal/runs/${id}`;
/* ---------- the audited comparison table ---------- */
const EV = window.HALFIG_EXPLORER_EV = [
 {r:1, c:"recipe", t:"Training objective: BC → offline Q learning", a:"behaviour cloning +0.16", b:"IQL −1.85", d:-2.01, l:-2.05, q:"weak", runs:["cqbbbg77","06fi5jms"],
  note:"Data, compute, value heads and losses also changed. The tested IQL system was clearly bad. This does not isolate IQL itself."},
 {r:2, c:"trunk", t:"Model size", a:"7M −3.62", b:"205M −2.09", d:1.52, l:1.61, q:"near", runs:[],
  note:"One family, one seed, at delay and replan of one frame. Not monotonic: 18M scored below 7M and 112M below 51M. At delay and replan of four frames the same span is −0.54."},
 {r:3, c:"loop", t:"Prediction delay", a:"d=1 +0.24", b:"d=12 −0.86", d:-1.11, l:-1.21, q:"near", runs:["1imfy8v3","2qs05au4"],
  note:"Near control. The Fox-only score changed by −0.98. The delay-12 run kept H=6 and R=1 but used a longer future-data buffer."},
 {r:4, c:"codec", t:"Legacy action setup", a:"modern codec −0.50", b:"legacy codec +0.24", d:0.75, l:0.69, q:"combined", runs:["k8nb00xv","1imfy8v3"],
  note:"Four changes at once: legacy output classes, the early-release reducer, one-frame timing, and half the action loss on the next frame. The 8k checkpoint gave +0.62."},
 {r:5, c:"recipe", t:"Training windows per replay", a:"K=4 −0.60", b:"K=1 +0.13", d:0.73, l:0.73, q:"strong", runs:["uw05bvm2","shjnxxsu"],
  note:"Only the recorded window setting changed. It must still be repeated on the current reservoir sampler."},
 {r:6, c:"groups", t:"Architecture package", a:"previous +0.16", b:"current −0.50", d:-0.66, l:-0.64, q:"combined", runs:["cqbbbg77","k8nb00xv"],
  note:"Architecture details and sampling code changed together. The later recovery on the same architecture shows that architecture is not the sole cause."},
 {r:7, c:"recipe", t:"Batch size at fixed data", a:"2,048 −0.79", b:"128 −0.32", d:0.47, l:0.47, q:"strong", runs:[],
  note:"All five arms saw 2.097M examples, so a smaller batch simply bought more optimizer updates. One seed."},
 {r:8, c:"dec", t:"Plan length and replan interval", a:"H=1", b:"H=6", d:-0.43, l:-0.44, q:"combined", runs:[],
  note:"Mean over four decoder cells, same checkpoint on each side. The old runs tied plan length to replan interval, so this is not a horizon-only test. Per-cell change ran from −1.00 to −0.05."},
 {r:9, c:"recipe", t:"Training data mix", a:"ranked only −0.50", b:"full mix −0.12", d:0.39, l:0.38, q:"strong", runs:["k8nb00xv","bevhq159"],
  note:"Strong control inside one experiment. Separately, an old professional-data run beat an old mixed run by +0.51, but that comparison has other changes."},
 {r:10, c:"dec", t:"Predicted-action feedback", a:"off", b:"on", d:0.36, l:0.35, q:"strong", runs:["98r9smrj","50q39o9j"],
  note:"Exact factorization at four executed frames: −0.70 → −0.20 without group conditioning, −0.31 → −0.10 with it. At one executed frame it is harmful, mean −0.18."},
 {r:11, c:"obs", t:"Observation feature set", a:"full −0.62", b:"lean −0.89", d:-0.27, l:-0.30, q:"near", runs:["pu8jpmbf"],
  note:"Good historical near-control. Removing or changing state features harmed play. The state schema changed too, so no single column is implicated."},
 {r:12, c:"obs", t:"Past-action input encoding", a:"one-hot −0.10", b:"learned codec +0.16", d:0.26, l:0.26, q:"near", runs:["1zzstd2d","cqbbbg77"],
  note:"The one-hot run saw twice the examples and still lost. At matched exposure its validation loss and its button-change F1 were also worse."},
 {r:13, c:"groups", t:"Controller-part conditioning", a:"independent groups", b:"chained groups", d:0.24, l:0.24, q:"strong", runs:["98r9smrj","a117chkw"],
  note:"Exact control at four executed frames: −0.70 → −0.31 without future conditioning, −0.20 → −0.10 with it. Letting later controller parts depend on earlier parts helped in both backgrounds."},
 {r:14, c:"recipe", t:"Learning-rate pair", a:"worst −0.59", b:"best −0.37", d:0.22, l:0.22, q:"strong", runs:[],
  note:"Fixed model, fixed data, six paired points. The best point is near .040 / .00170 and it is interior to the sweep."},
 {r:15, c:"dec", t:"Future targets and next-frame share", a:"{1,12} +1.10", b:"{1,12,18} +0.90", d:-0.20, l:null, q:"near", runs:["c6r7edxf","8fo0kube"],
  note:"Adding a target offset also reduced the relative weight of frame 1. This supports keeping half the loss on the next frame."},
 {r:16, c:"recipe", t:"Training examples seen", a:"8.39M +1.04", b:"16.78M +1.23", d:0.19, l:null, q:"near", runs:[],
  note:"Same run, two checkpoints. Training longer helps, but it does not explain the original gap."},
 {r:17, c:"dec", t:"Decoder type: categorical → flow", a:"categorical −0.10", b:"flow −0.28", d:-0.19, l:-0.16, q:"strong", runs:["5wfk2esf","t4s3gohe"],
  note:"Matched trunk and data. Decoder, objective, parameters and inference work changed together."},
 {r:18, c:"recipe", t:"Advantage-weighted regression", a:"behaviour cloning −0.62", b:"AWR −0.50", d:0.12, l:0.11, q:"near", runs:["5wfk2esf","hwzv0k9a"],
  note:"The modern matched pair gives +0.08 and the old reference about +0.13. The evidence is consistent but modest."},
 {r:19, c:"groups", t:"Action-head capacity", a:"linear head −0.69", b:"state-MLP head −0.58", d:0.11, l:0.08, q:"near", runs:["obx3o3az","q3aojgfm"],
  note:"Parameters and compute rose 12.6 %. A later head-to-head test did not confirm a large benefit."},
 {r:20, c:"obs", t:"Item and projectile features", a:"off −1.43", b:"on −1.34", d:0.09, l:0.11, q:"strong", runs:["v9xku05q","9wh7qhr1"],
  note:"Direct feature control. A small positive effect."},
 {r:21, c:"recipe", t:"Future-state auxiliary loss", a:"off +0.16", b:"on +0.11", d:-0.05, l:-0.06, q:"weak", runs:["cqbbbg77","qxcjjwl8"],
  note:"Also used twice the data, more parameters, more compute and another attention backend."},
 {r:22, c:"obs", t:"One-hot button class count", a:"256 classes −0.10", b:"52 classes −0.06", d:0.04, l:0.03, q:"strong", runs:["1zzstd2d","4wyxy1g2"],
  note:"Strong control inside one experiment. This concerns the one-hot history input, not the legacy six-class output codec."},
 {r:23, c:"recipe", t:"Optimizer and rate recipe", a:"AdamW −0.41", b:"Muon + AdamW −0.38", d:0.03, l:0.03, q:"strong", runs:["o3nihah5","vlim96s9"],
  note:"Exact recipe comparison, but the optimizer and the rates changed together. There is no evidence here for a large Muon benefit."},
 {r:24, c:"dec", t:"Future target offsets", a:"{1,2,3,4} −0.59", b:"{1,5,9,13} −0.56", d:0.03, l:0.05, q:"weak", runs:[],
  note:"Weak control, because the auxiliary-loss weight also changed."}
];

const COMPS = [
  ["obs","Observation","observation"], ["trunk","Trunk","trunk"], ["dec","Decoder","decoder"],
  ["groups","Group chain","groups"], ["codec","Codec","codec"], ["loop","Timing","timing"], ["recipe","Recipe","recipe"]
];
const QUALS = [["strong","strong"],["near","near"],["combined","combined"],["weak","weak"]];
const SEC_OF = Object.fromEntries(COMPS.map(([k,,s]) => [k,s]));

/* ---------- tiny helpers ---------- */
function fmt(v, sign){ if (v === null || v === undefined) return "—"; const s = v.toFixed(2); return sign && v > 0 ? "+" + s : s; }
function cls(v){ return v > 0 ? "pos" : v < 0 ? "neg" : ""; }

/* ---------- tooltip ---------- */
const tip = document.getElementById("xtip");
function showTip(html, x, y){
  tip.innerHTML = html; tip.classList.add("on");
  const r = tip.getBoundingClientRect();
  const left = Math.min(Math.max(10, x + 14), window.innerWidth - r.width - 10);
  const top = Math.min(Math.max(10, y + 14), window.innerHeight - r.height - 10);
  tip.style.left = left + "px"; tip.style.top = top + "px";
}
function hideTip(){ tip.classList.remove("on"); }

/* ---------- ranked chart ---------- */
const SCALE = 2.5;
const state = { comp:new Set(), qual:new Set(), sort:"d" };

function pillRow(host, items, key, onDone){
  const all = document.createElement("button");
  all.className = "pill"; all.type = "button"; all.textContent = "all";
  all.setAttribute("aria-pressed", "true");
  all.addEventListener("click", () => { state[key].clear(); sync(); });
  host.appendChild(all);
  items.forEach(([k, label]) => {
    const b = document.createElement("button");
    b.className = key === "comp" ? "pill k" : "pill"; b.type = "button"; b.textContent = label;
    b.dataset.k = k; b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => {
      state[key].has(k) ? state[key].delete(k) : state[key].add(k);
      sync();
    });
    host.appendChild(b);
  });
  function sync(){
    const s = state[key];
    all.setAttribute("aria-pressed", String(s.size === 0));
    host.querySelectorAll("[data-k]").forEach(b => b.setAttribute("aria-pressed", String(s.has(b.dataset.k))));
    onDone();
  }
}

const fComp = document.getElementById("f-comp");
const fCtl = document.getElementById("f-ctl");
const fSort = document.getElementById("f-sort");
pillRow(fComp, COMPS.map(([k,l]) => [k,l]), "comp", drawRank);
pillRow(fCtl, QUALS, "qual", drawRank);
[["d","change in score"],["l","change in LCB"]].forEach(([k,label], i) => {
  const b = document.createElement("button");
  b.className = "pill"; b.type = "button"; b.textContent = label; b.dataset.s = k;
  b.setAttribute("aria-pressed", String(i === 0));
  b.addEventListener("click", () => {
    state.sort = k;
    fSort.querySelectorAll("[data-s]").forEach(x => x.setAttribute("aria-pressed", String(x.dataset.s === k)));
    drawRank();
  });
  fSort.appendChild(b);
});

const rankBody = document.getElementById("rank-body");
function drawRank(){
  const rows = EV
    .filter(e => (state.comp.size === 0 || state.comp.has(e.c)) && (state.qual.size === 0 || state.qual.has(e.q)))
    .slice()
    .sort((x, y) => {
      const kx = state.sort === "l" && x.l !== null ? x.l : x.d;
      const ky = state.sort === "l" && y.l !== null ? y.l : y.d;
      return Math.abs(ky) - Math.abs(kx);
    });
  rankBody.replaceChildren();
  if (!rows.length){
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4" style="padding:22px 8px; color:var(--ink3)">No comparison matches these filters.</td>`;
    rankBody.appendChild(tr);
    return;
  }
  const compLabel = Object.fromEntries(COMPS.map(([k,l]) => [k,l]));
  for (const e of rows){
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    const w = Math.min(Math.abs(e.d) / SCALE, 1) * 50;
    const left = e.d >= 0 ? 50 : 50 - w;
    const lt = e.l === null ? "" :
      `<u style="left:${(50 + Math.max(Math.min(e.l / SCALE, 1), -1) * 50).toFixed(2)}%; background:var(--ink2)"></u>`;
    tr.innerHTML =
      `<td class="f">${e.t}<small>${compLabel[e.c]}</small></td>` +
      `<td class="d"><div class="bar"><span class="zero" style="left:50%"></span>` +
        `<i class="${e.d >= 0 ? "gain" : "loss"}" style="left:${left.toFixed(2)}%; width:${w.toFixed(2)}%"></i>${lt}</div></td>` +
      `<td class="v ${cls(e.d)}">${fmt(e.d, true)}<small class="lcbmini">LCB ${fmt(e.l, true)}</small></td>` +
      `<td class="q"><span class="badge ${e.q}">${e.q}</span></td>`;
    const go = () => { location.hash = "#" + SEC_OF[e.c]; flash(e); };
    tr.addEventListener("click", go);
    tr.addEventListener("keydown", ev => { if (ev.key === "Enter"){ ev.preventDefault(); go(); } });
    tr.addEventListener("mousemove", ev => showTip(
      `<span class="hd">${e.t}</span>${e.a} &nbsp;→&nbsp; ${e.b}<br>` +
      `score <b>${fmt(e.d, true)}</b> · LCB <b>${fmt(e.l, true)}</b><br>` +
      `<span style="color:var(--ink3)">${e.note}</span>`, ev.clientX, ev.clientY));
    tr.addEventListener("mouseleave", hideTip);
    rankBody.appendChild(tr);
  }
}
drawRank();

/* ---------- glossary: the netplay timing figure is drawn from three settings ---------- */
(function(){
  const NS = "http://www.w3.org/2000/svg";
  const fig = document.getElementById("fig-timing");
  const svg = document.getElementById("fig-timing-svg");
  const cap = document.getElementById("fig-timing-cap");
  if (!fig || !svg) return;
  const FRAME_MS = 1000 / 60;
  const state = {d:2, H:4, R:2, ms:15};

  function E(parent, tag, attrs, text){
    const node = document.createElementNS(NS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    parent.appendChild(node);
    return node;
  }
  const mono = {"font-family":"var(--mono)"};
  // "sₜ₊₂" as base text plus a small lowered tspan, so it renders in any mono font
  function T(parent, x, y, parts, attrs){
    const t = E(parent, "text", {x, y, ...mono, ...attrs});
    parts.forEach(([txt, isSub]) => {
      const span = E(t, "tspan", isSub ? {dy:"3", "font-size":"75%"} : {dy: "0"}, txt);
      if (isSub) E(t, "tspan", {dy:"-3"}, "");
    });
    return t;
  }
  const rel = k => k === 0 ? "t" : k > 0 ? `t+${k}` : `t−${-k}`;
  const PAST = 3, PITCH = 72, X0 = 100;
  const col = k => X0 + (k + PAST) * PITCH;
  const sx = k => col(k), ax = k => col(k) + 36;
  const BW = 30, BH = 26;
  const ROW = {A:44, B:104, C:164, D:230};
  const NOW = sx(0) + BW + 3;

  const maxR = () => Math.max(1, state.H - state.d);

  function render(){
    svg.replaceChildren();
    const defs = E(svg, "defs", {});
    for (const [id, colr] of [["tm-arrow", "var(--s5)"], ["tm-arrow-accent", "var(--accent)"]]){
      const m = E(defs, "marker", {id, viewBox:"0 0 10 10", refX:9, refY:5, markerWidth:5, markerHeight:5, orient:"auto"});
      E(m, "path", {d:"M0 1 L10 5 L0 9 Z", fill:colr});
    }
    const {d, H, ms} = state;
    const R = Math.min(state.R, maxR());
    const spare = H - d - R;          // written, never played
    const infFrames = ms / FRAME_MS;
    const fits = infFrames < R;
    const maxK = Math.max(H - 1, 5);

    for (let k = -PAST; k <= maxK; k++)
      E(svg, "text", {x:col(k) + 33, y:22, "text-anchor":"middle", "font-size":10, fill: k === 0 ? "var(--ink)" : "var(--ink3)", ...mono}, rel(k));
    E(svg, "line", {x1:NOW, y1:30, x2:NOW, y2:266, stroke:"var(--ink2)", "stroke-width":1.2, "stroke-dasharray":"5 4"});
    E(svg, "text", {x:NOW + 5, y:36, "font-size":9.5, fill:"var(--ink2)", ...mono}, "now");

    const label = (y, a, b) => {
      E(svg, "text", {x:6, y:y + 11, "font-size":10, fill:"var(--ink)", ...mono}, a);
      E(svg, "text", {x:6, y:y + 23, "font-size":9, fill:"var(--ink3)", ...mono}, b);
    };
    label(ROW.A, "model sees", "true present");
    label(ROW.B, "Slippi queue", d ? `held ${d} frames · locked` : "nothing ahead");
    label(ROW.C, "this plan", `H = ${H} slots`);
    label(ROW.D, "inference", `budget R = ${R} frame${R > 1 ? "s" : ""}`);

    /* row A: the observed stream, states and actions interleaved */
    const hist = E(svg, "g", {"data-part":"history"});
    for (let k = -PAST; k <= 0; k++){
      E(hist, "rect", {x:sx(k), y:ROW.A, width:BW, height:BH, rx:3, fill:"var(--s1)", opacity: k === 0 ? 1 : .55});
      T(hist, sx(k) + BW/2, ROW.A + 17, [["s", false], [rel(k), true]], {"text-anchor":"middle", "font-size":10, fill:"var(--panel)"});
      if (k < 0){
        E(hist, "rect", {x:ax(k), y:ROW.A + 4, width:BW, height:BH - 8, rx:3, fill:"var(--s2)", opacity:.5});
        T(hist, ax(k) + BW/2, ROW.A + 16, [["a", false], [rel(k), true]], {"text-anchor":"middle", "font-size":9.5, fill:"var(--panel)"});
      }
    }

    /* row B: actions already in Slippi's transport queue */
    const q = E(svg, "g", {"data-part":"committed"});
    for (let k = 0; k < d; k++){
      E(q, "rect", {x:ax(k), y:ROW.B, width:BW, height:BH, rx:3, fill:"var(--s5)", opacity:.28});
      E(q, "rect", {x:ax(k), y:ROW.B, width:BW, height:BH, rx:3, fill:"none", stroke:"var(--s5)", "stroke-width":1.2});
      T(q, ax(k) + BW/2, ROW.B + 17, [["a", false], [rel(k), true]], {"text-anchor":"middle", "font-size":9.5, fill:"var(--ink)"});
    }
    if (d){
      // a_t came from the plan that saw s_{t-d}: an elbow in the gap between rows A and B
      const yEl = ROW.B - 16, xTo = sx(-d) + BW/2, xFrom = ax(0) + BW/2;
      E(q, "path", {d:`M ${xFrom} ${ROW.B - 1} L ${xFrom} ${yEl} L ${xTo} ${yEl} L ${xTo} ${ROW.A + BH + 3}`,
        fill:"none", stroke:"var(--s5)", "stroke-width":1.1, "stroke-dasharray":"3 3", "marker-end":"url(#tm-arrow)"});
      T(q, xFrom - 8, yEl - 9, [["a", false], ["t", true], [" was decided from s", false], [rel(-d), true]],
        {"text-anchor":"end", "font-size":9, fill:"var(--ink3)"});
    } else {
      E(q, "text", {x:ax(0), y:ROW.B + 17, "font-size":9.5, fill:"var(--ink3)", ...mono}, "no transport delay · every action is still open");
    }

    /* row C: the decoder's H slots, d forced then R new */
    // siblings, never nested: a dimmed parent would multiply into a lit child
    const forced = E(svg, "g", {"data-part":"forced H"});
    const played = E(svg, "g", {"data-part":"R H"});
    const spareG = E(svg, "g", {"data-part":"H spare"});
    for (let k = 0; k < H; k++){
      const isForced = k < d, isPlayed = !isForced && (k - d) < R;
      const host = isForced ? forced : isPlayed ? played : spareG;
      if (isForced){
        E(host, "line", {x1:ax(k) + BW/2, y1:ROW.B + BH + 2, x2:ax(k) + BW/2, y2:ROW.C - 2, stroke:"var(--s3)", "stroke-width":1, "stroke-dasharray":"2 3", opacity:.7});
        E(host, "rect", {x:ax(k), y:ROW.C, width:BW, height:BH, rx:3, fill:"var(--s3)", opacity:.16});
        E(host, "rect", {x:ax(k), y:ROW.C, width:BW, height:BH, rx:3, fill:"none", stroke:"var(--s3)", "stroke-dasharray":"3 2"});
      } else if (isPlayed){
        E(host, "rect", {x:ax(k), y:ROW.C, width:BW, height:BH, rx:3, fill:"var(--s3)"});
      } else {
        E(host, "rect", {x:ax(k), y:ROW.C, width:BW, height:BH, rx:3, fill:"var(--s3)", opacity:.14});
        E(host, "rect", {x:ax(k), y:ROW.C, width:BW, height:BH, rx:3, fill:"none", stroke:"var(--ink3)", "stroke-width":1, opacity:.6});
        E(host, "line", {x1:ax(k) + 5, y1:ROW.C + 5, x2:ax(k) + BW - 5, y2:ROW.C + BH - 5, stroke:"var(--ink3)", "stroke-width":1, opacity:.6});
      }
      T(host, ax(k) + BW/2, ROW.C + 17, [["a", false], [rel(k), true]],
        {"text-anchor":"middle", "font-size":9.5, fill: isPlayed ? "var(--panel)" : isForced ? "var(--ink)" : "var(--ink3)"});
    }
    if (d) E(forced, "text", {x:ax(0), y:ROW.C + BH + 14, "font-size":9.5, fill:"var(--ink3)", ...mono}, "self-forced");
    E(played, "text", {x:ax(d), y:ROW.C + BH + 14, "font-size":9.5, fill:"var(--s3)", ...mono}, `${R} played`);
    if (spare > 0) E(spareG, "text", {x:ax(d + R), y:ROW.C + BH + 14, "font-size":9.5, fill:"var(--ink3)", ...mono}, `${spare} written, then replaced`);

    /* reaction: what s_t can first change. Label sits in the free space left of now. */
    const re = E(svg, "g", {"data-part":"reaction"});
    const xs = sx(0) + BW/2, xa = ax(d) + BW/2;
    E(re, "path", {d:`M ${xs} ${ROW.A + BH + 2} C ${xs} ${ROW.A + 72}, ${xa} ${ROW.C - 58}, ${xa} ${ROW.C - 3}`,
      fill:"none", stroke:"var(--accent)", "stroke-width":1.4, "marker-end":"url(#tm-arrow-accent)"});
    T(re, NOW - 10, ROW.C + 17, [["s", false], ["t", true], [" can first change a", false], [rel(d), true], [d ? `  →  lands ${d} frame${d > 1 ? "s" : ""} later` : "  →  next frame", false]],
      {"text-anchor":"end", "font-size":9.5, fill:"var(--accent)"});

    /* row D: inference time against the replan window */
    const lat = E(svg, "g", {"data-part":"latency"});
    const budgetW = R * PITCH;
    E(lat, "rect", {x:NOW, y:ROW.D, width:budgetW, height:24, rx:3, fill:"var(--s3)", opacity:.1});
    E(lat, "line", {x1:NOW + budgetW, y1:ROW.D - 4, x2:NOW + budgetW, y2:ROW.D + 28, stroke:"var(--s3)", "stroke-width":1.2});
    const w = Math.min(infFrames * PITCH, 9 * PITCH);
    T(lat, NOW + Math.max(budgetW, w) + 6, ROW.D + 16, [["next plan at s", false], [rel(R), true], [" · replaces the rest", false]], {"font-size":9.5, fill:"var(--s3)"});
    const okCol = fits ? "var(--gain)" : "var(--loss)";
    E(lat, "rect", {x:NOW, y:ROW.D + 6, width:w, height:12, rx:2, fill:okCol, opacity:.85});
    E(lat, "text", {x:NOW + 6, y:ROW.D + 15, "font-size":9, fill:"var(--panel)", ...mono}, `${ms} ms`);

    /* readout */
    const ro = E(svg, "g", {"data-part":"latency H R"});
    T(ro, 6, 296, [["d", false], ["transport", true], [` = ${d} · H = ${H} · R = ${R} · d`, false], ["inference", true], [` = ${ms} ms = ${infFrames.toFixed(1)} frames`, false]],
      {"font-size":10.5, fill:"var(--ink)"});
    const verdict = [["d", false], ["inference", true], [" < R ≤ H − d", false], ["transport", true],
      [` :  ${infFrames.toFixed(1)} ${fits ? "<" : "≮"} ${R} ≤ ${H - d}   ${fits ? "✓ fits" : "✗ the next plan is due before this one returns"}`, false]];
    T(ro, 6, 316, verdict, {"font-size":10.5, fill: fits ? "var(--gain)" : "var(--loss)"});

    cap.textContent =
      `${d ? `At frame t the game is applying the action decided from s(t−${d}). ` : ""}The decoder self-forces ${d} committed action${d === 1 ? "" : "s"} and writes ${H - d} new; ${R} of those ${R === 1 ? "is" : "are"} played before the next plan, ${spare} ${spare === 1 ? "is" : "are"} replaced unused. ${fits ? `A ${ms} ms pass fits inside the ${R}-frame window.` : `A ${ms} ms pass does not fit the ${R}-frame window: the queue runs dry before it returns.`}`;

    const key = fig.dataset.hi;
    if (key) fig.querySelectorAll("[data-part]").forEach(n => n.classList.toggle("lit", n.dataset.part.split(" ").includes(key)));
  }

  function syncPills(){
    for (const [id, key] of [["tm-d","d"],["tm-H","H"],["tm-R","R"],["tm-ms","ms"]]){
      document.getElementById(id).querySelectorAll("[data-v]").forEach(b => {
        const v = Number(b.dataset.v);
        const blocked = key === "R" && v > maxR();
        b.disabled = blocked;
        b.title = blocked ? `the decoder only writes H − d = ${maxR()} new actions` : "";
        b.setAttribute("aria-pressed", String(!blocked && v === (key === "R" ? Math.min(state.R, maxR()) : state[key])));
      });
    }
  }
  function pills(id, items, key){
    const host = document.getElementById(id);
    host.replaceChildren();
    items.forEach(([value, text]) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "pill"; b.textContent = text; b.dataset.v = String(value);
      b.addEventListener("click", () => { state[key] = value; syncPills(); render(); });
      host.appendChild(b);
    });
  }
  pills("tm-d", [[0, "0"], [1, "1"], [2, "2"], [3, "3"]], "d");
  pills("tm-H", [[4, "4 · live"], [5, "5"], [6, "6"]], "H");
  pills("tm-R", [[1, "1"], [2, "2"], [3, "3"], [4, "4"]], "R");
  pills("tm-ms", [[5, "5 ms"], [15, "15 ms"], [25, "25 ms"], [40, "40 ms"]], "ms");
  syncPills();
  render();
})();

/* ---------- glossary: term lights up its part of the toy figure ---------- */
function highlight(figId, key){
  const fig = document.getElementById(figId);
  if (!fig) return;
  fig.dataset.hi = key || "";
  fig.classList.toggle("hl", Boolean(key));
  fig.querySelectorAll("[data-part]").forEach(node =>
    node.classList.toggle("lit", Boolean(key) && node.dataset.part.split(" ").includes(key)));
}
document.querySelectorAll(".gterm[data-fig]").forEach(term => {
  const on = () => highlight(term.dataset.fig, term.dataset.hi);
  const offAll = () => highlight(term.dataset.fig, null);
  term.addEventListener("mouseenter", on);
  term.addEventListener("mouseleave", offAll);
  term.addEventListener("focus", on);
  term.addEventListener("blur", offAll);
});

/* ---------- per-section evidence tables ---------- */
function flash(e){
  const row = document.querySelector(`tr[data-r="${e.r}"]`);
  if (!row) return;
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1500);
}
document.querySelectorAll("tbody[data-ev]").forEach(tb => {
  const comp = tb.dataset.ev;
  EV.filter(e => e.c === comp).forEach(e => {
    const tr = document.createElement("tr");
    tr.dataset.r = e.r;
    const runs = e.runs.length
      ? `<div style="margin-top:4px">${e.runs.map(id => `<a class="chip" style="text-decoration:none" href="${WB(id)}">${id}</a>`).join(" ")}</div>` : "";
    tr.innerHTML =
      `<td><b style="font-weight:600">${e.t}</b><div class="small" style="margin-top:3px">${e.note}</div></td>` +
      `<td class="mono" style="font-size:.8rem; white-space:nowrap">${e.a}<br><span style="color:var(--ink3)">↓</span><br>${e.b}${runs}</td>` +
      `<td class="n ${cls(e.d)}">${fmt(e.d, true)}</td>` +
      `<td class="n ${e.l === null ? "" : cls(e.l)}">${fmt(e.l, true)}</td>` +
      `<td><span class="badge ${e.q}">${e.q}</span></td>`;
    tb.appendChild(tr);
  });
});

"use strict";
(function(){
const NS = "http://www.w3.org/2000/svg";
const A = "var(--indigo)", B = "var(--ochre)";
const tip = document.getElementById("xtip");
function put(html, x, y){
  tip.innerHTML = html; tip.classList.add("on");
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(Math.max(10, x + 14), window.innerWidth - r.width - 10) + "px";
  tip.style.top = Math.min(Math.max(10, y + 14), window.innerHeight - r.height - 10) + "px";
}
const off = () => tip.classList.remove("on");
function E(p, tag, a, txt){
  const n = document.createElementNS(NS, tag);
  for (const k in a) n.setAttribute(k, a[k]);
  if (txt !== undefined) n.textContent = txt;
  p.appendChild(n); return n;
}
function frame(id, W, H, pad){
  const s = document.getElementById(id);
  s.setAttribute("viewBox", `0 0 ${W} ${H}`);
  s.replaceChildren();
  return s;
}
function grid(s, pad, W, H, yTicks, y){
  yTicks.forEach(v => {
    const py = y(v);
    E(s, "line", {x1:pad.l, y1:py, x2:W - pad.r, y2:py,
      stroke: v === 0 ? "var(--ink3)" : "var(--hair)", "stroke-width": v === 0 ? 1 : 1});
    E(s, "text", {x:pad.l - 7, y:py + 3.6, "text-anchor":"end", "font-size":9.5,
      fill:"var(--ink3)", "font-family":"var(--mono)"}, v.toFixed(v % 1 === 0 ? 0 : 1));
  });
}
function legend(id, items){
  const host = document.getElementById(id);
  if (!host) return;
  host.replaceChildren();
  items.forEach(([c, label]) => {
    const sp = document.createElement("span");
    sp.innerHTML = `<i style="background:${c}"></i>${label}`;
    host.appendChild(sp);
  });
}

/* ---- 2. horizon panels ---- */
(function(){
  const xs = [1, 2, 4, 6];
  const panels = [
    ["chart-hz-a", "lg-hz",  {off:[-0.0453,-0.2924,-0.8457,-1.0581], on:[-0.0013,-0.0813,-0.4381,-0.6078]}],
    ["chart-hz-b", "lg-hz2", {off:[-0.3176,-0.2066,-0.3306,-0.3837], on:[-0.0740,-0.1428,-0.2504,-0.1644]}]
  ];
  panels.forEach(([id, lg, d]) => {
    legend(lg, [[B, "group chain off"], [A, "group chain on"]]);
    const W = 420, H = 240, pad = {l:38, r:52, t:14, b:36};
    const s = frame(id, W, H, pad);
    const x = i => pad.l + i / (xs.length - 1) * (W - pad.l - pad.r);
    const y = v => pad.t + (0.10 - v) / (0.10 + 1.20) * (H - pad.t - pad.b);
    grid(s, pad, W, H, [0, -0.4, -0.8, -1.2], y);
    xs.forEach((v, i) => E(s, "text", {x:x(i), y:H - pad.b + 16, "text-anchor":"middle",
      "font-size":10, fill:"var(--ink3)", "font-family":"var(--mono)"}, String(v)));
    E(s, "text", {x:(pad.l + W - pad.r)/2, y:H - 6, "text-anchor":"middle", "font-size":10,
      fill:"var(--ink3)", "font-family":"var(--mono)"}, "frames executed per plan");
    [["off", B, "off"], ["on", A, "on"]].forEach(([k, col, label]) => {
      const pts = d[k];
      E(s, "polyline", {points: pts.map((v, i) => `${x(i)},${y(v)}`).join(" "),
        fill:"none", stroke:col, "stroke-width":2, "stroke-linejoin":"round"});
      pts.forEach((v, i) => {
        E(s, "circle", {cx:x(i), cy:y(v), r:4.5, fill:col, stroke:"var(--panel)", "stroke-width":2});
        const hit = E(s, "circle", {cx:x(i), cy:y(v), r:12, fill:"transparent"});
        hit.addEventListener("mousemove", ev => put(
          `<span class="hd">group chain ${label}</span>${xs[i]} executed frame${xs[i] > 1 ? "s" : ""}<br>` +
          `stock LCB <b>${v.toFixed(4)}</b>`, ev.clientX, ev.clientY));
        hit.addEventListener("mouseleave", off);
      });
      E(s, "text", {x:x(xs.length - 1) + 9, y:y(pts[pts.length - 1]) + 3.6, "font-size":10.5,
        fill:col, "font-family":"var(--mono)"}, label);
    });
    E(s, "text", {x:pad.l, y:10, "font-size":9.5, fill:"var(--ink3)", "font-family":"var(--mono)"}, "stock LCB");
  });
})();

})();

"use strict";
(function(){
const CAP = {"models":[{"k":"L5","L":5,"d":320,"h":5,"p":6988015},{"k":"026base","L":8,"d":256,"h":4,"p":7081967},{"k":"L7","L":7,"d":448,"h":7,"p":17810159},{"k":"L10","L":10,"d":640,"h":10,"p":51154223},{"k":"L13","L":13,"d":832,"h":13,"p":111720303},{"k":"L16","L":16,"d":1024,"h":16,"p":205245039},{"k":"L18","L":18,"d":1152,"h":18,"p":293127087}],"budgets":[26,27,28,29,30],"positions":[67108864,134217728,268435456,536870912,1073741824],"delays":[1,2,4,6,8,10,12,14,16],"train":[[[1.3387,2.043,12.41,67108864],[1.29,1.9696,24.83,134217728],[1.2569,1.9181,49.66,268435456],[1.2329,1.8786,99.31,536870912],[1.2149,1.8489,198.63,1073741824]],[[1.3343,2.0359,12.02,67108864],[1.287,1.965,24.03,134217728],[1.2536,1.9136,48.07,268435456],[1.2301,1.875,96.13,536870912],[1.2129,1.8461,192.26,1073741824]],[[1.3327,2.0365,17.64,67108864],[1.2835,1.9621,35.29,134217728],[1.2486,1.9072,70.57,268435456],[1.2248,1.8685,141.14,536870912],[1.2077,1.8403,282.29,1073741824]],[[1.3016,1.9844,44.93,67108864],[1.2544,1.9122,89.86,134217728],[1.2206,1.8588,179.71,268435456],[1.1983,1.8218,359.43,536870912],[1.183,1.7956,718.85,1073741824]],[[1.288,1.9602,92.76,67108864],[1.24,1.8862,185.51,134217728],[1.2069,1.8336,371.03,268435456],[1.1851,1.7975,742.06,536870912],[1.2234,1.8479,1298.6,939524096]],[[1.2858,1.9581,132.07,67108864],[1.2382,1.8844,264.14,134217728],[1.205,1.8318,528.27,268435456],[1.1839,1.7966,1056.55,536870912],[1.1691,1.7711,2113.09,1073741824]],[[1.2799,1.947,202.81,67108864],[1.2287,1.8678,405.63,134217728],[1.1968,1.8159,811.25,268435456],[1.1761,1.781,1622.5,536870912],[7.3444,11.0217,2839.38,939524096]]],"cells":[[[[-3.484,-3.824,-74.7,-85.9,225,0.0,14.3],[-2.214,-2.421,-45.5,-53.9,174,0.0,20.0],[-2.108,-2.279,-44.7,-52.3,174,0.0,131.9],[-2.068,-2.235,-65.2,-72.8,170,0.0,38.7],[-1.831,-2.001,-72.8,-79.2,153,0.0,48.0],[-1.959,-2.131,-78.6,-85.7,166,0.0,56.5],[-2.322,-2.497,-88.7,-95.3,172,0.0,62.8],[-2.236,-2.39,-99.1,-105.4,174,0.0,73.4],[-2.302,-2.473,-108.8,-115.8,178,0.0,81.6]],[[-2.957,-3.276,-77.2,-86.4,202,0.0,14.0],[-2.22,-2.426,-52.6,-60.7,174,0.0,19.7],[-1.68,-1.852,-44.5,-52.2,148,0.0,118.4],[-1.572,-1.724,-61.8,-68.8,143,0.0,38.0],[-1.437,-1.581,-70.8,-78.1,140,0.0,45.2],[-1.512,-1.667,-81.1,-88.3,141,0.0,56.3],[-1.684,-1.83,-90.2,-96.9,143,0.0,61.1],[-1.728,-1.899,-97.9,-104.8,147,0.0,66.0],[-1.87,-2.02,-102.4,-109.1,156,0.0,72.4]],[[-3.435,-3.759,-85.5,-96.3,225,0.0,12.5],[-2.179,-2.411,-51.0,-60.4,179,0.0,18.4],[-1.32,-1.475,-34.6,-42.8,140,0.0,129.6],[-1.195,-1.355,-42.2,-49.8,130,0.0,38.4],[-1.181,-1.31,-60.3,-68.0,119,0.0,42.8],[-1.156,-1.296,-62.7,-70.0,122,0.0,49.2],[-1.581,-1.724,-79.6,-87.1,139,0.0,56.2],[-1.636,-1.787,-90.0,-97.5,144,0.0,63.7],[-1.77,-1.922,-94.7,-102.4,143,0.0,70.0]],[[-2.117,-2.411,-63.0,-75.9,171,0.0,13.3],[-1.992,-2.194,-62.2,-72.0,168,0.0,17.7],[-1.382,-1.53,-51.1,-59.1,134,0.0,125.0],[-1.119,-1.256,-50.8,-58.1,127,0.0,35.6],[-1.128,-1.285,-53.5,-61.0,121,0.008,43.0],[-1.247,-1.399,-64.4,-71.4,129,0.0,50.5],[-1.56,-1.705,-82.6,-89.0,139,0.0,60.6],[-1.598,-1.771,-93.6,-100.8,142,0.0,73.1],[-1.648,-1.801,-85.0,-92.1,150,0.0,79.8]],[[-3.617,-3.973,-92.6,-105.3,236,0.004,12.8],[-1.899,-2.075,-77.8,-88.0,162,0.0,18.0],[-1.071,-1.229,-44.5,-52.7,128,0.0,118.9],[-1.036,-1.184,-59.5,-67.8,91,0.0,47.6],[-0.968,-1.103,-40.1,-47.7,115,0.0,43.3],[-1.042,-1.183,-51.2,-58.5,114,0.0,53.7],[-1.192,-1.35,-65.1,-72.9,120,0.0,58.3],[-1.389,-1.548,-70.9,-78.3,138,0.0,63.6],[-1.523,-1.671,-80.6,-88.4,142,0.0,68.7]]],[[[-2.825,-3.109,-85.0,-96.4,200,0.005,17.8],[-2.37,-2.563,-64.8,-74.2,184,0.0,24.3],[-1.959,-2.147,-43.8,-51.5,163,0.006,170.2],[-1.762,-1.918,-63.1,-70.8,156,0.0,41.6],[-1.717,-1.889,-73.0,-80.8,146,0.0,54.6],[-1.816,-1.984,-82.7,-90.5,154,0.0,59.2],[-1.869,-2.015,-86.8,-94.2,151,0.0,66.6],[-1.838,-1.989,-98.3,-105.1,158,0.0,76.8],[-2.201,-2.371,-99.9,-107.1,168,0.0,78.6]],[[-3.561,-3.854,-101.3,-111.4,226,0.0,14.2],[-2.254,-2.46,-49.7,-58.4,178,0.0,20.7],[-1.638,-1.8,-46.5,-55.2,150,0.0,116.1],[-1.323,-1.463,-54.4,-62.1,129,0.0,36.3],[-1.457,-1.598,-70.6,-79.0,135,0.0,43.1],[-1.392,-1.533,-76.7,-83.9,133,0.0,51.5],[-1.532,-1.683,-78.6,-85.5,135,0.0,64.2],[-1.826,-1.97,-101.0,-108.1,152,0.0,64.8],[-1.906,-2.059,-101.9,-108.5,152,0.0,74.7]],[[-2.738,-2.998,-77.8,-88.6,198,0.0,13.9],[-1.39,-1.575,-33.7,-43.1,142,0.0,17.0],[-1.698,-1.864,-56.0,-64.0,155,0.0,119.3],[-1.242,-1.387,-44.1,-51.4,127,0.0,39.0],[-1.225,-1.367,-53.3,-60.6,124,0.0,42.5],[-1.344,-1.485,-73.1,-80.7,133,0.0,51.0],[-1.456,-1.602,-75.2,-82.6,130,0.0,56.0],[-1.845,-1.995,-85.3,-92.1,160,0.0,64.9],[-1.835,-1.988,-96.3,-103.2,147,0.0,77.8]],[[-2.592,-2.898,-87.3,-98.9,187,0.005,14.5],[-1.204,-1.36,-34.1,-43.1,125,0.0,19.5],[-1.387,-1.541,-63.7,-71.3,135,0.0,120.7],[-1.143,-1.277,-61.0,-68.9,116,0.0,37.7],[-1.01,-1.142,-60.2,-67.3,111,0.0,42.9],[-1.164,-1.308,-75.3,-82.5,115,0.0,52.2],[-1.401,-1.536,-77.3,-84.7,120,0.0,61.0],[-1.537,-1.685,-86.4,-93.4,133,0.0,70.0],[-1.678,-1.82,-98.7,-105.3,143,0.0,79.6]],[[-3.198,-3.533,-80.7,-92.5,218,0.0,13.4],[-1.719,-1.906,-44.3,-54.5,153,0.0,20.5],[-1.818,-1.998,-71.4,-80.2,145,0.0,136.2],null,null,null,null,null,null]],[[[-2.82,-3.108,-77.3,-88.0,200,0.0,13.8],[-2.238,-2.442,-50.9,-59.0,177,0.0,19.7],[-2.155,-2.366,-52.9,-60.6,173,0.006,116.4],[-1.6,-1.751,-67.1,-73.6,148,0.0,38.2],[-1.679,-1.841,-74.3,-81.2,145,0.0,45.8],[-1.811,-1.962,-81.1,-88.0,157,0.0,55.0],[-2.09,-2.246,-96.2,-103.1,170,0.0,62.6],[-2.214,-2.37,-104.3,-111.4,171,0.0,70.0],[-2.381,-2.531,-110.6,-117.2,172,0.0,80.9]],[[-2.541,-2.856,-62.1,-73.7,193,0.005,14.1],[-2.218,-2.447,-43.7,-52.1,182,0.0,20.4],[-1.91,-2.109,-42.8,-50.9,162,0.0,120.6],[-1.8,-1.961,-63.3,-70.5,155,0.0,38.6],[-1.414,-1.575,-56.8,-64.3,134,0.0,44.0],[-1.354,-1.501,-73.4,-80.7,125,0.0,51.6],[-1.684,-1.832,-87.5,-94.3,143,0.0,58.3],[-1.798,-1.945,-84.2,-91.0,148,0.0,64.1],[-1.745,-1.902,-101.6,-108.2,150,0.0,75.3]],[[-2.188,-2.484,-54.6,-67.1,173,0.0,19.5],[-1.336,-1.493,-20.1,-28.7,142,0.0,21.0],[-1.35,-1.505,-33.3,-41.1,135,0.0,139.3],[-1.405,-1.537,-54.6,-62.0,138,0.0,34.8],[-1.248,-1.386,-56.9,-64.9,131,0.0,44.0],[-1.494,-1.639,-76.4,-84.0,134,0.0,57.1],[-1.598,-1.751,-80.8,-87.9,141,0.0,61.4],[-1.69,-1.833,-83.3,-90.8,145,0.0,68.9],[-1.778,-1.929,-94.4,-101.7,155,0.0,78.7]],[[-0.542,-0.679,-5.2,-13.5,110,0.0,17.1],[-0.59,-0.741,8.3,0.0,111,0.009,22.2],[-0.755,-0.902,-7.5,-15.4,116,0.0,130.8],[-0.84,-0.99,-29.0,-36.7,112,0.0,34.6],[-0.909,-1.053,-48.0,-54.5,111,0.0,41.7],[-1.242,-1.391,-61.3,-69.1,129,0.0,49.8],[-1.113,-1.261,-63.4,-70.8,122,0.0,57.6],[-1.36,-1.525,-76.3,-83.2,130,0.008,64.7],[-1.574,-1.713,-82.0,-89.5,134,0.0,163.4]],[[-3.648,-4.039,-75.5,-87.9,232,0.0,13.5],[-1.569,-1.779,-32.1,-41.7,153,0.0,19.2],[-0.936,-1.08,-37.3,-44.6,113,0.0,123.0],[-1.418,-1.566,-94.3,-103.1,130,0.0,47.8],[-1.187,-1.327,-65.0,-72.8,120,0.0,45.4],[-1.374,-1.514,-86.6,-94.0,125,0.0,54.1],[-1.318,-1.467,-76.8,-84.2,133,0.0,60.5],[-1.501,-1.642,-80.9,-87.8,140,0.0,73.3],[-1.695,-1.842,-97.6,-104.9,145,0.0,180.7]]],[[[-3.434,-3.778,-87.2,-98.2,232,0.0,19.3],[-1.804,-1.993,-32.2,-39.7,166,0.0,26.6],[-1.617,-1.779,-47.1,-54.7,151,0.0,35.5],[-1.686,-1.847,-60.4,-68.1,149,0.0,45.0],[-1.751,-1.92,-70.8,-78.3,153,0.0,56.8],[-1.827,-1.989,-77.4,-85.1,155,0.0,68.9],[-1.87,-2.037,-89.4,-96.3,155,0.0,79.2],[-1.935,-2.097,-90.8,-98.0,159,0.0,90.3],[-2.206,-2.36,-105.0,-112.5,166,0.0,97.3]],[[-3.675,-4.1,-75.5,-87.8,242,0.0,14.4],[-2.283,-2.543,-40.2,-49.3,182,0.0,21.3],[-1.833,-2.031,-42.9,-51.3,157,0.0,31.1],[-1.536,-1.694,-57.8,-64.8,151,0.0,40.1],[-1.421,-1.568,-59.5,-67.1,139,0.0,55.8],[-1.501,-1.654,-68.9,-76.1,139,0.0,63.1],[-1.837,-1.99,-88.2,-95.5,153,0.0,68.8],[-1.722,-1.88,-84.8,-91.7,146,0.0,77.5],[-1.766,-1.913,-104.2,-111.0,148,0.0,87.2]],[[-3.34,-3.601,-72.5,-82.5,217,0.0,14.3],[-1.255,-1.436,-12.3,-20.8,138,0.0,21.3],[-1.465,-1.633,-30.0,-38.1,146,0.0,31.1],[-1.313,-1.466,-43.6,-50.9,136,0.0,41.8],[-1.757,-1.933,-60.0,-67.6,156,0.0,49.9],[-1.517,-1.672,-71.2,-78.8,141,0.0,58.6],[-1.784,-1.941,-85.9,-93.8,155,0.0,68.2],[-1.876,-2.021,-95.8,-103.6,160,0.0,79.6],[-1.986,-2.13,-103.3,-110.8,165,0.0,86.4]],[[-2.997,-3.312,-83.9,-94.8,208,0.0,13.6],[-1.624,-1.838,-38.8,-47.9,156,0.006,20.7],[-1.351,-1.519,-49.5,-58.3,137,0.0,30.6],[-1.129,-1.282,-40.7,-48.7,122,0.0,41.7],[-1.209,-1.345,-56.1,-62.6,123,0.0,50.7],[-1.214,-1.359,-62.2,-69.8,121,0.0,62.4],[-1.43,-1.569,-76.7,-83.6,133,0.0,73.1],[-1.641,-1.789,-105.0,-112.5,144,0.0,77.8],[-1.64,-1.781,-91.3,-98.4,139,0.0,88.2]],[[-2.704,-2.981,-93.5,-105.2,196,0.0,15.6],[-1.334,-1.528,-24.2,-34.3,136,0.0,26.5],[-1.044,-1.193,-26.8,-35.2,122,0.0,35.7],[-0.903,-1.033,-39.6,-47.2,106,0.0,41.5],[-1.009,-1.124,-51.6,-58.3,109,0.0,54.8],[-1.0,-1.14,-55.8,-63.9,116,0.008,61.3],[-1.658,-1.806,-96.2,-104.0,148,0.0,77.2],[-1.483,-1.624,-82.7,-90.6,104,0.0,87.7],[-1.332,-1.473,-80.0,-86.8,125,0.0,79.7]]],[[[-1.231,-1.393,-27.3,-35.7,126,0.0,16.0],[-1.622,-1.817,-45.7,-54.8,149,0.0,22.2],[-1.8,-1.962,-53.9,-62.5,156,0.0,32.8],[-1.624,-1.77,-68.7,-76.3,141,0.0,44.0],[-1.559,-1.716,-69.8,-77.0,137,0.0,54.9],[-1.85,-2.003,-78.4,-85.6,162,0.0,70.2],[-1.866,-2.014,-82.8,-90.2,159,0.0,74.5],[-1.837,-1.992,-93.3,-101.1,153,0.0,84.8],[-2.077,-2.22,-97.1,-104.5,163,0.0,92.8]],[[-0.809,-0.98,-11.9,-21.7,118,0.008,10.0],[-3.343,-3.585,-51.4,-60.5,217,0.0,14.4],[-1.572,-1.74,-43.7,-51.1,145,0.0,21.0],[-1.247,-1.409,-45.0,-51.4,128,0.0,27.5],[-1.08,-1.229,-49.0,-56.9,119,0.0,33.0],[-1.431,-1.579,-64.3,-71.6,138,0.0,39.2],[-1.473,-1.623,-84.5,-91.5,133,0.0,45.1],[-1.744,-1.895,-90.0,-96.3,149,0.0,51.4],[-1.804,-1.958,-95.2,-102.8,151,0.0,57.3]],[[-2.631,-2.94,-54.0,-65.2,191,0.0,16.4],[-1.915,-2.14,-42.1,-51.0,164,0.0,22.2],[-1.471,-1.641,-37.0,-45.2,146,0.007,34.0],[-1.319,-1.486,-52.8,-60.6,135,0.0,43.6],[-1.328,-1.471,-62.9,-70.7,132,0.0,58.1],[-1.766,-1.913,-77.8,-85.2,150,0.0,63.5],[-1.985,-2.154,-92.4,-100.5,164,0.0,81.1],[-1.625,-1.759,-79.7,-87.3,143,0.0,91.7],[-1.985,-2.133,-109.4,-116.4,162,0.0,102.9]],[[-1.681,-1.885,-82.4,-92.7,151,0.007,18.1],[-1.18,-1.368,-23.6,-32.6,140,0.0,22.5],[-1.515,-1.681,-44.0,-52.3,151,0.0,36.2],[-1.501,-1.668,-60.7,-69.0,140,0.0,47.0],[-0.845,-0.959,-37.2,-43.8,109,0.0,51.9],[-1.021,-1.157,-47.6,-55.7,116,0.0,58.2],[-1.332,-1.491,-58.8,-66.1,126,0.0,69.9],[-1.3,-1.435,-75.2,-82.8,124,0.0,77.3],[-1.548,-1.689,-83.9,-90.7,136,0.0,85.9]],[null,null,null,null,null,null,null,null,null]],[[[-1.849,-2.095,-53.8,-64.9,159,0.0,63.1],[-2.166,-2.357,-52.8,-61.5,172,0.0,63.9],[-2.334,-2.508,-67.2,-74.6,176,0.0,86.3],[-1.463,-1.626,-57.8,-65.3,138,0.0,91.2],[-1.545,-1.715,-70.7,-78.5,145,0.0,103.0],[-1.574,-1.737,-77.0,-84.1,134,0.0,111.7],[-1.865,-2.021,-93.8,-101.2,158,0.0,107.5],[-1.766,-1.908,-94.1,-100.5,149,0.0,111.5],[-2.048,-2.197,-97.5,-104.0,159,0.0,135.3]],[[-1.089,-1.381,-0.1,-10.8,135,0.0,45.1],[-1.78,-1.961,-35.7,-44.4,160,0.0,62.8],[-1.675,-1.848,-48.5,-56.8,149,0.0,72.9],[-1.383,-1.54,-55.8,-63.7,137,0.0,76.1],[-1.344,-1.498,-67.8,-75.2,132,0.0,86.8],[-1.587,-1.742,-74.6,-81.5,139,0.0,97.0],[-1.468,-1.63,-73.6,-80.5,134,0.0,107.1],[-1.625,-1.78,-90.0,-97.2,143,0.0,87.8],[-1.737,-1.894,-86.6,-93.4,141,0.0,97.6]],[[-0.82,-1.06,7.3,-3.0,122,0.0,49.1],[-1.558,-1.754,-17.9,-26.0,153,0.0,60.6],[-1.597,-1.836,-42.1,-53.4,79,0.0,80.2],[-0.899,-1.04,-23.3,-30.7,117,0.0,42.8],[-0.989,-1.135,-40.5,-48.1,114,0.0,50.1],[-1.251,-1.396,-57.6,-64.8,121,0.0,71.8],[-1.317,-1.459,-66.5,-73.7,129,0.0,70.7],[-1.457,-1.603,-77.6,-84.5,133,0.0,79.5],[-1.542,-1.688,-84.4,-91.8,131,0.0,88.4]],[[-0.218,-0.369,22.2,14.1,108,0.009,45.7],[-1.277,-1.453,-35.7,-44.9,141,0.0,61.1],[-0.664,-0.81,-12.8,-21.8,111,0.0,41.5],[-0.738,-0.866,-22.3,-29.7,108,0.0,41.9],[-0.877,-1.023,-34.2,-41.3,110,0.0,51.9],[-0.989,-1.136,-54.4,-61.3,116,0.0,63.2],[-1.091,-1.237,-62.1,-69.3,118,0.0,71.9],[-1.234,-1.352,-71.2,-79.1,117,0.0,82.3],[-1.458,-1.614,-78.4,-86.1,137,0.0,92.0]],[[-2.095,-2.367,-91.0,-104.2,170,0.006,54.3],[-1.357,-1.562,-24.4,-35.7,142,0.0,66.2],[-1.611,-1.77,-78.0,-86.8,149,0.0,81.7],[-1.425,-1.589,-66.2,-75.3,130,0.0,102.5],[-1.113,-1.257,-60.9,-68.9,122,0.0,101.5],[-1.484,-1.62,-90.9,-99.0,136,0.0,117.0],[-1.063,-1.191,-58.0,-65.4,110,0.0,134.5],[-1.776,-1.926,-111.7,-119.5,149,0.0,146.9],[-1.618,-1.76,-88.3,-95.0,139,0.0,291.6]]],[[[-2.927,-3.159,-86.7,-96.0,198,0.0,43.8],[-2.357,-2.573,-59.7,-68.7,181,0.0,52.4],[-2.135,-2.305,-63.8,-72.2,176,0.0,64.6],[-1.517,-1.689,-50.6,-57.7,136,0.0,76.7],[-1.668,-1.842,-70.8,-77.7,145,0.0,88.2],[-1.659,-1.821,-83.2,-90.2,149,0.007,103.7],[-1.958,-2.103,-86.4,-94.0,162,0.0,149.7],[-1.93,-2.083,-90.4,-97.0,160,0.0,135.1],[-1.991,-2.131,-102.2,-108.9,164,0.0,139.3]],[[-3.805,-4.141,-83.6,-95.0,240,0.0,41.3],[-2.529,-2.754,-42.4,-51.6,190,0.0,50.4],[-1.555,-1.738,-46.5,-54.5,143,0.0,70.5],[-1.166,-1.316,-34.6,-41.8,119,0.0,79.2],[-1.118,-1.258,-55.3,-63.2,120,0.0,93.6],[-1.499,-1.631,-82.9,-89.5,131,0.0,111.7],[-1.596,-1.751,-87.9,-96.0,135,0.0,120.2],[-1.591,-1.728,-95.6,-102.5,137,0.0,134.4],[-1.722,-1.876,-95.0,-101.4,147,0.0,143.7]],[[-0.713,-0.892,9.0,-1.2,120,0.0,49.9],[-1.607,-1.83,-16.9,-27.0,154,0.006,67.5],[-1.487,-1.648,-41.4,-49.2,146,0.0,88.2],[-1.582,-1.753,-57.9,-66.3,143,0.0,102.7],[-1.392,-1.536,-56.9,-64.3,131,0.0,116.0],[-1.334,-1.479,-68.8,-75.4,132,0.0,128.4],[-1.56,-1.699,-84.4,-91.7,140,0.0,148.3],[-1.782,-1.938,-81.3,-87.5,147,0.0,159.2],[-1.652,-1.784,-91.1,-98.1,145,0.0,321.9]],[[-0.42,-0.587,5.6,-4.3,108,0.0,60.4],[-1.427,-1.613,-34.0,-43.9,143,0.0,61.9],[-1.226,-1.382,-37.8,-46.0,127,0.0,90.6],[-1.172,-1.325,-57.2,-64.3,123,0.0,89.3],[-1.128,-1.268,-52.6,-60.7,119,0.0,119.6],[-0.988,-1.126,-56.0,-63.7,112,0.0,110.6],[-1.321,-1.448,-70.3,-78.1,124,0.0,122.7],[-1.396,-1.557,-70.4,-78.5,127,0.0,140.5],[-1.695,-1.835,-105.5,-112.8,146,0.0,145.8]],[null,null,null,null,null,null,null,null,null]]],"lat":[[[4.99,6.25,0.0,1],[6.36,8.05,0.0,1],[7.93,12.83,0.0,1],[9.26,11.62,0.0,1],[10.74,12.66,0.0,1],[12.1,14.22,0.0,1],[13.55,15.11,0.0,1],[14.93,16.42,0.0,1],[16.42,17.77,0.0,1]],[[5.18,5.99,0.0,1],[6.75,8.8,0.0,1],[8.14,11.06,0.0,1],[9.56,12.06,0.0,1],[11.06,12.86,0.0,1],[12.42,15.0,0.0,1],[13.87,15.34,0.0,1],[15.35,16.68,0.0,1],[16.89,18.21,0.0,1]],[[6.41,7.54,0.0,1],[8.21,10.31,0.0,1],[9.58,12.32,0.0,1],[10.87,14.95,0.0,1],[12.32,13.62,0.0,1],[13.75,15.25,0.0,1],[15.27,16.78,0.0,1],[16.77,18.43,0.0,1],[18.18,20.02,0.0,1]],[[10.76,11.66,0.0,1],[12.53,14.62,0.0,1],[14.22,17.3,0.0,1],[16.01,17.3,0.0,1],[17.62,19.39,0.0,1],[19.23,20.56,0.0,1],[21.12,22.26,0.0,1],[22.9,24.88,0.0,1],[24.67,26.57,0.0,1]],[[17.56,19.07,1.0,0],[19.6,21.67,0.0,1],[21.41,23.68,0.0,1],[23.39,25.18,0.01,0],[25.54,27.02,0.0,1],[27.41,29.01,0.0,1],[29.53,31.22,0.0,1],[31.64,33.41,0.0,1],[33.91,35.44,0.0,1]],[[28.92,29.66,1.0,0],[30.93,32.83,0.04,0],[32.96,35.92,0.0,1],[35.34,37.43,0.0,1],[37.25,39.25,0.01,0],[39.36,40.92,0.0,1],[41.07,43.02,0.0,1],[43.41,44.6,0.0,1],[45.44,46.85,0.0,1]],[[39.21,40.17,1.0,0],[41.19,43.07,1.0,0],[43.59,45.78,0.0,1],[45.91,47.93,0.0,1],[48.21,51.85,0.0,1],[50.69,52.65,0.0,1],[52.99,54.53,0.0,1],[55.61,57.32,0.0,1],[58.03,59.41,0.0,1]]]};
const NS = "http://www.w3.org/2000/svg";
const tip = document.getElementById("xtip");
function put(html, x, y){
  tip.innerHTML = html; tip.classList.add("on");
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(Math.max(10, x + 14), window.innerWidth - r.width - 10) + "px";
  tip.style.top = Math.min(Math.max(10, y + 14), window.innerHeight - r.height - 10) + "px";
}
const off = () => tip.classList.remove("on");
function E(parent, tag, attrs, text){
  const node = document.createElementNS(NS, tag);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  if (text !== undefined) node.textContent = text;
  parent.appendChild(node);
  return node;
}

const RAMP_LIGHT = ["#A9A2E9","#9188E0","#7A6FD4","#6255C5","#4B3EAE","#372B8F","#251C68"];
const RAMP_DARK  = ["#4A409B","#5A4FB8","#6C60D2","#8074E2","#9890EC","#B2ABF4","#CDC8FA"];
const isDark = () => {
  const stamped = document.documentElement.getAttribute("data-theme");
  if (stamped === "light" || stamped === "dark") return stamped === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
};
const ramp = i => (isDark() ? RAMP_DARK : RAMP_LIGHT)[i];

const METRICS = {
  stock:    {i:0, label:"net stocks / min",  short:"stocks",  diverging:true,  dp:2, max:4.2},
  stockLcb: {i:1, label:"stock LCB",         short:"LCB",     diverging:true,  dp:2, max:4.4},
  damage:   {i:2, label:"net damage / min",  short:"damage",  diverging:true,  dp:0, max:130},
  win:      {i:5, label:"win rate",          short:"win rate",diverging:false, dp:3, max:0.35}
};
const VIEWS = [
  ["grid",    "capability-grid",  "Capability grid"],
  ["data",    "data-curves",      "Data curves"],
  ["loss",    "loss-vs-play",     "Loss vs play"],
  ["latency", "latency-budget",   "Latency budget"]
];
const CONTROLS = {grid:["metric","budget"], data:["metric","delay"], loss:["delay"], latency:["delay"]};
const NOTES = {
  grid: "Every evaluated cell at one training budget. Colour and number are the selected metric; an empty cell was never run. Nothing here is positive: at this stage the model still loses stocks to the in-game CPU in every regime.",
  data: "The same metric against the number of positions the optimizer processed, one line per model size. If capacity were the binding constraint, the large models would separate cleanly from the small ones. They do not.",
  loss: "Validation loss on the x axis, closed-loop play on the y axis, one point per model and training budget. A tight upward line would mean loss predicts play. It is the strongest evidence on the page that it does not: the fit improves smoothly with size and data, and the ranking of the play does not follow. The 293M run at the largest budget is absent because it diverged.",
  latency: "Measured p95 replan latency on one RTX 3060 against the frame budget the chosen delay allows. Points under the dashed line fit inside the budget; points above it miss deadlines and cannot be deployed at that delay."
};

const state = {view:"grid", metric:"stock", budget:4, delay:0};
const fmtParams = p => p >= 1e6 ? (p/1e6).toFixed(p >= 1e8 ? 0 : 1) + "M" : (p/1e3).toFixed(0) + "k";
const fmtPos = v => (v/1e6 >= 1000) ? (v/1e9).toFixed(2) + "B" : (v/1e6).toFixed(0) + "M";
const fmtTick = v => v >= 1e8 ? (v/1e6).toFixed(0) + "M" : (v/1e6).toFixed(0) + "M";
function ylabel(svg, text){
  E(svg, "text", {x:14, y:H/2, "font-size":10, fill:"var(--ink3)", "font-family":"var(--mono)",
    "text-anchor":"middle", transform:`rotate(-90 14 ${H/2})`}, text);
}
const cell = (m, b, d) => CAP.cells[m][b][d];

function pillRow(host, items, key, onPick){
  host.replaceChildren();
  items.forEach(([value, label]) => {
    const btn = document.createElement("button");
    btn.type = "button"; btn.className = "pill"; btn.textContent = label; btn.dataset.v = String(value);
    btn.addEventListener("click", () => { onPick(value); syncPills(); render(); });
    host.appendChild(btn);
  });
}
function syncPills(){
  const marks = [
    ["cap-views", state.view], ["cap-metric", state.metric],
    ["cap-budget", String(state.budget)], ["cap-delay", String(state.delay)]
  ];
  marks.forEach(([id, value]) => document.getElementById(id).querySelectorAll("[data-v]")
    .forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(value)))));
  document.querySelectorAll(".exp-controls .fieldset").forEach(f => {
    f.hidden = !CONTROLS[state.view].includes(f.dataset.ctl);
  });
  document.getElementById("cap-title").textContent = VIEWS.find(v => v[0] === state.view)[2];
  document.getElementById("cap-note").textContent = NOTES[state.view];
  const legend = document.getElementById("cap-ramp");
  legend.hidden = state.view === "grid";
  if (!legend.hidden){
    legend.replaceChildren();
    const head = document.createElement("span");
    head.textContent = "model size";
    head.style.letterSpacing = ".14em";
    head.style.textTransform = "uppercase";
    legend.appendChild(head);
    CAP.models.forEach((m, i) => {
      const chip = document.createElement("span");
      chip.style.display = "inline-flex";
      chip.style.alignItems = "center";
      chip.style.gap = "5px";
      const swatch = document.createElement("i");
      swatch.style.background = ramp(i);
      chip.append(swatch, document.createTextNode(fmtParams(m.p)));
      legend.appendChild(chip);
    });
  }
}

const W = 900, H = 400;
function frame(){
  const svg = document.getElementById("cap-svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  return svg;
}
function axisY(svg, ticks, y, x0, x1, dp){
  ticks.forEach(v => {
    const py = y(v);
    E(svg, "line", {x1:x0, y1:py, x2:x1, y2:py, stroke: v === 0 ? "var(--ink3)" : "var(--hair)", "stroke-width":1});
    E(svg, "text", {x:x0 - 8, y:py + 3.6, "text-anchor":"end", "font-size":9.5, fill:"var(--ink3)",
      "font-family":"var(--mono)"}, v.toFixed(dp));
  });
}

/* ---- view: capability grid ---- */
function drawGrid(svg){
  const met = METRICS[state.metric];
  const cw = 78, ch = 34, x0 = 138, y0 = 78;
  E(svg, "text", {x:0, y:22, "font-size":11, fill:"var(--ink2)", "font-family":"var(--mono)"},
    `${met.label} · ${fmtPos(CAP.positions[state.budget])} positions processed`);
  E(svg, "text", {x:x0, y:46, "font-size":10, fill:"var(--ink3)", "font-family":"var(--mono)"},
    "control delay d, in frames — replan interval is also d, decode horizon is 2d−1");
  CAP.delays.forEach((d, i) => E(svg, "text", {x:x0 + i*(cw + 6) + cw/2, y:68, "text-anchor":"middle",
    "font-size":11, fill:"var(--ink2)", "font-family":"var(--mono)"}, "d=" + d));
  CAP.models.forEach((model, mi) => {
    const y = y0 + mi*(ch + 6);
    E(svg, "text", {x:126, y:y + 15, "text-anchor":"end", "font-size":11.5, fill:"var(--ink)",
      "font-family":"var(--mono)"}, fmtParams(model.p));
    E(svg, "text", {x:126, y:y + 27, "text-anchor":"end", "font-size":9.5, fill:"var(--ink3)",
      "font-family":"var(--mono)"}, `L${model.L}·d${model.d}`);
    CAP.delays.forEach((d, di) => {
      const x = x0 + di*(cw + 6);
      const c = cell(mi, state.budget, di);
      if (!c){
        E(svg, "rect", {x, y, width:cw, height:ch, rx:3, fill:"none", stroke:"var(--hair)",
          "stroke-dasharray":"3 3"});
        E(svg, "text", {x:x + cw/2, y:y + 21, "text-anchor":"middle", "font-size":11,
          fill:"var(--ink3)", "font-family":"var(--mono)"}, "—");
        return;
      }
      const v = c[met.i];
      let fill, alpha;
      if (met.diverging){
        alpha = Math.min(Math.abs(v) / met.max, 1) * 0.62;
        fill = v >= 0 ? "var(--gain)" : "var(--loss)";
      } else {
        alpha = Math.min(v / met.max, 1) * 0.7 + 0.06;
        fill = ramp(Math.min(6, Math.round(Math.min(v / met.max, 1) * 6)));
      }
      E(svg, "rect", {x, y, width:cw, height:ch, rx:3, fill, "fill-opacity":alpha.toFixed(3)});
      E(svg, "text", {x:x + cw/2, y:y + 21, "text-anchor":"middle", "font-size":11.5, fill:"var(--ink)",
        "font-family":"var(--mono)"}, v.toFixed(met.dp));
      const hit = E(svg, "rect", {x, y, width:cw, height:ch, rx:3, fill:"transparent", class:"cap-cell"});
      hit.addEventListener("mousemove", ev => put(
        `<span class="hd">${fmtParams(model.p)} · d=${d}</span>` +
        `net stocks <b>${c[0].toFixed(2)}</b> · LCB ${c[1].toFixed(2)}<br>` +
        `net damage <b>${c[2].toFixed(1)}</b> · LCB ${c[3].toFixed(1)}<br>` +
        `win rate ${(c[5]*100).toFixed(1)} % over ${c[4]} matches<br>` +
        `<span style="color:var(--ink3)">replan ${d} · horizon ${2*d - 1} · p95 ${c[6]} ms</span>`,
        ev.clientX, ev.clientY));
      hit.addEventListener("mouseleave", off);
    });
  });
  E(svg, "text", {x:0, y:372, "font-size":10, fill:"var(--ink3)", "font-family":"var(--mono)"},
    met.diverging ? "deeper red = worse · deeper green = better · 0 would mean even with the CPU"
                  : "deeper = higher");
}

/* ---- view: scaling curves ---- */
function drawData(svg){
  const met = METRICS[state.metric];
  const pad = {l:56, r:104, t:34, b:46};
  const xs = CAP.positions;
  const x = i => pad.l + i/(xs.length - 1) * (W - pad.l - pad.r);
  let lo = Infinity, hi = -Infinity;
  CAP.models.forEach((m, mi) => CAP.budgets.forEach((b, bi) => {
    const c = cell(mi, bi, state.delay);
    if (c){ lo = Math.min(lo, c[met.i]); hi = Math.max(hi, c[met.i]); }
  }));
  const span = (hi - lo) || 1;
  lo -= span*0.12; hi += span*0.12;
  const y = v => pad.t + (hi - v)/(hi - lo) * (H - pad.t - pad.b);
  const ticks = [0,1,2,3,4].map(k => lo + k*(hi - lo)/4);
  E(svg, "text", {x:0, y:20, "font-size":11, fill:"var(--ink2)", "font-family":"var(--mono)"},
    `${met.label} · control delay d = ${CAP.delays[state.delay]} frames`);
  ylabel(svg, met.label);
  axisY(svg, ticks, y, pad.l, W - pad.r, met.dp);
  xs.forEach((v, i) => E(svg, "text", {x:x(i), y:H - pad.b + 18, "text-anchor":"middle", "font-size":10,
    fill:"var(--ink3)", "font-family":"var(--mono)"}, fmtPos(v)));
  E(svg, "text", {x:(pad.l + W - pad.r)/2, y:H - 8, "text-anchor":"middle", "font-size":10,
    fill:"var(--ink3)", "font-family":"var(--mono)"}, "positions the optimizer processed");
  const ends = [];
  CAP.models.forEach((model, mi) => {
    const pts = [];
    CAP.budgets.forEach((b, bi) => {
      const c = cell(mi, bi, state.delay);
      if (c) pts.push([x(bi), y(c[met.i]), c, bi]);
    });
    if (!pts.length) return;
    const col = ramp(mi);
    E(svg, "polyline", {points: pts.map(p => `${p[0]},${p[1]}`).join(" "), fill:"none", stroke:col,
      "stroke-width":2, "stroke-linejoin":"round"});
    pts.forEach(p => {
      E(svg, "circle", {cx:p[0], cy:p[1], r:4, fill:col, stroke:"var(--panel)", "stroke-width":1.6});
      const hit = E(svg, "circle", {cx:p[0], cy:p[1], r:12, fill:"transparent"});
      hit.addEventListener("mousemove", ev => put(
        `<span class="hd">${fmtParams(model.p)} · ${fmtPos(xs[p[3]])} positions</span>` +
        `${met.label} <b>${p[2][met.i].toFixed(met.dp)}</b><br>` +
        `<span style="color:var(--ink3)">${p[2][4]} matches · d=${CAP.delays[state.delay]}</span>`,
        ev.clientX, ev.clientY));
      hit.addEventListener("mouseleave", off);
    });
    const last = pts[pts.length - 1];
    ends.push({x:last[0] + 9, y:last[1] + 3.6, col, text:fmtParams(model.p)});
  });
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++){
    if (ends[i].x === ends[i - 1].x && ends[i].y - ends[i - 1].y < 12) ends[i].y = ends[i - 1].y + 12;
  }
  ends.forEach(e => E(svg, "text", {x:e.x, y:e.y, "font-size":10, fill:e.col,
    "font-family":"var(--mono)"}, e.text));
}

/* ---- view: validation loss against closed-loop play ---- */
function drawLoss(svg){
  const pad = {l:56, r:96, t:34, b:52};
  const pts = [];
  CAP.models.forEach((model, mi) => CAP.budgets.forEach((b, bi) => {
    const t = CAP.train[mi][bi], c = cell(mi, bi, state.delay);
    if (t && c && t[0] < 2) pts.push({mi, bi, nll:t[0], score:c[0], c, model, pos:t[3]});
  }));
  const nx = pts.map(p => p.nll), ny = pts.map(p => p.score);
  const xlo = Math.min(...nx) - 0.006, xhi = Math.max(...nx) + 0.006;
  const ylo = Math.min(...ny) - 0.3, yhi = Math.max(...ny) + 0.3;
  const x = v => pad.l + (v - xlo)/(xhi - xlo) * (W - pad.l - pad.r);
  const y = v => pad.t + (yhi - v)/(yhi - ylo) * (H - pad.t - pad.b);
  E(svg, "text", {x:0, y:20, "font-size":11, fill:"var(--ink2)", "font-family":"var(--mono)"},
    `validation loss against play · control delay d = ${CAP.delays[state.delay]} frames`);
  axisY(svg, [0,1,2,3,4].map(k => ylo + k*(yhi - ylo)/4), y, pad.l, W - pad.r, 2);
  [0,1,2,3,4].forEach(k => {
    const v = xlo + k*(xhi - xlo)/4;
    E(svg, "text", {x:x(v), y:H - pad.b + 18, "text-anchor":"middle", "font-size":10,
      fill:"var(--ink3)", "font-family":"var(--mono)"}, v.toFixed(3));
  });
  E(svg, "text", {x:(pad.l + W - pad.r)/2, y:H - 22, "text-anchor":"middle", "font-size":10,
    fill:"var(--ink3)", "font-family":"var(--mono)"}, "validation next-frame loss, in bits");
  E(svg, "text", {x:pad.l, y:H - 6, "font-size":10, fill:"var(--ink3)", "font-family":"var(--mono)"},
    "← better fit");
  ylabel(svg, "net stocks / min  →  better play");
  CAP.models.forEach((model, mi) => {
    const own = pts.filter(p => p.mi === mi).sort((a, b) => a.bi - b.bi);
    if (own.length > 1) E(svg, "polyline", {points: own.map(p => `${x(p.nll)},${y(p.score)}`).join(" "),
      fill:"none", stroke:ramp(mi), "stroke-width":1, opacity:".28"});
  });
  pts.forEach(p => {
    const col = ramp(p.mi);
    E(svg, "circle", {cx:x(p.nll), cy:y(p.score), r:5.5, fill:col, stroke:"var(--panel)", "stroke-width":1.6});
    const hit = E(svg, "circle", {cx:x(p.nll), cy:y(p.score), r:13, fill:"transparent"});
    hit.addEventListener("mousemove", ev => put(
      `<span class="hd">${fmtParams(p.model.p)} · ${fmtPos(p.pos)} positions</span>` +
      `validation loss <b>${p.nll.toFixed(4)}</b> bits<br>net stocks <b>${p.score.toFixed(2)}</b> / min<br>` +
      `<span style="color:var(--ink3)">${p.c[4]} matches · d=${CAP.delays[state.delay]}</span>`,
      ev.clientX, ev.clientY));
    hit.addEventListener("mouseleave", off);
  });
  const n = pts.length;
  const rank = arr => { const idx = arr.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const out = Array(n); idx.forEach(([, i], k) => out[i] = k); return out; };
  const rx = rank(pts.map(p => p.nll)), ry = rank(pts.map(p => p.score));
  let d2 = 0; for (let i = 0; i < n; i++) d2 += (rx[i] - ry[i])**2;
  const rho = 1 - 6*d2/(n*(n*n - 1));
  E(svg, "text", {x:W - pad.r + 10, y:pad.t + 8, "font-size":11, fill:"var(--ink)",
    "font-family":"var(--mono)"}, "ρ = " + rho.toFixed(2));
  E(svg, "text", {x:W - pad.r + 10, y:pad.t + 22, "font-size":9.5, fill:"var(--ink3)",
    "font-family":"var(--mono)"}, "Spearman");
  E(svg, "text", {x:W - pad.r + 10, y:pad.t + 34, "font-size":9.5, fill:"var(--ink3)",
    "font-family":"var(--mono)"}, `n = ${n}`);
}

/* ---- view: latency against the frame budget ---- */
function drawLatency(svg){
  const pad = {l:56, r:30, t:34, b:52};
  const d = CAP.delays[state.delay];
  const budget = d * 1000/60;
  const rows = CAP.models.map((m, mi) => ({m, mi, l:CAP.lat[mi][state.delay]})).filter(r => r.l);
  const hi = Math.max(budget, ...rows.map(r => r.l[1])) * 1.16;
  const lx = v => Math.log10(v);
  const x = v => pad.l + (lx(v) - lx(5e6))/(lx(4e8) - lx(5e6)) * (W - pad.l - pad.r);
  const y = v => pad.t + (hi - v)/hi * (H - pad.t - pad.b);
  E(svg, "text", {x:0, y:20, "font-size":11, fill:"var(--ink2)", "font-family":"var(--mono)"},
    `measured p95 replan latency · delay d = ${d} allows ${budget.toFixed(1)} ms`);
  axisY(svg, [0, hi*0.25, hi*0.5, hi*0.75].map(v => Math.round(v)), y, pad.l, W - pad.r, 0);
  ylabel(svg, "p95 replan latency, ms");
  E(svg, "line", {x1:pad.l, y1:y(budget), x2:W - pad.r, y2:y(budget), stroke:"var(--accent)",
    "stroke-width":1.6, "stroke-dasharray":"6 4"});
  E(svg, "text", {x:W - pad.r - 4, y:y(budget) - 8, "text-anchor":"end", "font-size":10.5,
    fill:"var(--accent)", "font-family":"var(--mono)"}, `frame budget ${budget.toFixed(1)} ms`);
  [1e7, 3e7, 1e8, 3e8].forEach(v => E(svg, "text", {x:x(v), y:H - pad.b + 18, "text-anchor":"middle",
    "font-size":10, fill:"var(--ink3)", "font-family":"var(--mono)"}, fmtTick(v)));
  E(svg, "text", {x:(pad.l + W - pad.r)/2, y:H - 12, "text-anchor":"middle", "font-size":10,
    fill:"var(--ink3)", "font-family":"var(--mono)"}, "parameters, log scale");
  let lastX = -99, flip = false;
  rows.forEach(r => {
    const fits = r.l[1] <= budget;
    const col = fits ? ramp(r.mi) : "var(--loss)";
    const cx = x(r.m.p), cy = y(r.l[1]);
    E(svg, "line", {x1:cx, y1:y(r.l[0]), x2:cx, y2:cy, stroke:col, "stroke-width":1.4, opacity:".6"});
    E(svg, "circle", {cx, cy, r:6, fill: fits ? col : "var(--panel)", stroke:col, "stroke-width":2});
    flip = (cx - lastX) < 26 ? !flip : false;
    lastX = cx;
    E(svg, "text", {x:cx, y:cy + (flip ? 22 : -13), "text-anchor":"middle", "font-size":10, fill:col,
      "font-family":"var(--mono)"}, fmtParams(r.m.p));
    const hit = E(svg, "circle", {cx, cy, r:14, fill:"transparent"});
    hit.addEventListener("mousemove", ev => put(
      `<span class="hd">${r.m.k} · ${fmtParams(r.m.p)}</span>` +
      `p50 <b>${r.l[0]}</b> ms · p95 <b>${r.l[1]}</b> ms<br>` +
      `budget at d=${d} is ${budget.toFixed(1)} ms<br>` +
      `<span style="color:${fits ? "var(--gain)" : "var(--loss)"}">${fits ? "fits inside the budget" : "misses the deadline"}</span>` +
      `<br><span style="color:var(--ink3)">deadline misses ${(r.l[2]*100).toFixed(0)} %</span>`,
      ev.clientX, ev.clientY));
    hit.addEventListener("mouseleave", off);
  });
}

function render(){
  const svg = frame();
  if (state.view === "grid") drawGrid(svg);
  else if (state.view === "data") drawData(svg);
  else if (state.view === "loss") drawLoss(svg);
  else drawLatency(svg);
}

pillRow(document.getElementById("cap-views"), VIEWS.map(v => [v[0], v[1]]), "view", v => { state.view = v; });
function viewFromHash(){
  const slug = decodeURIComponent(location.hash.slice(1));
  const hit = VIEWS.find(v => v[1] === slug);
  if (!hit) return;
  state.view = hit[0]; syncPills(); render();
  document.getElementById("cap").scrollIntoView({block:"start"});
}
window.addEventListener("hashchange", viewFromHash);
pillRow(document.getElementById("cap-metric"),
  Object.entries(METRICS).map(([k, m]) => [k, m.short]), "metric", v => { state.metric = v; });
pillRow(document.getElementById("cap-budget"),
  CAP.positions.map((p, i) => [i, fmtPos(p)]), "budget", v => { state.budget = Number(v); });
pillRow(document.getElementById("cap-delay"),
  CAP.delays.map((d, i) => [i, "d=" + d]), "delay", v => { state.delay = Number(v); });
syncPills();
render();
viewFromHash();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
document.addEventListener("hal:theme", () => setTimeout(render, 0));
})();

})();
