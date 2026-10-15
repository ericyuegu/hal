/* Fig: reward → return → weight for one game, with the AWR knobs live. Data: figures_src/awr-ledger.json (inlined at build). */
(function () {
  'use strict';
  var DATA = window.HALFIG_AWR_LEDGER;
  var svg = document.getElementById('ledger-svg');
  if (!DATA || !svg) return;
  var NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, String(attrs[k])); });
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function unrle(pairs) {
    var out = [];
    pairs.forEach(function (p) { for (var i = 0; i < p[1]; i++) out.push(p[0]); });
    return out;
  }
  var FPS = DATA.fps || 60;
  var games = DATA.games.map(function (g) {
    return {label: g.label, p1: g.p1, p2: g.p2, stage: g.stage, n: g.frames,
      pct: [unrle(g.p1Percent).map(function (v) { return v / 10; }), unrle(g.p2Percent).map(function (v) { return v / 10; })],
      stock: [unrle(g.p1Stock), unrle(g.p2Stock)]};
  });
  var O52 = {hl: 8, beta: 199.5, cap: 3.5, stock: 120, win: 50, damage: 1};
  var state = {game: 0, ego: 1, base: 'ema', hl: 8, beta: 199.5, cap: 3.5, frame: null};

  var gamesEl = document.getElementById('ledger-games');
  games.forEach(function (g, i) {
    var b = document.createElement('button');
    b.type = 'button'; b.textContent = g.p1 + ' vs ' + g.p2; b.title = g.label + ' · ' + g.stage;
    b.setAttribute('aria-pressed', String(i === 0));
    b.addEventListener('click', function () { state.game = i; state.frame = null; sync(); });
    gamesEl.appendChild(b);
  });
  function bindSeg(id, key, parse) {
    var host = document.getElementById(id);
    host.querySelectorAll('button').forEach(function (b) {
      b.addEventListener('click', function () { state[key] = parse(b.dataset[key]); sync(); });
    });
  }
  bindSeg('ledger-ego', 'ego', Number);
  bindSeg('ledger-base', 'base', String);
  var hl = document.getElementById('ledger-hl'), beta = document.getElementById('ledger-beta'), cap = document.getElementById('ledger-cap');
  hl.addEventListener('input', function () { state.hl = Number(hl.value); render(); });
  beta.addEventListener('input', function () { state.beta = Math.pow(10, Number(beta.value)); render(); });
  cap.addEventListener('input', function () { state.cap = Number(cap.value); render(); });
  document.getElementById('ledger-reset').addEventListener('click', function () {
    state.hl = O52.hl; state.beta = O52.beta; state.cap = O52.cap;
    hl.value = O52.hl; beta.value = Math.log10(O52.beta); cap.value = O52.cap;
    render();
  });

  /* Reward parameters follow O52; the baseline and unnormalized weights are illustrative. */
  function compute() {
    var g = games[state.game];
    var e = state.ego - 1, o = 1 - e;
    var n = g.n, gamma = Math.pow(0.5, 1 / (state.hl * FPS));
    var r = new Float64Array(n), dealt = new Float64Array(n), taken = new Float64Array(n);
    var deathOpp = new Uint8Array(n), deathEgo = new Uint8Array(n);
    for (var t = 1; t < n; t++) {
      var dd = g.pct[o][t] - g.pct[o][t - 1], dt = g.pct[e][t] - g.pct[e][t - 1];
      dealt[t] = dd > 0 ? dd : 0; taken[t] = dt > 0 ? dt : 0;
      deathOpp[t] = g.stock[o][t] < g.stock[o][t - 1] ? 1 : 0;
      deathEgo[t] = g.stock[e][t] < g.stock[e][t - 1] ? 1 : 0;
      var winOpp = deathOpp[t] && g.stock[o][t] === 0 ? 1 : 0, winEgo = deathEgo[t] && g.stock[e][t] === 0 ? 1 : 0;
      r[t] = O52.damage * (dealt[t] - taken[t]) + O52.stock * (deathOpp[t] - deathEgo[t]) + O52.win * (winOpp - winEgo);
    }
    var G = new Float64Array(n), acc = 0;
    for (t = n - 1; t >= 0; t--) { acc = r[t] + gamma * acc; G[t] = acc; }
    var B = new Float64Array(n);
    if (state.base === 'mean') {
      var m = 0; for (t = 0; t < n; t++) m += G[t]; m /= n;
      B.fill(m);
    } else {
      var a = 1 - Math.pow(0.5, 1 / (10 * FPS)), mm = G[0];
      for (t = 0; t < n; t++) { B[t] = mm; mm += a * (G[t] - mm); }
    }
    var w = new Float64Array(n), clipped = 0;
    for (t = 0; t < n; t++) { var raw = Math.exp((G[t] - B[t]) / state.beta); w[t] = Math.min(raw, state.cap); if (raw >= state.cap) clipped++; }
    return {g: g, n: n, gamma: gamma, r: r, dealt: dealt, taken: taken, deathOpp: deathOpp, deathEgo: deathEgo, G: G, B: B, w: w, clipped: clipped, e: e, o: o};
  }
  document.getElementById('ledger-full').addEventListener('change', render);
  var last = null;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.awrLedgerRows = function () {
    if (!last) return [];
    var rows = [];
    for (var t = 0; t < last.n; t += 60) rows.push({second: t / FPS, ego_percent: last.g.pct[last.e][t], opp_percent: last.g.pct[last.o][t], ego_stocks: last.g.stock[last.e][t], opp_stocks: last.g.stock[last.o][t], reward: +last.r[t].toFixed(2), G: +last.G[t].toFixed(2), baseline: +last.B[t].toFixed(2), w: +last.w[t].toFixed(3)});
    return rows;
  };

  /* ---- drawing ---- */
  var W = 880, L = 60, R = 16, TOP = 22;
  var strips = [['events', 70], ['return G', 100], ['illustrative weight · log scale', 100]];
  var H = TOP + strips.reduce(function (s, x) { return s + x[1] + 22; }, 0) + 16;
  var plotStart = 0, plotEnd = 1;
  function X(t, n) { return L + ((t - plotStart) / (plotEnd - plotStart)) * (W - L - R); }

  function render() {
    var c = compute(); last = c;
    W = window.matchMedia('(max-width:760px)').matches ? 340 : 880;
    var event = c.deathEgo.findIndex(function(v){return v;});
    if (event < 0) event = c.deathOpp.findIndex(function(v){return v;});
    if (event < 0) event = Math.floor(c.n/2);
    var full = document.getElementById('ledger-full').checked;
    plotStart = full ? 0 : Math.max(0,event-12*FPS);
    plotEnd = full ? c.n-1 : Math.min(c.n-1,event+6*FPS);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var n = c.n, y = TOP;
    // time axis ticks every 30 s
    var seconds = plotEnd / FPS, tickStep = full ? 30 : 5;
    for (var s = Math.ceil(plotStart/FPS/tickStep)*tickStep; s <= seconds; s += tickStep) {
      var x = X(s * FPS, n);
      svg.appendChild(el('line', {x1: x, y1: TOP, x2: x, y2: H - 16, class: 'grid'}));
      svg.appendChild(el('text', {x: x, y: H - 4, class: 'tick', 'text-anchor': 'middle'}, s + ' s'));
    }
    // strip 1: events
    var evH = strips[0][1];
    svg.appendChild(el('text', {x: L, y: y - 6, class: 'lbl'}, 'replay events'));
    var mid = y + evH / 2;
    svg.appendChild(el('line', {x1: L, y1: mid, x2: W - R, y2: mid, class: 'axis'}));
    var maxD = 1; for (var t = plotStart; t <= plotEnd; t++) maxD = Math.max(maxD, c.dealt[t], c.taken[t]);
    for (t = Math.max(1,plotStart); t <= plotEnd; t++) {
      if (c.dealt[t]) svg.appendChild(el('line', {x1: X(t, n), y1: mid, x2: X(t, n), y2: mid - (c.dealt[t] / maxD) * (evH / 2 - 8), class: 'ev-dealt'}));
      if (c.taken[t]) svg.appendChild(el('line', {x1: X(t, n), y1: mid, x2: X(t, n), y2: mid + (c.taken[t] / maxD) * (evH / 2 - 8), class: 'ev-taken'}));
      if (c.deathOpp[t]) svg.appendChild(el('path', {d: 'M ' + X(t, n) + ' ' + (y + 2) + ' l -5 8 l 10 0 z', class: 'death-opp'}));
      if (c.deathEgo[t]) svg.appendChild(el('path', {d: 'M ' + X(t, n) + ' ' + (y + evH - 2) + ' l -5 -8 l 10 0 z', class: 'death-ego'}));
    }
    y += evH + 22;
    // strip 2: return + baseline
    var gH = strips[1][1];
    svg.appendChild(el('text', {x: L, y: y - 6, class: 'lbl'}, 'return G · baseline'));
    var lo = Infinity, hi = -Infinity;
    for (t = plotStart; t <= plotEnd; t++) { lo = Math.min(lo, c.G[t], c.B[t]); hi = Math.max(hi, c.G[t], c.B[t]); }
    if (hi - lo < 1) { hi += 1; lo -= 1; }
    function Yg(v) { return y + gH - ((v - lo) / (hi - lo)) * gH; }
    if (lo < 0 && hi > 0) svg.appendChild(el('line', {x1: L, y1: Yg(0), x2: W - R, y2: Yg(0), class: 'axis'}));
    svg.appendChild(el('text', {x: L - 6, y: y + 8, class: 'tick', 'text-anchor': 'end'}, hi.toFixed(0)));
    svg.appendChild(el('text', {x: L - 6, y: y + gH, class: 'tick', 'text-anchor': 'end'}, lo.toFixed(0)));
    svg.appendChild(el('path', {d: linePath(c.B, n, Yg), class: 'line-B'}));
    svg.appendChild(el('path', {d: linePath(c.G, n, Yg), class: 'line-G'}));
    y += gH + 22;
    // strip 3: weight, log scale
    var wH = strips[2][1];
    svg.appendChild(el('text', {x: L, y: y - 6, class: 'lbl'}, 'weight w'));
    var wmin = Infinity, wmax = 0;
    for (t = plotStart; t <= plotEnd; t++) { wmin = Math.min(wmin, c.w[t]); wmax = Math.max(wmax, c.w[t]); }
    var wlo = Math.log(Math.min(wmin / 1.3, 0.7)), whi = Math.log(Math.max(wmax * 1.3, 1.5));
    function Yw(v) { return y + wH - ((Math.log(Math.max(v, 0.05)) - wlo) / (whi - wlo)) * wH; }
    svg.appendChild(el('line', {x1: L, y1: Yw(1), x2: W - R, y2: Yw(1), class: 'rule-one'}));
    svg.appendChild(el('text', {x: L - 6, y: Yw(1) + 3, class: 'tick', 'text-anchor': 'end'}, '1'));
    if (Math.log(state.cap) <= whi) {
      svg.appendChild(el('line', {x1: L, y1: Yw(state.cap), x2: W - R, y2: Yw(state.cap), class: 'rule-cap'}));
      svg.appendChild(el('text', {x: L - 6, y: Yw(state.cap) + 3, class: 'tick', 'text-anchor': 'end'}, 'cap ' + state.cap.toFixed(1)));
    }
    svg.appendChild(el('path', {d: linePath(c.w, n, Yw) + ' L ' + X(plotEnd, n) + ' ' + Yw(1) + ' L ' + L + ' ' + Yw(1) + ' Z', class: 'area-w'}));
    svg.appendChild(el('path', {d: linePath(c.w, n, Yw), class: 'line-w'}));
    // playhead
    if (state.frame !== null) {
      var px = X(state.frame, n);
      svg.appendChild(el('line', {x1: px, y1: TOP, x2: px, y2: H - 16, class: 'head'}));
    }
    document.getElementById('ledger-hl-v').textContent = state.hl.toFixed(1) + ' s · γ ' + c.gamma.toFixed(5);
    document.getElementById('ledger-beta-v').textContent = state.beta.toFixed(0);
    document.getElementById('ledger-cap-v').textContent = state.cap.toFixed(1);
    readout(c);
  }
  function linePath(arr, n, Y) {
    var step = Math.max(1, Math.floor(n / 1400)), d = '';
    for (var t = plotStart; t <= plotEnd; t += step) d += (t > plotStart ? ' L ' : 'M ') + X(t, n).toFixed(1) + ' ' + Y(arr[t]).toFixed(1);
    return d;
  }
  function readout(c) {
    var host = document.getElementById('ledger-readout');
    var t = state.frame;
    if (t === null) {
      host.innerHTML = '<span>' + c.g.label + ' · ' + c.g.stage + '</span><span>ego <b>' + (state.ego === 1 ? c.g.p1 : c.g.p2) + '</b></span><span>frames clipped at cap <b>' + (100 * c.clipped / c.n).toFixed(1) + '%</b></span>';
      return;
    }
    host.innerHTML = '<span>t <b>' + (t / FPS).toFixed(2) + ' s</b></span><span>stocks <b>' + c.g.stock[c.e][t] + '–' + c.g.stock[c.o][t] + '</b></span><span>percent <b>' + c.g.pct[c.e][t].toFixed(0) + '% · ' + c.g.pct[c.o][t].toFixed(0) + '%</b></span><span>G <b>' + c.G[t].toFixed(1) + '</b></span><span>A <b>' + (c.G[t] - c.B[t]).toFixed(1) + '</b></span><span>w <b>' + c.w[t].toFixed(2) + '</b></span>';
  }
  svg.addEventListener('pointerdown', selectFrame);
  svg.addEventListener('pointermove', function(e){if(e.pointerType==='mouse') selectFrame(e);});
  function selectFrame(e) {
    var r = svg.getBoundingClientRect();
    var x = (e.clientX - r.left) / r.width * W;
    var n = games[state.game].n;
    state.frame = Math.max(0, Math.min(n - 1, Math.round(plotStart + (x - L) / (W - L - R) * (plotEnd - plotStart))));
    render();
  }
  svg.addEventListener('mouseleave', function () { state.frame = null; render(); });
  function sync() {
    gamesEl.querySelectorAll('button').forEach(function (b, i) { b.setAttribute('aria-pressed', String(i === state.game)); });
    document.querySelectorAll('#ledger-ego button').forEach(function (b) { b.setAttribute('aria-pressed', String(Number(b.dataset.ego) === state.ego)); });
    document.querySelectorAll('#ledger-base button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.base === state.base)); });
    render();
  }
  beta.value = Math.log10(state.beta);
  window.matchMedia('(max-width:760px)').addEventListener('change', render);
  sync();
}());
