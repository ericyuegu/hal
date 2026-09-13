/* Fig: the policy end to end — one causal trunk, one causal decoder, five cards. Labels only. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('arch-overview');
  var cards = document.getElementById('arch-cards');
  var detail = document.getElementById('arch-detail');
  if (!D || !svg || !cards || !detail) return;

  var FAINT = 'var(--ink-faint)';
  // sₜ is paired with the action that preceded it; the decoder writes aₜ, aₜ₊₁, …
  var frames = [['sₜ₋₂', 'aₜ₋₃'], ['sₜ₋₁', 'aₜ₋₂'], ['sₜ', 'aₜ₋₁']];
  var outs = ['aₜ', 'aₜ₊₁', 'aₜ₊₂', 'aₜ₊₃'];
  var TW = 96, TH = 34; // one token width for inputs and outputs

  /* ---------------------------------------------------------------- overview */
  var hits = [];
  function renderOverview() {
    D.empty(svg);
    hits = [];
    var mk = D.marker(svg, 'arrow', FAINT);
    svg.setAttribute('viewBox', '0 0 880 330');
    var centers = [155, 440, 725];

    var hist = D.part(svg, 'history');
    frames.forEach(function (f, i) {
      D.jointToken(hist, centers[i] - TW / 2, 24, TW, TH, f[0], f[1]);
      D.wire(hist, centers[i], 58, centers[i], 88, mk);
    });
    D.note(hist, 88, 45, '···', 'middle');
    hits.push(['history', 90, 18, 700, 48]);

    var trunk = D.part(svg, 'trunk');
    D.stacked(trunk, 70, 96, 740, 56, 'CAUSAL CONTEXT TRANSFORMER');
    hits.push(['trunk', 62, 96, 748, 64]);

    var temporal = D.part(svg, 'temporal');
    D.wire(temporal, centers[2], 152, centers[2], 184, mk);
    D.text(temporal, centers[2] + 8, 172, 'hₜ', {class: 'diagram-label'});
    D.box(temporal, centers[2] - 110, 188, 220, 40, 'CAUSAL ACTION DECODER');
    D.wire(temporal, centers[2], 228, centers[2], 252, mk);
    hits.push(['temporal', centers[2] - 114, 154, 232, 78]);

    var groups = D.part(svg, 'groups');
    var play = D.part(svg, 'play');
    var n = 3, gap = 8;
    var x0 = centers[2] - (TW * n + gap * (n - 1)) / 2;
    outs.slice(0, n).forEach(function (label, i) {
      D.token(i < 2 ? play : groups, x0 + i * (TW + gap), 258, TW, TH, label, i < 2 ? 'played' : '');
    });
    D.note(groups, x0 + n * (TW + gap) + 2, 279, '···');
    var rowEnd = x0 + n * (TW + gap) - gap;
    D.note(play, x0, 312, 'played', 'start', 'play');
    D.note(play, rowEnd, 312, 'replaced by the next plan', 'end');
    hits.push(['groups', x0 - 4, 254, TW * n + gap * (n - 1) + 8, 42]);
    hits.push(['play', x0 - 4, 300, TW * n + gap * (n - 1) + 8, 18]);

    hits.forEach(function (h) {
      var key = keyFor(h[0]);
      var r = D.el('rect', {x: h[1], y: h[2], width: h[3], height: h[4], class: 'hit', tabindex: 0, role: 'button', 'aria-label': SPEC[key].t});
      r.addEventListener('click', function () { select(selected === key ? null : key); });
      r.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(selected === key ? null : key); } });
      svg.appendChild(r);
    });
  }
  function keyFor(moduleId) { return {history: 'token', trunk: 'trunk', temporal: 'decoder', groups: 'frame', play: 'loop'}[moduleId]; }

  /* ---------------------------------------------------------------- card drawings: same grammar, one token width per drawing */
  function mechToken(s) {
    var mk = D.marker(s, 'arrow', FAINT);
    s.setAttribute('viewBox', '0 0 880 160');
    var rows = [['game state', 24], ['characters · stage · player identity', 64], ['previous controller frame  aₜ₋₁', 104]];
    rows.forEach(function (r) {
      s.appendChild(D.el('rect', {x: 16, y: r[1], width: 300, height: 30, class: 'model-token'}));
      D.text(s, 28, r[1] + 19, r[0], {class: 'diagram-label'});
    });
    D.path(s, 'M 320 39 C 380 39, 400 76, 452 76', mk);
    D.path(s, 'M 320 79 C 380 79, 400 82, 452 82', mk);
    D.path(s, 'M 320 119 C 380 119, 400 88, 452 88', mk);
    D.jointToken(s, 460, 65, 130, 34, 'sₜ', 'aₜ₋₁');
  }
  function mechTrunk(s) {
    var mk = D.marker(s, 'arrow', FAINT);
    s.setAttribute('viewBox', '0 0 880 170');
    var w = 96, gap = 8, y0 = 20, rowH = 30, step = 44, x0 = 16;
    var states = ['hₜ₋₂', 'hₜ₋₁', 'hₜ'];
    frames.forEach(function (_, i) {
      var y = y0 + i * step;
      var given = frames.slice(0, i + 1);
      given.forEach(function (f, k) { D.jointToken(s, x0 + k * (w + gap), y, w, rowH, f[0], f[1], k === i ? '' : 'given'); });
      var end = x0 + given.length * (w + gap) - gap;
      D.wire(s, end + 6, y + rowH / 2, 630, y + rowH / 2, mk);
      D.text(s, end + 30, y + rowH / 2 - 5, states[i], {class: 'diagram-label', 'text-anchor': 'middle'});
    });
    D.box(s, 640, y0, 220, 2 * step + rowH, 'CAUSAL ACTION DECODER');
  }
  function mechDecoder(s) {
    var mk = D.marker(s, 'arrow', FAINT);
    s.setAttribute('viewBox', '0 0 880 300');
    var hist = D.part(s, 'history');
    D.jointToken(hist, 16, 96, 130, 34, 'sₜ', 'aₜ₋₁');
    var temporal = D.part(s, 'temporal');
    D.wire(temporal, 146, 113, 178, 113, mk);
    D.text(temporal, 152, 105, 'hₜ', {class: 'diagram-label'});
    D.stacked(temporal, 186, 20, 670, 262, 'CAUSAL ACTION DECODER');
    var prefixes = [['aₜ₋₁'], ['aₜ₋₁', 'aₜ'], ['aₜ₋₁', 'aₜ', 'aₜ₊₁'], ['aₜ₋₁', 'aₜ', 'aₜ₊₁', 'aₜ₊₂']];
    var w = 48, gap = 6, x0 = 262, cells = 620;
    prefixes.forEach(function (p, i) {
      var y = 62 + i * 44;
      D.wire(temporal, 214, y + 14, 254, y + 14, mk);
      p.forEach(function (g, k) { D.token(temporal, x0 + k * (w + gap), y, w, 28, g, 'given'); });
      var px = x0 + p.length * (w + gap);
      D.token(temporal, px, y, w, 28, outs[i]);
      D.wire(temporal, px + w + 6, y + 14, cells - 7, y + 14, mk);
      ['C', 'M', 'T', 'B'].forEach(function (g, k) {
        var cx = cells + k * 38;
        D.token(temporal, cx, y, 31, 28, g);
        if (k) D.wire(temporal, cx - 7, y + 14, cx - 1, y + 14, mk);
      });
    });
  }
  function mechFrame(s) {
    var mk = D.marker(s, 'arrow', FAINT);
    s.setAttribute('viewBox', '0 0 880 150');
    var names = {C: 'C-stick', M: 'main stick', T: 'triggers', B: 'buttons'};
    var chain = ['C', 'M', 'T', 'B'];
    var w = 40, gap = 6, x0 = 60;
    for (var i = 1; i < chain.length; i++) {
      var y = 20 + (i - 1) * 40;
      D.wire(s, 16, y + 14, 52, y + 14, mk);
      chain.slice(0, i).forEach(function (g, k) { D.token(s, x0 + k * (w + gap), y, w, 28, g, 'given'); });
      D.token(s, x0 + i * (w + gap), y, w, 28, chain[i]);
    }
    chain.forEach(function (c, i) { D.note(s, 330, 38 + i * 26, c + '  ' + names[c]); });
  }
  function mechLoop(s) {
    s.setAttribute('viewBox', '0 0 880 150');
    var cellW = 72, gap = 8, x0 = 150;
    function planRow(y, label, offset) {
      D.note(s, 16, y + 19, label);
      for (var f = 0; f < 6; f++) {
        var cls = f < 2 ? 'queued' : f < 4 ? 'played' : '';
        D.token(s, x0 + (offset + f) * (cellW + gap), y, cellW, 30, offset + f === 0 ? 'aₜ' : 'aₜ₊' + subscript(offset + f), cls);
      }
    }
    D.note(s, x0 + cellW + gap / 2, 22, 'already in the game', 'middle');
    D.note(s, x0 + 3 * (cellW + gap) - gap / 2, 22, 'played', 'middle', 'play');
    D.note(s, x0 + 5 * (cellW + gap) - gap / 2, 22, 'replaced', 'middle');
    planRow(30, 'plan at t', 0);
    planRow(88, 'plan at t+2', 2);
  }
  function subscript(n) { return n === 0 ? '' : String(n).split('').map(function (c) { return '₀₁₂₃₄₅₆₇₈₉'[Number(c)]; }).join(''); }

  var SPEC = {
    token: {n: '01', t: 'Observation token', module: 'history', mech: mechToken, p: [
      'Every frame becomes one token: the game state at that frame, who is playing, and the controller frame that preceded it, concatenated and projected once.',
      'There are no hand-built geometric features; earlier experiments that added them never produced a clean win.']},
    trunk: {n: '02', t: 'Context trunk', module: 'trunk', mech: mechTrunk, p: [
      'A causal Transformer over the token sequence. Each position produces a state for its frame from that frame and the ones before it. All positions are computed in one pass, so one training window supervises the decoder at every position.',
      'In play only the newest state is used.']},
    decoder: {n: '03', t: 'Action decoder', module: 'temporal', mech: mechDecoder, p: [
      'A causal Transformer over the controller frames. It is handed the trunk state and the frames written so far and produces the next one, group by group. In training every prefix of the target chunk goes through it in parallel, teacher-forced; in play it advances one frame at a time on a cache.',
      'It is trained to write further ahead than the game uses.']},
    frame: {n: '04', t: 'Controller frame', module: 'groups', mech: mechFrame, p: [
      'Inside a frame the four controller groups are chosen in a fixed order, each conditioned on the ones before it.']},
    loop: {n: '05', t: 'Play loop', module: 'play', mech: mechLoop, p: [
      'Online play holds every input for two frames before the game applies it. A plan is handed the two frames already committed, writes the frames after them, and two of those are played before the next plan.',
      'The frame the game applied returns as the action half of the next token.']}
  };
  var ORDER = ['token', 'trunk', 'decoder', 'frame', 'loop'];
  var selected = null;

  function select(key) {
    selected = key;
    D.focus(svg, key ? SPEC[key].module : null);
    cards.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-selected', String(b.dataset.stage === key)); });
    detail.replaceChildren();
    if (!key) { detail.hidden = true; return; }
    var s = SPEC[key];
    var row = document.createElement('div'); row.className = 'row';
    var h = document.createElement('h5'); h.textContent = s.n + ' · ' + s.t;
    var close = document.createElement('button'); close.type = 'button'; close.className = 'close'; close.textContent = 'close';
    close.addEventListener('click', function () { select(null); });
    row.appendChild(h); row.appendChild(close);
    detail.appendChild(row);
    var mech = document.createElement('div'); mech.className = 'mech';
    var ms = D.el('svg', {class: 'model-diagram', id: 'arch-mech-' + key, role: 'img', 'aria-label': s.t});
    mech.appendChild(ms);
    detail.appendChild(mech);
    s.mech(ms);
    var prose = document.createElement('div'); prose.className = 'prose';
    s.p.forEach(function (t) { var p = document.createElement('p'); p.textContent = t; prose.appendChild(p); });
    detail.appendChild(prose);
    detail.hidden = false;
  }

  ORDER.forEach(function (key) {
    var s = SPEC[key];
    var b = document.createElement('button');
    b.type = 'button'; b.setAttribute('role', 'tab'); b.dataset.stage = key; b.setAttribute('aria-selected', 'false');
    b.innerHTML = '<span class="n">' + s.n + '</span><span class="t">' + s.t + '</span>';
    b.addEventListener('click', function () { select(selected === key ? null : key); });
    cards.appendChild(b);
  });
  renderOverview();
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && selected) select(null); });
}());
