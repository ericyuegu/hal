/* Fig: the policy end to end as one dataflow. Vectors are bars whose width is proportional to their
   size (no numbers, abstract names); sequences stack downward. Five bands; the cards focus a band. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('arch-overview');
  var cards = document.getElementById('arch-cards');
  var detail = document.getElementById('arch-detail');
  if (!D || !svg || !cards || !detail) return;

  var FAINT = 'var(--ink-faint)';
  var PX = 0.5; // one hidden unit → half a pixel; d = 384, d′ = 128, dₐ = 16, dₒ = 16, d_h = 256
  var W = {x: 268 * PX, cs: 20 * PX, ea: 16 * PX, d: 384 * PX, dd: 128 * PX, po: 16 * PX, dh: 256 * PX};
  var V = {c: 9, m: 65, t: 25, b: 256}; // group vocabularies of the current codec, for the logits bars
  var BH = 18;
  var mk, mka;

  function bar(target, x, y, w, label, cls) {
    target.appendChild(D.el('rect', {x: x, y: y, width: w, height: BH, class: 'model-token' + (cls ? ' ' + cls : '')}));
    if (label && w >= 34) D.text(target, x + w / 2, y + 13, label, {'text-anchor': 'middle', class: 'diagram-label' + (cls === 'given' ? ' given-label' : '')});
    return x + w;
  }
  function op(target, x, y, w, label) {
    target.appendChild(D.el('rect', {x: x, y: y, width: w, height: BH + 2, class: 'model-box'}));
    D.text(target, x + w / 2, y + 14, label, {'text-anchor': 'middle', class: 'diagram-label'});
    return x + w;
  }
  function stackedBox(target, x, y, w, h, title, note) {
    [6, 3].forEach(function (o) { target.appendChild(D.el('rect', {x: x - o, y: y + o, width: w, height: h, class: 'model-layer'})); });
    target.appendChild(D.el('rect', {x: x, y: y, width: w, height: h, class: 'model-box'}));
    D.text(target, x + w / 2, y + h / 2 + (note ? -2 : 4), title, {'text-anchor': 'middle', class: 'diagram-title'});
    if (note) D.text(target, x + w / 2, y + h / 2 + 13, note, {'text-anchor': 'middle', class: 'diagram-note'});
  }
  function under(target, x, w, y, label, cls) { D.note(target, x + w / 2, y, label, 'middle', cls); }
  function brace(target, x1, x2, y) { // a concat bracket under a row of bars
    D.path(target, 'M ' + x1 + ' ' + (y - 4) + ' L ' + x1 + ' ' + y + ' L ' + x2 + ' ' + y + ' L ' + x2 + ' ' + (y - 4), null, {ghost: true});
  }
  function bandTitle(target, y, n, title, note) {
    D.text(target, 16, y, n, {class: 'diagram-note'});
    D.text(target, 44, y, title, {class: 'diagram-title'});
    if (note) D.text(target, 884, y, note, {'text-anchor': 'end', class: 'diagram-note'});
  }
  function rule(y) { svg.appendChild(D.el('line', {x1: 16, y1: y, x2: 884, y2: y, class: 'model-wire ghost'})); }
  function sub(n) { return String(n).split('').map(function (c) { return '₀₁₂₃₄₅₆₇₈₉'[Number(c)]; }).join(''); }

  var hits = [];
  function render() {
    D.empty(svg);
    hits = [];
    mk = D.marker(svg, 'arrow', FAINT); mka = D.marker(svg, 'arrow-accent', 'var(--oxblood)');
    svg.setAttribute('viewBox', '0 0 900 864');

    /* ---- 01 observation token: the pieces of one frame, concatenated and projected once ---- */
    var g1 = D.part(svg, 'token');
    var y = 20;
    bandTitle(g1, y, '01', 'observation token', 'one per frame · width ∝ size');
    var r = y + 20, x = 40;
    x = bar(g1, x, r, W.x, 'xₜ');
    D.note(g1, 40, r + 32, 'xₜ · game state of both players and their followers');
    var cs0 = x + 4;
    x = bar(g1, x + 4, r, W.cs, '');
    for (var i = 0; i < 4; i++) x = bar(g1, x + 4, r, W.ea, '', 'given');
    D.note(g1, cs0, r + 46, 'characters · stage · e(aₜ₋₁), the previous controller frame');
    brace(g1, 40, x, r + 24);
    D.wire(g1, x + 8, r + 9, x + 34, r + 9, mk);
    x = op(g1, x + 36, r - 1, 56, 'linear');
    D.wire(g1, x + 4, r + 9, x + 30, r + 9, mk);
    var u0 = x + 32;
    bar(g1, u0, r, W.d, 'uₜ');
    under(g1, u0, W.d, r + 32, 'd');
    hits.push(['token', 10, y - 14, 880, 76]);
    rule(r + 58);

    /* ---- 02 context trunk: causal blocks over the frames seen; every position gets a state ---- */
    var g2 = D.part(svg, 'trunk');
    y = r + 80;
    bandTitle(g2, y, '02', 'context trunk', 'N causal Transformer blocks over the last L frames');
    var rows = ['uₜ₋₂', 'uₜ₋₁', 'uₜ'], hs = ['hₜ₋₂', 'hₜ₋₁', 'hₜ'];
    var bx = 300, bw = 180, by = y + 18, bh = 3 * 26 + 6;
    stackedBox(g2, bx, by, bw, bh, 'CAUSAL BLOCKS ×N', 'causal over frames');
    rows.forEach(function (label, i) {
      var ry = y + 20 + i * 26;
      bar(g2, 40, ry, W.d, label, i === 2 ? '' : 'given');
      D.wire(g2, 40 + W.d + 6, ry + 9, bx - 8, ry + 9, mk);
      D.wire(g2, bx + bw + 6, ry + 9, 540 - 8, ry + 9, i === 2 ? mka : mk, {accent: i === 2});
      bar(g2, 540, ry, W.d, hs[i], i === 2 ? 'played' : 'given');
    });
    under(g2, 540, W.d, y + 20 + 3 * 26 + 24, 'd · only hₜ is used in play', 'play');
    D.note(g2, 40, y + 20 + 3 * 26 + 24, 'each frame attends to itself and the frames before it');
    hits.push(['trunk', 10, y - 14, 880, 136]);
    rule(y + 132);

    /* ---- 03 action decoder: one token per future frame, causal blocks over those ---- */
    var g3 = D.part(svg, 'decoder');
    y = y + 154;
    bandTitle(g3, y, '03', 'action decoder', 'one token per future frame k · K frames · causal over k');
    var ks = [1, 2, 3];
    var dbx = 520, dbw = 160, dby = y + 18, dbh = 3 * 26 + 6;
    stackedBox(g3, dbx, dby, dbw, dbh, 'CAUSAL BLOCKS ×M', 'causal over k');
    ks.forEach(function (k, i) {
      var ry = y + 20 + i * 26, cx = 40;
      cx = bar(g3, cx, ry, W.d, 'norm(hₜ)', 'given');
      var e0 = cx + 4;
      for (var j = 0; j < 4; j++) cx = bar(g3, cx + 4, ry, W.ea, '', 'given');
      cx = bar(g3, cx + 4, ry, W.po, '', 'given');
      if (i === 2) {
        D.note(g3, e0, ry + 30, 'e(aₜ₊ₖ₋₂) · pₖ');
        D.note(g3, 40, ry + 46, 'norm(hₜ) is the trunk state, copied into every token · e(aₜ₊ₖ₋₂) the previous written frame · pₖ the offset embedding');
      }
      D.wire(g3, cx + 8, ry + 9, cx + 30, ry + 9, mk);
      cx = op(g3, cx + 32, ry - 1, 56, 'linear');
      D.wire(g3, cx + 4, ry + 9, cx + 24, ry + 9, mk);
      cx = bar(g3, cx + 26, ry, W.dd, 'v' + sub(k));
      D.wire(g3, cx + 6, ry + 9, dbx - 8, ry + 9, mk);
      D.wire(g3, dbx + dbw + 6, ry + 9, dbx + dbw + 30, ry + 9, mk);
      bar(g3, dbx + dbw + 32, ry, W.dd, 's' + sub(k), k === 2 ? 'played' : '');
      if (i === 2) under(g3, dbx + dbw + 32, W.dd, ry + 30, 'd′ · frame state');
    });
    D.note(g3, 438, y + 20 + 3 * 26 + 4, '···  k = K', 'middle');
    D.note(g3, 40, y + 20 + 3 * 26 + 52, 'vₖ attends only to v₁ … vₖ₋₁ · in play only the first few frames are written');
    hits.push(['decoder', 10, y - 14, 880, 170]);
    rule(y + 166);

    /* ---- 04 controller frame: the four heads of one frame state, in order (opened in Fig 5) ---- */
    var g4 = D.part(svg, 'frame');
    y = y + 188;
    bandTitle(g4, y, '04', 'controller frame', 'the four heads of s₂, in order · opened up in the O43 figure');
    var CX = [250, 420, 590, 760], names = ['C-stick', 'main stick', 'triggers', 'buttons'], vocab = [V.c, V.m, V.t, V.b];
    var YS = y + 30, YH = y + 46, YF = y + 62, YM = y + 96, YSUM = y + 132, YL = y + 150, YE = y + 188, YP = y + 218;
    bar(g4, 40, YS - 9, W.dd, 's₂', 'played');
    D.wire(g4, 40 + W.dd + 4, YS, CX[3], YS, null, {accent: true});
    D.note(g4, 40, YH + 4, 'hₜ', 'start');
    D.wire(g4, 60, YH, CX[3] + 60, YH);
    D.wire(g4, 140, YP, CX[3] + 40, YP, mk);
    D.note(g4, CX[3] + 46, YP + 4, 'aₜ₊₁');
    D.note(g4, 140, YP + 14, 'picks so far → later heads, and the next token');
    names.forEach(function (name, j) {
      var cx = CX[j];
      D.text(g4, cx, y + 14, (j + 1) + '. ' + name, {'text-anchor': 'middle', class: 'diagram-note'});
      g4.appendChild(D.el('circle', {cx: cx, cy: YS, r: 2.2, fill: 'var(--oxblood)'}));
      if (j === 0) {
        D.wire(g4, cx, YS, cx, YM - 2, mk);
      } else {
        D.wire(g4, cx, YS, cx, YF - 2, mk);
        op(g4, cx - 38, YF, 76, 'FiLM');
        var tx = cx - 62;
        g4.appendChild(D.el('circle', {cx: tx, cy: YP, r: 2.2, fill: FAINT}));
        D.path(g4, 'M ' + tx + ' ' + YP + ' L ' + tx + ' ' + (YF + 10) + ' L ' + (cx - 40) + ' ' + (YF + 10), mk);
        D.wire(g4, cx, YF + 20, cx, YM - 2, mk);
      }
      op(g4, cx - 38, YM, 76, 'MLP head');
      D.wire(g4, cx, YM + 20, cx, YSUM - 8, mk);
      g4.appendChild(D.el('circle', {cx: cx, cy: YSUM, r: 7, class: 'model-token'}));
      D.text(g4, cx, YSUM + 4, '+', {'text-anchor': 'middle', class: 'diagram-label'});
      var sx = cx + 68;
      g4.appendChild(D.el('circle', {cx: sx, cy: YH, r: 2.2, fill: FAINT}));
      D.wire(g4, sx, YH, sx, YM + 4);
      op(g4, sx - 22, YM + 4, 44, 'linear');
      D.path(g4, 'M ' + sx + ' ' + (YM + 24) + ' L ' + sx + ' ' + YSUM + ' L ' + (cx + 8) + ' ' + YSUM, mk);
      D.wire(g4, cx, YSUM + 7, cx, YL - 2, mk);
      var lw = Math.max(6, vocab[j] * PX);
      bar(g4, cx - lw / 2, YL, lw, lw >= 40 ? 'logits' : '');
      D.note(g4, cx + 8, YL + 30, '|V' + ['c', 'm', 't', 'b'][j] + '| · sample');
      D.wire(g4, cx, YL + 34, cx, YE - 2, mk);
      bar(g4, cx - W.ea / 2, YE, W.ea, '', 'given');
      D.note(g4, cx + 10, YE + 13, 'e(a' + ['ᶜ', 'ᵐ', 'ᵗ', 'ᵇ'][j] + ')');
      D.wire(g4, cx, YE + BH, cx, YP);
      g4.appendChild(D.el('circle', {cx: cx, cy: YP, r: 2.2, fill: FAINT}));
    });
    hits.push(['frame', 10, y - 14, 880, 250]);
    rule(y + 244);

    /* ---- 05 play loop: which written frames the game receives ---- */
    var g5 = D.part(svg, 'loop');
    y = y + 266;
    bandTitle(g5, y, '05', 'play loop', 'every input is held two frames before the game applies it');
    var cellW = 72, gap = 8, x0 = 150;
    function planRow(ry, label, offset) {
      D.note(g5, 40, ry + 19, label);
      for (var f = 0; f < 6; f++) {
        var cls = f < 2 ? 'queued' : f < 4 ? 'played' : '';
        D.token(g5, x0 + (offset + f) * (cellW + gap), ry, cellW, 30, offset + f === 0 ? 'aₜ' : 'aₜ₊' + sub(offset + f), cls);
      }
    }
    D.note(g5, x0 + cellW + gap / 2, y + 20, 'already in the game', 'middle');
    D.note(g5, x0 + 3 * (cellW + gap) - gap / 2, y + 20, 'played', 'middle', 'play');
    D.note(g5, x0 + 5 * (cellW + gap) - gap / 2, y + 20, 'replaced by the next plan', 'middle');
    planRow(y + 28, 'plan at t', 0);
    planRow(y + 72, 'plan at t+2', 2);
    D.note(g5, 40, y + 122, 'the frame the game applied comes back as aₜ₋₁ in the next observation token');
    hits.push(['loop', 10, y - 14, 880, 146]);

    hits.forEach(function (h) {
      var key = h[0];
      var rect = D.el('rect', {x: h[1], y: h[2], width: h[3], height: h[4], class: 'hit', tabindex: 0, role: 'button', 'aria-label': SPEC[key].t});
      rect.addEventListener('click', function () { select(selected === key ? null : key); });
      rect.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(selected === key ? null : key); } });
      svg.appendChild(rect);
    });
  }

  var SPEC = {
    token: {n: '01', t: 'Observation token', p: [
      'Every frame becomes one token: the game state for both players and their followers, the characters and stage, and the controller frame that preceded it as the same four class embeddings the decoder predicts. They are concatenated and projected once.',
      'There are no hand-built geometric features; earlier experiments that added them never produced a clean win.']},
    trunk: {n: '02', t: 'Context trunk', p: [
      'A causal Transformer over the token sequence. Each position produces a state for its frame from that frame and the ones before it. All positions are computed in one pass, so one training window supervises the decoder at every position.',
      'In play only the newest state, hₜ, is used.']},
    decoder: {n: '03', t: 'Action decoder', p: [
      'One token per future frame: the trunk state, the previous written frame, and an embedding of the offset, projected to a narrower width. Causal attention runs over these tokens only; the context enters through hₜ, which is copied into every one of them.',
      'In training every position of the window goes through it in parallel, teacher-forced; in play it advances one frame at a time on a cache, and only the first few offsets are written. It is trained to write further ahead than the game uses.']},
    frame: {n: '04', t: 'Controller frame', p: [
      'Inside a frame the four controller groups are chosen in a fixed order. Each later head sees the picks made so far through FiLM on the frame state; every head is a small MLP whose logits also receive a linear skip from hₜ. The picked class is embedded and carried forward.',
      'The button vocabulary is by far the largest.']},
    loop: {n: '05', t: 'Play loop', p: [
      'Online play holds every input for two frames before the game applies it. A plan is handed the two frames already committed, writes the frames after them, and two of those are played before the next plan.',
      'The frame the game applied returns as the action half of the next token.']}
  };
  var ORDER = ['token', 'trunk', 'decoder', 'frame', 'loop'];
  var selected = null;

  function select(key) {
    selected = key;
    D.focus(svg, key);
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
  render();
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && selected) select(null); });
}());
