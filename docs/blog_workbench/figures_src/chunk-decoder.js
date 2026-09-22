/* Fig: the decoder Transformer over one action chunk — one token per future frame, causal
   self-attention over the earlier tokens, cross-attention to the trunk states, and the
   written frame feeding the next token (experiments/059 CausalTemporalDecoder). */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('chunk-decoder-diagram');
  if (!D || !svg) return;

  var FAINT = 'var(--ink-faint)', ACCENT = 'var(--oxblood)';
  svg.setAttribute('viewBox', '0 0 900 560');
  var mk = D.marker(svg, 'arrow', FAINT), mka = D.marker(svg, 'arrow-accent', ACCENT);
  function dot(target, x, y, accent) { target.appendChild(D.el('circle', {cx: x, cy: y, r: 2.4, fill: accent ? ACCENT : FAINT})); }
  function op(target, cx, y, w, h, label, cls) {
    target.appendChild(D.el('rect', {x: cx - w / 2, y: y, width: w, height: h, class: 'model-token' + (cls ? ' ' + cls : '')}));
    D.text(target, cx, y + h / 2 + 4, label, {'text-anchor': 'middle', class: 'diagram-label'});
  }
  function rail(y, lines) {
    lines.forEach(function (line, i) { D.note(svg, 168, y + i * 13, line, 'end'); });
  }

  // — trunk states over the context, ending in hₜ
  var trunk = D.part(svg, 'trunk');
  D.note(trunk, 16, 14, 'trunk states over the context, one per frame');
  var STATES = [{x: 16, label: 'hₜ₋L', cls: 'given'}, {x: 118, label: 'hₜ₋₂', cls: 'given'}, {x: 172, label: 'hₜ₋₁', cls: 'given'}, {x: 226, label: 'hₜ', cls: 'played'}];
  STATES.forEach(function (s) { D.token(trunk, s.x, 28, 48, 28, s.label, s.cls); });
  D.note(trunk, 91, 46, '···', 'middle');

  // — columns: one decoder token per future frame
  var CX = [258, 388, 518, 648, 816], W = 108;
  var STEP = [
    {k: '1', prev: 'e(aₜ₋₁)', prevGiven: true, writes: 'aₜ'},
    {k: '2', prev: 'e(aₜ)', writes: 'aₜ₊₁'},
    {k: '3', prev: 'e(aₜ₊₁)', writes: 'aₜ₊₂'},
    {k: '4', prev: 'e(aₜ₊₂)', writes: 'aₜ₊₃'},
    {k: '28', prev: 'e(aₜ₊₂₆)', writes: 'aₜ₊₂₇'}
  ];
  var YTOK = 96, YLIN = 146, YATT = 200, YXATT = 258, YKV = 244, YBLK = 328, YNORM = 380, YHEAD = 418, YOUT = 470;
  D.note(svg, 874, 14, 'one decoder token per future frame, offset k', 'end');
  D.note(svg, 738, YTOK + 18, '···', 'middle');

  // hₜ bus: the same trunk state enters every token
  var hbus = D.part(svg, 'hbus');
  D.path(hbus, 'M 274 42 L 292 42 L 292 76 L ' + (CX[4] + 15) + ' 76', null, {accent: true});
  D.note(hbus, 300, 72, 'hₜ', 'start', 'play');

  // K, V: every trunk state, projected once, shared by every step's cross-attention
  var kv = D.part(svg, 'kv');
  D.wire(kv, 16, 62, 274, 62);
  D.note(kv, 16, 76, 'every trunk state → K, V');
  D.wire(kv, 190, 62, 190, 170);
  op(kv, 190, 170, 52, 20, 'linear');
  D.wire(kv, 190, 190, 190, YKV);
  D.path(kv, 'M 190 ' + YKV + ' L ' + (CX[4] + 40) + ' ' + YKV, null, {ghost: true});

  rail(YTOK + 12, ['previous frame,', 'hₜ, offset k']);
  rail(YLIN + 14, ['token']);
  rail(YATT + 12, ['causal attention', 'over earlier tokens']);
  rail(YXATT + 12, ['cross-attention', 'to trunk states']);
  rail(YBLK + 19, ['more causal blocks']);
  rail(YNORM + 14, ['frame state sₖ']);
  rail(YHEAD + 19, ['four heads, in order']);
  rail(YOUT + 18, ['written frame']);

  STEP.forEach(function (s, i) {
    var cx = CX[i], col = D.part(svg, 'step' + i);
    // the token: [ e(aₖ₋₁) | hₜ | k ] → linear → norm
    col.appendChild(D.el('rect', {x: cx - 54, y: YTOK, width: W, height: 28, class: 'model-token'}));
    D.wire(col, cx + 2, YTOK, cx + 2, YTOK + 28, null, {ghost: true});
    D.wire(col, cx + 32, YTOK, cx + 32, YTOK + 28, null, {ghost: true});
    D.text(col, cx - 26, YTOK + 18, s.prev, {'text-anchor': 'middle', class: 'diagram-label' + (s.prevGiven ? ' given-label' : '')});
    D.text(col, cx + 17, YTOK + 18, 'hₜ', {'text-anchor': 'middle', class: 'diagram-label'});
    D.text(col, cx + 43, YTOK + 18, s.k, {'text-anchor': 'middle', class: 'diagram-label'});
    dot(hbus, cx + 15, 76, true);
    D.wire(hbus, cx + 15, 76, cx + 15, YTOK - 2, mka, {accent: true});
    D.wire(col, cx, YTOK + 28, cx, YLIN - 2, mk);
    op(col, cx, YLIN, 96, 20, 'linear · norm');
    D.wire(col, cx, YLIN + 20, cx, YATT - 2, mk);
    // block 1: causal self-attention + MLP over the chunk tokens so far
    op(col, cx, YATT, W, 30, 'attention · MLP');
    D.wire(col, cx, YATT + 30, cx, YXATT - 2, mk);
    // history cross-attention: queries from the token, keys and values from the trunk states
    op(col, cx, YXATT, W, 30, 'cross-attention');
    dot(kv, cx + 32, YKV);
    D.wire(kv, cx + 32, YKV, cx + 32, YXATT - 2, mk);
    D.wire(col, cx, YXATT + 30, cx, YBLK - 2, mk);
    // the remaining causal blocks
    [8, 4].forEach(function (o) { col.appendChild(D.el('rect', {x: cx - 54 - o, y: YBLK + o, width: W, height: 30, class: 'model-layer'})); });
    op(col, cx, YBLK, W, 30, 'attention · MLP');
    D.wire(col, cx, YBLK + 30, cx, YNORM - 2, mk);
    op(col, cx, YNORM, 60, 20, 'norm');
    D.wire(col, cx, YNORM + 20, cx, YHEAD - 2, mk);
    // the four heads of Fig. 6 write this frame
    col.appendChild(D.el('rect', {x: cx - 54, y: YHEAD, width: W, height: 32, class: 'model-token queued'}));
    D.text(col, cx, YHEAD + 13, 'four heads', {'text-anchor': 'middle', class: 'diagram-label'});
    D.text(col, cx, YHEAD + 26, 'in order', {'text-anchor': 'middle', class: 'diagram-label'});
    D.wire(col, cx, YHEAD + 32, cx, YOUT - 2, mk);
    D.token(col, cx - 54, YOUT, W, 28, s.writes, 'played');
    // the written frame embeds into the next token
    var back = D.part(svg, 'back' + i);
    var gx = cx + 66, ny = YTOK + 14;
    if (i < 3) {
      D.path(back, 'M ' + (cx + 54) + ' ' + (YOUT + 14) + ' L ' + gx + ' ' + (YOUT + 14) + ' L ' + gx + ' ' + ny + ' L ' + (CX[i + 1] - 56) + ' ' + ny, mka, {accent: true});
    } else if (i === 3) {
      D.path(back, 'M ' + (cx + 54) + ' ' + (YOUT + 14) + ' L ' + gx + ' ' + (YOUT + 14) + ' L ' + gx + ' ' + ny + ' L ' + (gx + 12) + ' ' + ny, null, {accent: true});
    } else {
      D.wire(back, cx - 66, ny, cx - 56, ny, mka, {accent: true});
    }
  });
  // step 4 attends to steps 1–3; every step attends only to the steps before it
  var att = D.part(svg, 'attn');
  [0, 1, 2].forEach(function (i) {
    var x1 = CX[i] + 20, x2 = CX[3] - 20, lift = 10 + (3 - i) * 5;
    D.path(att, 'M ' + x1 + ' ' + YATT + ' Q ' + ((x1 + x2) / 2) + ' ' + (YATT - lift) + ' ' + x2 + ' ' + YATT, mk, {ghost: true});
  });

  D.note(svg, 204, 522, 'train: every step at once, teacher-forced with the true prefix');
  D.note(svg, 204, 536, 'play: one step at a time, keys and values cached');
}());
