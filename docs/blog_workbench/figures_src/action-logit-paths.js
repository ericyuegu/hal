/* Fig: one written frame opened up — the token strip, then the four heads in order with FiLM,
   the SiLU MLP head and the trunk skip — and what each of the three parts was worth (O43). */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('logit-anatomy');
  var forest = document.getElementById('logit-forest');
  if (!D || !svg || !forest) return;

  // baseline → treatment on both evaluation suites; the baseline of each row is O43 without that part
  var ROWS = [
    {comp: 'film', label: 'zero-init FiLM → ordinary init', sub: 'vzd242t6 → 1imfy8v3 · 4,096 updates',
      char: {delta: 1.09, lo: 0.75, hi: 1.43}, fox: {delta: 1.20, lo: 0.83, hi: 1.57}},
    {comp: 'skip', label: 'no trunk skip → trunk skip', sub: 'j4skyy83 → 1imfy8v3 · 4,096 updates',
      char: {delta: 0.89, lo: 0.49, hi: 1.29}, fox: {delta: 1.13, lo: 0.67, hi: 1.59}},
    {comp: 'head', label: 'linear head → SiLU MLP head', sub: 'apj3rw8b → 1imfy8v3 · 8,192 updates',
      char: {delta: 2.03, lo: 1.75, hi: 2.31}, fox: {delta: 2.41, lo: 2.09, hi: 2.73}}
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.logitPathRows = function () {
    return ROWS.map(function (r) {
      return {comparison: r.label, runs: r.sub.split(' · ')[0], updates: r.sub.split(' · ')[1],
        character_suite_delta: r.char.delta, character_suite_ci95_lo: r.char.lo, character_suite_ci95_hi: r.char.hi,
        fox_suite_delta: r.fox.delta, fox_suite_ci95_lo: r.fox.lo, fox_suite_ci95_hi: r.fox.hi};
    });
  };

  var FAINT = 'var(--ink-faint)', ACCENT = 'var(--oxblood)';

  /* ------------------------------------------------------------- anatomy */
  svg.setAttribute('viewBox', '0 0 900 560');
  var mk = D.marker(svg, 'arrow', FAINT), mka = D.marker(svg, 'arrow-accent', ACCENT);
  function dot(target, x, y) { target.appendChild(D.el('circle', {cx: x, cy: y, r: 2.4, fill: FAINT})); }
  function op(target, cx, y, w, label, cls) {
    target.appendChild(D.el('rect', {x: cx - w / 2, y: y, width: w, height: 20, class: 'model-token' + (cls ? ' ' + cls : '')}));
    D.text(target, cx, y + 14, label, {'text-anchor': 'middle', class: 'diagram-label'});
  }

  // — the token strip: hₜ and one token per future offset
  var strip = D.part(svg, 'strip');
  D.note(strip, 120, 18, 'one token per future frame · two causal attention blocks over them');
  D.token(strip, 24, 44, 56, 30, 'hₜ', 'given');
  D.note(strip, 52, 38, 'trunk state', 'middle');
  D.wire(strip, 80, 59, 118, 59, mk);
  var TOK = [
    {x: 120, label: '(hₜ, aₜ₋₁, 1)', writes: 'writes aₜ'},
    {x: 242, label: '(hₜ, aₜ, 2)', writes: 'writes aₜ₊₁', open: true},
    {x: 364, label: '(hₜ, aₜ₊₁, 3)', writes: 'writes aₜ₊₂'},
    {x: 486, label: '(hₜ, aₜ₊₂, 4)', writes: 'writes aₜ₊₃'},
    {x: 650, label: '(hₜ, aₜ₊₁₉, 20)', writes: 'writes aₜ₊₂₀'}
  ];
  TOK.forEach(function (t, i) {
    D.token(strip, t.x, 44, 112, 30, t.label, t.open ? 'played' : '');
    D.note(strip, t.x + 56, 92, t.writes, 'middle', t.open ? 'play' : '');
    if (i > 0 && i < 4) D.wire(strip, TOK[i - 1].x + 112, 59, t.x - 2, 59, mk);
  });
  D.note(strip, 624, 63, '···', 'middle');
  // causal attention: every token attends to the tokens before it
  [0, 1, 2].forEach(function (i) {
    var x1 = TOK[i].x + 56, x2 = TOK[3].x + 56, lift = 12 + (3 - i) * 6;
    D.path(strip, 'M ' + x1 + ' 44 Q ' + ((x1 + x2) / 2) + ' ' + (44 - lift) + ' ' + x2 + ' 44', mk, {ghost: true});
  });
  D.note(strip, 706, 112, 'attends only to earlier tokens', 'middle');

  // — the opened frame: frame state bus and trunk bus
  var opened = D.part(svg, 'frame');
  D.wire(opened, 298, 74, 298, 150, mka, {accent: true});
  D.note(opened, 306, 112, 'attention → norm → frame state s₂', 'start', 'play');
  var CX = [160, 340, 520, 700];
  var YS = 150, YH = 172, YF = 196, YM = 236, YSUM = 380, YL = 402, YSAMPLE = 436, YE = 470, YP = 516;
  D.wire(opened, 100, YS, 790, YS, null, {accent: true});
  D.note(opened, 96, YS + 4, 's₂', 'end', 'play');
  var skipbus = D.part(svg, 'skip');
  D.wire(skipbus, 52, 74, 52, YH);
  D.wire(skipbus, 52, YH, 790, YH);
  D.note(skipbus, 60, YH + 14, 'hₜ skip bus');
  var picksbus = D.part(svg, 'picks');
  D.wire(picksbus, 140, YP, 790, YP, mk);
  D.note(picksbus, 800, YP + 4, 'aₜ₊₁ → token 3');
  D.note(picksbus, 140, YP + 16, 'picks so far, concatenated');

  var GROUPS = ['C-stick', 'main stick', 'triggers', 'buttons'];
  GROUPS.forEach(function (name, j) {
    var cx = CX[j];
    D.text(svg, cx, 138, (j + 1) + '. ' + name, {'text-anchor': 'middle', class: 'diagram-title'});
    // frame state into the column
    var col = D.part(svg, 'col' + j);
    dot(col, cx, YS);
    if (j === 0) {
      D.wire(col, cx, YS, cx, YM + 8, mk);
      D.note(col, cx - 8, YF + 14, 'first: no FiLM', 'end');
    } else {
      D.wire(col, cx, YS, cx, YF, mk);
      var film = D.part(svg, 'film');
      op(film, cx, YF, 84, 'FiLM  γ, β');
      // γ, β come from the picks made earlier in this frame
      var tx = cx - 72;
      dot(film, tx, YP);
      D.path(film, 'M ' + tx + ' ' + YP + ' L ' + tx + ' ' + (YF + 10) + ' L ' + (cx - 44) + ' ' + (YF + 10), mk);
      D.wire(col, cx, YF + 20, cx, YM + 8, mk);
    }
    // the head: norm → linear → SiLU → linear
    var head = D.part(svg, 'head');
    head.appendChild(D.el('rect', {x: cx - 44, y: YM, width: 88, height: 130, class: 'model-token queued'}));
    if (j === 0) {
      ['norm', 'linear', 'SiLU', 'linear'].forEach(function (label, i) { op(head, cx, YM + 8 + i * 26, 64, label); });
    } else {
      op(head, cx, YM + 50, 78, 'SiLU MLP');
      D.note(head, cx, YM + 85, 'same structure', 'middle');
    }
    D.note(head, cx, YM + 121, 'SiLU MLP head', 'middle');
    D.wire(col, cx, YM + 130, cx, YSUM - 10, mk);
    // Σ with the trunk skip from the right
    col.appendChild(D.el('circle', {cx: cx, cy: YSUM, r: 9, class: 'model-token'}));
    D.text(col, cx, YSUM + 4, '+', {'text-anchor': 'middle', class: 'diagram-label'});
    var sx = cx + 72;
    dot(skipbus, sx, YH);
    D.wire(skipbus, sx, YH, sx, 300);
    op(skipbus, sx, 300, 52, 'linear');
    D.path(skipbus, 'M ' + sx + ' 320 L ' + sx + ' ' + YSUM + ' L ' + (cx + 10) + ' ' + YSUM, mk);
    // logits → sample → embed → picks bus
    D.wire(col, cx, YSUM + 9, cx, YL, mk);
    op(col, cx, YL, 84, 'logits · mask');
    D.wire(col, cx, YL + 20, cx, YSAMPLE, mk);
    op(col, cx, YSAMPLE, 84, 'sample');
    D.wire(col, cx, YSAMPLE + 20, cx, YE, mk);
    op(col, cx, YE, 84, 'embed');
    D.wire(col, cx, YE + 20, cx, YP);
    dot(picksbus, cx, YP);
  });
  D.note(svg, CX[3] + 104, 314, 'trunk skip');

  /* ------------------------------------------------------------- forest */
  var W = 900, top = 68, rowH = 54, L = 16, COLX = [430, 690], VALX = 704;
  forest.setAttribute('viewBox', '0 0 ' + W + ' ' + (top + ROWS.length * rowH + 40));
  var dom = D.domain(ROWS.map(function (r) { return r.char; }).concat(ROWS.map(function (r) { return r.fox; })));
  var x = D.scale(COLX, dom.lo, dom.hi);
  D.effectAxis(forest, x, dom, top - 14, top + ROWS.length * rowH - 8, 'Δ net stocks / min');
  var lg = top - 40;
  D.note(forest, L, lg, 'baseline → treatment · 95% interval · each baseline is O43 without that part');
  forest.appendChild(D.el('circle', {cx: 690, cy: lg - 4, r: 3.2, class: 'effect-pt'}));
  D.note(forest, 698, lg, 'character suite');
  forest.appendChild(D.el('circle', {cx: 820, cy: lg - 4, r: 3.2, class: 'effect-pt accent'}));
  D.note(forest, 828, lg, 'Fox suite');
  ROWS.forEach(function (r, i) {
    var y = top + i * rowH + rowH / 2 - 6;
    var g = D.el('g', {class: 'forest-row'});
    g.appendChild(D.el('rect', {x: 0, y: top + i * rowH - 4, width: W, height: rowH, class: 'hit'}));
    D.text(g, L, y, r.label, {class: 'diagram-title'});
    D.text(g, L, y + 14, r.sub.split(' · ')[1], {class: 'diagram-note'});
    D.effect(g, x, y - 7, r.char, {digits: 2, valueX: VALX});
    D.effect(g, x, y + 9, r.fox, {digits: 2, valueX: VALX, accent: true});
    g.setAttribute('tabindex', '0');
    g.setAttribute('role', 'button');
    g.setAttribute('aria-label', r.label);
    g.addEventListener('click', function () { D.focus(svg, r.comp); });
    g.addEventListener('focus', function () { D.focus(svg, r.comp); });
    g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); D.focus(svg, r.comp); } });
    g.addEventListener('mouseenter', function () { D.focus(svg, r.comp); });
    g.addEventListener('mouseleave', function () { D.focus(svg, null); });
    forest.appendChild(g);
  });
  D.note(forest, L, top + ROWS.length * rowH + 30, 'select a row to highlight the part it added');
}());
