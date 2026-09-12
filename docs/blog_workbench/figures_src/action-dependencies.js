/* Fig: what a predicted action is given — three decoder generations in the "given → produced" grammar. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('deps-diagram');
  if (!D || !svg) return;
  var FAINT = 'var(--ink-faint)';
  var mk = D.marker(svg, 'arrow', FAINT);
  svg.setAttribute('viewBox', '0 0 880 420');

  function panelTitle(y, title, note, accent) {
    D.text(svg, 16, y, title, {class: 'diagram-title' + (accent ? ' accent' : '')});
    D.note(svg, 16, y + 16, note);
  }
  function hLabel(x, y) { D.text(svg, x, y - 6, 'hₜ', {class: 'diagram-label'}); }
  var ROW = {w: 44, h: 28, gap: 6, arrow: 56, mk: mk};
  var P = 560; // the produced column, shared by all three panels
  var X = P - ROW.arrow - 0; // where an empty prefix's arrow starts

  // 1 · next-frame prediction
  panelTitle(24, 'Next-frame prediction', 'one frame, then observe again');
  var r1 = D.row(svg, P - ROW.arrow + ROW.gap - 6, 40, [], ['aₜ₊₁'], ROW);
  hLabel(r1.arrowStart + 18, 54);

  D.wire(svg, 16, 96, 864, 96, null, {ghost: true});

  // 2 · independent multi-token prediction
  panelTitle(122, 'Independent multi-token prediction', 'several frames at once, each from the trunk state alone');
  var r2 = D.row(svg, P - ROW.arrow + ROW.gap - 6, 138, [], ['aₜ₊₁', 'aₜ₊₂', 'aₜ₊₃', 'aₜ₊₄'], ROW);
  hLabel(r2.arrowStart + 18, 152);
  D.note(svg, P, 182, 'written together — no frame can see another');

  D.wire(svg, 16, 200, 864, 200, null, {ghost: true});

  // 3 · autoregressive multi-token prediction (the current decoder)
  panelTitle(220, 'Autoregressive multi-token prediction', 'each frame is handed the frames written before it', true);
  var prefixes = [[], ['aₜ₊₁'], ['aₜ₊₁', 'aₜ₊₂'], ['aₜ₊₁', 'aₜ₊₂', 'aₜ₊₃']];
  var outs = ['aₜ₊₁', 'aₜ₊₂', 'aₜ₊₃', 'aₜ₊₄'];
  prefixes.forEach(function (p, i) {
    var y = 246 + i * 40;
    // right-align the prefixes so the produced column lines up
    var x = P - ROW.arrow + ROW.gap - 6 - p.length * (ROW.w + ROW.gap);
    var r = D.row(svg, x, y, p, [outs[i]], ROW);
    if (i === 0) hLabel(r.arrowStart + 18, y + 14);
  });
  D.note(svg, 16, 410, 'hₜ rides on every arrow · inside a frame the four controller groups are written the same way, each knowing the ones before it');
}());
