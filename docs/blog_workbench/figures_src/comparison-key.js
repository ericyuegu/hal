/* Fig: the legend for every comparison — one annotated row, old score → new score, Δ with its 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('comparison-key-diagram');
  if (!D || !CI || !svg) return;
  var d = CI.codec_legacy_cartesian.stock;
  var ARMS = {base: {label: 'legacy codec', run: 'acbbeb76', score: -0.133}, treat: {label: 'Cartesian codec', run: 'cqbbbg77', score: 0.159}};
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.comparisonKey = function () {
    return [{baseline: ARMS.base.label, baseline_run: ARMS.base.run, baseline_stocks_per_min: ARMS.base.score, treatment: ARMS.treat.label, treatment_run: ARMS.treat.run, treatment_stocks_per_min: ARMS.treat.score, delta: d.delta, ci95_lo: d.lo, ci95_hi: d.hi, method: CI.codec_legacy_cartesian.method, boots: CI.codec_legacy_cartesian.n_boots}];
  };

  var W = 880, H = 196, y = 94;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var mk = D.marker(svg, 'arrow', 'var(--ink-faint)');

  // — the two arms
  function arm(x, a, cls) {
    svg.appendChild(D.el('rect', {x: x, y: y - 22, width: 150, height: 44, class: 'model-token' + (cls ? ' ' + cls : '')}));
    D.text(svg, x + 75, y - 4, a.label, {'text-anchor': 'middle', class: 'diagram-label' + (cls === 'given' ? ' given-label' : '')});
    D.text(svg, x + 75, y + 13, D.signed(a.score, 2) + ' stocks / min', {'text-anchor': 'middle', class: 'diagram-note'});
  }
  arm(24, ARMS.base, 'given');
  D.wire(svg, 178, y, 214, y, mk);
  arm(218, ARMS.treat, '');

  // — the effect column
  var dom = D.domain([d]), x = D.scale([480, 700], dom.lo, dom.hi);
  D.effectAxis(svg, x, dom, y - 30, y + 30, null);
  D.effect(svg, x, y, d, {accent: true, digits: 2, valueX: 716});
  D.text(svg, 716, y + 18, 'stocks / min', {class: 'diagram-note'});

  // — annotations: one label per part, a thin leader each; arms below, axis parts above
  function callout(x1, y1, x2, y2, label, anchor) {
    svg.appendChild(D.el('line', {x1: x1, y1: y1, x2: x2, y2: y2, class: 'model-wire ghost'}));
    D.text(svg, x2, y2 + (y2 > y1 ? 13 : -6), label, {class: 'diagram-note', 'text-anchor': anchor || 'middle'});
  }
  callout(40, y + 22, 40, 172, 'baseline · the model before the change', 'start');
  callout(234, y + 22, 234, 150, 'treatment · the same model, one thing changed', 'start');
  callout(x(0), y + 30, x(0), 172, '0 · no change', 'start');
  callout(x(d.delta), y - 6, x(d.delta), 52, 'Δ = treatment − baseline', 'middle');
  callout(x(d.hi), y - 3, x(d.hi) + 16, 30, '95% interval · 96 paired games', 'end');
}());
