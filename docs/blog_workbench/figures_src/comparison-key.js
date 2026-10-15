/* Fig: the legend for every comparison — baseline → treatment on the left, their difference with its 95% interval on the right, one strip. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var row = document.getElementById('comparison-key-row');
  var gloss = document.getElementById('comparison-key-glossary');
  if (!D || !CI || !row) return;
  var d = CI.codec_legacy_cartesian.stock;
  var ARMS = {base: {label: 'legacy codec', run: 'acbbeb76', score: -0.133}, treat: {label: 'Cartesian codec', run: 'cqbbbg77', score: 0.159}};
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.comparisonKey = function () {
    return [{baseline: ARMS.base.label, baseline_run: ARMS.base.run, baseline_stocks_per_min: ARMS.base.score, treatment: ARMS.treat.label, treatment_run: ARMS.treat.run, treatment_stocks_per_min: ARMS.treat.score, delta: d.delta, ci95_lo: d.lo, ci95_hi: d.hi, method: CI.codec_legacy_cartesian.method, boots: CI.codec_legacy_cartesian.n_boots}];
  };

  function arm(role, a, cls) {
    var el = document.createElement('div'); el.className = 'key-arm' + (cls ? ' ' + cls : '');
    el.innerHTML = '<span class="key-role">' + role + '</span><b>' + a.label + '</b><span class="key-score">' + D.signed(a.score, 2) + ' net stocks / min</span><span class="key-run">' + a.run + '</span>';
    return el;
  }
  var arrow = document.createElement('div'); arrow.className = 'key-arrow'; arrow.textContent = '→';
  var chart = document.createElement('div'); chart.className = 'key-chart';
  var svg = D.el('svg', {class: 'model-diagram reflow-chart', id: 'comparison-key-diagram', role: 'img', 'aria-label': 'The difference, plus 0.29 net stocks per minute, drawn as a dot with a horizontal 95% interval from plus 0.08 to plus 0.51 against a zero line.'});
  chart.appendChild(svg);
  row.appendChild(arm('baseline', ARMS.base, ''));
  row.appendChild(arrow);
  row.appendChild(arm('treatment', ARMS.treat, 'treat'));
  row.appendChild(chart);

  // the effect: zero line, interval, dot, value — labelled in place, no scale (the numbers are the point, not the axis)
  var W = 420, H = 128, y = 58;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var dom = D.domain([d]), x = D.scale([24, 286], dom.lo, dom.hi);
  dom.ticks.forEach(function (v) {
    svg.appendChild(D.el('line', {x1: x(v), y1: y - 22, x2: x(v), y2: y + 22, class: v === 0 ? 'effect-zero' : 'effect-grid'}));
  });
  D.effect(svg, x, y, d, {accent: true, digits: 2, valueX: 304});
  D.text(svg, 304, y + 18, 'stocks / min', {class: 'diagram-note'});
  D.text(svg, x(d.delta), y - 32, 'Δ = treatment − baseline', {class: 'diagram-label accent-text', 'text-anchor': 'middle'});
  D.text(svg, x(0), y + 56, '0 = no change', {class: 'diagram-note', 'text-anchor': 'middle'});
  svg.appendChild(D.el('path', {d: 'M ' + x(d.lo) + ' ' + (y + 26) + ' v 5 H ' + x(d.hi) + ' v -5', class: 'model-wire'}));
  D.text(svg, x(d.hi), y + 43, '95% interval · 96 paired boots', {class: 'diagram-note', 'text-anchor': 'end'});

  gloss.innerHTML = '<span><b>Read every comparison as baseline → treatment.</b></span><span>The dot is the change; the bar is its 95% interval from evaluation noise, not training seeds.</span><span>Unmatched or unfinished comparisons are labeled where they occur.</span>';
}());
