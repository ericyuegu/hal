/* Fig: O54 — four model sizes, each deployed at the delay its own inference forces, across four data budgets. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('capacity-panels-diagram');
  var readout = document.getElementById('capacity-panels-readout');
  var S = D && D.study('capacity_and_latency');
  if (!D || !svg || !S) return;
  var MODELS = [
    {w: 256, label: '5M', delay: 1}, {w: 512, label: '39M', delay: 2}, {w: 768, label: '125M', delay: 3, accent: true}, {w: 1024, label: '289M', delay: 6}
  ];
  var TIERS = [28, 29, 30, 31];
  var cell = {};
  S.arms.forEach(function (a) { cell[a.name.split('/')[0].slice(1) + ':' + a.data_exponent] = a; });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.capacityPanels = function () {
    return S.arms.map(function (a) { return {model: a.name.split(' at ')[0], run: a.run, data_exponent: a.data_exponent, bc_parameters: a.bc_parameters, delay_frames: a.delay_frames, latency_p99_ms: +a.latency_p99_ms.toFixed(1), stock: a.stock, stock_ci95_lo: a.stock_ci95[0], stock_ci95_hi: a.stock_ci95[1]}; });
  };

  var W = 880, H = 330, L = 64, R = 16, top = 54, bottom = 268, pw = (W - L - R - 3 * 18) / 4;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var lo = -1.3, hi = 0.2;
  var y = D.scale([bottom, top], lo, hi);
  var superscript = function (n) { return String(n).replace(/\d/g, function (c) { return '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]; }); };
  D.text(svg, L, 22, 'net stocks / min · dot and 95% interval · x = deployed delay in frames', {class: 'diagram-note'});
  [-1, -0.5, 0].forEach(function (v) {
    D.text(svg, L - 8, y(v) + 4, D.signed(v, 1), {class: 'diagram-note', 'text-anchor': 'end'});
  });
  TIERS.forEach(function (t, k) {
    var px = L + k * (pw + 18);
    var x = D.scale([px + 14, px + pw - 14], 0, 7);
    [-1, -0.5, 0].forEach(function (v) {
      svg.appendChild(D.el('line', {x1: px, y1: y(v), x2: px + pw, y2: y(v), class: v === 0 ? 'effect-zero' : 'effect-grid'}));
    });
    D.text(svg, px + pw / 2, top - 14, 'D = 2' + superscript(t) + (k === 0 ? ' positions' : ''), {class: 'diagram-title', 'text-anchor': 'middle'});
    [1, 2, 3, 6].forEach(function (d) { D.text(svg, x(d), bottom + 16, String(d), {class: 'diagram-note', 'text-anchor': 'middle'}); });
    MODELS.forEach(function (m) {
      var a = cell[m.w + ':' + t], cx = x(m.delay), cls = m.accent ? ' accent' : '';
      svg.appendChild(D.el('line', {x1: cx, y1: y(a.stock_ci95[0]), x2: cx, y2: y(a.stock_ci95[1]), class: 'effect-ci' + cls}));
      var pt = D.el('circle', {cx: cx, cy: y(a.stock), r: 4, class: 'effect-pt' + cls});
      svg.appendChild(pt);
      if (k === TIERS.length - 1) D.text(svg, cx + (m.delay === 6 ? -8 : 8), y(a.stock) - 8, m.label, {class: 'diagram-label' + (m.accent ? ' accent-text' : ''), 'text-anchor': m.delay === 6 ? 'end' : 'start'});
      var hit = D.el('rect', {x: cx - 12, y: top, width: 24, height: bottom - top, class: 'hit', fill: 'transparent'});
      svg.appendChild(hit);
      hit.addEventListener('mouseenter', function () {
        readout.innerHTML = '<span><b>' + m.label + '</b> · W' + m.w + ' · delay ' + m.delay + ' · p99 inference ' + a.latency_p99_ms.toFixed(0) + ' ms</span><span>D = 2' + superscript(t) + '</span><span>net stocks / min <b>' + D.signed(a.stock, 2) + '</b> [' + D.signed(a.stock_ci95[0], 2) + ', ' + D.signed(a.stock_ci95[1], 2) + ']</span><span>' + a.run + '</span>';
      });
    });
  });
  // legend for the accent
  svg.appendChild(D.el('circle', {cx: L + 6, cy: H - 14, r: 3.6, class: 'effect-pt accent'}));
  D.text(svg, L + 16, H - 10, '125M at 3 frames · strongest at the largest budget', {class: 'diagram-note'});
  svg.appendChild(D.el('circle', {cx: L + 430, cy: H - 14, r: 3.6, class: 'effect-pt'}));
  D.text(svg, L + 440, H - 10, '5M · 1 frame, 39M · 2 frames, 289M · 6 frames', {class: 'diagram-note'});
  svg.addEventListener('mouseleave', function () { readout.innerHTML = ''; });
}());
