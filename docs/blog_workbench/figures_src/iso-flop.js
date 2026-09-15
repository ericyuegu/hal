/* Fig: the O39 grid, L(N, D) = base + a/N^α + b/D^β fitted through it, and the iso-compute minima it implies. */
(function () {
  'use strict';
  var D = window.HALDIAG, DATA = window.HALFIG_DATA;
  var svg = document.getElementById('iso-flop-diagram');
  var readout = document.getElementById('iso-flop-readout');
  if (!D || !DATA || !svg) return;
  var I = DATA.iso_flop, F = I.fit, P = I.points;
  function L(n, d) { return F.base + F.aScale / Math.pow(n, F.aPower) + F.dScale / Math.pow(d, F.dPower); }
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.isoFlop = function () {
    return P.map(function (p) { return {run: p.run, model: p.model, layers: p.layers, d_model: p.d_model, total_parameters: p.total_parameters, n_eff: p.n_eff, d_positions_exp: p.d_exp, training_flops: p.training_flops, validation_nll: +p.nll.toFixed(4), fitted_nll: +p.fit_nll.toFixed(4), residual: +p.residual.toFixed(4)}; });
  };
  var superscript = function (n) { return String(n).replace(/\d/g, function (c) { return '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]; }); };
  var fmtN = function (n) { return n >= 1e9 ? (n / 1e9).toFixed(1) + 'B' : Math.round(n / 1e6) + 'M'; };

  var W = 880, H = 412, L0 = 64, top = 48, bottom = 328;
  var main = [L0, 600], side = [724, 864];
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var nLo = 1e7, nHi = 3e9;
  var x = function (n) { return main[0] + (Math.log10(n) - 7) / (Math.log10(nHi) - 7) * (main[1] - main[0]); };
  var y = D.scale([bottom, top], 1.7, 2.1);
  var nMin = Math.min.apply(null, P.map(function (p) { return p.n_eff; })), nMax = Math.max.apply(null, P.map(function (p) { return p.n_eff; }));

  // measured size range, grid, axes
  svg.appendChild(D.el('rect', {x: x(nMin), y: top, width: x(nMax) - x(nMin), height: bottom - top, class: 'chart-band'}));
  D.text(svg, (x(nMin) + x(nMax)) / 2, top - 6, 'measured sizes · curves are extrapolated outside', {class: 'diagram-note', 'text-anchor': 'middle'});
  [1.7, 1.8, 1.9, 2.0, 2.1].forEach(function (v) {
    svg.appendChild(D.el('line', {x1: main[0], y1: y(v), x2: main[1], y2: y(v), class: 'effect-grid'}));
    D.text(svg, main[0] - 8, y(v) + 4, v.toFixed(1), {class: 'diagram-note', 'text-anchor': 'end'});
  });
  [1e7, 1e8, 1e9].forEach(function (n) { D.text(svg, x(n), bottom + 16, fmtN(n), {class: 'diagram-note', 'text-anchor': 'middle'}); });
  D.text(svg, main[1], bottom + 32, 'effective parameters N = trunk + 36 · decoder · log scale', {class: 'diagram-note', 'text-anchor': 'end'});
  D.text(svg, main[0], 22, 'validation NLL · lower is better', {class: 'diagram-note'});

  // one fitted curve per data budget, light → dark with D; points on them
  var tiers = [26, 27, 28, 29, 30];
  var shade = function (i) { return 'color-mix(in srgb, var(--ink) ' + (35 + i * 16) + '%, var(--paper))'; };
  tiers.forEach(function (e, i) {
    var d = Math.pow(2, e), inside = [], before = [], after = [];
    for (var lg = 7; lg <= Math.log10(nHi) + 1e-9; lg += 0.02) {
      var n = Math.pow(10, lg), pt = x(n) + ',' + y(L(n, d));
      (n < nMin ? before : n > nMax ? after : inside).push(pt);
    }
    before.push(inside[0]); after.unshift(inside[inside.length - 1]);
    svg.appendChild(D.el('polyline', {points: before.join(' '), fill: 'none', stroke: shade(i), 'stroke-width': 1.3, 'stroke-dasharray': '3 3'}));
    svg.appendChild(D.el('polyline', {points: after.join(' '), fill: 'none', stroke: shade(i), 'stroke-width': 1.3, 'stroke-dasharray': '3 3'}));
    svg.appendChild(D.el('polyline', {points: inside.join(' '), fill: 'none', stroke: shade(i), 'stroke-width': 1.6}));
    D.text(svg, main[1] + 6, y(L(nHi, d)) + 4, 'D = 2' + superscript(e), {class: 'diagram-note'});
  });
  P.forEach(function (p) {
    var i = tiers.indexOf(p.d_exp);
    var c = D.el('circle', {cx: x(p.n_eff), cy: y(p.nll), r: 3.6, fill: shade(i), stroke: 'var(--paper)', 'stroke-width': 1.2});
    svg.appendChild(c);
    c.addEventListener('mouseenter', function () {
      readout.innerHTML = '<span><b>' + p.model + '</b> · ' + p.layers + ' layers · d ' + p.d_model + ' · N ' + fmtN(p.n_eff) + '</span><span>D = 2' + superscript(p.d_exp) + ' · ' + p.training_flops.toExponential(1).replace('e+', 'e') + ' FLOPs</span><span>NLL <b>' + p.nll.toFixed(3) + '</b> · fit ' + p.fit_nll.toFixed(3) + ' · residual ' + D.signed(p.residual, 3) + '</span><span>' + p.run + '</span>';
    });
  });
  // fixed-compute lines, C = 6 · N · D, with their minimum
  [1e17, 1e18].forEach(function (C) {
    var pts = [], best = null;
    for (var lg = 7; lg <= Math.log10(nHi) + 1e-9; lg += 0.02) {
      var n = Math.pow(10, lg), v = L(n, C / (6 * n));
      if (v <= 2.1) pts.push(x(n) + ',' + y(v));
      if (!best || v < best.v) best = {n: n, v: v};
    }
    svg.appendChild(D.el('polyline', {points: pts.join(' '), fill: 'none', stroke: 'var(--oxblood)', 'stroke-width': 1.2, 'stroke-dasharray': '1.5 3'}));
    svg.appendChild(D.el('circle', {cx: x(best.n), cy: y(best.v), r: 4, class: 'effect-pt accent'}));
    var lx = L0 + (C === 1e17 ? 0 : 250);
    svg.appendChild(D.el('circle', {cx: lx + 4, cy: H - 28, r: 3.6, class: 'effect-pt accent'}));
    D.text(svg, lx + 14, H - 24, '10' + superscript(Math.round(Math.log10(C))) + ' FLOPs · lowest at N ≈ ' + fmtN(best.n), {class: 'diagram-note accent-text'});
  });

  // residual panel: fitted against measured
  var lo = 1.76, hi = 2.06;
  var sx = D.scale(side, lo, hi), sy = D.scale([bottom, bottom - (side[1] - side[0])], lo, hi);
  svg.appendChild(D.el('line', {x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi), class: 'effect-grid'}));
  [1.8, 1.9, 2.0].forEach(function (v) {
    D.text(svg, sx(v), bottom + 16, v.toFixed(1), {class: 'diagram-note', 'text-anchor': 'middle'});
    D.text(svg, side[0] - 6, sy(v) + 4, v.toFixed(1), {class: 'diagram-note', 'text-anchor': 'end'});
  });
  D.text(svg, side[1], 22, 'fit against measured', {class: 'diagram-note', 'text-anchor': 'end'});
  D.text(svg, side[1], bottom + 32, 'measured', {class: 'diagram-note', 'text-anchor': 'end'});
  D.text(svg, side[0], sy(hi) - 6, 'fitted', {class: 'diagram-note'});
  var worst = Math.max.apply(null, P.map(function (p) { return Math.abs(p.residual); }));
  P.forEach(function (p) {
    svg.appendChild(D.el('circle', {cx: sx(p.nll), cy: sy(p.fit_nll), r: 3, fill: shade(tiers.indexOf(p.d_exp)), stroke: 'var(--paper)', 'stroke-width': 1}));
  });
  D.text(svg, side[1], 36, 'largest miss ' + worst.toFixed(3) + ' · none held out', {class: 'diagram-note', 'text-anchor': 'end'});
  D.text(svg, L0, H - 8, 'L(N, D) = ' + F.base.toFixed(3) + ' + ' + F.aScale.toFixed(1) + ' / N^' + F.aPower + ' + ' + F.dScale + ' / D^' + F.dPower, {class: 'diagram-note'});
  D.text(svg, W - 16, H - 8, 'D = supervised positions · compute = 6 · N · D', {class: 'diagram-note', 'text-anchor': 'end'});
  svg.addEventListener('mouseleave', function () { readout.innerHTML = ''; });
}());
