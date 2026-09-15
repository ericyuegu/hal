/* Fig: O54 hold-out — S(D) fitted through D = 2^30 per model, prediction against observation at 2^31. */
(function () {
  'use strict';
  var D = window.HALDIAG, DATA = window.HALFIG_DATA;
  var svg = document.getElementById('tier-prediction-diagram');
  var readout = document.getElementById('tier-prediction-readout');
  if (!D || !DATA || !svg) return;
  var T = DATA.tier;
  var LABEL = {256: '5M · 1 frame', 512: '39M · 2 frames', 768: '125M · 3 frames', 1024: '289M · 6 frames'};
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.tierPrediction = function () {
    return T.models.map(function (m) { return {model: 'W' + m.width, bc_parameters: m.bc_parameters, run: m.run, fit_s28: m.fit.s28, fit_amplitude: m.fit.amplitude, fit_alpha: m.fit.alpha, predicted_d31: m.predicted_d31, observed_d31: m.observed_d31, observed_ci95_lo: m.observed_d31_ci95[0], observed_ci95_hi: m.observed_d31_ci95[1]}; });
  };
  function curve(m, e) { return m.fit.s28 + m.fit.amplitude * (1 - Math.pow(Math.pow(2, e - 28), -m.fit.alpha)); }
  var superscript = function (n) { return String(n).replace(/\d/g, function (c) { return '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]; }); };

  var W = 880, H = 330, L = 64, R = 16, top = 54, bottom = 262, pw = (W - L - R - 3 * 18) / 4;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var y = D.scale([bottom, top], -1.3, 0.2);
  D.text(svg, L, 22, 'each model\u2019s own score against the CPU · fit through 2³⁰ · hollow = predicted at 2³¹ · filled = observed', {class: 'diagram-note'});
  [-1, -0.5, 0].forEach(function (v) { D.text(svg, L - 8, y(v) + 4, D.signed(v, 1), {class: 'diagram-note', 'text-anchor': 'end'}); });
  T.models.forEach(function (m, k) {
    var px = L + k * (pw + 18);
    var x = D.scale([px + 16, px + pw - 16], 28, 31);
    [-1, -0.5, 0].forEach(function (v) { svg.appendChild(D.el('line', {x1: px, y1: y(v), x2: px + pw, y2: y(v), class: v === 0 ? 'effect-zero' : 'effect-grid'})); });
    D.text(svg, px + pw / 2, top - 14, LABEL[m.width], {class: 'diagram-title' + (m.width === 768 ? ' accent' : ''), 'text-anchor': 'middle'});
    [28, 29, 30, 31].forEach(function (e) { D.text(svg, x(e), bottom + 16, '2' + superscript(e), {class: 'diagram-note', 'text-anchor': 'middle'}); });
    // fitted curve: solid through 2^30, dashed beyond
    var fitted = [], beyond = [];
    for (var e = 28; e <= 31.0001; e += 0.05) (e <= 30 ? fitted : beyond).push(x(e) + ',' + y(curve(m, e)));
    beyond.unshift(x(30) + ',' + y(curve(m, 30)));
    svg.appendChild(D.el('polyline', {points: fitted.join(' '), class: 'chart-line faint'}));
    svg.appendChild(D.el('polyline', {points: beyond.join(' '), class: 'chart-line faint', 'stroke-dasharray': '3 3'}));
    m.points.forEach(function (p) {
      var cx = x(p.d_exp), cls = p.d_exp === 31 ? ' accent' : '';
      svg.appendChild(D.el('line', {x1: cx, y1: y(p.ci95[0]), x2: cx, y2: y(p.ci95[1]), class: 'effect-ci' + cls}));
      svg.appendChild(D.el('circle', {cx: cx, cy: y(p.stock), r: p.d_exp === 31 ? 4.4 : 3.4, class: 'effect-pt' + cls}));
    });
    var pxp = x(31) - 14;
    svg.appendChild(D.el('circle', {cx: pxp, cy: y(m.predicted_d31), r: 4.4, class: 'pred-pt'}));
    var up = m.predicted_d31 > m.observed_d31;
    D.text(svg, up ? pxp - 9 : pxp, y(m.predicted_d31) + (up ? 4 : 17), D.signed(m.predicted_d31, 2), {class: 'diagram-note', 'text-anchor': up ? 'end' : 'middle'});
    D.text(svg, x(31), y(m.observed_d31_ci95[m.predicted_d31 > m.observed_d31 ? 0 : 1]) + (m.predicted_d31 > m.observed_d31 ? 15 : -7), D.signed(m.observed_d31, 2), {class: 'diagram-label accent-text', 'text-anchor': 'middle'});
    var hit = D.el('rect', {x: px, y: top, width: pw, height: bottom - top, fill: 'transparent'});
    svg.appendChild(hit);
    hit.addEventListener('mouseenter', function () {
      readout.innerHTML = '<span><b>' + LABEL[m.width] + '</b> · ' + m.run + '</span><span>fit S₂₈ ' + D.signed(m.fit.s28, 2) + ' · A ' + m.fit.amplitude.toFixed(2) + ' · α ' + m.fit.alpha.toFixed(2) + '</span><span>predicted at 2³¹ <b>' + D.signed(m.predicted_d31, 2) + '</b></span><span>observed <b>' + D.signed(m.observed_d31, 2) + '</b> [' + D.signed(m.observed_d31_ci95[0], 2) + ', ' + D.signed(m.observed_d31_ci95[1], 2) + ']</span>';
    });
  });
  var order = function (list) { return list.map(function (w) { return LABEL[w].split(' ·')[0]; }).join(' > '); };
  D.text(svg, L, H - 24, 'predicted order at 2³¹ · ' + order(T.holdout.predicted_d31_order), {class: 'diagram-note'});
  D.text(svg, L, H - 8, 'observed order at 2³¹ · ' + order(T.holdout.observed_d31_order) + ' · the fits are not used past 2³¹', {class: 'diagram-note'});
  svg.addEventListener('mouseleave', function () { readout.innerHTML = ''; });
}());
