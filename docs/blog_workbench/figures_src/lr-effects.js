/* Fig: the paired Muon/AdamW learning-rate sweep, the smallest pair → each arm with a 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('lr-effects-diagram');
  if (!D || !CI || !svg) return;
  var ARMS = [['0.014', '0.000595', '9luwu85n'], ['0.028', '0.00119', 'wmwellfd'], ['0.040', '0.00170', '7hy8xjjh'], ['0.056', '0.00238', '2rbed6ze'], ['0.080', '0.00340', '5adh9mtu'], ['0.112', '0.00476', 'zerhe8au']];
  var rows = ARMS.map(function (a) {
    var label = 'Muon ' + a[0] + ' · AdamW ' + a[1];
    if (a[0] === '0.014') return {label: label, sub: 'baseline', baseline: true, run: a[2]};
    return Object.assign({label: label, run: a[2], accent: a[0] === '0.040'}, CI['lr:' + a[0] + '-0.014']);
  });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.lrEffects = function () {
    return rows.filter(function (r) { return !r.baseline; }).map(function (r) { return {comparison: 'Muon 0.014 · AdamW 0.000595 → ' + r.label, treatment_run: r.run, baseline_run: '9luwu85n', method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 40, header: 'smallest pair → each row · 95% interval'});
}());
