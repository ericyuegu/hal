/* Fig: batch size at fixed examples, each arm minus batch 2,048 with a 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('batch-effects-diagram');
  if (!D || !CI || !svg) return;
  var ARMS = [[128, 'nt4gh4mk'], [256, 'phxqvexa'], [512, '980k3yla'], [1024, '7ok39dju'], [2048, 'rri7scjc']];
  var rows = ARMS.map(function (a) {
    var b = a[0], updates = (2097152 / b).toLocaleString('en-US') + ' updates';
    if (b === 2048) return {label: 'batch 2,048', sub: updates + ' · baseline', baseline: true, run: a[1]};
    return Object.assign({label: 'batch ' + b.toLocaleString('en-US'), sub: updates, run: a[1], accent: b === 128}, CI['batch:' + b + '-2048']);
  });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.batchEffects = function () {
    return rows.filter(function (r) { return !r.baseline; }).map(function (r) { return {comparison: r.label + ' vs batch 2,048', treatment_run: r.run, baseline_run: 'rri7scjc', method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 46});
}());
