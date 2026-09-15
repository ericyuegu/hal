/* Fig: ranked-anonymized data only → all 44 sources, at equal updates. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('data-source-effects-diagram');
  if (!D || !CI || !svg) return;
  var rows = [Object.assign({label: 'ranked only → all 44 sources', sub: 'k8nb00xv → bevhq159 · O42 · equal updates', accent: true}, CI.data_source_mix)];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.dataSourceEffects = function () {
    return rows.map(function (r) { return {comparison: r.label, runs: r.sub.split(' · ')[0], method: r.method, boots: r.n_boots, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 48});
}());
