/* Fig: each player identity against the masked identity, on one trained model. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('identity-effects-diagram');
  if (!D || !CI || !svg) return;
  var NAMES = ['Cody', 'Master', 'Mang0', 'Platinum'];
  var rows = [{label: 'identity masked', sub: 'u1gvby0r · baseline', baseline: true}].concat(NAMES.map(function (n) {
    return Object.assign({label: n, sub: 'dg0u00w5 · same checkpoint', accent: n === 'Cody'}, CI['player_identity:' + n]);
  }));
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.identityEffects = function () {
    return rows.filter(function (r) { return !r.baseline; }).map(function (r) { return {comparison: 'masked → ' + r.label, method: r.method, boots: r.n_boots, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 44, header: 'masked → each identity · 95% interval'});
}());
