/* Attach: replaces the codec figure's arm-level results table with the two matched differences. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var host = document.getElementById('codec-results');
  if (!D || !CI || !host) return;
  var rows = [
    Object.assign({label: 'Cartesian − legacy', sub: 'cqbbbg77 − acbbeb76 · O43 bridge'}, CI.codec_legacy_cartesian),
    Object.assign({label: 'polar − Cartesian', sub: 'yi7yyyu0 − yxy8brhx · O53'}, CI.codec_cartesian_polar)
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.codecEffects = function () {
    return rows.map(function (r) { return {comparison: r.label, runs: r.sub.split(' · ')[0], method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  host.innerHTML = '';
  var svg = D.el('svg', {class: 'model-diagram', role: 'img', 'aria-label': 'Two matched differences with 95% intervals: Cartesian minus legacy gains about 0.3 net stocks and 16 damage per minute, both intervals above zero; polar minus Cartesian is a tenth of a stock with an interval that includes zero.'});
  host.appendChild(svg);
  D.forest(svg, rows, {rowH: 48});
}());
