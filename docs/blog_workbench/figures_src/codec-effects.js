/* Fig: the two matched codec comparisons as baseline → treatment with 95% intervals. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('codec-effects-diagram');
  if (!D || !CI || !svg) return;
  var rows = [
    Object.assign({label: 'legacy → Cartesian', sub: 'acbbeb76 → cqbbbg77 · O43 bridge'}, CI.codec_legacy_cartesian),
    Object.assign({label: 'Cartesian → polar', sub: 'yxy8brhx → yi7yyyu0 · O53'}, CI.codec_cartesian_polar)
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.codecEffects = function () {
    return rows.map(function (r) { return {comparison: r.label, runs: r.sub.split(' · ')[0], method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 48});
}());
