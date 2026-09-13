/* Fig: AWR against behavior cloning, and the return half-life sweep, as differences with 95% intervals. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('awr-effects-diagram');
  if (!D || !CI || !svg) return;
  var ROWS = [
    {key: 'awr_actor_weighting', label: 'AWR', sub: 'vs behavior cloning · unpaired evaluations', runs: ['hwzv0k9a', '5wfk2esf']},
    {key: 'awr_gamma:gamma 0.99855', label: 'return half-life 8 s', sub: 'vs 3 s · O52 proxy', runs: ['0ksityvv', 'yxy8brhx'], accent: true},
    {key: 'awr_gamma:gamma 0.99920', label: 'return half-life 14 s', sub: 'vs 3 s · O52 proxy', runs: ['8t9koblm', 'yxy8brhx']}
  ];
  var rows = ROWS.map(function (r) { return Object.assign({}, r, CI[r.key]); });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.awrEffects = function () {
    return rows.map(function (r) { return {comparison: r.label + ' ' + r.sub.split(' · ')[0], treatment_run: r.runs[0], baseline_run: r.runs[1], method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 50});
}());
