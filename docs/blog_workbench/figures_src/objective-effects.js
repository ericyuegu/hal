/* Fig: alternative objectives and decoders, each as treatment − behavior cloning with a 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('objective-effects-diagram');
  if (!D || !CI || !svg) return;
  var ROWS = [
    {key: 'attached_iql', label: 'attached IQL', sub: 'actor + Q₁/Q₂/V losses · vs behavior cloning', runs: ['06fi5jms', 'cqbbbg77']},
    {key: 'endpoint_flow', label: 'endpoint flow decoder', sub: 'parallel flow over ten offsets · vs the causal decoder', runs: ['t4s3gohe', '5wfk2esf']},
    {key: 'exploration_modeling:XM K=1', label: 'exploration modeling · 1 plan', sub: 'one latent trajectory · vs behavior cloning', runs: ['mz8dk2gh', 'cqbbbg77']},
    {key: 'exploration_modeling:XM K=8', label: 'exploration modeling · 8 plans', sub: 'best of eight latent trajectories · vs behavior cloning', runs: ['feic18gt', 'cqbbbg77']}
  ];
  var rows = ROWS.map(function (r) { return Object.assign({}, r, CI[r.key]); });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.objectiveEffects = function () {
    return rows.map(function (r) { return {comparison: r.label, treatment_run: r.runs[0], baseline_run: r.runs[1], method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 50});
}());
