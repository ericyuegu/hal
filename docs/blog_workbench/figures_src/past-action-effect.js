/* Attach: the past-action figure's result block carries the difference with its 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var host = document.getElementById('past-action-list');
  if (!D || !CI || !host) return;
  var d = CI.past_action_encoding;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.pastActionEffect = function () {
    return [{comparison: '256-way one-hot → physical values', treatment_run: 'cqbbbg77', baseline_run: '1zzstd2d', method: d.method, stock_delta: d.stock.delta, stock_ci95_lo: d.stock.lo, stock_ci95_hi: d.stock.hi, damage_delta: d.damage.delta, damage_ci95_lo: d.damage.lo, damage_ci95_hi: d.damage.hi}];
  };
  var block = host.querySelector('.delta-block');
  if (!block) return;
  function line(name, m, digits) {
    return '<div class="delta-row" style="grid-template-columns:auto 1fr"><span class="delta-name">' + name + '</span><span class="delta-value">' + D.signed(m.delta, digits) + ' [' + D.signed(m.lo, digits) + ', ' + D.signed(m.hi, digits) + ']</span></div>';
  }
  // the budgets differ: the one-hot arm saw twice the examples (batch 1,024 · 16.78M) against batch 512 · 8.39M
  block.innerHTML = '<div class="delta-block-title">one-hot → physical · Δ and 95% interval</div>' + line('stocks / min', d.stock, 2) + line('damage / min', d.damage, 1) +
    '<div class="delta-block-title" style="margin-top:8px">one-hot: batch 1,024 · 16.8M examples<br>physical: batch 512 · 8.4M examples · fewer, and still ahead</div>';
}());
