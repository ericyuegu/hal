/* Attach: the past-action figure's result block carries the difference with its 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var host = document.getElementById('past-action-list');
  if (!D || !CI || !host) return;
  var d = CI.past_action_encoding;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.pastActionEffect = function () {
    return [{comparison: 'physical values − 256-way one-hot', treatment_run: 'cqbbbg77', baseline_run: '1zzstd2d', method: d.method, stock_delta: d.stock.delta, stock_ci95_lo: d.stock.lo, stock_ci95_hi: d.stock.hi, damage_delta: d.damage.delta, damage_ci95_lo: d.damage.lo, damage_ci95_hi: d.damage.hi}];
  };
  var block = host.querySelector('.delta-block');
  if (!block) return;
  function line(name, m, digits) {
    return '<div class="delta-row" style="grid-template-columns:auto 1fr"><span class="delta-name">' + name + '</span><span class="delta-value">' + D.signed(m.delta, digits) + ' [' + D.signed(m.lo, digits) + ', ' + D.signed(m.hi, digits) + ']</span></div>';
  }
  block.innerHTML = '<div class="delta-block-title">physical − one-hot · 95% interval</div>' + line('stocks / min', d.stock, 2) + line('damage / min', d.damage, 1);
}());
