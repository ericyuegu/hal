/* Fig: the two optimizer-recipe tests, with each arm's validation loss beside the play difference. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI, DATA = window.HALFIG_DATA;
  var svg = document.getElementById('optimizer-effects-diagram');
  if (!D || !CI || !DATA || !svg) return;
  var V = DATA.optimizer;
  var rows = [
    Object.assign({label: 'AdamW → Muon + AdamW', sub: 'o3nihah5 → vlim96s9 · 7M · unpaired', loss: [V.o3nihah5, V.vlim96s9]}, CI.optimizer_small),
    Object.assign({label: 'Muon + AdamW → AdamW', sub: 'p1fyyp1z → tgpuo1be · 216M production', accent: true, loss: [V.p1fyyp1z, V.tgpuo1be]}, CI.optimizer_full_scale)
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.optimizerEffects = function () {
    return rows.map(function (r) { return {comparison: r.label, runs: r.sub.split(' · ')[0], method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi, baseline_val_loss: r.loss[0].val_loss, treatment_val_loss: r.loss[1].val_loss, val_metric: r.loss[0].metric}; });
  };
  var top = 60, rowH = 56;
  D.forest(svg, rows, {rowH: rowH, top: top, cols: [[340, 480], [560, 690]], header: 'baseline → treatment · 95% interval'});
  // a third column: the validation loss of each arm, read baseline → treatment
  var vx = 766;
  D.text(svg, vx, top - 28, 'validation loss', {class: 'diagram-note'});
  D.text(svg, vx, top - 14, 'before → after', {class: 'diagram-note'});
  rows.forEach(function (r, i) {
    var y = top + i * rowH + rowH / 2 - 6;
    D.text(svg, vx, y + 4, r.loss[0].val_loss.toFixed(3) + ' → ' + r.loss[1].val_loss.toFixed(3), {class: 'diagram-label'});
    D.text(svg, vx, y + 18, r.loss[0].metric === 'val/loss' ? 'older loss metric' : 'validation NLL', {class: 'diagram-note'});
  });
}());
