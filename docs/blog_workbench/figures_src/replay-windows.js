/* Fig: four windows per replay → one, and the closed-loop difference with its 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('replay-windows-diagram');
  if (!D || !CI || !svg) return;
  var d = CI.windows_per_replay;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.replayWindows = function () {
    return [{comparison: 'four → one window per replay', treatment_run: 'shjnxxsu', baseline_run: 'uw05bvm2', method: d.method, stock_delta: d.stock.delta, stock_ci95_lo: d.stock.lo, stock_ci95_hi: d.stock.hi, damage_delta: d.damage.delta, damage_ci95_lo: d.damage.lo, damage_ci95_hi: d.damage.hi}];
  };
  var W = 26, H = 18, GAP = 4, y0 = 46, rowStep = H + 6;
  function grid(x, title, hot) {
    D.text(svg, x, 28, title, {class: 'diagram-title'});
    for (var r = 0; r < 4; r += 1) {
      var y = y0 + r * rowStep;
      D.text(svg, x, y + H / 2 + 4, 'replay ' + String.fromCharCode(65 + r), {class: 'diagram-note'});
      for (var s = 0; s < 8; s += 1) {
        svg.appendChild(D.el('rect', {x: x + 64 + s * (W + GAP), y: y, width: W, height: H, class: 'model-token window' + (hot(r, s) ? ' hot' : '')}));
      }
    }
    D.text(svg, x + 64, y0 + 4 * rowStep + 10, 'windows in time order · black = taken into the batch', {class: 'diagram-note'});
  }
  grid(16, 'four windows per replay · baseline', function (r, s) { return s >= 1 && s <= 4; });
  grid(456, 'one window per replay', function (r, s) { return s === 1 + r; });
  D.forest(svg, [Object.assign({label: 'four → one window per replay', sub: 'old evaluation, few matches per arm'}, d)], {top: 212, rowH: 50});
}());
