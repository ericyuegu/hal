/* Fig: the four O37 cells in the given→produced grammar, each cell's closed-loop result as its
   difference from the unconditioned cell with a 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('deps-diagram');
  if (!D || !CI || !svg) return;
  var FAINT = 'var(--ink-faint)';
  var mk = D.marker(svg, 'arrow', FAINT);
  svg.setAttribute('viewBox', '0 0 920 330');

  var CELLS = [
    {id: 'D0', name: 'nothing', across: false, within: false, stock: -0.702, damage: 2.77, run: '98r9smrj'},
    {id: 'D1', name: 'earlier groups', across: false, within: true, stock: -0.313, damage: 13.05, run: 'a117chkw'},
    {id: 'D2', name: 'earlier frames', across: true, within: false, stock: -0.196, damage: 18.06, run: '50q39o9j'},
    {id: 'D3', name: 'frames and groups', across: true, within: true, stock: -0.096, damage: 24.75, run: '5wfk2esf', cur: true}
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  CELLS.forEach(function (c) { c.vs = c.id === 'D0' ? null : CI['o37:' + c.id + '-D0']; });
  var MAIN = [['across frames', CI['o37:across']], ['within the frame', CI['o37:within']]];
  window.HALFIG_CUSTOM.data.o37Cells = function () {
    return CELLS.map(function (c) {
      var r = {o37_cell: c.id, conditioning: c.name, across_frames: c.across, within_frame: c.within, net_stocks_per_min: c.stock, net_damage_per_min: c.damage, run: c.run};
      if (c.vs) { r.stock_delta_vs_D0 = c.vs.stock.delta; r.stock_ci95_lo = c.vs.stock.lo; r.stock_ci95_hi = c.vs.stock.hi; r.damage_delta_vs_D0 = c.vs.damage.delta; r.damage_ci95_lo = c.vs.damage.lo; r.damage_ci95_hi = c.vs.damage.hi; r.method = c.vs.method; }
      return r;
    });
  };
  window.HALFIG_CUSTOM.data.o37MainEffects = function () {
    return MAIN.map(function (m) { return {conditioning: m[0], averaged_over: 'the other axis', method: m[1].method, stock_delta: m[1].stock.delta, stock_ci95_lo: m[1].stock.lo, stock_ci95_hi: m[1].stock.hi, damage_delta: m[1].damage.delta, damage_ci95_lo: m[1].damage.lo, damage_ci95_hi: m[1].damage.hi}; });
  };

  var W = 44, H = 28, GAP = 6;
  var X0 = 158;                  // arrow column
  var TOK = X0 + 56;             // first token column
  var GRP = 396;                 // the written frame, opened into its groups
  var COL = {stock: [604, 704], damage: [764, 864]};
  var y0 = 60, step = 62;

  D.text(svg, 16, 28, 'aₜ₊₂ conditioned on', {class: 'diagram-note'});
  D.text(svg, TOK, 28, 'across frames · given', {class: 'diagram-note'});
  D.text(svg, GRP, 28, 'within the frame · given', {class: 'diagram-note'});
  D.text(svg, (COL.stock[0] + COL.stock[1]) / 2, 28, 'Δ net stocks / min', {class: 'diagram-note', 'text-anchor': 'middle'});
  D.text(svg, (COL.damage[0] + COL.damage[1]) / 2, 28, 'Δ net damage / min', {class: 'diagram-note', 'text-anchor': 'middle'});
  D.text(svg, 908, 14, 'vs the cell conditioned on nothing · dot and 95% interval', {class: 'diagram-note', 'text-anchor': 'end'});

  var live = CELLS.filter(function (c) { return c.vs; });
  var sdom = D.domain(live.map(function (c) { return c.vs.stock; })), ddom = D.domain(live.map(function (c) { return c.vs.damage; }));
  var sx = D.scale(COL.stock, sdom.lo, sdom.hi), dx = D.scale(COL.damage, ddom.lo, ddom.hi);
  var yEnd = y0 + 3 * step + H + 8;
  D.effectAxis(svg, sx, sdom, y0 - 8, yEnd);
  D.effectAxis(svg, dx, ddom, y0 - 8, yEnd);

  CELLS.forEach(function (c, i) {
    var y = y0 + i * step;
    D.text(svg, 16, y + H / 2 + 4, c.name, {class: 'diagram-title' + (c.cur ? ' accent' : '')});
    // what the decoder is handed
    D.wire(svg, X0, y + H / 2, X0 + 48, y + H / 2, mk);
    if (i === 0) D.text(svg, X0 + 18, y + H / 2 - 6, 'hₜ', {class: 'diagram-label'});
    var given = c.across ? ['aₜ', 'aₜ₊₁'] : [];
    given.forEach(function (g, k) { D.token(svg, TOK + k * (W + GAP), y, W, H, g, 'given'); });
    D.token(svg, TOK + 2 * (W + GAP), y, W, H, 'aₜ₊₂');
    // the written frame, opened into its groups
    D.wire(svg, TOK + 3 * (W + GAP) - 2, y + H / 2, GRP - 8, y + H / 2, mk);
    ['C', 'M', 'T', 'B'].forEach(function (g, k) {
      var x = GRP + k * (k === 3 && c.within ? 52 : 38);
      var last = k === 3;
      D.token(svg, x, y, 30, H, g, c.within && !last ? 'given' : '');
    });
    if (c.within) D.wire(svg, GRP + 3 * 38 + 2, y + H / 2, GRP + 3 * 52 - 4, y + H / 2, mk);
    // results: difference from the unconditioned cell
    if (!c.vs) {
      D.text(svg, sx(0) + 6, y + H / 2 + 4, 'baseline', {class: 'diagram-note'});
      D.text(svg, dx(0) + 6, y + H / 2 + 4, 'baseline', {class: 'diagram-note'});
      return;
    }
    D.effect(svg, sx, y + H / 2, c.vs.stock, {accent: c.cur, digits: 2, valueX: COL.stock[1] + 8});
    D.effect(svg, dx, y + H / 2, c.vs.damage, {accent: c.cur, digits: 1, valueX: COL.damage[1] + 8});
  });
}());
