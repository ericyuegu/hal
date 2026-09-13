/* Fig: the four O37 cells in the given→produced grammar, with each cell's closed-loop result. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('deps-diagram');
  if (!D || !svg) return;
  var FAINT = 'var(--ink-faint)';
  var mk = D.marker(svg, 'arrow', FAINT);
  svg.setAttribute('viewBox', '0 0 880 330');

  var CELLS = [
    {id: 'D0', name: 'no feedback', across: false, within: false, stock: -0.702, damage: 2.77, run: '98r9smrj'},
    {id: 'D1', name: 'within the frame', across: false, within: true, stock: -0.313, damage: 13.05, run: 'a117chkw'},
    {id: 'D2', name: 'across frames', across: true, within: false, stock: -0.196, damage: 18.06, run: '50q39o9j'},
    {id: 'D3', name: 'across and within', across: true, within: true, stock: -0.096, damage: 24.75, run: '5wfk2esf', cur: true}
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.o37Cells = function () { return CELLS.map(function (c) { return {o37_cell: c.id, feedback: c.name, across_frames: c.across, within_frame: c.within, net_stocks_per_min: c.stock, net_damage_per_min: c.damage, run: c.run}; }); };

  var W = 44, H = 28, GAP = 6;
  var X0 = 158;                  // arrow column
  var TOK = X0 + 56;             // first token column
  var GRP = 400;                 // the written frame, opened into its groups
  var BAR = {stock: [612, 700], damage: [736, 852]};
  var y0 = 60, step = 62;

  D.text(svg, 16, 28, 'feedback', {class: 'diagram-note'});
  D.text(svg, TOK, 28, 'handed to the decoder', {class: 'diagram-note'});
  D.text(svg, GRP, 28, 'aₜ₊₂ by group', {class: 'diagram-note'});
  D.text(svg, (BAR.stock[0] + BAR.stock[1]) / 2, 28, 'net stocks / min', {class: 'diagram-note', 'text-anchor': 'middle'});
  D.text(svg, (BAR.damage[0] + BAR.damage[1]) / 2, 28, 'net damage / min', {class: 'diagram-note', 'text-anchor': 'middle'});

  function scale(range, lo, hi) { return function (v) { return range[0] + (v - lo) / (hi - lo) * (range[1] - range[0]); }; }
  var sx = scale(BAR.stock, -0.8, 0), dx = scale(BAR.damage, 0, 28);
  // zero lines
  svg.appendChild(D.el('line', {x1: sx(0), y1: y0 - 8, x2: sx(0), y2: y0 + 3 * step + H + 8, class: 'model-wire'}));
  svg.appendChild(D.el('line', {x1: dx(0), y1: y0 - 8, x2: dx(0), y2: y0 + 3 * step + H + 8, class: 'model-wire'}));

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
      var x = GRP + k * 40;
      D.token(svg, x, y, 30, H, g, c.within && k ? '' : (k ? 'given' : ''));
      if (c.within && k) D.wire(svg, x - 9, y + H / 2, x - 1, y + H / 2, mk);
    });
    // results
    svg.appendChild(D.el('rect', {x: Math.min(sx(0), sx(c.stock)), y: y + 6, width: Math.abs(sx(c.stock) - sx(0)), height: H - 12, fill: c.cur ? 'var(--oxblood)' : 'var(--ink-faint)'}));
    D.text(svg, sx(c.stock) - 6, y + H / 2 + 4, c.stock.toFixed(2), {class: 'diagram-label', 'text-anchor': 'end'});
    svg.appendChild(D.el('rect', {x: dx(0), y: y + 6, width: dx(c.damage) - dx(0), height: H - 12, fill: c.cur ? 'var(--oxblood)' : 'var(--ink-faint)'}));
    D.text(svg, dx(c.damage) + 6, y + H / 2 + 4, '+' + c.damage.toFixed(1), {class: 'diagram-label'});
  });
  D.text(svg, sx(0), y0 + 3 * step + H + 20, '0', {class: 'diagram-note', 'text-anchor': 'middle'});
  D.text(svg, dx(0), y0 + 3 * step + H + 20, '0', {class: 'diagram-note', 'text-anchor': 'middle'});
}());
