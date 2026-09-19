/* Fig: the previous controller frame as a one-hot class against a learned class embedding plus its
   physical values (experiments/043_legacy_codec.py group_embedding), with the matched difference. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('past-action-diagram');
  if (!D || !CI || !svg) return;
  var d = CI.past_action_encoding;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.pastActionEffect = function () {
    return [{comparison: '256-way one-hot → learned class + physical value', treatment_run: 'cqbbbg77', baseline_run: '1zzstd2d', method: d.method, stock_delta: d.stock.delta, stock_ci95_lo: d.stock.lo, stock_ci95_hi: d.stock.hi, damage_delta: d.damage.delta, damage_ci95_lo: d.damage.lo, damage_ci95_hi: d.damage.hi}];
  };
  svg.setAttribute('viewBox', '0 0 880 290');
  var mk = D.marker(svg, 'arrow', 'var(--ink-faint)');
  var H = 28, GX = 92, GW = 52, CX = 182, CW = 36, OUT = 580, OUTW = 128;   // given, class and produced columns
  var Y1 = 40, Y2 = 176;                                                    // row centres
  function arrow(x1, y1, x2, y2) { D.wire(svg, x1, y1, x2, y2, mk); }
  function label(x, y, s) { D.text(svg, x, y, s, {class: 'diagram-note', 'text-anchor': 'middle'}); }
  function bend(x1, y1, x2, y2) { D.path(svg, 'M ' + x1 + ' ' + y1 + ' C ' + (x1 + 30) + ' ' + y1 + ', ' + (x2 - 30) + ' ' + y2 + ', ' + x2 + ' ' + y2, mk); }

  // row 1 · one-hot: the class index as a 256-wide indicator vector, concatenated as is
  D.text(svg, 16, Y1 + 4, 'one-hot', {class: 'diagram-title'});
  D.token(svg, GX, Y1 - H / 2, GW, H, 'aₜ₋₁', 'given');
  arrow(GX + GW + 6, Y1, CX - 8, Y1);
  D.token(svg, CX, Y1 - H / 2, CW, H, 'c');
  label(CX + CW / 2, Y1 + H / 2 + 14, 'class');
  arrow(CX + CW + 6, Y1, OUT - 8, Y1);
  svg.appendChild(D.el('rect', {x: OUT, y: Y1 - H / 2, width: OUTW, height: H, class: 'model-token'}));
  svg.appendChild(D.el('rect', {x: OUT + 43, y: Y1 - H / 2, width: 2, height: H, class: 'model-token played'}));
  label(OUT + OUTW / 2, Y1 + H / 2 + 14, 'one-hot(c)');

  // row 2 · learned: class embedding plus a linear map of the physical values, normalised
  D.text(svg, 16, Y2 + 4, 'learned', {class: 'diagram-title'});
  D.token(svg, GX, Y2 - H / 2, GW, H, 'aₜ₋₁', 'given');
  arrow(GX + GW + 6, Y2, CX - 8, Y2);
  D.token(svg, CX, Y2 - H / 2, CW, H, 'c');
  label(CX + CW / 2, Y2 + H / 2 + 14, 'class');
  var UP = Y2 - 34, DOWN = Y2 + 34, BX = 274;
  bend(CX + CW + 6, Y2, BX - 8, UP);
  bend(CX + CW + 6, Y2, BX - 8, DOWN);
  D.box(svg, BX, UP - H / 2, 92, H, 'Embedding');
  D.token(svg, BX, DOWN - H / 2, 40, H, 'v(c)');
  label(BX + 20, DOWN + H / 2 + 14, 'stick · trigger values');
  arrow(BX + 46, DOWN, BX + 62, DOWN);
  D.box(svg, BX + 70, DOWN - H / 2, 56, H, 'linear');
  var SX = BX + 70 + 56 + 34;                                                // both branches into +
  bend(BX + 92 + 6, UP, SX - 14, Y2);
  bend(BX + 70 + 56 + 6, DOWN, SX - 14, Y2);
  svg.appendChild(D.el('circle', {cx: SX, cy: Y2, r: 11, class: 'model-box'}));
  D.text(svg, SX, Y2 + 4, '+', {'text-anchor': 'middle', class: 'diagram-title'});
  arrow(SX + 12, Y2, SX + 34, Y2);
  D.box(svg, SX + 40, Y2 - H / 2, 72, H, 'RMSNorm');
  arrow(SX + 40 + 72 + 6, Y2, OUT - 8, Y2);
  D.token(svg, OUT, Y2 - H / 2, 44, H, 'e(c)');
  label(OUT + 22, Y2 + H / 2 + 14, 'per controller group');

  // the matched difference
  D.effectBars(svg, [Object.assign({label: 'one-hot → learned', sub: 'O28'}, d)],
    {x: 720, y: 12, panelW: 156, panelH: 250, barW: 34, axisW: 44, metrics: ['stock']});
}());
