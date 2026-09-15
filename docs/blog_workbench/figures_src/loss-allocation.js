/* Fig: loss weight per predicted offset under the O26 split and the 50%-next-frame split, then the comparison. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('loss-allocation-diagram');
  if (!D || !CI || !svg) return;
  var OFFS = [1, 2, 3, 4, 5, 6, 9, 12, 16, 20];
  // experiments/043_legacy_codec.py _joint_objective: O26 = mean over offsets 1–4 plus mean over the rest
  // (total 2); the treatment = 2 · (½ · offset 1 + ½ · mean over the other nine)
  var ROWS = [
    {label: 'O26 split', sub: 'near 4 share 1 · far 6 share 1', run: 'acbbeb76', w: OFFS.map(function (o, i) { return i < 4 ? 0.25 : 1 / 6; })},
    {label: 'half on the next frame', sub: 'next frame 1 · other 9 share 1', run: '1imfy8v3', accent: true, w: OFFS.map(function (o, i) { return i === 0 ? 1 : 1 / 9; })}
  ];
  var d = CI.next_frame_loss_share;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.lossAllocation = function () {
    var out = [];
    ROWS.forEach(function (r) { OFFS.forEach(function (o, i) { out.push({allocation: r.label, run: r.run, offset: o, loss_weight: r.w[i]}); }); });
    out.push({comparison: 'O26 split → half on the next frame', runs: 'acbbeb76 → 1imfy8v3', method: d.method, stock_delta: d.stock.delta, stock_ci95_lo: d.stock.lo, stock_ci95_hi: d.stock.hi, damage_delta: d.damage.delta, damage_ci95_lo: d.damage.lo, damage_ci95_hi: d.damage.hi});
    return out;
  };

  var W = 880, H = 396;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var x0 = 250, tw = 48, gap = 12, barMax = 60;
  function sub(o) { return 'aₜ₊' + String(o).replace(/\d/g, function (c) { return '₀₁₂₃₄₅₆₇₈₉'[c]; }); }
  ROWS.forEach(function (r, k) {
    var base = 118 + k * 130, cls = r.accent ? ' accent' : '';
    D.text(svg, 16, base - 40, r.label, {class: 'diagram-title' + cls});
    D.text(svg, 16, base - 24, r.sub, {class: 'diagram-note'});
    D.text(svg, 16, base - 10, r.run, {class: 'diagram-note'});
    OFFS.forEach(function (o, i) {
      var x = x0 + i * (tw + gap), h = r.w[i] * barMax;
      svg.appendChild(D.el('rect', {x: x + 10, y: base - 8 - h, width: tw - 20, height: h, class: 'loss-bar' + cls}));
      D.text(svg, x + tw / 2, base - 12 - h, r.w[i].toFixed(2).replace(/^0/, ''), {class: 'diagram-note', 'text-anchor': 'middle'});
      D.token(svg, x, base, tw, 26, sub(o), k === 0 ? 'given' : '');
    });
  });
  D.text(svg, x0, 34, 'loss weight per predicted frame · frames sum to 2 in both rows', {class: 'diagram-note'});
  D.text(svg, x0 + OFFS.length * (tw + gap) - gap, 118 + 130 + 42, 'predicted frame, by offset from now', {class: 'diagram-note', 'text-anchor': 'end'});

  // the comparison row
  var y = 350;
  svg.appendChild(D.el('line', {x1: 16, y1: y - 44, x2: W - 16, y2: y - 44, stroke: 'var(--rule-dotted)', 'stroke-dasharray': '2 3'}));
  D.text(svg, 16, y - 4, 'O26 split → half on the next frame', {class: 'diagram-title accent'});
  D.text(svg, 16, y + 10, 'acbbeb76 → 1imfy8v3 · O43 · 95% interval', {class: 'diagram-note'});
  var doms = [D.domain([d.stock]), D.domain([d.damage])], cols = [[336, 540], [640, 826]];
  doms.forEach(function (dom, k) {
    var xs = D.scale(cols[k], dom.lo, dom.hi);
    D.effectAxis(svg, xs, dom, y - 12, y + 12, k ? 'Δ net damage / min' : 'Δ net stocks / min');
    D.effect(svg, xs, y, k ? d.damage : d.stock, {accent: true, digits: k ? 1 : 2, valueX: cols[k][1] + 8});
  });
}());
