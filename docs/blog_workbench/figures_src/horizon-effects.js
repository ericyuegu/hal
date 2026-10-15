/* Fig: execution horizon per O37 cell, as change from one frame per plan with 95% intervals. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('horizon-effects-diagram');
  if (!D || !CI || !svg) return;
  var CELLS = [
    {id: 'D0', name: 'nothing', run: '98r9smrj'},
    {id: 'D1', name: 'earlier groups', run: 'a117chkw'},
    {id: 'D2', name: 'earlier frames', run: '50q39o9j'},
    {id: 'D3', name: 'frames and groups', run: '5wfk2esf', accent: true}
  ];
  var HS = [2, 4, 6];
  var series = CELLS.map(function (c) {
    return Object.assign({}, c, {points: HS.map(function (h) { return Object.assign({h: h}, CI['horizon:' + c.id + ':H' + h + '-H1']); })});
  });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.horizonEffects = function () {
    var out = [];
    series.forEach(function (s) { s.points.forEach(function (p) { out.push({o37_cell: s.id, conditioning: s.name, run: s.run, frames_per_plan: p.h, baseline_frames_per_plan: 1, method: p.method, stock_delta: p.stock.delta, stock_ci95_lo: p.stock.lo, stock_ci95_hi: p.stock.hi, damage_delta: p.damage.delta, damage_ci95_lo: p.damage.lo, damage_ci95_hi: p.damage.hi}); }); });
    return out;
  };

  var W = 880, H = 360, L = 70, R = 250, top = 48, bottom = H - 44;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var all = [];
  series.forEach(function (s) { s.points.forEach(function (p) { all.push(p.stock); }); });
  var dom = D.domain(all);
  var y = D.scale([bottom, top], dom.lo, dom.hi);
  var lane = (W - L - R - 30) / 5;
  function x(h, k) { return L + 15 + (h - 1) * lane + (h === 1 ? 0 : (k - 1.5) * 8); }
  D.text(svg, 16, 22, 'aₜ₊₂ conditioned on · Δ net stocks / min from 1 frame per plan · dot and 95% interval', {class: 'diagram-note'});
  dom.ticks.forEach(function (v) {
    svg.appendChild(D.el('line', {x1: L, y1: y(v), x2: W - R, y2: y(v), class: v === 0 ? 'effect-zero' : 'effect-grid'}));
    D.text(svg, L - 8, y(v) + 4, D.signed(v, dom.digits), {class: 'diagram-note', 'text-anchor': 'end'});
  });
  [1,2,4,6].forEach(function (h) { D.text(svg, x(h,1.5), bottom + 18, h + ' frame' + (h===1?'':'s'), {class: 'diagram-note', 'text-anchor': 'middle'}); });
  D.text(svg, W - R - 8, top - 10, '1 frame per plan = baseline', {class: 'diagram-note', 'text-anchor': 'end'});
  svg.appendChild(D.el('circle',{cx:x(1,0),cy:y(0),r:4,class:'effect-pt'}));
  series.forEach(function (s, k) {
    var pts = [x(1,k) + ',' + y(0)].concat(s.points.map(function (p) { return x(p.h, k) + ',' + y(p.stock.delta); }));
    svg.appendChild(D.el('polyline', {points: pts.join(' '), class: 'effect-line' + (s.accent ? ' accent' : '')}));
    s.points.forEach(function (p) {
      var cx = x(p.h, k), cls = s.accent ? ' accent' : '';
      svg.appendChild(D.el('line', {x1: cx, y1: y(p.stock.lo), x2: cx, y2: y(p.stock.hi), class: 'effect-ci' + cls}));
      svg.appendChild(D.el('circle', {cx: cx, cy: y(p.stock.delta), r: 3.6, class: 'effect-pt' + cls}));
    });
    // legend: name and the six-frame value
    var last = s.points[s.points.length - 1];
    var ly = y(last.stock.delta) + (k===2?-22:k===3?22:0);
    svg.appendChild(D.el('line',{x1:x(6,k)+5,y1:y(last.stock.delta),x2:W-R+44,y2:ly,class:'effect-line'+(s.accent?' accent':'')}));
    svg.appendChild(D.el('line', {x1: W - R + 24, y1: ly, x2: W - R + 44, y2: ly, class: 'effect-line' + (s.accent ? ' accent' : '')}));
    svg.appendChild(D.el('circle', {cx: W - R + 34, cy: ly, r: 3.2, class: 'effect-pt' + (s.accent ? ' accent' : '')}));
    D.text(svg, W - R + 54, ly + 4, s.name, {class: 'diagram-title' + (s.accent ? ' accent' : '')});
    D.text(svg, W - R + 54, ly + 18, '1 → 6 frames: ' + D.signed(last.stock.delta, 2), {class: 'diagram-note'});
  });
}());
