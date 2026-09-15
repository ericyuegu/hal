/* Fig: validation NLL over gameplay, aligned on updates, for the O50 production run. */
(function () {
  'use strict';
  var D = window.HALDIAG, DATA = window.HALFIG_DATA;
  var svg = document.getElementById('training-curve-diagram');
  var readout = document.getElementById('training-curve-readout');
  if (!D || !DATA || !svg) return;
  var S = DATA.o50;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.trainingCurve = function () {
    var byUpdate = {};
    S.val_updates.forEach(function (u, i) { byUpdate[u] = {run: S.run, updates: u, val_nll: S.val_nll[i]}; });
    S.eval_updates.forEach(function (u, i) { byUpdate[u] = Object.assign(byUpdate[u] || {run: S.run, updates: u}, {net_stocks_per_min: S.eval_stock[i], net_damage_per_min: S.eval_damage[i]}); });
    return Object.keys(byUpdate).map(Number).sort(function (a, b) { return a - b; }).map(function (u) { return byUpdate[u]; });
  };

  var W = 880, H = 400, L = 76, R = 24;
  var top = {y0: 44, y1: 168}, bot = {y0: 216, y1: 352};
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var xmax = S.val_updates[S.val_updates.length - 1];
  var x = D.scale([L, W - R], 0, xmax);
  var nllLo = 1.35, nllHi = 1.60, stLo = -0.3, stHi = 0.4;
  var yN = D.scale([top.y1, top.y0], nllLo, nllHi);
  var yS = D.scale([bot.y1, bot.y0], stLo, stHi);

  // the dip: play fell from the 24,576 checkpoint to 57,344 while the loss kept falling
  var dip = [24576, 57344];
  [top, bot].forEach(function (p) {
    svg.appendChild(D.el('rect', {x: x(dip[0]), y: p.y0, width: x(dip[1]) - x(dip[0]), height: p.y1 - p.y0, class: 'chart-band'}));
  });
  D.text(svg, (x(dip[0]) + x(dip[1])) / 2, bot.y1 + 18, 'play falls while the loss holds', {class: 'diagram-note', 'text-anchor': 'middle'});

  // axes
  function yAxis(p, y, ticks, fmt, label) {
    ticks.forEach(function (v) {
      svg.appendChild(D.el('line', {x1: L, y1: y(v), x2: W - R, y2: y(v), class: v === 0 ? 'effect-zero' : 'effect-grid'}));
      D.text(svg, L - 8, y(v) + 4, fmt(v), {class: 'diagram-note', 'text-anchor': 'end'});
    });
    D.text(svg, L, p.y0 - 12, label, {class: 'diagram-note'});
  }
  yAxis(top, yN, [1.4, 1.5, 1.6], function (v) { return v.toFixed(1); }, 'validation NLL · lower is better');
  yAxis(bot, yS, [-0.2, 0, 0.2, 0.4], function (v) { return D.signed(v, 1); }, 'net stocks / min · 96 games per checkpoint');
  [0, 32768, 65536, 98304, 131072].forEach(function (u) {
    D.text(svg, x(u), bot.y1 + 34, u === 0 ? '0' : Math.round(u / 1000) + 'k', {class: 'diagram-note', 'text-anchor': 'middle'});
  });
  D.text(svg, W - R, bot.y1 + 48, 'optimizer updates', {class: 'diagram-note', 'text-anchor': 'end'});

  // series
  var nllPts = S.val_updates.map(function (u, i) { return x(u) + ',' + yN(S.val_nll[i]); });
  svg.appendChild(D.el('polyline', {points: nllPts.join(' '), class: 'chart-line'}));
  var stPts = S.eval_updates.map(function (u, i) { return x(u) + ',' + yS(S.eval_stock[i]); });
  svg.appendChild(D.el('polyline', {points: stPts.join(' '), class: 'effect-line accent'}));
  S.eval_updates.forEach(function (u, i) {
    svg.appendChild(D.el('circle', {cx: x(u), cy: yS(S.eval_stock[i]), r: 4, class: 'effect-pt accent'}));
  });
  var last = S.eval_updates.length - 1;
  D.text(svg, x(S.eval_updates[last]) - 8, yS(S.eval_stock[last]) - 8, D.signed(S.eval_stock[last], 2), {class: 'diagram-label accent-text', 'text-anchor': 'end'});
  D.text(svg, x(S.eval_updates[2]), yS(S.eval_stock[2]) - 9, D.signed(S.eval_stock[2], 2), {class: 'diagram-label', 'text-anchor': 'middle'});
  D.text(svg, x(S.eval_updates[4]), yS(S.eval_stock[4]) + 18, D.signed(S.eval_stock[4], 2), {class: 'diagram-label', 'text-anchor': 'middle'});

  // hover: nearest validation point, plus the checkpoint if one sits there
  var cursor = D.el('line', {x1: 0, x2: 0, y1: top.y0, y2: bot.y1, class: 'chart-cursor', visibility: 'hidden'});
  svg.appendChild(cursor);
  function show(u) {
    var i = S.val_updates.indexOf(u), k = S.eval_updates.indexOf(u);
    cursor.setAttribute('x1', x(u)); cursor.setAttribute('x2', x(u)); cursor.setAttribute('visibility', 'visible');
    readout.innerHTML = '<span><b>' + u.toLocaleString('en-US') + '</b> updates</span><span>validation NLL <b>' + S.val_nll[i].toFixed(3) + '</b></span>' +
      (k >= 0 ? '<span>net stocks / min <b>' + D.signed(S.eval_stock[k], 2) + '</b></span><span>net damage / min <b>' + D.signed(S.eval_damage[k], 1) + '</b></span>' : '<span>no gameplay evaluation at this checkpoint</span>');
  }
  svg.addEventListener('mousemove', function (e) {
    var pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    var p = pt.matrixTransform(svg.getScreenCTM().inverse());
    var best = S.val_updates[0], bd = Infinity;
    S.val_updates.forEach(function (u) { var d = Math.abs(x(u) - p.x); if (d < bd) { bd = d; best = u; } });
    show(best);
  });
  svg.addEventListener('mouseleave', function () { cursor.setAttribute('visibility', 'hidden'); readout.innerHTML = ''; });
  readout.innerHTML = '<span>hover for the values at each checkpoint</span>';
}());
