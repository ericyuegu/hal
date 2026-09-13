/* Fig: the physical-shard loader — keyed shard order, one decode per replay, the FIFO replay ring, batch composition. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('loader-diagram');
  if (!D || !svg) return;
  var FAINT = 'var(--ink-faint)', OX = 'var(--oxblood)';
  var mk = D.marker(svg, 'arrow', FAINT);
  var mkOx = D.marker(svg, 'arrow-ox', OX);
  svg.setAttribute('viewBox', '0 0 880 490');
  var W = 4;          // windows per replay drawn (the real value is in the caption)
  var COHORTS = 14;   // cohorts drawn in the ring

  /* ---- 1 · shards in one keyed order → one decode per replay → W windows ---- */
  D.text(svg, 16, 22, 'physical shards · one keyed order, read once', {class: 'diagram-note'});
  for (var i = 0; i < 6; i++) {
    var x = 16 + i * 46;
    svg.appendChild(D.el('rect', {x: x, y: 32, width: 40, height: 28, class: 'model-token' + (i === 0 ? '' : ' given')}));
    D.text(svg, x + 20, 50, 'shard', {'text-anchor': 'middle', class: 'diagram-label' + (i ? ' given-label' : '')});
  }
  D.text(svg, 16 + 6 * 46 + 2, 50, '···', {class: 'diagram-note'});
  D.text(svg, 36, 76, 'next', {class: 'diagram-note', 'text-anchor': 'middle'});
  D.wire(svg, 322, 46, 366, 46, mk);
  D.text(svg, 344, 38, 'decode', {class: 'diagram-note', 'text-anchor': 'middle'});
  // one replay, W windows at distinct starts
  svg.appendChild(D.el('rect', {x: 372, y: 32, width: 300, height: 28, class: 'model-token'}));
  D.text(svg, 372, 22, 'one replay', {class: 'diagram-note'});
  [0.05, 0.31, 0.58, 0.8].forEach(function (f, k) {
    var x = 372 + f * 300;
    svg.appendChild(D.el('rect', {x: x, y: 34, width: 44, height: 24, fill: 'color-mix(in srgb,var(--oxblood) 12%,transparent)', stroke: OX, 'stroke-width': 1}));
    D.text(svg, x + 22, 76, 'window ' + (k + 1), {'text-anchor': 'middle', class: 'diagram-note'});
  });
  D.text(svg, 690, 50, 'W windows · context + chunk', {class: 'diagram-note'});

  /* ---- 2 · the replay ring ---- */
  var ry = 138;
  D.text(svg, 60, ry - 10, 'replay ring · C slots · FIFO cohorts of L replays · each slot keeps its W windows', {class: 'diagram-note'});
  var cw = 52, ch = 92, cx0 = 16;
  for (var c = 0; c < COHORTS; c++) {
    var x = cx0 + c * (cw + 6);
    var newest = c === 0, oldest = c === COHORTS - 1;
    svg.appendChild(D.el('rect', {x: x, y: ry, width: cw, height: ch, class: 'model-token' + (oldest ? ' queued' : ''), 'stroke': newest ? OX : undefined}));
    for (var k = 0; k < W; k++) {
      svg.appendChild(D.el('line', {x1: x + 6, y1: ry + 12 + k * 22, x2: x + cw - 6, y2: ry + 12 + k * 22, stroke: 'var(--rule-dotted)'}));
      if (c === 0) D.text(svg, x + cw / 2, ry + 9 + k * 22, 'w' + (k + 1), {'text-anchor': 'middle', class: 'diagram-note'});
    }
  }
  var ringRight = cx0 + COHORTS * (cw + 6) - 6;
  D.path(svg, 'M 380 62 L 380 96 L 42 96 L 42 ' + (ry - 4), mk);
  D.text(svg, 42, ry + ch + 16, 'newest cohort', {'text-anchor': 'middle', class: 'diagram-note'});
  D.text(svg, ringRight - cw / 2, ry + ch + 16, 'oldest · retired next batch', {'text-anchor': 'middle', class: 'diagram-note'});
  D.text(svg, (cx0 + ringRight) / 2, ry + ch + 16, '→ age · P batches around the ring', {'text-anchor': 'middle', class: 'diagram-note'});

  /* ---- 3 · one batch: window k comes from a cohort about k·P/W batches old ---- */
  var by = 340;

  var gw = 150, gx0 = 16;
  var srcCohorts = [1, 4, 8, 12];
  for (k = 0; k < W; k++) {
    var gx = gx0 + k * (gw + 14);
    svg.appendChild(D.el('rect', {x: gx, y: by, width: gw, height: 34, class: 'model-token'}));
    D.text(svg, gx + gw / 2, by + 21, 'L replays · window ' + (k + 1), {'text-anchor': 'middle', class: 'diagram-label'});
    var sx = cx0 + srcCohorts[k] * (cw + 6) + cw / 2;
    var sy = ry + 12 + k * 22;
    D.path(svg, 'M ' + sx + ' ' + sy + ' L ' + sx + ' ' + (ry + ch + 24) + ' C ' + sx + ' ' + (by - 30) + ', ' + (gx + gw / 2) + ' ' + (ry + ch + 40) + ', ' + (gx + gw / 2) + ' ' + (by - 4), mkOx);
  }
  D.text(svg, 16, by + 54, 'one batch · B = L × W windows · every replay appears once', {class: 'diagram-note'});
  D.text(svg, 16, by + 70, 'window k of a replay is used about k·P/W batches after the replay entered, jittered within a phase block', {class: 'diagram-note'});
  D.text(svg, 16, by + 86, 'two windows of one replay are never closer than P − block + 1 batches', {class: 'diagram-note'});

  /* ---- 4 · what a checkpoint keeps ---- */
  var ky = 470;
  svg.appendChild(D.el('line', {x1: 16, y1: ky - 14, x2: 864, y2: ky - 14, stroke: 'var(--rule-dotted)', 'stroke-dasharray': '2 3'}));
  D.text(svg, 16, ky + 6, 'checkpoint', {class: 'diagram-label'});
  D.text(svg, 110, ky + 6, 'cursor (epoch · shard · row) + one locator per ring slot · no decoded arrays', {class: 'diagram-note'});
}());
