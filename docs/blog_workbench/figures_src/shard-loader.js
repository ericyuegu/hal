/* Fig: the physical-shard loader — one disk read per replay, W windows, spread over batches ≈ P apart. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('loader-diagram');
  if (!D || !svg) return;
  var FAINT = 'var(--ink-faint)';
  var mk = D.marker(svg, 'arrow', FAINT);
  var W = 4;          // windows per replay drawn (O52 uses 8)
  svg.setAttribute('viewBox', '0 0 880 400');

  /* ---- 1 · shards in one keyed order → one decode → W windows ---- */
  D.text(svg, 16, 22, 'shards · one fixed order · each read once', {class: 'diagram-note'});
  for (var i = 0; i < 5; i++) {
    var x = 16 + i * 46;
    svg.appendChild(D.el('rect', {x: x, y: 32, width: 40, height: 28, class: 'model-token' + (i === 0 ? ' accent' : ' given')}));
    D.text(svg, x + 20, 50, 'shard', {'text-anchor': 'middle', class: 'diagram-label' + (i ? ' given-label' : '')});
  }
  D.text(svg, 16 + 5 * 46 + 2, 50, '···', {class: 'diagram-note'});
  D.wire(svg, 278, 46, 322, 46, mk);
  D.text(svg, 300, 38, 'decode', {class: 'diagram-note', 'text-anchor': 'middle'});
  svg.appendChild(D.el('rect', {x: 330, y: 32, width: 330, height: 28, class: 'model-token'}));
  D.text(svg, 330, 22, 'one replay · read once', {class: 'diagram-note'});
  var cuts = [0.04, 0.3, 0.56, 0.8];
  cuts.forEach(function (f, k) {
    var x = 330 + f * 330;
    svg.appendChild(D.el('rect', {x: x, y: 34, width: 50, height: 24, class: 'loader-win r'}));
    D.text(svg, x + 25, 50, 'w' + (k + 1), {'text-anchor': 'middle', class: 'diagram-label accent-text'});
  });
  D.text(svg, 672, 50, 'W windows · context + chunk', {class: 'diagram-note'});

  /* ---- 2 · the batches that follow, in order ---- */
  var by = 130, bh = 150, bw = 38, slots = 6, gap = 6, x0 = 16;
  D.text(svg, x0, by - 30, 'batches, in training order →', {class: 'diagram-note'});
  // clusters of three batches, ≈ P batches apart; window k of the replay lands in cluster k
  var clusters = [['b', 'b+1', 'b+2'], ['b+P−1', 'b+P', 'b+P+1'], ['b+2P−1', 'b+2P', 'b+2P+1'], ['b+3P−1', 'b+3P', 'b+3P+1']];
  var placeR = [[0, 1], [1, 2], [0, 3], [2, 1]];   // [column in cluster, slot] for the followed replay's window k
  var placeQ = [[1, 4], [2, 0], [1, 5], [0, 2]];   // an older replay, entered ≈ P/2 batches earlier
  var cw = 3 * (bw + gap) - gap, cgap = 44;
  clusters.forEach(function (cl, k) {
    var cx = x0 + k * (cw + cgap);
    cl.forEach(function (label, j) {
      var bx = cx + j * (bw + gap);
      svg.appendChild(D.el('rect', {x: bx, y: by, width: bw, height: bh, class: 'model-token'}));
      for (var s = 0; s < slots; s++) {
        var sy = by + 8 + s * ((bh - 16) / slots);
        var cls = 'loader-win';
        if (placeR[k][0] === j && placeR[k][1] === s) cls += ' r';
        else if (placeQ[k][0] === j && placeQ[k][1] === s) cls += ' q';
        svg.appendChild(D.el('rect', {x: bx + 6, y: sy, width: bw - 12, height: (bh - 16) / slots - 5, class: cls}));
      }
      D.text(svg, bx + bw / 2, by + bh + 14, label, {'text-anchor': 'middle', class: 'diagram-note'});
    });
    if (k < clusters.length - 1) D.text(svg, cx + cw + cgap / 2, by + bh / 2 + 4, '···', {'text-anchor': 'middle', class: 'diagram-note'});
  });
  // the followed replay's windows: leader lines from the replay bar down to their batches
  cuts.forEach(function (f, k) {
    var sx = 330 + f * 330 + 25, cx = x0 + k * (cw + cgap) + placeR[k][0] * (bw + gap) + bw / 2;
    var sy = by + 8 + placeR[k][1] * ((bh - 16) / slots) + 2;
    D.path(svg, 'M ' + sx + ' 60 L ' + sx + ' 76 L ' + cx + ' 76 L ' + cx + ' ' + (sy - 2), null, {accent: true, ghost: true});
  });
  // brackets: ≈ P batches between consecutive windows
  var y2 = by + bh + 34;
  [0, 1, 2].forEach(function (k) {
    var a = x0 + k * (cw + cgap) + placeR[k][0] * (bw + gap) + bw / 2, b = x0 + (k + 1) * (cw + cgap) + placeR[k + 1][0] * (bw + gap) + bw / 2;
    svg.appendChild(D.el('path', {d: 'M ' + a + ' ' + (y2 - 6) + ' L ' + a + ' ' + y2 + ' L ' + b + ' ' + y2 + ' L ' + b + ' ' + (y2 - 6), fill: 'none', stroke: FAINT}));
    D.text(svg, (a + b) / 2, y2 + 13, '≈ P batches', {'text-anchor': 'middle', class: 'diagram-note'});
  });
  // legend
  var ly = 366;
  svg.appendChild(D.el('rect', {x: x0, y: ly - 9, width: 14, height: 10, class: 'loader-win r'}));
  D.text(svg, x0 + 20, ly, 'the replay above · one window per batch', {class: 'diagram-note'});
  svg.appendChild(D.el('rect', {x: x0 + 330, y: ly - 9, width: 14, height: 10, class: 'loader-win q'}));
  D.text(svg, x0 + 350, ly, 'an older replay, same schedule', {class: 'diagram-note'});
  svg.appendChild(D.el('rect', {x: x0 + 600, y: ly - 9, width: 14, height: 10, class: 'loader-win'}));
  D.text(svg, x0 + 620, ly, 'other replays', {class: 'diagram-note'});
  D.text(svg, x0, ly + 18, 'every window in a batch comes from a different replay', {class: 'diagram-note'});
}());
