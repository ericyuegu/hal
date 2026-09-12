/* Fig: run-length action spans — what BPE found in the packed controller stream. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var host = document.getElementById('rle-diagram');
  if (!host) return;
  function svgEl(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, String(attrs[k])); });
    if (text !== undefined) n.textContent = text;
    return n;
  }
  var s = svgEl('svg', {viewBox: '0 0 920 254', role: 'img', 'aria-label': 'A frame-aligned stream of eight controller symbols, A A A B B C C A, is segmented by BPE into four span tokens: A times three, B times two, C times two, and A. The vocabulary compressed the stream 13.8 times; the first 256 merges were all pure holds.'});
  var defs = svgEl('defs');
  var m = svgEl('marker', {id: 'rle-arrow', viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '5', markerHeight: '5', orient: 'auto', markerUnits: 'strokeWidth'});
  m.appendChild(svgEl('path', {d: 'M0 1 L10 5 L0 9 Z', fill: 'var(--ink-faint)'}));
  defs.appendChild(m);
  s.appendChild(defs);

  var stream = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'A'];
  var x0 = 24, fw = 56, gap = 6, y1 = 44;
  s.appendChild(svgEl('text', {x: x0, y: 22, 'class': 'lbl'}, 'Frame-aligned controller stream'));
  s.appendChild(svgEl('text', {x: x0 + 250, y: 22, 'class': 'sub'}, 'one symbol per frame · the four controller groups packed together'));
  stream.forEach(function (sym, i) {
    var x = x0 + i * (fw + gap);
    s.appendChild(svgEl('rect', {x: x, y: y1, width: fw, height: 36, 'class': 'frame ' + sym.toLowerCase()}));
    s.appendChild(svgEl('text', {x: x + fw / 2, y: y1 + 23, 'class': 'sym', 'text-anchor': 'middle'}, sym));
    s.appendChild(svgEl('text', {x: x + fw / 2, y: y1 + 50, 'class': 'tiny', 'text-anchor': 'middle'}, 't+' + (i + 1)));
  });
  var right = x0 + stream.length * (fw + gap) - gap;
  s.appendChild(svgEl('text', {x: right + 14, y: y1 + 23, 'class': 'sub'}, '8 frames · 8 decisions for a frame-level head'));

  // segmentation arrow
  s.appendChild(svgEl('path', {d: 'M ' + (x0 + 250) + ' 104 L ' + (x0 + 250) + ' 126', fill: 'none', stroke: 'var(--ink-faint)', 'stroke-width': 1.1, 'marker-end': 'url(#rle-arrow)'}));
  s.appendChild(svgEl('text', {x: x0 + 262, y: 120, 'class': 'tiny'}, 'greedy BPE segmentation'));

  var y2 = 136;
  s.appendChild(svgEl('text', {x: x0, y: y2 - 10, 'class': 'lbl'}, 'Learned span tokens'));
  var spans = [['A × 3', 3], ['B × 2', 2], ['C × 2', 2], ['A', 1]];
  var cursor = x0;
  spans.forEach(function (sp) {
    var w = sp[1] * (fw + gap) - gap;
    s.appendChild(svgEl('rect', {x: cursor, y: y2, width: w, height: 36, 'class': 'span'}));
    s.appendChild(svgEl('text', {x: cursor + w / 2, y: y2 + 23, 'class': 'sym', 'text-anchor': 'middle'}, sp[0]));
    cursor += w + gap;
  });
  s.appendChild(svgEl('text', {x: right + 14, y: y2 + 23, 'class': 'sub'}, '4 tokens · a token could span up to 16 frames'));
  s.appendChild(svgEl('text', {x: x0, y: y2 + 52, 'class': 'tiny'}, 'a sampled token was executed to completion before the next plan — that cadence change is part of the failed package'));

  // stats block
  var sx = 620, sy = 206;
  s.appendChild(svgEl('line', {x1: x0, y1: sy - 4, x2: 896, y2: sy - 4, stroke: 'var(--rule-dotted)', 'stroke-dasharray': '2 3'}));
  [['13.8×', 'compression at vocab 4,096'], ['100%', 'of the first 256 merges were pure holds'], ['−2.64 / −172', 'stocks · damage per min vs. the frame-aligned reference']].forEach(function (st, i) {
    var x = x0 + i * 300;
    s.appendChild(svgEl('text', {x: x, y: sy + 22, 'class': 'stat'}, st[0]));
    s.appendChild(svgEl('text', {x: x, y: sy + 38, 'class': 'tiny'}, st[1]));
  });
  host.appendChild(s);
}());
