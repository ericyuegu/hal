/* Fig: the ranked corpus before and after the compact policy layout (docs/experiments/data_pipeline.md). */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('storage-bars-diagram');
  if (!D || !svg) return;
  var ROWS = [
    {label: 'decoded training arrays', sub: 'what the loader reads', before: 802.47, after: 76.19},
    {label: 'compressed shards on disk', sub: 'Zstd, as stored', before: 29.82, after: 13.34}
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.storageBars = function () {
    return ROWS.map(function (r) { return {quantity: r.label, corpus: 'ranked-anonymized-1 train · 112,409 replays', mds_v7_gb: r.before, compact_policy_v7_gb: r.after, reduction: +(r.before / r.after).toFixed(2)}; });
  };
  var W = 880, H = 200, x0 = 250, x1 = 720;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var x = D.scale([x0, x1], 0, 820);
  function bar(y, v, cls, name) {
    svg.appendChild(D.el('rect', {x: x0, y: y, width: Math.max(2, x(v) - x0), height: 16, class: 'loss-bar' + cls}));
    D.text(svg, x(v) + 8, y + 12, v.toFixed(0) + ' GB · ' + name, {class: 'diagram-label'});
  }
  D.text(svg, x0, 22, 'gigabytes · one scale for all four bars', {class: 'diagram-note'});
  ROWS.forEach(function (r, i) {
    var y = 44 + i * 74;
    D.text(svg, 16, y + 12, r.label, {class: 'diagram-title'});
    D.text(svg, 16, y + 28, r.sub, {class: 'diagram-note'});
    bar(y, r.before, '', 'before');
    bar(y + 22, r.after, ' accent', 'after · ' + (r.before / r.after).toFixed(1) + '× smaller');
  });
  [0, 200, 400, 600, 800].forEach(function (v) {
    svg.appendChild(D.el('line', {x1: x(v), y1: 34, x2: x(v), y2: 176, class: v ? 'effect-grid' : 'effect-zero'}));
    D.text(svg, x(v), H - 8, String(v), {class: 'diagram-note', 'text-anchor': 'middle'});
  });
}());
