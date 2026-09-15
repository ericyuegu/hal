/* Fig: O55 — dense (every prefix) versus final-prefix supervision of the decoder, then the history
   cross-attention variants inside the sparse setup. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('supervision-density-diagram');
  var S = D && D.study('o55_history_cross_attention');
  if (!D || !svg || !S) return;
  var cmp = {};
  S.comparisons.forEach(function (c) { cmp[c.treatment + '<' + c.control] = {delta: c.stock_delta, lo: c.delta_ci95[0], hi: c.delta_ci95[1]}; });
  var ROWS = [
    {label: 'dense → sparse (A)', sub: 'O52 → A · 128 trained positions → 1 · same windows and updates', d: cmp['A<O52 dense baseline'], accent: true},
    {label: 'A → B · + history cross-attention', sub: 'one decoder layer reads the trunk history', d: cmp['B<A']},
    {label: 'A → C · add history attention, drop the hₜ input', sub: 'the history reaches the decoder only through cross-attention', d: cmp['C<A']},
    {label: 'A → D · replace action attention, drop the hₜ input', sub: 'every decoder attention layer reads the history · all arms keep the trunk skip', d: cmp['D<A']}
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.supervisionDensity = function () {
    return S.comparisons.map(function (c) { return {comparison: c.control + ' → ' + c.treatment, stock_delta: c.stock_delta, ci95_lo: c.delta_ci95[0], ci95_hi: c.delta_ci95[1], method: 'paired bootstrap, 96 boots'}; });
  };

  var W = 880, H = 452;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var mk = D.marker(svg, 'arrow', 'var(--ink-faint)');

  /* ---- panel 1: one window, where the decoder is trained ---- */
  var N = 16, tw = 34, gap = 4, x0 = 250, half = N / 2;
  function strip(y, name, note, supervised) {
    D.text(svg, 16, y + 18, name, {class: 'diagram-title' + (supervised.length === 1 ? '' : ' accent')});
    D.text(svg, 16, y + 34, note, {class: 'diagram-note'});
    for (var i = 0; i < N; i++) {
      var x = x0 + i * (tw + gap);
      D.token(svg, x, y, tw, 26, i === N - 1 ? 'sₜ' : i === 0 ? 's₁' : '', 'given');
      if (supervised.indexOf(i) >= 0) {
        svg.appendChild(D.el('rect', {x: x, y: y - 22, width: tw, height: 16, class: 'model-token accent'}));
        D.text(svg, x + tw / 2, y - 10, 'loss', {class: 'diagram-note accent-text', 'text-anchor': 'middle'});
      }
    }
    D.text(svg, x0 + half * (tw + gap) - gap / 2, y + 40, '256 context frames', {class: 'diagram-note', 'text-anchor': 'middle'});
  }
  var dense = []; for (var i = half; i < N; i++) dense.push(i);
  D.text(svg, x0, 20, 'one training window · a decoder chunk is trained wherever “loss” sits', {class: 'diagram-note'});
  strip(58, 'dense · O52', 'the last 128 positions each train a chunk', dense);
  strip(150, 'sparse · O55 A–D', 'only the final position trains a chunk', [N - 1]);
  svg.appendChild(D.el('line', {x1: x0, y1: 84 + 40, x2: x0 + N * (tw + gap) - gap, y2: 84 + 40, class: 'model-wire ghost'}));

  /* ---- panel 2: the comparisons, net stocks only (O55 reported stocks) ---- */
  var top = 262, rowH = 44;
  svg.appendChild(D.el('line', {x1: 16, y1: top - 42, x2: W - 16, y2: top - 42, stroke: 'var(--rule-dotted)', 'stroke-dasharray': '2 3'}));
  var dom = D.domain(ROWS.map(function (r) { return r.d; }));
  var x = D.scale([560, 780], dom.lo, dom.hi);
  D.text(svg, 16, top - 20, 'baseline → treatment · 95% interval', {class: 'diagram-note'});
  D.effectAxis(svg, x, dom, top - 6, top + ROWS.length * rowH - 8, 'Δ net stocks / min');
  ROWS.forEach(function (r, i) {
    var y = top + i * rowH + rowH / 2 - 6;
    D.text(svg, 16, y, r.label, {class: 'diagram-title' + (r.accent ? ' accent' : '')});
    D.text(svg, 16, y + 14, r.sub, {class: 'diagram-note'});
    D.effect(svg, x, y, r.d, {accent: r.accent, digits: 2, valueX: 792});
  });
}());
