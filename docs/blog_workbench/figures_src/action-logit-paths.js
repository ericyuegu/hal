/* Fig: three paths into the action logits, and what removing each one cost. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var pathHost = document.getElementById('logit-path');
  var forestHost = document.getElementById('logit-forest');
  if (!pathHost || !forestHost) return;
  function svgEl(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, String(attrs[k])); });
    if (text !== undefined) n.textContent = text;
    return n;
  }

  var ROWS = [
    {id: 'film', change: 'Zero-initialize FiLM', at: 'U4,096', run: 'vzd242t6', char: [-1.09, 0.34], fox: [-1.20, 0.37], comp: 'film'},
    {id: 'skip', change: 'Remove the trunk skip', at: 'U4,096', run: 'j4skyy83', char: [-0.89, 0.40], fox: [-1.13, 0.46], comp: 'skip'},
    {id: 'head', change: 'Linear heads instead of SiLU MLP', at: 'U8,192', run: 'apj3rw8b', char: [-2.03, 0.28], fox: [-2.41, 0.32], comp: 'head'}
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.logitPathRows = function () {
    return ROWS.map(function (r) { return {change: r.change, updates: r.at, run: r.run, control: '1imfy8v3', character_suite_delta: r.char[0], character_suite_ci95: r.char[1], fox_suite_delta: r.fox[0], fox_suite_ci95: r.fox[1]}; });
  };

  /* ---- the path diagram ---- */
  var s = svgEl('svg', {viewBox: '0 0 410 250', role: 'img', 'aria-label': 'The trunk state feeds the temporal decoder, whose future-frame state is conditioned by FiLM on earlier controller parts and read by a SiLU MLP head; a direct skip from the trunk state also adds into the action logits.'});
  var defs = svgEl('defs');
  var m = svgEl('marker', {id: 'lp-arrow', viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '5', markerHeight: '5', orient: 'auto', markerUnits: 'strokeWidth'});
  m.appendChild(svgEl('path', {d: 'M0 1 L10 5 L0 9 Z', fill: 'var(--ink-faint)'}));
  defs.appendChild(m);
  s.appendChild(defs);
  function comp(key, x, y, w, h, title, note) {
    var g = svgEl('g', {'class': 'comp', 'data-comp': key});
    g.appendChild(svgEl('rect', {x: x, y: y, width: w, height: h, 'class': 'plate'}));
    g.appendChild(svgEl('text', {x: x + w / 2, y: y + 15, 'class': 'sym', 'text-anchor': 'middle'}, title));
    if (note) g.appendChild(svgEl('text', {x: x + w / 2, y: y + 28, 'class': 'tiny', 'text-anchor': 'middle'}, note));
    s.appendChild(g);
    return g;
  }
  function wire(d, key) {
    var p = svgEl('path', {d: d, 'class': 'wire', 'marker-end': 'url(#lp-arrow)'});
    if (key) { var g = svgEl('g', {'class': 'comp', 'data-comp': key}); g.appendChild(p); s.appendChild(g); return g; }
    s.appendChild(p);
    return p;
  }
  s.appendChild(svgEl('text', {x: 8, y: 16, 'class': 'lbl'}, 'Complete O43 control · arm A'));
  comp('trunk', 8, 60, 92, 36, 'trunk state', 'hₜ');
  wire('M 100 78 L 128 78');
  comp('dec', 130, 60, 120, 36, 'temporal decoder', 'future-frame state');
  wire('M 250 78 L 292 78');
  // FiLM is a modifier on the head's input, not a stage of its own
  var film = svgEl('g', {'class': 'comp', 'data-comp': 'film'});
  film.appendChild(svgEl('path', {d: 'M 271 128 L 271 84', 'class': 'wire', 'marker-end': 'url(#lp-arrow)'}));
  film.appendChild(svgEl('text', {x: 271, y: 142, 'class': 'tiny', 'text-anchor': 'middle'}, 'FiLM · earlier controller parts'));
  s.appendChild(film);
  comp('head', 294, 60, 112, 36, 'SiLU MLP head');
  wire('M 350 96 L 350 180');
  // the direct skip
  wire('M 54 96 L 54 194 L 336 194', 'skip');
  var skipLabel = svgEl('g', {'class': 'comp', 'data-comp': 'skip'});
  skipLabel.appendChild(svgEl('text', {x: 64, y: 188, 'class': 'tiny'}, 'direct trunk skip · hₜ → linear logits'));
  s.appendChild(skipLabel);
  s.appendChild(svgEl('circle', {cx: 350, cy: 194, r: 12, 'class': 'plate'}));
  s.appendChild(svgEl('text', {x: 350, y: 198.5, 'class': 'sym', 'text-anchor': 'middle'}, 'Σ'));
  wire('M 362 194 L 382 194');
  s.appendChild(svgEl('rect', {x: 384, y: 182, width: 16, height: 24, 'class': 'plate', 'stroke-dasharray': '2 2'}));
  s.appendChild(svgEl('text', {x: 8, y: 236, 'class': 'tiny'}, 'hover a row to see which path the test removed'));
  pathHost.appendChild(s);

  /* ---- the forest plot ---- */
  var W = 460, H = 276, L = 12, R = 24, top = 44, rowH = 58;
  var xmin = -3.2, xmax = 0.3;
  function X(v) { return L + 150 + (v - xmin) / (xmax - xmin) * (W - L - R - 150); }
  var f = svgEl('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Forest plot of three one-change ablations: each lost between 0.9 and 2.4 stocks per minute on both evaluation suites, with intervals that exclude zero.'});
  f.appendChild(svgEl('text', {x: L, y: 16, 'class': 'lbl'}, 'Change in net stocks / min'));
  f.appendChild(svgEl('text', {x: L, y: 30, 'class': 'sub'}, 'treatment minus control · point and 95% interval'));
  [-3, -2.5, -2, -1.5, -1, -0.5].forEach(function (v) {
    f.appendChild(svgEl('line', {x1: X(v), y1: top, x2: X(v), y2: top + rowH * 3, 'class': 'grid'}));
    f.appendChild(svgEl('text', {x: X(v), y: top + rowH * 3 + 14, 'class': 'tiny', 'text-anchor': 'middle'}, String(v)));
  });
  f.appendChild(svgEl('line', {x1: X(0), y1: top - 6, x2: X(0), y2: top + rowH * 3, 'class': 'zero'}));
  f.appendChild(svgEl('text', {x: X(0), y: top + rowH * 3 + 14, 'class': 'tiny', 'text-anchor': 'middle'}, '0'));
  f.appendChild(svgEl('text', {x: X(0), y: top + rowH * 3 + 25, 'class': 'tiny', 'text-anchor': 'middle'}, 'control'));
  ROWS.forEach(function (r, i) {
    var y0 = top + i * rowH;
    var g = svgEl('g', {'class': 'row', 'data-comp': r.comp});
    g.appendChild(svgEl('rect', {x: 0, y: y0, width: W, height: rowH, 'class': 'hit'}));
    g.appendChild(svgEl('line', {x1: L, y1: y0 + rowH, x2: W - R, y2: y0 + rowH, 'class': 'grid'}));
    g.appendChild(svgEl('text', {x: L, y: y0 + 18, 'class': 'sym', 'font-size': '10'}, r.change));
    g.appendChild(svgEl('text', {x: L, y: y0 + 31, 'class': 'tiny'}, r.at + ' · ' + r.run));
    g.appendChild(svgEl('text', {x: L, y: y0 + 46, 'class': 'val', 'font-size': '9'}, r.char[0].toFixed(2) + ' ± ' + r.char[1].toFixed(2) + '  ·  ' + r.fox[0].toFixed(2) + ' ± ' + r.fox[1].toFixed(2)));
    [['char', r.char, 22, 'character suite'], ['fox', r.fox, 40, 'Fox suite']].forEach(function (series) {
      var v = series[1], y = y0 + series[2];
      g.appendChild(svgEl('line', {x1: X(v[0] - v[1]), y1: y, x2: X(v[0] + v[1]), y2: y, 'class': 'ci ' + series[0]}));
      g.appendChild(svgEl('circle', {cx: X(v[0]), cy: y, r: 3.4, 'class': 'pt ' + series[0]}));
    });
    g.addEventListener('mouseenter', function () { pathHost.setAttribute('data-hl', r.comp); s.querySelectorAll('.comp').forEach(function (c) { c.classList.toggle('is-hl', c.dataset.comp === r.comp); }); });
    g.addEventListener('mouseleave', function () { pathHost.removeAttribute('data-hl'); s.querySelectorAll('.comp').forEach(function (c) { c.classList.remove('is-hl'); }); });
    f.appendChild(g);
  });
  var ly = top + rowH * 3 + 44;
  f.appendChild(svgEl('circle', {cx: L + 4, cy: ly, r: 3.4, 'class': 'pt char'}));
  f.appendChild(svgEl('text', {x: L + 12, y: ly + 3.5, 'class': 'tiny'}, 'character suite'));
  f.appendChild(svgEl('circle', {cx: L + 124, cy: ly, r: 3.4, 'class': 'pt fox'}));
  f.appendChild(svgEl('text', {x: L + 132, y: ly + 3.5, 'class': 'tiny'}, 'Fox suite'));
  f.appendChild(svgEl('text', {x: W - R, y: ly + 3.5, 'class': 'tiny', 'text-anchor': 'end'}, 'left is worse · every interval excludes zero'));
  forestHost.appendChild(f);
}());
