/* Fig: the three controller codecs — button classes, trigger targets and main-stick targets. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, txt) {
    var n = document.createElementNS(NS, name);
    Object.keys(attrs).forEach(function (k) { n.setAttribute(k, String(attrs[k])); });
    if (txt !== undefined) n.textContent = txt;
    return n;
  }
  var many = function (sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); };

  // main-stick class centers, as drawn by the retired anatomy page
  var legacyStickPoints = [
    [0, 0], [.35, 0], [-.35, 0], [0, .35], [0, -.35], [.675, 0], [-.675, 0], [0, .675], [0, -.675],
    [1, 0], [0, 1], [-1, 0], [0, -1], [.95, -.3], [-.95, -.3], [.95, .3], [-.95, .3],
    [.85, -.5], [.85, .5], [-.85, -.5], [-.85, .5], [.7, -.7], [-.7, -.7], [.7, .7], [-.7, .7],
    [.5, .5], [-.5, .5], [.5, -.5], [-.5, -.5], [.5, .85], [-.5, .85], [.5, -.85], [-.5, -.85],
    [.3, -.95], [.3, .95], [-.3, -.95], [-.3, .95]
  ];
  function ringPoints(radius, count) {
    return Array.from({length: count}, function (_, i) {
      var a = i / count * Math.PI * 2;
      return [radius * Math.cos(a), radius * Math.sin(a)];
    });
  }
  var stickPoints = {
    legacy: legacyStickPoints,
    cartesian: [[0, 0], [.35, 0], [-.35, 0], [0, .35], [0, -.35], [.675, 0], [-.675, 0], [0, .675], [0, -.675]]
      .concat(ringPoints(.5, 8), ringPoints(.85, 8), ringPoints(1, 40)),
    polar: [[0, 0]].concat(ringPoints(23 / 80, 4), ringPoints(Math.sqrt(23 * 80) / 80, 16), ringPoints(1, 64))
  };
  var stickRings = {legacy: [], cartesian: [.5, .85], polar: [23 / 80, Math.sqrt(23 * 80) / 80]};
  many('[data-stick-codec]').forEach(function (svg) {
    var name = svg.getAttribute('data-stick-codec');
    svg.appendChild(el('line', {class: 'stick-axis', x1: 62, y1: 66, x2: 178, y2: 66}));
    svg.appendChild(el('line', {class: 'stick-axis', x1: 120, y1: 8, x2: 120, y2: 124}));
    svg.appendChild(el('circle', {class: 'stick-rim', cx: 120, cy: 66, r: 56}));
    stickRings[name].forEach(function (r) { svg.appendChild(el('circle', {class: 'stick-ring', cx: 120, cy: 66, r: (r * 56).toFixed(2)})); });
    var radius = name === 'polar' ? 1.55 : name === 'cartesian' ? 1.8 : 2.1;
    stickPoints[name].forEach(function (p) {
      svg.appendChild(el('circle', {class: 'stick-point', cx: (120 + p[0] * 56).toFixed(2), cy: (66 - p[1] * 56).toFixed(2), r: radius}));
    });
  });

  // trigger targets: legacy fuses L and R by max onto one line of five; the mask codecs keep L × R
  var TRIG = [0, 0.35, 0.6, 0.85, 1.0];
  var tickLabel = function (v) { return v === 0 ? '0' : v === 1 ? '1' : String(v).replace(/^0/, ''); };
  many('[data-trigger-codec]').forEach(function (svg) {
    var name = svg.getAttribute('data-trigger-codec');
    if (name === 'legacy') {
      var x0 = 70, x1 = 200, y = 44, sx = function (v) { return x0 + v * (x1 - x0); };
      svg.appendChild(el('text', {x: 14, y: 36}, 'L'));
      svg.appendChild(el('text', {x: 14, y: 60}, 'R'));
      svg.appendChild(el('path', {class: 'trig-fuse', d: 'M 24 32 C 38 32, 36 ' + y + ', 48 ' + y + ' M 24 56 C 38 56, 36 ' + y + ', 48 ' + y}));
      svg.appendChild(el('text', {x: 50, y: y + 3}, 'max'));
      svg.appendChild(el('line', {class: 'trig-axis', x1: x0, y1: y, x2: x1, y2: y}));
      TRIG.forEach(function (v) {
        svg.appendChild(el('circle', {class: 'trig-point', cx: sx(v).toFixed(1), cy: y, r: 2.4}));
        svg.appendChild(el('text', {x: sx(v).toFixed(1), y: y + 20, 'text-anchor': 'middle'}, tickLabel(v)));
      });
    } else {
      var gx = 96, gy = 12, span = 72, px = function (v) { return gx + v * span; }, py = function (v) { return gy + span - v * span; };
      TRIG.forEach(function (v) {
        svg.appendChild(el('line', {class: 'trig-grid', x1: px(v), y1: py(0), x2: px(v), y2: py(1)}));
        svg.appendChild(el('line', {class: 'trig-grid', x1: px(0), y1: py(v), x2: px(1), y2: py(v)}));
      });
      svg.appendChild(el('line', {class: 'trig-axis', x1: px(0), y1: py(0), x2: px(1), y2: py(0)}));
      svg.appendChild(el('line', {class: 'trig-axis', x1: px(0), y1: py(0), x2: px(0), y2: py(1)}));
      TRIG.forEach(function (l) { TRIG.forEach(function (r) {
        svg.appendChild(el('circle', {class: 'trig-point', cx: px(l).toFixed(1), cy: py(r).toFixed(1), r: 2.4}));
      }); });
      [0, 1].forEach(function (v) {
        svg.appendChild(el('text', {x: px(v), y: py(0) + 11, 'text-anchor': 'middle'}, tickLabel(v)));
        svg.appendChild(el('text', {x: px(0) - 5, y: py(v) + 3, 'text-anchor': 'end'}, tickLabel(v)));
      });
      svg.appendChild(el('text', {x: px(1) + 8, y: py(0) + 3}, 'L'));
      svg.appendChild(el('text', {x: px(0), y: py(1) - 5, 'text-anchor': 'middle'}, 'R'));
    }
  });
}());
