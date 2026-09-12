/* Fig: where a predicted action may look — three decoder generations as dependency wires. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var host = document.getElementById('deps-diagram');
  var axes = document.getElementById('deps-axes');
  if (!host || !axes) return;

  function svgEl(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, String(attrs[k])); });
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function sub(x, y, main, s, cls) {
    var t = svgEl('text', {x: x, y: y, 'text-anchor': 'middle', 'class': cls || 'sym'});
    t.appendChild(svgEl('tspan', {}, main));
    t.appendChild(svgEl('tspan', {'baseline-shift': 'sub', 'font-size': '75%'}, s));
    return t;
  }
  var svg = svgEl('svg', {viewBox: '0 0 920 312', role: 'img', 'aria-label': 'Three panels. Next-frame prediction: the trunk state for the current frame produces one action. Independent multi-token prediction: the same state fans out to four actions with no wires between them. Autoregressive multi-token prediction: each action also reads every earlier action, and inside a frame each controller group reads the groups already chosen. In every panel the trunk state reads earlier frames.'});
  var defs = svgEl('defs');
  [['faint', 'var(--ink-faint)'], ['ox', 'var(--oxblood)']].forEach(function (pair) {
    var m = svgEl('marker', {id: 'deps-arrow-' + pair[0], viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '5', markerHeight: '5', orient: 'auto', markerUnits: 'strokeWidth'});
    m.appendChild(svgEl('path', {d: 'M0 1 L10 5 L0 9 Z', fill: pair[1]}));
    defs.appendChild(m);
  });
  svg.appendChild(defs);

  function wire(d, family) {
    return svgEl('path', {d: d, 'class': 'w ' + family, 'marker-end': 'url(#deps-arrow-' + (family === 'chunk' || family === 'frame' ? 'ox' : 'faint') + ')'});
  }
  function arcUp(x1, x2, y, lift, family) {
    return wire('M ' + x1 + ' ' + y + ' Q ' + ((x1 + x2) / 2) + ' ' + (y - lift) + ' ' + x2 + ' ' + y, family);
  }

  var PANELS = [
    {x: 0, title: 'Next-frame prediction', note: 'O11 · one action per trunk state', actions: 1, chunk: false, frame: false},
    {x: 306, title: 'Independent multi-token', note: 'O12 · four actions, no wires between them', actions: 4, chunk: false, frame: false},
    {x: 612, title: 'Autoregressive multi-token', note: 'O37 D3 → O50 · the current decoder', actions: 4, chunk: true, frame: true, cur: true}
  ];
  var W = 306;
  var HY = 232;      // trunk state row
  var AY = 128;      // action row
  var TX = [42, 100, 158]; // trunk states t−2, t−1, t (panel-local)

  PANELS.forEach(function (p, idx) {
    var g = svgEl('g', {transform: 'translate(' + p.x + ' 0)'});
    if (idx) svg.appendChild(svgEl('line', {x1: p.x - 2, y1: 30, x2: p.x - 2, y2: 268, 'class': 'divider', stroke: 'var(--rule-dotted)'}));
    g.appendChild(svgEl('text', {x: 16, y: 24, 'class': 'title' + (p.cur ? ' cur' : '')}, p.title));
    g.appendChild(svgEl('text', {x: 16, y: 38, 'class': 'sub'}, p.note));

    // trunk states with across-time wires (the family every generation shares)
    TX.forEach(function (x, i) {
      var isNow = i === TX.length - 1;
      g.appendChild(svgEl('rect', {x: x - 20, y: HY - 12, width: 40, height: 24, 'class': 'h', 'stroke-opacity': isNow ? 1 : 0.55}));
      g.appendChild(sub(x, HY + 4, 'h', isNow ? 't' : 't−' + (TX.length - 1 - i), 'sym'));
      for (var j = i + 1; j < TX.length; j++) g.appendChild(arcUp(x, TX[j] - 4, HY + 12, -(14 + 8 * (j - i)), 'time'));
    });
    if (idx === 0) g.appendChild(svgEl('text', {x: 100, y: 282, 'class': 'tiny', 'text-anchor': 'middle'}, 'dotted: trunk states read earlier frames'));

    // actions
    var ax = [];
    var span = p.actions === 1 ? 0 : 186;
    for (var k = 0; k < p.actions; k++) ax.push(p.actions === 1 ? 158 : 62 + (span / (p.actions - 1)) * k);
    ax.forEach(function (x, k) {
      var box = svgEl('rect', {x: x - 22, y: AY - 14, width: 44, height: 28, 'class': 'a' + (p.cur ? ' cur' : '')});
      g.appendChild(box);
      g.appendChild(sub(x, AY + 4, 'a', 't+' + (k + 1)));
      // trunk → action
      g.appendChild(wire('M ' + TX[2] + ' ' + (HY - 12) + ' C ' + TX[2] + ' ' + (HY - 60) + ', ' + x + ' ' + (AY + 60) + ', ' + x + ' ' + (AY + 15), 'trunk'));
    });
    if (p.chunk) {
      // every earlier action → this one
      for (var i = 0; i < ax.length; i++) for (var j = i + 1; j < ax.length; j++) {
        g.appendChild(arcUp(ax[i] + 4 * (j - i - 1), ax[j] - 4 * (j - i - 1) - 2, AY - 14, 14 + 10 * (j - i), 'chunk'));
      }
      g.appendChild(svgEl('text', {x: 155, y: 62, 'class': 'tiny', 'text-anchor': 'middle'}, 'each action reads every action before it'));
    } else if (p.actions > 1) {
      g.appendChild(svgEl('text', {x: 155, y: 100, 'class': 'tiny', 'text-anchor': 'middle'}, 'actions cannot see one another'));
    } else {
      g.appendChild(svgEl('text', {x: 158, y: 100, 'class': 'tiny', 'text-anchor': 'middle'}, 'one frame, then re-observe'));
    }
    if (p.frame) {
      // open one action to show the within-frame chain
      var fx = 196, fy = 168;
      g.appendChild(svgEl('path', {d: 'M ' + (ax[1] + 22) + ' ' + (AY + 14) + ' L ' + fx + ' ' + fy, stroke: 'var(--rule-dotted)', 'stroke-dasharray': '2 3', fill: 'none'}));
      g.appendChild(svgEl('rect', {x: fx, y: fy, width: 104, height: 44, fill: 'var(--paper)', stroke: 'var(--rule-dotted)'}));
      g.appendChild(svgEl('text', {x: fx + 52, y: fy + 12, 'class': 'tiny', 'text-anchor': 'middle'}, 'inside one frame'));
      ['C', 'M', 'T', 'B'].forEach(function (name, i) {
        var cx = fx + 12 + i * 24;
        g.appendChild(svgEl('rect', {x: cx - 8, y: fy + 20, width: 16, height: 16, 'class': 'g'}));
        g.appendChild(svgEl('text', {x: cx, y: fy + 31.5, 'class': 'tiny', 'text-anchor': 'middle', 'font-size': '8'}, name));
        if (i) g.appendChild(wire('M ' + (cx - 16) + ' ' + (fy + 28) + ' L ' + (cx - 9) + ' ' + (fy + 28), 'frame'));
      });
      // wires from earlier groups skipping: C→T, C→B, M→B as small arcs above
      for (var a = 0; a < 4; a++) for (var b = a + 2; b < 4; b++) {
        g.appendChild(arcUp(fx + 12 + a * 24 + 2, fx + 12 + b * 24 - 6, fy + 20, 6 + 3 * (b - a), 'frame'));
      }
    }
    svg.appendChild(g);
  });

  // legend
  var lg = svgEl('g', {transform: 'translate(322 296)'});
  var items = [['time', 'across time · trunk'], ['chunk', 'across the chunk · decoder'], ['frame', 'within a frame · group chain']];
  items.forEach(function (it, i) {
    var x = i * 200;
    lg.appendChild(svgEl('path', {d: 'M ' + x + ' 6 L ' + (x + 16) + ' 6', 'class': 'w ' + it[0]}));
    lg.appendChild(svgEl('text', {x: x + 20, y: 9.5, 'class': 'legend'}, it[1]));
  });
  svg.appendChild(lg);
  host.appendChild(svg);

  axes.querySelectorAll('button').forEach(function (b) {
    b.addEventListener('click', function () {
      axes.querySelectorAll('button').forEach(function (o) { o.setAttribute('aria-pressed', String(o === b)); });
      if (b.dataset.axis === 'all') host.removeAttribute('data-axis'); else host.setAttribute('data-axis', b.dataset.axis);
    });
  });
}());
