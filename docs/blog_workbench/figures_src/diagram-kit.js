/* Shared drawing primitives for the hand-written figures: the page's model-diagram grammar
   (joint tokens, stacked modules, decoder head, wires) plus "given → produced" rows. */
window.HALDIAG = (function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, String(attrs[k])); });
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function empty(target) { while (target.firstChild) target.removeChild(target.firstChild); }
  function text(target, x, y, value, attrs) {
    var a = Object.assign({}, attrs || {}); a.x = x; a.y = y;
    var t = el('text', a, value); target.appendChild(t); return t;
  }
  function marker(svg, suffix, color) {
    var id = (svg.id || 'd') + '-' + suffix;
    var defs = svg.querySelector('defs');
    if (!defs) { defs = el('defs'); svg.insertBefore(defs, svg.firstChild); }
    var m = el('marker', {id: id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse'});
    m.appendChild(el('path', {d: 'M 0 0 L 10 5 L 0 10 z', fill: color}));
    defs.appendChild(m);
    return 'url(#' + id + ')';
  }
  function part(target, moduleId) { var g = el('g', {'data-module': moduleId}); target.appendChild(g); return g; }
  function token(target, x, y, w, h, label, cls) {
    target.appendChild(el('rect', {x: x, y: y, width: w, height: h, class: 'model-token' + (cls ? ' ' + cls : '')}));
    text(target, x + w / 2, y + h / 2 + 4, label, {'text-anchor': 'middle', class: 'diagram-label' + (cls === 'given' ? ' given-label' : '')});
  }
  function jointToken(target, x, y, w, h, state, action, cls) {
    target.appendChild(el('rect', {x: x, y: y, width: w, height: h, class: 'model-token' + (cls ? ' ' + cls : '')}));
    text(target, x + w / 2, y + h / 2 + 4, '(' + state + ', ' + action + ')', {'text-anchor': 'middle', class: 'diagram-label' + (cls === 'given' ? ' given-label' : '')});
  }
  function stacked(target, x, y, w, h, title, note) {
    [8, 4].forEach(function (o) { target.appendChild(el('rect', {x: x - o, y: y + o, width: w, height: h, class: 'model-layer'})); });
    target.appendChild(el('rect', {x: x, y: y, width: w, height: h, class: 'model-box'}));
    text(target, x + w / 2, y + (note || h > 90 ? 29 : h / 2 + 4), title, {'text-anchor': 'middle', class: 'diagram-title'});
    if (note) text(target, x + w / 2, y + 50, note, {'text-anchor': 'middle', class: 'diagram-note'});
  }
  function box(target, x, y, w, h, title) {
    target.appendChild(el('rect', {x: x - 4, y: y + 4, width: w, height: h, class: 'model-layer'}));
    target.appendChild(el('rect', {x: x, y: y, width: w, height: h, class: 'model-box'}));
    text(target, x + w / 2, y + h / 2 + 4, title, {'text-anchor': 'middle', class: 'diagram-title'});
  }
  function wire(target, x1, y1, x2, y2, mk, opts) {
    var a = {x1: x1, y1: y1, x2: x2, y2: y2, class: 'model-wire' + (opts && opts.ghost ? ' ghost' : '') + (opts && opts.accent ? ' accent' : '')};
    if (mk) a['marker-end'] = mk;
    var l = el('line', a); target.appendChild(l); return l;
  }
  function path(target, d, mk, opts) {
    var a = {d: d, class: 'model-wire' + (opts && opts.ghost ? ' ghost' : '') + (opts && opts.accent ? ' accent' : '')};
    if (mk) a['marker-end'] = mk;
    var p = el('path', a); target.appendChild(p); return p;
  }
  function note(target, x, y, value, anchor, cls) {
    return text(target, x, y, value, {'text-anchor': anchor || 'start', class: 'diagram-note' + (cls ? ' ' + cls : '')});
  }
  /* A "given → produced" row: grey prefix tokens, an arrow, then the produced token(s). */
  function row(target, x, y, given, produced, opts) {
    var o = Object.assign({w: 44, h: 28, gap: 6, arrow: 40, mk: null, producedCls: '', joint: false}, opts || {});
    var cx = x;
    given.forEach(function (g) {
      if (o.joint && Array.isArray(g)) jointToken(target, cx, y, o.w, o.h, g[0], g[1], 'given');
      else token(target, cx, y, o.w, o.h, g, 'given');
      cx += o.w + o.gap;
    });
    if (!given.length) cx = x;
    var arrowStart = cx - o.gap + 6;
    wire(target, arrowStart, y + o.h / 2, arrowStart + o.arrow - 8, y + o.h / 2, o.mk);
    cx = arrowStart + o.arrow;
    var start = cx;
    produced.forEach(function (p) {
      token(target, cx, y, o.w, o.h, p, o.producedCls);
      cx += o.w + o.gap;
    });
    return {end: cx - o.gap, producedStart: start, arrowStart: arrowStart};
  }
  function focus(svg, moduleId) {
    Array.prototype.forEach.call(svg.querySelectorAll('[data-module]'), function (p) {
      p.classList.remove('module-dim', 'module-focus');
      if (!moduleId) return;
      p.classList.add(p.getAttribute('data-module') === moduleId ? 'module-focus' : 'module-dim');
    });
  }
  return {el: el, empty: empty, text: text, marker: marker, part: part, token: token, jointToken: jointToken, stacked: stacked, box: box, wire: wire, path: path, note: note, row: row, focus: focus};
}());
