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

  /* ---- baseline → treatment: Δ = after − before with its 95% interval ---- */
  function signed(v, digits) { return (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(digits); }
  function scale(range, lo, hi) { return function (v) { return range[0] + (v - lo) / (hi - lo) * (range[1] - range[0]); }; }
  function niceStep(span) {
    var raw = span / 4, mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), r = raw / mag;
    return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
  }
  /* Axis domain covering every interval in `series` (list of {lo, hi}) plus zero. */
  function domain(series) {
    var lo = 0, hi = 0;
    series.forEach(function (d) { lo = Math.min(lo, d.lo); hi = Math.max(hi, d.hi); });
    var step = niceStep(hi - lo || 1), pad = step * 0.35;
    lo = Math.floor((lo - pad) / step) * step; hi = Math.ceil((hi + pad) / step) * step;
    var ticks = []; for (var v = lo; v <= hi + step / 2; v += step) ticks.push(Math.abs(v) < step / 1e6 ? 0 : v);
    return {lo: lo, hi: hi, ticks: ticks, digits: step >= 1 ? 0 : step >= 0.1 ? 1 : 2};
  }
  /* One effect: whisker over [lo, hi], dot at delta, value text at a fixed column. */
  function effect(target, x, y, d, opts) {
    var o = Object.assign({accent: false, digits: 2, valueX: null}, opts || {});
    var cls = o.accent ? ' accent' : '';
    target.appendChild(el('line', {x1: x(d.lo), y1: y, x2: x(d.hi), y2: y, class: 'effect-ci' + cls}));
    target.appendChild(el('circle', {cx: x(d.delta), cy: y, r: 3.6, class: 'effect-pt' + cls}));
    if (o.valueX !== null) text(target, o.valueX, y + 4, signed(d.delta, o.digits), {class: 'diagram-label effect-val' + cls});
  }
  /* Zero line, grid ticks and a column title for one effect column. */
  function effectAxis(target, x, dom, yTop, yBottom, title) {
    var px = Math.abs(x(dom.ticks[1]) - x(dom.ticks[0]));
    var every = px >= 40 ? 1 : px >= 20 ? 2 : 4;
    var zi = dom.ticks.indexOf(0);
    dom.ticks.forEach(function (v, i) {
      target.appendChild(el('line', {x1: x(v), y1: yTop, x2: x(v), y2: yBottom, class: v === 0 ? 'effect-zero' : 'effect-grid'}));
      var labelled = v === 0 || ((i - zi) % every === 0 && Math.abs(i - zi) >= every);
      if (labelled) text(target, x(v), yBottom + 14, signed(v, dom.digits), {class: 'diagram-note', 'text-anchor': 'middle'});
    });
    if (title) text(target, (x(dom.lo) + x(dom.hi)) / 2, yTop - 14, title, {class: 'diagram-note', 'text-anchor': 'middle'});
  }
  /* A whole forest: rows of {label, sub, stock, damage, accent, baseline}; stocks and damage side by side. */
  function forest(svg, rows, opts) {
    var o = Object.assign({width: 880, labelX: 16, cols: [[336, 540], [640, 826]], rowH: 44, top: 60, sub: true}, opts || {});
    var H = o.top + rows.length * o.rowH + 34;
    svg.setAttribute('viewBox', '0 0 ' + o.width + ' ' + H);
    var live = rows.filter(function (r) { return !r.baseline; });
    var doms = [domain(live.map(function (r) { return r.stock; })), domain(live.map(function (r) { return r.damage; }))];
    var xs = doms.map(function (d, k) { return scale(o.cols[k], d.lo, d.hi); });
    var yEnd = o.top + rows.length * o.rowH - 8;
    effectAxis(svg, xs[0], doms[0], o.top - 14, yEnd, 'Δ net stocks / min');
    effectAxis(svg, xs[1], doms[1], o.top - 14, yEnd, 'Δ net damage / min');
    text(svg, o.labelX, o.top - 14 - 14, o.header || 'baseline → treatment · 95% interval', {class: 'diagram-note'});
    rows.forEach(function (r, i) {
      var y = o.top + i * o.rowH + o.rowH / 2 - 6;
      text(svg, o.labelX, y + (r.sub ? 0 : 4), r.label, {class: 'diagram-title' + (r.accent ? ' accent' : '')});
      if (r.sub) text(svg, o.labelX, y + 14, r.sub, {class: 'diagram-note'});
      if (r.baseline) {
        xs.forEach(function (x, k) { text(svg, x(0) + 6, y + 4, 'baseline', {class: 'diagram-note'}); });
        return;
      }
      effect(svg, xs[0], y, r.stock, {accent: r.accent, digits: 2, valueX: o.cols[0][1] + 8});
      effect(svg, xs[1], y, r.damage, {accent: r.accent, digits: 1, valueX: o.cols[1][1] + 8});
    });
    return {xs: xs, doms: doms};
  }
  function focus(svg, moduleId) {
    Array.prototype.forEach.call(svg.querySelectorAll('[data-module]'), function (p) {
      p.classList.remove('module-dim', 'module-focus');
      if (!moduleId) return;
      p.classList.add(p.getAttribute('data-module') === moduleId ? 'module-focus' : 'module-dim');
    });
  }
  return {el: el, empty: empty, text: text, marker: marker, part: part, token: token, jointToken: jointToken, stacked: stacked, box: box, wire: wire, path: path, note: note, row: row, focus: focus, signed: signed, scale: scale, domain: domain, effect: effect, effectAxis: effectAxis, forest: forest};
}());
