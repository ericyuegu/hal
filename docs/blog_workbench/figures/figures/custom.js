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
      if (r.sub) text(svg, o.labelX, y + 14, r.sub.replace(/^[a-z0-9]{8} → [a-z0-9]{8} · /, ''), {class: 'diagram-note'});
      if (r.baseline) {
        xs.forEach(function (x, k) { text(svg, x(0) + 6, y + 4, 'baseline', {class: 'diagram-note'}); });
        return;
      }
      effect(svg, xs[0], y, r.stock, {accent: r.accent, digits: 2, valueX: o.cols[0][1] + 8});
      effect(svg, xs[1], y, r.damage, {accent: r.accent, digits: 1, valueX: o.cols[1][1] + 8});
    });

    if (o.top <= 80) {
      var mobile = document.createElement('div');
      mobile.className = 'mobile-forest';
      svg.classList.add('desktop-forest');
      ['stock', 'damage'].forEach(function (metric, k) {
        var heading = document.createElement('h5');
        heading.textContent = k ? 'Change in net damage / min' : 'Change in net stocks / min';
        mobile.appendChild(heading);
        var panel = el('svg', {class:'model-diagram', viewBox: '0 0 320 ' + (rows.length * 76 + 32), role: 'img', 'aria-label': heading.textContent + '; treatment minus baseline, 95% intervals'});
        var mx = scale([30, 266], doms[k].lo, doms[k].hi);
        effectAxis(panel, mx, doms[k], 32, rows.length * 76 - 4, null);
        rows.forEach(function (r, i) {
          var yy = i * 76;
          text(panel, 8, yy + 18, r.label.replace('behavior cloning', 'BC'), {class:'diagram-title'});
          if (r.sub) text(panel, 8, yy + 33, r.sub.replace(/^[a-z0-9]{8} → [a-z0-9]{8} · /, ''), {class:'diagram-note'});
          if (r.baseline) text(panel, mx(0) + 6, yy + 54, 'baseline', {class:'diagram-note'});
          else effect(panel, mx, yy + 54, r[metric], {accent:r.accent, digits:k ? 1 : 2, valueX:278});
        });
        mobile.appendChild(panel);
      });
      rows.filter(function(r){return r.loss;}).forEach(function(r){var note=document.createElement('p');note.className='figure-note';note.textContent=r.label+' · validation loss '+r.loss[0].val_loss.toFixed(3)+' → '+r.loss[1].val_loss.toFixed(3)+' ('+r.loss[0].metric+')';mobile.appendChild(note);});
      svg.after(mobile);
    }
    return {xs: xs, doms: doms};
  }
  /* Vertical Δ bars with 95% whiskers: rows of {label, sub, stock, damage, accent}; one panel per metric,
     laid out left to right from (o.x, o.y). Returns the drawn width and height. */
  function effectBars(target, rows, opts) {
    var o = Object.assign({x: 0, y: 0, panelW: 300, panelH: 190, gap: 44, barW: 26, axisW: 40, labelH: 34,
      metrics: ['stock', 'damage'], titles: {stock: 'Δ net stocks / min', damage: 'Δ net damage / min'},
      digits: {stock: 2, damage: 1}}, opts || {});
    o.metrics.forEach(function (m, k) {
      var g = el('g', {transform: 'translate(' + (o.x + k * (o.panelW + o.gap)) + ',' + o.y + ')'});
      target.appendChild(g);
      var dom = domain(rows.map(function (r) { return r[m]; }));
      var top = 30, bottom = o.panelH - o.labelH;
      var yy = scale([bottom, top], dom.lo, dom.hi);
      text(g, 0, 12, o.titles[m], {class: 'diagram-note'});
      dom.ticks.forEach(function (v) {
        g.appendChild(el('line', {x1: o.axisW, y1: yy(v), x2: o.panelW, y2: yy(v), class: v === 0 ? 'effect-zero' : 'effect-grid'}));
        text(g, o.axisW - 6, yy(v) + 4, signed(v, dom.digits), {class: 'diagram-note', 'text-anchor': 'end'});
      });
      var slot = (o.panelW - o.axisW) / rows.length;
      rows.forEach(function (r, i) {
        var d = r[m], cx = o.axisW + slot * (i + 0.5), cls = r.accent ? ' accent' : '';
        var y0 = yy(0), y1 = yy(d.delta);
        g.appendChild(el('rect', {x: cx - o.barW / 2, y: Math.min(y0, y1), width: o.barW, height: Math.max(1, Math.abs(y1 - y0)), class: 'effect-bar' + cls}));
        g.appendChild(el('line', {x1: cx, y1: yy(d.lo), x2: cx, y2: yy(d.hi), class: 'effect-ci' + cls}));
        [d.lo, d.hi].forEach(function (v) { g.appendChild(el('line', {x1: cx - 5, y1: yy(v), x2: cx + 5, y2: yy(v), class: 'effect-ci' + cls})); });
        text(g, cx, yy(d.hi) - 7, signed(d.delta, o.digits[m]), {class: 'diagram-label effect-val' + cls, 'text-anchor': 'middle'});
        text(g, cx, bottom + 18, r.label, {class: 'diagram-label', 'text-anchor': 'middle'});
        if (r.sub) text(g, cx, bottom + 31, r.sub, {class: 'diagram-note', 'text-anchor': 'middle'});
      });
    });
    return {width: o.metrics.length * o.panelW + (o.metrics.length - 1) * o.gap, height: o.panelH};
  }
  /* The study table the page embeds (evidence.json without the boot rows). */
  var evidenceCache;
  function evidence() {
    if (evidenceCache === undefined) {
      var node = document.getElementById('wb-evidence');
      try { evidenceCache = node ? JSON.parse(node.textContent) : null; } catch (err) { evidenceCache = null; }
    }
    return evidenceCache;
  }
  function study(id) {
    var ev = evidence();
    return ev && ev.studies ? ev.studies.filter(function (s) { return s.id === id; })[0] || null : null;
  }
  function focus(svg, moduleId) {
    Array.prototype.forEach.call(svg.querySelectorAll('[data-module]'), function (p) {
      p.classList.remove('module-dim', 'module-focus');
      if (!moduleId) return;
      p.classList.add(p.getAttribute('data-module') === moduleId ? 'module-focus' : 'module-dim');
    });
  }

  function bindReadout(target, callback, label) {
    target.setAttribute('tabindex', '0');
    target.setAttribute('role', 'button');
    if (label) target.setAttribute('aria-label', label);
    target.addEventListener('mouseenter', callback);
    target.addEventListener('focus', callback);
    target.addEventListener('click', callback);
    target.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); callback(e); }
    });
  }
  function sweep(svg, rows, values, labels, secondary, axisTitle) {
    var readout=document.createElement('p');readout.className='chart-readout';readout.setAttribute('aria-live','polite');svg.after(readout);
    function render() {
      empty(svg);
      var small = window.matchMedia('(max-width:760px)').matches;
      var width = small ? 340 : 880, panelWidth = small ? 340 : 430;
      svg.classList.add('reflow-chart');
      svg.setAttribute('viewBox', '0 0 ' + width + ' ' + (small ? 660 : 330));
      ['stock', 'damage'].forEach(function (metric, k) {
        var g = el('g', {transform:'translate(' + (small ? 0 : k * 450) + ',' + (small ? k * 330 : 0) + ')'});
        svg.appendChild(g);
        var domainY = domain(rows.filter(function(r){return !r.baseline;}).map(function(r){return r[metric];}));
        var yy = scale([228, 38], domainY.lo, domainY.hi);
        var xx = function(v) {return 50 + Math.log(v/values[0])/Math.log(values[values.length-1]/values[0])*(panelWidth-84);};
        text(g, 12, 18, k ? 'Δ net damage / min' : 'Δ net stocks / min', {class:'diagram-title'});
        domainY.ticks.forEach(function(v){
          g.appendChild(el('line',{x1:50,x2:panelWidth-24,y1:yy(v),y2:yy(v),class:v===0?'effect-zero':'effect-grid'}));
          text(g,42,yy(v)+4,signed(v,domainY.digits),{class:'diagram-note','text-anchor':'end'});
        });
        var points=[];
        rows.forEach(function(r,i){
          var d=r.baseline?{delta:0,lo:0,hi:0}:r[metric], cx=xx(values[i]);
          points.push(cx+','+yy(d.delta));
          g.appendChild(el('line',{x1:cx,x2:cx,y1:yy(d.lo),y2:yy(d.hi),class:'effect-ci'+(r.accent?' accent':'')}));
          var dot=el('circle',{cx:cx,cy:yy(d.delta),r:4,class:'effect-pt'+(r.accent?' accent':'')});
          var title=el('title',{});title.textContent=r.label+': '+signed(d.delta,k?1:2)+' ['+signed(d.lo,2)+', '+signed(d.hi,2)+']';dot.appendChild(title);g.appendChild(dot);
          bindReadout(dot,function(){readout.textContent=title.textContent+(r.sub?' · '+r.sub:'');},title.textContent);
          var lineY=small && i%2 ? 268 : 248;
          text(g,cx,lineY,labels[i],{class:'diagram-note','text-anchor':'middle'});
          if(secondary) text(g,cx,lineY+12,secondary[i],{class:'diagram-note','text-anchor':'middle'});
        });
        g.insertBefore(el('polyline',{points:points.join(' '),class:'chart-line faint'}),g.firstChild);
        text(g,12,310,axisTitle,{class:'diagram-note'});
      });
    }
    render();
    window.matchMedia('(max-width:760px)').addEventListener('change',render);
  }
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.fig-src.custom svg.model-diagram').forEach(function(svg) {
      if (svg.closest('.mobile-forest') || svg.classList.contains('reflow-chart') || svg.classList.contains('desktop-forest')) return;
      var wrap = document.createElement('div'); wrap.className = 'svg-scroll';
      wrap.tabIndex = 0; wrap.setAttribute('role','region'); wrap.setAttribute('aria-label',svg.getAttribute('aria-label') || 'Figure detail');
      svg.before(wrap); wrap.appendChild(svg);
      var hint=document.createElement('p');hint.className='scroll-hint';hint.textContent='Swipe sideways to inspect the diagram.';wrap.before(hint);
    });
  });
  return {bindReadout:bindReadout, sweep:sweep, el: el, empty: empty, text: text, marker: marker, part: part, token: token, jointToken: jointToken, stacked: stacked, box: box, wire: wire, path: path, note: note, row: row, focus: focus, signed: signed, scale: scale, domain: domain, effect: effect, effectAxis: effectAxis, forest: forest, effectBars: effectBars, evidence: evidence, study: study};
}());

window.HALFIG_AWR_LEDGER = {"fps":60,"games":[{"file":"diamond-diamond-00018278aa8bb1c30871ab4c.slp","label":"ranked \u00b7 diamond vs diamond","p1":"Sheik","p2":"Fox","stage":"Battlefield","frames":6929,"p1Percent":[[0,156],[50,119],[95,578],[235,373],[278,33],[368,373],[518,227],[608,69],[728,303],[811,129],[937,114],[1006,169],[1056,30],[1236,238],[0,239],[20,9],[38,2],[56,5],[75,195],[199,190],[246,134],[289,681],[319,6],[346,5],[365,850],[515,90],[545,32],[565,43],[704,439],[834,107],[904,70],[954,142],[974,12],[992,4],[1010,3],[1029,24],[1079,736]],"p2Percent":[[0,343],[90,219],[230,47],[357,107],[487,60],[572,235],[602,48],[652,8],[682,98],[728,8],[755,146],[855,130],[975,183],[1101,153],[0,523],[99,1076],[229,490],[259,48],[309,8],[339,58],[367,47],[413,8],[440,64],[466,47],[509,8],[535,115],[665,230],[755,57],[825,37],[954,307],[0,1375],[64,40],[93,47],[141,8],[170,43],[306,74],[333,47],[380,8],[408,45],[532,57],[559,26],[584,47],[629,8],[657,63],[757,45],[827,88]],"p1Stock":[[4,2866],[3,4063]],"p2Stock":[[4,1725],[3,3123],[2,736],[1,1344],[0,1]]},{"file":"master-master-000551fbfa951d4cce5f3f7a.slp","label":"ranked \u00b7 master vs master","p1":"Fox","p2":"Fox","stage":"Pok\u00e9mon Stadium","frames":12487,"p1Percent":[[0,1913],[120,327],[170,432],[320,122],[366,125],[408,80],[438,30],[458,9],[476,2],[494,2],[513,143],[629,44],[659,108],[749,226],[861,134],[889,30],[909,10],[926,2],[944,2],[961,27],[1091,449],[1141,98],[1160,10],[1177,2],[1194,2],[1211,32],[1261,193],[1331,91],[1481,114],[0,805],[50,839],[70,8],[88,2],[106,2],[125,262],[171,51],[189,9],[206,1],[223,2],[239,23],[419,417],[509,61],[639,54],[789,181],[836,85],[956,89],[1001,87],[0,469],[30,141],[60,7],[87,15],[131,260],[311,74],[331,9],[349,2],[368,2],[386,67],[436,6],[554,50],[672,160],[718,602],[738,5],[756,87],[846,49],[891,397],[961,133],[991,32],[1011,10],[1029,2],[1047,53],[1227,218],[0,493],[130,251],[160,760],[180,11],[207,5],[235,380]],"p2Percent":[[0,166],[50,129],[170,21],[216,102],[299,21],[342,64],[522,63],[572,91],[614,75],[764,66],[877,68],[907,180],[927,5],[954,5],[982,19],[1102,92],[1120,10],[1137,12],[1153,373],[1193,107],[1203,149],[0,192],[19,8],[37,17],[87,96],[107,8],[125,3],[143,1],[161,183],[207,117],[249,440],[288,454],[378,125],[415,537],[515,141],[605,138],[665,151],[755,29],[825,382],[855,64],[945,136],[995,61],[1041,31],[1221,176],[0,343],[42,26],[82,18],[122,21],[165,26],[195,6],[222,14],[263,203],[313,5],[431,54],[551,51],[671,69],[780,347],[930,67],[960,293],[1008,204],[1036,60],[1081,1321],[1141,66],[1171,546],[1221,781],[1267,225],[1357,158],[1399,592],[1419,49],[1569,126],[0,409],[20,5],[38,18],[88,42],[134,79],[153,8],[170,2],[187,2],[205,21],[295,140],[475,55],[595,129],[613,12],[631,1],[648,31],[698,20],[812,66],[842,6],[869,93],[919,71],[964,75],[1034,57],[1054,10],[1071,31],[1161,53],[1281,72]],"p1Stock":[[4,4699],[3,2978],[2,2865],[1,1945]],"p2Stock":[[4,1758],[3,3515],[2,5646],[1,1567],[0,1]]},{"file":"h2h-bc-baseline-wins-cptfalcon-vs-marth-yoshis_story.slp","label":"model vs model \u00b7 BC baseline wins","p1":"Captain Falcon","p2":"Marth","stage":"Yoshi\u2019s Story","frames":4646,"p1Percent":[[0,312],[100,112],[230,113],[270,12],[361,94],[391,33],[431,60],[467,89],[591,49],[678,180],[808,230],[818,90],[0,431],[130,342],[0,1061],[100,217],[230,1218],[240,3]],"p2Percent":[[0,1275],[70,74],[0,353],[60,16],[124,188],[178,15],[236,33],[356,298],[536,263],[0,1166],[30,34],[70,7],[100,51],[240,133],[300,17],[364,32],[484,277],[544,121],[614,84],[735,209]],"p1Stock":[[4,1314],[3,773],[2,824],[1,1734],[0,1]],"p2Stock":[[4,1289],[3,1166],[2,2131],[1,60]]},{"file":"h2h-lopsided-awr-strong-b0.3-detach-wins-4-0-falco-vs-falco.slp","label":"model vs model \u00b7 AWR wins 4\u20130","p1":"Falco","p2":"Falco","stage":"Dreamland","frames":4220,"p1Percent":[[0,484],[30,57],[110,461],[0,484],[80,306],[0,468],[150,130],[286,30],[361,51],[488,268],[605,248],[635,27],[662,25],[736,53],[776,117],[802,24],[826,92],[986,147],[0,219],[20,4],[38,3],[56,2],[75,487],[145,33]],"p2Percent":[[0,116],[30,258],[150,1265],[230,47],[350,362],[470,27],[550,107],[580,1418],[610,204],[760,151],[920,66],[980,7],[1071,192]],"p1Stock":[[4,942],[3,790],[2,1680],[1,807],[0,1]],"p2Stock":[[4,4220]]}]};
window.HALFIG_DATA = {"o50": {"run": "p1fyyp1z", "val_updates": [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536, 69632, 73728, 77824, 81920, 86016, 90112, 94208, 98304, 102400, 106496, 110592, 114688, 118784, 122880, 126976, 131072], "val_nll": [1.5791820561203818, 1.5103761385449477, 1.4795180697047676, 1.4634260317213854, 1.448551541973239, 1.4455258612086794, 1.447027123446336, 1.4423722957420981, 1.4468302051954895, 1.4458469939950989, 1.452989943003546, 1.4459457159372022, 1.4419165996472352, 1.4409629425272865, 1.4449993813532684, 1.43389530684944, 1.4311546797024775, 1.4244409537506784, 1.4211250418336558, 1.4172396296652974, 1.4153978236144047, 1.4044030808711607, 1.4019645986272031, 1.3958003246837993, 1.3926876962799717, 1.3878077738693424, 1.386225900948772, 1.3826830670396386, 1.3797439728959595, 1.37916600218562, 1.3784998608054468, 1.3793123233211932], "evals": [{"updates": 8192, "evaluation_id": "eval-2c70e0bd1e98b121", "boots": 96, "stock": -0.2175287496076083, "stock_ci95": [-0.34505186256697096, -0.09014582412726471], "damage": 9.538520243221672, "damage_ci95": [0.2577713923484313, 18.92569181702497]}, {"updates": 16384, "evaluation_id": "eval-0296ae51b377699a", "boots": 96, "stock": 0.058385292803717806, "stock_ci95": [-0.07960562782305144, 0.19638755278654949], "damage": 45.02695437240303, "damage_ci95": [36.21477002424307, 53.607311543896834]}, {"updates": 24576, "evaluation_id": "eval-c5d76bac60cd0baa", "boots": 96, "stock": 0.1643922788658701, "stock_ci95": [0.037105423576079384, 0.29692903795701064], "damage": 50.31414286786696, "damage_ci95": [40.76983192946281, 59.89840302291335]}, {"updates": 32768, "evaluation_id": "eval-44fd030a47dd71cc", "boots": 96, "stock": 0.09553168246330214, "stock_ci95": [-0.04243709614435894, 0.2335220970355575], "damage": 25.059970496130568, "damage_ci95": [16.04224469027401, 34.327991616344626]}, {"updates": 40960, "evaluation_id": "eval-d7dbc72ac93e635f", "boots": 96, "stock": -0.1540118311770694, "stock_ci95": [-0.2868328981896649, -0.021240003186000486], "damage": 39.58321000414505, "damage_ci95": [31.091801503917964, 48.27435917984737]}, {"updates": 49152, "evaluation_id": "eval-bf2778dbf16370e6", "boots": 96, "stock": 0.0, "stock_ci95": [-0.12735698871475576, 0.1271762745594461], "damage": 21.428472232053, "damage_ci95": [11.513013970098987, 31.0698602855801]}, {"updates": 57344, "evaluation_id": "eval-0e6ccc26c00a136e", "boots": 96, "stock": -0.17509908279327258, "stock_ci95": [-0.3130865998240218, -0.03713232368777022], "damage": 27.191259956458545, "damage_ci95": [17.820320145005095, 36.55525094830308]}, {"updates": 65536, "evaluation_id": "eval-afc4c215a78ba5fd", "boots": 96, "stock": -0.07958368888083228, "stock_ci95": [-0.20157638495748414, 0.04244463406977722], "damage": 19.31362356262325, "damage_ci95": [10.576002063891133, 27.86091904671398]}, {"updates": 73728, "evaluation_id": "eval-c3bf13076345c32b", "boots": 96, "stock": -0.0742919052778208, "stock_ci95": [-0.20699355135474629, 0.05311380222847505], "damage": 24.609878265051414, "damage_ci95": [15.427614078282568, 33.645228691542066]}, {"updates": 81920, "evaluation_id": "eval-e0d3bb525bae1480", "boots": 96, "stock": 0.0795836888808323, "stock_ci95": [-0.04246002786439329, 0.20164856546502996], "damage": 39.0541814099931, "damage_ci95": [30.46235277581308, 47.59954311756489]}, {"updates": 90112, "evaluation_id": "eval-ac2ec42b6ad07ffa", "boots": 96, "stock": 0.11672824867833266, "stock_ci95": [-0.010611658970757514, 0.24942003771479235], "damage": 33.16911890117338, "damage_ci95": [24.248227656890695, 41.99491384524602]}, {"updates": 98304, "evaluation_id": "eval-1b5c4c936c2bc5be", "boots": 96, "stock": 0.13784510981218606, "stock_ci95": [0.021203099186331075, 0.2544832796532666], "damage": 31.63140892724937, "damage_ci95": [23.093274978700016, 40.369370243107255]}, {"updates": 106496, "evaluation_id": "eval-e75cfa5fb73a084c", "boots": 96, "stock": 0.24929682676964654, "stock_ci95": [0.12194743426110095, 0.3768023324615048], "damage": 38.28511609340493, "damage_ci95": [29.216328289072763, 47.51026218077205]}, {"updates": 114688, "evaluation_id": "eval-3dee7d6e8a1b2ee2", "boots": 96, "stock": 0.25989180014850966, "stock_ci95": [0.1591194858060636, 0.35548481230932494], "damage": 37.51009966720893, "damage_ci95": [29.095367560528718, 46.13829591940387]}, {"updates": 122880, "evaluation_id": "eval-5ae4230bed5e2f47", "boots": 96, "stock": 0.29164825452938586, "stock_ci95": [0.16962479349064863, 0.408605524528215], "damage": 41.47654631197902, "damage_ci95": [34.267636214662495, 48.7415592216571]}, {"updates": 131072, "evaluation_id": "eval-2e8548534cb977d9", "boots": 96, "stock": 0.3449592461608543, "stock_ci95": [0.22285405146905785, 0.46764979407466467], "damage": 37.59018899188393, "damage_ci95": [29.14668985455726, 46.04813064402752]}]}, "optimizer": {"o3nihah5": {"arm": "AdamW", "val_loss": 1.012882363210714, "metric": "val/loss"}, "vlim96s9": {"arm": "Muon + AdamW", "val_loss": 0.9836503000609116, "metric": "val/loss"}, "p1fyyp1z": {"arm": "Muon + AdamW", "val_loss": 1.3793123233211932, "metric": "val/nll", "updates": 131072}, "tgpuo1be": {"arm": "AdamW", "val_loss": 1.3300914395989125, "metric": "val/nll", "updates": 131072}}, "iso_flop": {"fit": {"base": 1.635, "aScale": 89.11, "aPower": 0.368, "dScale": 2009, "dPower": 0.498}, "compute": "6 \u00b7 D \u00b7 N_eff", "n_eff": "trunk + 36 \u00b7 decoder", "source_csv": "/home/ericgu/src/hal/results/039_capacity_scaling/experiment_039_raw_outcomes.csv", "points": [{"run": "dt361n1f", "model": "L5", "layers": 5, "d_model": 320, "total_parameters": 6988015, "n_eff": 30830680, "d_exp": 26, "training_flops": 1.241407146688512e+16, "nll": 2.04296626119388, "fit_nll": 2.04554761255216, "residual": -0.0025813513582799708}, {"run": "accppjyl", "model": "L5", "layers": 5, "d_model": 320, "total_parameters": 6988015, "n_eff": 30830680, "d_exp": 27, "training_flops": 2.482814293377024e+16, "nll": 1.969640321004709, "fit_nll": 1.9713318901336587, "residual": -0.0016915691289496326}, {"run": "5tewqgz7", "model": "L5", "layers": 5, "d_model": 320, "total_parameters": 6988015, "n_eff": 30830680, "d_exp": 28, "training_flops": 4.965628586754048e+16, "nll": 1.9181448393649845, "fit_nll": 1.9187806485244439, "residual": -0.0006358091594593773}, {"run": "d7zifq1s", "model": "L5", "layers": 5, "d_model": 320, "total_parameters": 6988015, "n_eff": 30830680, "d_exp": 29, "training_flops": 9.931257173508096e+16, "nll": 1.878577972742213, "fit_nll": 1.8815697597171197, "residual": -0.0029917869749067627}, {"run": "oixqvhj2", "model": "L5", "layers": 5, "d_model": 320, "total_parameters": 6988015, "n_eff": 30830680, "d_exp": 30, "training_flops": 1.9862514347016192e+17, "nll": 1.848850233358216, "fit_nll": 1.8552211863355872, "residual": -0.006370952977371225}, {"run": "18266r7m", "model": "026base", "layers": 8, "d_model": 256, "total_parameters": 7081967, "n_eff": 29842712, "d_exp": 26, "training_flops": 1.2016263005995008e+16, "nll": 2.035924514698792, "fit_nll": 2.0474323320046994, "residual": -0.01150781730590733}, {"run": "fufjdhue", "model": "026base", "layers": 8, "d_model": 256, "total_parameters": 7081967, "n_eff": 29842712, "d_exp": 27, "training_flops": 2.4032526011990016e+16, "nll": 1.9650361186948464, "fit_nll": 1.973216609586198, "residual": -0.008180490891351644}, {"run": "gpfbsp9a", "model": "026base", "layers": 8, "d_model": 256, "total_parameters": 7081967, "n_eff": 29842712, "d_exp": 28, "training_flops": 4.806505202398003e+16, "nll": 1.913620997546124, "fit_nll": 1.9206653679769832, "residual": -0.007044370430859148}, {"run": "z0q5csaf", "model": "026base", "layers": 8, "d_model": 256, "total_parameters": 7081967, "n_eff": 29842712, "d_exp": 29, "training_flops": 9.613010404796006e+16, "nll": 1.8750435634464069, "fit_nll": 1.883454479169659, "residual": -0.008410915723252144}, {"run": "bpsr9sji", "model": "026base", "layers": 8, "d_model": 256, "total_parameters": 7081967, "n_eff": 29842712, "d_exp": 30, "training_flops": 1.9226020809592013e+17, "nll": 1.8460757223151592, "fit_nll": 1.8571059057881265, "residual": -0.011030183472967359}, {"run": "la5p7wpo", "model": "L7", "layers": 7, "d_model": 448, "total_parameters": 17810159, "n_eff": 43816664, "d_exp": 26, "training_flops": 1.7642919271858176e+16, "nll": 2.03654572370649, "fit_nll": 2.0265817486491096, "residual": 0.009963975057380559}, {"run": "54gkmcl9", "model": "L7", "layers": 7, "d_model": 448, "total_parameters": 17810159, "n_eff": 43816664, "d_exp": 27, "training_flops": 3.528583854371635e+16, "nll": 1.9621021836340145, "fit_nll": 1.9523660262306082, "residual": 0.009736157403406276}, {"run": "zzwng0rb", "model": "L7", "layers": 7, "d_model": 448, "total_parameters": 17810159, "n_eff": 43816664, "d_exp": 28, "training_flops": 7.05716770874327e+16, "nll": 1.907247679181244, "fit_nll": 1.8998147846213933, "residual": 0.0074328945598507445}, {"run": "jgqts6t4", "model": "L7", "layers": 7, "d_model": 448, "total_parameters": 17810159, "n_eff": 43816664, "d_exp": 29, "training_flops": 1.411433541748654e+17, "nll": 1.868542129334363, "fit_nll": 1.8626038958140692, "residual": 0.0059382335202937675}, {"run": "kwzr1hp4", "model": "L7", "layers": 7, "d_model": 448, "total_parameters": 17810159, "n_eff": 43816664, "d_exp": 30, "training_flops": 2.822867083497308e+17, "nll": 1.8402611210082496, "fit_nll": 1.8362553224325366, "residual": 0.004005798575712927}, {"run": "24m9fz5k", "model": "L10", "layers": 10, "d_model": 640, "total_parameters": 51154223, "n_eff": 111580568, "d_exp": 26, "training_flops": 4.492827097772851e+16, "nll": 1.984402606095476, "fit_nll": 1.9866072708009819, "residual": -0.0022046647055058433}, {"run": "9idtvi78", "model": "L10", "layers": 10, "d_model": 640, "total_parameters": 51154223, "n_eff": 111580568, "d_exp": 27, "training_flops": 8.985654195545702e+16, "nll": 1.912155025529341, "fit_nll": 1.9123915483824805, "residual": -0.00023652285313957577}, {"run": "2g1pgvvz", "model": "L10", "layers": 10, "d_model": 640, "total_parameters": 51154223, "n_eff": 111580568, "d_exp": 28, "training_flops": 1.7971308391091405e+17, "nll": 1.8587825080113989, "fit_nll": 1.8598403067732656, "residual": -0.001057798761866735}, {"run": "79d8l0af", "model": "L10", "layers": 10, "d_model": 640, "total_parameters": 51154223, "n_eff": 111580568, "d_exp": 29, "training_flops": 3.594261678218281e+17, "nll": 1.8218180628041216, "fit_nll": 1.8226294179659415, "residual": -0.0008113551618198489}, {"run": "9885eba2", "model": "L10", "layers": 10, "d_model": 640, "total_parameters": 51154223, "n_eff": 111580568, "d_exp": 30, "training_flops": 7.188523356436562e+17, "nll": 1.7956324093955631, "fit_nll": 1.796280844584409, "residual": -0.0006484351888458306}, {"run": "u7t65xfs", "model": "L13", "layers": 13, "d_model": 832, "total_parameters": 111720303, "n_eff": 230364248, "d_exp": 26, "training_flops": 9.275689793696563e+16, "nll": 1.9601956102725355, "fit_nll": 1.963808701397762, "residual": -0.003613091125226564}, {"run": "m2cpo3rn", "model": "L13", "layers": 13, "d_model": 832, "total_parameters": 111720303, "n_eff": 230364248, "d_exp": 27, "training_flops": 1.8551379587393126e+17, "nll": 1.8862407563798804, "fit_nll": 1.8895929789792607, "residual": -0.0033522225993802834}, {"run": "z2h1ain0", "model": "L13", "layers": 13, "d_model": 832, "total_parameters": 111720303, "n_eff": 230364248, "d_exp": 28, "training_flops": 3.710275917478625e+17, "nll": 1.8335726348811832, "fit_nll": 1.8370417373700458, "residual": -0.0034691024888626742}, {"run": "2epkrfmj", "model": "L13", "layers": 13, "d_model": 832, "total_parameters": 111720303, "n_eff": 230364248, "d_exp": 29, "training_flops": 7.42055183495725e+17, "nll": 1.797502125240838, "fit_nll": 1.7998308485627215, "residual": -0.002328723321883386}, {"run": "spwekixh", "model": "L16", "layers": 16, "d_model": 1024, "total_parameters": 205245039, "n_eff": 327994904, "d_exp": 26, "training_flops": 1.3206819243137434e+17, "nll": 1.9580559016337016, "fit_nll": 1.9547165944100204, "residual": 0.0033393072236811427}, {"run": "t2o011wf", "model": "L16", "layers": 16, "d_model": 1024, "total_parameters": 205245039, "n_eff": 327994904, "d_exp": 27, "training_flops": 2.6413638486274867e+17, "nll": 1.8843847255524508, "fit_nll": 1.880500871991519, "residual": 0.0038838535609317937}, {"run": "0ls85skw", "model": "L16", "layers": 16, "d_model": 1024, "total_parameters": 205245039, "n_eff": 327994904, "d_exp": 28, "training_flops": 5.282727697254973e+17, "nll": 1.8317713941546805, "fit_nll": 1.8279496303823042, "residual": 0.0038217637723763076}, {"run": "to0fx9ig", "model": "L16", "layers": 16, "d_model": 1024, "total_parameters": 205245039, "n_eff": 327994904, "d_exp": 29, "training_flops": 1.0565455394509948e+18, "nll": 1.7966224093384078, "fit_nll": 1.7907387415749798, "residual": 0.005883667763427969}, {"run": "tlgvpsqh", "model": "L16", "layers": 16, "d_model": 1024, "total_parameters": 205245039, "n_eff": 327994904, "d_exp": 30, "training_flops": 2.113091078901989e+18, "nll": 1.7711337818716562, "fit_nll": 1.7643901681934473, "residual": 0.00674361367820886}, {"run": "wcdj8arn", "model": "L18", "layers": 18, "d_model": 1152, "total_parameters": 293127087, "n_eff": 503691672, "d_exp": 26, "training_flops": 2.0281305548508365e+17, "nll": 1.9470057232359543, "fit_nll": 1.9451549990016954, "residual": 0.0018507242342589159}, {"run": "84z3js0r", "model": "L18", "layers": 18, "d_model": 1152, "total_parameters": 293127087, "n_eff": 503691672, "d_exp": 27, "training_flops": 4.056261109701673e+17, "nll": 1.8677958767601075, "fit_nll": 1.870939276583194, "residual": -0.0031433998230865345}, {"run": "rz25c5pr", "model": "L18", "layers": 18, "d_model": 1152, "total_parameters": 293127087, "n_eff": 503691672, "d_exp": 28, "training_flops": 8.112522219403346e+17, "nll": 1.815900191321837, "fit_nll": 1.8183880349739792, "residual": -0.002487843652142052}, {"run": "03f39e4a", "model": "L18", "layers": 18, "d_model": 1152, "total_parameters": 293127087, "n_eff": 503691672, "d_exp": 29, "training_flops": 1.6225044438806692e+18, "nll": 1.7809871689620178, "fit_nll": 1.7811771461666548, "residual": -0.00018997720463698187}], "checkpoints": [{"run": "ldlin05b", "model": "L3", "n_eff": 23742936, "positions": 350981584.0, "training_flops": 4.999999971654374e+16, "nll": 1.9237731207336688, "exact": true}, {"run": "ui30df7n", "model": "L3", "n_eff": 23742936, "positions": 701963168.0, "training_flops": 9.999999943308749e+16, "nll": 1.8892911459614743, "exact": true}, {"run": "9h8otw5m", "model": "L4", "n_eff": 26696984, "positions": 312145120.0, "training_flops": 4.999999964590848e+16, "nll": 1.916986007707574, "exact": true}, {"run": "vea6s9sm", "model": "L4", "n_eff": 26696984, "positions": 624290248.0, "training_flops": 1.000000005732722e+17, "nll": 1.882034288426402, "exact": true}, {"run": "dt361n1f", "model": "L5", "n_eff": 30830680, "positions": 67108864.0, "training_flops": 1.241407146688512e+16, "nll": 2.04296626119388, "exact": true}, {"run": "accppjyl", "model": "L5", "n_eff": 30830680, "positions": 134217728.0, "training_flops": 2.482814293377024e+16, "nll": 1.969640321004709, "exact": true}, {"run": "5tewqgz7", "model": "L5", "n_eff": 30830680, "positions": 268435456.0, "training_flops": 4.965628586754048e+16, "nll": 1.9181448393649845, "exact": true}, {"run": "d7zifq1s", "model": "L5", "n_eff": 30830680, "positions": 536870912.0, "training_flops": 9.931257173508096e+16, "nll": 1.878577972742213, "exact": true}, {"run": "oixqvhj2", "model": "L5", "n_eff": 30830680, "positions": 1073741824.0, "training_flops": 1.9862514347016192e+17, "nll": 1.848850233358216, "exact": true}, {"run": "p4sqr8hx", "model": "L5", "n_eff": 30830680, "positions": 1621761184.0, "training_flops": 3.000000006019507e+17, "nll": 1.8360281943763683, "exact": true}, {"run": "la5p7wpo", "model": "L7", "n_eff": 43816664, "positions": 67108864.0, "training_flops": 1.7642919271858176e+16, "nll": 2.03654572370649, "exact": true}, {"run": "54gkmcl9", "model": "L7", "n_eff": 43816664, "positions": 134217728.0, "training_flops": 3.528583854371635e+16, "nll": 1.9621021836340145, "exact": true}, {"run": "zzwng0rb", "model": "L7", "n_eff": 43816664, "positions": 268435456.0, "training_flops": 7.05716770874327e+16, "nll": 1.907247679181244, "exact": true}, {"run": "jgqts6t4", "model": "L7", "n_eff": 43816664, "positions": 536870912.0, "training_flops": 1.411433541748654e+17, "nll": 1.868542129334363, "exact": true}, {"run": "kwzr1hp4", "model": "L7", "n_eff": 43816664, "positions": 1073741824.0, "training_flops": 2.822867083497308e+17, "nll": 1.8402611210082496, "exact": true}, {"run": "2ubngnfw", "model": "L7", "n_eff": 43816664, "positions": 1141118368.0, "training_flops": 3.000000006893061e+17, "nll": 1.838295597502493, "exact": true}, {"run": "24m9fz5k", "model": "L10", "n_eff": 111580568, "positions": 67108864.0, "training_flops": 4.492827097772851e+16, "nll": 1.984402606095476, "exact": true}, {"run": "9idtvi78", "model": "L10", "n_eff": 111580568, "positions": 134217728.0, "training_flops": 8.985654195545702e+16, "nll": 1.912155025529341, "exact": true}, {"run": "2g1pgvvz", "model": "L10", "n_eff": 111580568, "positions": 268435456.0, "training_flops": 1.7971308391091405e+17, "nll": 1.8587825080113989, "exact": true}, {"run": "79d8l0af", "model": "L10", "n_eff": 111580568, "positions": 536870912.0, "training_flops": 3.594261678218281e+17, "nll": 1.8218180628041216, "exact": true}, {"run": "9885eba2", "model": "L10", "n_eff": 111580568, "positions": 1073741824.0, "training_flops": 7.188523356436562e+17, "nll": 1.7956324093955631, "exact": true}, {"run": "u7t65xfs", "model": "L13", "n_eff": 230364248, "positions": 67108864.0, "training_flops": 9.275689793696563e+16, "nll": 1.9601956102725355, "exact": true}, {"run": "m2cpo3rn", "model": "L13", "n_eff": 230364248, "positions": 134217728.0, "training_flops": 1.8551379587393126e+17, "nll": 1.8862407563798804, "exact": true}, {"run": "z2h1ain0", "model": "L13", "n_eff": 230364248, "positions": 268435456.0, "training_flops": 3.710275917478625e+17, "nll": 1.8335726348811832, "exact": true}, {"run": "2epkrfmj", "model": "L13", "n_eff": 230364248, "positions": 536870912.0, "training_flops": 7.42055183495725e+17, "nll": 1.797502125240838, "exact": true}, {"run": "piba1f7t", "model": "L13", "n_eff": 230364248, "positions": 723491896.0, "training_flops": 9.999999993368052e+17, "nll": 1.784338300827231, "exact": true}, {"run": "spwekixh", "model": "L16", "n_eff": 327994904, "positions": 67108864.0, "training_flops": 1.3206819243137434e+17, "nll": 1.9580559016337016, "exact": true}, {"run": "t2o011wf", "model": "L16", "n_eff": 327994904, "positions": 134217728.0, "training_flops": 2.6413638486274867e+17, "nll": 1.8843847255524508, "exact": true}, {"run": "0ls85skw", "model": "L16", "n_eff": 327994904, "positions": 268435456.0, "training_flops": 5.282727697254973e+17, "nll": 1.8317713941546805, "exact": true}, {"run": "to0fx9ig", "model": "L16", "n_eff": 327994904, "positions": 536870912.0, "training_flops": 1.0565455394509948e+18, "nll": 1.7966224093384078, "exact": true}, {"run": "tlgvpsqh", "model": "L16", "n_eff": 327994904, "positions": 1073741824.0, "training_flops": 2.113091078901989e+18, "nll": 1.7711337818716562, "exact": true}, {"run": "wcdj8arn", "model": "L18", "n_eff": 503691672, "positions": 67108864.0, "training_flops": 2.0281305548508365e+17, "nll": 1.9470057232359543, "exact": true}, {"run": "84z3js0r", "model": "L18", "n_eff": 503691672, "positions": 134217728.0, "training_flops": 4.056261109701673e+17, "nll": 1.8677958767601075, "exact": true}, {"run": "rz25c5pr", "model": "L18", "n_eff": 503691672, "positions": 268435456.0, "training_flops": 8.112522219403346e+17, "nll": 1.815900191321837, "exact": true}, {"run": "03f39e4a", "model": "L18", "n_eff": 503691672, "positions": 536870912.0, "training_flops": 1.6225044438806692e+18, "nll": 1.7809871689620178, "exact": true}], "levels": [{"c": 5e+16, "points": [{"model": "L3", "n_eff": 23742936, "nll": 1.9237731207336688, "exact": true, "runs": ["ldlin05b"]}, {"model": "L4", "n_eff": 26696984, "nll": 1.916986007707574, "exact": true, "runs": ["9h8otw5m"]}, {"model": "L5", "n_eff": 30830680, "nll": 1.9181448393649845, "exact": true, "runs": ["5tewqgz7"]}, {"model": "L7", "n_eff": 43816664, "nll": 1.9345192104495925, "exact": false, "runs": ["54gkmcl9", "zzwng0rb"], "between_positions": [134217728.0, 268435456.0]}, {"model": "L10", "n_eff": 111580568, "nll": 1.9732544754911112, "exact": false, "runs": ["24m9fz5k", "9idtvi78"], "between_positions": [67108864.0, 134217728.0]}]}, {"c": 1e+17, "points": [{"model": "L3", "n_eff": 23742936, "nll": 1.8892911459614743, "exact": true, "runs": ["ui30df7n"]}, {"model": "L4", "n_eff": 26696984, "nll": 1.882034288426402, "exact": true, "runs": ["vea6s9sm"]}, {"model": "L5", "n_eff": 30830680, "nll": 1.878577972742213, "exact": true, "runs": ["d7zifq1s"]}, {"model": "L7", "n_eff": 43816664, "nll": 1.8877850269493925, "exact": false, "runs": ["zzwng0rb", "jgqts6t4"], "between_positions": [268435456.0, 536870912.0]}, {"model": "L10", "n_eff": 111580568, "nll": 1.9039194030861901, "exact": false, "runs": ["9idtvi78", "2g1pgvvz"], "between_positions": [134217728.0, 268435456.0]}, {"model": "L13", "n_eff": 230364248, "nll": 1.952173466797727, "exact": false, "runs": ["u7t65xfs", "m2cpo3rn"], "between_positions": [67108864.0, 134217728.0]}]}, {"c": 3e+17, "points": [{"model": "L5", "n_eff": 30830680, "nll": 1.8360281943763683, "exact": true, "runs": ["p4sqr8hx"]}, {"model": "L7", "n_eff": 43816664, "nll": 1.838295597502493, "exact": true, "runs": ["2ubngnfw"]}, {"model": "L10", "n_eff": 111580568, "nll": 1.8314559115209248, "exact": false, "runs": ["2g1pgvvz", "79d8l0af"], "between_positions": [268435456.0, 536870912.0]}, {"model": "L13", "n_eff": 230364248, "nll": 1.8497187836807258, "exact": false, "runs": ["m2cpo3rn", "z2h1ain0"], "between_positions": [134217728.0, 268435456.0]}, {"model": "L16", "n_eff": 327994904, "nll": 1.8747207375749384, "exact": false, "runs": ["t2o011wf", "0ls85skw"], "between_positions": [134217728.0, 268435456.0]}, {"model": "L18", "n_eff": 503691672, "nll": 1.9022670533131452, "exact": false, "runs": ["wcdj8arn", "84z3js0r"], "between_positions": [67108864.0, 134217728.0]}]}, {"c": 1e+18, "points": [{"model": "L13", "n_eff": 230364248, "nll": 1.784338300827231, "exact": true, "runs": ["piba1f7t"]}, {"model": "L16", "n_eff": 327994904, "nll": 1.7994116553400201, "exact": false, "runs": ["0ls85skw", "to0fx9ig"], "between_positions": [268435456.0, 536870912.0]}, {"model": "L18", "n_eff": 503691672, "nll": 1.8053642242598231, "exact": false, "runs": ["rz25c5pr", "03f39e4a"], "between_positions": [268435456.0, 536870912.0]}]}], "family": {"n_min": 23742936, "n_max": 503691672}, "u_total": 2409583026.0, "production": {"run": "p1fyyp1z", "positions": 8589934592, "trunk": 203606471, "decoder": 12890323, "passes": 20, "n_eff": 461412931, "flops": 2.3781041383158055e+19}}, "tier": {"fit_through_d_exp": 30, "models": [{"width": 256, "delay_frames": 1, "bc_parameters": 5436825, "run": "ss478s9q", "fit": {"s28": -1.0315678334453389, "amplitude": 1.4396033313595876e-10, "alpha": 0.01000000015055048}, "points": [{"d_exp": 28, "stock": -0.9794223714619034, "ci95": [-1.135445020624736, -0.8246032351701547]}, {"d_exp": 29, "stock": -1.0180920102875266, "ci95": [-1.1784660321561935, -0.8684415773914115]}, {"d_exp": 30, "stock": -1.0971891186535467, "ci95": [-1.247368877778886, -0.957499048720489]}, {"d_exp": 31, "stock": -0.8990238409774004, "ci95": [-1.0271218278108327, -0.770704972537818]}], "predicted_d31": -1.0315678334423761, "observed_d31": -0.8990238409774004, "observed_d31_ci95": [-1.0271218278108327, -0.770704972537818]}, {"width": 512, "delay_frames": 2, "bc_parameters": 39024281, "run": "wacv61om", "fit": {"s28": -0.9540348835660117, "amplitude": 2.4134045728708053, "alpha": 0.21402414024912061}, "points": [{"d_exp": 28, "stock": -0.9540348835659942, "ci95": [-1.120672777228789, -0.7770251689944919]}, {"d_exp": 29, "stock": -0.6212966691595239, "ci95": [-0.7602894957890335, -0.4776632892648078]}, {"d_exp": 30, "stock": -0.334433365479045, "ci95": [-0.451305568723057, -0.21748833918538124]}, {"d_exp": 31, "stock": -0.44620065454566915, "ci95": [-0.5898707637739997, -0.2972959342489634]}], "predicted_d31": -0.0871201581723483, "observed_d31": -0.44620065454566915, "observed_d31_ci95": [-0.5898707637739997, -0.2972959342489634]}, {"width": 768, "delay_frames": 3, "bc_parameters": 124712857, "run": "w3o895iq", "fit": {"s28": -0.7825441335899738, "amplitude": 0.2938806652363727, "alpha": 3.9999999999999494}, "points": [{"d_exp": 28, "stock": -0.7874248488108821, "ci95": [-0.9428476306736213, -0.6272410662344224]}, {"d_exp": 29, "stock": -0.42405886172589036, "ci95": [-0.5458809999297058, -0.30205583441193534]}, {"d_exp": 30, "stock": -0.5679028726862604, "ci95": [-0.7012262605693038, -0.42959033545413694]}, {"d_exp": 31, "stock": -0.010599629896256069, "ci95": [-0.11661704552482091, 0.09009685411817707]}], "predicted_d31": -0.4887352165628873, "observed_d31": -0.010599629896256069, "observed_d31_ci95": [-0.11661704552482091, 0.09009685411817707]}, {"width": 1024, "delay_frames": 6, "bc_parameters": 288848025, "run": "x7thsb6d", "fit": {"s28": -0.8002034589857908, "amplitude": 16.210232372006033, "alpha": 0.010000000168085587}, "points": [{"d_exp": 28, "stock": -0.7592483618198955, "ci95": [-0.876967980527292, -0.6371228287147135]}, {"d_exp": 29, "stock": -0.770426258252195, "ci95": [-0.9138890891797041, -0.6267371795284166]}, {"d_exp": 30, "stock": -0.5357924378997998, "ci95": [-0.6470820490390933, -0.4137249266523094]}, {"d_exp": 31, "stock": -0.41906479590483514, "ci95": [-0.5413678099919623, -0.29148982214704344]}], "predicted_d31": -0.4666016951738756, "observed_d31": -0.41906479590483514, "observed_d31_ci95": [-0.5413678099919623, -0.29148982214704344]}], "holdout": {"fit_through_data_exponent": 30, "predicted_d31_order": [512, 1024, 768, 256], "observed_d31_order": [768, 1024, 512, 256], "ordering_passed": false, "bootstrap_order_match_fraction": 0.0015, "bootstrap_resamples": 2000, "d34_winner_reported": false}}};
window.HALFIG_CI = {"codec_legacy_cartesian": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.2937801586712318, "lo": 0.0766177023566447, "hi": 0.5109426149858189}, "damage": {"delta": 16.270286146859085, "lo": 3.3990961120039316, "hi": 29.14147618171424}, "treatment": "cartesian", "control": "legacy"}, "codec_cartesian_polar": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.0963166416795991, "lo": -0.0843685269289834, "hi": 0.2770018102881816}, "damage": {"delta": 0.9011057331179781, "lo": -10.807762126068662, "hi": 12.609973592304618}, "treatment": "polar", "control": "cartesian"}, "projectile_inputs": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.09631518780752102, "lo": -0.13869931401631586, "hi": 0.3313296896313579}, "damage": {"delta": 10.550027056866792, "lo": -1.2948801991349814, "hi": 22.394934312868564}, "treatment": "projectiles", "control": "no projectiles"}, "past_action_encoding": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.26044941001407124, "lo": 0.08288385514199131, "hi": 0.43801496488615116}, "damage": {"delta": 23.68595106037023, "lo": 13.777944515811546, "hi": 33.59395760492891}, "treatment": "learned class plus physical value", "control": "256-way one-hot"}, "onehot_button_pruning": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.03667906451223713, "lo": -0.12469082756466318, "hi": 0.19804895658913743}, "damage": {"delta": -7.665238706452084, "lo": -18.047890746479084, "hi": 2.717413333574915}, "treatment": "52 retained button classes", "control": "256 button classes"}, "lean_feature_bundle": {"method": "paired", "n_boots": 95, "stock": {"delta": -0.25539462156458753, "lo": -0.5055880681966373, "hi": -0.0052011749325378}, "damage": {"delta": -13.157282511573772, "lo": -25.933284086102162, "hi": -0.381280937045382}, "treatment": "lean features", "control": "full features"}, "rle_bpe_actions": {"method": "welch", "n_boots": {"treatment": 25, "control": 95}, "stock": {"delta": -2.6772013228537497, "lo": -3.066794400077777, "hi": -2.2876082456297224}, "damage": {"delta": -172.13145411104648, "lo": -192.9918570085884, "hi": -151.27105121350456}, "treatment": "BPE/RLE spans", "control": "frame-aligned actions"}, "o26_to_o42_package": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.21882446127123592, "lo": -0.37267698222077184, "hi": -0.06497194032169998}, "damage": {"delta": -10.84463590561372, "lo": -22.40588349459832, "hi": 0.716611683370882}, "treatment": "O42 ranked-only", "control": "O26"}, "old_head_capacity": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.11051036142542887, "lo": -0.11247398590085562, "hi": 0.3334947087517134}, "damage": {"delta": 0.914330671356464, "lo": -12.202287448680737, "hi": 14.030948791393664}, "treatment": "larger state MLP", "control": "linear head"}, "future_state_expert": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.047535884535570866, "lo": -0.20918767540697078, "hi": 0.11411590633582905}, "damage": {"delta": -12.587044436640944, "lo": -23.133650070808017, "hi": -2.040438802473872}, "treatment": "future-state expert", "control": "policy only"}, "fixed_long_delay": {"method": "welch", "n_boots": {"treatment": 32, "control": 96}, "stock": {"delta": -1.1097536064220481, "lo": -1.4213418544579155, "hi": -0.7981653583861806}, "damage": {"delta": -85.82883474360841, "lo": -101.74296162015557, "hi": -69.91470786706125}, "treatment": "delay 12", "control": "delay 1"}, "rank_weighted_bc": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.12234229710092714, "lo": -0.2674956202900011, "hi": 0.022811026088146835}, "damage": {"delta": 0.9394848501856453, "lo": -9.846786185276422, "hi": 11.725755885647711}, "treatment": "rank weights 1/2/4", "control": "uniform behavior cloning"}, "awr_gamma:gamma 0.99855": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.3674220305623483, "lo": 0.194982459872588, "hi": 0.5398616012521086}, "damage": {"delta": 34.044282169320205, "lo": 20.35558759300563, "hi": 47.73297674563478}, "treatment": "gamma 0.99855", "control": "gamma 0.99618"}, "awr_gamma:gamma 0.99920": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.31933719227777085, "lo": 0.16016701063418923, "hi": 0.47850737392135245}, "damage": {"delta": 30.883513082392614, "lo": 18.06262249294203, "hi": 43.704403671843195}, "treatment": "gamma 0.99920", "control": "gamma 0.99618"}, "endpoint_flow": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.18707774154383353, "lo": -0.37640582322875316, "hi": 0.0022503401410861046}, "damage": {"delta": -32.99474527006023, "lo": -44.60206551631221, "hi": -21.387425023808248}, "treatment": "parallel endpoint flow", "control": "categorical autoregressive decoder"}, "attached_iql": {"method": "paired", "n_boots": 96, "stock": {"delta": -2.0143445823294925, "lo": -2.2229282241327333, "hi": -1.8057609405262518}, "damage": {"delta": -95.87821586445216, "lo": -105.5891359107609, "hi": -86.16729581814342}, "treatment": "attached IQL", "control": "behavior cloning"}, "exploration_modeling:XM K=1": {"method": "paired", "n_boots": 96, "stock": {"delta": -1.679074311722296, "lo": -1.9538935438977956, "hi": -1.4042550795467963}, "damage": {"delta": -89.74327084899316, "lo": -103.6913616487217, "hi": -75.79518004926463}, "treatment": "XM K=1", "control": "O26 behavior cloning"}, "data_source_mix": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.05851125597069115, "lo": -0.21593226122724338, "hi": 0.09890974928586108}, "damage": {"delta": 6.314509878517144, "lo": -4.755710026864878, "hi": 17.384729783899164}, "treatment": "all 44 sources", "control": "ranked-anonymized-1"}, "next_frame_loss_share": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.1398614323357623, "lo": -0.08072994858002566, "hi": 0.3604528132515502}, "damage": {"delta": 3.6512932495071984, "lo": -10.010407022121417, "hi": 17.312993521135812}, "treatment": "50% next-frame loss", "control": "O26 loss allocation"}, "player_identity:Cody": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.08268811551475534, "lo": -0.09854028422901599, "hi": 0.26391651525852666}, "damage": {"delta": 31.250556578834374, "lo": 17.14170175018313, "hi": 45.35941140748562}, "treatment": "Cody", "control": "identity masked"}, "player_identity:Master": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.021757767966640345, "lo": -0.2042873870242125, "hi": 0.16077185109093184}, "damage": {"delta": 4.217397826002823, "lo": -8.53345620996414, "hi": 16.968251861969787}, "treatment": "Master", "control": "identity masked"}, "player_identity:Mang0": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.06693389306694252, "lo": -0.27401563009909224, "hi": 0.1401478439652072}, "damage": {"delta": 7.114431725724953, "lo": -7.245123333898939, "hi": 21.473986785348846}, "treatment": "Mang0", "control": "identity masked"}, "player_identity:Platinum": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.3207430577221377, "lo": -0.4847450257887264, "hi": -0.15674108965554895}, "damage": {"delta": -9.249706364433925, "lo": -21.335957329222282, "hi": 2.836544600354431}, "treatment": "Platinum", "control": "identity masked"}, "optimizer_full_scale": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.1480642032114674, "lo": -0.03041356215387553, "hi": 0.32654196857681034}, "damage": {"delta": 15.155278013826193, "lo": 2.8446009585417915, "hi": 27.465955069110592}, "treatment": "all AdamW", "control": "Muon plus AdamW"}, "o37:D1-D0": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.39004278984123514, "lo": 0.24413996787593834, "hi": 0.535945611806532}, "damage": {"delta": 10.3039262725621, "lo": -1.5627389313246454, "hi": 22.170591476448845}}, "o37:D2-D0": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.5068228773050397, "lo": 0.34247461902948473, "hi": 0.6711711355805947}, "damage": {"delta": 15.294455721283782, "lo": 3.3954938385127456, "hi": 27.19341760405482}}, "o37:D3-D0": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.6087780470728391, "lo": 0.4235959937316125, "hi": 0.7939601004140657}, "damage": {"delta": 22.042008321433013, "lo": 9.797457289273538, "hi": 34.28655935359249}}, "o37:across": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.3627790672683218, "lo": 0.23344790567624565, "hi": 0.492110228860398}, "damage": {"delta": 13.516268885077345, "lo": 5.071343992221914, "hi": 21.961193777932777}}, "o37:within": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.24599897980451738, "lo": 0.1358480888767216, "hi": 0.3561498707323132}, "damage": {"delta": 8.525739436355659, "lo": 0.5568721225096676, "hi": 16.49460675020165}}, "horizon:D0:H2-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.25529948211761444, "lo": -0.4190575100391979, "hi": -0.09154145419603102}, "damage": {"delta": -14.441195412017715, "lo": -25.57800432035932, "hi": -3.3043865036761115}}, "horizon:D0:H4-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.7998107065623361, "lo": -0.9603186902218792, "hi": -0.639302722902793}, "damage": {"delta": -32.738868509235395, "lo": -45.32874526320782, "hi": -20.14899175526297}}, "horizon:D0:H6-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -1.0048223164325607, "lo": -1.2040647819031087, "hi": -0.8055798509620127}, "damage": {"delta": -54.46167677414758, "lo": -65.61164236880786, "hi": -43.31171117948729}}, "horizon:D1:H2-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.08484983560523578, "lo": -0.2525069975041002, "hi": 0.08280732629362865}, "damage": {"delta": -4.688898334630816, "lo": -16.579675528431228, "hi": 7.201878859169594}}, "horizon:D1:H4-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.44669671217723533, "lo": -0.6101335336904057, "hi": -0.283259890664065}, "damage": {"delta": -27.760663081910185, "lo": -40.632638245890675, "hi": -14.888687917929698}}, "horizon:D1:H6-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.6013426564089039, "lo": -0.7826336416323983, "hi": -0.42005167118540954}, "damage": {"delta": -42.204456189458966, "lo": -56.65409256782489, "hi": -27.754819811093043}}, "horizon:D2:H2-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": 0.1221562437917942, "lo": -0.03270686626038173, "hi": 0.27701935384397014}, "damage": {"delta": -1.3871765571301495, "lo": -12.65203794978981, "hi": 9.877684835529509}}, "horizon:D2:H4-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.010445205641988689, "lo": -0.17328466002056345, "hi": 0.15239424873658605}, "damage": {"delta": -11.428309448363414, "lo": -22.464098423319832, "hi": -0.3925204734069965}}, "horizon:D2:H6-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.04811748451261554, "lo": -0.22180976220358517, "hi": 0.12557479317835407}, "damage": {"delta": -13.768310563416314, "lo": -24.714989530099672, "hi": -2.821631596732958}}, "horizon:D3:H2-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.05354333448085182, "lo": -0.2199930586351637, "hi": 0.11290638967346003}, "damage": {"delta": -4.769236755994029, "lo": -15.266121457262415, "hi": 5.727647945274356}}, "horizon:D3:H4-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.147961002180563, "lo": -0.3401483562407635, "hi": 0.04422635187963753}, "damage": {"delta": -11.55434998358109, "lo": -22.956689710520973, "hi": -0.15201025664120849}}, "horizon:D3:H6-H1": {"method": "paired", "n_boots": 96, "stock": {"delta": -0.07381774836432718, "lo": -0.23193655971078636, "hi": 0.08430106298213198}, "damage": {"delta": -10.49017711161718, "lo": -21.414883288567946, "hi": 0.43452906533358515}}, "awr_actor_weighting": {"method": "independent", "runs": ["hwzv0k9a", "5wfk2esf"], "stock": {"delta": 0.07568118740602148, "lo": -0.10404662733610323, "hi": 0.2554090021481462}, "damage": {"delta": 10.79505930582518, "lo": -1.4089537332887385, "hi": 22.9990723449391}}, "exploration_modeling:XM K=8": {"method": "independent", "runs": ["feic18gt@shared_k", "cqbbbg77"], "stock": {"delta": -0.8074643747713892, "lo": -0.9825115399021882, "hi": -0.6324172096405902}, "damage": {"delta": -44.33370128083352, "lo": -55.41414939887559, "hi": -33.25325316279146}}, "windows_per_replay": {"method": "independent", "runs": ["shjnxxsu", "uw05bvm2"], "stock": {"delta": 0.7282775231496745, "lo": -0.07244833281289842, "hi": 1.5290033791122475}, "damage": {"delta": -6.22803789863525, "lo": -46.53127834784616, "hi": 34.07520255057566}}, "optimizer_small": {"method": "independent", "runs": ["vlim96s9", "o3nihah5"], "stock": {"delta": 0.030274837702697233, "lo": -0.7704510182598755, "hi": 0.83100069366527}, "damage": {"delta": -30.106985725507116, "lo": -70.41022617471802, "hi": 10.196254723703781}}, "batch:128-2048": {"method": "independent", "runs": ["nt4gh4mk", "rri7scjc"], "stock": {"delta": 0.4689792573245629, "lo": 0.2527729982533079, "hi": 0.6851855163958179}, "damage": {"delta": 27.92360026214979, "lo": 15.799860848955479, "hi": 40.047339675344105}}, "batch:256-2048": {"method": "independent", "runs": ["phxqvexa", "rri7scjc"], "stock": {"delta": 0.30429806864227293, "lo": 0.11356734982161071, "hi": 0.49502878746293516}, "damage": {"delta": 21.662850789805674, "lo": 9.409174553614342, "hi": 33.916527025997006}}, "batch:512-2048": {"method": "independent", "runs": ["980k3yla", "rri7scjc"], "stock": {"delta": 0.074714310182045, "lo": -0.1331723543620434, "hi": 0.2826009747261334}, "damage": {"delta": 11.872178396041477, "lo": -0.7747140806555155, "hi": 24.519070872738467}}, "batch:1024-2048": {"method": "independent", "runs": ["7ok39dju", "rri7scjc"], "stock": {"delta": 0.234892077938069, "lo": 0.04363067063627782, "hi": 0.4261534852398602}, "damage": {"delta": 7.608187000918335, "lo": -4.032334730553366, "hi": 19.248708732390035}}, "lr:0.028-0.014": {"method": "independent", "runs": ["wmwellfd", "9luwu85n"], "stock": {"delta": 0.16004583146173001, "lo": -0.025627432858824822, "hi": 0.3457190957822849}, "damage": {"delta": 9.385692319456425, "lo": -3.2112212115822363, "hi": 21.982605850495084}}, "lr:0.040-0.014": {"method": "independent", "runs": ["7hy8xjjh", "9luwu85n"], "stock": {"delta": 0.21825072960501024, "lo": 0.02036931461696123, "hi": 0.41613214459305925}, "damage": {"delta": 6.037815287038697, "lo": -6.702105880458486, "hi": 18.77773645453588}}, "lr:0.056-0.014": {"method": "independent", "runs": ["2rbed6ze", "9luwu85n"], "stock": {"delta": 0.13348081502051878, "lo": -0.05216742334036198, "hi": 0.3191290533813995}, "damage": {"delta": 2.9052148198165924, "lo": -9.592578052541997, "hi": 15.403007692175182}}, "lr:0.080-0.014": {"method": "independent", "runs": ["5adh9mtu", "9luwu85n"], "stock": {"delta": 0.04814737151056736, "lo": -0.14189352868134764, "hi": 0.23818827170248236}, "damage": {"delta": 2.2363985734250065, "lo": -11.007500685608923, "hi": 15.480297832458936}}, "lr:0.112-0.014": {"method": "independent", "runs": ["zerhe8au", "9luwu85n"], "stock": {"delta": 0.07982374518168855, "lo": -0.11808028271209917, "hi": 0.27772777307547625}, "damage": {"delta": -8.70738580292793, "lo": -20.5356947698586, "hi": 3.1209231640027397}}};
/* Fig: the legend for every comparison — baseline → treatment on the left, their difference with its 95% interval on the right, one strip. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var row = document.getElementById('comparison-key-row');
  var gloss = document.getElementById('comparison-key-glossary');
  if (!D || !CI || !row) return;
  var d = CI.codec_legacy_cartesian.stock;
  var ARMS = {base: {label: 'legacy codec', run: 'acbbeb76', score: -0.133}, treat: {label: 'Cartesian codec', run: 'cqbbbg77', score: 0.159}};
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.comparisonKey = function () {
    return [{baseline: ARMS.base.label, baseline_run: ARMS.base.run, baseline_stocks_per_min: ARMS.base.score, treatment: ARMS.treat.label, treatment_run: ARMS.treat.run, treatment_stocks_per_min: ARMS.treat.score, delta: d.delta, ci95_lo: d.lo, ci95_hi: d.hi, method: CI.codec_legacy_cartesian.method, boots: CI.codec_legacy_cartesian.n_boots}];
  };

  function arm(role, a, cls) {
    var el = document.createElement('div'); el.className = 'key-arm' + (cls ? ' ' + cls : '');
    el.innerHTML = '<span class="key-role">' + role + '</span><b>' + a.label + '</b><span class="key-score">' + D.signed(a.score, 2) + ' net stocks / min</span><span class="key-run">' + a.run + '</span>';
    return el;
  }
  var arrow = document.createElement('div'); arrow.className = 'key-arrow'; arrow.textContent = '→';
  var chart = document.createElement('div'); chart.className = 'key-chart';
  var svg = D.el('svg', {class: 'model-diagram reflow-chart', id: 'comparison-key-diagram', role: 'img', 'aria-label': 'The difference, plus 0.29 net stocks per minute, drawn as a dot with a horizontal 95% interval from plus 0.08 to plus 0.51 against a zero line.'});
  chart.appendChild(svg);
  row.appendChild(arm('baseline', ARMS.base, ''));
  row.appendChild(arrow);
  row.appendChild(arm('treatment', ARMS.treat, 'treat'));
  row.appendChild(chart);

  // the effect: zero line, interval, dot, value — labelled in place, no scale (the numbers are the point, not the axis)
  var W = 420, H = 128, y = 58;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var dom = D.domain([d]), x = D.scale([24, 286], dom.lo, dom.hi);
  dom.ticks.forEach(function (v) {
    svg.appendChild(D.el('line', {x1: x(v), y1: y - 22, x2: x(v), y2: y + 22, class: v === 0 ? 'effect-zero' : 'effect-grid'}));
  });
  D.effect(svg, x, y, d, {accent: true, digits: 2, valueX: 304});
  D.text(svg, 304, y + 18, 'stocks / min', {class: 'diagram-note'});
  D.text(svg, x(d.delta), y - 32, 'Δ = treatment − baseline', {class: 'diagram-label accent-text', 'text-anchor': 'middle'});
  D.text(svg, x(0), y + 56, '0 = no change', {class: 'diagram-note', 'text-anchor': 'middle'});
  svg.appendChild(D.el('path', {d: 'M ' + x(d.lo) + ' ' + (y + 26) + ' v 5 H ' + x(d.hi) + ' v -5', class: 'model-wire'}));
  D.text(svg, x(d.hi), y + 43, '95% interval · 96 paired boots', {class: 'diagram-note', 'text-anchor': 'end'});

  gloss.innerHTML = '<span><b>Read every comparison as baseline → treatment.</b></span><span>The dot is the change; the bar is its 95% interval from evaluation noise, not training seeds.</span><span>Unmatched or unfinished comparisons are labeled where they occur.</span>';
}());

/* Fig: validation NLL over gameplay, aligned on updates, for the O50 production run. */
(function () {
  'use strict';
  var D = window.HALDIAG, DATA = window.HALFIG_DATA;
  var svg = document.getElementById('training-curve-diagram');
  var readout = document.getElementById('training-curve-readout');
  if (!D || !DATA || !svg) return;
  var S = DATA.o50;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  var EV = S.evals, evalAt = {};
  EV.forEach(function (e) { evalAt[e.updates] = e; });
  window.HALFIG_CUSTOM.data.trainingCurve = function () {
    var byUpdate = {};
    S.val_updates.forEach(function (u, i) { byUpdate[u] = {run: S.run, updates: u, val_nll: S.val_nll[i]}; });
    EV.forEach(function (e) { byUpdate[e.updates] = Object.assign(byUpdate[e.updates] || {run: S.run, updates: e.updates}, {evaluation_id: e.evaluation_id, boots: e.boots, net_stocks_per_min: e.stock, stock_ci95_lo: e.stock_ci95[0], stock_ci95_hi: e.stock_ci95[1], net_damage_per_min: e.damage, damage_ci95_lo: e.damage_ci95[0], damage_ci95_hi: e.damage_ci95[1]}); });
    return Object.keys(byUpdate).map(Number).sort(function (a, b) { return a - b; }).map(function (u) { return byUpdate[u]; });
  };

  var W = 880, H = 400, L = 76, R = 24;
  var top = {y0: 44, y1: 168}, bot = {y0: 216, y1: 352};
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var xmax = S.val_updates[S.val_updates.length - 1];
  var x = D.scale([L, W - R], 0, xmax);
  var nllLo = 1.35, nllHi = 1.60, stLo = -0.4, stHi = 0.5;
  var yN = D.scale([top.y1, top.y0], nllLo, nllHi);
  var yS = D.scale([bot.y1, bot.y0], stLo, stHi);

  // the dip: play fell from the 24,576 checkpoint to 57,344 while the loss kept falling
  var dip = [24576, 57344];
  [top, bot].forEach(function (p) {
    svg.appendChild(D.el('rect', {x: x(dip[0]), y: p.y0, width: x(dip[1]) - x(dip[0]), height: p.y1 - p.y0, class: 'chart-band'}));
  });
  D.text(svg, (x(dip[0]) + x(dip[1])) / 2, bot.y1 + 18, 'play falls while the loss holds', {class: 'diagram-note', 'text-anchor': 'middle'});

  // axes
  function yAxis(p, y, ticks, fmt, label) {
    ticks.forEach(function (v) {
      svg.appendChild(D.el('line', {x1: L, y1: y(v), x2: W - R, y2: y(v), class: v === 0 ? 'effect-zero' : 'effect-grid'}));
      D.text(svg, L - 8, y(v) + 4, fmt(v), {class: 'diagram-note', 'text-anchor': 'end'});
    });
    D.text(svg, L, p.y0 - 12, label, {class: 'diagram-note'});
  }
  yAxis(top, yN, [1.4, 1.5, 1.6], function (v) { return v.toFixed(1); }, 'validation NLL · lower is better');
  yAxis(bot, yS, [-0.2, 0, 0.2, 0.4], function (v) { return D.signed(v, 1); }, 'net stocks / min · 96 two-minute episodes per checkpoint · dot and 95% interval');
  [0, 32768, 65536, 98304, 131072].forEach(function (u) {
    D.text(svg, x(u), bot.y1 + 34, u === 0 ? '0' : Math.round(u / 1000) + 'k', {class: 'diagram-note', 'text-anchor': 'middle'});
  });
  D.text(svg, W - R, bot.y1 + 48, 'optimizer updates', {class: 'diagram-note', 'text-anchor': 'end'});

  // series
  var nllPts = S.val_updates.map(function (u, i) { return x(u) + ',' + yN(S.val_nll[i]); });
  svg.appendChild(D.el('polyline', {points: nllPts.join(' '), class: 'chart-line'}));
  var stPts = EV.map(function (e) { return x(e.updates) + ',' + yS(e.stock); });
  svg.appendChild(D.el('polyline', {points: stPts.join(' '), class: 'effect-line accent'}));
  EV.forEach(function (e) {
    svg.appendChild(D.el('line', {x1: x(e.updates), y1: yS(e.stock_ci95[0]), x2: x(e.updates), y2: yS(e.stock_ci95[1]), class: 'effect-ci accent'}));
    svg.appendChild(D.el('circle', {cx: x(e.updates), cy: yS(e.stock), r: 3.6, class: 'effect-pt accent'}));
  });
  function mark(u, dy, anchor) { var e = evalAt[u]; D.text(svg, x(u) + (anchor === 'end' ? -8 : 0), yS(dy > 0 ? e.stock_ci95[0] : e.stock_ci95[1]) + dy, D.signed(e.stock, 2), {class: 'diagram-label' + (u === 131072 ? ' accent-text' : ''), 'text-anchor': anchor || 'middle'}); }
  mark(24576, -7); mark(57344, 15); mark(131072, -7, 'end');

  // hover: nearest validation point, plus the checkpoint if one sits there
  var cursor = D.el('line', {x1: 0, x2: 0, y1: top.y0, y2: bot.y1, class: 'chart-cursor', visibility: 'hidden'});
  svg.appendChild(cursor);
  function show(u) {
    var i = S.val_updates.indexOf(u), e = evalAt[u];
    cursor.setAttribute('x1', x(u)); cursor.setAttribute('x2', x(u)); cursor.setAttribute('visibility', 'visible');
    readout.innerHTML = '<span><b>' + u.toLocaleString('en-US') + '</b> updates</span><span>validation NLL <b>' + S.val_nll[i].toFixed(3) + '</b></span>' +
      (e ? '<span>net stocks / min <b>' + D.signed(e.stock, 2) + '</b> [' + D.signed(e.stock_ci95[0], 2) + ', ' + D.signed(e.stock_ci95[1], 2) + ']</span><span>net damage / min <b>' + D.signed(e.damage, 1) + '</b> [' + D.signed(e.damage_ci95[0], 1) + ', ' + D.signed(e.damage_ci95[1], 1) + ']</span>' : '<span>not played at this checkpoint</span>');
  }
  function selectPoint(e) {
    var pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    var p = pt.matrixTransform(svg.getScreenCTM().inverse());
    var best = S.val_updates[0], bd = Infinity;
    S.val_updates.forEach(function (u) { var d = Math.abs(x(u) - p.x); if (d < bd) { bd = d; best = u; } });
    show(best);
  }
  svg.addEventListener('pointerdown', selectPoint);
  svg.addEventListener('pointermove', function(e){if(e.pointerType==='mouse')selectPoint(e);});
  svg.addEventListener('mouseleave', function () { cursor.setAttribute('visibility', 'hidden'); readout.innerHTML = ''; });
  readout.innerHTML = '<span>hover for the values at each checkpoint</span>';
}());

/* The overview uses semantic HTML so each stage stays readable at narrow widths. */

/* Fig: the four O37 cells in the given→produced grammar, each cell's closed-loop result as its
   difference from the unconditioned cell with a 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('deps-diagram');
  if (!D || !CI || !svg) return;
  var FAINT = 'var(--ink-faint)';
  var mk = D.marker(svg, 'arrow', FAINT);
  svg.setAttribute('viewBox', '0 0 920 330');

  var CELLS = [
    {id: 'D0', name: 'nothing', across: false, within: false, stock: -0.702, damage: 2.77, run: '98r9smrj'},
    {id: 'D1', name: 'earlier groups', across: false, within: true, stock: -0.313, damage: 13.05, run: 'a117chkw'},
    {id: 'D2', name: 'earlier frames', across: true, within: false, stock: -0.196, damage: 18.06, run: '50q39o9j'},
    {id: 'D3', name: 'frames and groups', across: true, within: true, stock: -0.096, damage: 24.75, run: '5wfk2esf', cur: true}
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  CELLS.forEach(function (c) { c.vs = c.id === 'D0' ? null : CI['o37:' + c.id + '-D0']; });
  var MAIN = [['across frames', CI['o37:across']], ['within the frame', CI['o37:within']]];
  window.HALFIG_CUSTOM.data.o37Cells = function () {
    return CELLS.map(function (c) {
      var r = {o37_cell: c.id, conditioning: c.name, across_frames: c.across, within_frame: c.within, net_stocks_per_min: c.stock, net_damage_per_min: c.damage, run: c.run};
      if (c.vs) { r.stock_delta_vs_D0 = c.vs.stock.delta; r.stock_ci95_lo = c.vs.stock.lo; r.stock_ci95_hi = c.vs.stock.hi; r.damage_delta_vs_D0 = c.vs.damage.delta; r.damage_ci95_lo = c.vs.damage.lo; r.damage_ci95_hi = c.vs.damage.hi; r.method = c.vs.method; }
      return r;
    });
  };
  window.HALFIG_CUSTOM.data.o37MainEffects = function () {
    return MAIN.map(function (m) { return {conditioning: m[0], averaged_over: 'the other axis', method: m[1].method, stock_delta: m[1].stock.delta, stock_ci95_lo: m[1].stock.lo, stock_ci95_hi: m[1].stock.hi, damage_delta: m[1].damage.delta, damage_ci95_lo: m[1].damage.lo, damage_ci95_hi: m[1].damage.hi}; });
  };

  var W = 44, H = 28, GAP = 6;
  var X0 = 158;                  // arrow column
  var TOK = X0 + 56;             // first token column
  var GRP = 396;                 // the written frame, opened into its groups
  var COL = {stock: [604, 704], damage: [764, 864]};
  var y0 = 60, step = 62;

  D.text(svg, 16, 28, 'aₜ₊₂ conditioned on', {class: 'diagram-note'});
  D.text(svg, TOK, 28, 'across frames · given', {class: 'diagram-note'});
  D.text(svg, GRP, 28, 'within the frame · given', {class: 'diagram-note'});
  D.text(svg, (COL.stock[0] + COL.stock[1]) / 2, 28, 'Δ net stocks / min', {class: 'diagram-note', 'text-anchor': 'middle'});
  D.text(svg, (COL.damage[0] + COL.damage[1]) / 2, 28, 'Δ net damage / min', {class: 'diagram-note', 'text-anchor': 'middle'});
  D.text(svg, 908, 14, 'conditioned on nothing → each cell · 95% interval', {class: 'diagram-note', 'text-anchor': 'end'});

  var live = CELLS.filter(function (c) { return c.vs; });
  var sdom = D.domain(live.map(function (c) { return c.vs.stock; })), ddom = D.domain(live.map(function (c) { return c.vs.damage; }));
  var sx = D.scale(COL.stock, sdom.lo, sdom.hi), dx = D.scale(COL.damage, ddom.lo, ddom.hi);
  var yEnd = y0 + 3 * step + H + 8;
  D.effectAxis(svg, sx, sdom, y0 - 8, yEnd);
  D.effectAxis(svg, dx, ddom, y0 - 8, yEnd);

  CELLS.forEach(function (c, i) {
    var y = y0 + i * step;
    D.text(svg, 16, y + H / 2 + 4, c.name, {class: 'diagram-title' + (c.cur ? ' accent' : '')});
    // what the decoder is handed
    D.wire(svg, X0, y + H / 2, X0 + 48, y + H / 2, mk);
    if (i === 0) D.text(svg, X0 + 18, y + H / 2 - 6, 'hₜ', {class: 'diagram-label'});
    var given = c.across ? ['aₜ', 'aₜ₊₁'] : [];
    given.forEach(function (g, k) { D.token(svg, TOK + k * (W + GAP), y, W, H, g, 'given'); });
    D.token(svg, TOK + 2 * (W + GAP), y, W, H, 'aₜ₊₂');
    // the written frame, opened into its groups
    D.wire(svg, TOK + 3 * (W + GAP) - 2, y + H / 2, GRP - 8, y + H / 2, mk);
    ['C', 'M', 'T', 'B'].forEach(function (g, k) {
      var x = GRP + k * (k === 3 && c.within ? 52 : 38);
      var last = k === 3;
      D.token(svg, x, y, 30, H, g, c.within && !last ? 'given' : '');
    });
    if (c.within) D.wire(svg, GRP + 3 * 38 + 2, y + H / 2, GRP + 3 * 52 - 4, y + H / 2, mk);
    // results: difference from the unconditioned cell
    if (!c.vs) {
      D.text(svg, sx(0) + 6, y + H / 2 + 4, 'baseline', {class: 'diagram-note'});
      D.text(svg, dx(0) + 6, y + H / 2 + 4, 'baseline', {class: 'diagram-note'});
      return;
    }
    D.effect(svg, sx, y + H / 2, c.vs.stock, {accent: c.cur, digits: 2, valueX: COL.stock[1] + 8});
    D.effect(svg, dx, y + H / 2, c.vs.damage, {accent: c.cur, digits: 1, valueX: COL.damage[1] + 8});
  });
}());

/* Fig: the decoder Transformer over one action chunk — one token per future frame, causal
   self-attention over the earlier tokens, cross-attention to the trunk states, and the
   written frame feeding the next token (experiments/059 CausalTemporalDecoder). */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('chunk-decoder-diagram');
  if (!D || !svg) return;

  var FAINT = 'var(--ink-faint)', ACCENT = 'var(--oxblood)';
  svg.setAttribute('viewBox', '0 0 900 560');
  var mk = D.marker(svg, 'arrow', FAINT), mka = D.marker(svg, 'arrow-accent', ACCENT);
  function dot(target, x, y, accent) { target.appendChild(D.el('circle', {cx: x, cy: y, r: 2.4, fill: accent ? ACCENT : FAINT})); }
  function op(target, cx, y, w, h, label, cls) {
    target.appendChild(D.el('rect', {x: cx - w / 2, y: y, width: w, height: h, class: 'model-token' + (cls ? ' ' + cls : '')}));
    D.text(target, cx, y + h / 2 + 4, label, {'text-anchor': 'middle', class: 'diagram-label'});
  }
  function rail(y, lines) {
    lines.forEach(function (line, i) { D.note(svg, 168, y + i * 13, line, 'end'); });
  }

  // — trunk states over the context, ending in hₜ
  var trunk = D.part(svg, 'trunk');
  D.note(trunk, 16, 14, 'trunk states over the context, one per frame');
  var STATES = [{x: 16, label: 'hₜ₋L', cls: 'given'}, {x: 118, label: 'hₜ₋₂', cls: 'given'}, {x: 172, label: 'hₜ₋₁', cls: 'given'}, {x: 226, label: 'hₜ', cls: 'played'}];
  STATES.forEach(function (s) { D.token(trunk, s.x, 28, 48, 28, s.label, s.cls); });
  D.note(trunk, 91, 46, '···', 'middle');

  // — columns: one decoder token per future frame
  var CX = [258, 388, 518, 648, 816], W = 108;
  var STEP = [
    {k: '1', prev: 'e(aₜ₋₁)', prevGiven: true, writes: 'aₜ'},
    {k: '2', prev: 'e(aₜ)', writes: 'aₜ₊₁'},
    {k: '3', prev: 'e(aₜ₊₁)', writes: 'aₜ₊₂'},
    {k: '4', prev: 'e(aₜ₊₂)', writes: 'aₜ₊₃'},
    {k: '28', prev: 'e(aₜ₊₂₆)', writes: 'aₜ₊₂₇'}
  ];
  var YTOK = 96, YLIN = 146, YATT = 200, YXATT = 258, YKV = 244, YBLK = 328, YNORM = 380, YHEAD = 418, YOUT = 470;
  D.note(svg, 874, 14, 'one decoder token per future frame, offset k', 'end');
  D.note(svg, 738, YTOK + 18, '···', 'middle');

  // hₜ bus: the same trunk state enters every token
  var hbus = D.part(svg, 'hbus');
  D.path(hbus, 'M 274 42 L 292 42 L 292 76 L ' + (CX[4] + 15) + ' 76', null, {accent: true});
  D.note(hbus, 300, 72, 'hₜ', 'start', 'play');

  // K, V: every trunk state, projected once, shared by every step's cross-attention
  var kv = D.part(svg, 'kv');
  D.wire(kv, 16, 62, 274, 62);
  D.note(kv, 16, 76, 'every trunk state → K, V');
  D.wire(kv, 190, 62, 190, 170);
  op(kv, 190, 170, 52, 20, 'linear');
  D.wire(kv, 190, 190, 190, YKV);
  D.path(kv, 'M 190 ' + YKV + ' L ' + (CX[4] + 40) + ' ' + YKV, null, {ghost: true});

  rail(YTOK + 12, ['previous frame,', 'hₜ, offset k']);
  rail(YLIN + 14, ['token']);
  rail(YATT + 12, ['causal attention', 'over earlier tokens']);
  rail(YXATT + 12, ['cross-attention', 'to trunk states']);
  rail(YBLK + 19, ['more causal blocks']);
  rail(YNORM + 14, ['frame state sₖ']);
  rail(YHEAD + 19, ['four heads, in order']);
  rail(YOUT + 18, ['written frame']);

  STEP.forEach(function (s, i) {
    var cx = CX[i], col = D.part(svg, 'step' + i);
    // the token: [ e(aₖ₋₁) | hₜ | k ] → linear → norm
    col.appendChild(D.el('rect', {x: cx - 54, y: YTOK, width: W, height: 28, class: 'model-token'}));
    D.wire(col, cx + 2, YTOK, cx + 2, YTOK + 28, null, {ghost: true});
    D.wire(col, cx + 32, YTOK, cx + 32, YTOK + 28, null, {ghost: true});
    D.text(col, cx - 26, YTOK + 18, s.prev, {'text-anchor': 'middle', class: 'diagram-label' + (s.prevGiven ? ' given-label' : '')});
    D.text(col, cx + 17, YTOK + 18, 'hₜ', {'text-anchor': 'middle', class: 'diagram-label'});
    D.text(col, cx + 43, YTOK + 18, s.k, {'text-anchor': 'middle', class: 'diagram-label'});
    dot(hbus, cx + 15, 76, true);
    D.wire(hbus, cx + 15, 76, cx + 15, YTOK - 2, mka, {accent: true});
    D.wire(col, cx, YTOK + 28, cx, YLIN - 2, mk);
    op(col, cx, YLIN, 96, 20, 'linear · norm');
    D.wire(col, cx, YLIN + 20, cx, YATT - 2, mk);
    // block 1: causal self-attention + MLP over the chunk tokens so far
    op(col, cx, YATT, W, 30, 'attention · MLP');
    D.wire(col, cx, YATT + 30, cx, YXATT - 2, mk);
    // history cross-attention: queries from the token, keys and values from the trunk states
    op(col, cx, YXATT, W, 30, 'cross-attention');
    dot(kv, cx + 32, YKV);
    D.wire(kv, cx + 32, YKV, cx + 32, YXATT - 2, mk);
    D.wire(col, cx, YXATT + 30, cx, YBLK - 2, mk);
    // the remaining causal blocks
    [8, 4].forEach(function (o) { col.appendChild(D.el('rect', {x: cx - 54 - o, y: YBLK + o, width: W, height: 30, class: 'model-layer'})); });
    op(col, cx, YBLK, W, 30, 'attention · MLP');
    D.wire(col, cx, YBLK + 30, cx, YNORM - 2, mk);
    op(col, cx, YNORM, 60, 20, 'norm');
    D.wire(col, cx, YNORM + 20, cx, YHEAD - 2, mk);
    // the four heads of Fig. 6 write this frame
    col.appendChild(D.el('rect', {x: cx - 54, y: YHEAD, width: W, height: 32, class: 'model-token queued'}));
    D.text(col, cx, YHEAD + 13, 'four heads', {'text-anchor': 'middle', class: 'diagram-label'});
    D.text(col, cx, YHEAD + 26, 'in order', {'text-anchor': 'middle', class: 'diagram-label'});
    D.wire(col, cx, YHEAD + 32, cx, YOUT - 2, mk);
    D.token(col, cx - 54, YOUT, W, 28, s.writes, 'played');
    // the written frame embeds into the next token
    var back = D.part(svg, 'back' + i);
    var gx = cx + 66, ny = YTOK + 14;
    if (i < 3) {
      D.path(back, 'M ' + (cx + 54) + ' ' + (YOUT + 14) + ' L ' + gx + ' ' + (YOUT + 14) + ' L ' + gx + ' ' + ny + ' L ' + (CX[i + 1] - 56) + ' ' + ny, mka, {accent: true});
    } else if (i === 3) {
      D.path(back, 'M ' + (cx + 54) + ' ' + (YOUT + 14) + ' L ' + gx + ' ' + (YOUT + 14) + ' L ' + gx + ' ' + ny + ' L ' + (gx + 12) + ' ' + ny, null, {accent: true});
    } else {
      D.wire(back, cx - 66, ny, cx - 56, ny, mka, {accent: true});
    }
  });
  // step 4 attends to steps 1–3; every step attends only to the steps before it
  var att = D.part(svg, 'attn');
  [0, 1, 2].forEach(function (i) {
    var x1 = CX[i] + 20, x2 = CX[3] - 20, lift = 10 + (3 - i) * 5;
    D.path(att, 'M ' + x1 + ' ' + YATT + ' Q ' + ((x1 + x2) / 2) + ' ' + (YATT - lift) + ' ' + x2 + ' ' + YATT, mk, {ghost: true});
  });

  D.note(svg, 204, 522, 'train: every step at once, teacher-forced with the true prefix');
  D.note(svg, 204, 536, 'play: one step at a time, keys and values cached');
}());

/* Fig: one written frame opened up — the token strip, then the four heads in order with FiLM,
   the SiLU MLP head and the trunk skip — and what each of the three parts was worth (O43). */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('logit-anatomy');
  var forest = document.getElementById('logit-forest');
  if (!D || !svg || !forest) return;

  // baseline → treatment on both evaluation suites; the baseline of each row is O43 without that part
  var ROWS = [
    {comp: 'film', label: 'zero-init FiLM → ordinary init', sub: 'vzd242t6 → 1imfy8v3 · 4,096 updates',
      char: {delta: 1.09, lo: 0.75, hi: 1.43}, fox: {delta: 1.20, lo: 0.83, hi: 1.57}},
    {comp: 'skip', label: 'no trunk skip → trunk skip', sub: 'j4skyy83 → 1imfy8v3 · 4,096 updates',
      char: {delta: 0.89, lo: 0.49, hi: 1.29}, fox: {delta: 1.13, lo: 0.67, hi: 1.59}},
    {comp: 'head', label: 'linear head → SiLU MLP head', sub: 'apj3rw8b → 1imfy8v3 · 8,192 updates',
      char: {delta: 2.03, lo: 1.75, hi: 2.31}, fox: {delta: 2.41, lo: 2.09, hi: 2.73}}
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.logitPathRows = function () {
    return ROWS.map(function (r) {
      return {comparison: r.label, runs: r.sub.split(' · ')[0], updates: r.sub.split(' · ')[1],
        character_suite_delta: r.char.delta, character_suite_ci95_lo: r.char.lo, character_suite_ci95_hi: r.char.hi,
        fox_suite_delta: r.fox.delta, fox_suite_ci95_lo: r.fox.lo, fox_suite_ci95_hi: r.fox.hi};
    });
  };

  var FAINT = 'var(--ink-faint)', ACCENT = 'var(--oxblood)';

  /* ------------------------------------------------------------- anatomy */
  svg.setAttribute('viewBox', '0 0 900 560');
  var mk = D.marker(svg, 'arrow', FAINT), mka = D.marker(svg, 'arrow-accent', ACCENT);
  function dot(target, x, y) { target.appendChild(D.el('circle', {cx: x, cy: y, r: 2.4, fill: FAINT})); }
  function op(target, cx, y, w, label, cls) {
    target.appendChild(D.el('rect', {x: cx - w / 2, y: y, width: w, height: 20, class: 'model-token' + (cls ? ' ' + cls : '')}));
    D.text(target, cx, y + 14, label, {'text-anchor': 'middle', class: 'diagram-label'});
  }

  // — the token strip: hₜ and one token per future offset
  var strip = D.part(svg, 'strip');
  D.note(strip, 120, 18, 'one token per future frame · two causal attention blocks over them');
  D.token(strip, 24, 44, 56, 30, 'hₜ', 'given');
  D.note(strip, 52, 38, 'trunk state', 'middle');
  D.wire(strip, 80, 59, 118, 59, mk);
  var TOK = [
    {x: 120, label: '(hₜ, aₜ₋₁, 1)', writes: 'writes aₜ'},
    {x: 242, label: '(hₜ, aₜ, 2)', writes: 'writes aₜ₊₁', open: true},
    {x: 364, label: '(hₜ, aₜ₊₁, 3)', writes: 'writes aₜ₊₂'},
    {x: 486, label: '(hₜ, aₜ₊₂, 4)', writes: 'writes aₜ₊₃'},
    {x: 650, label: '(hₜ, aₜ₊₁₉, 20)', writes: 'writes aₜ₊₂₀'}
  ];
  TOK.forEach(function (t, i) {
    D.token(strip, t.x, 44, 112, 30, t.label, t.open ? 'played' : '');
    D.note(strip, t.x + 56, 92, t.writes, 'middle', t.open ? 'play' : '');
    if (i > 0 && i < 4) D.wire(strip, TOK[i - 1].x + 112, 59, t.x - 2, 59, mk);
  });
  D.note(strip, 624, 63, '···', 'middle');
  // causal attention: every token attends to the tokens before it
  [0, 1, 2].forEach(function (i) {
    var x1 = TOK[i].x + 56, x2 = TOK[3].x + 56, lift = 12 + (3 - i) * 6;
    D.path(strip, 'M ' + x1 + ' 44 Q ' + ((x1 + x2) / 2) + ' ' + (44 - lift) + ' ' + x2 + ' 44', mk, {ghost: true});
  });
  D.note(strip, 706, 112, 'attends only to earlier tokens', 'middle');

  // — the opened frame: frame state bus and trunk bus
  var opened = D.part(svg, 'frame');
  D.wire(opened, 298, 74, 298, 150, mka, {accent: true});
  D.note(opened, 306, 112, 'attention → norm → frame state s₂', 'start', 'play');
  var CX = [160, 340, 520, 700];
  var YS = 150, YH = 172, YF = 196, YM = 236, YSUM = 380, YL = 402, YSAMPLE = 436, YE = 470, YP = 516;
  D.wire(opened, 100, YS, 790, YS, null, {accent: true});
  D.note(opened, 96, YS + 4, 's₂', 'end', 'play');
  var skipbus = D.part(svg, 'skip');
  D.wire(skipbus, 52, 74, 52, YH);
  D.wire(skipbus, 52, YH, 790, YH);
  D.note(skipbus, 60, YH + 14, 'hₜ skip bus');
  var picksbus = D.part(svg, 'picks');
  D.wire(picksbus, 140, YP, 790, YP, mk);
  D.note(picksbus, 800, YP + 4, 'aₜ₊₁ → token 3');
  D.note(picksbus, 140, YP + 16, 'picks so far, concatenated');

  var GROUPS = ['C-stick', 'main stick', 'triggers', 'buttons'];
  GROUPS.forEach(function (name, j) {
    var cx = CX[j];
    D.text(svg, cx, 138, (j + 1) + '. ' + name, {'text-anchor': 'middle', class: 'diagram-title'});
    // frame state into the column
    var col = D.part(svg, 'col' + j);
    dot(col, cx, YS);
    if (j === 0) {
      D.wire(col, cx, YS, cx, YM + 8, mk);
      D.note(col, cx - 8, YF + 14, 'first: no FiLM', 'end');
    } else {
      D.wire(col, cx, YS, cx, YF, mk);
      var film = D.part(svg, 'film');
      op(film, cx, YF, 84, 'FiLM  γ, β');
      // γ, β come from the picks made earlier in this frame
      var tx = cx - 72;
      dot(film, tx, YP);
      D.path(film, 'M ' + tx + ' ' + YP + ' L ' + tx + ' ' + (YF + 10) + ' L ' + (cx - 44) + ' ' + (YF + 10), mk);
      D.wire(col, cx, YF + 20, cx, YM + 8, mk);
    }
    // the head: norm → linear → SiLU → linear
    var head = D.part(svg, 'head');
    head.appendChild(D.el('rect', {x: cx - 44, y: YM, width: 88, height: 130, class: 'model-token queued'}));
    if (j === 0) {
      ['norm', 'linear', 'SiLU', 'linear'].forEach(function (label, i) { op(head, cx, YM + 8 + i * 26, 64, label); });
    } else {
      op(head, cx, YM + 50, 78, 'SiLU MLP');
      D.note(head, cx, YM + 85, 'same structure', 'middle');
    }
    D.note(head, cx, YM + 121, 'SiLU MLP head', 'middle');
    D.wire(col, cx, YM + 130, cx, YSUM - 10, mk);
    // Σ with the trunk skip from the right
    col.appendChild(D.el('circle', {cx: cx, cy: YSUM, r: 9, class: 'model-token'}));
    D.text(col, cx, YSUM + 4, '+', {'text-anchor': 'middle', class: 'diagram-label'});
    var sx = cx + 72;
    dot(skipbus, sx, YH);
    D.wire(skipbus, sx, YH, sx, 300);
    op(skipbus, sx, 300, 52, 'linear');
    D.path(skipbus, 'M ' + sx + ' 320 L ' + sx + ' ' + YSUM + ' L ' + (cx + 10) + ' ' + YSUM, mk);
    // logits → sample → embed → picks bus
    D.wire(col, cx, YSUM + 9, cx, YL, mk);
    op(col, cx, YL, 84, 'logits · mask');
    D.wire(col, cx, YL + 20, cx, YSAMPLE, mk);
    op(col, cx, YSAMPLE, 84, 'sample');
    D.wire(col, cx, YSAMPLE + 20, cx, YE, mk);
    op(col, cx, YE, 84, 'embed');
    D.wire(col, cx, YE + 20, cx, YP);
    dot(picksbus, cx, YP);
  });
  D.note(svg, CX[3] + 104, 314, 'trunk skip');

  /* ------------------------------------------------------------- forest */
  var W = 900, top = 68, rowH = 54, L = 16, COLX = [430, 690], VALX = 704;
  forest.setAttribute('viewBox', '0 0 ' + W + ' ' + (top + ROWS.length * rowH + 40));
  var dom = D.domain(ROWS.map(function (r) { return r.char; }).concat(ROWS.map(function (r) { return r.fox; })));
  var x = D.scale(COLX, dom.lo, dom.hi);
  D.effectAxis(forest, x, dom, top - 14, top + ROWS.length * rowH - 8, 'Δ net stocks / min');
  var lg = top - 40;
  D.note(forest, L, lg, 'baseline → treatment · 95% interval · each baseline is O43 without that part');
  forest.appendChild(D.el('circle', {cx: 690, cy: lg - 4, r: 3.2, class: 'effect-pt'}));
  D.note(forest, 698, lg, 'character suite');
  forest.appendChild(D.el('circle', {cx: 820, cy: lg - 4, r: 3.2, class: 'effect-pt accent'}));
  D.note(forest, 828, lg, 'Fox suite');
  ROWS.forEach(function (r, i) {
    var y = top + i * rowH + rowH / 2 - 6;
    var g = D.el('g', {class: 'forest-row'});
    g.appendChild(D.el('rect', {x: 0, y: top + i * rowH - 4, width: W, height: rowH, class: 'hit'}));
    D.text(g, L, y, r.label, {class: 'diagram-title'});
    D.text(g, L, y + 14, r.sub.split(' · ')[1], {class: 'diagram-note'});
    D.effect(g, x, y - 7, r.char, {digits: 2, valueX: VALX});
    D.effect(g, x, y + 9, r.fox, {digits: 2, valueX: VALX, accent: true});
    g.setAttribute('tabindex', '0');
    g.setAttribute('role', 'button');
    g.setAttribute('aria-label', r.label);
    g.addEventListener('click', function () { D.focus(svg, r.comp); });
    g.addEventListener('focus', function () { D.focus(svg, r.comp); });
    g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); D.focus(svg, r.comp); } });
    g.addEventListener('mouseenter', function () { D.focus(svg, r.comp); });
    g.addEventListener('mouseleave', function () { D.focus(svg, null); });
    forest.appendChild(g);
  });
  D.note(forest, L, top + ROWS.length * rowH + 30, 'select a row to highlight the part it added');
}());

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

/* Fig: the observation token, opened up — every source, the two shared encoders, one tuple. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('obs-diagram');
  if (!D || !svg) return;
  var FAINT = 'var(--ink-faint)';
  var mk = D.marker(svg, 'arrow', FAINT);
  svg.setAttribute('viewBox', '0 0 880 500');

  var SW = 196, SH = 28, STEP = 36;       // source boxes
  var EX = 270, EW = 190;                 // encoder column
  var TX = 720, TW = 150, TY = 220;       // the token

  function source(y, label, optional, withMask) {
    // optional sources are dashed; every player block travels with its validity mask
    var w = withMask ? SW - 46 : SW;
    svg.appendChild(D.el('rect', {x: 16, y: y, width: w, height: SH, class: 'model-token' + (optional ? ' queued' : '')}));
    D.text(svg, 28, y + 18, label, {class: 'diagram-label'});
    if (withMask) {
      svg.appendChild(D.el('rect', {x: 16 + w + 6, y: y, width: 40, height: SH, class: 'model-token'}));
      D.text(svg, 16 + w + 26, y + 18, 'mask', {'text-anchor': 'middle', class: 'diagram-label given-label'});
    }
  }
  var BUS = 626;                          // wires run straight to this column, then bend once into the token
  function into(x1, y1, x2, y2) {
    D.path(svg, 'M ' + x1 + ' ' + y1 + ' L ' + BUS + ' ' + y1 + ' C ' + (BUS + 70) + ' ' + y1 + ', ' + (x2 - 70) + ' ' + y2 + ', ' + x2 + ' ' + y2, mk);
  }

  // players → one shared encoder
  var py = 20;
  ['ego', 'ego follower', 'opponent follower', 'opponent'].forEach(function (l, i) {
    var y = py + i * STEP;
    source(y, l, l.indexOf('follower') !== -1, true);
    D.wire(svg, 16 + SW + 6, y + SH / 2, EX - 6, y + SH / 2, mk);
  });
  D.box(svg, EX, py, EW, 3 * STEP + SH, 'PLAYER ENCODER');

  // projectiles → one shared encoder → Σ
  var qy = py + 4 * STEP + 22;
  ['projectile', 'projectile', 'projectile', 'projectile'].forEach(function (l, i) {
    var y = qy + i * STEP;
    source(y, l, true, false);
    D.wire(svg, 16 + SW + 6, y + SH / 2, EX - 6, y + SH / 2, mk);
  });
  D.box(svg, EX, qy, EW, 3 * STEP + SH, 'PROJECTILE ENCODER');
  var sy = qy + (3 * STEP + SH) / 2;
  D.wire(svg, EX + EW + 6, sy, EX + EW + 24, sy, mk);
  svg.appendChild(D.el('rect', {x: EX + EW + 30, y: sy - 14, width: 76, height: SH, class: 'model-token'}));
  D.text(svg, EX + EW + 68, sy + 4, '× present', {'text-anchor': 'middle', class: 'diagram-label'});
  D.wire(svg, EX + EW + 112, sy, EX + EW + 130, sy, mk);
  svg.appendChild(D.el('circle', {cx: EX + EW + 148, cy: sy, r: 13, class: 'model-box'}));
  D.text(svg, EX + EW + 148, sy + 4.5, 'Σ', {'text-anchor': 'middle', class: 'diagram-title'});

  // conditioning
  var cy = qy + 4 * STEP + 22;
  source(cy, 'characters · stage');
  source(cy + STEP, 'player identity');

  // the action half: the previous frame, as the decoder's classes
  var ay = cy + 2 * STEP + 22;
  source(ay, 'aₜ₋₁');
  D.wire(svg, 16 + SW + 6, ay + SH / 2, EX - 6, ay + SH / 2, mk);
  svg.appendChild(D.el('rect', {x: EX, y: ay, width: 110, height: SH, class: 'model-token'}));
  D.text(svg, EX + 55, ay + 18, 'classes', {'text-anchor': 'middle', class: 'diagram-label'});
  D.wire(svg, EX + 116, ay + SH / 2, EX + 150, ay + SH / 2, mk);
  svg.appendChild(D.el('rect', {x: EX + 156, y: ay, width: 160, height: SH, class: 'model-token'}));
  D.text(svg, EX + 236, ay + 18, 'embedding + value', {'text-anchor': 'middle', class: 'diagram-label'});

  // everything joins the token
  var tokenMid = TY + 17;
  into(EX + EW + 6, py + (3 * STEP + SH) / 2, TX - 6, tokenMid - 12);
  into(EX + EW + 162, sy, TX - 6, tokenMid - 4);
  into(16 + SW + 6, cy + SH / 2, TX - 6, tokenMid + 4);
  into(16 + SW + 6, cy + STEP + SH / 2, TX - 6, tokenMid + 10);
  D.path(svg, 'M ' + (EX + 322) + ' ' + (ay + SH / 2) + ' L ' + BUS + ' ' + (ay + SH / 2) + ' C ' + (BUS + 70) + ' ' + (ay + SH / 2) + ', ' + (TX - 70) + ' ' + (tokenMid + 16) + ', ' + (TX - 6) + ' ' + (tokenMid + 16), mk);
  D.note(svg, TX + TW / 2, TY - 12, 'concatenate', 'middle');
  D.jointToken(svg, TX, TY, TW, 34, 'sₜ', 'aₜ₋₁');
  svg.appendChild(D.el('rect', {x: 16, y: 478, width: 26, height: 14, class: 'model-token queued'}));
  D.note(svg, 50, 489, 'may be absent · absent block: values zero, mask set · empty slot: gate zero');

  // the matched difference for the projectile slots
  var CI = window.HALFIG_CI;
  if (CI && CI.projectile_inputs) {
    D.effectBars(svg, [Object.assign({label: 'projectiles', sub: 'off → on · O41'}, CI.projectile_inputs)],
      {x: 716, y: 300, panelW: 160, panelH: 180, barW: 34, axisW: 44, metrics: ['stock']});
  }
}());

/* Fig: reward → return → weight for one game, with the AWR knobs live. Data: figures_src/awr-ledger.json (inlined at build). */
(function () {
  'use strict';
  var DATA = window.HALFIG_AWR_LEDGER;
  var svg = document.getElementById('ledger-svg');
  if (!DATA || !svg) return;
  var NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, String(attrs[k])); });
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function unrle(pairs) {
    var out = [];
    pairs.forEach(function (p) { for (var i = 0; i < p[1]; i++) out.push(p[0]); });
    return out;
  }
  var FPS = DATA.fps || 60;
  var games = DATA.games.map(function (g) {
    return {label: g.label, p1: g.p1, p2: g.p2, stage: g.stage, n: g.frames,
      pct: [unrle(g.p1Percent).map(function (v) { return v / 10; }), unrle(g.p2Percent).map(function (v) { return v / 10; })],
      stock: [unrle(g.p1Stock), unrle(g.p2Stock)]};
  });
  var O52 = {hl: 8, beta: 199.5, cap: 3.5, stock: 120, win: 50, damage: 1};
  var state = {game: 0, ego: 1, base: 'ema', hl: 8, beta: 199.5, cap: 3.5, frame: null};

  var gamesEl = document.getElementById('ledger-games');
  games.forEach(function (g, i) {
    var b = document.createElement('button');
    b.type = 'button'; b.textContent = g.p1 + ' vs ' + g.p2; b.title = g.label + ' · ' + g.stage;
    b.setAttribute('aria-pressed', String(i === 0));
    b.addEventListener('click', function () { state.game = i; state.frame = null; sync(); });
    gamesEl.appendChild(b);
  });
  function bindSeg(id, key, parse) {
    var host = document.getElementById(id);
    host.querySelectorAll('button').forEach(function (b) {
      b.addEventListener('click', function () { state[key] = parse(b.dataset[key]); sync(); });
    });
  }
  bindSeg('ledger-ego', 'ego', Number);
  bindSeg('ledger-base', 'base', String);
  var hl = document.getElementById('ledger-hl'), beta = document.getElementById('ledger-beta'), cap = document.getElementById('ledger-cap');
  hl.addEventListener('input', function () { state.hl = Number(hl.value); render(); });
  beta.addEventListener('input', function () { state.beta = Math.pow(10, Number(beta.value)); render(); });
  cap.addEventListener('input', function () { state.cap = Number(cap.value); render(); });
  document.getElementById('ledger-reset').addEventListener('click', function () {
    state.hl = O52.hl; state.beta = O52.beta; state.cap = O52.cap;
    hl.value = O52.hl; beta.value = Math.log10(O52.beta); cap.value = O52.cap;
    render();
  });

  /* Reward parameters follow O52; the baseline and unnormalized weights are illustrative. */
  function compute() {
    var g = games[state.game];
    var e = state.ego - 1, o = 1 - e;
    var n = g.n, gamma = Math.pow(0.5, 1 / (state.hl * FPS));
    var r = new Float64Array(n), dealt = new Float64Array(n), taken = new Float64Array(n);
    var deathOpp = new Uint8Array(n), deathEgo = new Uint8Array(n);
    for (var t = 1; t < n; t++) {
      var dd = g.pct[o][t] - g.pct[o][t - 1], dt = g.pct[e][t] - g.pct[e][t - 1];
      dealt[t] = dd > 0 ? dd : 0; taken[t] = dt > 0 ? dt : 0;
      deathOpp[t] = g.stock[o][t] < g.stock[o][t - 1] ? 1 : 0;
      deathEgo[t] = g.stock[e][t] < g.stock[e][t - 1] ? 1 : 0;
      var winOpp = deathOpp[t] && g.stock[o][t] === 0 ? 1 : 0, winEgo = deathEgo[t] && g.stock[e][t] === 0 ? 1 : 0;
      r[t] = O52.damage * (dealt[t] - taken[t]) + O52.stock * (deathOpp[t] - deathEgo[t]) + O52.win * (winOpp - winEgo);
    }
    var G = new Float64Array(n), acc = 0;
    for (t = n - 1; t >= 0; t--) { acc = r[t] + gamma * acc; G[t] = acc; }
    var B = new Float64Array(n);
    if (state.base === 'mean') {
      var m = 0; for (t = 0; t < n; t++) m += G[t]; m /= n;
      B.fill(m);
    } else {
      var a = 1 - Math.pow(0.5, 1 / (10 * FPS)), mm = G[0];
      for (t = 0; t < n; t++) { B[t] = mm; mm += a * (G[t] - mm); }
    }
    var w = new Float64Array(n), clipped = 0;
    for (t = 0; t < n; t++) { var raw = Math.exp((G[t] - B[t]) / state.beta); w[t] = Math.min(raw, state.cap); if (raw >= state.cap) clipped++; }
    return {g: g, n: n, gamma: gamma, r: r, dealt: dealt, taken: taken, deathOpp: deathOpp, deathEgo: deathEgo, G: G, B: B, w: w, clipped: clipped, e: e, o: o};
  }
  document.getElementById('ledger-full').addEventListener('change', render);
  var last = null;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.awrLedgerRows = function () {
    if (!last) return [];
    var rows = [];
    for (var t = 0; t < last.n; t += 60) rows.push({second: t / FPS, ego_percent: last.g.pct[last.e][t], opp_percent: last.g.pct[last.o][t], ego_stocks: last.g.stock[last.e][t], opp_stocks: last.g.stock[last.o][t], reward: +last.r[t].toFixed(2), G: +last.G[t].toFixed(2), baseline: +last.B[t].toFixed(2), w: +last.w[t].toFixed(3)});
    return rows;
  };

  /* ---- drawing ---- */
  var W = 880, L = 60, R = 16, TOP = 22;
  var strips = [['events', 70], ['return G', 100], ['illustrative weight · log scale', 100]];
  var H = TOP + strips.reduce(function (s, x) { return s + x[1] + 22; }, 0) + 16;
  var plotStart = 0, plotEnd = 1;
  function X(t, n) { return L + ((t - plotStart) / (plotEnd - plotStart)) * (W - L - R); }

  function render() {
    var c = compute(); last = c;
    W = window.matchMedia('(max-width:760px)').matches ? 340 : 880;
    var event = c.deathEgo.findIndex(function(v){return v;});
    if (event < 0) event = c.deathOpp.findIndex(function(v){return v;});
    if (event < 0) event = Math.floor(c.n/2);
    var full = document.getElementById('ledger-full').checked;
    plotStart = full ? 0 : Math.max(0,event-12*FPS);
    plotEnd = full ? c.n-1 : Math.min(c.n-1,event+6*FPS);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var n = c.n, y = TOP;
    // time axis ticks every 30 s
    var seconds = plotEnd / FPS, tickStep = full ? 30 : 5;
    for (var s = Math.ceil(plotStart/FPS/tickStep)*tickStep; s <= seconds; s += tickStep) {
      var x = X(s * FPS, n);
      svg.appendChild(el('line', {x1: x, y1: TOP, x2: x, y2: H - 16, class: 'grid'}));
      svg.appendChild(el('text', {x: x, y: H - 4, class: 'tick', 'text-anchor': 'middle'}, s + ' s'));
    }
    // strip 1: events
    var evH = strips[0][1];
    svg.appendChild(el('text', {x: L, y: y - 6, class: 'lbl'}, 'replay events'));
    var mid = y + evH / 2;
    svg.appendChild(el('line', {x1: L, y1: mid, x2: W - R, y2: mid, class: 'axis'}));
    var maxD = 1; for (var t = plotStart; t <= plotEnd; t++) maxD = Math.max(maxD, c.dealt[t], c.taken[t]);
    for (t = Math.max(1,plotStart); t <= plotEnd; t++) {
      if (c.dealt[t]) svg.appendChild(el('line', {x1: X(t, n), y1: mid, x2: X(t, n), y2: mid - (c.dealt[t] / maxD) * (evH / 2 - 8), class: 'ev-dealt'}));
      if (c.taken[t]) svg.appendChild(el('line', {x1: X(t, n), y1: mid, x2: X(t, n), y2: mid + (c.taken[t] / maxD) * (evH / 2 - 8), class: 'ev-taken'}));
      if (c.deathOpp[t]) svg.appendChild(el('path', {d: 'M ' + X(t, n) + ' ' + (y + 2) + ' l -5 8 l 10 0 z', class: 'death-opp'}));
      if (c.deathEgo[t]) svg.appendChild(el('path', {d: 'M ' + X(t, n) + ' ' + (y + evH - 2) + ' l -5 -8 l 10 0 z', class: 'death-ego'}));
    }
    y += evH + 22;
    // strip 2: return + baseline
    var gH = strips[1][1];
    svg.appendChild(el('text', {x: L, y: y - 6, class: 'lbl'}, 'return G · baseline'));
    var lo = Infinity, hi = -Infinity;
    for (t = plotStart; t <= plotEnd; t++) { lo = Math.min(lo, c.G[t], c.B[t]); hi = Math.max(hi, c.G[t], c.B[t]); }
    if (hi - lo < 1) { hi += 1; lo -= 1; }
    function Yg(v) { return y + gH - ((v - lo) / (hi - lo)) * gH; }
    if (lo < 0 && hi > 0) svg.appendChild(el('line', {x1: L, y1: Yg(0), x2: W - R, y2: Yg(0), class: 'axis'}));
    svg.appendChild(el('text', {x: L - 6, y: y + 8, class: 'tick', 'text-anchor': 'end'}, hi.toFixed(0)));
    svg.appendChild(el('text', {x: L - 6, y: y + gH, class: 'tick', 'text-anchor': 'end'}, lo.toFixed(0)));
    svg.appendChild(el('path', {d: linePath(c.B, n, Yg), class: 'line-B'}));
    svg.appendChild(el('path', {d: linePath(c.G, n, Yg), class: 'line-G'}));
    y += gH + 22;
    // strip 3: weight, log scale
    var wH = strips[2][1];
    svg.appendChild(el('text', {x: L, y: y - 6, class: 'lbl'}, 'weight w'));
    var wmin = Infinity, wmax = 0;
    for (t = plotStart; t <= plotEnd; t++) { wmin = Math.min(wmin, c.w[t]); wmax = Math.max(wmax, c.w[t]); }
    var wlo = Math.log(Math.min(wmin / 1.3, 0.7)), whi = Math.log(Math.max(wmax * 1.3, 1.5));
    function Yw(v) { return y + wH - ((Math.log(Math.max(v, 0.05)) - wlo) / (whi - wlo)) * wH; }
    svg.appendChild(el('line', {x1: L, y1: Yw(1), x2: W - R, y2: Yw(1), class: 'rule-one'}));
    svg.appendChild(el('text', {x: L - 6, y: Yw(1) + 3, class: 'tick', 'text-anchor': 'end'}, '1'));
    if (Math.log(state.cap) <= whi) {
      svg.appendChild(el('line', {x1: L, y1: Yw(state.cap), x2: W - R, y2: Yw(state.cap), class: 'rule-cap'}));
      svg.appendChild(el('text', {x: L - 6, y: Yw(state.cap) + 3, class: 'tick', 'text-anchor': 'end'}, 'cap ' + state.cap.toFixed(1)));
    }
    svg.appendChild(el('path', {d: linePath(c.w, n, Yw) + ' L ' + X(plotEnd, n) + ' ' + Yw(1) + ' L ' + L + ' ' + Yw(1) + ' Z', class: 'area-w'}));
    svg.appendChild(el('path', {d: linePath(c.w, n, Yw), class: 'line-w'}));
    // playhead
    if (state.frame !== null) {
      var px = X(state.frame, n);
      svg.appendChild(el('line', {x1: px, y1: TOP, x2: px, y2: H - 16, class: 'head'}));
    }
    document.getElementById('ledger-hl-v').textContent = state.hl.toFixed(1) + ' s · γ ' + c.gamma.toFixed(5);
    document.getElementById('ledger-beta-v').textContent = state.beta.toFixed(0);
    document.getElementById('ledger-cap-v').textContent = state.cap.toFixed(1);
    readout(c);
  }
  function linePath(arr, n, Y) {
    var step = Math.max(1, Math.floor(n / 1400)), d = '';
    for (var t = plotStart; t <= plotEnd; t += step) d += (t > plotStart ? ' L ' : 'M ') + X(t, n).toFixed(1) + ' ' + Y(arr[t]).toFixed(1);
    return d;
  }
  function readout(c) {
    var host = document.getElementById('ledger-readout');
    var t = state.frame;
    if (t === null) {
      host.innerHTML = '<span>' + c.g.label + ' · ' + c.g.stage + '</span><span>ego <b>' + (state.ego === 1 ? c.g.p1 : c.g.p2) + '</b></span><span>frames clipped at cap <b>' + (100 * c.clipped / c.n).toFixed(1) + '%</b></span>';
      return;
    }
    host.innerHTML = '<span>t <b>' + (t / FPS).toFixed(2) + ' s</b></span><span>stocks <b>' + c.g.stock[c.e][t] + '–' + c.g.stock[c.o][t] + '</b></span><span>percent <b>' + c.g.pct[c.e][t].toFixed(0) + '% · ' + c.g.pct[c.o][t].toFixed(0) + '%</b></span><span>G <b>' + c.G[t].toFixed(1) + '</b></span><span>A <b>' + (c.G[t] - c.B[t]).toFixed(1) + '</b></span><span>w <b>' + c.w[t].toFixed(2) + '</b></span>';
  }
  svg.addEventListener('pointerdown', selectFrame);
  svg.addEventListener('pointermove', function(e){if(e.pointerType==='mouse') selectFrame(e);});
  function selectFrame(e) {
    var r = svg.getBoundingClientRect();
    var x = (e.clientX - r.left) / r.width * W;
    var n = games[state.game].n;
    state.frame = Math.max(0, Math.min(n - 1, Math.round(plotStart + (x - L) / (W - L - R) * (plotEnd - plotStart))));
    render();
  }
  svg.addEventListener('mouseleave', function () { state.frame = null; render(); });
  function sync() {
    gamesEl.querySelectorAll('button').forEach(function (b, i) { b.setAttribute('aria-pressed', String(i === state.game)); });
    document.querySelectorAll('#ledger-ego button').forEach(function (b) { b.setAttribute('aria-pressed', String(Number(b.dataset.ego) === state.ego)); });
    document.querySelectorAll('#ledger-base button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.base === state.base)); });
    render();
  }
  beta.value = Math.log10(state.beta);
  window.matchMedia('(max-width:760px)').addEventListener('change', render);
  sync();
}());

/* Fig: the previous controller frame as a one-hot class against a learned class embedding plus its
   physical values (experiments/043_legacy_codec.py group_embedding), with the matched difference. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('past-action-diagram');
  if (!D || !CI || !svg) return;
  var d = CI.past_action_encoding;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.pastActionEffect = function () {
    return [{comparison: '256-way one-hot → learned class + physical value', treatment_run: 'cqbbbg77', baseline_run: '1zzstd2d', method: d.method, stock_delta: d.stock.delta, stock_ci95_lo: d.stock.lo, stock_ci95_hi: d.stock.hi, damage_delta: d.damage.delta, damage_ci95_lo: d.damage.lo, damage_ci95_hi: d.damage.hi}];
  };
  svg.setAttribute('viewBox', '0 0 880 290');
  var mk = D.marker(svg, 'arrow', 'var(--ink-faint)');
  var H = 28, GX = 92, GW = 52, CX = 182, CW = 36, OUT = 580, OUTW = 128;   // given, class and produced columns
  var Y1 = 40, Y2 = 176;                                                    // row centres
  function arrow(x1, y1, x2, y2) { D.wire(svg, x1, y1, x2, y2, mk); }
  function label(x, y, s) { D.text(svg, x, y, s, {class: 'diagram-note', 'text-anchor': 'middle'}); }
  function bend(x1, y1, x2, y2) { D.path(svg, 'M ' + x1 + ' ' + y1 + ' C ' + (x1 + 30) + ' ' + y1 + ', ' + (x2 - 30) + ' ' + y2 + ', ' + x2 + ' ' + y2, mk); }

  // row 1 · one-hot: the class index as a 256-wide indicator vector, concatenated as is
  D.text(svg, 16, Y1 + 4, 'one-hot', {class: 'diagram-title'});
  D.token(svg, GX, Y1 - H / 2, GW, H, 'aₜ₋₁', 'given');
  arrow(GX + GW + 6, Y1, CX - 8, Y1);
  D.token(svg, CX, Y1 - H / 2, CW, H, 'c');
  label(CX + CW / 2, Y1 + H / 2 + 14, 'class');
  arrow(CX + CW + 6, Y1, OUT - 8, Y1);
  svg.appendChild(D.el('rect', {x: OUT, y: Y1 - H / 2, width: OUTW, height: H, class: 'model-token'}));
  svg.appendChild(D.el('rect', {x: OUT + 43, y: Y1 - H / 2, width: 2, height: H, class: 'model-token played'}));
  label(OUT + OUTW / 2, Y1 + H / 2 + 14, 'one-hot(c)');

  // row 2 · learned: class embedding plus a linear map of the physical values, normalised
  D.text(svg, 16, Y2 + 4, 'learned', {class: 'diagram-title'});
  D.token(svg, GX, Y2 - H / 2, GW, H, 'aₜ₋₁', 'given');
  arrow(GX + GW + 6, Y2, CX - 8, Y2);
  D.token(svg, CX, Y2 - H / 2, CW, H, 'c');
  label(CX + CW / 2, Y2 + H / 2 + 14, 'class');
  var UP = Y2 - 34, DOWN = Y2 + 34, BX = 274;
  bend(CX + CW + 6, Y2, BX - 8, UP);
  bend(CX + CW + 6, Y2, BX - 8, DOWN);
  D.box(svg, BX, UP - H / 2, 92, H, 'Embedding');
  D.token(svg, BX, DOWN - H / 2, 40, H, 'v(c)');
  label(BX + 20, DOWN + H / 2 + 14, 'stick · trigger values');
  arrow(BX + 46, DOWN, BX + 62, DOWN);
  D.box(svg, BX + 70, DOWN - H / 2, 56, H, 'linear');
  var SX = BX + 70 + 56 + 34;                                                // both branches into +
  bend(BX + 92 + 6, UP, SX - 14, Y2);
  bend(BX + 70 + 56 + 6, DOWN, SX - 14, Y2);
  svg.appendChild(D.el('circle', {cx: SX, cy: Y2, r: 11, class: 'model-box'}));
  D.text(svg, SX, Y2 + 4, '+', {'text-anchor': 'middle', class: 'diagram-title'});
  arrow(SX + 12, Y2, SX + 34, Y2);
  D.box(svg, SX + 40, Y2 - H / 2, 72, H, 'RMSNorm');
  arrow(SX + 40 + 72 + 6, Y2, OUT - 8, Y2);
  D.token(svg, OUT, Y2 - H / 2, 44, H, 'e(c)');
  label(OUT + 22, Y2 + H / 2 + 14, 'per controller group');

  // the matched difference
  D.effectBars(svg, [Object.assign({label: 'one-hot → learned', sub: 'O28'}, d)],
    {x: 720, y: 12, panelW: 156, panelH: 250, barW: 34, axisW: 44, metrics: ['stock']});
}());

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
  D.text(svg, 672, 50, 'W windows', {class: 'diagram-note'});

  /* ---- 2 · the batches that follow, in order ---- */
  var by = 130, bh = 150, bw = 38, slots = 6, gap = 6, x0 = 16;
  D.text(svg, x0, by - 30, 'batches, in training order →', {class: 'diagram-note'});
  // clusters of three batches, ≈ P batches apart; window k of the replay lands in cluster k
  var clusters = [['b', 'b+1', 'b+2'], ['b+P−1', 'b+P', 'b+P+1'], ['b+2P−1', 'b+2P', 'b+2P+1'], ['b+3P−1', 'b+3P', 'b+3P+1']];
  var placeR = [[0, 1], [1, 2], [0, 3], [2, 1]];   // [column in cluster, slot] for the followed replay's window k

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

        svg.appendChild(D.el('rect', {x: bx + 6, y: sy, width: bw - 12, height: (bh - 16) / slots - 5, class: cls}));
      }
      D.text(svg, bx + bw / 2, by + bh + 14, 'batch ' + (k * 219 + j + 1), {'text-anchor': 'middle', class: 'diagram-note'});
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
    D.text(svg, (a + b) / 2, y2 + 13, 'about 219 batches later', {'text-anchor': 'middle', class: 'diagram-note'});
  });
  // legend
  var ly = 366;
  svg.appendChild(D.el('rect', {x: x0, y: ly - 9, width: 14, height: 10, class: 'loader-win r'}));
  D.text(svg, x0 + 20, ly, 'the replay above · one window per batch', {class: 'diagram-note'});

  svg.appendChild(D.el('rect', {x: x0, y: ly + 9, width: 14, height: 10, class: 'loader-win'}));
  D.text(svg, x0 + 20, ly + 18, 'other replays · every window in a batch comes from a different replay', {class: 'diagram-note'});
}());

/* Fig: alternative objectives and decoders, each as behavior cloning → the alternative with a 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('objective-effects-diagram');
  if (!D || !CI || !svg) return;
  var ROWS = [
    {key: 'attached_iql', label: 'behavior cloning → attached IQL', sub: 'actor + Q₁/Q₂/V losses', runs: ['06fi5jms', 'cqbbbg77']},
    {key: 'endpoint_flow', label: 'causal decoder → endpoint flow', sub: 'parallel flow over ten offsets', runs: ['t4s3gohe', '5wfk2esf']},
    {key: 'exploration_modeling:XM K=1', label: 'behavior cloning → exploration modeling', sub: '1 plan · one latent trajectory', runs: ['mz8dk2gh', 'cqbbbg77']},
    {key: 'exploration_modeling:XM K=8', label: 'behavior cloning → exploration modeling', sub: '8 plans · best of eight latent trajectories', runs: ['feic18gt', 'cqbbbg77']}
  ];
  var rows = ROWS.map(function (r) { return Object.assign({}, r, CI[r.key]); });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.objectiveEffects = function () {
    return rows.map(function (r) { return {comparison: r.label + ' · ' + r.sub.split(' · ')[0], treatment_run: r.runs[0], baseline_run: r.runs[1], method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 50});
}());

/* Fig: AWR against behavior cloning, and the return half-life sweep, as differences with 95% intervals. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('awr-effects-diagram');
  if (!D || !CI || !svg) return;
  var ROWS = [
    {key: 'awr_actor_weighting', label: 'behavior cloning → AWR', sub: 'unpaired evaluations', runs: ['hwzv0k9a', '5wfk2esf']},
    {key: 'awr_gamma:gamma 0.99855', label: 'return half-life 3 s → 8 s', sub: 'O52 proxy', runs: ['0ksityvv', 'yxy8brhx'], accent: true},
    {key: 'awr_gamma:gamma 0.99920', label: 'return half-life 3 s → 14 s', sub: 'O52 proxy', runs: ['8t9koblm', 'yxy8brhx']}
  ];
  var rows = ROWS.map(function (r) { return Object.assign({}, r, CI[r.key]); });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.awrEffects = function () {
    return rows.map(function (r) { return {comparison: r.label, treatment_run: r.runs[0], baseline_run: r.runs[1], method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 50});
}());

/* Fig: four windows per replay → one, and the closed-loop difference with its 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('replay-windows-diagram');
  if (!D || !CI || !svg) return;
  var d = CI.windows_per_replay;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.replayWindows = function () {
    return [{comparison: 'four → one window per replay', treatment_run: 'shjnxxsu', baseline_run: 'uw05bvm2', method: d.method, stock_delta: d.stock.delta, stock_ci95_lo: d.stock.lo, stock_ci95_hi: d.stock.hi, damage_delta: d.damage.delta, damage_ci95_lo: d.damage.lo, damage_ci95_hi: d.damage.hi}];
  };

  var demo = document.createElement('div'); demo.className='replay-batches';
  [4,16].forEach(function(replays){
    var section=document.createElement('section');
    var h=document.createElement('h5');h.textContent=replays===4?'4 replays × 4 windows':'16 replays × 1 window';section.appendChild(h);
    var grid=document.createElement('div');grid.className='batch-grid';
    for(var i=0;i<16;i++){
      var cell=document.createElement('span');var replay=replays===4?Math.floor(i/4):i;
      cell.textContent=String.fromCharCode(65+replay)+(replays===4?String(i%4+1):'1');
      cell.style.background='color-mix(in srgb, var(--oxblood) '+(6+(replay%4)*7)+'%, var(--paper))';
      cell.setAttribute('aria-label','Replay '+String.fromCharCode(65+replay)+', window '+(replays===4?i%4+1:1));
      grid.appendChild(cell);
    }
    section.appendChild(grid);
    var p=document.createElement('p');p.className='figure-note';p.textContent='16 windows in this illustrative batch';section.appendChild(p);demo.appendChild(section);
  });
  svg.before(demo);
  D.forest(svg,[Object.assign({label:'four → one window per replay',sub:'old evaluation; few matches per arm'},d)],{rowH:50});
}());

/* Fig: execution horizon per O37 cell, as change from one frame per plan with 95% intervals. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('horizon-effects-diagram');
  if (!D || !CI || !svg) return;
  var CELLS = [
    {id: 'D0', name: 'nothing', run: '98r9smrj'},
    {id: 'D1', name: 'earlier groups', run: 'a117chkw'},
    {id: 'D2', name: 'earlier frames', run: '50q39o9j'},
    {id: 'D3', name: 'frames and groups', run: '5wfk2esf', accent: true}
  ];
  var HS = [2, 4, 6];
  var series = CELLS.map(function (c) {
    return Object.assign({}, c, {points: HS.map(function (h) { return Object.assign({h: h}, CI['horizon:' + c.id + ':H' + h + '-H1']); })});
  });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.horizonEffects = function () {
    var out = [];
    series.forEach(function (s) { s.points.forEach(function (p) { out.push({o37_cell: s.id, conditioning: s.name, run: s.run, frames_per_plan: p.h, baseline_frames_per_plan: 1, method: p.method, stock_delta: p.stock.delta, stock_ci95_lo: p.stock.lo, stock_ci95_hi: p.stock.hi, damage_delta: p.damage.delta, damage_ci95_lo: p.damage.lo, damage_ci95_hi: p.damage.hi}); }); });
    return out;
  };

  var W = 880, H = 360, L = 70, R = 250, top = 48, bottom = H - 44;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var all = [];
  series.forEach(function (s) { s.points.forEach(function (p) { all.push(p.stock); }); });
  var dom = D.domain(all);
  var y = D.scale([bottom, top], dom.lo, dom.hi);
  var lane = (W - L - R - 30) / 5;
  function x(h, k) { return L + 15 + (h - 1) * lane + (h === 1 ? 0 : (k - 1.5) * 8); }
  D.text(svg, 16, 22, 'aₜ₊₂ conditioned on · Δ net stocks / min from 1 frame per plan · dot and 95% interval', {class: 'diagram-note'});
  dom.ticks.forEach(function (v) {
    svg.appendChild(D.el('line', {x1: L, y1: y(v), x2: W - R, y2: y(v), class: v === 0 ? 'effect-zero' : 'effect-grid'}));
    D.text(svg, L - 8, y(v) + 4, D.signed(v, dom.digits), {class: 'diagram-note', 'text-anchor': 'end'});
  });
  [1,2,4,6].forEach(function (h) { D.text(svg, x(h,1.5), bottom + 18, h + ' frame' + (h===1?'':'s'), {class: 'diagram-note', 'text-anchor': 'middle'}); });
  D.text(svg, W - R - 8, top - 10, '1 frame per plan = baseline', {class: 'diagram-note', 'text-anchor': 'end'});
  svg.appendChild(D.el('circle',{cx:x(1,0),cy:y(0),r:4,class:'effect-pt'}));
  series.forEach(function (s, k) {
    var pts = [x(1,k) + ',' + y(0)].concat(s.points.map(function (p) { return x(p.h, k) + ',' + y(p.stock.delta); }));
    svg.appendChild(D.el('polyline', {points: pts.join(' '), class: 'effect-line' + (s.accent ? ' accent' : '')}));
    s.points.forEach(function (p) {
      var cx = x(p.h, k), cls = s.accent ? ' accent' : '';
      svg.appendChild(D.el('line', {x1: cx, y1: y(p.stock.lo), x2: cx, y2: y(p.stock.hi), class: 'effect-ci' + cls}));
      svg.appendChild(D.el('circle', {cx: cx, cy: y(p.stock.delta), r: 3.6, class: 'effect-pt' + cls}));
    });
    // legend: name and the six-frame value
    var last = s.points[s.points.length - 1];
    var ly = y(last.stock.delta) + (k===2?-22:k===3?22:0);
    svg.appendChild(D.el('line',{x1:x(6,k)+5,y1:y(last.stock.delta),x2:W-R+44,y2:ly,class:'effect-line'+(s.accent?' accent':'')}));
    svg.appendChild(D.el('line', {x1: W - R + 24, y1: ly, x2: W - R + 44, y2: ly, class: 'effect-line' + (s.accent ? ' accent' : '')}));
    svg.appendChild(D.el('circle', {cx: W - R + 34, cy: ly, r: 3.2, class: 'effect-pt' + (s.accent ? ' accent' : '')}));
    D.text(svg, W - R + 54, ly + 4, s.name, {class: 'diagram-title' + (s.accent ? ' accent' : '')});
    D.text(svg, W - R + 54, ly + 18, '1 → 6 frames: ' + D.signed(last.stock.delta, 2), {class: 'diagram-note'});
  });
}());

/* Fig: batch size at fixed examples, batch 2,048 → each arm with a 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('batch-effects-diagram');
  if (!D || !CI || !svg) return;
  var ARMS = [[128, 'nt4gh4mk'], [256, 'phxqvexa'], [512, '980k3yla'], [1024, '7ok39dju'], [2048, 'rri7scjc']];
  var rows = ARMS.map(function (a) {
    var b = a[0], updates = (2097152 / b).toLocaleString('en-US') + ' updates';
    if (b === 2048) return {label: 'batch 2,048', sub: updates + ' · baseline', baseline: true, run: a[1]};
    return Object.assign({label: 'batch ' + b.toLocaleString('en-US'), sub: updates, run: a[1], accent: b === 128}, CI['batch:' + b + '-2048']);
  });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.batchEffects = function () {
    return rows.filter(function (r) { return !r.baseline; }).map(function (r) { return {comparison: 'batch 2,048 → ' + r.label, treatment_run: r.run, baseline_run: 'rri7scjc', method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.sweep(svg, rows, ARMS.map(function(a){return a[0];}), ARMS.map(function(a){return String(a[0]);}), ['16k','8k','4k','2k','1k'], 'batch size · log scale / updates below');
}());

/* Fig: the paired Muon/AdamW learning-rate sweep, the smallest pair → each arm with a 95% interval. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('lr-effects-diagram');
  if (!D || !CI || !svg) return;
  var ARMS = [['0.014', '0.000595', '9luwu85n'], ['0.028', '0.00119', 'wmwellfd'], ['0.040', '0.00170', '7hy8xjjh'], ['0.056', '0.00238', '2rbed6ze'], ['0.080', '0.00340', '5adh9mtu'], ['0.112', '0.00476', 'zerhe8au']];
  var rows = ARMS.map(function (a) {
    var label = 'Muon ' + a[0] + ' · AdamW ' + a[1];
    if (a[0] === '0.014') return {label: label, sub: 'baseline', baseline: true, run: a[2]};
    return Object.assign({label: label, run: a[2], accent: a[0] === '0.040'}, CI['lr:' + a[0] + '-0.014']);
  });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.lrEffects = function () {
    return rows.filter(function (r) { return !r.baseline; }).map(function (r) { return {comparison: 'Muon 0.014 · AdamW 0.000595 → ' + r.label, treatment_run: r.run, baseline_run: '9luwu85n', method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.sweep(svg, rows, ARMS.map(function(a){return Number(a[0]);}), ARMS.map(function(a){return a[0];}), null, 'Muon learning rate · log scale');
}());

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

/* Fig: the two matched codec comparisons as baseline → treatment Δ bars with 95% intervals. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('codec-effects-diagram');
  if (!D || !CI || !svg) return;
  var rows = [
    Object.assign({label: 'legacy → Cartesian', sub: 'O43 bridge', runs: 'acbbeb76 → cqbbbg77'}, CI.codec_legacy_cartesian),
    Object.assign({label: 'Cartesian → polar', sub: 'O53', runs: 'yxy8brhx → yi7yyyu0'}, CI.codec_cartesian_polar)
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.codecEffects = function () {
    return rows.map(function (r) { return {comparison: r.label, runs: r.runs, method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  svg.setAttribute('viewBox', '0 0 880 240');
  D.effectBars(svg, rows, {x: 8, y: 8, panelW: 390, panelH: 224, gap: 70, barW: 34, axisW: 48});
}());

/* Fig: each player identity against the masked identity, on one trained model. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('identity-effects-diagram');
  if (!D || !CI || !svg) return;
  var NAMES = ['Cody', 'Master', 'Mang0', 'Platinum'];
  var rows = [{label: 'identity masked', sub: 'baseline', baseline: true}].concat(NAMES.map(function (n) {
    return Object.assign({label: n, sub: (n === 'Master' || n === 'Platinum') ? 'pooled ranked-player identity' : 'individual player identity', accent: n === 'Cody'}, CI['player_identity:' + n]);
  }));
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.identityEffects = function () {
    return rows.filter(function (r) { return !r.baseline; }).map(function (r) { return {comparison: 'masked → ' + r.label, method: r.method, boots: r.n_boots, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 44, header: 'masked → each identity · 95% interval'});
}());

/* Fig: the two optimizer-recipe tests, with each arm's validation loss beside the play difference. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI, DATA = window.HALFIG_DATA;
  var svg = document.getElementById('optimizer-effects-diagram');
  if (!D || !CI || !DATA || !svg) return;
  var V = DATA.optimizer;
  var rows = [
    Object.assign({label: 'AdamW → Muon + AdamW', sub: 'o3nihah5 → vlim96s9 · 7M · unpaired', loss: [V.o3nihah5, V.vlim96s9]}, CI.optimizer_small),
    Object.assign({label: 'Muon + AdamW → AdamW', sub: 'p1fyyp1z → tgpuo1be · 216M production', accent: true, loss: [V.p1fyyp1z, V.tgpuo1be]}, CI.optimizer_full_scale)
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.optimizerEffects = function () {
    return rows.map(function (r) { return {comparison: r.label, runs: r.sub.split(' · ')[0], method: r.method, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi, baseline_val_loss: r.loss[0].val_loss, treatment_val_loss: r.loss[1].val_loss, val_metric: r.loss[0].metric}; });
  };
  var top = 60, rowH = 56;
  D.forest(svg, rows, {rowH: rowH, top: top, cols: [[340, 480], [560, 690]], header: 'baseline → treatment · 95% interval'});
  // a third column: the validation loss of each arm, read baseline → treatment
  var vx = 766;
  D.text(svg, vx, top - 28, 'validation loss', {class: 'diagram-note'});
  D.text(svg, vx, top - 14, 'before → after', {class: 'diagram-note'});
  rows.forEach(function (r, i) {
    var y = top + i * rowH + rowH / 2 - 6;
    D.text(svg, vx, y + 4, r.loss[0].val_loss.toFixed(3) + ' → ' + r.loss[1].val_loss.toFixed(3), {class: 'diagram-label'});
    D.text(svg, vx, y + 18, r.loss[0].metric === 'val/loss' ? 'older loss metric' : 'validation NLL', {class: 'diagram-note'});
  });
}());

/* Fig: ranked-anonymized data only → all 44 sources, at equal updates. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('data-source-effects-diagram');
  if (!D || !CI || !svg) return;
  var rows = [Object.assign({label: 'ranked only → all 44 sources', sub: 'k8nb00xv → bevhq159 · O42 · equal updates', accent: true}, CI.data_source_mix)];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.dataSourceEffects = function () {
    return rows.map(function (r) { return {comparison: r.label, runs: r.sub.split(' · ')[0], method: r.method, boots: r.n_boots, stock_delta: r.stock.delta, stock_ci95_lo: r.stock.lo, stock_ci95_hi: r.stock.hi, damage_delta: r.damage.delta, damage_ci95_lo: r.damage.lo, damage_ci95_hi: r.damage.hi}; });
  };
  D.forest(svg, rows, {rowH: 48});
}());

/* Fig: loss weight per predicted offset under the O26 split and the 50%-next-frame split, then the comparison. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var svg = document.getElementById('loss-allocation-diagram');
  if (!D || !CI || !svg) return;
  var OFFS = [1, 2, 3, 4, 5, 6, 9, 12, 16, 20];
  // experiments/043_legacy_codec.py _joint_objective: O26 = mean over offsets 1–4 plus mean over the rest
  // (total 2); the treatment = 2 · (½ · offset 1 + ½ · mean over the other nine)
  var ROWS = [
    {label: 'O26 split', sub: 'near 4 share 1 · far 6 share 1', run: 'acbbeb76', w: OFFS.map(function (o, i) { return i < 4 ? 0.25 : 1 / 6; })},
    {label: 'half on the next frame', sub: 'next frame 1 · other 9 share 1', run: '1imfy8v3', accent: true, w: OFFS.map(function (o, i) { return i === 0 ? 1 : 1 / 9; })}
  ];
  var d = CI.next_frame_loss_share;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.lossAllocation = function () {
    var out = [];
    ROWS.forEach(function (r) { OFFS.forEach(function (o, i) { out.push({allocation: r.label, run: r.run, offset: o, loss_weight: r.w[i]}); }); });
    out.push({comparison: 'O26 split → half on the next frame', runs: 'acbbeb76 → 1imfy8v3', method: d.method, stock_delta: d.stock.delta, stock_ci95_lo: d.stock.lo, stock_ci95_hi: d.stock.hi, damage_delta: d.damage.delta, damage_ci95_lo: d.damage.lo, damage_ci95_hi: d.damage.hi});
    return out;
  };

  var W = 880, H = 396;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var x0 = 262, tw = 48, gap = 11, barMax = 60;
  function sub(o) { return 'aₜ₊' + String(o).replace(/\d/g, function (c) { return '₀₁₂₃₄₅₆₇₈₉'[c]; }); }
  ROWS.forEach(function (r, k) {
    var base = 118 + k * 130, cls = r.accent ? ' accent' : '';
    D.text(svg, 16, base - 40, r.label, {class: 'diagram-title' + cls});
    D.text(svg, 16, base - 24, r.sub, {class: 'diagram-note'});
    D.text(svg, 16, base - 10, r.run, {class: 'diagram-note'});
    OFFS.forEach(function (o, i) {
      var x = x0 + i * (tw + gap), h = r.w[i] * barMax;
      svg.appendChild(D.el('rect', {x: x + 10, y: base - 8 - h, width: tw - 20, height: h, class: 'loss-bar' + cls}));
      D.text(svg, x + tw / 2, base - 12 - h, r.w[i].toFixed(2).replace(/^0/, ''), {class: 'diagram-note', 'text-anchor': 'middle'});
      D.token(svg, x, base, tw, 26, sub(o), k === 0 ? 'given' : '');
    });
  });
  D.text(svg, x0, 34, 'loss weight per predicted frame · frames sum to 2 in both rows', {class: 'diagram-note'});
  D.text(svg, x0 + OFFS.length * (tw + gap) - gap, 118 + 130 + 42, 'predicted frame, by offset from now', {class: 'diagram-note', 'text-anchor': 'end'});

  // the comparison row
  var y = 350;
  svg.appendChild(D.el('line', {x1: 16, y1: y - 44, x2: W - 16, y2: y - 44, stroke: 'var(--rule-dotted)', 'stroke-dasharray': '2 3'}));
  D.text(svg, 16, y - 4, 'O26 split → half on the next frame', {class: 'diagram-title accent'});
  D.text(svg, 16, y + 10, 'acbbeb76 → 1imfy8v3 · O43 · 95% interval', {class: 'diagram-note'});
  var doms = [D.domain([d.stock]), D.domain([d.damage])], cols = [[336, 540], [640, 826]];
  doms.forEach(function (dom, k) {
    var xs = D.scale(cols[k], dom.lo, dom.hi);
    D.effectAxis(svg, xs, dom, y - 12, y + 12, k ? 'Δ net damage / min' : 'Δ net stocks / min');
    D.effect(svg, xs, y, k ? d.damage : d.stock, {accent: true, digits: k ? 1 : 2, valueX: cols[k][1] + 8});
  });
}());

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

  var W = 880, H = 492;
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
  D.effectAxis(svg, x, dom, top - 6, top + ROWS.length * rowH + 24, 'Δ net stocks / min');
  ROWS.forEach(function (r, i) {
    var y = top + i * rowH + rowH / 2 - 6 + (i ? 32 : 0);
    if (i === 1) {
      svg.appendChild(D.el('line',{x1:16,x2:864,y1:y-37,y2:y-37,class:'effect-grid'}));
      D.text(svg,16,y-18,'Sparse baseline A → cross-attention variants',{class:'diagram-note'});
    }
    D.text(svg, 16, y, r.label, {class: 'diagram-title' + (r.accent ? ' accent' : '')});
    D.text(svg, 16, y + 14, r.sub, {class: 'diagram-note'});
    D.effect(svg, x, y, r.d, {accent: r.accent, digits: 2, valueX: 792});
  });
}());

/* Fig: the compact table of ideas that did not help or did not finish, built from the study table. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var table = document.getElementById('failed-ideas-table');
  if (!D || !CI || !table) return;
  var GROUPS = [
    {title: 'Completed comparisons · no demonstrated gain', rows: [
      {id: 'rle_bpe_actions', idea: 'Merge held inputs into span tokens', what: 'One symbol per frame, BPE merges repeated holds into spans of up to 16 frames; a span plays to completion.', also: 'Execution cadence, parameter count and crash rate changed with it; 25 vs 95 boots.'},
      {id: 'fixed_long_delay', idea: 'Predict 12 frames ahead', what: 'Delay 1 → delay 12 on the O43 model.', also: 'A longer future-data buffer came with the delay; 32 vs 96 boots.'},
      {id: 'rank_weighted_bc', idea: 'Weight replays by player rank', what: 'Platinum, Diamond and Master replays weighted 1, 2 and 4.', also: 'Nothing else changed.'},
      {id: 'future_state_expert', idea: 'Predict the next game state as a side task', what: 'A training-only head that predicts future state next to the action heads.', also: 'The treatment had 8% more parameters and twice the examples, and still did not win.'},
      {id: 'lean_feature_bundle', idea: 'A slimmer state-feature bundle', what: 'Several state columns removed or changed at once.', also: 'Old evaluation protocol; no single column is identified.'}
    ]},
    {title: 'Inconclusive or unfinished', rows: [
      {id: 'hand_features', idea: 'Hand-made geometry features', what: 'Relative position, velocity, range and stage geometry appended to the frame token.', also: 'Both runs stopped before a valid closed-loop result.'},
      {id: 'joint_action_chords', idea: 'One token for the whole controller', what: 'Sticks, triggers and buttons packed into one data-derived class.', also: 'The run stopped after three diagnostic updates.'},
      {id: 'flattened_action_tokens', idea: 'Controller parts as tokens in the main sequence', what: 'Each frame became one state token plus four controller tokens; 256 frames became 1,280 tokens.', also: 'About five times the trunk work and different context lengths; no equal-work control.'}
    ]}
  ];
  function fmt(d) { return D.signed(d.delta, 2) + ' [' + D.signed(d.lo, 2) + ', ' + D.signed(d.hi, 2) + ']'; }
  function result(r) {
    var ci = CI[r.id], s = D.study(r.id);
    if (ci) return {text: fmt(ci.stock), unit: (ci.control || '') + ' → ' + (ci.treatment || '') + ' · stocks / min · ' + (ci.method === 'welch' ? 'unpaired' : 'paired'), num: true};
    var arms = (s && s.arms || []).filter(function (a) { return typeof a.stock === 'number'; });
    if (arms.length) return {text: arms.map(function (a) { return D.signed(a.stock, 2); }).join(' · '), unit: 'arm scores only · no matched control', num: false};
    return {text: 'no result', unit: 'run did not complete', num: false};
  }
  var rows = [];
  var html = '<thead><tr><th>idea</th><th>what was tried</th><th>result</th><th>what else changed</th><th>evidence</th></tr></thead>';
  GROUPS.forEach(function (g) {
    html += '<tbody><tr class="group"><td colspan="5">' + g.title + '</td></tr>';
    g.rows.forEach(function (r) {
      var s = D.study(r.id) || {}, res = result(r);
      rows.push({idea: r.idea, study: r.id, evidence: s.evidence, result: res.text, result_basis: res.unit, what_else_changed: r.also, runs: (s.arms || []).map(function (a) { return a.run; }).filter(Boolean).join(' · ')});
      html += '<tr><td class="idea">' + r.idea + '</td><td>' + r.what + '</td><td class="res' + (res.num ? '' : ' soft') + '"><b>' + res.text + '</b><span>' + res.unit + '</span></td><td>' + r.also + '</td><td><span class="ev ev-' + (s.evidence || 'missing') + '">' + (s.evidence === 'strong_rejection' ? 'no demonstrated gain' : (s.evidence || '').replace('_', ' ')) + '</span></td></tr>';
    });
    html += '</tbody>';
  });
  table.innerHTML = html;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.failedIdeas = function () { return rows; };
}());

/* Fig: the ranked corpus before and after the compact policy layout (docs/experiments/data_pipeline.md). */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('storage-bars-diagram');
  if (!D || !svg) return;
  var ROWS = [
    {label: 'decoded training arrays', sub: 'train split · what the loader reads', before: 802.47, after: 76.19},
    {label: 'compressed shards on disk', sub: 'whole artifact · Zstd, as stored', before: 29.82, after: 13.34}
  ];
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.storageBars = function () {
    return ROWS.map(function (r) { return {quantity: r.label, corpus: 'ranked-anonymized-1 train · 112,409 replays', mds_v7_gb: r.before, compact_policy_v7_gb: r.after, reduction: +(r.before / r.after).toFixed(2)}; });
  };

  var panel=document.createElement('div');panel.className='storage-summary';
  var title=document.createElement('h5');title.textContent='Decoded training arrays';panel.appendChild(title);
  [ROWS[0].before,ROWS[0].after].forEach(function(v,i){
    var label=document.createElement('p');label.textContent=(i?'After: ':'Before: ')+v.toFixed(0)+' GB';panel.appendChild(label);
    var bar=document.createElement('div');bar.className='storage-bar'+(i?' after':'');bar.style.width=(v/ROWS[0].before*100)+'%';panel.appendChild(bar);
  });
  var note=document.createElement('p');note.className='figure-note';
  note.textContent='10.5× smaller decoded arrays. Compressed artifact: 30 → 13 GB (2.2× smaller). These are storage sizes, not a measured throughput gain.';
  panel.appendChild(note);svg.replaceWith(panel);
}());

/* Fig: O46 real-time chunking — committed prefix given, the rest written; same rule at training time. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('rtc-timeline-diagram');
  if (!D || !svg) return;
  var S = D.study('rtc');
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.rtcTimeline = function () {
    return (S && S.arms || []).map(function (a) { return {arm: a.name, run: a.run, updates: a.updates, character_suite_stock: a.character_suite_stock, fox_suite_stock: a.fox_suite_stock, control: 'not run'}; });
  };
  var W = 880, H = 440;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var mk = D.marker(svg, 'arrow', 'var(--ink-faint)');
  function sub(n) { return String(n).replace(/\d/g, function (c) { return '₀₁₂₃₄₅₆₇₈₉'[c]; }); }
  var d = 3, tw = 52, gap = 6, x0 = 150;

  var tl = 40;  // the frame each token belongs to, placed over the token itself (drawn after the rows)

  function row(y, title, note, givenLabels, producedLabels, givenNote, producedNote, producedCls) {
    D.text(svg, 16, y + 18, title, {class: 'diagram-title'});
    D.text(svg, 16, y + 34, note, {class: 'diagram-note'});
    var r = D.row(svg, x0, y, givenLabels, producedLabels, {w: tw, h: 28, gap: gap, arrow: 40, mk: mk, producedCls: producedCls});
    D.text(svg, x0 + 62, y - 10, givenNote, {class: 'diagram-note'});
    D.text(svg, r.producedStart, y - 10, producedNote, {class: 'diagram-note' + (producedCls ? ' accent-text' : '')});
    D.text(svg, r.end + 6, y + 18, '···', {class: 'diagram-note'});
    return r;
  }
  var given = ['hₜ']; for (var k = 0; k < d; k++) given.push(k ? 'aₜ₊' + sub(k) : 'aₜ');
  var produced = []; for (k = d; k < 7; k++) produced.push('aₜ₊' + sub(k));
  var play = row(80, 'playing', 'd = ' + d + ' queued', given, produced, 'committed · cannot change', 'written now · can still change', 'played');

  // frame labels over the play row, one per action token, given and produced alike
  for (var i = 0; i < 7; i++) {
    var cx = i < d ? x0 + (tw + gap) * (i + 1) + tw / 2 : play.producedStart + (i - d) * (tw + gap) + tw / 2;
    D.text(svg, cx, tl, i === 0 ? 't' : 't+' + i, {class: 'diagram-note', 'text-anchor': 'middle'});
  }
  D.text(svg, play.end + 8, tl, '···', {class: 'diagram-note'});

  // the same rule at training time
  var y2 = 190;
  svg.appendChild(D.el('line', {x1: 16, y1: y2 - 44, x2: W - 16, y2: y2 - 44, stroke: 'var(--rule-dotted)', 'stroke-dasharray': '2 3'}));

  row(y2, 'training · O46', 'd random in 0–4', given, produced, 'true actions · no loss', 'loss · offsets to 20', '');
  D.text(svg,16,278,'Timing example · no network delay · inference 3 frames · replan every 3 frames',{class:'diagram-title'});
  var tx=function(f){return 200+f*84;};
  D.text(svg,16,326,'game applies',{class:'diagram-note'});
  for(var f=0;f<6;f++){
    D.token(svg,tx(f),306,78,28,'aₜ'+(f?'₊'+sub(f):''),f<3?'given':'played');
    D.text(svg,tx(f),296,f?'t+'+f:'t',{class:'diagram-note'});
  }
  D.text(svg,16,365,'inference',{class:'diagram-note'});
  svg.appendChild(D.el('rect',{x:tx(0),y:344,width:tx(3)-tx(0),height:28,class:'model-token'}));
  D.text(svg,tx(0)+10,362,'compute while committed inputs play',{class:'diagram-note'});
  svg.appendChild(D.el('line',{x1:tx(3),x2:tx(3),y1:288,y2:392,class:'effect-zero'}));
  D.text(svg,tx(3)+8,388,'plan returns; next replan starts',{class:'diagram-note'});
  D.text(svg,16,422,'Illustrative schedule, not measured latency. The prediction horizon is six frames.',{class:'diagram-note'});

}());

/* Fig: O54 — four model sizes, each deployed at the delay its own inference forces, across four data budgets. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('capacity-panels-diagram');
  var readout = document.getElementById('capacity-panels-readout');
  var S = D && D.study('capacity_and_latency');
  if (!D || !svg || !S) return;
  var MODELS = [
    {w: 256, label: '5M', delay: 1}, {w: 512, label: '39M', delay: 2}, {w: 768, label: '125M', delay: 3, accent: true}, {w: 1024, label: '289M', delay: 6}
  ];
  var TIERS = [28, 29, 30, 31];
  var cell = {};
  S.arms.forEach(function (a) { cell[a.name.split('/')[0].slice(1) + ':' + a.data_exponent] = a; });
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.capacityPanels = function () {
    return S.arms.map(function (a) { return {model: a.name.split(' at ')[0], run: a.run, data_exponent: a.data_exponent, bc_parameters: a.bc_parameters, delay_frames: a.delay_frames, latency_p99_ms: +a.latency_p99_ms.toFixed(1), stock: a.stock, stock_ci95_lo: a.stock_ci95[0], stock_ci95_hi: a.stock_ci95[1]}; });
  };


  function render(){
    D.empty(svg);
    var small=window.matchMedia('(max-width:760px)').matches;
    var W=small?340:880, pw=small?278:182, gap=18;
    svg.classList.add('reflow-chart');
    svg.setAttribute('viewBox','0 0 '+W+' '+(small?1180:310));
    TIERS.forEach(function(t,k){
      var g=D.el('g',{transform:'translate('+(small?0:k*(pw+gap))+','+(small?k*295:0)+')'});svg.appendChild(g);
      var x=D.scale([70,40+pw],0,7), y=D.scale([244,46],-1.3,0.2);
      D.text(g,50,20,'D = 2'+String(t).replace(/\d/g,function(c){return '⁰¹²³⁴⁵⁶⁷⁸⁹'[c];}),{class:'diagram-title'});
      [-1,-0.5,0].forEach(function(v){
        g.appendChild(D.el('line',{x1:50,x2:pw+50,y1:y(v),y2:y(v),class:v===0?'effect-zero':'effect-grid'}));
        if(small||k===0)D.text(g,44,y(v)+4,D.signed(v,1),{class:'diagram-note','text-anchor':'end'});
      });
      MODELS.forEach(function(m){
        var a=cell[m.w+':'+t],cx=x(m.delay),cls=m.accent?' accent':'';
        g.appendChild(D.el('line',{x1:cx,x2:cx,y1:y(a.stock_ci95[0]),y2:y(a.stock_ci95[1]),class:'effect-ci'+cls}));
        g.appendChild(D.el('circle',{cx:cx,cy:y(a.stock),r:4,class:'effect-pt'+cls}));
        D.text(g,cx,264,String(m.delay),{class:'diagram-note','text-anchor':'middle'});
        D.text(g,cx,280,m.label,{class:'diagram-note','text-anchor':'middle'});
        var hit=D.el('rect',{x:cx-10,y:40,width:20,height:208,fill:'transparent'});g.appendChild(hit);
        D.bindReadout(hit,function(){
          readout.textContent=m.label+' · '+m.delay+' frames · net stocks/min '+D.signed(a.stock,2)+' ['+D.signed(a.stock_ci95[0],2)+', '+D.signed(a.stock_ci95[1],2)+'] · p99 '+a.latency_p99_ms.toFixed(1)+' ms · '+a.run;
        },m.label+' at data exponent '+t);
      });
    });
  }
  render();
  window.matchMedia('(max-width:760px)').addEventListener('change',render);
}());

/* Fig: O54 hold-out — S(D) fitted through D = 2^30 per model, prediction against observation at 2^31. */
(function () {
  'use strict';
  var D = window.HALDIAG, DATA = window.HALFIG_DATA;
  var svg = document.getElementById('tier-prediction-diagram');
  var readout = document.getElementById('tier-prediction-readout');
  if (!D || !DATA || !svg) return;
  var T = DATA.tier;
  var LABEL = {256: '5M · 1 frame', 512: '39M · 2 frames', 768: '125M · 3 frames', 1024: '289M · 6 frames'};
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.tierPrediction = function () {
    return T.models.map(function (m) { return {model: 'W' + m.width, bc_parameters: m.bc_parameters, run: m.run, fit_s28: m.fit.s28, fit_amplitude: m.fit.amplitude, fit_alpha: m.fit.alpha, predicted_d31: m.predicted_d31, observed_d31: m.observed_d31, observed_ci95_lo: m.observed_d31_ci95[0], observed_ci95_hi: m.observed_d31_ci95[1]}; });
  };
  function curve(m, e) { return m.fit.s28 + m.fit.amplitude * (1 - Math.pow(Math.pow(2, e - 28), -m.fit.alpha)); }
  var superscript = function (n) { return String(n).replace(/\d/g, function (c) { return '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]; }); };

  function render(){
    D.empty(svg);
  var small=window.matchMedia('(max-width:760px)').matches;
  var W = small?340:880, H = small?1260:330, L = 50, R = 16, top = 54, bottom = 262, pw = small?274:(W - L - R - 3 * 18) / 4;
  svg.classList.add('reflow-chart');
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var y = D.scale([bottom, top], -1.3, 0.2);


  T.models.forEach(function (m, k) {
    var g=D.el('g',{transform:'translate(0,'+(small?k*310:0)+')'});svg.appendChild(g);
    var px = L + (small?0:k*(pw+18));
    if(small||k===0) [-1,-0.5,0].forEach(function(v){D.text(g,px-8,y(v)+4,D.signed(v,1),{class:'diagram-note','text-anchor':'end'});});
    var x = D.scale([px + 16, px + pw - 16], 28, 31);
    [-1, -0.5, 0].forEach(function (v) { g.appendChild(D.el('line', {x1: px, y1: y(v), x2: px + pw, y2: y(v), class: v === 0 ? 'effect-zero' : 'effect-grid'})); });
    D.text(g, px + pw / 2, top - 14, LABEL[m.width], {class: 'diagram-title' + (m.width === 768 ? ' accent' : ''), 'text-anchor': 'middle'});
    [28, 29, 30, 31].forEach(function (e) { D.text(g, x(e), bottom + 16, '2' + superscript(e), {class: 'diagram-note', 'text-anchor': 'middle'}); });
    // fitted curve: solid through 2^30, dashed beyond
    var fitted = [], beyond = [];
    for (var e = 28; e <= 31.0001; e += 0.05) (e <= 30 ? fitted : beyond).push(x(e) + ',' + y(curve(m, e)));
    beyond.unshift(x(30) + ',' + y(curve(m, 30)));
    g.appendChild(D.el('polyline', {points: fitted.join(' '), class: 'chart-line faint'}));
    g.appendChild(D.el('polyline', {points: beyond.join(' '), class: 'chart-line faint', 'stroke-dasharray': '3 3'}));
    m.points.forEach(function (p) {
      var cx = x(p.d_exp), cls = p.d_exp === 31 ? ' accent' : '';
      g.appendChild(D.el('line', {x1: cx, y1: y(p.ci95[0]), x2: cx, y2: y(p.ci95[1]), class: 'effect-ci' + cls}));
      g.appendChild(D.el('circle', {cx: cx, cy: y(p.stock), r: p.d_exp === 31 ? 4.4 : 3.4, class: 'effect-pt' + cls}));
    });
    var pxp = x(31);
    g.appendChild(D.el('circle', {cx: pxp, cy: y(m.predicted_d31), r: 4.4, class: 'pred-pt'}));
    var up = m.predicted_d31 > m.observed_d31;
    D.text(g, up ? pxp - 9 : pxp, y(m.predicted_d31) + (up ? 4 : 17), D.signed(m.predicted_d31, 2), {class: 'diagram-note', 'text-anchor': up ? 'end' : 'middle'});
    D.text(g, x(31), y(m.observed_d31_ci95[m.predicted_d31 > m.observed_d31 ? 0 : 1]) + (m.predicted_d31 > m.observed_d31 ? 15 : -7), D.signed(m.observed_d31, 2), {class: 'diagram-label accent-text', 'text-anchor': 'middle'});
    var hit = D.el('rect', {x: px, y: top, width: pw, height: bottom - top, fill: 'transparent'});
    g.appendChild(hit);
    D.bindReadout(hit, function () {
      readout.innerHTML = '<span><b>' + LABEL[m.width] + '</b> · ' + m.run + '</span><span>fit S₂₈ ' + D.signed(m.fit.s28, 2) + ' · A ' + m.fit.amplitude.toFixed(2) + ' · α ' + m.fit.alpha.toFixed(2) + '</span><span>predicted at 2³¹ <b>' + D.signed(m.predicted_d31, 2) + '</b></span><span>observed <b>' + D.signed(m.observed_d31, 2) + '</b> [' + D.signed(m.observed_d31_ci95[0], 2) + ', ' + D.signed(m.observed_d31_ci95[1], 2) + ']</span>';
    }, LABEL[m.width]);
  });
  }
  render();
  window.matchMedia('(max-width:760px)').addEventListener('change',render);
}());

/* O39 fit: measured data curves, iso-FLOP parabolas with their measured points and predicted minima,
   and the compute-optimal size N_opt(C) the fit implies, extrapolated to the O50 production budget. */
(function(){
 'use strict';
 var D=window.HALDIAG, data=window.HALFIG_DATA;
 var measured=document.getElementById('iso-flop-diagram'), compute=document.getElementById('iso-compute-diagram');
 var optimal=document.getElementById('iso-optimal-diagram'), table=document.getElementById('iso-optimal-table'), residual=document.getElementById('iso-residual-diagram');
 var readout=document.getElementById('iso-flop-readout');
 if(!D||!data||!measured||!compute||!optimal||!table||!residual)return;
 var I=data.iso_flop,F=I.fit,P=I.points,K=I.checkpoints,LEVELS=I.levels,FAM=I.family,U=I.u_total,O50=I.production;
 var EXP=F.dPower/(F.aPower+F.dPower);
 function loss(n,d){return F.base+F.aScale/Math.pow(n,F.aPower)+F.dScale/Math.pow(d,F.dPower);}
 function nOpt(C){return Math.pow(F.aScale*F.aPower/(F.dScale*F.dPower),1/(F.aPower+F.dPower))*Math.pow(C/6,EXP);}
 function sup(n){return String(n).replace(/\d/g,function(c){return '⁰¹²³⁴⁵⁶⁷⁸⁹'[c];}).replace('-','⁻');}
 function fmtN(n){if(n<1e9)return Math.round(n/1e6)+'M';var b=(n/1e9).toFixed(1);return (b.slice(-2)==='.0'?b.slice(0,-2):b)+'B';}
 function fmtE(v){var e=Math.floor(Math.log10(v)),m=v/Math.pow(10,e);if(Math.abs(m-10)<0.02){m=1;e+=1;}if(Math.abs(m-1)<0.02)return '10'+sup(e);var t=m.toFixed(1);return (t.slice(-2)==='.0'?t.slice(0,-2):t)+'×10'+sup(e);}
 function fmtD(d){return fmtE(d)+' ('+(d/U<0.95?(d/U).toFixed(2):(d/U).toFixed(1))+' epochs)';}
 function fmtInt(n){return Math.round(n).toLocaleString('en-US');}
 function fmtC(v){var e=Math.floor(Math.log10(v)),m=v/Math.pow(10,e);return m.toFixed(2)+'×10'+sup(e)+' FLOPs';}
 function pow2(d){var e=Math.log2(d);return Math.abs(e-Math.round(e))<1e-9?' = 2'+sup(Math.round(e)):'';}
 function epochs(d){return (d/U).toFixed(3)+' epochs of the corpus';}
 var dMin=Math.min.apply(null,K.map(function(c){return c.positions;})),dMax=Math.max.apply(null,K.map(function(c){return c.positions;}));
 function supported(n,d){return n>=FAM.n_min&&n<=FAM.n_max&&d>=dMin&&d<=dMax;}
 function optimum(C){var n=nOpt(C),d=C/(6*n);return {c:C,n:n,d:d,nll:loss(n,d),supported:supported(n,d)};}
 function describe(o){return 'C = '+fmtE(o.c)+' FLOPs · N_opt ≈ '+fmtN(o.n)+' · D = '+fmtD(o.d)+' · predicted NLL '+o.nll.toFixed(3)+(o.supported?' · inside the observed ranges':' · extrapolated');}
 function optimumLines(title,o){return [title,'C = '+fmtC(o.c),'N_opt = '+fmtInt(o.n),'D = '+fmtInt(o.d)+' positions · '+epochs(o.d),'predicted NLL '+o.nll.toFixed(4),o.supported?'inside the observed N and D ranges':'extrapolated beyond the observed ranges'];}
 window.HALFIG_CUSTOM=window.HALFIG_CUSTOM||{data:{}};
 window.HALFIG_CUSTOM.data.isoFlop=function(){return P.map(function(p){return {run:p.run,model:p.model,layers:p.layers,d_model:p.d_model,total_parameters:p.total_parameters,compute_equivalent_parameters:p.n_eff,d_positions_exp:p.d_exp,training_flops:p.training_flops,validation_nll:p.nll,fitted_nll:p.fit_nll,residual:p.residual};});};
 window.HALFIG_CUSTOM.data.isoFlopLevels=function(){var rows=[];LEVELS.forEach(function(l){l.points.forEach(function(p){rows.push({training_flops:l.c,model:p.model,compute_equivalent_parameters:p.n_eff,validation_nll:p.nll,exact:p.exact,runs:p.runs});});});return rows;};

 var tip=document.createElement('div');tip.className='iso-tip';tip.setAttribute('aria-hidden','true');document.body.appendChild(tip);
 function placeTip(x,y){var w=tip.offsetWidth,h=tip.offsetHeight,X=x+14,Y=y+16;if(X+w>window.innerWidth-8)X=Math.max(8,x-w-14);if(Y+h>window.innerHeight-8)Y=Math.max(8,y-h-16);tip.style.left=X+'px';tip.style.top=Y+'px';}
 function showTip(lines,x,y){while(tip.firstChild)tip.removeChild(tip.firstChild);lines.forEach(function(l,i){var d=document.createElement('div');if(!i)d.className='iso-tip-title';d.textContent=l;tip.appendChild(d);});tip.classList.add('on');placeTip(x,y);}
 function hideTip(){tip.classList.remove('on');}
 document.addEventListener('pointerdown',function(e){if(!e.target.closest||!e.target.closest('.iso-hit'))hideTip();});
 function highlight(svg,series){Array.prototype.forEach.call(svg.querySelectorAll('[data-series]'),function(e){e.classList.toggle('dim',!!series&&e.getAttribute('data-series')!==series);});}
 function clearable(svg){svg.addEventListener('mouseleave',function(){highlight(svg,null);});svg.addEventListener('focusout',function(e){if(!svg.contains(e.relatedTarget))highlight(svg,null);});}
 // a generous transparent hit target over a mark: pointer → tooltip at the pointer, focus → tooltip at the mark, both → readout line
 function bind(svg,el,series,lines){
  var message=lines.join(' · '),hit=D.el('circle',{cx:el.getAttribute('cx')||el.getAttribute('data-cx'),cy:el.getAttribute('cy')||el.getAttribute('data-cy'),r:9,fill:'transparent',class:'iso-hit'});
  svg.appendChild(hit);
  D.bindReadout(hit,function(e){
   readout.textContent=message;highlight(svg,series);
   var r=el.getBoundingClientRect(),pointer=e&&typeof e.clientX==='number'&&e.type!=='focus';
   showTip(lines,pointer?e.clientX:r.left+r.width/2,pointer?e.clientY:r.bottom);
  },message);
  hit.addEventListener('mousemove',function(e){placeTip(e.clientX,e.clientY);});
  hit.addEventListener('mouseleave',hideTip);hit.addEventListener('blur',hideTip);
 }
 function inkRamp(i,n){return 'color-mix(in srgb,var(--ink) '+Math.round(25+75*i/(n-1))+'%,var(--paper))';}
 function oxRamp(i,n){return 'color-mix(in srgb,var(--oxblood) '+Math.round(35+65*i/(n-1))+'%,var(--paper))';}
 function diamond(cx,cy,r){return 'M'+cx+' '+(cy-r)+'L'+(cx+r)+' '+cy+'L'+cx+' '+(cy+r)+'L'+(cx-r)+' '+cy+'Z';}
 function star(cx,cy,r){var d='';for(var i=0;i<10;i++){var a=-Math.PI/2+i*Math.PI/5,rr=i%2?r*0.45:r;d+=(i?'L':'M')+(cx+rr*Math.cos(a)).toFixed(1)+' '+(cy+rr*Math.sin(a)).toFixed(1);}return d+'Z';}
 // draw v(n) over the size axis as solid segments where (n, d(n)) is inside the observed ranges and dashed ones outside
 function curve(svg,x,y,fn,dOf,style,series,range){
  var segment=[],segmentSupported=null,lo=range?range[0]:x.lo,hi=range?range[1]:x.hi,last=null;
  function flush(){if(segment.length>1)svg.appendChild(D.el('polyline',{points:segment.join(' '),fill:'none',stroke:style,'stroke-width':1.6,'data-series':series,'stroke-dasharray':segmentSupported?null:'4 4'}));segment=[];}
  for(var j=0;j<200;j++){
   var n0=lo*Math.pow(hi/lo,j/200),n1=lo*Math.pow(hi/lo,(j+1)/200),v0=fn(n0),v1=fn(n1);
   if(v0<y.lo||v0>y.hi||v1<y.lo||v1>y.hi){flush();continue;}
   var sup_=supported(n0,dOf(n0))&&supported(n1,dOf(n1));
   if(segmentSupported!==sup_){flush();segmentSupported=sup_;}
   if(!segment.length)segment.push(x(n0)+','+y(v0));segment.push(x(n1)+','+y(v1));last={n:n1,v:v1};
  }
  flush();
  return last;
 }

 function render(){
  var W=window.matchMedia('(max-width:760px)').matches?340:420,H=340;
  [measured,compute,optimal,residual].forEach(function(svg){D.empty(svg);svg.setAttribute('viewBox','0 0 '+W+' '+H);});
  function sizeAxis(nLo,nHi){var x=function(n){return 50+Math.log(n/nLo)/Math.log(nHi/nLo)*(W-76);};x.lo=nLo;x.hi=nHi;return x;}
  var xm=sizeAxis(FAM.n_min/1.4,FAM.n_max*3),xc=sizeAxis(FAM.n_min/5,FAM.n_max*2.5);
  var y=D.scale([250,32],1.7,2.1);y.lo=1.7;y.hi=2.1;
  function lossAxes(svg,x){
   [1.7,1.8,1.9,2,2.1].forEach(function(v){svg.appendChild(D.el('line',{x1:50,x2:W-20,y1:y(v),y2:y(v),class:'effect-grid'}));D.text(svg,42,y(v)+4,v.toFixed(1),{class:'diagram-note','text-anchor':'end'});});
   [1e7,3e7,1e8,3e8,1e9].forEach(function(n){if(x(n)>=50&&x(n)<=W-15)D.text(svg,x(n),270,fmtN(n),{class:'diagram-note','text-anchor':'middle'});});
   D.text(svg,50,18,'validation NLL',{class:'diagram-note'});
   D.text(svg,50,292,'compute-equivalent parameters · log scale',{class:'diagram-note'});
  }

  // measured runs on the fitted data curves
  lossAxes(measured,xm);
  [26,27,28,29,30].forEach(function(e,i){
   var shade=inkRamp(i,5),d=Math.pow(2,e),series='d'+e,x=xm;
   var end=curve(measured,x,y,function(n){return loss(n,d);},function(){return d;},shade,series,[FAM.n_min,FAM.n_max]);
   D.text(measured,x(end.n)+5,y(end.v)+4,'2'+sup(e),{class:'diagram-note','data-series':series});
   P.filter(function(p){return p.d_exp===e;}).forEach(function(p){
    var dot=D.el('circle',{cx:x(p.n_eff),cy:y(p.nll),r:4,fill:shade,stroke:'var(--paper)','stroke-width':1,'data-series':series});measured.appendChild(dot);
    bind(measured,dot,series,[p.model+' · run '+p.run,'N = '+fmtInt(p.n_eff)+' (trunk + 36 × decoder; '+fmtInt(p.total_parameters)+' weights)','D = '+fmtInt(d)+' positions'+pow2(d),'C = '+fmtC(p.training_flops),'NLL '+p.nll.toFixed(4)+' · fit '+p.fit_nll.toFixed(4)+' · residual '+D.signed(p.residual,4)]);
   });
  });
  D.text(measured,50,320,'Curve labels: D = supervised positions',{class:'diagram-note'});

  // iso-FLOP parabolas: L(N, C/6N) at four budgets, the runs measured at each, and the predicted minimum
  lossAxes(compute,xc);
  var lastTopLabel=-1e9;
  LEVELS.forEach(function(level,i){
   var C=level.c,shade=oxRamp(i,LEVELS.length),series='c'+i,o=optimum(C),x=xc;
   var end=curve(compute,x,y,function(n){return loss(n,C/(6*n));},function(n){return C/(6*n);},shade,series);
   if(end.n>=x.hi/1.02)D.text(compute,x(end.n),y(end.v)-8,fmtE(C),{class:'diagram-note','data-series':series,'text-anchor':'end'});
   else{var lift=x(end.n)-lastTopLabel<44?17:5;lastTopLabel=x(end.n);D.text(compute,x(end.n)+2,y(end.v)-lift,fmtE(C),{class:'diagram-note','data-series':series,'text-anchor':'middle'});}
   level.points.forEach(function(p){
    var dot=D.el('circle',{cx:x(p.n_eff),cy:y(p.nll),r:4,fill:p.exact?shade:'var(--paper)',stroke:shade,'stroke-width':1.6,'data-series':series});compute.appendChild(dot);
    var lines=[p.model+' at C = '+fmtE(C)+' FLOPs','N = '+fmtInt(p.n_eff),'NLL '+p.nll.toFixed(4)];
    if(p.exact){var ck=K.filter(function(c){return c.run===p.runs[0];})[0];lines.push('run '+p.runs[0]+' trained to this budget','D = '+fmtInt(ck.positions)+' positions'+pow2(ck.positions),'C = '+fmtC(ck.training_flops));}
    else lines.push('interpolated in log C between runs '+p.runs[0]+' and '+p.runs[1],'D = '+fmtInt(p.between_positions[0])+' and '+fmtInt(p.between_positions[1])+' positions');
    bind(compute,dot,series,lines);
   });
   var mark=D.el('path',{d:diamond(x(o.n),y(o.nll),5.5),fill:'var(--paper)',stroke:shade,'stroke-width':1.8,'data-series':series,'data-cx':x(o.n),'data-cy':y(o.nll)});compute.appendChild(mark);
   bind(compute,mark,series,optimumLines('Predicted minimum at C = '+fmtE(C),o));
  });
  D.text(compute,50,320,'Curve labels: C = training FLOPs',{class:'diagram-note'});
  clearable(measured);clearable(compute);

  // compute-optimal size against budget, log–log, extrapolated to the O50 budget
  var cLo=1e16,cHi=1e20,mLo=1e7,mHi=3e9;
  var cx=function(C){return 60+Math.log(C/cLo)/Math.log(cHi/cLo)*(W-84);};
  var my=function(n){return 250-Math.log(n/mLo)/Math.log(mHi/mLo)*(250-32);};
  var band=D.el('rect',{x:60,y:my(FAM.n_max),width:W-84,height:my(FAM.n_min)-my(FAM.n_max),fill:'var(--rule)'});optimal.appendChild(band);
  [1e7,1e8,1e9].forEach(function(n){optimal.appendChild(D.el('line',{x1:60,x2:W-24,y1:my(n),y2:my(n),class:'effect-grid'}));D.text(optimal,52,my(n)+4,fmtN(n),{class:'diagram-note','text-anchor':'end'});});
  [1e16,1e17,1e18,1e19,1e20].forEach(function(C){D.text(optimal,cx(C),270,fmtE(C),{class:'diagram-note','text-anchor':'middle'});});
  D.text(optimal,60,18,'compute-optimal parameters',{class:'diagram-note'});
  D.text(optimal,60,292,'training FLOPs · log scale',{class:'diagram-note'});
  D.text(optimal,W-24,my(FAM.n_min)-5,'sizes the fit saw',{class:'diagram-note','text-anchor':'end'});
  var pts=[];for(var j=0;j<=100;j++){var C=cLo*Math.pow(cHi/cLo,j/100);pts.push(cx(C)+','+my(nOpt(C)));}
  optimal.appendChild(D.el('polyline',{points:pts.join(' '),class:'chart-line'}));
  D.text(optimal,cx(2e18)+10,my(nOpt(2e18))+16,'N_opt ∝ C'+sup(EXP.toFixed(2).replace('0.','⁰·')),{class:'diagram-note'});
  var prod=optimum(O50.flops),actualNll=loss(O50.n_eff,O50.positions);
  var cursor=D.el('g',{class:'iso-cursor',visibility:'hidden'});
  cursor.appendChild(D.el('line',{x1:0,x2:0,y1:32,y2:250}));cursor.appendChild(D.el('circle',{r:4}));
  optimal.appendChild(cursor);
  LEVELS.forEach(function(level){var o=optimum(level.c),mark=D.el('path',{d:diamond(cx(o.c),my(o.n),5.5),fill:'var(--paper)',stroke:'var(--ink)','stroke-width':1.8,'data-cx':cx(o.c),'data-cy':my(o.n)});optimal.appendChild(mark);bind(optimal,mark,null,optimumLines('Predicted optimum at C = '+fmtE(o.c),o));});
  var starMark=D.el('path',{d:star(cx(prod.c),my(prod.n),9),fill:'var(--oxblood)','data-cx':cx(prod.c),'data-cy':my(prod.n)});optimal.appendChild(starMark);
  bind(optimal,starMark,null,optimumLines('Predicted optimum at the O50 budget',prod).concat(['O50 as trained sat at N = '+fmtInt(O50.n_eff)+', fit NLL '+actualNll.toFixed(4)]));
  var ring=D.el('circle',{cx:cx(O50.flops),cy:my(O50.n_eff),r:5,fill:'var(--oxblood)',stroke:'var(--paper)','stroke-width':1.2});optimal.appendChild(ring);
  bind(optimal,ring,null,['O50 as trained · run '+O50.run,'C = '+fmtC(O50.flops)+' = 6 · D · N','N = '+fmtInt(O50.n_eff)+' = '+fmtInt(O50.trunk)+' + '+O50.passes+' × '+fmtInt(O50.decoder),'D = '+fmtInt(O50.positions)+' positions'+pow2(O50.positions)+' · '+epochs(O50.positions),'fit NLL '+actualNll.toFixed(4)+' here vs '+prod.nll.toFixed(4)+' at N_opt = '+fmtInt(prod.n)]);
  D.text(optimal,cx(prod.c)-13,my(prod.n)-9,'O50 budget',{class:'diagram-note','text-anchor':'end'});
  function sweep(e){
   if(e.target.classList.contains('iso-hit')){cursor.setAttribute('visibility','hidden');return;}
   var r=optimal.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*W;
   if(px<60||px>W-24){cursor.setAttribute('visibility','hidden');return;}
   var C=cLo*Math.pow(cHi/cLo,(px-60)/(W-84)),o=optimum(C);
   cursor.setAttribute('visibility','visible');cursor.setAttribute('transform','translate('+cx(C)+',0)');
   cursor.querySelector('circle').setAttribute('cy',my(o.n));
   readout.textContent=describe(o);
  }
  optimal.onpointermove=sweep;optimal.onpointerdown=sweep;
  optimal.onpointerleave=function(){cursor.setAttribute('visibility','hidden');};

  // the same optima as numbers, for readers without a pointer
  var body=table.querySelector('tbody');while(body.firstChild)body.removeChild(body.firstChild);
  function row(cells,cls){var tr=document.createElement('tr');if(cls)tr.className=cls;cells.forEach(function(c){var td=document.createElement('td');td.textContent=c;tr.appendChild(td);});body.appendChild(tr);}
  LEVELS.map(function(l){return optimum(l.c);}).concat([prod]).forEach(function(o,i){
   row([fmtE(o.c),fmtN(o.n),fmtE(o.d),(o.d/U).toFixed(o.d/U<0.95?2:1),o.nll.toFixed(3),i===LEVELS.length?'O50 budget':o.supported?'':'extrapolated'],i===LEVELS.length?'accent':null);
  });
  row([fmtE(O50.flops),fmtN(O50.n_eff),fmtE(O50.positions),(O50.positions/U).toFixed(1),actualNll.toFixed(3),'O50 as trained'],'accent');

  // residuals of the published fit
  var worst=Math.max.apply(null,P.map(function(p){return Math.abs(p.residual);})),rx=D.scale([50,W-24],1.76,2.06),ry=D.scale([250,36],-worst*1.3,worst*1.3);
  [-0.01,0,0.01].forEach(function(v){residual.appendChild(D.el('line',{x1:50,x2:W-24,y1:ry(v),y2:ry(v),class:v===0?'effect-zero':'effect-grid'}));D.text(residual,42,ry(v)+4,D.signed(v,2),{class:'diagram-note','text-anchor':'end'});});
  P.forEach(function(p){var dot=D.el('circle',{cx:rx(p.fit_nll),cy:ry(p.nll-p.fit_nll),r:3.5,class:'effect-pt'});residual.appendChild(dot);bind(residual,dot,null,[p.model+' · run '+p.run+' · D = 2'+sup(p.d_exp),'measured '+p.nll.toFixed(4)+' − fitted '+p.fit_nll.toFixed(4)+' = '+D.signed(p.residual,4)]);});
  [1.8,1.9,2].forEach(function(v){D.text(residual,rx(v),272,v.toFixed(1),{class:'diagram-note','text-anchor':'middle'});});
  D.text(residual,50,20,'measured − fitted NLL',{class:'diagram-note'});
  D.text(residual,50,294,'fitted NLL',{class:'diagram-note'});
  D.text(residual,50,322,'largest absolute residual '+worst.toFixed(3)+' · none held out',{class:'diagram-note'});
 }
 render();window.matchMedia('(max-width:760px)').addEventListener('change',render);
}());
