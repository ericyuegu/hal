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
