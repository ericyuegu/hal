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
  var W = 880, H = 250;
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
}());
