/* Fig: the compact table of ideas that did not help or did not finish, built from the study table. */
(function () {
  'use strict';
  var D = window.HALDIAG, CI = window.HALFIG_CI;
  var table = document.getElementById('failed-ideas-table');
  if (!D || !CI || !table) return;
  var GROUPS = [
    {title: 'Tested against a matched baseline · did not help', rows: [
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
    if (ci) return {text: fmt(ci.stock), unit: 'stocks / min · ' + (ci.method === 'welch' ? 'unpaired' : 'paired'), num: true};
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
      html += '<tr><td class="idea">' + r.idea + '</td><td>' + r.what + '</td><td class="res' + (res.num ? '' : ' soft') + '"><b>' + res.text + '</b><span>' + res.unit + '</span></td><td>' + r.also + '</td><td><span class="ev ev-' + (s.evidence || 'missing') + '">' + (s.evidence || '').replace('_', ' ') + '</span></td></tr>';
    });
    html += '</tbody>';
  });
  table.innerHTML = html;
  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.failedIdeas = function () { return rows; };
}());
