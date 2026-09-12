/* Fig: the policy end to end — the page's model-diagram grammar, one causal trunk, one causal decoder, five cards. */
(function () {
  'use strict';
  var D = window.HALDIAG;
  var svg = document.getElementById('arch-overview');
  var cards = document.getElementById('arch-cards');
  var detail = document.getElementById('arch-detail');
  if (!D || !svg || !cards || !detail) return;

  var FAINT = 'var(--ink-faint)', OX = 'var(--oxblood)';
  var frames = [['sₜ₋₂', 'aₜ₋₂'], ['sₜ₋₁', 'aₜ₋₁'], ['sₜ', 'aₜ']];

  /* ---------------------------------------------------------------- overview */
  function renderOverview() {
    D.empty(svg);
    var mk = D.marker(svg, 'arrow', FAINT);
    svg.setAttribute('viewBox', '0 0 880 346');
    var centers = [155, 440, 725], tokenW = 174;

    D.note(svg, 440, 16, 'ONE JOINT TOKEN PER FRAME', 'middle');
    var hist = D.part(svg, 'history');
    frames.forEach(function (f, i) {
      D.jointToken(hist, centers[i] - tokenW / 2, 31, tokenW, 38, f[0], f[1]);
      D.wire(hist, centers[i], 69, centers[i], 98, mk);
    });
    D.note(hist, 68, 55, '···', 'middle');

    var trunk = D.part(svg, 'trunk');
    D.stacked(trunk, 70, 106, 740, 68, 'CAUSAL CONTEXT TRANSFORMER', 'every position reads only the frames before it · all positions run in parallel');

    // one decoder; the trunk state is the arrow, not a box
    var temporal = D.part(svg, 'temporal');
    D.wire(temporal, centers[2], 174, centers[2], 206, mk);
    D.text(temporal, centers[2] + 8, 194, 'hₜ', {class: 'diagram-label'});
    D.box(temporal, centers[2] - 100, 210, 200, 40, 'CAUSAL ACTION DECODER');
    D.note(temporal, 297, 222, 'ONE DECODER', 'middle');
    D.note(temporal, 297, 238, 'in play it is handed the state for the current frame', 'middle');
    D.note(temporal, 297, 252, 'in training every position’s state goes through the same decoder at once', 'middle');
    D.wire(temporal, centers[2], 250, centers[2], 274, mk);

    var groups = D.part(svg, 'groups');
    var play = D.part(svg, 'play');
    var cellW = 40, gap = 6, n = 4;
    var x0 = centers[2] - (cellW * n + gap * (n - 1)) / 2 - 10;
    ['aₜ₊₁', 'aₜ₊₂', 'aₜ₊₃', 'aₜ₊₄'].forEach(function (label, i) {
      D.token(i < 2 ? play : groups, x0 + i * (cellW + gap), 282, cellW, 34, label, i < 2 ? 'played' : '');
    });
    D.note(groups, x0 + n * (cellW + gap) + 4, 304, '…');
    var rowEnd = x0 + n * (cellW + gap) - gap;
    D.note(play, x0, 334, 'played', 'start', 'play');
    D.note(play, rowEnd, 334, 'then replaced', 'end');

    // click targets over each module
    [['history', 60, 24, 760, 52], ['trunk', 60, 100, 760, 82], ['temporal', 190, 184, 640, 74], ['groups', 620, 262, 220, 82]].forEach(function (h) {
      var r = D.el('rect', {x: h[1], y: h[2], width: h[3], height: h[4], class: 'hit', 'data-hit': h[0], tabindex: 0, role: 'button'});
      r.addEventListener('click', function () { select(selected === keyFor(h[0]) ? null : keyFor(h[0])); });
      r.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(selected === keyFor(h[0]) ? null : keyFor(h[0])); } });
      svg.appendChild(r);
    });
  }
  function keyFor(moduleId) { return {history: 'token', trunk: 'trunk', temporal: 'decoder', groups: 'frame', play: 'loop'}[moduleId]; }

  /* ---------------------------------------------------------------- card content: mechanism drawings in the same grammar */
  function mechToken(s) {
    var mk = D.marker(s, 'arrow', FAINT);
    s.setAttribute('viewBox', '0 0 880 190');
    D.note(s, 16, 16, 'WHAT ONE FRAME CONTRIBUTES');
    var boxes = [['GAME STATE', 'both players · projectiles · stage', 30], ['WHO IS PLAYING', 'characters · the player being imitated', 80], ['PREVIOUS CONTROLLER FRAME', 'the action the game just applied', 130]];
    boxes.forEach(function (b) {
      s.appendChild(D.el('rect', {x: 16, y: b[2], width: 340, height: 40, class: 'model-token'}));
      D.text(s, 28, b[2] + 17, b[0], {class: 'diagram-label'});
      D.note(s, 28, b[2] + 31, b[1]);
    });
    var g = D.part(s, 'history');
    D.jointToken(g, 470, 66, 174, 38, 'sₜ', 'aₜ');
    D.note(s, 513, 56, 'state half', 'middle');
    D.note(s, 600, 56, 'action half', 'middle');
    D.path(s, 'M 360 50 C 420 50, 420 78, 464 78', mk);
    D.path(s, 'M 360 100 C 420 100, 420 92, 464 92', mk);
    D.path(s, 'M 360 150 C 480 150, 600 150, 600 110', mk);
    D.note(s, 660, 82, 'one linear projection');
    D.note(s, 660, 96, 'no hand-built features');
  }
  function mechTrunk(s) {
    var mk = D.marker(s, 'arrow', FAINT);
    s.setAttribute('viewBox', '0 0 880 210');
    D.note(s, 16, 16, 'GIVEN · this frame and the frames before it');
    D.note(s, 700, 16, 'HANDED TO THE DECODER · one state per frame', 'middle');
    var w = 52, gap = 8, y0 = 32, rowH = 30, step = 44;
    var right = 560; // the prefixes end here so every arrow starts on the same column
    var outs = ['hₜ₋₂', 'hₜ₋₁', 'hₜ'];
    frames.forEach(function (_, i) {
      var y = y0 + i * step;
      var given = frames.slice(0, i + 1);
      var x = right - given.length * (w * 2 + gap) + gap;
      given.forEach(function (f, k) { D.jointToken(s, x + k * (w * 2 + gap), y, w * 2, rowH, f[0], f[1], 'given'); });
      D.wire(s, right + 6, y + rowH / 2, 630, y + rowH / 2, mk);
      D.text(s, right + 40, y + rowH / 2 - 5, outs[i], {class: 'diagram-label', 'text-anchor': 'middle'});
    });
    D.box(s, 640, y0, 220, 2 * step + rowH, 'CAUSAL ACTION DECODER');
    D.note(s, 440, 190, 'every row runs at once · a state never sees the frames after its own · the state is what the arrow carries', 'middle');
  }
  function mechDecoder(s) {
    var mk = D.marker(s, 'arrow', FAINT);
    s.setAttribute('viewBox', '0 0 880 330');
    D.note(s, 16, 16, 'ONE DECODER, ENLARGED');
    var hist = D.part(s, 'history');
    D.note(hist, 81, 94, 'CURRENT FRAME', 'middle');
    D.jointToken(hist, 16, 106, 130, 40, 'sₜ', 'aₜ');
    var temporal = D.part(s, 'temporal');
    D.wire(temporal, 146, 126, 178, 126, mk);
    D.text(temporal, 152, 118, 'hₜ', {class: 'diagram-label'});
    D.stacked(temporal, 186, 32, 670, 270, 'CAUSAL ACTION DECODER');
    D.note(temporal, 220, 77, 'GIVEN · the frames written so far');
    D.note(temporal, 700, 77, 'PRODUCED · the next frame, group by group', 'middle');
    var prefixes = [['aₜ'], ['aₜ', 'aₜ₊₁'], ['aₜ', 'aₜ₊₁', 'aₜ₊₂'], ['aₜ', 'aₜ₊₁', 'aₜ₊₂', 'aₜ₊₃']];
    var outs = ['aₜ₊₁', 'aₜ₊₂', 'aₜ₊₃', 'aₜ₊₄'];
    prefixes.forEach(function (p, i) {
      var y = 92 + i * 44;
      D.row(temporal, 220, y, p, [outs[i]], {w: 44, h: 28, gap: 6, arrow: 40, mk: mk});
      var cx = 606;
      ['C', 'M', 'T', 'B'].forEach(function (gname, k) {
        var x = cx + k * 38;
        D.token(temporal, x, y, 31, 28, gname);
        if (k) D.wire(temporal, x - 7, y + 14, x - 1, y + 14, mk);
      });
      D.wire(temporal, 220 + p.length * 50 - 6 + 40 + 44 + 6, y + 14, 599, y + 14, mk);
    });
    D.note(temporal, 521, 285, 'all rows at once in training · one row per frame in play', 'middle');
  }
  function mechFrame(s) {
    var mk = D.marker(s, 'arrow', FAINT);
    s.setAttribute('viewBox', '0 0 880 200');
    D.note(s, 16, 16, 'GIVEN · the groups already chosen in this frame');
    D.note(s, 300, 16, 'PRODUCED');
    var names = {C: 'C-stick', M: 'main stick', T: 'triggers', B: 'buttons'};
    var chain = ['C', 'M', 'T', 'B'];
    var w = 40, gap = 6, right = 200;
    for (var i = 1; i < chain.length; i++) {
      var y = 32 + (i - 1) * 40;
      var given = chain.slice(0, i);
      var x = right - given.length * (w + gap) + gap;
      D.row(s, x, y, given, [chain[i]], {w: w, h: 28, gap: gap, arrow: 48, mk: mk});
      D.note(s, 300, y + 18, names[chain[i]] + ', knowing ' + given.map(function (c) { return names[c]; }).join(' and '));
    }
    D.note(s, 16, 176, 'C-stick first, then main stick, triggers, buttons · each choice shifts the state that makes the next one');
  }
  function mechLoop(s) {
    s.setAttribute('viewBox', '0 0 880 190');
    D.note(s, 16, 16, 'TWO PLANS, TWO FRAMES APART');
    var cellW = 64, gap = 8, x0 = 150;
    function planRow(y, label, offset) {
      D.note(s, 16, y + 20, label);
      for (var f = 0; f < 6; f++) {
        var frame = offset + f + 1;
        var cls = f < 2 ? 'queued' : f < 4 ? 'played' : '';
        D.token(s, x0 + (offset + f) * (cellW + gap), y, cellW, 30, 'aₜ₊' + subscript(frame), cls);
      }
    }
    planRow(38, 'plan made at t', 0);
    planRow(96, 'plan made at t+2', 2);
    var lx = x0;
    D.note(s, lx + cellW + gap / 2, 30, 'already in the game', 'middle');
    D.note(s, lx + 3 * (cellW + gap) + cellW / 2 - gap / 2, 30, 'played', 'middle', 'play');
    D.note(s, lx + 5 * (cellW + gap) + cellW / 2 - gap / 2, 30, 'written, then replaced', 'middle');
    D.note(s, 16, 158, 'the game is always two frames ahead of the model');
    D.note(s, 16, 172, 'a plan is handed the two frames it cannot change, and must be ready before its own two are used up');
  }
  function subscript(n) { return String(n).split('').map(function (c) { return '₀₁₂₃₄₅₆₇₈₉'[Number(c)]; }).join(''); }

  var SPEC = {
    token: {n: '01', t: 'Observation token', r: 'one frame, one token', module: 'history', mech: mechToken, p: [
      'Every frame becomes one token that carries both halves of the moment: what the game looked like and what the player did. The state half holds both players, any projectiles, and the stage; the action half is the controller frame the game just applied, entered through the same class set the decoder predicts rather than as a raw code.',
      'Everything is projected together in one linear map. There are no hand-built geometric features — earlier experiments that added them never produced a clean win, and the plain projection is what scaled.']},
    trunk: {n: '02', t: 'Context trunk', r: 'reads the frames before', module: 'trunk', mech: mechTrunk, p: [
      'A plain causal Transformer. Each position produces a state for its frame from that frame and the ones before it, never the ones after. Because every position is computed in the same pass, one training window supervises the decoder at many positions at once.',
      'In play only the newest state matters; the rest exist so that training sees a whole trajectory of decisions for the price of one forward pass.']},
    decoder: {n: '03', t: 'Action decoder', r: 'writes frames in order', module: 'temporal', mech: mechDecoder, p: [
      'One small causal Transformer, not a head per frame. It is handed the trunk state for the current frame and the controller frames written so far, and produces the next one. In training every prefix of the target chunk goes through it in parallel, teacher-forced; in play it advances one frame at a time on a cache.',
      'It is trained to write further ahead than the game ever uses. The far frames are a training signal that shapes the trunk, not a plan.']},
    frame: {n: '04', t: 'Controller frame', r: 'C → M → T → B', module: 'groups', mech: mechFrame, p: [
      'Inside a frame the four controller groups are chosen in a fixed order, and each choice is made knowing the ones before it: the C-stick class conditions the main-stick choice, both condition the triggers, and all three condition the buttons.',
      'This within-frame feedback and the across-frame feedback above are separate axes; the factorization ablation later in the post measures each on its own.']},
    loop: {n: '05', t: 'Play loop', r: 'a new plan every two frames', module: 'play', mech: mechLoop, p: [
      'Online play holds every input for two frames before the game applies it, so when a plan is made the next two frames are already committed. The decoder is handed those two, writes the frames after them, and the runtime plays two of those before planning again.',
      'The frame the game actually played comes back as the action half of the next token. The loop closes through the data, not inside the model.']}
  };
  var ORDER = ['token', 'trunk', 'decoder', 'frame', 'loop'];
  var selected = null;

  function select(key) {
    selected = key;
    D.focus(svg, key ? SPEC[key].module : null);
    cards.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-selected', String(b.dataset.stage === key)); });
    detail.replaceChildren();
    if (!key) { detail.hidden = true; return; }
    var s = SPEC[key];
    var row = document.createElement('div'); row.className = 'row';
    var h = document.createElement('h5'); h.textContent = s.n + ' · ' + s.t;
    var close = document.createElement('button'); close.type = 'button'; close.className = 'close'; close.textContent = 'close';
    close.addEventListener('click', function () { select(null); });
    row.appendChild(h); row.appendChild(close);
    detail.appendChild(row);
    var mech = document.createElement('div'); mech.className = 'mech';
    var ms = D.el('svg', {class: 'model-diagram', id: 'arch-mech-' + key, role: 'img', 'aria-label': s.t});
    mech.appendChild(ms);
    detail.appendChild(mech);
    s.mech(ms);
    var prose = document.createElement('div'); prose.className = 'prose';
    s.p.forEach(function (t) { var p = document.createElement('p'); p.textContent = t; prose.appendChild(p); });
    detail.appendChild(prose);
    detail.hidden = false;
  }

  ORDER.forEach(function (key) {
    var s = SPEC[key];
    var b = document.createElement('button');
    b.type = 'button'; b.setAttribute('role', 'tab'); b.dataset.stage = key; b.setAttribute('aria-selected', 'false');
    b.innerHTML = '<span class="n">' + s.n + '</span><span class="t">' + s.t + '</span><span class="r">' + s.r + '</span>';
    b.addEventListener('click', function () { select(selected === key ? null : key); });
    cards.appendChild(b);
  });
  renderOverview();
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && selected) select(null); });
}());
