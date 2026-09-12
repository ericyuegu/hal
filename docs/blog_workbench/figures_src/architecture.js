/* Fig: the policy end to end — one causal trunk, one causal decoder, five clickable stages. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var host = document.getElementById('arch-diagram');
  var cards = document.getElementById('arch-cards');
  var detail = document.getElementById('arch-detail');
  if (!host || !cards || !detail) return;

  function svgEl(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, String(attrs[k])); });
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function sub(parent, main, s) {
    // "a" + subscript via tspans
    var t = svgEl('text', {x: parent.x, y: parent.y, 'text-anchor': parent.anchor || 'middle', 'class': parent.cls || 'sym'});
    t.appendChild(svgEl('tspan', {}, main));
    t.appendChild(svgEl('tspan', {'baseline-shift': 'sub', 'font-size': '75%'}, s));
    return t;
  }
  function markerFor(cls) { return 'url(#arch-arrow-' + (cls.indexOf('chunk') !== -1 ? 'ox' : 'faint') + ')'; }
  function arc(x1, x2, y, lift, cls, step) {
    var p = svgEl('path', {d: 'M ' + x1 + ' ' + y + ' Q ' + ((x1 + x2) / 2) + ' ' + (y - lift) + ' ' + x2 + ' ' + y, 'class': cls, 'marker-end': markerFor(cls)});
    if (step !== undefined) p.style.setProperty('--step', step);
    return p;
  }
  function line(x1, y1, x2, y2, cls, step, arrow) {
    var l = svgEl('path', {d: 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2, 'class': cls, 'marker-end': arrow === false ? '' : markerFor(cls)});
    if (step !== undefined) l.style.setProperty('--step', step);
    return l;
  }

  var svg = svgEl('svg', {viewBox: '0 0 920 262', role: 'img', 'aria-label': 'Observed frames become joint tokens, one causal context Transformer reads them, its state for the current frame feeds one causal action decoder, which writes controller frames one at a time; the game plays the first two.'});
  var defs = svgEl('defs');
  [['faint', 'var(--ink-faint)'], ['ox', 'var(--oxblood)']].forEach(function (pair) {
    var marker = svgEl('marker', {id: 'arch-arrow-' + pair[0], viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '5', markerHeight: '5', orient: 'auto', markerUnits: 'strokeWidth'});
    marker.appendChild(svgEl('path', {d: 'M0 1 L10 5 L0 9 Z', fill: pair[1]}));
    defs.appendChild(marker);
  });
  svg.appendChild(defs);

  var stages = {};
  function stage(key, label) {
    var g = svgEl('g', {'class': 'stage', 'data-stage': key, tabindex: '0', role: 'button', 'aria-label': label});
    stages[key] = g;
    svg.appendChild(g);
    return g;
  }

  /* ---- 01 observed frames → joint tokens, sitting over the trunk ---- */
  var tok = stage('token', 'Stage 1, observed frames become joint tokens');
  tok.appendChild(svgEl('text', {x: 32, y: 22, 'class': 'lbl'}, '01 · Observed frames → tokens'));
  var tokenX = [96, 196, 296];
  var tokenLabels = ['t−2', 't−1', 't'];
  tok.appendChild(svgEl('text', {x: 52, y: 58, 'class': 'sub', 'text-anchor': 'middle'}, '···'));
  tok.appendChild(svgEl('text', {x: 52, y: 72, 'class': 'tiny', 'text-anchor': 'middle'}, '256 frames'));
  tokenX.forEach(function (x, i) {
    var g = svgEl('g', {'class': 'pulse'});
    g.style.setProperty('--step', i);
    g.appendChild(svgEl('rect', {x: x - 44, y: 38, width: 88, height: 28, 'class': 'plate'}));
    g.appendChild(svgEl('line', {x1: x, y1: 38, x2: x, y2: 66, stroke: 'var(--rule-dotted)'}));
    g.appendChild(sub({x: x - 22, y: 57}, 's', tokenLabels[i]));
    g.appendChild(sub({x: x + 22, y: 57}, 'a', tokenLabels[i]));
    tok.appendChild(g);
    tok.appendChild(line(x, 66, x, 90, 'wire pulse-wire', i));
  });
  tok.appendChild(svgEl('text', {x: 196, y: 34, 'class': 'tiny', 'text-anchor': 'middle'}, 'one joint token per frame: state + the action taken'));

  /* ---- 02 causal context transformer ---- */
  var trunk = stage('trunk', 'Stage 2, causal context Transformer');
  var trunkPlate = svgEl('rect', {x: 32, y: 92, width: 360, height: 60, 'class': 'plate pulse'});
  trunkPlate.style.setProperty('--step', 3);
  trunk.appendChild(trunkPlate);
  trunk.appendChild(svgEl('text', {x: 212, y: 112, 'class': 'lbl', 'text-anchor': 'middle'}, '02 · Causal context Transformer'));
  trunk.appendChild(svgEl('text', {x: 212, y: 126, 'class': 'sub', 'text-anchor': 'middle'}, '16 blocks · d1024 · 16 heads · causal'));
  // across-time wires between the token drop points
  tokenX.forEach(function (x, i) {
    trunk.appendChild(svgEl('circle', {cx: x, cy: 142, r: 2.4, fill: 'var(--ink)'}));
    for (var j = i + 1; j < tokenX.length; j++) trunk.appendChild(arc(x, tokenX[j] - 3, 142, 14 + 6 * (j - i), 'wire time'));
  });
  trunk.appendChild(svgEl('text', {x: 360, y: 146, 'class': 'tiny', 'text-anchor': 'middle'}, 'across time'));
  // h_t leaves the trunk
  trunk.appendChild(line(392, 122, 424, 122, 'wire pulse-wire', 4));
  var hg = svgEl('g', {'class': 'pulse'});
  hg.style.setProperty('--step', 5);
  hg.appendChild(svgEl('rect', {x: 426, y: 108, width: 34, height: 28, 'class': 'plate'}));
  hg.appendChild(sub({x: 443, y: 127}, 'h', 't'));
  trunk.appendChild(hg);
  trunk.appendChild(svgEl('text', {x: 443, y: 152, 'class': 'tiny', 'text-anchor': 'middle'}, 'state at t'));

  /* ---- 03 causal action decoder ---- */
  var dec = stage('decoder', 'Stage 3, causal action decoder');
  dec.appendChild(line(460, 122, 492, 122, 'wire pulse-wire', 6));
  var decPlate = svgEl('rect', {x: 494, y: 92, width: 150, height: 60, 'class': 'plate pulse'});
  decPlate.style.setProperty('--step', 7);
  dec.appendChild(decPlate);
  dec.appendChild(svgEl('text', {x: 569, y: 112, 'class': 'lbl', 'text-anchor': 'middle'}, '03 · Causal action'));
  dec.appendChild(svgEl('text', {x: 569, y: 125, 'class': 'lbl', 'text-anchor': 'middle'}, 'decoder'));
  dec.appendChild(svgEl('text', {x: 569, y: 141, 'class': 'sub', 'text-anchor': 'middle'}, '4 blocks · d512'));

  /* ---- 04 the frames it writes ---- */
  var chunk = stage('frame', 'Stage 4, predicted controller frames');
  var actX = [684, 746, 808, 870];
  var groups = ['C', 'M', 'T', 'B'];
  actX.forEach(function (x, i) {
    dec.appendChild(line(644, 122, x - 24, 122, 'wire pulse-wire', 8 + i * 0.5, false));
    var g = svgEl('g', {'class': 'pulse'});
    g.style.setProperty('--step', 9 + i);
    g.appendChild(svgEl('rect', {x: x - 24, y: 92, width: 48, height: 60, 'class': 'plate'}));
    g.appendChild(sub({x: x, y: 108}, 'a', 't+' + (i + 1)));
    groups.forEach(function (gr, k) {
      g.appendChild(svgEl('rect', {x: x - 20 + k * 10.5, y: 116, width: 9, height: 12, 'class': 'group' + (gr === 'B' ? ' b' : '')}));
      g.appendChild(svgEl('text', {x: x - 15.5 + k * 10.5, y: 125, 'class': 'tiny', 'text-anchor': 'middle', 'font-size': '7'}, gr));
    });
    g.appendChild(svgEl('text', {x: x, y: 144, 'class': 'tiny', 'text-anchor': 'middle'}, i === 0 ? 'next frame' : '+' + (i + 1)));
    chunk.appendChild(g);
    for (var j = i + 1; j < actX.length; j++) chunk.appendChild(arc(x + 6 * (j - i - 1) - 6, actX[j] - 6 * (j - i - 1) - 2, 92, 12 + 9 * (j - i), 'wire chunk'));
  });
  chunk.appendChild(svgEl('text', {x: 777, y: 22, 'class': 'lbl', 'text-anchor': 'middle'}, '04 · Controller frames, one at a time'));
  chunk.appendChild(svgEl('text', {x: 777, y: 36, 'class': 'tiny', 'text-anchor': 'middle'}, 'each reads the frames before it · inside: C → M → T → B'));
  chunk.appendChild(svgEl('text', {x: 900, y: 128, 'class': 'tiny', 'text-anchor': 'middle'}, '… +20'));

  /* ---- 05 the play loop ---- */
  var loop = stage('loop', 'Stage 5, the play loop');
  loop.appendChild(svgEl('path', {d: 'M 660 166 L 660 176 L 770 176 L 770 166', fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1}));
  loop.appendChild(svgEl('text', {x: 715, y: 190, 'class': 'tiny', 'text-anchor': 'middle', fill: 'var(--ink)'}, 'played · 2 frames'));
  loop.appendChild(svgEl('path', {d: 'M 784 166 L 784 176 L 894 176 L 894 166', fill: 'none', stroke: 'var(--ink-faint)', 'stroke-width': 1, 'stroke-dasharray': '2 3'}));
  loop.appendChild(svgEl('text', {x: 839, y: 192, 'class': 'tiny', 'text-anchor': 'middle'}, 'replaced by the next plan'));
  loop.appendChild(svgEl('text', {x: 32, y: 192, 'class': 'lbl'}, '05 · Play loop'));
  loop.appendChild(svgEl('text', {x: 32, y: 208, 'class': 'sub'}, '2 frames are already committed when a plan is made · a new plan every 2 frames · 60 Hz'));
  loop.appendChild(svgEl('path', {d: 'M 715 200 C 715 236, 296 236, 296 216', fill: 'none', stroke: 'var(--ink-faint)', 'stroke-width': 1, 'marker-end': 'url(#arch-arrow-faint)'}));
  loop.appendChild(svgEl('text', {x: 505, y: 252, 'class': 'tiny', 'text-anchor': 'middle'}, 'the played frame becomes the next token’s action'));

  host.appendChild(svg);

  /* ---------------------------------------------------------------- cards + detail */
  var SPEC = {
    token: {
      n: '01', t: 'Observation token', r: 'one frame → one 1024-d vector',
      p: ['Four player blocks in a fixed order (ego, ego follower, opponent follower, opponent): six normalized continuous values each, with a validity mask for every float and learned embeddings for the categorical fields. Four projectile slots go through one shared encoder, are gated by presence, and are summed, so an empty slot adds exactly zero.',
          'The previous controller frame does not enter as a one-hot code. It uses the same class set the decoder predicts: a learned class embedding plus a linear map of the physical stick and trigger values. Characters, stage, the pooled projectile vector and the ego-player identity are concatenated and projected once.'],
      dl: [['d_model', '1024'], ['context', '256 frames'], ['action embed', '48'], ['character / stage', '8 / 4'], ['item slots', '4 × 32'], ['player identity', '32']],
      mech: mechToken
    },
    trunk: {
      n: '02', t: 'Context trunk', r: 'causal over 256 frames',
      p: ['A plain pre-norm Transformer with rotary positions and no bias. A query reads only earlier frames and never its own sample’s cold-start padding. Every position gets a state, so one forward pass over a training window supervises the last 128 positions at once.',
          'Only the state for the current frame drives a plan at play time. The rest exist for training.'],
      dl: [['blocks', '16'], ['d_model', '1024'], ['heads', '16 × 64'], ['attention', 'full causal'], ['supervised / window', '128 of 256']],
      mech: mechTrunk
    },
    decoder: {
      n: '03', t: 'Action decoder', r: 'writes frames in causal order',
      p: ['One small causal Transformer, not one head per frame. Its input at each step is the trunk state, an offset embedding, and the controller frame chosen at the previous step. In training every prefix of the target chunk goes through it in parallel (teacher-forced); in play it advances one frame at a time on a KV cache.',
          'It is trained on ten offsets — a dense 1 to 6 and a sparse 9, 12, 16, 20 — but only the first few ever reach the game.'],
      dl: [['blocks', '4'], ['d_model', '512'], ['heads', '8 × 64'], ['offsets', '1–6 · 9 · 12 · 16 · 20'], ['chunk sampled', '20 frames'], ['inputs', 'h_t + offset + a_prev']],
      mech: mechDecoder
    },
    frame: {
      n: '04', t: 'Controller frame', r: 'C → M → T → B',
      p: ['Inside one frame the four controller groups are chosen in order. Each pick conditions the next through FiLM: the C-stick class shifts and scales the state that chooses the main stick, and so on to triggers and buttons. Within-frame feedback was worth +0.25 stocks/min on its own in the O37 matrix.',
          'The classes are the Cartesian codec: fixed stick centres and an 8-bit button mask, so chords are one class rather than “the latest button”.'],
      dl: [['C-stick', '9 classes'], ['main stick', '65 classes'], ['triggers', '25 classes'], ['buttons', '256 (8-bit mask)'], ['conditioning', 'FiLM from chosen groups']],
      mech: mechFrame
    },
    loop: {
      n: '05', t: 'Play loop', r: 'prediction 4 · delay 2 · replan 2',
      p: ['Slippi holds every input for two frames before the game applies it, so when a plan is made the next two frames are already committed. The decoder is handed those two, writes the frames after them, and the runtime plays two of them before planning again. Inference has to finish inside those two frames.',
          'The frame the game actually played comes back as the action half of the next observation token — the loop closes through the data, not through the model.'],
      dl: [['prediction', '4 frames'], ['transport delay', '2 frames'], ['replan interval', '2 frames'], ['budget', '< 33 ms per plan'], ['rate', '60 Hz']],
      mech: mechLoop
    }
  };
  var ORDER = ['token', 'trunk', 'decoder', 'frame', 'loop'];

  function mechSvg(w, h, label) {
    return svgEl('svg', {viewBox: '0 0 ' + w + ' ' + h, role: 'img', 'aria-label': label});
  }
  function mechToken() {
    var s = mechSvg(320, 128, 'Inputs concatenated into one token');
    var rows = ['player blocks ×4', 'past action (class + value)', 'characters · stage', 'projectiles ×4 → pooled', 'ego identity'];
    rows.forEach(function (r, i) {
      var y = 12 + i * 22;
      s.appendChild(svgEl('rect', {x: 4, y: y, width: 150, height: 16, 'class': 'cell', fill: 'var(--paper)', stroke: 'var(--rule-dotted)'}));
      s.appendChild(svgEl('text', {x: 10, y: y + 11.5, 'class': 'tiny', 'font-size': '9'}, r));
      s.appendChild(svgEl('path', {d: 'M 156 ' + (y + 8) + ' C 190 ' + (y + 8) + ', 200 64, 232 64', fill: 'none', stroke: 'var(--ink-faint)', 'stroke-width': 1}));
    });
    s.appendChild(svgEl('rect', {x: 234, y: 50, width: 80, height: 28, 'class': 'plate', fill: 'var(--paper)', stroke: 'var(--ink)'}));
    s.appendChild(svgEl('text', {x: 274, y: 62, 'class': 'tiny', 'text-anchor': 'middle', 'font-size': '9', fill: 'var(--ink)'}, 'concat → linear'));
    s.appendChild(svgEl('text', {x: 274, y: 73, 'class': 'tiny', 'text-anchor': 'middle', 'font-size': '9'}, 'one 1024-d token'));
    return s;
  }
  function mechTrunk() {
    var s = mechSvg(320, 96, 'Every frame gets a state; each reads only earlier frames');
    var xs = [30, 90, 150, 210, 270];
    xs.forEach(function (x, i) {
      s.appendChild(svgEl('rect', {x: x - 18, y: 56, width: 36, height: 22, fill: 'var(--paper)', stroke: i === xs.length - 1 ? 'var(--ink)' : 'var(--ink-faint)'}));
      s.appendChild(sub({x: x, y: 71, cls: 'tiny'}, 'h', i === xs.length - 1 ? 't' : 't−' + (xs.length - 1 - i)));
      for (var j = i + 1; j < xs.length; j++) if (j - i <= 2) s.appendChild(arc(x, xs[j] - 4, 56, 12 + 10 * (j - i), 'wire time'));
    });
    s.appendChild(svgEl('text', {x: 160, y: 92, 'class': 'tiny', 'text-anchor': 'middle'}, 'dotted = may attend · the last 128 states are supervised'));
    return s;
  }
  function mechDecoder() {
    var s = mechSvg(320, 132, 'Growing action prefixes each produce the next frame');
    s.appendChild(svgEl('text', {x: 4, y: 10, 'class': 'tiny'}, 'prefix the decoder is given'));
    s.appendChild(svgEl('text', {x: 316, y: 10, 'class': 'tiny', 'text-anchor': 'end'}, 'frame it writes'));
    for (var r = 0; r < 4; r++) {
      var y = 20 + r * 27;
      var g = svgEl('g', {'class': 'prefix-row'});
      g.style.setProperty('--step', r);
      for (var k = 0; k <= r; k++) {
        g.appendChild(svgEl('rect', {x: 4 + k * 44, y: y, width: 40, height: 18, fill: 'var(--paper)', stroke: 'var(--ink-faint)'}));
        g.appendChild(sub({x: 24 + k * 44, y: y + 12.5, cls: 'tiny'}, 'a', k === 0 ? 't' : 't+' + k));
      }
      g.appendChild(line(4 + (r + 1) * 44 + 2, y + 9, 250, y + 9, 'wire', undefined, true));
      g.appendChild(svgEl('rect', {x: 256, y: y, width: 56, height: 18, fill: 'color-mix(in srgb,var(--oxblood) 8%,var(--paper))', stroke: 'var(--oxblood)'}));
      g.appendChild(sub({x: 284, y: y + 12.5, cls: 'tiny'}, 'a', 't+' + (r + 1)));
      s.appendChild(g);
    }
    s.appendChild(svgEl('text', {x: 160, y: 128, 'class': 'tiny', 'text-anchor': 'middle'}, 'all rows at once in training · one row per frame in play'));
    return s;
  }
  function mechFrame() {
    var s = mechSvg(320, 92, 'C-stick, main stick, triggers, buttons chosen in order');
    var names = [['C', 'C-stick', '9'], ['M', 'main stick', '65'], ['T', 'triggers', '25'], ['B', 'buttons', '256']];
    names.forEach(function (n, i) {
      var x = 14 + i * 78;
      var rect = svgEl('rect', {x: x, y: 22, width: 62, height: 34, 'class': 'grp-step', fill: 'var(--paper)', stroke: 'var(--ink)'});
      rect.style.setProperty('--step', i);
      s.appendChild(rect);
      s.appendChild(svgEl('text', {x: x + 31, y: 36, 'class': 'tiny', 'text-anchor': 'middle', 'font-size': '9', fill: 'var(--ink)'}, n[1]));
      s.appendChild(svgEl('text', {x: x + 31, y: 49, 'class': 'tiny', 'text-anchor': 'middle'}, n[2] + ' classes'));
      if (i < 3) {
        s.appendChild(line(x + 64, 39, x + 76, 39, 'wire chunk', undefined, true));
        s.appendChild(svgEl('text', {x: x + 70, y: 18, 'class': 'tiny', 'text-anchor': 'middle', 'font-size': '7.5'}, 'FiLM'));
      }
    });
    s.appendChild(svgEl('text', {x: 160, y: 80, 'class': 'tiny', 'text-anchor': 'middle'}, 'each pick shifts and scales the state that makes the next pick'));
    return s;
  }
  function mechLoop() {
    var s = mechSvg(320, 104, 'Timeline of one plan: two committed frames, two played, the rest replaced');
    for (var f = 0; f < 8; f++) {
      var x = 20 + f * 36;
      var kind = f < 2 ? 'committed' : f < 4 ? 'played' : 'replaced';
      s.appendChild(svgEl('rect', {x: x, y: 34, width: 30, height: 22, fill: kind === 'played' ? 'color-mix(in srgb,var(--oxblood) 12%,var(--paper))' : 'var(--paper)', stroke: kind === 'replaced' ? 'var(--ink-faint)' : 'var(--ink)', 'stroke-dasharray': kind === 'replaced' ? '2 2' : ''}));
      s.appendChild(svgEl('text', {x: x + 15, y: 48, 'class': 'tiny', 'text-anchor': 'middle'}, 't+' + (f + 1)));
    }
    s.appendChild(svgEl('text', {x: 56, y: 26, 'class': 'tiny', 'text-anchor': 'middle'}, 'already in the game'));
    s.appendChild(svgEl('text', {x: 128, y: 26, 'class': 'tiny', 'text-anchor': 'middle'}, 'played'));
    s.appendChild(svgEl('text', {x: 236, y: 26, 'class': 'tiny', 'text-anchor': 'middle'}, 'written, then replaced'));
    s.appendChild(svgEl('path', {d: 'M 20 70 L 306 70', stroke: 'var(--rule-dotted)'}));
    s.appendChild(svgEl('text', {x: 20, y: 84, 'class': 'tiny'}, 'plan made at t'));
    s.appendChild(svgEl('text', {x: 124, y: 84, 'class': 'tiny'}, 'next plan at t+2 · must be ready in 33 ms'));
    return s;
  }

  var selected = null;
  function select(key) {
    selected = key;
    if (!key) {
      host.querySelector('svg').removeAttribute('data-sel');
      Object.keys(stages).forEach(function (k) { stages[k].classList.remove('is-sel'); });
      cards.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-selected', 'false'); });
      detail.hidden = true;
      detail.replaceChildren();
      return;
    }
    host.querySelector('svg').setAttribute('data-sel', key);
    host.classList.add('has-sel');
    Object.keys(stages).forEach(function (k) { stages[k].classList.toggle('is-sel', k === key); });
    cards.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-selected', String(b.dataset.stage === key)); });
    var s = SPEC[key];
    detail.replaceChildren();
    var row = document.createElement('div');
    row.className = 'row';
    var h = document.createElement('h5');
    h.textContent = s.n + ' · ' + s.t;
    var close = document.createElement('button');
    close.type = 'button'; close.className = 'close'; close.textContent = 'close';
    close.addEventListener('click', function () { select(null); });
    row.appendChild(h); row.appendChild(close);
    detail.appendChild(row);
    var left = document.createElement('div');
    s.p.forEach(function (t) { var p = document.createElement('p'); p.textContent = t; left.appendChild(p); });
    var mech = document.createElement('div');
    mech.className = 'mech';
    mech.appendChild(s.mech());
    left.appendChild(mech);
    var right = document.createElement('div');
    var dl = document.createElement('dl');
    s.dl.forEach(function (pair) {
      var dt = document.createElement('dt'); dt.textContent = pair[0];
      var dd = document.createElement('dd'); dd.textContent = pair[1];
      dl.appendChild(dt); dl.appendChild(dd);
    });
    right.appendChild(dl);
    detail.appendChild(left);
    detail.appendChild(right);
    detail.hidden = false;
  }

  ORDER.forEach(function (key) {
    var s = SPEC[key];
    var b = document.createElement('button');
    b.type = 'button'; b.role = 'tab'; b.dataset.stage = key; b.setAttribute('aria-selected', 'false');
    b.innerHTML = '<span class="n">' + s.n + '</span><span class="t">' + s.t + '</span><span class="r">' + s.r + '</span>';
    b.addEventListener('click', function () { select(selected === key ? null : key); });
    cards.appendChild(b);
    var g = stages[key];
    g.addEventListener('click', function () { select(selected === key ? null : key); });
    g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(selected === key ? null : key); } });
  });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && selected) select(null); });
}());
