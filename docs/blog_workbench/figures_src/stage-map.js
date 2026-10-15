/* Fig: the five stages of the 043 policy on one time axis. Selecting a stage dims the others and
   fills the spec panel from the table below. Configuration is experiments/043_legacy_codec.py as
   retired in 5764958c; parameter counts were taken from that model by the standalone anatomy page
   (stages sum to the 15,053,039 total; 8 × 1,769,472 = 4·d² + 2·d·4d per block at d = 384). */
(function () {
  'use strict';
  var diagram = document.getElementById('stage-map-diagram');
  var spec = document.getElementById('stage-map-spec');
  if (!diagram || !spec) return;

  var TOTAL_PARAMS = 15053039;
  var STAGES = {
    obs: {
      n: '01', name: 'Observation encoder', params: 185388,
      role: 'Turns one frame of gamestate, the previous controller frame, the item slots and the match conditioning into a single token.',
      shapes: ['MDS columns', '[B, 128, 384]'],
      cfg: [['d_model', '384'], ['L_ctx', '128'], ['action_state_embed_dim', '48'], ['char_dim / stage_dim', '8 / 4'], ['observation_bundle', 'base'], ['item slots', '4, spawn-ordered']],
      parts: 'ctx_proj 135,552 · categorical tables 49,190 · character 256 · stage 128'
    },
    trunk: {
      n: '02', name: 'Transformer trunk', params: 14155776,
      role: 'Eight causal pre-norm blocks read the 128-frame window and produce one state per frame.',
      shapes: ['[B, 128, 384]', '[B, 128, 384]'],
      cfg: [['n_layers', '8'], ['n_heads', '6 × 64'], ['d_model', '384'], ['attn_window', '0 (full)'], ['mlp', '384 → 1536 → 384'], ['norm', 'RMS, no affine']],
      parts: '1,769,472 per block · no bias anywhere · rotary positions'
    },
    dec: {
      n: '03', name: 'Temporal action decoder', params: 489568,
      role: 'Expands each trunk state into ten selected future offsets. Every step reads the action the step before it chose.',
      shapes: ['[B, 128, 384]', '[B, 128, 10, 128]'],
      cfg: [['head_offsets', '1 2 3 4 5 6 9 12 16 20'], ['temporal_d_model', '128'], ['temporal_layers', '2'], ['temporal_heads', '4 × 32'], ['temporal_ff_dim', '256'], ['offset_embed_dim', '16']],
      parts: 'token projection 59,520 · 2 blocks 262,144 · group conditioning 25,344 · trunk-logit skip 136,320 · shared codec 5,904'
    },
    groups: {
      n: '04', name: 'Within-frame group chain', params: 222307,
      role: 'Picks four controller groups in a fixed order. Each pick scales and shifts the state for the next.',
      shapes: ['[B, 128, 10, 128]', '14-channel controller frame'],
      cfg: [['group_order', 'c_stick → main_stick → triggers → buttons'], ['group_head_dim', '256'], ['head', 'RMSNorm → SiLU → linear'], ['conditioning', 'state × (1 + tanh γ) + β'], ['legality', 'shoulder click masks buttons']],
      parts: 'four nonlinear heads over the retained legacy classes: 9, 37, 5 and 6'
    },
    loop: {
      n: '05', name: 'Closed-loop execution', params: 0,
      role: 'The runtime, not the model. It decides when a predicted frame reaches the emulator.',
      shapes: ['controller frame', 'Dolphin, at 60 Hz'],
      cfg: [['prediction delay d', '1'], ['plan length H', '1'], ['replan interval R', '1'], ['decode', 'ancestral, one group at a time'], ['temperature', '1.0, frozen'], ['inference', 'compiled, bucketed']],
      parts: 'a plan covers frames [t+d, t+d+H) · the newest eligible plan wins'
    }
  };
  var ORDER = ['obs', 'trunk', 'dec', 'groups', 'loop'];

  window.HALFIG_CUSTOM = window.HALFIG_CUSTOM || {data: {}};
  window.HALFIG_CUSTOM.data.stageSpecs = function () {
    return ORDER.map(function (key) {
      var s = STAGES[key];
      return {stage: s.n, name: s.name, parameters: s.params, share_pct: +(100 * s.params / TOTAL_PARAMS).toFixed(2),
        input: s.shapes[0], output: s.shapes[1], config: s.cfg.map(function (kv) { return kv[0] + ' = ' + kv[1]; }).join('; '), parts: s.parts};
    });
  };

  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function renderSpec(key) {
    var s = STAGES[key];
    var pct = s.params ? 100 * s.params / TOTAL_PARAMS : 0;
    var shapes = s.shapes.map(function (x) { return '<span class="sm-chip">' + esc(x) + '</span>'; }).join('<span class="sm-arw">→</span>');
    var cfg = s.cfg.map(function (kv) { return '<dt>' + esc(kv[0]) + '</dt><dd>' + esc(kv[1]) + '</dd>'; }).join('');
    var params = s.params ? s.params.toLocaleString('en-US') + ' · ' + pct.toFixed(2) + ' %' : 'none — this stage is runtime';
    spec.innerHTML =
      '<div class="sm-spec-grid">' +
        '<div><span class="sm-tag">Stage ' + s.n + '</span><h5>' + esc(s.name) + '</h5><p class="sm-role">' + esc(s.role) + '</p></div>' +
        '<div><dl><dt>Shapes</dt><dd><span class="sm-shapes">' + shapes + '</span></dd>' +
          '<dt>Parameters</dt><dd>' + params + '<span class="sm-parambar"><i style="width:' + Math.max(pct, 0.6) + '%"></i></span></dd></dl>' +
          '<p class="sm-parts">' + esc(s.parts) + '</p></div>' +
        '<div><dl>' + cfg + '</dl></div>' +
      '</div>';
  }

  var selected = null;
  var stages = diagram.querySelectorAll('.stage');
  function select(key) {
    selected = key;
    diagram.setAttribute('data-selected', key);
    Array.prototype.forEach.call(stages, function (g) { g.classList.toggle('is-sel', g.getAttribute('data-stage') === key); });
    renderSpec(key);
  }
  function clearSelect() {
    selected = null;
    diagram.removeAttribute('data-selected');
    Array.prototype.forEach.call(stages, function (g) { g.classList.remove('is-sel'); });
    renderSpec('trunk');
  }
  Array.prototype.forEach.call(stages, function (g) {
    var key = g.getAttribute('data-stage');
    g.addEventListener('click', function () { if (selected === key) clearSelect(); else select(key); });
    g.addEventListener('mouseenter', function () { if (!selected) renderSpec(key); });
    g.addEventListener('focus', function () { renderSpec(key); });
    g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(key); } });
  });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && selected) clearSelect(); });
  renderSpec('trunk');
}());
