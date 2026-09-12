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
  D.jointToken(svg, TX, TY, TW, 34, 'sₜ', 'aₜ₋₁');
  svg.appendChild(D.el('rect', {x: 16, y: 478, width: 26, height: 14, class: 'model-token queued'}));
  D.note(svg, 50, 489, 'may be absent · absent block: values zero, mask set · empty slot: gate zero');
}());
