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
