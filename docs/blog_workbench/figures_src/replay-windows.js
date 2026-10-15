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
