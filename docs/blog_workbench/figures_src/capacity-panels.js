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
