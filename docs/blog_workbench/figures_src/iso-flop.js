/* O39 fit: measured data curves, iso-FLOP parabolas with their measured points and predicted minima,
   and the compute-optimal size N_opt(C) the fit implies, extrapolated to the O50 production budget. */
(function(){
 'use strict';
 var D=window.HALDIAG, data=window.HALFIG_DATA;
 var measured=document.getElementById('iso-flop-diagram'), compute=document.getElementById('iso-compute-diagram');
 var optimal=document.getElementById('iso-optimal-diagram'), table=document.getElementById('iso-optimal-table'), residual=document.getElementById('iso-residual-diagram');
 var readout=document.getElementById('iso-flop-readout');
 if(!D||!data||!measured||!compute||!optimal||!table||!residual)return;
 var I=data.iso_flop,F=I.fit,P=I.points,K=I.checkpoints,LEVELS=I.levels,FAM=I.family,U=I.u_total,O50=I.production;
 var EXP=F.dPower/(F.aPower+F.dPower);
 function loss(n,d){return F.base+F.aScale/Math.pow(n,F.aPower)+F.dScale/Math.pow(d,F.dPower);}
 function nOpt(C){return Math.pow(F.aScale*F.aPower/(F.dScale*F.dPower),1/(F.aPower+F.dPower))*Math.pow(C/6,EXP);}
 function sup(n){return String(n).replace(/\d/g,function(c){return '⁰¹²³⁴⁵⁶⁷⁸⁹'[c];}).replace('-','⁻');}
 function fmtN(n){if(n<1e9)return Math.round(n/1e6)+'M';var b=(n/1e9).toFixed(1);return (b.slice(-2)==='.0'?b.slice(0,-2):b)+'B';}
 function fmtE(v){var e=Math.floor(Math.log10(v)),m=v/Math.pow(10,e);if(Math.abs(m-10)<0.02){m=1;e+=1;}if(Math.abs(m-1)<0.02)return '10'+sup(e);var t=m.toFixed(1);return (t.slice(-2)==='.0'?t.slice(0,-2):t)+'×10'+sup(e);}
 function fmtD(d){return fmtE(d)+' ('+(d/U<0.95?(d/U).toFixed(2):(d/U).toFixed(1))+' epochs)';}
 var dMin=Math.min.apply(null,K.map(function(c){return c.positions;})),dMax=Math.max.apply(null,K.map(function(c){return c.positions;}));
 function supported(n,d){return n>=FAM.n_min&&n<=FAM.n_max&&d>=dMin&&d<=dMax;}
 function optimum(C){var n=nOpt(C),d=C/(6*n);return {c:C,n:n,d:d,nll:loss(n,d),supported:supported(n,d)};}
 function describe(o){return 'C = '+fmtE(o.c)+' FLOPs · N_opt ≈ '+fmtN(o.n)+' · D = '+fmtD(o.d)+' · predicted NLL '+o.nll.toFixed(3)+(o.supported?' · inside the observed ranges':' · extrapolated');}
 window.HALFIG_CUSTOM=window.HALFIG_CUSTOM||{data:{}};
 window.HALFIG_CUSTOM.data.isoFlop=function(){return P.map(function(p){return {run:p.run,model:p.model,layers:p.layers,d_model:p.d_model,total_parameters:p.total_parameters,compute_equivalent_parameters:p.n_eff,d_positions_exp:p.d_exp,training_flops:p.training_flops,validation_nll:p.nll,fitted_nll:p.fit_nll,residual:p.residual};});};
 window.HALFIG_CUSTOM.data.isoFlopLevels=function(){var rows=[];LEVELS.forEach(function(l){l.points.forEach(function(p){rows.push({training_flops:l.c,model:p.model,compute_equivalent_parameters:p.n_eff,validation_nll:p.nll,exact:p.exact,runs:p.runs});});});return rows;};

 function highlight(svg,series){Array.prototype.forEach.call(svg.querySelectorAll('[data-series]'),function(e){e.classList.toggle('dim',!!series&&e.getAttribute('data-series')!==series);});}
 function clearable(svg){svg.addEventListener('mouseleave',function(){highlight(svg,null);});svg.addEventListener('focusout',function(e){if(!svg.contains(e.relatedTarget))highlight(svg,null);});}
 function bind(svg,el,series,message){D.bindReadout(el,function(){readout.textContent=message;highlight(svg,series);},message);}
 function inkRamp(i,n){return 'color-mix(in srgb,var(--ink) '+Math.round(25+75*i/(n-1))+'%,var(--paper))';}
 function oxRamp(i,n){return 'color-mix(in srgb,var(--oxblood) '+Math.round(35+65*i/(n-1))+'%,var(--paper))';}
 function diamond(cx,cy,r){return 'M'+cx+' '+(cy-r)+'L'+(cx+r)+' '+cy+'L'+cx+' '+(cy+r)+'L'+(cx-r)+' '+cy+'Z';}
 function star(cx,cy,r){var d='';for(var i=0;i<10;i++){var a=-Math.PI/2+i*Math.PI/5,rr=i%2?r*0.45:r;d+=(i?'L':'M')+(cx+rr*Math.cos(a)).toFixed(1)+' '+(cy+rr*Math.sin(a)).toFixed(1);}return d+'Z';}
 // draw v(n) over the size axis as solid segments where (n, d(n)) is inside the observed ranges and dashed ones outside
 function curve(svg,x,y,fn,dOf,style,series,range){
  var segment=[],segmentSupported=null,lo=range?range[0]:x.lo,hi=range?range[1]:x.hi,last=null;
  function flush(){if(segment.length>1)svg.appendChild(D.el('polyline',{points:segment.join(' '),fill:'none',stroke:style,'stroke-width':1.6,'data-series':series,'stroke-dasharray':segmentSupported?null:'4 4'}));segment=[];}
  for(var j=0;j<200;j++){
   var n0=lo*Math.pow(hi/lo,j/200),n1=lo*Math.pow(hi/lo,(j+1)/200),v0=fn(n0),v1=fn(n1);
   if(v0<y.lo||v0>y.hi||v1<y.lo||v1>y.hi){flush();continue;}
   var sup_=supported(n0,dOf(n0))&&supported(n1,dOf(n1));
   if(segmentSupported!==sup_){flush();segmentSupported=sup_;}
   if(!segment.length)segment.push(x(n0)+','+y(v0));segment.push(x(n1)+','+y(v1));last={n:n1,v:v1};
  }
  flush();
  return last;
 }

 function render(){
  var W=window.matchMedia('(max-width:760px)').matches?340:420,H=340;
  [measured,compute,optimal,residual].forEach(function(svg){D.empty(svg);svg.setAttribute('viewBox','0 0 '+W+' '+H);});
  function sizeAxis(nLo,nHi){var x=function(n){return 50+Math.log(n/nLo)/Math.log(nHi/nLo)*(W-76);};x.lo=nLo;x.hi=nHi;return x;}
  var xm=sizeAxis(FAM.n_min/1.4,FAM.n_max*3),xc=sizeAxis(FAM.n_min/5,FAM.n_max*2.5);
  var y=D.scale([250,32],1.7,2.1);y.lo=1.7;y.hi=2.1;
  function lossAxes(svg,x){
   [1.7,1.8,1.9,2,2.1].forEach(function(v){svg.appendChild(D.el('line',{x1:50,x2:W-20,y1:y(v),y2:y(v),class:'effect-grid'}));D.text(svg,42,y(v)+4,v.toFixed(1),{class:'diagram-note','text-anchor':'end'});});
   [1e7,3e7,1e8,3e8,1e9].forEach(function(n){if(x(n)>=50&&x(n)<=W-15)D.text(svg,x(n),270,fmtN(n),{class:'diagram-note','text-anchor':'middle'});});
   D.text(svg,50,18,'validation NLL',{class:'diagram-note'});
   D.text(svg,50,292,'compute-equivalent parameters · log scale',{class:'diagram-note'});
  }

  // measured runs on the fitted data curves
  lossAxes(measured,xm);
  [26,27,28,29,30].forEach(function(e,i){
   var shade=inkRamp(i,5),d=Math.pow(2,e),series='d'+e,x=xm;
   var end=curve(measured,x,y,function(n){return loss(n,d);},function(){return d;},shade,series,[FAM.n_min,FAM.n_max]);
   D.text(measured,x(end.n)+5,y(end.v)+4,'2'+sup(e),{class:'diagram-note','data-series':series});
   P.filter(function(p){return p.d_exp===e;}).forEach(function(p){
    var dot=D.el('circle',{cx:x(p.n_eff),cy:y(p.nll),r:4,fill:shade,stroke:'var(--paper)','stroke-width':1,'data-series':series});measured.appendChild(dot);
    bind(measured,dot,series,p.model+' · N = '+fmtN(p.n_eff)+' · D = 2'+sup(e)+' · NLL '+p.nll.toFixed(3)+' (fit '+p.fit_nll.toFixed(3)+') · '+p.run);
   });
  });
  D.text(measured,50,320,'Curve labels: D = supervised positions',{class:'diagram-note'});

  // iso-FLOP parabolas: L(N, C/6N) at four budgets, the runs measured at each, and the predicted minimum
  lossAxes(compute,xc);
  var lastTopLabel=-1e9;
  LEVELS.forEach(function(level,i){
   var C=level.c,shade=oxRamp(i,LEVELS.length),series='c'+i,o=optimum(C),x=xc;
   var end=curve(compute,x,y,function(n){return loss(n,C/(6*n));},function(n){return C/(6*n);},shade,series);
   if(end.n>=x.hi/1.02)D.text(compute,x(end.n),y(end.v)-8,fmtE(C),{class:'diagram-note','data-series':series,'text-anchor':'end'});
   else{var lift=x(end.n)-lastTopLabel<44?17:5;lastTopLabel=x(end.n);D.text(compute,x(end.n)+2,y(end.v)-lift,fmtE(C),{class:'diagram-note','data-series':series,'text-anchor':'middle'});}
   level.points.forEach(function(p){
    var dot=D.el('circle',{cx:x(p.n_eff),cy:y(p.nll),r:4,fill:p.exact?shade:'var(--paper)',stroke:shade,'stroke-width':1.6,'data-series':series});compute.appendChild(dot);
    bind(compute,dot,series,p.model+' at C = '+fmtE(C)+' · N = '+fmtN(p.n_eff)+' · NLL '+p.nll.toFixed(3)+(p.exact?' · trained to this budget · '+p.runs[0]:' · interpolated between D = '+fmtE(p.between_positions[0])+' and '+fmtE(p.between_positions[1])));
   });
   var mark=D.el('path',{d:diamond(x(o.n),y(o.nll),5.5),fill:'var(--paper)',stroke:shade,'stroke-width':1.8,'data-series':series});compute.appendChild(mark);
   bind(compute,mark,series,'Predicted minimum · '+describe(o));
  });
  D.text(compute,50,320,'Curve labels: C = training FLOPs',{class:'diagram-note'});
  D.text(compute,W>400?265:50,W>400?320:334,'diamonds = predicted minima',{class:'diagram-note'});
  clearable(measured);clearable(compute);

  // compute-optimal size against budget, log–log, extrapolated to the O50 budget
  var cLo=1e16,cHi=1e20,mLo=1e7,mHi=3e9;
  var cx=function(C){return 60+Math.log(C/cLo)/Math.log(cHi/cLo)*(W-84);};
  var my=function(n){return 250-Math.log(n/mLo)/Math.log(mHi/mLo)*(250-32);};
  var band=D.el('rect',{x:60,y:my(FAM.n_max),width:W-84,height:my(FAM.n_min)-my(FAM.n_max),fill:'var(--rule)'});optimal.appendChild(band);
  [1e7,1e8,1e9].forEach(function(n){optimal.appendChild(D.el('line',{x1:60,x2:W-24,y1:my(n),y2:my(n),class:'effect-grid'}));D.text(optimal,52,my(n)+4,fmtN(n),{class:'diagram-note','text-anchor':'end'});});
  [1e16,1e17,1e18,1e19,1e20].forEach(function(C){D.text(optimal,cx(C),270,fmtE(C),{class:'diagram-note','text-anchor':'middle'});});
  D.text(optimal,60,18,'compute-optimal parameters',{class:'diagram-note'});
  D.text(optimal,60,292,'training FLOPs · log scale',{class:'diagram-note'});
  D.text(optimal,W-24,my(FAM.n_min)-5,'sizes the fit saw',{class:'diagram-note','text-anchor':'end'});
  var pts=[];for(var j=0;j<=100;j++){var C=cLo*Math.pow(cHi/cLo,j/100);pts.push(cx(C)+','+my(nOpt(C)));}
  optimal.appendChild(D.el('polyline',{points:pts.join(' '),class:'chart-line'}));
  D.text(optimal,cx(2e18)+10,my(nOpt(2e18))+16,'N_opt ∝ C'+sup(EXP.toFixed(2).replace('0.','⁰·')),{class:'diagram-note'});
  var prod=optimum(O50.flops),actualNll=loss(O50.n_eff,O50.positions);
  var cursor=D.el('g',{class:'iso-cursor',visibility:'hidden'});
  cursor.appendChild(D.el('line',{x1:0,x2:0,y1:32,y2:250}));cursor.appendChild(D.el('circle',{r:4}));
  optimal.appendChild(cursor);
  LEVELS.forEach(function(level){var o=optimum(level.c),dot=D.el('circle',{cx:cx(o.c),cy:my(o.n),r:4,class:'effect-pt'});optimal.appendChild(dot);bind(optimal,dot,null,describe(o));});
  var starMark=D.el('path',{d:star(cx(prod.c),my(prod.n),9),fill:'var(--oxblood)'});optimal.appendChild(starMark);
  bind(optimal,starMark,null,'O50 budget · '+describe(prod));
  var ring=D.el('circle',{cx:cx(O50.flops),cy:my(O50.n_eff),r:5,fill:'var(--paper)',stroke:'var(--oxblood)','stroke-width':1.8});optimal.appendChild(ring);
  bind(optimal,ring,null,'O50 as trained · C = '+fmtE(O50.flops)+' FLOPs · N = '+fmtN(O50.n_eff)+' · D = '+fmtD(O50.positions)+' · fit predicts NLL '+actualNll.toFixed(3)+' here vs '+prod.nll.toFixed(3)+' at N_opt');
  D.text(optimal,cx(prod.c)-13,my(prod.n)-9,'O50 budget',{class:'diagram-note','text-anchor':'end'});
  function sweep(e){
   if(e.target.getAttribute('role')==='button'){cursor.setAttribute('visibility','hidden');return;}
   var r=optimal.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*W;
   if(px<60||px>W-24){cursor.setAttribute('visibility','hidden');return;}
   var C=cLo*Math.pow(cHi/cLo,(px-60)/(W-84)),o=optimum(C);
   cursor.setAttribute('visibility','visible');cursor.setAttribute('transform','translate('+cx(C)+',0)');
   cursor.querySelector('circle').setAttribute('cy',my(o.n));
   readout.textContent=describe(o);
  }
  optimal.onpointermove=sweep;optimal.onpointerdown=sweep;
  optimal.onpointerleave=function(){cursor.setAttribute('visibility','hidden');};

  // the same optima as numbers, for readers without a pointer
  var body=table.querySelector('tbody');while(body.firstChild)body.removeChild(body.firstChild);
  function row(cells,cls){var tr=document.createElement('tr');if(cls)tr.className=cls;cells.forEach(function(c){var td=document.createElement('td');td.textContent=c;tr.appendChild(td);});body.appendChild(tr);}
  LEVELS.map(function(l){return optimum(l.c);}).concat([prod]).forEach(function(o,i){
   row([fmtE(o.c),fmtN(o.n),fmtE(o.d),(o.d/U).toFixed(o.d/U<0.95?2:1),o.nll.toFixed(3),i===LEVELS.length?'O50 budget':o.supported?'':'extrapolated'],i===LEVELS.length?'accent':null);
  });
  row([fmtE(O50.flops),fmtN(O50.n_eff),fmtE(O50.positions),(O50.positions/U).toFixed(1),actualNll.toFixed(3),'O50 as trained'],'accent');

  // residuals of the published fit
  var worst=Math.max.apply(null,P.map(function(p){return Math.abs(p.residual);})),rx=D.scale([50,W-24],1.76,2.06),ry=D.scale([250,36],-worst*1.3,worst*1.3);
  [-0.01,0,0.01].forEach(function(v){residual.appendChild(D.el('line',{x1:50,x2:W-24,y1:ry(v),y2:ry(v),class:v===0?'effect-zero':'effect-grid'}));D.text(residual,42,ry(v)+4,D.signed(v,2),{class:'diagram-note','text-anchor':'end'});});
  P.forEach(function(p){var dot=D.el('circle',{cx:rx(p.fit_nll),cy:ry(p.nll-p.fit_nll),r:3.5,class:'effect-pt'});residual.appendChild(dot);bind(residual,dot,null,p.model+' · D = 2'+sup(p.d_exp)+' · measured '+p.nll.toFixed(3)+' − fitted '+p.fit_nll.toFixed(3)+' = '+D.signed(p.residual,3));});
  [1.8,1.9,2].forEach(function(v){D.text(residual,rx(v),272,v.toFixed(1),{class:'diagram-note','text-anchor':'middle'});});
  D.text(residual,50,20,'measured − fitted NLL',{class:'diagram-note'});
  D.text(residual,50,294,'fitted NLL',{class:'diagram-note'});
  D.text(residual,50,322,'largest absolute residual '+worst.toFixed(3)+' · none held out',{class:'diagram-note'});
 }
 render();window.matchMedia('(max-width:760px)').addEventListener('change',render);
}());
