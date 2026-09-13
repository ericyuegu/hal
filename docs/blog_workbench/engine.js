(function () {
  'use strict';

  /* ------------------------------------------------------------------ registry + shell refs */
  var REGISTRY = JSON.parse(document.getElementById('wb-registry').textContent);
  var REG = {};
  REGISTRY.forEach(function (entry) { REG[entry.key] = entry; });
  var EVIDENCE = (function () {
    var node = document.getElementById('wb-evidence');
    try { return node ? JSON.parse(node.textContent) : null; } catch (err) { return null; }
  }());
  var WB_RUN = 'https://wandb.ai/ericyuegu/hal/runs/';

  var pool = document.getElementById('wb-fig-pool');
  var docEl = document.getElementById('wb-doc');
  var parkedEl = document.getElementById('wb-parked');
  var parkedList = document.getElementById('wb-parked-list');
  var parkedCount = document.getElementById('wb-parked-count');
  var statusEl = document.getElementById('wb-status');
  var footMeta = document.getElementById('wb-foot-meta');

  var state = {
    db: null,
    online: false,
    outline: null,          // {order: [], title, updatedAt}
    blocks: {},             // id -> block data
    els: {},                // id -> rendered element
    rendered: {},           // id -> JSON of block used for the current element
    edit: false,
    showData: false,
    editing: null,          // block id currently in the editor
    dragId: null
  };

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'class') node.className = attrs[k];
      else if (k === 'text') node.textContent = attrs[k];
      else if (k === 'html') node.innerHTML = attrs[k];
      else if (k.slice(0, 2) === 'on') node.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] === false || attrs[k] === null || attrs[k] === undefined) { /* skip */ }
      else node.setAttribute(k, attrs[k] === true ? '' : attrs[k]);
    });
    (children || []).forEach(function (c) { if (c) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return node;
  }
  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function newId() { return 'b_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }
  function nowIso() { return new Date().toISOString(); }
  function hhmm() { var d = new Date(); return ('0' + d.getHours()).slice(-2) + ':' + ('0' + d.getMinutes()).slice(-2); }

  var toastEl = el('div', {class: 'wb-toast', role: 'status'});
  document.body.appendChild(toastEl);
  var toastTimer = null;
  function toast(msg) {
    toastEl.textContent = msg;
    toastEl.classList.add('on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove('on'); }, 1800);
  }
  function status(msg, warn) {
    statusEl.textContent = msg;
    statusEl.classList.toggle('warn', Boolean(warn));
  }

  /* ------------------------------------------------------------------ document model */
  function order() { return (state.outline && state.outline.order) || []; }
  function visibleIds() { return order().filter(function (id) { var b = state.blocks[id]; return b && !b.hidden; }); }
  /* Figures number in order; a figure with `sub: true` shares the previous figure's number as 2a, 2b, … */
  function figureNumbers() {
    var n = 0, map = {}, group = [], owner = {};
    function label(k) { return group.length > 1 ? n + String.fromCharCode(96 + k) : String(n); }
    function relabel() { group.forEach(function (gid, k) { map[gid] = label(k + 1); }); }
    visibleIds().forEach(function (id) {
      var b = state.blocks[id];
      if (b.type !== 'figure') return;
      if (b.sub && group.length) group.push(id); else { n += 1; group = [id]; }
      if (b.fig && !(b.fig in owner)) owner[b.fig] = id;
      relabel();
    });
    Object.keys(owner).forEach(function (f) { map[f] = map[owner[f]]; });
    return map;
  }
  function sectionNumbers() {
    var n = 0, map = {};
    visibleIds().forEach(function (id) {
      var b = state.blocks[id];
      if (b.type === 'heading' && (b.level || 2) === 2) { n += 1; map[id] = n; }
    });
    return map;
  }

  /* ------------------------------------------------------------------ persistence */
  function writeBlock(id, data) {
    state.blocks[id] = data;
    render();
    if (!state.online) { status('offline · change kept in this tab only', true); return Promise.resolve(); }
    return state.db.doc('blocks/' + id).set(data).then(function () { status('saved ' + hhmm()); }, function (err) { status('save failed: ' + err.message, true); });
  }
  function patchBlock(id, patch) {
    var next = Object.assign({}, state.blocks[id], patch, {updatedAt: nowIso()});
    return writeBlock(id, next);
  }
  function deleteBlock(id) {
    delete state.blocks[id];
    var ids = order().filter(function (x) { return x !== id; });
    var p = writeOutline(ids);
    if (state.online) state.db.doc('blocks/' + id).delete().catch(function (err) { status('delete failed: ' + err.message, true); });
    return p;
  }
  function writeOutline(ids) {
    state.outline = Object.assign({}, state.outline, {order: ids, updatedAt: nowIso()});
    render();
    if (!state.online) { status('offline · order kept in this tab only', true); return Promise.resolve(); }
    return state.db.doc('doc/outline').set(state.outline).then(function () { status('saved ' + hhmm()); }, function (err) { status('save failed: ' + err.message, true); });
  }
  function insertAt(index, block) {
    var id = newId();
    block.createdAt = block.updatedAt = nowIso();
    state.blocks[id] = block;
    var ids = order().slice();
    ids.splice(Math.max(0, Math.min(index, ids.length)), 0, id);
    if (state.online) state.db.doc('blocks/' + id).set(block).catch(function (err) { status('save failed: ' + err.message, true); });
    writeOutline(ids);
    return id;
  }
  function moveBlock(id, targetId, after) {
    var ids = order().filter(function (x) { return x !== id; });
    var at = ids.indexOf(targetId);
    if (at === -1) return;
    ids.splice(after ? at + 1 : at, 0, id);
    writeOutline(ids);
  }
  function shift(id, delta) {
    var ids = order().slice();
    var i = ids.indexOf(id);
    // skip over hidden neighbours so a move is always visible
    var j = i + delta;
    while (j >= 0 && j < ids.length && state.blocks[ids[j]] && state.blocks[ids[j]].hidden) j += delta;
    if (j < 0 || j >= ids.length) return;
    ids.splice(i, 1);
    ids.splice(j, 0, id);
    writeOutline(ids);
  }

  function useLocalSeed(reason) {
    // an offline preview (no claude host) renders the seed shipped with the preview build
    if (window.WB_SEED && !order().length) {
      state.outline = window.WB_SEED.outline;
      state.blocks = JSON.parse(JSON.stringify(window.WB_SEED.blocks));
    }
    status(reason, true);
    render();
  }
  function connect() {
    if (!window.claude || typeof window.claude.use !== 'function') {
      useLocalSeed('offline preview · storage unavailable');
      return;
    }
    window.claude.use('db').then(function (db) {
      if (!db) {
        useLocalSeed('storage unavailable in this view · read-only');
        return;
      }
      state.db = db;
      state.online = true;
      db.doc('doc/outline').onSnapshot(function (snap) {
        state.outline = snap.exists ? snap.data() : {order: []};
        render();
        if (!snap.exists) status('empty document · seed it from the session', true);
        else if (!snap.metadata.hasPendingWrites) status('synced ' + hhmm());
      }, function (err) { status('outline: ' + err.message, true); });
      db.collection('blocks').onSnapshot(function (qs) {
        qs.docChanges().forEach(function (ch) {
          if (ch.type === 'removed') delete state.blocks[ch.doc.id];
          else state.blocks[ch.doc.id] = ch.doc.data();
        });
        render();
      }, function (err) { status('blocks: ' + err.message, true); });
    }, function () {
      useLocalSeed('storage unavailable · read-only');
    });
  }

  /* ------------------------------------------------------------------ markdown */
  if (window.marked) {
    window.marked.setOptions({gfm: true, breaks: false, headerIds: false, mangle: false});
  }
  function resolveFigRefs(md, figNums) {
    return md.replace(/\{\{\s*fig:([a-z0-9-]+)\s*\}\}/gi, function (_, key) {
      var n = figNums[key];
      if (!n) return '<a class="figref missing" title="figure ' + escapeHtml(key) + ' is not placed">Fig ?</a>';
      return '<a class="figref" href="#fig-' + escapeHtml(key) + '">Fig ' + n + '</a>';
    });
  }
  /* TeX spans ($$…$$ display, $…$ inline) bypass markdown untouched; MathJax typesets them after paint. */
  function protectMath(src) {
    var spans = [];
    var out = src.replace(/\$\$([\s\S]+?)\$\$|(?<!\\)\$([^\n$]+?)\$/g, function (m, display, inline) {
      spans.push(display !== undefined ? '<div class="wb-math">$$' + display + '$$</div>' : '<span class="wb-math">\\(' + inline + '\\)</span>');
      return '\u0000MATH' + (spans.length - 1) + '\u0000';
    });
    return {src: out, spans: spans};
  }
  function md2html(md, figNums) {
    var src = resolveFigRefs(md || '', figNums || figureNumbers());
    if (!window.marked) return '<pre>' + escapeHtml(src) + '</pre>';
    var kept = protectMath(src);
    var html = window.marked.parse(kept.src).replace(/\u0000MATH(\d+)\u0000/g, function (_, i) { return kept.spans[Number(i)]; });
    // wide tables scroll on their own
    return html.replace(/<table>/g, '<div class="wb-body-tablewrap"><table>').replace(/<\/table>/g, '</table></div>');
  }
  document.addEventListener('mathjax-ready', function () { typesetMath(); });
  var typesetQueued = false;
  function typesetMath() {
    if (typesetQueued || !window.MathJax || !window.MathJax.typesetPromise) return;
    typesetQueued = true;
    setTimeout(function () {
      typesetQueued = false;
      var nodes = $$('.wb-math').filter(function (n) { return !n.querySelector('mjx-container'); });
      if (nodes.length) window.MathJax.typesetPromise(nodes).catch(function () { /* leave the TeX source visible */ });
    }, 0);
  }
  function headingHtml(b, secNums, id) {
    var level = b.level || 2;
    var text = md2html((b.md || '').replace(/^#+\s*/, ''), figureNumbers()).replace(/^<p>|<\/p>\s*$/g, '');
    var num = level === 2 && secNums[id] ? '<span class="section-number">§ ' + secNums[id] + '</span>' : '';
    return '<h' + level + ' id="' + escapeHtml(b.anchor || id) + '">' + text + num + '</h' + level + '>';
  }

  /* ------------------------------------------------------------------ figures */
  function figSource(key) { return $('.fig-src[data-fig="' + key + '"]'); }
  function claimFigure(key, host) {
    var src = figSource(key);
    if (!src) { host.appendChild(el('div', {class: 'fig-missing', text: 'figure "' + key + '" is not in this build'})); return; }
    if (src.parentNode !== pool && src.parentNode !== host) {
      host.appendChild(el('div', {class: 'fig-missing', text: 'figure "' + key + '" is already placed above; each figure can appear once'}));
      return;
    }
    host.appendChild(src);
  }
  function releaseFigure(key) {
    var src = figSource(key);
    if (src && src.parentNode !== pool) pool.appendChild(src);
  }

  function anatomyRows(name) {
    var apis = [window.HALFIG_ANATOMY, window.HALFIG_CUSTOM];
    for (var i = 0; i < apis.length; i++) {
      var api = apis[i];
      if (api && api.data && api.data[name]) { try { return api.data[name](); } catch (err) { return null; } }
    }
    return null;
  }
  function figureDatasets(entry) {
    var sets = [];
    (entry.data || []).forEach(function (name) {
      var rows = anatomyRows(name);
      if (rows !== null && rows !== undefined) sets.push({name: name, rows: rows});
    });
    if (entry.key === 'evidence-ranking' && window.HALFIG_EXPLORER_EV) sets.push({name: 'EV (24 comparisons)', rows: window.HALFIG_EXPLORER_EV});
    return sets;
  }
  function linkRuns(text) {
    return escapeHtml(text).replace(/\b([a-z0-9]{8})\b/g, function (m) {
      return /\d/.test(m) && /[a-z]/.test(m) ? '<a href="' + WB_RUN + m + '" target="_blank" rel="noopener">' + m + '</a>' : m;
    });
  }
  function cellHtml(v) {
    if (v === null || v === undefined) return '<td class="num">—</td>';
    if (typeof v === 'number') return '<td class="num">' + (Number.isInteger(v) ? v.toLocaleString('en-US') : v.toLocaleString('en-US', {maximumFractionDigits: 4})) + '</td>';
    if (typeof v === 'boolean') return '<td>' + (v ? 'yes' : 'no') + '</td>';
    if (typeof v === 'object') return '<td><pre>' + linkRuns(JSON.stringify(v, null, 0).slice(0, 600)) + '</pre></td>';
    return '<td>' + linkRuns(String(v)) + '</td>';
  }
  function tableHtml(rows) {
    if (Array.isArray(rows)) {
      if (!rows.length) return '<p class="empty">no rows</p>';
      if (typeof rows[0] !== 'object' || rows[0] === null) {
        return '<div class="scroll"><table><thead><tr><th>#</th><th>value</th></tr></thead><tbody>' +
          rows.map(function (v, i) { return '<tr><td class="num">' + i + '</td>' + cellHtml(v) + '</tr>'; }).join('') + '</tbody></table></div>';
      }
      var cols = [];
      rows.forEach(function (r) { Object.keys(r).forEach(function (k) { if (cols.indexOf(k) === -1) cols.push(k); }); });
      return '<div class="scroll"><table><thead><tr>' + cols.map(function (c) { return '<th>' + escapeHtml(c) + '</th>'; }).join('') + '</tr></thead><tbody>' +
        rows.map(function (r) { return '<tr>' + cols.map(function (c) { return cellHtml(r[c]); }).join('') + '</tr>'; }).join('') + '</tbody></table></div>';
    }
    if (rows && typeof rows === 'object') {
      return '<div class="scroll"><table><thead><tr><th>key</th><th>value</th></tr></thead><tbody>' +
        Object.keys(rows).map(function (k) { return '<tr><td>' + escapeHtml(k) + '</td>' + cellHtml(rows[k]) + '</tr>'; }).join('') + '</tbody></table></div>';
    }
    return '<p class="empty">no rows</p>';
  }
  function studyHtml(s) {
    var head = '<b>' + escapeHtml(s.id || '') + '</b>' + (s.title ? ' · ' + escapeHtml(s.title) : '') + (s.evidence ? ' · <span style="color:var(--oxblood)">' + escapeHtml(s.evidence) + '</span>' : '');
    var parts = ['<div class="wb-drawer-head" style="margin-top:12px">' + head + '</div>'];
    if (s.claim) parts.push('<p style="margin:0 0 6px;font-family:var(--font-serif);font-size:.86rem">' + escapeHtml(s.claim) + '</p>');
    if (s.setup) parts.push('<p class="empty" style="margin:0 0 6px">' + escapeHtml(s.setup) + '</p>');
    var rest = {};
    Object.keys(s).forEach(function (k) { if (['id', 'title', 'evidence', 'claim', 'setup', 'section', 'arms', 'comparison', 'comparisons', 'series', 'code'].indexOf(k) === -1) rest[k] = s[k]; });
    if (Array.isArray(s.arms)) parts.push(tableHtml(s.arms));
    if (s.comparison) parts.push(tableHtml(s.comparison));
    if (Array.isArray(s.comparisons)) parts.push(tableHtml(s.comparisons));
    if (Array.isArray(s.series)) parts.push(tableHtml(s.series));
    if (Object.keys(rest).length) parts.push(tableHtml(rest));
    if (Array.isArray(s.code) && s.code.length) parts.push('<p class="empty">code · ' + s.code.map(escapeHtml).join(' · ') + '</p>');
    return parts.join('');
  }
  function studiesHtml(entry, query) {
    if (!EVIDENCE || !Array.isArray(EVIDENCE.studies)) return '<p class="empty">evidence.json is not embedded in this build</p>';
    var q = (query || '').toLowerCase();
    var wanted = entry.studies || [];
    var list = EVIDENCE.studies.filter(function (s) {
      if (q) return JSON.stringify(s).toLowerCase().indexOf(q) !== -1;
      return wanted.indexOf(s.id) !== -1;
    });
    if (!list.length) return '<p class="empty">' + (q ? 'no study matches “' + escapeHtml(query) + '”' : 'no studies are linked to this figure yet — search all ' + EVIDENCE.studies.length + ' above') + '</p>';
    return list.map(studyHtml).join('');
  }
  function buildDrawer(entry) {
    var drawer = el('div', {class: 'wb-drawer', hidden: !state.showData});
    var sets = figureDatasets(entry);
    var tabs = el('div', {class: 'wb-drawer-tabs'});
    var body = el('div');
    var search = el('input', {type: 'search', placeholder: 'search studies, runs, numbers…', 'aria-label': 'Search the study table', id: 'wb-search-' + entry.key,
      style: 'font:.72rem/1.4 var(--font-mono);border:1px solid var(--rule-dotted);background:var(--paper);color:var(--ink);padding:3px 8px;min-width:220px'});
    var active = sets.length ? 0 : -1;
    function paint() {
      $$('button', tabs).forEach(function (b, i) { b.setAttribute('aria-pressed', String(i === active + 1 || (active === -1 && i === 0))); });
      if (active === -1) body.innerHTML = studiesHtml(entry, search.value);
      else body.innerHTML = tableHtml(sets[active].rows);
      search.hidden = active !== -1;
    }
    tabs.appendChild(el('button', {type: 'button', text: 'studies', onclick: function () { active = -1; paint(); }}));
    sets.forEach(function (s, i) {
      tabs.appendChild(el('button', {type: 'button', text: s.name, onclick: function () { active = i; paint(); }}));
    });
    search.addEventListener('input', paint);
    var head = el('div', {class: 'wb-drawer-head'}, [
      el('span', {html: 'raw data · <b>' + escapeHtml(entry.key) + '</b> · ' + escapeHtml(entry.source) + (EVIDENCE && EVIDENCE.as_of ? ' · evidence as of ' + escapeHtml(EVIDENCE.as_of) : '')}),
      search
    ]);
    drawer.appendChild(head);
    drawer.appendChild(tabs);
    drawer.appendChild(body);
    paint();
    return drawer;
  }

  /* ------------------------------------------------------------------ block rendering */
  function blockElement(id) {
    var node = state.els[id];
    if (node) return node;
    node = el('div', {class: 'wb-block', 'data-id': id});
    var handle = el('span', {class: 'wb-handle', title: 'drag to reorder', 'aria-hidden': 'true', text: '⋮⋮'});
    handle.addEventListener('mousedown', function () { node.setAttribute('draggable', 'true'); });
    handle.addEventListener('mouseup', function () { node.removeAttribute('draggable'); });
    node.appendChild(handle);
    node.appendChild(el('button', {type: 'button', class: 'wb-menu-btn', 'aria-label': 'Block actions', 'aria-haspopup': 'menu', 'aria-expanded': 'false', text: '⋯', onclick: function (e) { e.stopPropagation(); toggleMenu(id, node); }}));
    node.appendChild(el('div', {class: 'wb-body'}));
    node.appendChild(el('div', {class: 'wb-note', 'data-empty': ''}));
    node.addEventListener('dragstart', function (e) { if (!state.edit || !e.target.classList.contains('wb-block')) return; state.dragId = id; node.classList.add('dragging'); e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', id); });
    node.addEventListener('dragend', function () { node.classList.remove('dragging'); node.removeAttribute('draggable'); clearDropMarks(); state.dragId = null; });
    node.addEventListener('dragover', function (e) {
      if (!state.dragId || state.dragId === id) return;
      e.preventDefault();
      var r = node.getBoundingClientRect();
      var after = e.clientY > r.top + r.height / 2;
      clearDropMarks();
      node.classList.add(after ? 'drop-after' : 'drop-before');
    });
    node.addEventListener('drop', function (e) {
      if (!state.dragId || state.dragId === id) return;
      e.preventDefault();
      var after = node.classList.contains('drop-after');
      clearDropMarks();
      moveBlock(state.dragId, id, after);
    });
    node.addEventListener('dblclick', function (e) {
      if (!state.edit || state.editing) return;
      var b = state.blocks[id];
      if (b && b.type !== 'figure' && !e.target.closest('a')) openEditor(id);
    });
    state.els[id] = node;
    return node;
  }
  function clearDropMarks() { $$('.drop-before,.drop-after').forEach(function (n) { n.classList.remove('drop-before', 'drop-after'); }); }

  function paintBlock(id, b, figNums, secNums) {
    var node = blockElement(id);
    node.setAttribute('data-type', b.type);
    var body = $('.wb-body', node);
    var note = $('.wb-note', node);
    note.textContent = b.note || '';
    if (b.note) note.removeAttribute('data-empty'); else note.setAttribute('data-empty', '');
    if (state.editing === id) return node; // the editor owns the body while open

    var signature = JSON.stringify([b, figNums[id], secNums[id], b.type === 'figure' ? figNums : null]);
    if (state.rendered[id] === signature && body.childNodes.length) return node;
    state.rendered[id] = signature;

    if (b.type === 'figure') {
      var entry = REG[b.fig] || {key: b.fig, name: b.fig, source: 'missing', data: []};
      if (b.fig) releaseFigure(b.fig);
      body.innerHTML = '';
      node.id = 'fig-' + (b.fig || id);
      var host = el('div', {class: 'wb-figwrap'});
      claimFigure(b.fig, host);
      body.appendChild(host);
      var cap = el('div', {class: 'wb-fig-caption'});
      cap.appendChild(el('span', {class: 'fig-num', text: 'Fig ' + (figNums[id] || '?')}));
      cap.appendChild(el('span', {class: 'fig-text', html: b.caption ? md2html(b.caption, figNums) : '<p><em>no caption yet</em></p>'}));
      cap.appendChild(el('span', {class: 'fig-key', text: entry.key}));
      var drawer = buildDrawer(entry);
      var btn = el('button', {type: 'button', class: 'wb-data-btn', 'aria-expanded': String(state.showData), text: state.showData ? 'data ▴' : 'data ▾', onclick: function () {
        var open = drawer.hidden;
        drawer.hidden = !open;
        btn.setAttribute('aria-expanded', String(open));
        btn.textContent = open ? 'data ▴' : 'data ▾';
      }});
      cap.appendChild(btn);
      body.appendChild(cap);
      body.appendChild(drawer);
      return node;
    }
    if (b.type === 'heading') {
      body.innerHTML = headingHtml(b, secNums, id);
      return node;
    }
    if (b.type === 'card') {
      var lines = (b.md || '').split('\n');
      var tab = lines.shift() || '';
      body.innerHTML = '<div class="wb-card"><span class="wb-card-tab">' + escapeHtml(tab.replace(/^#+\s*/, '')) + '</span>' + md2html(lines.join('\n').trim(), figNums) + '</div>';
      return node;
    }
    body.innerHTML = md2html(b.md || '', figNums);
    typesetMath();
    return node;
  }

  function insertBar(index) {
    var bar = el('div', {class: 'wb-insert'});
    var btn = el('button', {type: 'button', 'aria-expanded': 'false', 'aria-label': 'Insert a block here', text: '+ insert'});
    bar.appendChild(btn);
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      closeMenus();
      var menu = el('div', {class: 'wb-insert-menu', role: 'menu'});
      [['heading', 'heading'], ['text', 'text'], ['callout', 'callout'], ['card', 'card'], ['figure', 'figure…']].forEach(function (pair) {
        menu.appendChild(el('button', {type: 'button', role: 'menuitem', text: pair[1], onclick: function (ev) {
          ev.stopPropagation();
          closeMenus();
          if (pair[0] === 'figure') { openPicker(index); return; }
          var block = pair[0] === 'heading' ? {type: 'heading', level: 2, md: 'New section'} : {type: pair[0], md: pair[0] === 'callout' ? '**NB.** ' : pair[0] === 'card' ? 'Card title\n\n- item' : 'New paragraph.'};
          var id = insertAt(index, block);
          setTimeout(function () { openEditor(id); }, 0);
        }}));
      });
      bar.appendChild(menu);
      btn.setAttribute('aria-expanded', 'true');
    });
    return bar;
  }

  function render() {
    var ids = order();
    var figNums = figureNumbers();
    var secNums = sectionNumbers();
    var frag = document.createDocumentFragment();
    var missing = 0;
    frag.appendChild(insertBar(0));
    ids.forEach(function (id, index) {
      var b = state.blocks[id];
      if (!b) { missing += 1; return; }
      if (b.hidden) {
        if (b.type === 'figure' && b.fig && state.els[id] && state.els[id].parentNode) releaseFigure(b.fig);
        return;
      }
      frag.appendChild(paintBlock(id, b, figNums, secNums));
      frag.appendChild(insertBar(index + 1));
    });
    // detach elements that no longer belong, returning their figures to the pool
    Object.keys(state.els).forEach(function (id) {
      var b = state.blocks[id];
      if (!b || b.hidden || ids.indexOf(id) === -1) {
        var node = state.els[id];
        if (b && b.type === 'figure' && b.fig) releaseFigure(b.fig);
        if (!b || ids.indexOf(id) === -1) { delete state.els[id]; delete state.rendered[id]; }
        if (node.parentNode) node.parentNode.removeChild(node);
      }
    });
    docEl.replaceChildren(frag);
    renderParked();
    renderToc();
    var figCount = Object.keys(figNums).filter(function (k) { return k.slice(0, 2) === 'b_' || state.blocks[k]; }).length;
    footMeta.textContent = ids.length + ' blocks · ' + visibleIds().filter(function (id) { return state.blocks[id].type === 'figure'; }).length + ' figures placed · ' + REGISTRY.length + ' in the registry' + (missing ? ' · ' + missing + ' outline ids without a block' : '');
    void figCount;
    if (window.HALFIG_ANATOMY && window.HALFIG_ANATOMY.rerender && !render.warmed) {
      render.warmed = true;
      setTimeout(function () { try { window.HALFIG_ANATOMY.rerender(); } catch (err) { /* figure code owns its errors */ } }, 0);
    }
  }

  var tocList = document.getElementById('wb-toc-list');
  var tocEl = document.getElementById('wb-toc');
  function renderToc() {
    if (!tocList) return;
    var secNums = sectionNumbers();
    tocList.replaceChildren();
    visibleIds().forEach(function (id) {
      var b = state.blocks[id];
      if (b.type !== 'heading') return;
      var level = b.level || 2;
      if (level === 1 || level > 3) return;
      var li = el('li', {'data-level': String(level)});
      var a = el('a', {href: '#' + (b.anchor || id)});
      if (level === 2 && secNums[id]) a.appendChild(el('span', {class: 'num', text: '§ ' + secNums[id]}));
      a.appendChild(document.createTextNode((b.md || '').replace(/^#+\s*/, '').replace(/[*_`]/g, '')));
      a.addEventListener('click', function (e) {
        e.preventDefault();
        var target = document.getElementById(b.anchor || id);
        if (target) { target.scrollIntoView({block: 'start'}); history.replaceState(null, '', '#' + (b.anchor || id)); }
      });
      li.appendChild(a);
      tocList.appendChild(li);
    });
    spyToc();
  }
  function spyToc() {
    if (!tocList) return;
    var links = $$('a', tocList);
    var current = null;
    links.forEach(function (a) {
      var target = document.getElementById(a.getAttribute('href').slice(1));
      if (target && target.getBoundingClientRect().top <= 120) current = a;
    });
    links.forEach(function (a) { a.setAttribute('aria-current', String(a === current)); });
  }
  window.addEventListener('scroll', function () { if (!spyToc.tick) { spyToc.tick = true; requestAnimationFrame(function () { spyToc.tick = false; spyToc(); }); } }, {passive: true});
  var tocToggle = document.getElementById('wb-toc-toggle');
  if (tocToggle) {
    tocToggle.addEventListener('click', function () {
      var open = tocEl.getAttribute('data-open') !== 'true';
      tocEl.setAttribute('data-open', String(open));
      tocToggle.setAttribute('aria-expanded', String(open));
      try { localStorage.setItem('wb-toc', open ? '1' : '0'); } catch (err) { /* per-viewer convenience only */ }
    });
    try { if (localStorage.getItem('wb-toc') === '0' || window.innerWidth < 1180) { tocEl.setAttribute('data-open', 'false'); tocToggle.setAttribute('aria-expanded', 'false'); } } catch (err) { /* ignore */ }
  }

  function renderParked() {
    var hidden = order().filter(function (id) { return state.blocks[id] && state.blocks[id].hidden; });
    parkedEl.hidden = !hidden.length || !state.edit;
    parkedCount.textContent = hidden.length ? String(hidden.length) : '';
    parkedList.replaceChildren();
    hidden.forEach(function (id) {
      var b = state.blocks[id];
      var preview = b.type === 'figure' ? (REG[b.fig] ? REG[b.fig].name : b.fig) : (b.md || '').replace(/[#*_>`]/g, '').slice(0, 140);
      parkedList.appendChild(el('div', {class: 'wb-parked-item'}, [
        el('span', {class: 't', text: b.type}),
        el('span', {class: 'p', text: preview, title: preview}),
        el('button', {type: 'button', text: 'restore', onclick: function () { patchBlock(id, {hidden: false}); }}),
        el('button', {type: 'button', class: 'danger', text: 'delete', onclick: function () { if (confirm('Delete this parked block for good?')) deleteBlock(id); }})
      ]));
    });
  }

  /* ------------------------------------------------------------------ menus, editor, picker */
  function closeMenus() {
    $$('.wb-menu, .wb-insert-menu').forEach(function (m) { m.remove(); });
    $$('[aria-expanded="true"].wb-menu-btn, .wb-insert button[aria-expanded="true"]').forEach(function (b) { b.setAttribute('aria-expanded', 'false'); });
  }
  document.addEventListener('click', function () { closeMenus(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeMenus(); });

  function toggleMenu(id, node) {
    var open = $('.wb-menu', node);
    closeMenus();
    if (open) return;
    var tpl = document.getElementById('wb-block-menu').content.cloneNode(true);
    var menu = tpl.firstElementChild;
    var b = state.blocks[id];
    if (b.type === 'figure') { $('[data-act="duplicate"]', menu).remove(); $('[data-act="edit"]', menu).textContent = 'caption'; }
    menu.addEventListener('click', function (e) {
      var act = e.target.getAttribute('data-act');
      if (!act) return;
      e.stopPropagation();
      closeMenus();
      if (act === 'edit' || act === 'note') openEditor(id, act === 'note');
      if (act === 'below-text' || act === 'below-heading' || act === 'below-callout') {
        var kind = act.slice(6);
        var nid = insertAt(order().indexOf(id) + 1, kind === 'heading' ? {type: 'heading', level: 3, md: 'New section'} : {type: kind, md: kind === 'callout' ? '**NB.** ' : 'New paragraph.'});
        setTimeout(function () { openEditor(nid); }, 0);
      }
      if (act === 'below-figure') openPicker(order().indexOf(id) + 1);
      if (act === 'up') shift(id, -1);
      if (act === 'down') shift(id, 1);
      if (act === 'park') patchBlock(id, {hidden: true});
      if (act === 'duplicate') { var copy = Object.assign({}, b); delete copy.createdAt; insertAt(order().indexOf(id) + 1, copy); }
      if (act === 'delete') { if (confirm('Delete this block? Park it instead if you might want it back.')) deleteBlock(id); }
    });
    node.appendChild(menu);
    $('.wb-menu-btn', node).setAttribute('aria-expanded', 'true');
  }

  function openEditor(id, focusNote) {
    if (state.editing && state.editing !== id) closeEditor(false);
    var b = state.blocks[id];
    if (!b) return;
    state.editing = id;
    var node = blockElement(id);
    var body = $('.wb-body', node);
    var isFigure = b.type === 'figure';
    var editor = el('div', {class: 'wb-editor'});
    var fields = {};
    if (b.type === 'heading') {
      var lvl = el('select', {id: 'wb-level-' + id});
      [2, 3, 4].forEach(function (n) { lvl.appendChild(el('option', {value: String(n), text: 'h' + n, selected: (b.level || 2) === n})); });
      editor.appendChild(el('label', {}, ['level', lvl]));
      fields.level = lvl;
    }
    if (isFigure) {
      var keyIn = el('input', {type: 'text', id: 'wb-fig-' + id, value: b.fig || '', list: 'wb-fig-keys', spellcheck: 'false'});
      editor.appendChild(el('label', {}, ['figure key', keyIn]));
      fields.fig = keyIn;
    }
    var ta = el('textarea', {id: 'wb-ta-' + id, spellcheck: 'true', placeholder: isFigure ? 'caption (markdown, one paragraph)' : 'markdown · use {{fig:key}} to reference a figure by its stable key'});
    ta.value = isFigure ? (b.caption || '') : (b.md || '');
    editor.appendChild(ta);
    var preview = el('div', {class: 'wb-editor-preview'});
    editor.appendChild(preview);
    var noteIn = el('textarea', {id: 'wb-note-' + id, placeholder: 'editorial note (never exported)', style: 'min-height:48px;border-top:1px dotted var(--rule-dotted);border-bottom:0;font-size:.76rem'});
    noteIn.value = b.note || '';
    editor.appendChild(noteIn);
    var bar = el('div', {class: 'wb-editor-bar'});
    bar.appendChild(el('span', {class: 'hint', text: '⌘/ctrl+S saves · esc cancels'}));
    bar.appendChild(el('button', {type: 'button', class: 'quiet', text: 'cancel', onclick: function () { closeEditor(false); }}));
    bar.appendChild(el('button', {type: 'button', class: 'primary', text: 'save', onclick: function () { closeEditor(true); }}));
    editor.appendChild(bar);

    function paintPreview() {
      if (isFigure) { preview.innerHTML = ta.value ? md2html(ta.value) : ''; return; }
      preview.innerHTML = b.type === 'heading' ? headingHtml({md: ta.value, level: Number(fields.level.value)}, {}, id) : md2html(ta.value);
      typesetMath();
    }
    ta.addEventListener('input', function () { autosize(ta); paintPreview(); });
    editor.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); closeEditor(true); }
      if (e.key === 'Escape') { e.preventDefault(); closeEditor(false); }
    });
    state.editorFields = {ta: ta, note: noteIn, fields: fields, isFigure: isFigure, level: b.level};
    body.replaceChildren(editor);
    paintPreview();
    autosize(ta);
    (focusNote ? noteIn : ta).focus();
  }
  function autosize(ta) { ta.style.height = 'auto'; ta.style.height = Math.min(600, Math.max(120, ta.scrollHeight + 4)) + 'px'; }
  function closeEditor(save) {
    var id = state.editing;
    if (!id) return;
    var f = state.editorFields;
    state.editing = null;
    state.editorFields = null;
    delete state.rendered[id];
    if (!save) { render(); return; }
    var patch = {note: f.note.value.trim()};
    if (f.isFigure) {
      patch.caption = f.ta.value.trim();
      var key = f.fields.fig.value.trim();
      if (key && key !== state.blocks[id].fig) { releaseFigure(state.blocks[id].fig); patch.fig = key; }
    } else {
      patch.md = f.ta.value.replace(/\s+$/, '');
      if (f.fields.level) patch.level = Number(f.fields.level.value);
    }
    patchBlock(id, patch);
  }

  function openPicker(index) {
    $$('.wb-picker').forEach(function (p) { p.remove(); });
    var placed = {};
    order().forEach(function (id) { var b = state.blocks[id]; if (b && b.type === 'figure' && b.fig) placed[b.fig] = b.hidden ? 'parked' : 'placed'; });
    var sheet = el('div', {class: 'wb-picker', role: 'dialog', 'aria-label': 'Choose a figure'});
    var inner = el('div', {class: 'wb-picker-inner'});
    inner.appendChild(el('div', {class: 'wb-picker-head'}, [
      el('span', {text: 'Insert a figure'}),
      el('button', {type: 'button', class: 'wb-tool', text: 'close', onclick: function () { sheet.remove(); }})
    ]));
    var grid = el('div', {class: 'wb-picker-grid'});
    REGISTRY.forEach(function (entry) {
      var stateLabel = placed[entry.key] || '';
      var btn = el('button', {type: 'button', disabled: stateLabel === 'placed'}, [
        el('span', {class: 'k', text: entry.key + ' · ' + entry.source}),
        el('span', {class: 'n', text: entry.name}),
        stateLabel ? el('span', {class: 's', text: stateLabel === 'parked' ? 'parked — inserting restores it here' : 'already placed'}) : null
      ]);
      btn.addEventListener('click', function () {
        sheet.remove();
        if (stateLabel === 'parked') {
          var pid = order().filter(function (id) { return state.blocks[id].fig === entry.key; })[0];
          var ids = order().filter(function (id) { return id !== pid; });
          ids.splice(index, 0, pid);
          state.blocks[pid] = Object.assign({}, state.blocks[pid], {hidden: false});
          if (state.online) state.db.doc('blocks/' + pid).set(state.blocks[pid]);
          writeOutline(ids);
          return;
        }
        insertAt(index, {type: 'figure', fig: entry.key, caption: entry.caption || ''});
      });
      grid.appendChild(btn);
    });
    inner.appendChild(grid);
    sheet.appendChild(inner);
    sheet.addEventListener('click', function (e) { e.stopPropagation(); });
    document.body.appendChild(sheet);
  }

  /* ------------------------------------------------------------------ export */
  function exportMarkdown() {
    var figNums = figureNumbers();
    var out = [];
    visibleIds().forEach(function (id) {
      var b = state.blocks[id];
      if (b.type === 'heading') { out.push('#'.repeat(b.level || 2) + ' ' + (b.md || '').replace(/^#+\s*/, '')); out.push(''); return; }
      if (b.type === 'figure') {
        var entry = REG[b.fig] || {name: b.fig};
        out.push('> **FIGURE ' + figNums[id] + ' — ' + entry.name + '** `' + b.fig + '`');
        if (b.caption) { out.push('>'); out.push('> ' + b.caption.replace(/\n/g, '\n> ')); }
        out.push('');
        return;
      }
      var md = (b.md || '').replace(/\{\{\s*fig:([a-z0-9-]+)\s*\}\}/gi, function (_, k) { return figNums[k] ? 'Fig ' + figNums[k] : 'Fig ?'; });
      if (b.type === 'callout' || b.type === 'card') md = md.split('\n').map(function (l) { return '> ' + l; }).join('\n');
      out.push(md);
      out.push('');
    });
    return out.join('\n').replace(/\n{3,}/g, '\n\n');
  }
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(text);
    var ta = el('textarea', {style: 'position:fixed;left:-9999px'});
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } finally { ta.remove(); }
    return Promise.resolve();
  }

  /* ------------------------------------------------------------------ shell controls */
  var editBtn = document.getElementById('wb-edit');
  var dataBtn = document.getElementById('wb-data');
  var notesBtn = document.getElementById('wb-notes');
  var themeBtn = document.getElementById('wb-theme');
  function setEdit(on) {
    state.edit = on;
    document.body.classList.toggle('wb-edit', on);
    editBtn.setAttribute('aria-pressed', String(on));
    if (!on && state.editing) closeEditor(true);
    try { localStorage.setItem('wb-edit', on ? '1' : '0'); } catch (err) { /* per-viewer convenience only */ }
    render();
  }
  function setData(on) {
    state.showData = on;
    document.body.classList.toggle('wb-showdata', on);
    dataBtn.setAttribute('aria-pressed', String(on));
    $$('.wb-drawer').forEach(function (d) { d.hidden = !on; });
    $$('.wb-data-btn').forEach(function (b) { b.setAttribute('aria-expanded', String(on)); b.textContent = on ? 'data ▴' : 'data ▾'; });
    try { localStorage.setItem('wb-data', on ? '1' : '0'); } catch (err) { /* per-viewer convenience only */ }
  }
  function setNotes(on) {
    state.showNotes = on;
    document.body.classList.toggle('wb-notes', on);
    notesBtn.setAttribute('aria-pressed', String(on));
    try { localStorage.setItem('wb-notes', on ? '1' : '0'); } catch (err) { /* per-viewer convenience only */ }
  }
  notesBtn.addEventListener('click', function () { setNotes(!state.showNotes); });
  editBtn.addEventListener('click', function () { setEdit(!state.edit); });
  dataBtn.addEventListener('click', function () { setData(!state.showData); });
  document.getElementById('wb-export').addEventListener('click', function () {
    copyText(exportMarkdown()).then(function () { toast('markdown copied · ' + visibleIds().length + ' blocks'); }, function () { toast('copy failed — select the text manually'); });
  });

  function effectiveTheme() {
    var stamped = document.documentElement.getAttribute('data-theme');
    if (stamped === 'dark' || stamped === 'ink') return 'dark';
    if (stamped === 'light') return 'light';
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function paintTheme() { themeBtn.textContent = effectiveTheme() === 'dark' ? 'paper' : 'ink'; themeBtn.setAttribute('aria-label', 'Switch to ' + themeBtn.textContent + ' mode'); }
  themeBtn.addEventListener('click', function () {
    var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('wb-theme', next); } catch (err) { /* per-viewer convenience only */ }
    paintTheme();
    document.dispatchEvent(new CustomEvent('hal:theme', {detail: next}));
    if (window.HALFIG_ANATOMY && window.HALFIG_ANATOMY.rerender) setTimeout(window.HALFIG_ANATOMY.rerender, 0);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', paintTheme);

  // figure-key datalist for the editor
  var dl = el('datalist', {id: 'wb-fig-keys'});
  REGISTRY.forEach(function (r) { dl.appendChild(el('option', {value: r.key, text: r.name})); });
  document.body.appendChild(dl);

  /* ------------------------------------------------------------------ boot */
  try {
    var savedTheme = localStorage.getItem('wb-theme');
    if (savedTheme === 'dark' || savedTheme === 'light') document.documentElement.setAttribute('data-theme', savedTheme);
    if (localStorage.getItem('wb-edit') === '1') setEdit(true);
    if (localStorage.getItem('wb-data') === '1') setData(true);
    if (localStorage.getItem('wb-notes') === '1') setNotes(true);
  } catch (err) { /* private windows still render */ }
  paintTheme();
  render();
  connect();
}());
