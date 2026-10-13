'use strict';
/* rtMRI Inspector front end. One timeline (seconds) drives the video frame, the audio views and the area-function views. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, d = 2) => (v == null || !isFinite(v)) ? '–' : (+v).toFixed(d);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).error || m; } catch (e) {} throw new Error(m); }
  return r;
}
const getJSON = async p => (await api(p)).json();
const postJSON = async (p, b) => (await api(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b || {}) })).json();
function debounce(fn, ms) { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; }
function toast(msg, ms = 6000) { const b = $('#banner'); b.textContent = msg; b.classList.remove('hidden'); clearTimeout(toast.h); if (ms) toast.h = setTimeout(() => b.classList.add('hidden'), ms); }

/* ------------------------------------------------------------------ state */
const S = {
  meta: null, ft: null, t: 0, frame: 0, win: [0, 1], sel: null, playing: false, gen: 0,
  chunks: new Map(), loading: new Map(), C: 1, gain: null, vmax: 255,
  over: null, wave: null, spec: null, fmt: null, tractRes: null, batch: null, hoverGl: -1,
  lmMode: null, lm: {}, seq: { wave: 0, spec: 0, fmt: 0, tract: 0 }, hoverT: null,
  img: { bri: 0, con: 1, gam: 1, smooth: true, flat: false },
};
const LM_ORDER = ['glot', 'mpal', 'dent', 'mlab'];
const LM_HELP = {
  glot: 'Click the glottis (bottom of the airway, at the level of the vocal folds).',
  mpal: 'Click the highest point of the hard palate roof (dark airway just below it).',
  dent: 'Click the alveolar ridge, just behind the upper front teeth.',
  mlab: 'Click the mid-lip position: the front limit of the vocal tract.',
};
const ML = 46, MR = 10; // plot margins shared by the audio canvases

/* ------------------------------------------------------------------ colour maps */
function lutFromStops(stops) {
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255 * (stops.length - 1), k = Math.min(Math.floor(x), stops.length - 2), f = x - k;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = stops[k][c] * (1 - f) + stops[k + 1][c] * f;
  }
  return out;
}
const CMAPS = {
  ink: lutFromStops([[255, 255, 255], [225, 225, 225], [120, 120, 120], [20, 20, 20], [0, 0, 0]]),
  gray: lutFromStops([[0, 0, 0], [255, 255, 255]]),
  magma: lutFromStops([[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]]),
  viridis: lutFromStops([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]),
};
const theme = () => getComputedStyle(document.documentElement);
function setupCanvas(c) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); return { g, w, h };
}

/* ------------------------------------------------------------------ time helpers */
function frameAt(t) {
  const a = S.ft; let lo = 0, hi = a.length - 1;
  if (t <= a[0]) return 0; if (t >= a[hi]) return hi;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] <= t) lo = m; else hi = m; }
  return (t - a[lo] <= a[hi] - t) ? lo : hi;
}
const tEnd = () => S.meta.t_end;
const x2t = (x, w) => S.win[0] + (x - ML) / (w - ML - MR) * (S.win[1] - S.win[0]);
const t2x = (t, w) => ML + (t - S.win[0]) / (S.win[1] - S.win[0]) * (w - ML - MR);
const inVideo = t => t >= S.ft[0] - 0.5 / S.meta.fps && t <= S.ft[S.ft.length - 1] + 0.5 / S.meta.fps;

/* ------------------------------------------------------------------ drawing scheduler */
const dirty = new Set(); let raf = 0;
function draw(...what) { what.forEach(w => dirty.add(w)); if (!raf) raf = requestAnimationFrame(flush); }
function flush() {
  raf = 0; const d = [...dirty]; dirty.clear();
  if (!S.meta) return;
  if (d.includes('video')) drawVideo();
  if (d.includes('over')) drawOver();
  if (d.includes('wave')) drawWave();
  if (d.includes('spec')) drawSpec();
  if (d.includes('af')) drawAFplot();
}
const drawAudioAll = () => draw('over', 'wave', 'spec');

/* ------------------------------------------------------------------ files */
async function loadFiles() {
  const f = await getJSON('/api/files');
  const sel = $('#fileSel'); sel.innerHTML = '';
  if (!f.files.length) { sel.innerHTML = '<option value="">(no videos found)</option>'; }
  f.files.forEach((e, i) => { const o = document.createElement('option'); o.value = i; o.textContent = e.rel + (e.wav ? '' : '  (no audio)'); o.dataset.v = e.video; o.dataset.w = e.wav || ''; sel.appendChild(o); });
  const cur = f.files.findIndex(e => e.token === f.open);
  if (cur >= 0) sel.value = cur; else sel.value = '';
  if (cur < 0 && f.files.length) { sel.insertBefore(Object.assign(document.createElement('option'), { value: '', textContent: 'Choose a recording…' }), sel.firstChild); sel.value = ''; }
  $('#folderPath').textContent = f.root;
}
async function openSelected() {
  const o = $('#fileSel').selectedOptions[0]; if (!o || !o.dataset.v) return;
  stop(); toast('Loading ' + o.textContent + ' …', 0);
  try { const m = await postJSON('/api/open', { video: o.dataset.v, wav: o.dataset.w || null }); init(m); $('#banner').classList.add('hidden'); }
  catch (e) { toast('Could not open: ' + e.message, 12000); }
}

/* ------------------------------------------------------------------ init per recording */
function init(m) {
  S.gen++; S.meta = m; S.ft = Float64Array.from(m.frame_times); S.chunks.clear(); S.loading.clear(); S.C = m.chunk;
  S.over = S.wave = S.spec = S.fmt = S.tractRes = S.batch = null; S.sel = null; S.gain = null; S.lmMode = null;
  $('#prog').classList.add('hidden');
  const v = m.view || {};
  S.t = v.t != null ? v.t : (m.t_end > 2 ? 2 : 0); S.t = clamp(S.t, 0, m.t_end);
  const wl = v.winLen || Math.min(1, m.t_end); S.win = [S.t - wl * 0.3, S.t + wl * 0.7]; fixWindow();
  S.frame = frameAt(S.t);
  $('#slider').max = m.n - 1; $('#frameNo').max = m.n;
  $('#noAudio').classList.toggle('hidden', m.has_audio);
  ['cOver', 'cWave', 'cSpec'].forEach(id => $('#' + id).classList.toggle('hidden', !m.has_audio));
  $('#showFmt').disabled = !m.has_audio;
  const aud = $('#aud'); if (m.has_audio) { aud.src = '/api/audio.wav?' + Date.now(); aud.load(); } else aud.removeAttribute('src');
  applySettingsToUI(m);
  const banners = (m.warnings || []);
  $('#warnList').innerHTML = banners.map(w => `<div>⚠ ${w}</div>`).join('');
  if (banners.length) toast(banners[0] + (banners.length > 1 ? `  (+${banners.length - 1} more in "Sync & data")` : ''), 14000);
  applyTractMeta(m.tract);
  renderLabels();
  prefetchAll(S.gen); loadGain();
  updateTimeUI(); refreshOverview(); refreshWindowData(); requestTract();
  draw('video', 'af');
}
function applySettingsToUI(m) {
  const s = m.settings;
  $('#avOff').value = s.av_offset; $('#fov').value = s.fov_mm; $('#flipH').checked = !!s.flip_h; $('#iFlat').checked = !!s.flatten; S.img.flat = !!s.flatten;
  $('#burstS').value = s.burst_s; $('#burstA').value = s.burst_atten; $('#pxOverride').value = s.pixel_mm || '';
  $('#pxInfo').textContent = `${fmt(m.pixel_mm, 3)} mm/px  (${m.w}×${m.h} px)`;
  const v = m.view || {};
  for (const [id, key] of [['sgWin', 'sgWin'], ['sgStep', 'sgStep'], ['sgFmax', 'sgFmax'], ['sgDr', 'sgDr'], ['sgTop', 'sgTop'], ['sgMap', 'sgMap'], ['fN', 'fN'], ['fMax', 'fMax'], ['fWin', 'fWin'], ['fStep', 'fStep'], ['fOrd', 'fOrd'], ['fBw', 'fBw'], ['fLvl', 'fLvl'], ['fScale', 'fScale']])
    if (v[key] != null) $('#' + id).value = v[key];
  if (v.sgPre != null) $('#sgPre').checked = v.sgPre;
  if (v.showFmt != null) $('#showFmt').checked = v.showFmt;
}

/* ------------------------------------------------------------------ video frames */
async function ensureChunk(ci, gen = S.gen) {
  if (S.chunks.has(ci) || ci < 0 || ci * S.C >= S.meta.n) return;
  if (S.loading.has(ci)) return S.loading.get(ci);
  const p = (async () => {
    try {
      const r = await api(`/api/frames?start=${ci * S.C}&count=${S.C}`); const buf = new Uint8Array(await r.arrayBuffer());
      if (gen !== S.gen) return;
      evictIfNeeded(ci); S.chunks.set(ci, buf);
      if (Math.floor(S.frame / S.C) === ci) draw('video');
    } finally { S.loading.delete(ci); }
  })();
  S.loading.set(ci, p); return p;
}
function evictIfNeeded(keep) {
  const cap = 600e6; let total = 0; S.chunks.forEach(b => total += b.length);
  while (total > cap && S.chunks.size) {
    let far = null, fd = -1; const here = Math.floor(S.frame / S.C);
    S.chunks.forEach((b, k) => { const d = Math.abs(k - here); if (k !== keep && d > fd) { fd = d; far = k; } });
    if (far == null) break; total -= S.chunks.get(far).length; S.chunks.delete(far);
  }
}
async function prefetchAll(gen) {
  const nC = Math.ceil(S.meta.n / S.C);
  const order = [...Array(nC).keys()].sort((a, b) => Math.abs(a - Math.floor(S.frame / S.C)) - Math.abs(b - Math.floor(S.frame / S.C)));
  let i = 0;
  const worker = async () => { while (i < order.length && gen === S.gen) { const ci = order[i++]; try { await ensureChunk(ci, gen); } catch (e) { return; } } };
  await Promise.all([worker(), worker()]);
  if (gen === S.gen) { const c0 = S.chunks.get(0); if (c0) S.vmax = Math.max(60, percentile(c0, 99.7, 7)); draw('video'); }
}
function percentile(arr, p, stride = 1) {
  const h = new Uint32Array(256); for (let i = 0; i < arr.length; i += stride) h[arr[i]]++;
  let tot = 0; h.forEach(v => tot += v); let acc = 0; for (let v = 0; v < 256; v++) { acc += h[v]; if (acc >= tot * p / 100) return v; } return 255;
}
async function loadGain() { try { const r = await api('/api/gain'); S.gain = new Float32Array(await r.arrayBuffer()); draw('video'); } catch (e) {} }
function getFrame(i) {
  const ci = Math.floor(i / S.C), ch = S.chunks.get(ci); if (!ch) { ensureChunk(ci); return null; }
  const sz = S.meta.w * S.meta.h, off = (i - ci * S.C) * sz; return ch.subarray(off, off + sz);
}
const offscreen = document.createElement('canvas');
function drawVideo() {
  const c = $('#video'); const { g, w, h } = setupCanvas(c); const m = S.meta;
  g.fillStyle = '#000'; g.fillRect(0, 0, w, h);
  const data = getFrame(S.frame);
  const sc = Math.min(w / m.w, h / m.h), ox = (w - m.w * sc) / 2, oy = (h - m.h * sc) / 2;
  if (data) {
    offscreen.width = m.w; offscreen.height = m.h; const og = offscreen.getContext('2d'); const id = og.createImageData(m.w, m.h); const px = id.data;
    const { bri, con, gam, flat } = S.img; const lut = new Uint8ClampedArray(256);
    for (let v = 0; v < 256; v++) { let x = v / S.vmax; x = ((x - 0.5) * con + 0.5) + bri / 255; lut[v] = 255 * Math.pow(clamp(x, 0, 1), 1 / gam); }
    const gain = flat && S.gain && S.gain.length === data.length ? S.gain : null;
    for (let i = 0, j = 0; i < data.length; i++, j += 4) { let v = data[i]; if (gain) v = Math.min(255, v / gain[i]); const o = lut[v | 0]; px[j] = px[j + 1] = px[j + 2] = o; px[j + 3] = 255; }
    og.putImageData(id, 0, 0);
    g.imageSmoothingEnabled = S.img.smooth; g.imageSmoothingQuality = 'high';
    g.drawImage(offscreen, ox, oy, m.w * sc, m.h * sc);
  } else { g.fillStyle = '#888'; g.font = '12px system-ui'; g.fillText('loading frames…', 12, 20); }
  const P = (x, y) => [ox + (x + 0.5) * sc, oy + (y + 0.5) * sc];
  drawTractOverlay(g, P, sc);
  $('#vmsg').classList.toggle('hidden', inVideo(S.t)); $('#vmsg').textContent = 'no video at this time';
}
function drawTractOverlay(g, P, sc) {
  const tr = S.meta.tract; if (!tr) return;
  const grid = tr.grid, res = S.tractRes;
  if (grid && $('#showGrid').checked) {
    const sh = res && res.shift ? res.shift : [0, 0];
    g.lineWidth = 1; g.strokeStyle = 'rgba(255,110,110,.55)';
    grid.ends.forEach((e, k) => { g.strokeStyle = (res && k >= res.ngl) ? 'rgba(120,120,120,.35)' : 'rgba(255,110,110,.55)'; g.beginPath(); g.moveTo(...P(e[0][0] + sh[0], e[0][1] + sh[1])); g.lineTo(...P(e[1][0] + sh[0], e[1][1] + sh[1])); g.stroke(); });
  }
  if (res && $('#showBnd').checked) {
    const line = (pts, col) => { g.strokeStyle = col; g.lineWidth = 1.2; g.beginPath(); pts.forEach((p, i) => { const q = P(p[0], p[1]); i ? g.lineTo(...q) : g.moveTo(...q); }); g.stroke(); };
    line(res.lf, 'rgba(60,255,120,.9)'); line(res.rt, 'rgba(70,200,255,.9)'); if ($('#showCen').checked) line(res.cen, 'rgba(255,230,70,.8)');
    const dot = (p, col, r) => { const q = P(p[0], p[1]); g.fillStyle = col; g.beginPath(); g.arc(q[0], q[1], r, 0, 7); g.fill(); };
    res.lf.forEach(p => dot(p, '#3dff78', 2.2)); res.rt.forEach(p => dot(p, '#46c8ff', 2.2));
    (res.edited || []).forEach(k => { dot(res.lf[k], '#fff', 3); dot(res.rt[k], '#fff', 3); });
    if (S.hoverGl >= 0 && S.hoverGl < res.ngl) { const e = tr.grid.ends[S.hoverGl], sh = res.shift || [0, 0]; g.strokeStyle = '#ff3b30'; g.lineWidth = 2.2; g.beginPath(); g.moveTo(...P(e[0][0] + sh[0], e[0][1] + sh[1])); g.lineTo(...P(e[1][0] + sh[0], e[1][1] + sh[1])); g.stroke(); }
  }
  g.font = '11px system-ui'; LM_ORDER.forEach((k, i) => {
    const p = S.lm[k] || (tr.landmarks && tr.landmarks[k]); if (!p) return; const q = P(p[0], p[1]);
    g.strokeStyle = '#ff4df0'; g.lineWidth = 1.6; g.beginPath(); g.moveTo(q[0] - 6, q[1]); g.lineTo(q[0] + 6, q[1]); g.moveTo(q[0], q[1] - 6); g.lineTo(q[0], q[1] + 6); g.stroke();
    g.fillStyle = '#ff9af6'; g.fillText(String(i + 1), q[0] + 7, q[1] - 5);
  });
}

/* ------------------------------------------------------------------ time / transport */
function fixWindow() {
  const T = tEnd(); let [a, b] = S.win; let len = clamp(b - a, 0.02, Math.max(T, 0.02));
  a = clamp(a, 0, Math.max(0, T - len)); S.win = [a, a + len];
}
function setTime(t, opt = {}) {
  S.t = clamp(t, 0, tEnd()); const f = frameAt(S.t); const fc = f !== S.frame; S.frame = f;
  if (!opt.keepWindow && (S.t < S.win[0] || S.t > S.win[1])) { const L = S.win[1] - S.win[0]; S.win = [S.t - L * 0.15, S.t + L * 0.85]; fixWindow(); windowChanged(); }
  updateTimeUI(); draw('video', 'over', 'wave', 'spec');
  if (fc) { requestTractDebounced(); }
  scheduleViewSave();
}
function updateTimeUI() {
  const m = S.meta; $('#slider').value = S.frame; if (document.activeElement !== $('#frameNo')) $('#frameNo').value = S.frame + 1;
  if (document.activeElement !== $('#timeNo')) $('#timeNo').value = S.t.toFixed(3);
  $('#frameInfo').textContent = `${S.frame + 1} / ${m.n}  ·  ${m.fps.toFixed(2)} fps`;
  $('#readout').textContent = `t = ${S.t.toFixed(3)} s   frame ${S.frame + 1}`;
  $('#winLen').value = (S.win[1] - S.win[0]).toFixed(2);
  $('#selInfo').textContent = S.sel ? `selection ${S.sel[0].toFixed(3)}–${S.sel[1].toFixed(3)} s (${((S.sel[1] - S.sel[0]) * 1000).toFixed(0)} ms)` : '';
}
/* playback: audio element is the clock while it is sounding, wall-clock otherwise */
let pb = null;
function play(t0, t1) {
  if (S.playing) stop();
  const rate = +$('#rate').value; const aud = $('#aud'), m = S.meta, off = m.settings.av_offset;
  if (t1 == null) { t1 = (S.sel && S.t >= S.sel[0] && S.t < S.sel[1]) ? S.sel[1] : tEnd(); }
  if (t0 >= t1 - 0.01) t0 = (S.sel && t1 === S.sel[1]) ? S.sel[0] : 0;
  pb = { t0, t1, rate, wall: performance.now(), t: t0, audioOn: false, off };
  S.playing = true; $('#bPlay').textContent = '⏸'; aud.playbackRate = rate; setTime(t0, { keepWindow: false }); tick();
}
function stop() { S.playing = false; pb = null; $('#aud').pause(); $('#bPlay').textContent = '▶'; }
function tick() {
  if (!S.playing || !pb) return;
  const aud = $('#aud'), m = S.meta, now = performance.now();
  let t = pb.t + (now - pb.wall) / 1000 * pb.rate; pb.wall = now;
  if (m.has_audio) {
    const ta = t - pb.off, inA = ta >= 0 && ta < m.audio_dur;
    if (inA && !pb.audioOn) { aud.currentTime = ta; aud.play().catch(() => {}); pb.audioOn = true; }
    else if (inA && pb.audioOn && !aud.paused) { const tc = aud.currentTime + pb.off; if (Math.abs(tc - t) < 0.5) t = tc; }
    else if (!inA && pb.audioOn) { aud.pause(); pb.audioOn = false; }
  }
  pb.t = t;
  if (t >= pb.t1) { if ($('#loop').checked) { pb.t = pb.t0; pb.wall = now; pb.audioOn = false; aud.pause(); } else { setTime(pb.t1); stop(); return; } }
  setTime(pb.t); requestAnimationFrame(tick);
}
const winRange = () => S.win.slice();
const selOrWin = () => S.sel ? S.sel.slice() : S.win.slice();
function rangeFor(kind) { return kind === 'all' ? [0, tEnd()] : kind === 'win' ? winRange() : (S.sel ? S.sel.slice() : winRange()); }

/* ------------------------------------------------------------------ audio views */
const refreshOverview = debounce(async () => {
  if (!S.meta || !S.meta.has_audio) return; const c = $('#cOver'); const n = Math.max(200, Math.round(c.clientWidth - ML - MR)); const seq = ++S.seq.wave + 'o';
  try { const d = await getJSON(`/api/waveform?t0=0&t1=${tEnd()}&n=${n}`); S.over = d; draw('over'); } catch (e) {}
}, 60);
let waveSeq = 0, specSeq = 0, fmtSeq = 0;
const refreshWindowData = debounce(async () => {
  if (!S.meta) return; if (S.meta.has_audio) { fetchWave(); fetchSpec(); if ($('#showFmt').checked) fetchFmt(); }
}, 80);
function windowChanged() { drawAudioAll(); refreshWindowData(); updateTimeUI(); scheduleViewSave(); }
async function fetchWave() {
  const c = $('#cWave'); const n = Math.max(100, Math.round(c.clientWidth - ML - MR)); const [a, b] = S.win; const seq = ++waveSeq;
  try { const d = await getJSON(`/api/waveform?t0=${a}&t1=${b}&n=${n}`); if (seq === waveSeq) { S.wave = d; draw('wave'); } } catch (e) {}
}
function specQuery(a, b, stepMs) {
  return `t0=${a}&t1=${b}&win=${$('#sgWin').value}&step=${stepMs}&fmax=${$('#sgFmax').value}&preemph=${$('#sgPre').checked ? 1 : 0}&dr=${$('#sgDr').value}&top=${$('#sgTop').value}`;
}
async function fetchSpec() {
  const c = $('#cSpec'); const pw = Math.max(200, c.clientWidth - ML - MR); const [a, b] = S.win; const seq = ++specSeq;
  const step = Math.max(+$('#sgStep').value, (b - a) * 1000 / (pw * 1.5));
  try {
    const r = await api('/api/spectrogram?' + specQuery(a, b, step)); const meta = JSON.parse(r.headers.get('X-Meta')); const bmp = await createImageBitmap(await r.blob());
    if (seq !== specSeq) return;
    const t = document.createElement('canvas'); t.width = bmp.width; t.height = bmp.height; const tg = t.getContext('2d'); tg.drawImage(bmp, 0, 0);
    const px = tg.getImageData(0, 0, t.width, t.height).data; const gray = new Uint8Array(t.width * t.height); for (let i = 0; i < gray.length; i++) gray[i] = px[i * 4];
    S.spec = { ...meta, gray, win: [a, b] }; colorizeSpec(); draw('spec');
  } catch (e) { if (seq === specSeq) toast('Spectrogram: ' + e.message); }
}
function colorizeSpec() {
  const s = S.spec; if (!s) return; const lut = CMAPS[$('#sgMap').value]; const c = document.createElement('canvas'); c.width = s.cols; c.height = s.rows;
  const g = c.getContext('2d'), id = g.createImageData(s.cols, s.rows);
  for (let i = 0; i < s.gray.length; i++) { const v = s.gray[i] * 3; id.data[i * 4] = lut[v]; id.data[i * 4 + 1] = lut[v + 1]; id.data[i * 4 + 2] = lut[v + 2]; id.data[i * 4 + 3] = 255; }
  g.putImageData(id, 0, 0); s.img = c;
}
function fmtQuery(a, b) {
  return `t0=${a}&t1=${b}&nf=${$('#fN').value}&fmaxf=${$('#fMax').value}&fwin=${$('#fWin').value}&fstep=${$('#fStep').value}&order=${$('#fOrd').value}&maxbw=${$('#fBw').value}&minlvl=${$('#fLvl').value}&scale=${$('#fScale').value}`;
}
async function fetchFmt() {
  const [a, b] = S.win; const L = b - a; const seq = ++fmtSeq;
  try { const d = await getJSON('/api/formants?' + fmtQuery(Math.max(0, a - L * .05), b + L * .05)); if (seq === fmtSeq) { S.fmt = d; draw('spec'); updateFmtReadout(); } } catch (e) { if (seq === fmtSeq) toast('Formants: ' + e.message); }
}
function axisTicks(lo, hi, target) {
  const span = hi - lo, raw = span / target, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p; const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10)); return out;
}
function drawTimeGrid(g, w, h, labels) {
  const th = theme(); g.strokeStyle = 'rgba(128,140,155,.25)'; g.fillStyle = th.getPropertyValue('--dim'); g.font = '11px system-ui'; g.lineWidth = 1;
  axisTicks(S.win[0], S.win[1], Math.max(3, (w - ML - MR) / 90)).forEach(t => { const x = Math.round(t2x(t, w)) + .5; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h - (labels ? 14 : 0)); g.stroke(); if (labels) { g.textAlign = 'center'; g.fillText(t.toFixed(Math.max(0, 2 - Math.floor(Math.log10(S.win[1] - S.win[0])))) , x, h - 3); } });
  g.textAlign = 'left';
}
function drawCursorAndSel(g, w, h) {
  if (S.sel) { const x0 = t2x(S.sel[0], w), x1 = t2x(S.sel[1], w); g.fillStyle = 'rgba(224,178,79,.20)'; g.fillRect(x0, 0, x1 - x0, h); g.strokeStyle = 'rgba(224,178,79,.9)'; g.beginPath(); g.moveTo(x0 + .5, 0); g.lineTo(x0 + .5, h); g.moveTo(x1 + .5, 0); g.lineTo(x1 + .5, h); g.stroke(); }
  const x = t2x(S.t, w); if (x >= ML && x <= w - MR) { g.strokeStyle = '#ff4d4d'; g.setLineDash([5, 3]); g.lineWidth = 1.3; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); g.setLineDash([]); }
  if (S.hoverT != null) { const xh = t2x(S.hoverT, w); if (xh >= ML && xh <= w - MR) { g.strokeStyle = 'rgba(255,255,255,.35)'; g.lineWidth = 1; g.beginPath(); g.moveTo(xh + .5, 0); g.lineTo(xh + .5, h); g.stroke(); } }
}
function drawWaveArr(g, d, w, h, t0, t1, col) {
  if (!d) return; const n = d.min.length, pw = w - ML - MR; g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
  for (let i = 0; i < n; i++) { if (d.min[i] == null) continue; const x = ML + (i + .5) / n * pw; g.moveTo(x, h / 2 - d.max[i] * h * .46); g.lineTo(x, h / 2 - d.min[i] * h * .46 + .5); } g.stroke();
}
function drawOver() {
  const c = $('#cOver'); if (c.classList.contains('hidden')) return; const { g, w, h } = setupCanvas(c); const th = theme(); g.clearRect(0, 0, w, h);
  g.fillStyle = th.getPropertyValue('--panel2'); g.fillRect(ML, 0, w - ML - MR, h);
  if (S.over) drawWaveArr(g, S.over, w, h, 0, tEnd(), '#7d8aa0');
  const T = tEnd(), X = t => ML + t / T * (w - ML - MR);
  g.fillStyle = 'rgba(76,194,176,.18)'; g.strokeStyle = '#4cc2b0'; g.lineWidth = 1.5; const x0 = X(S.win[0]), x1 = X(S.win[1]); g.fillRect(x0, 0, x1 - x0, h); g.strokeRect(x0 + .5, .5, Math.max(2, x1 - x0), h - 1);
  if (S.sel) { g.fillStyle = 'rgba(224,178,79,.35)'; g.fillRect(X(S.sel[0]), h - 5, X(S.sel[1]) - X(S.sel[0]), 5); }
  g.strokeStyle = '#ff4d4d'; g.lineWidth = 1.3; g.beginPath(); g.moveTo(X(S.t), 0); g.lineTo(X(S.t), h); g.stroke();
  g.fillStyle = th.getPropertyValue('--dim'); g.font = '11px system-ui'; g.fillText('0 s', 6, 12); g.textAlign = 'right'; g.fillText(T.toFixed(1) + ' s', w - 12, 12); g.textAlign = 'left';
  if (S.meta.video_dur < S.meta.audio_dur + S.meta.settings.av_offset - 0.05) { g.fillStyle = 'rgba(229,115,106,.25)'; const xv = X(S.meta.video_dur); g.fillRect(xv, 0, w - MR - xv, 6); }
}
function drawWave() {
  const c = $('#cWave'); if (c.classList.contains('hidden')) return; const { g, w, h } = setupCanvas(c); const th = theme(); g.clearRect(0, 0, w, h);
  g.fillStyle = '#1b1d17'; if (document.documentElement.dataset.theme === 'light') g.fillStyle = '#fbf7e8'; g.fillRect(ML, 0, w - ML - MR, h);
  drawTimeGrid(g, w, h, false);
  g.strokeStyle = 'rgba(128,140,155,.3)'; g.beginPath(); g.moveTo(ML, h / 2 + .5); g.lineTo(w - MR, h / 2 + .5); g.stroke();
  if (S.wave && S.wave.t0 === S.win[0] && S.wave.t1 === S.win[1]) drawWaveArr(g, S.wave, w, h, S.win[0], S.win[1], '#e6edf5');
  else if (S.wave) { g.save(); g.globalAlpha = .35; const a = S.wave.t0, b = S.wave.t1, x0 = t2x(a, w), x1 = t2x(b, w); g.beginPath(); g.rect(ML, 0, w - ML - MR, h); g.clip(); g.translate(x0, 0); g.scale((x1 - x0) / (w - ML - MR), 1); g.translate(-ML, 0); drawWaveArr(g, S.wave, w, h, a, b, '#e6edf5'); g.restore(); }
  g.fillStyle = th.getPropertyValue('--dim'); g.font = '11px system-ui'; g.fillText('amp', 8, 14);
  drawCursorAndSel(g, w, h);
}
function drawSpec() {
  const c = $('#cSpec'); if (c.classList.contains('hidden')) return; const { g, w, h } = setupCanvas(c); const th = theme(); g.clearRect(0, 0, w, h);
  const bot = 16, ph = h - bot, fmax = +$('#sgFmax').value; g.fillStyle = '#000'; g.fillRect(ML, 0, w - ML - MR, ph);
  const s = S.spec;
  if (s && s.img) {
    const step = s.step, x0 = t2x(s.t0 - step / 2, w), x1 = t2x(s.t1 + step / 2, w); const fm = s.fmax; const ydraw = ph * (fmax / fm); // image covers 0..s.fmax
    g.save(); g.beginPath(); g.rect(ML, 0, w - ML - MR, ph); g.clip(); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    g.drawImage(s.img, x0, 0, x1 - x0, ph * (s.fmax / fmax)); g.restore();
  }
  g.fillStyle = th.getPropertyValue('--dim'); g.font = '11px system-ui'; g.strokeStyle = 'rgba(128,140,155,.35)'; g.textAlign = 'right';
  axisTicks(0, fmax, 6).forEach(f => { const y = Math.round(ph - f / fmax * ph) + .5; g.beginPath(); g.moveTo(ML - 4, y); g.lineTo(ML, y); g.stroke(); g.fillText((f / 1000).toFixed(f % 1000 ? 1 : 0) + 'k', ML - 6, y + 4); });
  g.textAlign = 'left'; g.save(); g.beginPath(); g.rect(0, 0, w, ph); g.clip(); drawTimeGrid(g, w, h, false); g.restore();
  g.strokeStyle = 'rgba(128,140,155,.3)'; g.beginPath(); g.rect(ML + .5, .5, w - ML - MR - 1, ph - 1); g.stroke();
  // time axis labels
  g.fillStyle = th.getPropertyValue('--dim'); g.textAlign = 'center';
  const dec = Math.max(0, 2 - Math.floor(Math.log10(S.win[1] - S.win[0])));
  axisTicks(S.win[0], S.win[1], Math.max(3, (w - ML - MR) / 90)).forEach(t => g.fillText(t.toFixed(dec), t2x(t, w), h - 3)); g.textAlign = 'left';
  // formants
  if ($('#showFmt').checked && S.fmt) {
    const cols = ['#ff5d5d', '#ffc94d', '#6dff8a', '#5ad0ff', '#c58bff', '#ffa0e0']; const F = S.fmt.F, T = S.fmt.t;
    for (let k = 0; k < T.length; k++) { const x = t2x(T[k], w); if (x < ML || x > w - MR) continue; (F[k] || []).forEach((f, i) => { if (f == null || f > fmax) return; g.fillStyle = cols[i % cols.length]; g.beginPath(); g.arc(x, ph - f / fmax * ph, 2.1, 0, 7); g.fill(); }); }
  }
  g.save(); g.beginPath(); g.rect(0, 0, w, ph); g.clip(); drawCursorAndSel(g, w, ph);
  if (S.hoverF != null && S.hoverT != null) {
    const yh = ph - S.hoverF / fmax * ph, xh = t2x(S.hoverT, w); g.strokeStyle = 'rgba(255,255,255,.55)'; g.lineWidth = 1; g.beginPath(); g.moveTo(ML, Math.round(yh) + .5); g.lineTo(w - MR, Math.round(yh) + .5); g.stroke();
    const txt = `${S.hoverT.toFixed(3)} s · ${Math.round(S.hoverF)} Hz`; g.font = '11px system-ui'; const tw = g.measureText(txt).width + 10; let bx = xh + 10; if (bx + tw > w - MR) bx = xh - tw - 10; const by = clamp(yh - 20, 2, ph - 20);
    g.fillStyle = 'rgba(15,17,20,.88)'; g.fillRect(bx, by, tw, 17); g.fillStyle = '#fff'; g.fillText(txt, bx + 5, by + 12);
  }
  g.restore();
  g.fillStyle = 'rgba(255,255,255,.0)';
}
/* ------------------------------------------------------------------ audio-canvas mouse */
function bindAudioMouse(c, isSpec) {
  let drag = null;
  const pos = e => { const r = c.getBoundingClientRect(); return e.clientX - r.left; };
  c.addEventListener('mousedown', e => { if (e.button !== 0) return; drag = { x0: pos(e), moved: false }; });
  window.addEventListener('mousemove', e => {
    if (!drag) return; const x = pos(e); if (Math.abs(x - drag.x0) > 4) drag.moved = true;
    if (drag.moved) { const w = c.clientWidth; const a = x2t(drag.x0, w), b = x2t(x, w); S.sel = [clamp(Math.min(a, b), 0, tEnd()), clamp(Math.max(a, b), 0, tEnd())]; updateTimeUI(); draw('over', 'wave', 'spec'); }
  });
  window.addEventListener('mouseup', e => { if (!drag) return; if (!drag.moved) { setTime(x2t(drag.x0, c.clientWidth), { keepWindow: true }); } drag = null; });
  c.addEventListener('dblclick', () => { S.sel = null; updateTimeUI(); drawAudioAll(); });
  c.addEventListener('mousemove', e => { const t = x2t(pos(e), c.clientWidth); S.hoverT = t; S.hoverF = null; let hz = '';
    if (isSpec) { const r = c.getBoundingClientRect(), ph = r.height - 16, y = e.clientY - r.top; if (y >= 0 && y <= ph) { S.hoverF = (ph - y) / ph * (+$('#sgFmax').value); hz = `  ${Math.round(S.hoverF)} Hz`; } }
    $('#readout').textContent = `t = ${S.t.toFixed(3)} s   frame ${S.frame + 1}   ·  cursor ${t.toFixed(3)} s${hz}`; draw('wave', 'spec'); });
  c.addEventListener('mouseleave', () => { S.hoverT = null; S.hoverF = null; draw('wave', 'spec'); updateTimeUI(); });
  c.addEventListener('wheel', e => {
    e.preventDefault(); const w = c.clientWidth; const tc = x2t(pos(e), w); const [a, b] = S.win, L = b - a;
    if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) { const d = (e.deltaX || e.deltaY) / 500 * L; S.win = [a + d, b + d]; }
    else { const k = e.deltaY > 0 ? 1.2 : 1 / 1.2; const nl = clamp(L * k, 0.02, tEnd()); const f = (tc - a) / L; S.win = [tc - f * nl, tc + (1 - f) * nl]; }
    fixWindow(); windowChanged();
  }, { passive: false });
}
function bindOverview() {
  const c = $('#cOver'); let drag = null;
  const T2 = e => { const r = c.getBoundingClientRect(); return clamp((e.clientX - r.left - ML) / (c.clientWidth - ML - MR), 0, 1) * tEnd(); };
  c.addEventListener('mousedown', e => {
    const t = T2(e); const L = S.win[1] - S.win[0];
    if (t >= S.win[0] && t <= S.win[1]) drag = { off: t - S.win[0], L, moved: false };
    else { S.win = [t - L / 2, t + L / 2]; fixWindow(); setTime(t, { keepWindow: true }); windowChanged(); drag = { off: L / 2, L, moved: true }; }
  });
  window.addEventListener('mousemove', e => { if (!drag) return; const t = T2(e); drag.moved = true; S.win = [t - drag.off, t - drag.off + drag.L]; fixWindow(); windowChanged(); });
  window.addEventListener('mouseup', () => { drag = null; });
}

/* ------------------------------------------------------------------ video mouse (landmarks / edit) */
function videoXY(e) {
  const c = $('#video'), r = c.getBoundingClientRect(), m = S.meta, sc = Math.min(r.width / m.w, r.height / m.h);
  const ox = (r.width - m.w * sc) / 2, oy = (r.height - m.h * sc) / 2; return [(e.clientX - r.left - ox) / sc - 0.5, (e.clientY - r.top - oy) / sc - 0.5];
}
async function onVideoClick(e) {
  if (!S.meta) return; const [x, y] = videoXY(e);
  if (S.lmMode) {
    S.lm[S.lmMode] = [x, y]; const i = LM_ORDER.indexOf(S.lmMode); S.lmMode = i < 3 ? LM_ORDER[i + 1] : null; updateLmUI(); draw('video');
    if (!S.lmMode && LM_ORDER.every(k => S.lm[k])) {
      try { const tr = await postJSON('/api/tract/landmarks', { landmarks: S.lm }); applyTractMeta(tr); toast('Grid built. Use "Set reference frame" on a frame with a clearly visible palate and pharyngeal wall for steadier results.', 8000); requestTract(); }
      catch (err) { toast(err.message, 10000); }
    }
    return;
  }
  const mode = $('#editMode').value;
  if (mode && S.meta.tract && S.meta.tract.grid) {
    try { const r = await postJSON('/api/tract/edit', { frame: S.frame, x, y, side: mode }); setTractRes(r); } catch (err) { toast(err.message); }
  }
}
function updateLmUI() {
  $$('button.lm').forEach(b => { const k = b.dataset.lm; b.classList.toggle('active', S.lmMode === k); b.classList.toggle('done', !!S.lm[k] && S.lmMode !== k); });
  $('#lmHelp').textContent = S.lmMode ? LM_HELP[S.lmMode] : '';
  $('#video').style.cursor = S.lmMode || $('#editMode').value ? 'crosshair' : 'default';
}

/* ------------------------------------------------------------------ area function */
function applyTractMeta(tr) {
  S.meta.tract = tr; if (!tr) return;
  if (tr.landmarks && Object.keys(tr.landmarks).length) S.lm = { ...tr.landmarks }; else if (!tr.grid) S.lm = {};
  $$('#afParams [data-p]').forEach(i => { if (document.activeElement !== i && tr.params[i.dataset.p] != null) i.value = tr.params[i.dataset.p]; });
  $$('#afParams [data-pb]').forEach(i => i.checked = !!tr.params[i.dataset.pb]);
  $$('#afParams [data-pr]').forEach(i => { const v = tr.params[i.dataset.pr] || (tr.grid ? tr.grid[i.dataset.pr] : null); i.value = v ? v.join(',') : ''; });
  S.meta.pixel_mm = tr.pixel_mm; $('#pxInfo').textContent = `${fmt(tr.pixel_mm, 3)} mm/px  (${S.meta.w}×${S.meta.h} px)`;
  $('#afRef').textContent = tr.ref_frame != null ? `Reference: frame ${tr.ref_frame + 1} ✓ (reset)` : 'Set reference frame';
  updateLmUI(); draw('video', 'af');
}
const requestTractDebounced = debounce(() => requestTract(), 35);
async function requestTract() {
  const tr = S.meta && S.meta.tract; if (!tr || !tr.grid || !$('#afAuto').checked) { if (!tr || !tr.grid) { S.tractRes = null; draw('video', 'af'); } return; }
  const seq = ++S.seq.tract; try { const r = await postJSON('/api/tract/frame', { frame: S.frame }); if (seq === S.seq.tract) setTractRes(r); } catch (e) { if (seq === S.seq.tract) { S.tractRes = null; $('#afMsg').textContent = 'Extraction failed: ' + e.message; draw('video', 'af'); } }
}
function setTractRes(r) { S.tractRes = r; $('#afMsg').textContent = ''; draw('video', 'af'); updateAFReadouts(); }
async function updateAFReadouts() {
  const r = S.tractRes; if (!r) return;
  $('#afStats').textContent = `tract ${fmt(r.vt_len_mm, 0)} mm · ${r.ngl} gridlines · mid-sag. area ${fmt(r.mid_area_mm2, 0)} mm²` + (r.shift && (r.shift[0] || r.shift[1]) ? ` · shift ${fmt(r.shift[0], 1)},${fmt(r.shift[1], 1)} px` : '');
  let line = `tube model F1–F${r.predicted_formants.length}: ${r.predicted_formants.map(f => Math.round(f)).join('  ')} Hz`;
  if (S.meta.has_audio) { const m = await measuredFormants(r.time); if (m) line += `\nmeasured (LPC) at ${r.time.toFixed(3)} s: ${m.map(f => f == null ? '–' : Math.round(f)).join('  ')} Hz`; }
  $('#afFmt').textContent = line; $('#afFmt').style.whiteSpace = 'pre';
}
const fmtCache = new Map();
async function measuredFormants(t) {
  const key = [t.toFixed(3), $('#fN').value, $('#fMax').value, $('#fWin').value, $('#fOrd').value, $('#fBw').value, $('#fLvl').value, $('#fScale').value, S.meta.settings.av_offset, S.meta.settings.burst_s].join('|');
  if (fmtCache.has(key)) return fmtCache.get(key);
  try {
    const d = await getJSON('/api/formants?' + fmtQuery(Math.max(0, t - 0.03), t + 0.03)); const n = +$('#fN').value; const out = [];
    for (let i = 0; i < n; i++) { const v = d.F.map(r => r[i]).filter(x => x != null).sort((a, b) => a - b); out.push(v.length ? v[v.length >> 1] : null); }
    if (fmtCache.size > 400) fmtCache.clear(); fmtCache.set(key, out); return out;
  } catch (e) { return null; }
}
async function updateFmtReadout() {
  if (!S.meta || !S.meta.has_audio) return; const m = await measuredFormants(S.t);
  $('#fmtReadout').textContent = m ? `t = ${S.t.toFixed(3)} s\n` + m.map((f, i) => `F${i + 1} = ${f == null ? '–' : Math.round(f) + ' Hz'}`).join('\n') : '—'; $('#fmtReadout').style.whiteSpace = 'pre';
}
function drawAFplot() {
  const c = $('#cAFplot'); const { g, w, h } = setupCanvas(c); const th = theme(); g.clearRect(0, 0, w, h); const r = S.tractRes;
  g.fillStyle = th.getPropertyValue('--dim'); g.font = '11px system-ui';
  if (!r) { g.fillText(S.meta.tract && S.meta.tract.grid ? 'Waiting for extraction…' : 'Place the four landmarks to build the grid.', 14, 24); return; }
  const unit = $('#afUnit').value, vals = unit === 'mm' ? r.aperture_mm : r.area_cm2, x = r.x_mm;
  const L = 44, R = 12, T = 12, B = 28, pw = w - L - R, ph = h - T - B;
  const xmax = Math.max(40, Math.ceil(x[x.length - 1] / 10) * 10); const vmax = Math.max(unit === 'mm' ? 10 : 2, ...vals) * 1.1;
  const X = v => L + v / xmax * pw, Y = v => T + ph - v / vmax * ph;
  g.strokeStyle = 'rgba(128,140,155,.25)'; g.textAlign = 'right';
  axisTicks(0, vmax, 5).forEach(v => { const y = Math.round(Y(v)) + .5; g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke(); g.fillText(v.toFixed(unit === 'mm' ? 0 : 1), L - 5, y + 4); });
  g.textAlign = 'center'; axisTicks(0, xmax, 8).forEach(v => { const xx = Math.round(X(v)) + .5; g.beginPath(); g.moveTo(xx, T); g.lineTo(xx, T + ph); g.stroke(); g.fillText(v, xx, h - 14); });
  g.fillText('distance from glottis (mm)  →  lips', L + pw / 2, h - 2); g.save(); g.translate(11, T + ph / 2); g.rotate(-Math.PI / 2); g.fillText(unit === 'mm' ? 'aperture (mm)' : 'area (cm²)', 0, 0); g.restore(); g.textAlign = 'left';
  g.strokeStyle = '#4cc2b0'; g.lineWidth = 2; g.beginPath(); let prev = 0;
  for (let k = 0; k < x.length; k++) { const a = X(prev), b = X(x[k]), y = Y(vals[k]); if (k === 0) g.moveTo(a, y); else g.lineTo(a, y); g.lineTo(b, y); prev = x[k]; } g.stroke();
  if (S.hoverGl >= 0 && S.hoverGl < x.length) { const k = S.hoverGl, a = k ? x[k - 1] : 0; g.fillStyle = 'rgba(255,59,48,.25)'; g.fillRect(X(a), T, X(x[k]) - X(a), ph); }
}
function bindAFplot() {
  const c = $('#cAFplot');
  c.addEventListener('mousemove', e => {
    const r = S.tractRes; if (!r) return; const rect = c.getBoundingClientRect(), L = 44, R = 12, pw = rect.width - L - R; const xmax = Math.max(40, Math.ceil(r.x_mm[r.x_mm.length - 1] / 10) * 10);
    const d = (e.clientX - rect.left - L) / pw * xmax; let k = r.x_mm.findIndex(v => v >= d); if (k < 0) k = -1; S.hoverGl = d < 0 ? -1 : k;
    if (k >= 0) $('#afMsg').textContent = `gridline ${k + 1}: ${fmt(d, 0)} mm from glottis · aperture ${fmt(r.aperture_mm[k], 1)} mm · area ${fmt(r.area_cm2[k], 2)} cm²`;
    draw('video', 'af');
  });
  c.addEventListener('mouseleave', () => { S.hoverGl = -1; draw('video', 'af'); });
}

async function exportRange() {
  const tr = S.meta.tract; if (!tr || !tr.grid) { toast('Place the four landmarks first.'); return; }
  const [a, b] = rangeFor($('#batchRange').value); const i0 = frameAt(a), i1 = frameAt(b); const step = Math.max(1, +$('#batchStep').value || 1);
  const prog = $('#prog'); prog.classList.remove('hidden'); prog.firstElementChild.style.width = '0%'; $('#afBatch').disabled = true;
  try {
    const job = await postJSON('/api/tract/batch', { start: i0, end: i1, step });
    for (;;) { await new Promise(r => setTimeout(r, 250)); const j = await getJSON('/api/job/' + job.id); prog.firstElementChild.style.width = (j.progress * 100).toFixed(0) + '%'; if (j.error) throw new Error(j.error); if (j.done) break; }
    download('/api/tract/batch.csv?kind=' + $('#batchKind').value); toast(`Exported ${Math.floor((i1 - i0) / step) + 1} frames (frames ${i0 + 1}–${i1 + 1}).`, 5000);
  } catch (e) { toast('Export failed: ' + e.message, 10000); }
  prog.classList.add('hidden'); $('#afBatch').disabled = false;
}
let synthBlob = null;
async function synth(playIt) {
  if (!S.tractRes) { toast('Extract an area function first (place the landmarks).'); return; }
  try {
    const r = await api('/api/synth', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ frame: S.frame, f0: +$('#synF0').value, dur: +$('#synDur').value }) });
    synthBlob = await r.blob(); const u = URL.createObjectURL(synthBlob);
    if (playIt) { stop(); const a = new Audio(u); a.onended = () => URL.revokeObjectURL(u); a.play(); } else { download(u, `${S.meta.token}_frame${S.frame + 1}_synth.wav`); }
  } catch (e) { toast('Synthesis failed: ' + e.message); }
}

/* ------------------------------------------------------------------ labels */
function renderLabels() {
  const L = S.meta.labels || []; const el = $('#labList'); el.innerHTML = '';
  L.forEach((l, i) => { const d = document.createElement('div'); d.className = 'lab'; d.innerHTML = `<span>${l.text.replace(/</g, '&lt;')}</span><span class="mono">${l.t0.toFixed(3)}–${l.t1.toFixed(3)}</span><span class="spacer"></span><button data-i="${i}">✕</button>`; d.onclick = ev => { if (ev.target.tagName === 'BUTTON') { L.splice(i, 1); saveLabels(); renderLabels(); return; } S.sel = [l.t0, l.t1]; setTime(l.t0); S.win = [l.t0 - (l.t1 - l.t0) * .5, l.t1 + (l.t1 - l.t0) * .5]; fixWindow(); windowChanged(); }; el.appendChild(d); });
}
const saveLabels = () => postJSON('/api/labels', { labels: S.meta.labels }).catch(e => toast(e.message));

/* ------------------------------------------------------------------ settings & view persistence */
async function pushSettings(patch) {
  try {
    const r = await postJSON('/api/settings', patch); S.meta.settings = r.settings; S.meta.t_end = r.t_end; S.meta.pixel_mm = r.pixel_mm;
    if (r.flip_changed) { S.chunks.clear(); S.loading.clear(); S.tractRes = null; S.lm = {}; S.batch = null; prefetchAll(++S.gen); loadGain(); }
    applyTractMeta(r.tract); applySettingsToUI(S.meta); fmtCache.clear();
    fixWindow(); refreshOverview(); refreshWindowData(); requestTract(); drawAudioAll(); draw('video');
  } catch (e) { toast(e.message); }
}
const scheduleViewSave = debounce(() => {
  if (!S.meta) return; const v = { t: S.t, winLen: S.win[1] - S.win[0], showFmt: $('#showFmt').checked, sgPre: $('#sgPre').checked };
  ['sgWin', 'sgStep', 'sgFmax', 'sgDr', 'sgTop', 'sgMap', 'fN', 'fMax', 'fWin', 'fStep', 'fOrd', 'fBw', 'fLvl', 'fScale'].forEach(id => v[id] = $('#' + id).value);
  S.meta.view = v; postJSON('/api/view', v).catch(() => {});
}, 1500);

/* ------------------------------------------------------------------ export helpers */
function download(url, name) { const a = document.createElement('a'); a.href = url; if (name) a.download = name; document.body.appendChild(a); a.click(); a.remove(); }
function saveCanvasPng(canvas, name) { canvas.toBlob(b => { const u = URL.createObjectURL(b); download(u, name); setTimeout(() => URL.revokeObjectURL(u), 5000); }); }

/* ------------------------------------------------------------------ wiring */
function bind() {
  $('#fileSel').onchange = openSelected;
  $('#btnFolder').onclick = async () => {
    const p = prompt('Folder (or video file) to scan, on the computer running this app:', $('#folderPath').textContent || ''); if (!p) return;
    try { await postJSON('/api/scan', { path: p }); await loadFiles(); if ($('#fileSel').options.length > 1 || $('#fileSel').value) { /* user picks */ } } catch (e) { toast(e.message); }
  };
  $('#bPlay').onclick = () => S.playing ? stop() : play(S.t, null);
  $('#bPrev').onclick = () => stepFrame(-1); $('#bNext').onclick = () => stepFrame(1);
  $('#slider').oninput = e => { const f = +e.target.value; setTime(S.ft[f]); };
  $('#frameNo').onchange = e => { const f = clamp((+e.target.value | 0) - 1, 0, S.meta.n - 1); setTime(S.ft[f]); };
  $('#timeNo').onchange = e => setTime(+e.target.value);
  $('#rate').onchange = () => { if (S.playing) play(S.t, pb && pb.t1); };
  $('#bPlayWin').onclick = () => play(S.win[0], S.win[1]);
  $('#bPlaySel').onclick = () => { if (!S.sel) { toast('Drag on the waveform or spectrogram to select an interval first.'); return; } play(S.sel[0], S.sel[1]); };
  $('#bSavePng').onclick = () => { const m = S.meta, d = getFrame(S.frame); if (!d) return; const c = document.createElement('canvas'); const k = 4; c.width = m.w * k; c.height = m.h * k; const g = c.getContext('2d'); g.imageSmoothingQuality = 'high'; g.drawImage(offscreen, 0, 0, c.width, c.height); saveCanvasPng(c, `${m.token}_f${S.frame + 1}.png`); };
  $('#video').addEventListener('click', onVideoClick);
  // image controls
  const readImg = () => { S.img.bri = +$('#iBri').value; S.img.con = +$('#iCon').value / 100; S.img.gam = +$('#iGam').value / 100; S.img.smooth = $('#iSmooth').checked; draw('video'); };
  ['iBri', 'iCon', 'iGam', 'iSmooth'].forEach(id => $('#' + id).oninput = readImg);
  $('#iFlat').onchange = e => { S.img.flat = e.target.checked; pushSettings({ flatten: e.target.checked }); };
  $('#iReset').onclick = () => { $('#iBri').value = 0; $('#iCon').value = 100; $('#iGam').value = 100; readImg(); };
  // window controls
  $('#winLen').onchange = e => { const L = clamp(+e.target.value, 0.02, tEnd()); const c = (S.win[0] + S.win[1]) / 2; S.win = [c - L / 2, c + L / 2]; fixWindow(); windowChanged(); };
  $('#winMinus').onclick = () => { const L = (S.win[1] - S.win[0]) / 1.5, c = S.t; S.win = [c - L / 2, c + L / 2]; fixWindow(); windowChanged(); };
  $('#winPlus').onclick = () => { const L = (S.win[1] - S.win[0]) * 1.5, c = S.t; S.win = [c - L / 2, c + L / 2]; fixWindow(); windowChanged(); };
  $('#winAll').onclick = () => { S.win = [0, tEnd()]; windowChanged(); };
  $('#showFmt').onchange = () => { if ($('#showFmt').checked) fetchFmt(); draw('spec'); scheduleViewSave(); };
  bindAudioMouse($('#cWave')); bindAudioMouse($('#cSpec'), true);  bindOverview(); bindAFplot();
  // display params
  const respec = () => { fetchSpec(); scheduleViewSave(); };
  ['sgWin', 'sgStep', 'sgFmax', 'sgDr', 'sgTop', 'sgPre'].forEach(id => $('#' + id).onchange = respec);
  $('#sgMap').onchange = () => { colorizeSpec(); draw('spec'); scheduleViewSave(); };
  ['fN', 'fMax', 'fWin', 'fStep', 'fOrd', 'fBw', 'fLvl', 'fScale'].forEach(id => $('#' + id).onchange = () => { fmtCache.clear(); if ($('#showFmt').checked) fetchFmt(); updateFmtReadout(); scheduleViewSave(); });
  // tabs
  $$('.tabs nav button').forEach(b => b.onclick = () => { $$('.tabs nav button').forEach(x => x.classList.toggle('on', x === b)); $$('.tab').forEach(t => t.classList.toggle('on', t.id === b.dataset.tab)); draw('af'); });
  // sync / geometry
  $('#avOff').onchange = e => pushSettings({ av_offset: +e.target.value });
  $$('[data-nudge]').forEach(b => b.onclick = () => { const v = Math.round(((+$('#avOff').value) + +b.dataset.nudge) * 1000) / 1000; $('#avOff').value = v; pushSettings({ av_offset: v }); });
  $('#syncEst').onclick = estimateSync;
  $('#fov').onchange = e => pushSettings({ fov_mm: +e.target.value });
  $('#pxOverride').onchange = e => pushSettings({ pixel_mm: e.target.value ? +e.target.value : null });
  $('#flipH').onchange = e => pushSettings({ flip_h: e.target.checked });
  $('#burstS').onchange = e => pushSettings({ burst_s: +e.target.value }); $('#burstA').onchange = e => pushSettings({ burst_atten: +e.target.value });
  // area function
  $$('button.lm').forEach(b => b.onclick = () => { S.lmMode = (S.lmMode === b.dataset.lm) ? null : b.dataset.lm; updateLmUI(); if (S.lmMode) toast(LM_HELP[S.lmMode], 5000); else $('#banner').classList.add('hidden'); });
  $('#afClearLm').onclick = async () => { S.lm = {}; S.lmMode = null; S.tractRes = null; applyTractMeta(await postJSON('/api/tract/clear')); };
  $('#afRef').onclick = async () => { try { applyTractMeta(await postJSON('/api/tract/ref', { frame: S.frame })); requestTract(); toast(`Frame ${S.frame + 1} is now the reference for palate and pharyngeal wall.`); } catch (e) { toast(e.message); } };
  $('#afRefClear').onclick = async () => { applyTractMeta(await postJSON('/api/tract/ref', { frame: S.frame, clear: true })); requestTract(); };
  $('#afAuto').onchange = () => { if ($('#afAuto').checked) requestTract(); };
  ['showGrid', 'showBnd', 'showCen'].forEach(id => $('#' + id).onchange = () => draw('video'));
  $('#editMode').onchange = updateLmUI;
  $('#afClearEdits').onclick = async () => { try { setTractRes(await postJSON('/api/tract/clear_edits', { frame: S.frame })); } catch (e) { toast(e.message); } };
  $('#afBatch').onclick = exportRange;
  $('#afCsvFrame').onclick = () => { if (!S.tractRes) { toast('Extract an area function first (place the landmarks).'); return; } download('/api/tract/frame.csv?frame=' + S.frame); };
  $('#afSynth').onclick = () => synth(true); $('#afSynthSave').onclick = () => synth(false); $('#afUnit').onchange = () => draw('af');
  $$('#afParams [data-p]').forEach(i => i.onchange = async () => { try { applyTractMeta(await postJSON('/api/tract/params', { [i.dataset.p]: i.value === '' ? null : +i.value })); requestTract(); } catch (e) { toast(e.message); } });
  $$('#afParams [data-pb]').forEach(i => i.onchange = async () => { try { applyTractMeta(await postJSON('/api/tract/params', { [i.dataset.pb]: i.checked })); requestTract(); } catch (e) { toast(e.message); } });
  $$('#afParams [data-pr]').forEach(i => i.onchange = async () => { const v = i.value.split(',').map(s => parseInt(s, 10) - 0).filter(n => !isNaN(n)); if (v.length !== 2) return; try { applyTractMeta(await postJSON('/api/tract/params', { [i.dataset.pr]: v })); requestTract(); } catch (e) { toast(e.message); } });
  // labels / export
  $('#labAdd').onclick = () => { const txt = $('#labText').value.trim(); if (!txt) { toast('Type a label first.'); return; } const [a, b] = selOrWin(); (S.meta.labels = S.meta.labels || []).push({ text: txt, t0: a, t1: b }); saveLabels(); renderLabels(); };
  $('#expClip').onclick = () => { const [a, b] = rangeFor($('#expRange').value); $('#expMsg').textContent = 'Encoding…'; fetch(`/api/export/clip?t0=${a}&t1=${b}`).then(async r => { if (!r.ok) throw new Error((await r.json()).error); const bl = await r.blob(); const u = URL.createObjectURL(bl); download(u, `${S.meta.token}_${a.toFixed(2)}-${b.toFixed(2)}.mp4`); $('#expMsg').textContent = 'Saved (also written to the output folder).'; }).catch(e => $('#expMsg').textContent = e.message); };
  $('#expWav').onclick = () => { const [a, b] = rangeFor($('#expRange').value); download(`/api/export/clip?kind=wav&t0=${a}&t1=${b}`); };
  $('#expSpec').onclick = () => saveCanvasPng($('#cSpec'), `${S.meta.token}_spectrogram_${S.win[0].toFixed(2)}-${S.win[1].toFixed(2)}.png`);
  $('#expFmt').onclick = () => { const [a, b] = rangeFor($('#expRange').value); download(`/api/formants.csv?${fmtQuery(a, b)}`); };
  $('#expSession').onclick = async () => { const r = await postJSON('/api/save_session'); $('#expMsg').textContent = 'Saved: ' + r.path; };
  // help
  $('#btnHelp').onclick = showHelp; $('#modal').onclick = e => { if (e.target.id === 'modal') $('#modal').classList.add('hidden'); };
  window.addEventListener('keydown', onKey); window.addEventListener('resize', debounce(() => { refreshOverview(); refreshWindowData(); draw('video', 'af'); drawAudioAll(); }, 120));
  $('#aud').addEventListener('error', () => { if (S.meta && S.meta.has_audio) toast('Audio playback failed in this browser.'); });
}
function stepFrame(d) { if (S.playing) stop(); const f = clamp(S.frame + d, 0, S.meta.n - 1); setTime(S.ft[f]); }
function onKey(e) {
  const tg = e.target.tagName; if (tg === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox' || tg === 'SELECT' || tg === 'TEXTAREA') return; if (!S.meta) return;
  if (e.key === ' ') { e.preventDefault(); S.playing ? stop() : play(S.t, null); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); stepFrame(e.shiftKey ? 10 : e.ctrlKey ? Math.round(S.meta.fps) : 1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); stepFrame(e.shiftKey ? -10 : e.ctrlKey ? -Math.round(S.meta.fps) : -1); }
  else if (e.key === 'Home') setTime(0); else if (e.key === 'End') setTime(tEnd());
  else if (e.key === 'Escape') { S.lmMode = null; S.sel = null; $('#editMode').value = ''; updateLmUI(); updateTimeUI(); drawAudioAll(); $('#modal').classList.add('hidden'); }
  else if (e.key === '?') showHelp();
  else if (e.key === 'f') { $('#showFmt').checked = !$('#showFmt').checked; $('#showFmt').onchange(); }
  else if (e.key === 'w') play(S.win[0], S.win[1]);
  else if (e.key === 's' && S.sel) play(S.sel[0], S.sel[1]);
}
function showHelp() {
  $('#modalBox').innerHTML = `<h3 style="margin-top:0">Shortcuts &amp; mouse</h3><table>
  <tr><td><kbd>Space</kbd></td><td>play / pause (to end of selection if inside one)</td></tr>
  <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>previous / next frame (<kbd>Shift</kbd> ×10, <kbd>Ctrl</kbd> ±1 s)</td></tr>
  <tr><td><kbd>w</kbd> / <kbd>s</kbd></td><td>play window / selection</td></tr>
  <tr><td><kbd>f</kbd></td><td>toggle formant overlay</td></tr>
  <tr><td>click waveform / spectrogram</td><td>move cursor (video follows)</td></tr>
  <tr><td>drag waveform / spectrogram</td><td>select an interval (double-click clears)</td></tr>
  <tr><td>wheel</td><td>zoom the window around the pointer; <kbd>Shift</kbd>+wheel pans</td></tr>
  <tr><td>overview</td><td>click or drag the teal box to move the window</td></tr>
  <tr><td><kbd>Esc</kbd></td><td>cancel landmark / edit mode, clear selection</td></tr></table>
  <p><button onclick="document.getElementById('modal').classList.add('hidden')">Close</button></p>`; $('#modal').classList.remove('hidden');
}
async function estimateSync() {
  const out = $('#syncOut'); out.textContent = 'Working…';
  try {
    const r = await getJSON('/api/estimate_sync'); if (!r.ok) { out.textContent = r.reason; return; }
    const conf = r.peak / Math.max(r.second, 1e-6);
    out.innerHTML = `Best lag: <b>${r.offset.toFixed(2)} s</b> (peak ${r.peak.toFixed(3)}, next best elsewhere ${r.second.toFixed(3)}). ` +
      (conf > 1.5 ? 'Peak is clearly separated.' : 'Peak is not much higher than other lags (repetitive speech or noise): treat as a hint and verify by eye.') +
      ` <button id="applySync">Apply ${r.offset.toFixed(2)} s</button>`;
    $('#applySync').onclick = () => { $('#avOff').value = r.offset; pushSettings({ av_offset: r.offset }); };
    const c = $('#cSync'), { g, w, h } = setupCanvas(c); g.clearRect(0, 0, w, h); const mx = Math.max(...r.corr), mn = Math.min(...r.corr); g.strokeStyle = '#4cc2b0'; g.beginPath();
    r.corr.forEach((v, i) => { const x = i / (r.corr.length - 1) * w, y = h - 6 - (v - mn) / (mx - mn || 1) * (h - 16); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
    g.fillStyle = theme().getPropertyValue('--dim'); g.font = '10px system-ui'; g.fillText(`${r.lags[0]} s`, 3, h - 2); g.textAlign = 'right'; g.fillText(`${r.lags[r.lags.length - 1]} s`, w - 3, h - 2); g.textAlign = 'left';
  } catch (e) { out.textContent = e.message; }
}

/* ------------------------------------------------------------------ boot */
(async function boot() {
  bind(); updateLmUI();
  try { await loadFiles(); } catch (e) { toast(e.message); }
  try { const m = await getJSON('/api/meta'); if (m.open === false) { if ($('#fileSel').options.length > 1) toast('Choose a recording from the list.', 4000); } else init(m); } catch (e) { toast(e.message); }
  window.__S = S; // handy for debugging / tests
})();
