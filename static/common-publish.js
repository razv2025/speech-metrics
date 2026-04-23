'use strict';

function _getUserId() {
  let uid = localStorage.getItem('speech-metrics-uid');
  if (!uid) {
    uid = 'u' + Date.now().toString(36) + Math.random().toString(36).slice(2);
    localStorage.setItem('speech-metrics-uid', uid);
  }
  return uid;
}

const _publishedEntries = { 1: [], 2: [], 3: [] };
const _pubFilter = { 1: 'all', 2: 'all', 3: 'all' };

async function fetchPublished(task) {
  try {
    const resp = await fetch(`${SERVER_URL}/published/${TASK_SLUG[task]}`);
    if (!resp.ok) return;
    const rows = await resp.json();
    const myId = _getUserId();
    _publishedEntries[task] = rows.map(r => ({
      ...r.metrics,
      id: r.id,
      filename: r.filename,
      username: r.username || '',
      timestamp: r.published_at,
      audio_sr: r.audio_sr,
      duration_s: r.duration_s,
      version: r.version,
      _published: true,
      _mine: r.user_id === myId,
    }));
    renderLogTable(task);
  } catch (e) { /* server may be unreachable */ }
}

function cycleViewFilter(task) {
  const order = ['all', 'mine', 'published'];
  _pubFilter[task] = order[(order.indexOf(_pubFilter[task]) + 1) % order.length];
  renderLogTable(task);
}

function setViewFilter(task, f) {
  _pubFilter[task] = f;
  renderLogTable(task);
}

async function publishEntry(task, localId) {
  if (!logDB) return;
  const btn = document.querySelector(`.log-pub-btn[data-id="${localId}"][data-task="${task}"]`);
  if (btn) { btn.disabled = true; btn.textContent = '.'; }

  try {
    const entry = await new Promise((res, rej) => {
      const req = logDB.transaction('task' + task, 'readonly').objectStore('task' + task).get(localId);
      req.onsuccess = e => res(e.target.result);
      req.onerror = () => rej(new Error('DB read failed'));
    });
    if (!entry || !entry.audioData) throw new Error('No audio data');

    const skipKeys = new Set(['duration_s', 'analysis_duration_s']);
    const metrics = {};
    LOG_METRICS[task].forEach(m => {
      if (!skipKeys.has(m.key) && entry[m.key] != null) metrics[m.key] = entry[m.key];
    });

    const metadata = {
      user_id: _getUserId(),
      username: entry.username || _currentUsername,
      filename: entry.filename,
      metrics,
      reference_text: entry.referenceText || null,
      audio_sr: entry.audioSR || asrSampleRate,
      duration_s: entry.duration_s || null,
      version: entry.version || null,
    };

    const form = new FormData();
    form.append('file', new Blob([entry.audioData], { type: 'audio/wav' }), 'recording.wav');
    form.append('metadata', JSON.stringify(metadata));

    const resp = await fetch(`${SERVER_URL}/publish/${TASK_SLUG[task]}`, { method: 'POST', body: form });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const result = await resp.json();

    await new Promise((res, rej) => {
      const tx = logDB.transaction('task' + task, 'readwrite');
      const store = tx.objectStore('task' + task);
      const req = store.get(localId);
      req.onsuccess = ev => {
        const e = ev.target.result;
        if (!e) { rej(new Error('Entry gone')); return; }
        e.published_id = result.id;
        store.put(e).onsuccess = res;
      };
      req.onerror = rej;
    });

    await fetchPublished(task);
    renderLogTable(task);
  } catch (err) {
    console.error('Publish failed:', err);
    if (btn) { btn.disabled = false; btn.textContent = '??'; }
  }
}

async function deletePublishedEntry(task, pubId) {
  try {
    const resp = await fetch(`${SERVER_URL}/published/${pubId}?user_id=${encodeURIComponent(_getUserId())}`, { method: 'DELETE' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    _publishedEntries[task] = _publishedEntries[task].filter(e => e.id !== pubId);
    renderLogTable(task);
  } catch (err) {
    console.error('Delete published failed:', err);
  }
}

async function replayPublishedEntry(task, pubId) {
  const btn = document.querySelector(`.log-play-btn[data-pub-id="${pubId}"]`);
  if (btn) { btn.disabled = true; btn.textContent = '?'; btn.classList.add('loading'); }

  try {
    const pub = (_publishedEntries[task] || []).find(e => e.id === pubId);
    if (!pub) throw new Error('Entry not in cache; refresh page');

    const audioResp = await fetch(`${SERVER_URL}/audio/${pubId}`);
    if (!audioResp.ok) throw new Error('Audio download failed');
    const arrayBuf = await audioResp.arrayBuffer();

    const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await tempCtx.decodeAudioData(arrayBuf.slice(0));
    await tempCtx.close();
    const mono = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += ch[i];
    }
    for (let i = 0; i < mono.length; i++) mono[i] /= decoded.numberOfChannels;

    asrAudioBuf = Array.from(mono);
    asrSampleRate = pub.audio_sr || decoded.sampleRate;
    currentFilename = pub.filename || 'published';

    const form = new FormData();
    form.append('file', new Blob([arrayBuf], { type: 'audio/wav' }), 'recording.wav');
    if (task === 3 && pub.reference_text) form.append('reference_text', pub.reference_text);
    const t0 = Date.now();
    const serverPromise = fetch(`${SERVER_URL}/analyze/${TASK_SLUG[task]}`, { method: 'POST', body: form })
      .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`));

    _replayMode = true;
    await processUploadedAudio();

    const d = await serverPromise;
    const analysis_duration_s = (Date.now() - t0) / 1000;
    _applyServerMetricsToUI(task, d);
    setStatus('Analysis complete', false);

    addPendingLogEntry(task);
    _fillPendingLogEntry(task, d, analysis_duration_s);

    if (btn) { btn.disabled = false; btn.textContent = '\x18'; btn.classList.remove('loading'); }
  } catch (err) {
    console.error('Published replay failed:', err);
    if (btn) { btn.disabled = false; btn.textContent = '\x18'; btn.classList.remove('loading'); }
  }
}
