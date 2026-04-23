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

