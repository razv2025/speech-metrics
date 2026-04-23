'use strict';

async function _decodeMonoBuffer(arrayBuf) {
  const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const decoded = await tempCtx.decodeAudioData(arrayBuf.slice(0));
    const mono = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += ch[i];
    }
    for (let i = 0; i < mono.length; i++) mono[i] /= decoded.numberOfChannels;
    return { decoded, mono };
  } finally {
    await tempCtx.close();
  }
}

function _setReplayBuffer(mono, sampleRate, filename) {
  asrAudioBuf = Array.from(mono);
  asrSampleRate = sampleRate;
  currentFilename = filename;
}

function _toggleReplayButton(btn, loading) {
  if (!btn) return;
  btn.disabled = loading;
  btn.textContent = loading ? '?' : '\x18';
  btn.classList.toggle('loading', loading);
}

async function replayPublishedEntry(task, pubId) {
  const btn = document.querySelector(`.log-play-btn[data-pub-id="${pubId}"]`);
  _toggleReplayButton(btn, true);

  try {
    const pub = (_publishedEntries[task] || []).find(e => e.id === pubId);
    if (!pub) throw new Error('Entry not in cache; refresh page');

    const audioResp = await fetch(`${SERVER_URL}/audio/${pubId}`);
    if (!audioResp.ok) throw new Error('Audio download failed');
    const arrayBuf = await audioResp.arrayBuffer();
    const { decoded, mono } = await _decodeMonoBuffer(arrayBuf);
    _setReplayBuffer(mono, pub.audio_sr || decoded.sampleRate, pub.filename || 'published');

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
    _toggleReplayButton(btn, false);
  } catch (err) {
    console.error('Published replay failed:', err);
    _toggleReplayButton(btn, false);
  }
}

async function replayLogEntry(task, id) {
  if (!logDB) return;

  const btn = document.querySelector('.log-play-btn[data-id="' + id + '"][data-task="' + task + '"]');
  _toggleReplayButton(btn, true);

  try {
    const entry = await new Promise((resolve, reject) => {
      const req = logDB.transaction('task' + task, 'readonly').objectStore('task' + task).get(id);
      req.onsuccess = e => resolve(e.target.result);
      req.onerror = () => reject(new Error('DB read failed'));
    });
    if (!entry || !entry.audioData) throw new Error('No audio stored for this entry');

    const { decoded, mono } = await _decodeMonoBuffer(entry.audioData);
    _setReplayBuffer(mono, entry.audioSR || decoded.sampleRate, entry.filename || 'replay');

    const blob = new Blob([entry.audioData], { type: 'audio/wav' });
    const form = new FormData();
    form.append('file', blob, 'recording.wav');
    const refText = entry.referenceText || (task === 3 ? activePassageRef : null);
    if (task === 3 && refText) form.append('reference_text', refText);
    const t0 = Date.now();
    const serverPromise = fetch(`${SERVER_URL}/analyze/${TASK_SLUG[task]}`, { method: 'POST', body: form })
      .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`));

    _replayMode = true;
    await processUploadedAudio();

    const d = await serverPromise;
    const analysis_duration_s = (Date.now() - t0) / 1000;
    _applyServerMetricsToUI(task, d);
    setStatus('Analysis complete', false);

    const newMetrics = _logFlatten(task, d);
    if (task === 3 && !refText) {
      delete newMetrics.articulation_pct;
      delete newMetrics.consonant_precision_pct;
    }
    newMetrics.analysis_duration_s = analysis_duration_s;
    await new Promise((resolve, reject) => {
      const tx = logDB.transaction('task' + task, 'readwrite');
      const store = tx.objectStore('task' + task);
      const req = store.get(id);
      req.onsuccess = ev => {
        const existing = ev.target.result;
        if (!existing) {
          reject(new Error('Entry gone'));
          return;
        }
        Object.assign(existing, newMetrics);
        store.put(existing).onsuccess = resolve;
      };
      req.onerror = reject;
    });

    renderLogTable(task);
    _toggleReplayButton(btn, false);
  } catch (err) {
    console.error('Re-analysis failed:', err);
    _toggleReplayButton(btn, false);
  }
}
