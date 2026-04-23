'use strict';

let logDB = null;
let lastLogId = { 1: null, 2: null, 3: null };

function initLogDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('speech-metrics-log', 1);
    req.onupgradeneeded = e => {
      const d = e.target.result;
      [1, 2, 3].forEach(t => {
        if (!d.objectStoreNames.contains('task' + t)) {
          d.createObjectStore('task' + t, { keyPath: 'id', autoIncrement: true });
        }
      });
    };
    req.onsuccess = e => {
      logDB = e.target.result;
      resolve();
    };
    req.onerror = e => reject(e);
  });
}

function _logFlatten(task, d) {
  if (task === 1) return {
    f0_mean_hz: d.f0_mean_hz, spl_mean_db: d.spl_mean_db, mpt_s: d.mpt_s,
    jitter_pct: d.jitter_pct, shimmer_pct: d.shimmer_pct, cpp_db: d.cpp_db,
    hnr_db: d.hnr_db, noise_floor_db: d.noise_floor_db,
  };
  if (task === 2) return {
    f0_min_hz: d.f0_min_hz, f0_max_hz: d.f0_max_hz,
    f0_range_st: d.f0_range_st, noise_floor_db: d.noise_floor_db,
  };
  return {
    f0_mean_hz: d.f0_mean_hz, f0_std_hz: d.f0_std_hz, cpp_db: d.cpp_db,
    spl_mean_db: d.spl_mean_db, spl_variability_db: d.spl_variability_db,
    speech_rate_wpm: d.speech_rate_wpm, prosody_st: d.prosody_st,
    noise_floor_db: d.noise_floor_db, avg_pause_s: d.avg_pause_s,
    vsa_khz2: d.vsa_hz2 != null ? d.vsa_hz2 / 1000 : null,
    articulation_pct: d.articulation_pct,
    consonant_precision_pct: d.consonant_precision_pct,
  };
}

function addPendingLogEntry(task) {
  if (!logDB) return;
  const audioData = (asrAudioBuf && asrAudioBuf.length)
    ? encodeWAV(new Float32Array(asrAudioBuf), asrSampleRate)
    : null;
  const duration_s = asrAudioBuf.length / asrSampleRate;
  const entry = { filename: currentFilename, username: _currentUsername, timestamp: new Date().toISOString(), duration_s };
  if (audioData) { entry.audioData = audioData; entry.audioSR = asrSampleRate; }
  if (task === 3 && activePassageRef) entry.referenceText = activePassageRef;
  const tx = logDB.transaction('task' + task, 'readwrite');
  const req = tx.objectStore('task' + task).add(entry);
  req.onsuccess = e => {
    lastLogId[task] = e.target.result;
    renderLogTable(task);
  };
}

function _fillPendingLogEntry(task, data, analysis_duration_s) {
  if (!logDB || !lastLogId[task]) return;
  const tx = logDB.transaction('task' + task, 'readwrite');
  const store = tx.objectStore('task' + task);
  const req = store.get(lastLogId[task]);
  req.onsuccess = ev => {
    const entry = ev.target.result;
    if (!entry) return;
    Object.assign(entry, _logFlatten(task, data), { version: data.version || null });
    if (analysis_duration_s != null) entry.analysis_duration_s = analysis_duration_s;
    store.put(entry).onsuccess = () => renderLogTable(task);
  };
}

function updateLastLogFilename(task, filename) {
  if (!logDB || !lastLogId[task]) return;
  const tx = logDB.transaction('task' + task, 'readwrite');
  const store = tx.objectStore('task' + task);
  const req = store.get(lastLogId[task]);
  req.onsuccess = e => {
    const entry = e.target.result;
    if (!entry) return;
    entry.filename = filename;
    store.put(entry).onsuccess = () => renderLogTable(task);
  };
}

function _getAllLogEntries(task) {
  return new Promise(resolve => {
    if (!logDB) return resolve([]);
    const req = logDB.transaction('task' + task, 'readonly').objectStore('task' + task).getAll();
    req.onsuccess = e => resolve(e.target.result || []);
    req.onerror = () => resolve([]);
  });
}
