'use strict';

const TASK_SLUG = { 1: 'sustained-phonation', 2: 'pitch-glides', 3: 'reading-passage' };

/* ════════════════════════════════════════════
   SESSION STATE
════════════════════════════════════════════ */
let currentTask   = CURRENT_TASK;
let isRecording   = false;
let audioCtx      = null;
let mediaStream   = null;
let scriptProc    = null;
let analyserNode  = null;
let animId        = null;
let sampleRate    = 44100;

// Accumulators — reset per recording session
let f0History     = [];   // voiced F0 values, cap 500
let splHistory    = [];   // SPL (dB) values,  cap 500
let rmsHistory    = [];   // RMS values,        cap 500
let cppHistory    = [];   // CPP values,        cap 500
let noiseHistory  = [];   // SPL of silent/unvoiced frames, cap 500
let hnrHistory    = [];   // HNR (dB) of voiced frames, cap 500
let formantHistory = [];  // {f1,f2} Hz pairs from voiced frames, cap 500
let pauseDurations = [];  // seconds, each inter-speech pause
let pauseFrameCount = 0;  // silent frames in current pause
let hasSpeechStarted = false;
let inPause = false;
let pitchContour  = [];   // last 300 frames (0 = unvoiced)
let recentF0      = [];   // ring buffer last 31 voiced F0  (for live graph)
let recentRms     = [];   // ring buffer last 31 voiced RMS (for live graph)
let prevVoicedF0  = null; // previous voiced F0 for full-recording jitter
let prevVoicedRms = null; // previous voiced RMS for full-recording shimmer
let jitterDiffSum = 0;    // running sum of |F0[i]-F0[i-1]| over all voiced pairs
let jitterDiffCount = 0;
let shimmerDiffSum = 0;   // running sum of |RMS[i]-RMS[i-1]| over all voiced pairs
let shimmerDiffCount = 0;
let mptStart      = null; // timestamp of voiced onset
let mptTotal      = 0;    // accumulated voiced seconds
let minF0         = Infinity;
let maxF0         = -Infinity;
let sessionStart  = null;
let lastHnrCorr   = null; // last autocorrelation peak value

// Audio buffer state
let asrAudioBuf      = [];          // raw PCM collected during recording (at asrSampleRate)
let asrSampleRate    = 44100;       // sample rate of asrAudioBuf
let currentFilename  = 'New recording'; // filename for log entry

const CAP             = 500;
const CONTOUR_CAP     = 300;
const JITTER_WIN      = 30;
const SNR_GATE_DB     = 10;  // frame must be ≥10 dB above noise floor to be "speech"
const CPP_VOICE_THRESH = 1.0; // minimum CPP (dB) for a frame to count as voiced human speech
const SERVER_URL      = ''; // relative — HTML and API served from same origin


/* ════════════════════════════════════════════
   SESSION RESET
════════════════════════════════════════════ */
function _defaultFilename(task) {
  const now = new Date();
  const ts  = now.getFullYear() + '-'
    + String(now.getMonth() + 1).padStart(2, '0') + '-'
    + String(now.getDate()).padStart(2, '0') + '_'
    + String(now.getHours()).padStart(2, '0') + '-'
    + String(now.getMinutes()).padStart(2, '0');
  const names = TASK_SLUG;
  const ext   = task === 3 ? 'zip' : 'wav';
  return `${names[task]}_${ts}.${ext}`;
}

function resetSession() {
  f0History = []; splHistory = []; rmsHistory = []; cppHistory = []; noiseHistory = []; hnrHistory = [];
  formantHistory = []; pauseDurations = [];
  pauseFrameCount = 0; hasSpeechStarted = false; inPause = false;
  pitchContour = []; recentF0 = []; recentRms = [];
  prevVoicedF0 = null; prevVoicedRms = null;
  jitterDiffSum = 0; jitterDiffCount = 0;
  shimmerDiffSum = 0; shimmerDiffCount = 0;
  mptStart = null; mptTotal = 0;
  minF0 = Infinity; maxF0 = -Infinity;
  sessionStart = null; lastHnrCorr = null;
  asrAudioBuf = [];
  currentFilename = _defaultFilename(CURRENT_TASK);
  _resetPlaybackState();
  const dlId  = CURRENT_TASK === 3 ? 'download-btn' : 'download-btn-' + CURRENT_TASK;
  const dlBtn = document.getElementById(dlId);
  if (dlBtn) dlBtn.disabled = true;
}

function enableDownloadBtn(taskN) {
  const id  = taskN === 3 ? 'download-btn' : 'download-btn-' + taskN;
  const btn = document.getElementById(id);
  if (btn) btn.disabled = false;
}

/* ════════════════════════════════════════════
   RECORDING CONTROL
════════════════════════════════════════════ */
function toggleRecording() {
  if (isRecording) stopRecording();
  else             startRecording();
}

async function startRecording() {
  if (passageEditing) commitPassageEdit(); // commit any pending edit before recording
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    setStatus('Microphone access denied. Please allow microphone use and try again.', true);
    return;
  }

  resetSession();
  sessionStart = Date.now();

  audioCtx     = new (window.AudioContext || window.webkitAudioContext)();
  sampleRate   = audioCtx.sampleRate;

  const source = audioCtx.createMediaStreamSource(mediaStream);

  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 4096;

  scriptProc = audioCtx.createScriptProcessor(4096, 1, 1);
  scriptProc.onaudioprocess = onAudioProcess;

  source.connect(analyserNode);
  source.connect(scriptProc);
  scriptProc.connect(audioCtx.destination);

  asrSampleRate = audioCtx.sampleRate;
  isRecording = true;
  setBtn(true);
  setStatus('Recording\u2026', false);
  animId = requestAnimationFrame(renderLoop);

}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  if (mptStart !== null) { mptTotal += (Date.now() - mptStart) / 1000; mptStart = null; }

  if (scriptProc)   { scriptProc.onaudioprocess = null; scriptProc.disconnect(); }
  if (analyserNode) { analyserNode.disconnect(); }
  if (mediaStream)  { mediaStream.getTracks().forEach(t => t.stop()); }
  if (audioCtx)     { audioCtx.close(); audioCtx = null; }
  if (animId)       { cancelAnimationFrame(animId); animId = null; }

  scriptProc = null; analyserNode = null; mediaStream = null;
  setBtn(false);
  setStatus('Stopped', false);

  // Enable download for current task
  if (asrAudioBuf.length > 0) {
    enableDownloadBtn(currentTask);
    _enablePlaybackBtn();
    if (currentTask === 1) drawFullWaveform(null);
    else drawPitchContour(currentTask, null);
  }
  // Log immediately (pending), then send to server to fill in metrics
  if (asrAudioBuf.length > 0) {
    addPendingLogEntry(currentTask);
    sendToServer(currentTask);
  }
}

/* ════════════════════════════════════════════
   SERVER — Clinical-grade analysis (Praat + Whisper)
════════════════════════════════════════════ */

// IDs that belong to each task's server-computed metrics
const SERVER_METRIC_IDS = {
  1: ['t1-f0','t1-spl','t1-mpt','t1-jitter','t1-shimmer','t1-cpp','t1-hnr','t1-noise'],
  2: ['t2-minf0','t2-maxf0','t2-range','t2-noise'],
  3: ['t3-meanf0','t3-hspl','t3-f0std','t3-cpp','t3-mspl','t3-splvar',
      't3-rate','t3-prosody','t3-noise','t3-pause','t3-vsa','t3-artic','t3-consonants'],
};

function setPendingMetrics(task) {
  for (const id of (SERVER_METRIC_IDS[task] || [])) {
    const el = document.getElementById(id);
    if (el) el.textContent = '\u2026';   // "…"
  }
}

function clearMetrics(task) {
  for (const id of (SERVER_METRIC_IDS[task] || [])) {
    const el = document.getElementById(id);
    if (el) { el.textContent = '\u2014'; updateScale(id, null); }
  }
}

async function sendToServer(task) {
  setPendingMetrics(task);
  setStatus('Analyzing with Praat\u2026', false);
  try {
    const wav  = encodeWAV(new Float32Array(asrAudioBuf), asrSampleRate);
    const blob = new Blob([wav], { type: 'audio/wav' });
    const form = new FormData();
    form.append('file', blob, 'recording.wav');
    if (task === 3 && activePassageRef) form.append('reference_text', activePassageRef);

    const t0   = Date.now();
    const resp = await fetch(`${SERVER_URL}/analyze/${TASK_SLUG[task]}`, {
      method: 'POST',
      body: form,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const analysis_duration_s = (Date.now() - t0) / 1000;
    applyServerResults(task, data, analysis_duration_s);
    setStatus('Analysis complete', false);
    document.getElementById('server-error-banner').style.display = 'none';
  } catch (err) {
    console.error('Praat server error:', err);
    clearMetrics(task);
    setStatus('Server error — see banner below', true);
    document.getElementById('server-error-banner').style.display = 'block';
  }
}

function _applyServerMetricsToUI(task, d) {
  const set = (id, val, dec) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = (val !== null && val !== undefined) ? fmt(val, dec) : '\u2014';
    updateScale(id, (val !== null && val !== undefined) ? val : null);
  };

  if (task === 1) {
    set('t1-f0',      d.f0_mean_hz,    1);
    set('t1-spl',     d.spl_mean_db,   1);
    set('t1-mpt',     d.mpt_s,         1);
    set('t1-jitter',  d.jitter_pct,    2);
    set('t1-shimmer', d.shimmer_pct,   2);
    set('t1-cpp',     d.cpp_db,        1);
    set('t1-hnr',     d.hnr_db,        1);
    set('t1-noise',   d.noise_floor_db,1);
  } else if (task === 2) {
    set('t2-minf0',   d.f0_min_hz,     1);
    set('t2-maxf0',   d.f0_max_hz,     1);
    set('t2-range',   d.f0_range_st,   1);
    set('t2-noise',   d.noise_floor_db,1);
  } else if (task === 3) {
    set('t3-meanf0',  d.f0_mean_hz,        1);
    set('t3-hspl',    d.spl_mean_db,       1);
    set('t3-f0std',   d.f0_std_hz,         1);
    set('t3-cpp',     d.cpp_db,            1);
    set('t3-mspl',    d.spl_mean_db,       1);
    set('t3-splvar',  d.spl_variability_db,1);
    set('t3-prosody', d.prosody_st,        1);
    set('t3-noise',   d.noise_floor_db,    1);
    set('t3-pause',   d.avg_pause_s,       2);
    if (d.speech_rate_wpm !== null && d.speech_rate_wpm !== undefined)
      set('t3-rate', d.speech_rate_wpm, 0);
    if (d.vsa_hz2 !== null && d.vsa_hz2 !== undefined) {
      const el = document.getElementById('t3-vsa');
      if (el) { el.textContent = fmt(d.vsa_hz2 / 1000, 0); updateScale('t3-vsa', d.vsa_hz2 / 1000); }
    }
    if (d.transcript || d.articulation_pct !== undefined) {
      showArticResult(d.articulation_pct ?? null, d.consonant_precision_pct ?? null, d.transcript ?? '');
    }
  }
}

function applyServerResults(task, d, analysis_duration_s) {
  _applyServerMetricsToUI(task, d);
  _fillPendingLogEntry(task, d, analysis_duration_s);
}

function setBtn(active) {
  const btn = document.getElementById('record-btn');
  const lbl = document.getElementById('btn-label');
  btn.className = 'record-btn ' + (active ? 'active' : 'idle');
  lbl.textContent = active ? 'Stop Recording' : 'Start Recording';
}

function setStatus(msg, isErr) {
  const el = document.getElementById('status-text');
  el.textContent = msg;
  el.className   = isErr ? 'error' : '';
}

/* ════════════════════════════════════════════
   AUDIO PROCESS CALLBACK
════════════════════════════════════════════ */
function onAudioProcess(e) {
  const buf = e.inputBuffer.getChannelData(0);

  const rms       = computeRMS(buf);
  const spl       = 20 * Math.log10(rms + 1e-12) + 90;
  const isSilence = rms < 0.008;

  // Collect noise floor from quiet-but-non-zero frames
  if (isSilence && rms > 1e-6) push(noiseHistory, spl, CAP);

  // Speech gate: must be above silence threshold AND ≥SNR_GATE_DB above noise floor
  const noiseEst = noiseHistory.length >= 5 ? arrMedian(noiseHistory) : null;
  const isSpeech = !isSilence && (noiseEst === null || spl > noiseEst + SNR_GATE_DB);

  const pitchResult = isSpeech ? detectPitch(buf) : null;
  const f0          = pitchResult ? pitchResult.f0   : null;
  const hnrPeak     = pitchResult ? pitchResult.peak : null;
  const cpp         = isSpeech ? computeCPP(buf) : null;

  // Voiced gate: speech + pitch detected + sufficient CPP → confirmed human voice
  const isVoiced = isSpeech && f0 !== null && cpp !== null && cpp > CPP_VOICE_THRESH;

  if (isSpeech) {
    push(splHistory, spl, CAP);
    push(rmsHistory, rms, CAP);
    if (cpp !== null) push(cppHistory, cpp, CAP);
  }

  if (isVoiced) {
    push(f0History, f0, CAP);
    push(recentF0,  f0, JITTER_WIN + 1);
    push(recentRms, rms, JITTER_WIN + 1);
    if (prevVoicedF0 !== null) {
      jitterDiffSum  += Math.abs(f0  - prevVoicedF0);  jitterDiffCount++;
      shimmerDiffSum += Math.abs(rms - prevVoicedRms); shimmerDiffCount++;
    }
    prevVoicedF0 = f0; prevVoicedRms = rms;
    if (f0 < minF0) minF0 = f0;
    if (f0 > maxF0) maxF0 = f0;
    if (hnrPeak !== null) {
      lastHnrCorr = hnrPeak;
      if (hnrPeak > 0 && hnrPeak < 1) push(hnrHistory, 10 * Math.log10(hnrPeak / (1 - hnrPeak)), CAP);
    }
    if (mptStart === null) mptStart = Date.now();
  } else {
    if (mptStart !== null) {
      mptTotal += (Date.now() - mptStart) / 1000;
      mptStart = null;
    }
  }

  push(pitchContour, isVoiced ? f0 : 0, CONTOUR_CAP);

  // Pause tracking (speech-gate based)
  if (isSpeech) {
    if (inPause && hasSpeechStarted) {
      pauseDurations.push(pauseFrameCount * 4096 / sampleRate);
      pauseFrameCount = 0; inPause = false;
    }
    hasSpeechStarted = true;
  } else if (hasSpeechStarted) {
    if (!inPause) inPause = true;
    pauseFrameCount++;
  }

  // Formant estimation for VSA (task 3, voiced frames only)
  if (isVoiced && currentTask === 3) {
    const fm = lpcFormants(buf, sampleRate);
    if (fm) push(formantHistory, fm, CAP);
  }

  // Collect raw audio for all tasks (download + task-3 Whisper transcription)
  {
    const maxSamples = sampleRate * 300; // cap at 5 min
    for (let i = 0; i < buf.length; i++) asrAudioBuf.push(buf[i]);
    if (asrAudioBuf.length > maxSamples) asrAudioBuf.splice(0, asrAudioBuf.length - maxSamples);
  }
}

/* ════════════════════════════════════════════
   RENDER LOOP
════════════════════════════════════════════ */
function renderLoop() {
  if (!isRecording) return;
  animId = requestAnimationFrame(renderLoop);

  if      (currentTask === 1) { drawWaveform();         updateTask1(); }
  else if (currentTask === 2) { drawPitchContour(2);    updateTask2(); }
  else if (currentTask === 3) { drawPitchContour(3);    updateTask3(); }
}

/* ════════════════════════════════════════════
   CANVAS — WAVEFORM
════════════════════════════════════════════ */
function drawWaveform() {
  const canvas = document.getElementById('waveform-canvas');
  if (!canvas || !analyserNode) return;
  const ctx = canvas.getContext('2d');
  const W   = canvas.offsetWidth;
  const H   = canvas.offsetHeight;
  canvas.width  = W;
  canvas.height = H;

  const bufLen = analyserNode.frequencyBinCount;
  const data   = new Uint8Array(bufLen);
  analyserNode.getByteTimeDomainData(data);

  ctx.fillStyle = '#1a1d27';
  ctx.fillRect(0, 0, W, H);

  // Centre guide line
  ctx.strokeStyle = '#252836';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();

  // Waveform
  ctx.strokeStyle = '#5cb85c';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  const step = W / bufLen;
  let x = 0;
  for (let i = 0; i < bufLen; i++) {
    const y = ((data[i] / 128) * H) / 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    x += step;
  }
  ctx.stroke();
}

/* ════════════════════════════════════════════
   CANVAS — PITCH CONTOUR
════════════════════════════════════════════ */
function drawPitchContour(taskN, progress = null) {
  const canvas = document.getElementById('pitch-canvas-' + taskN);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W   = canvas.offsetWidth;
  const H   = canvas.offsetHeight;
  canvas.width  = W;
  canvas.height = H;

  // Log-frequency mapping: 60–600 Hz
  const LOG_MIN = Math.log2(60);
  const LOG_MAX = Math.log2(600);
  function freqToY(f) {
    f = Math.max(60, Math.min(600, f));
    return H - ((Math.log2(f) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * H;
  }

  ctx.fillStyle = '#1a1d27';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  const GRIDS = [100, 150, 200, 300, 400];
  ctx.font      = '11px monospace';
  ctx.textAlign = 'left';
  GRIDS.forEach(freq => {
    const y = freqToY(freq);
    ctx.strokeStyle = '#252836';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillStyle = '#555c75';
    ctx.fillText(freq + ' Hz', 4, y - 3);
  });

  if (pitchContour.length < 2) return;

  const n     = pitchContour.length;
  const stepW = W / (n - 1);                              // stretch contour to full canvas width
  const maxI  = progress !== null ? Math.floor(progress * (n - 1)) : n - 1;

  // Draw voiced segments up to maxI
  ctx.strokeStyle = '#5cb85c';
  ctx.lineWidth   = 2;
  ctx.beginPath();
  let penDown = false;
  for (let i = 0; i <= maxI; i++) {
    const f = pitchContour[i];
    const x = i * stepW;
    if (f <= 0) { penDown = false; continue; }
    const y = freqToY(f);
    if (!penDown) { ctx.moveTo(x, y); penDown = true; }
    else            ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Dot + label at last drawn voiced point
  for (let i = maxI; i >= 0; i--) {
    const f = pitchContour[i];
    if (f <= 0) continue;
    const x = i * stepW;
    const y = freqToY(f);
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#5cb85c';
    ctx.fill();
    const label = Math.round(f) + ' Hz';
    ctx.fillStyle   = '#e0e4ef';
    ctx.font        = '12px monospace';
    ctx.textAlign   = 'left';
    const lx = Math.min(x + 9, W - label.length * 7.5);
    ctx.fillText(label, lx, y - 8);
    break;
  }

  // Playback cursor
  if (progress !== null) {
    const cx = progress * W;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth   = 1.5;
    ctx.stroke();
  }
}


/* ════════════════════════════════════════════
   DOM UPDATE — TASK 1
════════════════════════════════════════════ */
function updateTask1() {
  const recent30 = f0History.slice(-30);
  const avgF0    = recent30.length ? arrMean(recent30) : null;
  document.getElementById('t1-f0').textContent = avgF0 ? fmt(avgF0, 1) : '\u2014';
  updateScale('t1-f0', avgF0);
  updateHistogram('t1-f0', f0History, avgF0);

  const lastSpl = splHistory.length ? splHistory[splHistory.length-1] : null;
  const meanSpl1 = splHistory.length ? arrMean(splHistory) : null;
  document.getElementById('t1-spl').textContent = lastSpl !== null ? fmt(lastSpl, 1) : '\u2014';
  updateScale('t1-spl', lastSpl);
  updateHistogram('t1-spl', splHistory, meanSpl1);

  const mpt = mptStart !== null
    ? mptTotal + (Date.now() - mptStart) / 1000
    : mptTotal;
  document.getElementById('t1-mpt').textContent = fmt(mpt, 1);
  updateScale('t1-mpt', mpt);

  const jitter = computeJitter();
  document.getElementById('t1-jitter').textContent  = fmt(jitter, 2);
  updateScale('t1-jitter', parseFloat(jitter));

  const shimmer = computeShimmer();
  document.getElementById('t1-shimmer').textContent = fmt(shimmer, 2);
  updateScale('t1-shimmer', parseFloat(shimmer));

  const avgCPP = cppHistory.length ? arrMean(cppHistory.slice(-20)) : null;
  document.getElementById('t1-cpp').textContent = avgCPP !== null ? fmt(avgCPP, 1) : '\u2014';
  updateScale('t1-cpp', avgCPP);
  updateHistogram('t1-cpp', cppHistory, avgCPP);

  const hnrVal = hnrHistory.length ? arrMean(hnrHistory) : null;
  document.getElementById('t1-hnr').textContent = hnrVal !== null ? fmt(hnrVal, 1) : '\u2014';
  updateScale('t1-hnr', hnrVal);
  updateHistogram('t1-hnr', hnrHistory, hnrVal);

  const noise1 = computeNoiseFloor();
  document.getElementById('t1-noise').textContent = noise1 !== null ? fmt(noise1, 1) : '\u2014';
  updateScale('t1-noise', noise1);
  updateHistogram('t1-noise', noiseHistory, noise1);
}

/* ════════════════════════════════════════════
   DOM UPDATE — TASK 2
════════════════════════════════════════════ */
function updateTask2() {
  const hasMin = minF0 < Infinity;
  const hasMax = maxF0 > -Infinity;
  document.getElementById('t2-minf0').textContent = hasMin ? fmt(minF0, 1) : '\u2014';
  updateScale('t2-minf0', hasMin ? minF0 : null);
  document.getElementById('t2-maxf0').textContent = hasMax ? fmt(maxF0, 1) : '\u2014';
  updateScale('t2-maxf0', hasMax ? maxF0 : null);
  const rng = computeProsody();
  document.getElementById('t2-range').textContent  = rng !== null ? fmt(rng, 1) : '\u2014';
  updateScale('t2-range', rng);

  const noise2 = computeNoiseFloor();
  document.getElementById('t2-noise').textContent = noise2 !== null ? fmt(noise2, 1) : '\u2014';
  updateScale('t2-noise', noise2);
  updateHistogram('t2-noise', noiseHistory, noise2);
}

/* ════════════════════════════════════════════
   DOM UPDATE — TASK 3
════════════════════════════════════════════ */
function updateTask3() {
  const voiced = f0History.filter(v => v > 0);

  const avgF0 = voiced.length ? arrMean(voiced) : null;
  document.getElementById('t3-meanf0').textContent = avgF0 ? fmt(avgF0, 1) : '\u2014';
  updateScale('t3-meanf0', avgF0);
  updateHistogram('t3-meanf0', voiced, avgF0);

  const hspl = splHistory.length > 5 ? arrMedian(splHistory) : null;
  document.getElementById('t3-hspl').textContent = hspl !== null ? fmt(hspl, 1) : '\u2014';
  updateScale('t3-hspl', hspl);
  updateHistogram('t3-hspl', splHistory, hspl);

  const f0sd = voiced.length > 2 ? arrStd(voiced) : null;
  document.getElementById('t3-f0std').textContent = f0sd !== null ? fmt(f0sd, 1) : '\u2014';
  updateScale('t3-f0std', f0sd);

  const avgCPP = cppHistory.length ? arrMean(cppHistory.slice(-20)) : null;
  document.getElementById('t3-cpp').textContent = avgCPP !== null ? fmt(avgCPP, 1) : '\u2014';
  updateScale('t3-cpp', avgCPP);
  updateHistogram('t3-cpp', cppHistory, avgCPP);

  const mSpl = splHistory.length ? arrMean(splHistory) : null;
  document.getElementById('t3-mspl').textContent = mSpl !== null ? fmt(mSpl, 1) : '\u2014';
  updateScale('t3-mspl', mSpl);
  updateHistogram('t3-mspl', splHistory, mSpl);

  const splVar = computeSPLVariability();
  document.getElementById('t3-splvar').textContent = splVar !== null ? fmt(splVar, 1) : '\u2014';
  updateScale('t3-splvar', splVar);

  const rate = computeSpeechRate();
  const ratePM = rate !== null ? rate * 60 : null;
  document.getElementById('t3-rate').textContent = ratePM !== null ? fmt(ratePM, 0) : '\u2014';
  updateScale('t3-rate', ratePM);

  const prosody = computeProsody();
  document.getElementById('t3-prosody').textContent = prosody !== null ? fmt(prosody, 1) : '\u2014';
  updateScale('t3-prosody', prosody);

  const noise3 = computeNoiseFloor();
  document.getElementById('t3-noise').textContent = noise3 !== null ? fmt(noise3, 1) : '\u2014';
  updateScale('t3-noise', noise3);
  updateHistogram('t3-noise', noiseHistory, noise3);

  const pause = computeAvgPause();
  document.getElementById('t3-pause').textContent = pause !== null ? fmt(pause, 2) : '\u2014';
  updateScale('t3-pause', pause);

  const vsa = computeVSA();
  document.getElementById('t3-vsa').textContent = vsa !== null ? fmt(vsa / 1000, 0) : '\u2014';
  updateScale('t3-vsa', vsa !== null ? vsa / 1000 : null);
}

/* ════════════════════════════════════════════
   PASSAGE SETS
════════════════════════════════════════════ */
const RAINBOW_REF = 'when the sunlight strikes raindrops in the air they act as a prism and form a rainbow the rainbow is a division of white light into many beautiful colors these take the shape of a long round arch with its path high above and its two ends apparently beyond the horizon there is according to legend a boiling pot of gold at one end people look but no one ever finds it when a man looks for something beyond his reach his friends say he is looking for the pot of gold at the end of the rainbow';

const PASSAGE_SETS = {
  word: [
    'Sunshine','Harmony','Laughter','Kindness','Gratitude','Courage','Wisdom',
    'Serenity','Radiance','Freedom','Beauty','Delight','Warmth','Blossom',
    'Clarity','Balance','Flourish','Celebrate','Cherish','Sparkle','Triumph',
    'Vibrant','Tranquil','Graceful','Peaceful','Nurture','Uplift','Wonder',
    'Inspire','Joyful'
  ],
  sentence: [
    'The morning light filled the room with golden warmth and quiet hope.',
    'Every act of kindness ripples outward and touches more lives than we know.',
    'Spring arrived with a burst of color and the sweet smell of renewal.',
    'With patience and care, every seed grows into something beautiful and strong.',
    'The warm breeze carried the scent of blossoms across the sunlit garden.',
    'Gratitude opens the heart and fills even the quietest moments with meaning.',
    'Joy is contagious, spreading gently from one smiling face to another.',
    'The river flowed steadily, reflecting the blue sky and the bright afternoon sun.',
    'He greeted each day with enthusiasm and a heart full of possibilities.',
    'Music filled the air, and the whole room swayed together in harmony.',
    'The old tree stood tall, its branches wide and welcoming like open arms.',
    'A kind word at the right moment can change the entire course of someone\'s day.',
    'Courage grows stronger each time we face our challenges with an open heart.',
    'Laughter echoed through the house, warm and bright as a summer afternoon.',
    'She found beauty in every small thing — a dewdrop, a birdsong, a friendly wave.',
    'Together, they built something lasting out of trust, laughter, and shared dreams.',
    'The children laughed freely as they chased bright butterflies through the meadow.',
    'A garden in full bloom is one of the most cheerful sights in the world.',
    'They sat together on the porch, watching the stars appear one by one.',
    'Every new morning brings with it the chance to begin something wonderful.'
  ],
  paragraph: [
    'When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.',
    'Autumn is a season of warmth and color, when the world puts on its finest display. The leaves turn brilliant shades of gold, crimson, and amber, drifting softly to the ground with each gentle breeze. The air smells of wood smoke and ripe apples, and the afternoons grow sweet and still. Families gather to share warm meals, and neighbors wave to each other across frost-touched lawns. There is a quiet joy in autumn, a sense of abundance and gratitude for everything the year has brought.',
    'True friendship is one of life\'s greatest gifts. A good friend listens without judgment, celebrates your victories, and stands beside you during difficult times. Friendships grow through shared laughter, honest conversations, and the small kindnesses that become the foundation of trust. Over the years, friends shape who we are and remind us of who we want to be. The bonds we form with the people we love are among the most enduring and meaningful parts of a well-lived life.',
    'There is something magical about early morning, when the world is still and the day holds nothing but possibility. The sky softens from deep blue to pale gold as the sun rises above the horizon. Birds begin their songs one by one, filling the air with cheerful sound. In these early hours, everything feels fresh and full of promise, as though the world has been made new overnight. It is a perfect moment to breathe deeply, set a quiet intention, and step forward with a grateful heart.',
    'The ocean has a way of putting things in perspective. Standing at the water\'s edge, watching the waves roll in and out in their ancient rhythm, the mind grows calm and the heart feels lighter. The sea breeze carries salt and freedom, and the sound of the surf is one of the most soothing sounds in the world. The ocean reminds us how wide and wonderful the world truly is, and how much beauty there is to explore in a lifetime filled with curiosity and courage.',
    'A well-tended garden is a place of peace and constant wonder. Seeds pressed into dark soil become green shoots within days, then blossoms of every shape and color. Bees move unhurriedly from flower to flower, and butterflies drift on warm air. The gardener learns patience, attention, and deep respect for the rhythms of nature. There is great satisfaction in tending something living, watching it grow, and sharing its harvest with others. A garden teaches us that with care and time, beautiful things always flourish.',
    'Music has a remarkable power to lift the spirit and bring people together. A familiar melody can carry us instantly back to a happy memory, filling us with warmth all at once. Rhythm makes us want to move, and harmony reminds us that different voices can create something far more beautiful together than any single one could alone. Whether played softly in a quiet room or performed before a joyful crowd, music speaks a language that everyone understands, crossing every border and bridging every distance.',
    'Laughter is one of the most powerful forces in human life. It breaks tension, deepens friendship, and reminds us not to take everything too seriously. A genuine laugh shared between two people creates a bond that words alone cannot form. Children laugh freely and often, finding delight in the simplest things, and that natural joy is something worth holding onto through every season of life. When we laugh together, we are at our most human, our most connected, and our most alive.',
    'Kindness costs nothing and yet its value is immeasurable. A warm smile offered to a stranger, a helping hand extended without expectation, a few encouraging words spoken at the right moment — each of these small acts can change someone\'s entire day. Kindness spreads in ways we cannot always see or measure. It creates ripples that travel far beyond their starting point, touching lives in ways we may never know. Choosing kindness, day after day, is one of the finest things a person can do.',
    'The stars have guided travelers, inspired poets, and filled children with wonder for thousands of years. On a clear night, far from the glow of city lights, the sky becomes a dazzling map of light and possibility. Each star is a distant sun, many of them ancient beyond imagination, yet their light still reaches us here on earth, still bright and full of meaning. Looking up at the night sky, it is easy to feel both small and connected — part of something vast, beautiful, and endlessly alive.'
  ]
};

let passageType = 'paragraph';
let activePassageRef = RAINBOW_REF;

function pickRandom(arr, exclude) {
  const pool = arr.length > 1 ? arr.filter(x => x !== exclude) : arr;
  return pool[Math.floor(Math.random() * pool.length)];
}

function setPassageType(type) {
  passageType = type;
  ['word','sentence','paragraph'].forEach(t => {
    const b = document.getElementById('ptype-' + t);
    if (b) b.classList.toggle('active', t === type);
  });
  shufflePassage();
}

function shufflePassage() {
  if (passageEditing) commitPassageEdit();
  const items = PASSAGE_SETS[passageType];
  const chosen = pickRandom(items, activePassageRef);
  activePassageRef = chosen;

  const box     = document.getElementById('passage-box');
  const heading = document.getElementById('passage-heading');
  const textEl  = document.getElementById('passage-text');
  if (!box || !heading || !textEl) return;

  const labels = { word: 'Word', sentence: 'Sentence', paragraph: 'Paragraph' };
  heading.textContent = labels[passageType] + ' — read aloud';
  textEl.textContent  = chosen;

  box.classList.toggle('mode-word', passageType === 'word');
}

let passageEditing = false;

function startPassageEdit() {
  if (passageEditing) return;
  const textEl  = document.getElementById('passage-text');
  if (!textEl) return;
  passageEditing = true;

  const ta = document.createElement('textarea');
  ta.id        = 'passage-textarea';
  ta.className = 'passage-edit-area';
  ta.value     = textEl.textContent;
  ta.rows      = passageType === 'word' ? 2 : passageType === 'sentence' ? 3 : 6;
  ta.addEventListener('blur', commitPassageEdit);
  textEl.replaceWith(ta);
  ta.focus();

  const box = document.getElementById('passage-box');
  if (box) box.classList.remove('mode-word');

  const _editBtn = document.getElementById('passage-edit-btn');
  if (_editBtn) _editBtn.style.display = 'none';
  document.getElementById('passage-done-btn').style.display = '';
}

function commitPassageEdit() {
  if (!passageEditing) return;
  passageEditing = false;
  const ta  = document.getElementById('passage-textarea');
  const val = ta ? ta.value.trim() : activePassageRef;
  activePassageRef = val || activePassageRef;

  const span = document.createElement('span');
  span.id          = 'passage-text';
  span.textContent = activePassageRef;
  span.onclick     = startPassageEdit;
  if (ta) ta.replaceWith(span);

  const box = document.getElementById('passage-box');
  if (box) box.classList.toggle('mode-word', passageType === 'word');

  const _editBtn2 = document.getElementById('passage-edit-btn');
  if (_editBtn2) _editBtn2.style.display = '';
  document.getElementById('passage-done-btn').style.display = 'none';
}


function showArticResult(score, cp, transcript) {
  const articEl = document.getElementById('t3-artic');
  if (articEl) { articEl.textContent = score !== null ? fmt(score, 1) : '\u2014'; updateScale('t3-artic', score); }
  const cpEl = document.getElementById('t3-consonants');
  if (cpEl) { cpEl.textContent = cp !== null ? fmt(cp, 1) : '\u2014'; updateScale('t3-consonants', cp); }
  const box = document.getElementById('transcript-box');
  const txt = document.getElementById('transcript-text');
  if (box && txt && transcript) {
    box.style.display = 'block';
    txt.textContent   = transcript.trim();
  }
}



/* ════════════════════════════════════════════
   AUDIO DOWNLOAD
════════════════════════════════════════════ */
function encodeWAV(samples, sr) {
  const len    = samples.length;
  const buf    = new ArrayBuffer(44 + len * 2);
  const view   = new DataView(buf);
  const str    = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  str(0,  'RIFF');
  view.setUint32(4,  36 + len * 2, true);
  str(8,  'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16,       true);
  view.setUint16(20, 1,        true);
  view.setUint16(22, 1,        true);
  view.setUint32(24, sr,       true);
  view.setUint32(28, sr * 2,   true);
  view.setUint16(32, 2,        true);
  view.setUint16(34, 16,       true);
  str(36, 'data');
  view.setUint32(40, len * 2,  true);
  let off = 44;
  for (let i = 0; i < len; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    off += 2;
  }
  return buf;
}

function downloadAudioWAV(taskN) {
  if (!asrAudioBuf.length) return;
  const wav  = encodeWAV(new Float32Array(asrAudioBuf), asrSampleRate);
  const blob = new Blob([wav], { type: 'audio/wav' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  const fname = _defaultFilename(taskN);
  a.download  = fname;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  currentFilename = fname;
  updateLastLogFilename(taskN, fname);
}

async function downloadRecording() {
  if (!asrAudioBuf.length) return;
  if (passageEditing) commitPassageEdit();

  const wav         = encodeWAV(new Float32Array(asrAudioBuf), asrSampleRate);
  const textEl      = document.getElementById('passage-text');
  const passageText = textEl ? textEl.textContent.trim() : activePassageRef;

  const zip = new JSZip();
  zip.file('recording.wav', wav);
  zip.file('passage.txt',   passageText);

  const blob = await zip.generateAsync({ type: 'blob' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  const fname = _defaultFilename(3);
  a.download  = fname;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  currentFilename = fname;
  updateLastLogFilename(3, fname);
}

/* ════════════════════════════════════════════
   AUDIO UPLOAD
════════════════════════════════════════════ */
async function handleAudioUpload(file) {
  if (!file) return;
  currentFilename = file.name;
  if (passageEditing) commitPassageEdit();
  setStatus('Reading ' + file.name + '\u2026', false);
  try {
    const zipData = await file.arrayBuffer();
    const zip     = await JSZip.loadAsync(zipData);

    const txtFile = zip.file('passage.txt');
    if (!txtFile) throw new Error('passage.txt not found in ZIP');
    const passageText = await txtFile.async('string');
    activePassageRef  = passageText.trim();

    const textEl = document.getElementById('passage-text');
    if (textEl) textEl.textContent = activePassageRef;

    const wavFile = zip.file('recording.wav');
    if (!wavFile) throw new Error('recording.wav not found in ZIP');
    const wavBuf   = await wavFile.async('arraybuffer');

    setStatus('Decoding audio\u2026', false);

    const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await tempCtx.decodeAudioData(wavBuf);
    await tempCtx.close();

    const mono = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += ch[i];
    }
    for (let i = 0; i < mono.length; i++) mono[i] /= decoded.numberOfChannels;

    asrAudioBuf   = Array.from(mono);
    asrSampleRate = decoded.sampleRate;

    await processUploadedAudio();
  } catch (err) {
    setStatus('Upload error: ' + (err.message || String(err)), true);
  }
}

async function handleAudioUploadRaw(file) {
  if (!file) return;
  currentFilename = file.name;
  setStatus('Decoding ' + file.name + '\u2026', false);
  try {
    const arrayBuf = await file.arrayBuffer();
    const tempCtx  = new (window.AudioContext || window.webkitAudioContext)();
    const decoded  = await tempCtx.decodeAudioData(arrayBuf);
    await tempCtx.close();

    const mono = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += ch[i];
    }
    for (let i = 0; i < mono.length; i++) mono[i] /= decoded.numberOfChannels;

    asrAudioBuf   = Array.from(mono);
    asrSampleRate = decoded.sampleRate;

    await processUploadedAudio();
  } catch (err) {
    setStatus('Decode error: ' + (err.message || String(err)), true);
  }
}

async function processUploadedAudio() {
  sampleRate = asrSampleRate;

  const savedBuf      = asrAudioBuf;
  const savedSR       = asrSampleRate;
  const savedTask     = currentTask;
  const savedFilename = currentFilename;
  resetSession();
  asrAudioBuf     = savedBuf;
  asrSampleRate   = savedSR;
  currentFilename = savedFilename;
  enableDownloadBtn(savedTask);

  sessionStart = Date.now() - (asrAudioBuf.length / asrSampleRate) * 1000;

  const samples     = new Float32Array(asrAudioBuf);
  const FRAME       = 4096;
  const totalFrames = Math.floor(samples.length / FRAME);
  const frameDur    = FRAME / asrSampleRate;
  const decimation  = Math.max(1, Math.floor(totalFrames / CONTOUR_CAP));
  const BATCH       = 40;
  let voicedFrames  = 0;

  const noiseScan = [];
  for (let i = 0; i < totalFrames; i++) {
    const r = computeRMS(samples.subarray(i * FRAME, (i + 1) * FRAME));
    if (r < 0.008 && r > 1e-6) noiseScan.push(20 * Math.log10(r + 1e-12) + 90);
  }
  const batchNoiseFloor = noiseScan.length >= 5 ? arrMedian(noiseScan) : null;
  if (batchNoiseFloor !== null) push(noiseHistory, batchNoiseFloor, CAP);

  for (let i = 0; i < totalFrames; i++) {
    const frame      = samples.subarray(i * FRAME, (i + 1) * FRAME);
    const rms        = computeRMS(frame);
    const spl        = 20 * Math.log10(rms + 1e-12) + 90;
    const isSilence  = rms < 0.008;

    const isSpeech   = !isSilence && (batchNoiseFloor === null || spl > batchNoiseFloor + SNR_GATE_DB);

    const pitchResult = isSpeech ? detectPitch(frame) : null;
    const f0         = pitchResult ? pitchResult.f0   : null;
    const hnrPeak    = pitchResult ? pitchResult.peak : null;
    const cpp        = isSpeech   ? computeCPP(frame) : null;

    const isVoiced   = isSpeech && f0 !== null && cpp !== null && cpp > CPP_VOICE_THRESH;

    if (isSpeech) {
      push(splHistory, spl, CAP);
      push(rmsHistory, rms, CAP);
      if (cpp !== null) push(cppHistory, cpp, CAP);
    }
    if (isVoiced) {
      push(f0History, f0, CAP);
      push(recentF0,  f0, JITTER_WIN + 1);
      push(recentRms, rms, JITTER_WIN + 1);
      if (prevVoicedF0 !== null) {
        jitterDiffSum  += Math.abs(f0  - prevVoicedF0);  jitterDiffCount++;
        shimmerDiffSum += Math.abs(rms - prevVoicedRms); shimmerDiffCount++;
      }
      prevVoicedF0 = f0; prevVoicedRms = rms;
      if (f0 < minF0) minF0 = f0;
      if (f0 > maxF0) maxF0 = f0;
      if (hnrPeak !== null) {
        lastHnrCorr = hnrPeak;
        if (hnrPeak > 0 && hnrPeak < 1) push(hnrHistory, 10 * Math.log10(hnrPeak / (1 - hnrPeak)), CAP);
      }
      voicedFrames++;
    }
    if (i % decimation === 0) {
      push(pitchContour, isVoiced ? f0 : 0, CONTOUR_CAP);
    }

    if (isSpeech) {
      if (inPause && hasSpeechStarted) {
        pauseDurations.push(pauseFrameCount * FRAME / asrSampleRate);
        pauseFrameCount = 0; inPause = false;
      }
      hasSpeechStarted = true;
    } else if (hasSpeechStarted) {
      if (!inPause) inPause = true;
      pauseFrameCount++;
    }

    if (isVoiced && savedTask === 3) {
      const fm = lpcFormants(frame, asrSampleRate);
      if (fm) push(formantHistory, fm, CAP);
    }

    if (i % BATCH === BATCH - 1) {
      const pct = Math.round((i / totalFrames) * 100);
      setStatus(`Analyzing\u2026 ${pct}%`, false);
      if      (currentTask === 1) updateTask1();
      else if (currentTask === 2) { updateTask2(); drawPitchContour(2); }
      else if (currentTask === 3) { updateTask3(); drawPitchContour(3); }
      await new Promise(r => setTimeout(r, 0));
    }
  }

  mptTotal = voicedFrames * frameDur;
  mptStart = null;

  setStatus('Done', false);

  if      (currentTask === 1) updateTask1();
  else if (currentTask === 2) { updateTask2(); drawPitchContour(2); }
  else if (currentTask === 3) { updateTask3(); drawPitchContour(3); }

  _enablePlaybackBtn();
  if (currentTask === 1) drawFullWaveform(null);

  if (_replayMode) { _replayMode = false; } else { addPendingLogEntry(savedTask); sendToServer(savedTask); }
}

/* ════════════════════════════════════════════
   MULTI-FILE UPLOAD QUEUE
════════════════════════════════════════════ */
let _uploadQueue = [];
let _uploadBusy  = false;
let _replayMode  = false;

// Draw the full recorded buffer as a static waveform (used after recording + during playback)
function drawFullWaveform(progress) {
  const canvas = document.getElementById('waveform-canvas');
  if (!canvas || !asrAudioBuf.length) return;
  const ctx = canvas.getContext('2d');
  const W   = canvas.offsetWidth;
  const H   = canvas.offsetHeight;
  canvas.width  = W;
  canvas.height = H;

  ctx.fillStyle = '#1a1d27';
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = '#252836';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

  const samples = asrAudioBuf;
  const maxX    = progress !== null ? Math.floor(progress * W) : W;
  ctx.strokeStyle = '#5cb85c';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  for (let x = 0; x < maxX; x++) {
    const i = Math.floor(x / W * samples.length);
    const y = (0.5 - samples[i] * 0.4) * H;
    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (progress !== null) {
    const cx = progress * W;
    ctx.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx.lineWidth   = 1.5;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  }
}

function enqueueFiles(fileList, handler) {
  for (const f of fileList) _uploadQueue.push({ file: f, handler });
  _drainUploadQueue();
}

async function _drainUploadQueue() {
  if (_uploadBusy || _uploadQueue.length === 0) return;
  _uploadBusy = true;
  while (_uploadQueue.length > 0) {
    const { file, handler } = _uploadQueue.shift();
    await handler(file);
  }
  _uploadBusy = false;
}

/* ════════════════════════════════════════════
   DRAG-AND-DROP
════════════════════════════════════════════ */
function setupDragDrop() {
  const task = CURRENT_TASK;
  let dragDepth = 0;
  document.addEventListener('dragover', e => { e.preventDefault(); });
  document.addEventListener('dragenter', e => {
    e.preventDefault();
    dragDepth++;
    document.body.classList.add('drag-over');
  });
  document.addEventListener('dragleave', () => {
    dragDepth--;
    if (dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('drag-over'); }
  });
  document.addEventListener('drop', e => {
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove('drag-over');
    const files = [...e.dataTransfer.files];
    if (task === 3) {
      const zips = files.filter(f => f.name.toLowerCase().endsWith('.zip') || f.type === 'application/zip');
      if (zips.length) enqueueFiles(zips, handleAudioUpload);
    } else {
      const audio = files.filter(f => f.type.startsWith('audio/'));
      if (audio.length) enqueueFiles(audio, handleAudioUploadRaw);
    }
  });
}

/* ════════════════════════════════════════════
   ANALYSIS LOG — IndexedDB
════════════════════════════════════════════ */
const LOG_METRICS = {
  1: [
    { key: 'f0_mean_hz',          label: 'F0μ<br>(Hz)',   dec: 1, title: 'Mean fundamental frequency (Hz)' },
    { key: 'spl_mean_db',         label: 'SPL<br>(dB)',   dec: 1, title: 'Mean sound pressure level — loudness (dB)' },
    { key: 'mpt_s',               label: 'MPT<br>(s)',    dec: 1, title: 'Maximum phonation time (s)' },
    { key: 'jitter_pct',          label: 'Jitr<br>(%)',   dec: 2, title: 'Jitter — cycle-to-cycle frequency variability (%)' },
    { key: 'shimmer_pct',         label: 'Shim<br>(%)',   dec: 2, title: 'Shimmer — cycle-to-cycle amplitude variability (%)' },
    { key: 'cpp_db',              label: 'CPP<br>(dB)',   dec: 1, title: 'Cepstral peak prominence — overall voice quality (dB)' },
    { key: 'hnr_db',              label: 'HNR<br>(dB)',   dec: 1, title: 'Harmonics-to-noise ratio (dB)' },
    { key: 'noise_floor_db',      label: 'Nois<br>(dB)',  dec: 1, title: 'Ambient noise floor (dB)' },
    { key: 'duration_s',          label: 'Dur<br>(s)',    dec: 1, title: 'Recording duration (s)' },
    { key: 'analysis_duration_s', label: 'Anls<br>(s)',   dec: 1, title: 'Server analysis duration (s)' },
  ],
  2: [
    { key: 'f0_min_hz',           label: 'F0↓<br>(Hz)',   dec: 1, title: 'Lowest achievable pitch (Hz)' },
    { key: 'f0_max_hz',           label: 'F0↑<br>(Hz)',   dec: 1, title: 'Highest achievable pitch (Hz)' },
    { key: 'f0_range_st',         label: 'F0R<br>(ST)',   dec: 1, title: 'Pitch range — highest minus lowest (semitones)' },
    { key: 'noise_floor_db',      label: 'Nois<br>(dB)',  dec: 1, title: 'Ambient noise floor (dB)' },
    { key: 'duration_s',          label: 'Dur<br>(s)',    dec: 1, title: 'Recording duration (s)' },
    { key: 'analysis_duration_s', label: 'Anls<br>(s)',   dec: 1, title: 'Server analysis duration (s)' },
  ],
  3: [
    { key: 'f0_mean_hz',              label: 'F0μ<br>(Hz)',   dec: 1, title: 'Mean fundamental frequency during speech (Hz)' },
    { key: 'f0_std_hz',               label: 'F0σ<br>(Hz)',   dec: 1, title: 'Pitch standard deviation — variability (Hz)' },
    { key: 'cpp_db',                  label: 'CPP<br>(dB)',   dec: 1, title: 'Cepstral peak prominence — voice quality (dB)' },
    { key: 'spl_mean_db',             label: 'SPL<br>(dB)',   dec: 1, title: 'Mean sound pressure level (dB)' },
    { key: 'spl_variability_db',      label: 'SPLV<br>(dB)',  dec: 1, title: 'SPL variability — loudness modulation (dB)' },
    { key: 'speech_rate_wpm',         label: 'Rate<br>(syl)', dec: 0, title: 'Speech rate (syllables per minute)' },
    { key: 'prosody_st',              label: 'Pros<br>(ST)',  dec: 1, title: 'Prosody — pitch range used during speech (semitones)' },
    { key: 'noise_floor_db',          label: 'Nois<br>(dB)',  dec: 1, title: 'Ambient noise floor (dB)' },
    { key: 'avg_pause_s',             label: 'Paus<br>(s)',   dec: 2, title: 'Mean inter-speech pause length (s)' },
    { key: 'vsa_khz2',                label: 'VSA<br>(kHz²)', dec: 0, title: 'Vowel space area — shrinks with vowel centralisation (kHz²)' },
    { key: 'articulation_pct',        label: 'Art<br>(%)',    dec: 1, title: 'Word accuracy vs. passage (%)' },
    { key: 'consonant_precision_pct', label: 'Con<br>(%)',    dec: 1, title: 'Consonant precision vs. passage (%)' },
    { key: 'duration_s',              label: 'Dur<br>(s)',    dec: 1, title: 'Recording duration (s)' },
    { key: 'analysis_duration_s',     label: 'Anls<br>(s)',   dec: 1, title: 'Server analysis duration (s)' },
  ],
};


function _escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}



let _currentUsername = '';

function initUsername() {
  fetch('/me').then(r => r.json()).then(data => {
    const name = data.username || '';
    _currentUsername = name;
    if (!name) return;
    const el = document.getElementById('nav-user');
    if (el) el.textContent = 'Hi, ' + name;
  }).catch(() => {});
}

// Initialise on load
resetSession();
setBtn(false);
setStatus('Ready', false);
initLogDB().then(() => { renderLogTable(CURRENT_TASK); });
setupDragDrop();
initScales();
initAudioPlayback();
if (CURRENT_TASK === 3) setPassageType('paragraph');
fetchPublished(CURRENT_TASK);
initUsername();
