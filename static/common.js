'use strict';

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
  const names = { 1: 'phonation', 2: 'pitch-glides', 3: 'reading' };
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
    const resp = await fetch(`${SERVER_URL}/analyze/task${task}`, {
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
   DSP — FFT (radix-2, in-place)
════════════════════════════════════════════ */
function fftInPlace(re, im) {
  const N = re.length;
  // Bit-reversal
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  // Butterfly
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const ang  = -Math.PI / half;          // -2pi/len
    const wr   = Math.cos(ang);
    const wi   = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const ur = re[i+k],      ui = im[i+k];
        const vr = re[i+k+half]*cr - im[i+k+half]*ci;
        const vi = re[i+k+half]*ci + im[i+k+half]*cr;
        re[i+k]      = ur + vr;  im[i+k]      = ui + vi;
        re[i+k+half] = ur - vr;  im[i+k+half] = ui - vi;
        const ncr = cr*wr - ci*wi;
        ci = cr*wi + ci*wr;
        cr = ncr;
      }
    }
  }
}

function ifftInPlace(re, im) {
  const N = re.length;
  for (let i = 0; i < N; i++) im[i] = -im[i];
  fftInPlace(re, im);
  for (let i = 0; i < N; i++) { re[i] /= N; im[i] = -im[i] / N; }
}

/* ════════════════════════════════════════════
   DSP — PITCH (normalised autocorrelation)
════════════════════════════════════════════ */
function detectPitch(buf) {
  // YIN pitch detection (de Cheveigné & Kawahara 2002).
  const N       = 2048;
  const W       = N >> 1;                                 // 1024-sample analysis window
  const MIN_LAG = Math.max(2, Math.floor(sampleRate / 600)); // ~73 @ 44100 Hz
  const MAX_LAG = Math.min(Math.floor(sampleRate / 75), W - 1); // ~588 @ 44100 Hz
  const THRESH  = 0.10;   // standard YIN threshold

  // Energy check
  let e = 0;
  for (let i = 0; i < W; i++) e += buf[i] * buf[i];
  if (e / W < 1e-8) return null;

  // Step 1 — Difference function d(τ) = Σ (x[j] – x[j+τ])²
  const df = new Float32Array(MAX_LAG + 1);
  for (let tau = 1; tau <= MAX_LAG; tau++) {
    let s = 0;
    for (let j = 0; j < W; j++) {
      const d = buf[j] - buf[j + tau];
      s += d * d;
    }
    df[tau] = s;
  }

  // Step 2 — Cumulative mean normalised difference function (CMNDF)
  const cmndf = new Float32Array(MAX_LAG + 1);
  cmndf[0] = 1;
  let runSum = 0;
  for (let tau = 1; tau <= MAX_LAG; tau++) {
    runSum += df[tau];
    cmndf[tau] = runSum > 0 ? (df[tau] * tau) / runSum : 1;
  }

  // Step 3 — Absolute threshold
  let tau = -1;
  for (let t = MIN_LAG; t <= MAX_LAG; t++) {
    if (cmndf[t] < THRESH) {
      while (t + 1 <= MAX_LAG && cmndf[t + 1] < cmndf[t]) t++;
      tau = t;
      break;
    }
  }
  // Fallback: global minimum
  if (tau === -1) {
    let minVal = Infinity;
    for (let t = MIN_LAG; t <= MAX_LAG; t++) {
      if (cmndf[t] < minVal) { minVal = cmndf[t]; tau = t; }
    }
    if (minVal > 0.5) return null;
  }

  // Step 4 — Parabolic interpolation
  let refTau = tau;
  if (tau > 1 && tau < MAX_LAG) {
    const s0 = cmndf[tau - 1], s1 = cmndf[tau], s2 = cmndf[tau + 1];
    const denom = 2 * (2 * s1 - s2 - s0);
    if (Math.abs(denom) > 1e-10) refTau = tau + (s0 - s2) / denom;
  }

  const f0 = sampleRate / refTau;
  if (f0 < 75 || f0 > 600) return null;

  const confidence = Math.max(0, Math.min(1, 1 - cmndf[tau]));
  return { f0, peak: confidence };
}

/* ════════════════════════════════════════════
   DSP — RMS / SPL
════════════════════════════════════════════ */
function computeRMS(buf) {
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / buf.length);
}

/* ════════════════════════════════════════════
   DSP — CPP (cepstral peak prominence)
════════════════════════════════════════════ */
function computeCPP(buf) {
  const NFFT = 512;
  const re   = new Float32Array(NFFT);
  const im   = new Float32Array(NFFT);

  // Hann window + frame
  for (let i = 0; i < NFFT; i++) {
    const w = 0.5 * (1 - Math.cos(2 * Math.PI * i / (NFFT - 1)));
    re[i] = (buf[i] || 0) * w;
    im[i] = 0;
  }

  fftInPlace(re, im);

  // Log power spectrum
  const logSpec = new Float32Array(NFFT);
  for (let k = 0; k < NFFT; k++) {
    logSpec[k] = Math.log(re[k]*re[k] + im[k]*im[k] + 1e-12);
  }

  // IFFT(logSpec) -> cepstrum
  const cRe = logSpec.slice();
  const cIm  = new Float32Array(NFFT);
  ifftInPlace(cRe, cIm);

  const qMin = Math.max(1, Math.floor(sampleRate / 600));
  const qMax = Math.min(Math.floor(sampleRate / 60), (NFFT >> 1) - 1);
  if (qMin >= qMax) return null;

  // Linear regression over quefrency range
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let q = qMin; q <= qMax; q++) {
    const y = cRe[q];
    sx += q; sy += y; sxx += q*q; sxy += q*y; n++;
  }
  const det = n*sxx - sx*sx;
  let slope = 0, intercept = 0;
  if (Math.abs(det) > 1e-12) {
    slope     = (n*sxy - sx*sy) / det;
    intercept = (sy - slope*sx) / n;
  }

  // Peak quefrency
  let peakVal = -Infinity, peakQ = qMin;
  for (let q = qMin; q <= qMax; q++) {
    if (cRe[q] > peakVal) { peakVal = cRe[q]; peakQ = q; }
  }

  const regAtPeak = slope * peakQ + intercept;
  const cpp       = (peakVal - regAtPeak) * 10;
  return isFinite(cpp) ? cpp : null;
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

function push(arr, val, cap) {
  arr.push(val);
  if (arr.length > cap) arr.shift();
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
   STATISTICS HELPERS
════════════════════════════════════════════ */
function arrMean(a) {
  if (!a.length) return 0;
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}
function arrStd(a) {
  if (a.length < 2) return 0;
  const m = arrMean(a);
  let s = 0; for (let i = 0; i < a.length; i++) s += (a[i]-m)*(a[i]-m);
  return Math.sqrt(s / a.length);
}
function arrMedian(a) {
  if (!a.length) return 0;
  const s = a.slice().sort((x,y) => x-y);
  const m = s.length >> 1;
  return s.length & 1 ? s[m] : (s[m-1]+s[m])/2;
}
function fmt(v, dec) {
  if (v === null || v === undefined || !isFinite(v)) return '\u2014';
  return v.toFixed(dec);
}

/* ════════════════════════════════════════════
   METRIC COMPUTATIONS
════════════════════════════════════════════ */
function computeJitter() {
  if (jitterDiffCount < 2 || !f0History.length) return null;
  const m = arrMean(f0History);
  return m > 0 ? (jitterDiffSum / jitterDiffCount / m) * 100 : null;
}
function computeShimmer() {
  if (shimmerDiffCount < 2 || !rmsHistory.length) return null;
  const m = arrMean(rmsHistory);
  return m > 0 ? (shimmerDiffSum / shimmerDiffCount / m) * 100 : null;
}

function computeNoiseFloor() {
  if (noiseHistory.length < 3) return null;
  return arrMedian(noiseHistory);
}

/* ════════════════════════════════════════════
   LPC FORMANT ESTIMATION
════════════════════════════════════════════ */
const LPC_ORDER = 14;

function computeLPCCoeffs(frame) {
  const n = Math.min(frame.length, 1024);
  const x = new Float32Array(n);
  x[0] = frame[0];
  for (let i = 1; i < n; i++) x[i] = frame[i] - 0.97 * frame[i - 1];
  for (let i = 0; i < n; i++) x[i] *= 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1));

  const r = new Float32Array(LPC_ORDER + 1);
  for (let k = 0; k <= LPC_ORDER; k++)
    for (let j = 0; j < n - k; j++) r[k] += x[j] * x[j + k];
  if (r[0] < 1e-10) return null;

  const a = new Float32Array(LPC_ORDER + 1);
  a[0] = 1;
  let err = r[0];
  for (let m = 1; m <= LPC_ORDER; m++) {
    let lam = r[m];
    for (let j = 1; j < m; j++) lam += a[j] * r[m - j];
    const km = -lam / err;
    if (Math.abs(km) >= 1) return null;
    const tmp = a.slice();
    for (let j = 1; j < m; j++) a[j] = tmp[j] + km * tmp[m - j];
    a[m] = km;
    err *= (1 - km * km);
    if (err <= 0) return null;
  }
  return a;
}

function lpcFormants(frame, sr) {
  let workFrame = frame;
  let effSr = sr;
  if (sr > 16000) {
    const step = Math.round(sr / 11025);
    const h = [0.0625, 0.25, 0.375, 0.25, 0.0625]; // 5-tap low-pass anti-alias
    const dLen = Math.floor(frame.length / step);
    const ds = new Float32Array(dLen);
    for (let i = 0; i < dLen; i++) {
      let s = 0;
      for (let k = 0; k < h.length; k++) {
        const idx = i * step - 2 + k;
        if (idx >= 0 && idx < frame.length) s += h[k] * frame[idx];
      }
      ds[i] = s;
    }
    workFrame = ds;
    effSr = sr / step;
  }

  const off = Math.max(0, Math.floor((workFrame.length - 1024) / 2));
  const sub = workFrame.subarray
    ? workFrame.subarray(off, off + 1024)
    : workFrame.slice(off, off + 1024);
  const a = computeLPCCoeffs(sub);
  if (!a) return null;

  const N = 512;
  const spec = new Float32Array(N);
  let maxV = 0;
  for (let k = 0; k < N; k++) {
    const omega = Math.PI * k / N;
    let re = 0, im = 0;
    for (let j = 0; j <= LPC_ORDER; j++) {
      re += a[j] * Math.cos(j * omega);
      im += a[j] * Math.sin(j * omega);
    }
    spec[k] = 1 / (re * re + im * im + 1e-12);
    if (spec[k] > maxV) maxV = spec[k];
  }

  const binHz = effSr / (2 * N);
  const thresh = maxV * 0.01;
  const peaks = [];
  for (let k = 1; k < N - 1; k++) {
    if (spec[k] > thresh && spec[k] > spec[k - 1] && spec[k] > spec[k + 1])
      peaks.push(k * binHz);
  }

  const f1 = peaks.find(f => f >= 200 && f <= 1000) ?? null;
  const f2 = f1 ? (peaks.find(f => f > f1 + 150 && f <= 3000) ?? null) : null;
  return (f1 && f2) ? { f1, f2 } : null;
}

function convexHullArea(pts) {
  if (pts.length < 3) return null;
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (O, A, B) => (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0]);
  const h = [];
  for (const pt of p) {
    while (h.length >= 2 && cross(h[h.length-2], h[h.length-1], pt) <= 0) h.pop();
    h.push(pt);
  }
  const lo = h.length + 1;
  for (let i = p.length - 2; i >= 0; i--) {
    while (h.length >= lo && cross(h[h.length-2], h[h.length-1], p[i]) <= 0) h.pop();
    h.push(p[i]);
  }
  h.pop();
  let area = 0;
  for (let i = 0; i < h.length; i++) {
    const j = (i + 1) % h.length;
    area += h[i][0] * h[j][1] - h[j][0] * h[i][1];
  }
  return Math.abs(area) / 2;
}

function computeVSA() {
  if (formantHistory.length < 10) return null;
  return convexHullArea(formantHistory.map(f => [f.f1, f.f2]));
}

function computeAvgPause() {
  if (pauseDurations.length === 0) return null;
  return arrMean(pauseDurations);
}


function computeSPLVariability() {
  if (splHistory.length < 5) return null;
  const s = splHistory.slice().sort((a,b) => a-b);
  const p10 = s[Math.floor(s.length * 0.10)];
  const p90 = s[Math.floor(s.length * 0.90)];
  return p90 - p10;
}

function computeSpeechRate() {
  if (splHistory.length < 5 || !sessionStart) return null;
  const elapsed = (Date.now() - sessionStart) / 1000;
  if (elapsed < 1) return null;

  // Smooth with window-3
  const sm = [];
  for (let i = 1; i < splHistory.length - 1; i++) {
    sm.push((splHistory[i-1] + splHistory[i] + splHistory[i+1]) / 3);
  }
  const mSpl    = arrMean(sm);
  const thresh  = mSpl - 5;

  let peaks = 0;
  for (let i = 1; i < sm.length - 1; i++) {
    if (sm[i] > thresh && sm[i] > sm[i-1] && sm[i] > sm[i+1]) peaks++;
  }
  return peaks / elapsed;
}

function computeProsody() {
  if (minF0 < Infinity && maxF0 > -Infinity && maxF0 > minF0) {
    return 12 * Math.log2(maxF0 / minF0);
  }
  return null;
}

/* ════════════════════════════════════════════
   METRIC SCALES  (literature-based fixed ranges)
════════════════════════════════════════════ */
const METRIC_SCALES = {
  't1-f0':      { min: 50,  max: 400, good: 'center', gLo: 80,  gHi: 260 },
  't1-spl':     { min: 40,  max: 100, good: 'center', gLo: 55,  gHi: 80  },
  't1-mpt':     { min: 0,   max: 30,  good: 'high'                        },
  't1-jitter':  { min: 0,   max: 5,   good: 'low'                         },
  't1-shimmer': { min: 0,   max: 10,  good: 'low'                         },
  't1-cpp':     { min: 0,   max: 25,  good: 'high'                        },
  't1-hnr':     { min: 0,   max: 35,  good: 'high'                        },
  't1-noise':   { min: 30,  max: 80,  good: 'low'                         },
  't2-minf0':   { min: 50,  max: 300, good: 'low'                         },
  't2-maxf0':   { min: 100, max: 600, good: 'high'                        },
  't2-range':   { min: 0,   max: 48,  good: 'high'                        },
  't2-noise':   { min: 30,  max: 80,  good: 'low'                         },
  't3-meanf0':  { min: 50,  max: 400, good: 'center', gLo: 80,  gHi: 260 },
  't3-hspl':    { min: 40,  max: 100, good: 'center', gLo: 55,  gHi: 80  },
  't3-f0std':   { min: 0,   max: 100, good: 'center', gLo: 15,  gHi: 50  },
  't3-cpp':     { min: 0,   max: 25,  good: 'high'                        },
  't3-mspl':    { min: 40,  max: 100, good: 'center', gLo: 55,  gHi: 80  },
  't3-splvar':  { min: 0,   max: 30,  good: 'center', gLo: 4,   gHi: 15  },
  't3-rate':    { min: 0,   max: 600, good: 'center', gLo: 180, gHi: 420 },
  't3-prosody': { min: 0,   max: 24,  good: 'center', gLo: 4,   gHi: 15  },
  't3-artic':      { min: 0,   max: 100,  good: 'high'                         },
  't3-noise':      { min: 30,  max: 80,   good: 'low'                          },
  't3-consonants': { min: 0,   max: 100,  good: 'high'                         },
  't3-pause':      { min: 0,   max: 3,    good: 'center', gLo: 0.1, gHi: 0.8  },
  't3-vsa':        { min: 0,   max: 500,  good: 'high'                         },
};

function scaleGradient(cfg) {
  const R = '#e05555', A = '#f5a623', G = '#5cb85c';
  if (cfg.good === 'high') return `linear-gradient(to right,${R},${A} 45%,${G})`;
  if (cfg.good === 'low')  return `linear-gradient(to right,${G},${A} 55%,${R})`;
  const span = cfg.max - cfg.min;
  const lo   = ((cfg.gLo - cfg.min) / span * 100).toFixed(1);
  const hi   = ((cfg.gHi - cfg.min) / span * 100).toFixed(1);
  const loA  = Math.max(0,   (cfg.gLo - cfg.min) / span * 100 - 10).toFixed(1);
  const hiA  = Math.min(100, (cfg.gHi - cfg.min) / span * 100 + 10).toFixed(1);
  return `linear-gradient(to right,${R} 0%,${A} ${loA}%,${G} ${lo}%,${G} ${hi}%,${A} ${hiA}%,${R} 100%)`;
}

const HIST_IDS = new Set([
  't1-f0','t1-spl','t1-cpp','t1-hnr','t1-noise',
  't2-noise',
  't3-meanf0','t3-hspl','t3-cpp','t3-mspl','t3-noise',
]);

function initScales() {
  for (const [id, cfg] of Object.entries(METRIC_SCALES)) {
    const valueEl = document.getElementById(id);
    if (!valueEl) continue;
    const card = valueEl.closest('.metric-card');
    if (!card) continue;
    const desc = card.querySelector('.metric-desc');
    const wrap = document.createElement('div');
    if (HIST_IDS.has(id)) {
      wrap.className = 'metric-scale hist-enabled';
      wrap.innerHTML = `<svg class="metric-hist" id="hist-${id}" viewBox="0 0 100 48" preserveAspectRatio="none"></svg>`
                     + `<div class="scale-track" style="background:${scaleGradient(cfg)}"></div>`;
    } else {
      wrap.className = 'metric-scale';
      wrap.innerHTML = `<div class="scale-thumb" id="${id}-thumb">&#9660;</div>`
                     + `<div class="scale-track" style="background:${scaleGradient(cfg)}"></div>`;
    }
    card.insertBefore(wrap, desc);
  }
}

function updateScale(id, value) {
  const thumb = document.getElementById(id + '-thumb');
  if (!thumb) return;
  if (value === null || value === undefined || isNaN(value)) {
    thumb.style.display = 'none';
    return;
  }
  const cfg = METRIC_SCALES[id];
  const pct = Math.max(0, Math.min(100, (value - cfg.min) / (cfg.max - cfg.min) * 100));
  thumb.style.left    = pct + '%';
  thumb.style.display = 'block';
}

function updateHistogram(id, data, avgValue) {
  const svg = document.getElementById('hist-' + id);
  if (!svg) return;
  if (!data || data.length < 2) { svg.innerHTML = ''; return; }
  const cfg = METRIC_SCALES[id];
  const N = 28;
  const bins = new Int32Array(N);
  for (const v of data) {
    const b = Math.floor((v - cfg.min) / (cfg.max - cfg.min) * N);
    bins[Math.max(0, Math.min(N - 1, b))]++;
  }
  const maxBin = Math.max(...bins);
  if (maxBin === 0) { svg.innerHTML = ''; return; }
  const W = 100, H = 48, bw = W / N;
  let html = '';
  for (let i = 0; i < N; i++) {
    if (!bins[i]) continue;
    const h = (bins[i] / maxBin) * H;
    html += `<rect x="${(i*bw).toFixed(2)}" y="${(H-h).toFixed(2)}" `
          + `width="${(bw-0.5).toFixed(2)}" height="${h.toFixed(2)}" `
          + `fill="rgba(255,255,255,0.3)" rx="0.8"/>`;
  }
  if (avgValue !== null && avgValue !== undefined && isFinite(avgValue)) {
    const x = Math.max(0, Math.min(W, (avgValue - cfg.min) / (cfg.max - cfg.min) * W));
    html += `<line x1="${x.toFixed(2)}" y1="0" x2="${x.toFixed(2)}" y2="${H}" `
          + `stroke="white" stroke-width="1.5" opacity="0.9"/>`;
  }
  svg.innerHTML = html;
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

/* ════════════════════════════════════════════
   AUDIO PLAYBACK (canvas play button)
════════════════════════════════════════════ */
let _pbCtx    = null;   // AudioContext
let _pbNode   = null;   // AudioBufferSourceNode
let _pbStart  = 0;      // ctx.currentTime when playback began
let _pbOffset = 0;      // seconds into audio at start (for pause/resume)
let _pbDur    = 0;      // total audio duration in seconds
let _pbAnimId = null;
let _pbActive = false;

function initAudioPlayback() {
  const ids = { 1: 'waveform-canvas', 2: 'pitch-canvas-2', 3: 'pitch-canvas-3' };
  const canvas = document.getElementById(ids[CURRENT_TASK]);
  if (!canvas) return;
  const wrap = canvas.closest('.canvas-wrap');
  if (!wrap) return;
  const btn = document.createElement('button');
  btn.id        = 'canvas-play-btn';
  btn.className = 'canvas-play-btn';
  btn.title     = 'Play recording';
  btn.innerHTML = '&#9654; Play';
  btn.disabled  = true;
  btn.onclick   = _togglePlayback;
  wrap.appendChild(btn);
}

function _enablePlaybackBtn() {
  const btn = document.getElementById('canvas-play-btn');
  if (btn) btn.disabled = false;
}

function _resetPlaybackState() {
  _pbActive = false;
  _pbOffset = 0;
  if (_pbNode)   { try { _pbNode.stop(); } catch(e) {} _pbNode = null; }
  if (_pbCtx)    { _pbCtx.close(); _pbCtx = null; }
  if (_pbAnimId) { cancelAnimationFrame(_pbAnimId); _pbAnimId = null; }
  const btn = document.getElementById('canvas-play-btn');
  if (btn) { btn.innerHTML = '&#9654; Play'; btn.title = 'Play recording'; btn.disabled = true; }
}

function _togglePlayback() {
  if (_pbActive) _pausePlayback(); else _startPlayback();
}

function _startPlayback() {
  if (!asrAudioBuf.length) return;
  const samples = new Float32Array(asrAudioBuf);
  _pbDur = samples.length / asrSampleRate;
  if (_pbOffset >= _pbDur) _pbOffset = 0;

  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  _pbCtx = ctx;
  const audioBuf = ctx.createBuffer(1, samples.length, asrSampleRate);
  audioBuf.getChannelData(0).set(samples);

  const src = ctx.createBufferSource();
  src.buffer = audioBuf;
  src.connect(ctx.destination);
  src.start(0, _pbOffset);
  src.onended = () => { if (_pbActive) _stopPlayback(); };
  _pbNode   = src;
  _pbStart  = ctx.currentTime;
  _pbActive = true;

  const btn = document.getElementById('canvas-play-btn');
  if (btn) { btn.innerHTML = '&#9646;&#9646; Pause'; btn.title = 'Pause'; }
  _pbAnimId = requestAnimationFrame(_pbAnimLoop);
}

function _pausePlayback() {
  if (_pbCtx) _pbOffset += _pbCtx.currentTime - _pbStart;
  _pbActive = false;
  if (_pbNode)   { try { _pbNode.stop(); } catch(e) {} _pbNode = null; }
  if (_pbCtx)    { _pbCtx.close(); _pbCtx = null; }
  if (_pbAnimId) { cancelAnimationFrame(_pbAnimId); _pbAnimId = null; }
  const btn = document.getElementById('canvas-play-btn');
  if (btn) { btn.innerHTML = '&#9654; Play'; btn.title = 'Play recording'; }
  _drawStaticGraph(null);
}

function _stopPlayback() {
  _pbOffset = 0;
  _pbActive = false;
  if (_pbNode)   { try { _pbNode.stop(); } catch(e) {} _pbNode = null; }
  if (_pbCtx)    { _pbCtx.close(); _pbCtx = null; }
  if (_pbAnimId) { cancelAnimationFrame(_pbAnimId); _pbAnimId = null; }
  const btn = document.getElementById('canvas-play-btn');
  if (btn) { btn.innerHTML = '&#9654; Play'; btn.title = 'Play recording'; }
  _drawStaticGraph(null);
}

function _pbAnimLoop() {
  if (!_pbActive || !_pbCtx) return;
  const elapsed  = _pbOffset + (_pbCtx.currentTime - _pbStart);
  const progress = Math.min(1, elapsed / _pbDur);
  _drawStaticGraph(progress);
  if (progress < 1) {
    _pbAnimId = requestAnimationFrame(_pbAnimLoop);
  }
}

function _drawStaticGraph(progress) {
  if (CURRENT_TASK === 1) drawFullWaveform(progress);
  else                    drawPitchContour(CURRENT_TASK, progress);
}

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
    { key: 'duration_s',          label: 'Audio\n(s)',      dec: 1 },
    { key: 'analysis_duration_s', label: 'Analysis\n(s)',   dec: 1 },
    { key: 'f0_mean_hz',          label: 'F0 Mean\n(Hz)',   dec: 1 },
    { key: 'spl_mean_db',    label: 'SPL\n(dB)',        dec: 1 },
    { key: 'mpt_s',          label: 'MPT\n(s)',         dec: 1 },
    { key: 'jitter_pct',     label: 'Jitter\n(%)',      dec: 2 },
    { key: 'shimmer_pct',    label: 'Shimmer\n(%)',     dec: 2 },
    { key: 'cpp_db',         label: 'CPP\n(dB)',        dec: 1 },
    { key: 'hnr_db',         label: 'HNR\n(dB)',        dec: 1 },
    { key: 'noise_floor_db', label: 'Noise\n(dB)',      dec: 1 },
  ],
  2: [
    { key: 'duration_s',          label: 'Audio\n(s)',      dec: 1 },
    { key: 'analysis_duration_s', label: 'Analysis\n(s)',   dec: 1 },
    { key: 'f0_min_hz',           label: 'F0 Min\n(Hz)',    dec: 1 },
    { key: 'f0_max_hz',      label: 'F0 Max\n(Hz)',     dec: 1 },
    { key: 'f0_range_st',    label: 'F0 Range\n(ST)',   dec: 1 },
    { key: 'noise_floor_db', label: 'Noise\n(dB)',      dec: 1 },
  ],
  3: [
    { key: 'duration_s',              label: 'Audio\n(s)',       dec: 1 },
    { key: 'analysis_duration_s',     label: 'Analysis\n(s)',    dec: 1 },
    { key: 'f0_mean_hz',              label: 'F0 Mean\n(Hz)',    dec: 1 },
    { key: 'f0_std_hz',               label: 'F0 Std\n(Hz)',     dec: 1 },
    { key: 'cpp_db',                  label: 'CPP\n(dB)',        dec: 1 },
    { key: 'spl_mean_db',             label: 'SPL\n(dB)',        dec: 1 },
    { key: 'spl_variability_db',      label: 'SPL Var\n(dB)',    dec: 1 },
    { key: 'speech_rate_wpm',         label: 'Rate\n(WPM)',      dec: 0 },
    { key: 'prosody_st',              label: 'Prosody\n(ST)',    dec: 1 },
    { key: 'noise_floor_db',          label: 'Noise\n(dB)',      dec: 1 },
    { key: 'avg_pause_s',             label: 'Pause\n(s)',       dec: 2 },
    { key: 'vsa_khz2',                label: 'VSA\n(kHz²)',      dec: 0 },
    { key: 'articulation_pct',        label: 'Artic.\n(%)',      dec: 1 },
    { key: 'consonant_precision_pct', label: 'Conson.\n(%)',     dec: 1 },
  ],
};

let logDB = null;
let lastLogId = { 1: null, 2: null, 3: null };
const _sortState = { 1: null, 2: null, 3: null }; // null = default (newest first)

function sortLogTable(task, key) {
  const cur = _sortState[task];
  if (cur && cur.key === key) {
    _sortState[task] = cur.dir === 'asc' ? { key, dir: 'desc' } : null;
  } else {
    _sortState[task] = { key, dir: 'asc' };
  }
  renderLogTable(task);
}

function _thSort(task, key, label) {
  const s = _sortState[task];
  const active = s && s.key === key;
  const cls    = active ? ' sort-' + s.dir : '';
  const ind    = active ? (s.dir === 'asc' ? ' ▲' : ' ▼') : '';
  return '<th class="sortable' + cls + '" onclick="sortLogTable(' + task + ',\'' + key + '\')">' + label + ind + '</th>';
}

function initLogDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('speech-metrics-log', 1);
    req.onupgradeneeded = e => {
      const d = e.target.result;
      [1, 2, 3].forEach(t => {
        if (!d.objectStoreNames.contains('task' + t))
          d.createObjectStore('task' + t, { keyPath: 'id', autoIncrement: true });
      });
    };
    req.onsuccess = e => { logDB = e.target.result; resolve(); };
    req.onerror   = e => reject(e);
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

// Step 1: create entry immediately with audio data, no metrics yet
function addPendingLogEntry(task) {
  if (!logDB) return;
  const audioData = (asrAudioBuf && asrAudioBuf.length)
    ? encodeWAV(new Float32Array(asrAudioBuf), asrSampleRate)
    : null;
  const duration_s = asrAudioBuf.length / asrSampleRate;
  const entry = { filename: currentFilename, timestamp: new Date().toISOString(), duration_s };
  if (audioData) { entry.audioData = audioData; entry.audioSR = asrSampleRate; }
  if (task === 3 && activePassageRef) entry.referenceText = activePassageRef;
  const tx  = logDB.transaction('task' + task, 'readwrite');
  const req = tx.objectStore('task' + task).add(entry);
  req.onsuccess = e => { lastLogId[task] = e.target.result; renderLogTable(task); };
}

// Step 2: fill in server metrics on the pending entry once the server responds
function _fillPendingLogEntry(task, data, analysis_duration_s) {
  if (!logDB || !lastLogId[task]) return;
  const tx    = logDB.transaction('task' + task, 'readwrite');
  const store = tx.objectStore('task' + task);
  const req   = store.get(lastLogId[task]);
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
  const tx    = logDB.transaction('task' + task, 'readwrite');
  const store = tx.objectStore('task' + task);
  const req   = store.get(lastLogId[task]);
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
    req.onerror   = () => resolve([]);
  });
}

function _escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function _fmtTs(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function renderLogTable(task) {
  _getAllLogEntries(task).then(entries => {
    const wrap   = document.getElementById('log-table-' + task);
    const delBtn = document.getElementById('log-delete-' + task);
    const expBtn = document.getElementById('log-export-' + task);
    if (!wrap) return;
    if (entries.length === 0) {
      wrap.innerHTML = '<p class="log-empty">No analyses recorded yet.</p>';
      if (delBtn) delBtn.disabled = true;
      if (expBtn) expBtn.disabled = true;
      return;
    }
    if (expBtn) expBtn.disabled = false;
    const metrics = LOG_METRICS[task];

    // Sort entries
    const sort = _sortState[task];
    let display;
    if (sort) {
      display = entries.slice().sort((a, b) => {
        const av = a[sort.key], bv = b[sort.key];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        const an = Number(av), bn = Number(bv);
        const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : String(av).localeCompare(String(bv));
        return sort.dir === 'asc' ? cmp : -cmp;
      });
    } else {
      display = entries.slice().reverse(); // newest first (default)
    }

    let html = '<div class="log-table-wrap"><table class="log-table"><thead><tr>';
    html += '<th><input type="checkbox" onchange="logToggleAll(this,' + task + ')" title="Select all"></th>';
    html += _thSort(task, 'filename', 'File');
    metrics.forEach(m => { html += _thSort(task, m.key, m.label.replace('\n', '<br>')); });
    html += _thSort(task, 'timestamp', 'Time');
    html += _thSort(task, 'version', 'Ver.');
    html += '</tr></thead><tbody>';
    display.forEach(e => {
      html += '<tr>';
      const playBtn = e.audioData
        ? '<button class="log-play-btn" data-id="' + e.id + '" data-task="' + task + '" onclick="replayLogEntry(' + task + ',' + e.id + ')" title="Re-analyze">↑</button>'
        : '';
      const dlLabel = task === 3 ? '↓ zip' : '↓ wav';
      const dlTitle = task === 3 ? 'Download ZIP (audio + passage)' : 'Download WAV';
      const dlBtn = e.audioData
        ? '<button class="log-dl-btn" onclick="downloadLogAudio(event,' + task + ',' + e.id + ')" title="' + dlTitle + '">' + dlLabel + '</button>'
        : '';
      html += '<td class="log-ctrl-cell"><input type="checkbox" class="log-chk" data-id="' + e.id + '" data-task="' + task + '" onchange="logChkChange(' + task + ')">' + playBtn + dlBtn + '</td>';
      const fname = _escHtml(e.filename || '—');
      html += '<td class="log-filename" title="' + _escHtml(e.filename || '') + ' (click to rename)" onclick="renameLogEntry(this,' + task + ',' + e.id + ')">' + fname + '</td>';
      metrics.forEach(m => {
        const v = e[m.key];
        html += '<td>' + (v != null ? fmt(parseFloat(v), m.dec) : '—') + '</td>';
      });
      html += '<td class="log-ts">' + _fmtTs(e.timestamp) + '</td>';
      html += '<td class="log-ts" title="' + _escHtml(e.version||'') + '">' + _escHtml(e.version || '—') + '</td>';
      html += '</tr>';
    });
    html += '</tbody></table></div>';
    wrap.innerHTML = html;
    if (delBtn) delBtn.disabled = true;
  });
}

function logChkChange(task) {
  const any = [...document.querySelectorAll('.log-chk[data-task="' + task + '"]')].some(c => c.checked);
  const btn = document.getElementById('log-delete-' + task);
  if (btn) btn.disabled = !any;
}

function logToggleAll(masterChk, task) {
  document.querySelectorAll('.log-chk[data-task="' + task + '"]').forEach(c => c.checked = masterChk.checked);
  logChkChange(task);
}

function deleteSelected(task) {
  if (!logDB) return;
  const ids = [...document.querySelectorAll('.log-chk[data-task="' + task + '"]:checked')].map(c => +c.dataset.id);
  if (!ids.length) return;
  const tx    = logDB.transaction('task' + task, 'readwrite');
  const store = tx.objectStore('task' + task);
  ids.forEach(id => store.delete(id));
  tx.oncomplete = () => renderLogTable(task);
}

function exportLogTSV(task) {
  _getAllLogEntries(task).then(entries => {
    if (!entries.length) return;
    const metrics = LOG_METRICS[task];
    const headers = ['filename', 'timestamp', ...metrics.map(m => m.key), 'version'];
    const lines   = [headers.join('\t')];
    entries.forEach(e => {
      lines.push(headers.map(h => { const v = e[h]; return v != null ? String(v) : ''; }).join('\t'));
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/tab-separated-values' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    const names = { 1: 'sustained-phonation', 2: 'pitch-glides', 3: 'reading-passage' };
    const now   = new Date();
    const ts    = now.getFullYear() + '-' + String(now.getMonth()+1).padStart(2,'0') + '-' +
                  String(now.getDate()).padStart(2,'0') + '_' +
                  String(now.getHours()).padStart(2,'0') + '-' + String(now.getMinutes()).padStart(2,'0');
    a.download  = names[task] + '_' + ts + '.tsv';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  });
}

function downloadLogAudio(event, task, id) {
  event.stopPropagation();
  if (!logDB) return;
  const req = logDB.transaction('task' + task, 'readonly').objectStore('task' + task).get(id);
  req.onsuccess = async e => {
    const entry = e.target.result;
    if (!entry || !entry.audioData) return;
    const base = (entry.filename || 'recording').replace(/\.(wav|zip)$/i, '');
    let blob, filename;
    if (task === 3) {
      const zip = new JSZip();
      zip.file('recording.wav', entry.audioData);
      if (entry.referenceText) zip.file('passage.txt', entry.referenceText);
      blob     = await zip.generateAsync({ type: 'blob' });
      filename = base + '.zip';
    } else {
      blob     = new Blob([entry.audioData], { type: 'audio/wav' });
      filename = base + '.wav';
    }
    const url = URL.createObjectURL(blob);
    const a   = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };
}

function renameLogEntry(cell, task, id) {
  if (cell.querySelector('input')) return;
  const current = cell.textContent.trim();

  function restoreCell(name) {
    cell.textContent = name;
    cell.title = name + ' (click to rename)';
  }

  const input = document.createElement('input');
  input.className = 'log-rename-input';
  input.value = current;
  cell.innerHTML = '';
  cell.appendChild(input);
  input.focus();
  input.select();

  let committed = false;
  function commit() {
    if (committed) return;
    committed = true;
    const newName = input.value.trim() || current;
    if (!logDB) { restoreCell(current); return; }
    const tx    = logDB.transaction('task' + task, 'readwrite');
    const store = tx.objectStore('task' + task);
    const req   = store.get(id);
    req.onsuccess = ev => {
      const entry = ev.target.result;
      if (!entry) { restoreCell(current); return; }
      entry.filename = newName;
      store.put(entry).onsuccess = () => restoreCell(newName);
    };
    req.onerror = () => restoreCell(current);
  }
  function cancel() { committed = true; restoreCell(current); }

  input.addEventListener('blur', commit);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter')  { e.preventDefault(); input.blur(); }
    if (e.key === 'Escape') { e.preventDefault(); cancel(); }
  });
}

async function replayLogEntry(task, id) {
  if (!logDB) return;

  const btn = document.querySelector('.log-play-btn[data-id="' + id + '"][data-task="' + task + '"]');
  if (btn) { btn.disabled = true; btn.textContent = '↻'; btn.classList.add('loading'); }

  try {
    const entry = await new Promise((resolve, reject) => {
      const req = logDB.transaction('task' + task, 'readonly').objectStore('task' + task).get(id);
      req.onsuccess = e => resolve(e.target.result);
      req.onerror   = () => reject(new Error('DB read failed'));
    });
    if (!entry || !entry.audioData) throw new Error('No audio stored for this entry');

    const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await tempCtx.decodeAudioData(entry.audioData.slice(0));
    await tempCtx.close();
    const mono = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += ch[i];
    }
    for (let i = 0; i < mono.length; i++) mono[i] /= decoded.numberOfChannels;
    asrAudioBuf   = Array.from(mono);
    asrSampleRate = entry.audioSR || decoded.sampleRate;
    currentFilename = entry.filename || 'replay';

    const blob = new Blob([entry.audioData], { type: 'audio/wav' });
    const form = new FormData();
    form.append('file', blob, 'recording.wav');
    const refText = entry.referenceText || (task === 3 ? activePassageRef : null);
    if (task === 3 && refText) form.append('reference_text', refText);
    const t0 = Date.now();
    const serverPromise = fetch(`${SERVER_URL}/analyze/task${task}`, { method: 'POST', body: form })
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
      const tx    = logDB.transaction('task' + task, 'readwrite');
      const store = tx.objectStore('task' + task);
      const req   = store.get(id);
      req.onsuccess = ev => {
        const existing = ev.target.result;
        if (!existing) { reject(new Error('Entry gone')); return; }
        Object.assign(existing, newMetrics);
        store.put(existing).onsuccess = resolve;
      };
      req.onerror = reject;
    });

    renderLogTable(task);
  } catch (err) {
    console.error('Re-analysis failed:', err);
    if (btn) { btn.disabled = false; btn.textContent = '↑'; btn.classList.remove('loading'); }
  }
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
