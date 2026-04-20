'use strict';

const SC_SERVER = 'https://98.87.120.209:8765';
const SILENCE_MS   = 600;
const RMS_THRESH   = 0.012;
const MIN_SPEECH   = 150;

// ── State ──
let scAllSentences = [];   // from server
let scQueue        = [];   // shuffled ids for current category
let scQueueIdx     = 0;
let scListening    = false;
let scProcessing   = false;
let scSpeechStart  = 0;
let scSilenceTimer = null;
let scAudioCtx     = null;
let scStream       = null;
let scProcessor    = null;
let scPcmChunks    = [];
let scSampleRate   = 16000;
let scScore        = 0;
let scWrong        = 0;
let scSkipped      = 0;
let scLog          = [];    // {sentence, heard, canonical, correct, skipped, ms}

// ── Init ──
(async function init() {
  try {
    const resp = await fetch(`${SC_SERVER}/sentence-completion/sentences`, {credentials:'include'});
    scAllSentences = await resp.json();
  } catch(e) {
    document.getElementById('sc-status').textContent = 'Could not load sentences.';
    return;
  }
  initUsername();
  buildQueue();
  showSentence();
})();

function initUsername() {
  fetch(`${SC_SERVER}/me`, {credentials:'include'})
    .then(r => r.json())
    .then(d => {
      const el = document.getElementById('nav-user');
      if (el && d.username) el.textContent = d.username;
    }).catch(() => {});
}

// ── Queue management ──
function currentCategory() {
  return document.getElementById('sc-category').value;
}

function buildQueue() {
  const cat = currentCategory();
  let pool = scAllSentences;
  if (cat !== 'all') pool = scAllSentences.filter(s => s.cat === cat);
  // Fisher-Yates shuffle
  const ids = pool.map(s => s.id);
  for (let i = ids.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [ids[i], ids[j]] = [ids[j], ids[i]];
  }
  scQueue = ids;
  scQueueIdx = 0;
}

function onSCCategoryChange() {
  stopSC();
  buildQueue();
  showSentence();
}

function currentSentence() {
  if (!scQueue.length) return null;
  const id = scQueue[scQueueIdx % scQueue.length];
  return scAllSentences.find(s => s.id === id) || null;
}

// ── Display ──
function showSentence() {
  const s = currentSentence();
  if (!s) {
    document.getElementById('sc-sentence').textContent = 'No sentences available.';
    return;
  }
  document.getElementById('sc-cat-badge').textContent = s.cat;
  document.getElementById('sc-cat-badge').className = 'sc-cat-badge sc-cat-' + s.cat.toLowerCase().replace(' ', '-');
  document.getElementById('sc-sentence').innerHTML = renderBlank(s.text, 'idle');
  document.getElementById('sc-progress').textContent =
    `${(scQueueIdx % scQueue.length) + 1} / ${scQueue.length}`;
  document.getElementById('sc-result').style.display = 'none';
  document.getElementById('sc-heard').style.display  = 'none';
  document.getElementById('sc-status').textContent   = '';
  setBtnState('idle');
}

function renderBlank(text, state) {
  let cls = 'sc-blank';
  if (state === 'listening') cls += ' sc-blank-pulse';
  if (state === 'correct')   cls += ' sc-blank-correct';
  if (state === 'wrong')     cls += ' sc-blank-wrong';
  return text.replace(/___/, `<span class="${cls}" id="sc-blank-span">___</span>`);
}

function fillBlank(word, state) {
  const span = document.getElementById('sc-blank-span');
  if (!span) return;
  span.textContent = word;
  span.className   = 'sc-blank sc-blank-' + state;
}

function setBtnState(state) {
  const btn   = document.getElementById('sc-btn');
  const label = document.getElementById('sc-btn-label');
  btn.className = 'record-btn ' + state;
  if (state === 'idle')       { label.textContent = 'Start Listening'; btn.disabled = false; }
  else if (state === 'active'){ label.textContent = 'Listening…';      btn.disabled = false; }
  else if (state === 'busy')  { label.textContent = 'Processing…';     btn.disabled = true;  }
}

// ── VAD & recording ──
async function toggleSC() {
  if (scProcessing) return;
  if (scListening) { stopSC(); return; }
  await startSC();
}

async function startSC() {
  try {
    scStream    = await navigator.mediaDevices.getUserMedia({audio:true});
    scAudioCtx  = new AudioContext();
    scSampleRate= scAudioCtx.sampleRate;
    const src   = scAudioCtx.createMediaStreamSource(scStream);
    scProcessor = scAudioCtx.createScriptProcessor(2048, 1, 1);
    src.connect(scProcessor);
    scProcessor.connect(scAudioCtx.destination);
  } catch(e) {
    document.getElementById('sc-status').textContent = 'Mic access denied.';
    return;
  }
  scPcmChunks  = [];
  scListening  = true;
  scSpeechStart= 0;
  if (scSilenceTimer) { clearTimeout(scSilenceTimer); scSilenceTimer = null; }

  const s = currentSentence();
  if (s) document.getElementById('sc-sentence').innerHTML = renderBlank(s.text, 'listening');

  setBtnState('active');
  document.getElementById('sc-status').textContent = 'Say the missing word…';

  scProcessor.onaudioprocess = function(e) {
    if (!scListening) return;
    const data = e.inputBuffer.getChannelData(0);
    let rms = 0;
    for (let i = 0; i < data.length; i++) rms += data[i] * data[i];
    rms = Math.sqrt(rms / data.length);
    if (rms >= RMS_THRESH) {
      if (!scSpeechStart) scSpeechStart = Date.now();
      clearTimeout(scSilenceTimer);
      scSilenceTimer = null;
      scPcmChunks.push(new Float32Array(data));
    } else if (scSpeechStart) {
      if (!scSilenceTimer) {
        scSilenceTimer = setTimeout(() => {
          if (Date.now() - scSpeechStart >= MIN_SPEECH) {
            const dur = Date.now() - scSpeechStart;
            const pcm = mergePCM(scPcmChunks);
            stopSC();
            sendClip(pcm, scSampleRate, dur);
          } else {
            scPcmChunks  = [];
            scSpeechStart= 0;
          }
        }, SILENCE_MS);
      }
    }
  };
}

function stopSC() {
  scListening = false;
  clearTimeout(scSilenceTimer);
  scSilenceTimer = null;
  if (scProcessor) { try { scProcessor.disconnect(); } catch(e){} scProcessor = null; }
  if (scStream)    { scStream.getTracks().forEach(t => t.stop()); scStream = null; }
  if (scAudioCtx)  { try { scAudioCtx.close(); } catch(e){} scAudioCtx = null; }
  if (!scProcessing) setBtnState('idle');
}

function mergePCM(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out   = new Float32Array(total);
  let offset  = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out;
}

function pcmToWav(pcm, sr) {
  const buf    = new ArrayBuffer(44 + pcm.length * 2);
  const view   = new DataView(buf);
  const write  = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  write(0,  'RIFF'); view.setUint32(4,  36 + pcm.length * 2, true);
  write(8,  'WAVE'); write(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, 'data'); view.setUint32(40, pcm.length * 2, true);
  let o = 44;
  for (let i = 0; i < pcm.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buf;
}

// ── Send clip ──
async function sendClip(pcm, sr, speechMs) {
  scProcessing = true;
  setBtnState('busy');
  document.getElementById('sc-status').textContent = 'Processing…';

  const sentence = currentSentence();
  if (!sentence) { scProcessing = false; setBtnState('idle'); return; }

  const t0  = Date.now();
  const wav = pcmToWav(pcm, sr);
  const fd  = new FormData();
  fd.append('file',        new Blob([wav], {type:'audio/wav'}), 'clip.wav');
  fd.append('sentence_id', sentence.id);

  let result;
  try {
    const resp = await fetch(`${SC_SERVER}/sentence-completion/check`, {
      method: 'POST', body: fd, credentials: 'include'
    });
    result = await resp.json();
  } catch(e) {
    document.getElementById('sc-status').textContent = 'Server error.';
    scProcessing = false;
    setBtnState('idle');
    return;
  }

  const ms = Date.now() - t0;
  showResult(result, sentence, speechMs, ms);
}

// ── Show result ──
function showResult(result, sentence, speechMs, serverMs) {
  const heard     = (result.transcript || '').trim();
  const correct   = result.correct;
  const canonical = result.canonical;

  // Fill blank
  fillBlank(correct ? canonical : canonical, correct ? 'correct' : 'wrong');

  // Result banner
  const rDiv = document.getElementById('sc-result');
  rDiv.style.display = 'block';
  if (correct) {
    rDiv.className   = 'sc-result-banner sc-result-win';
    rDiv.textContent = '✓ Correct!';
    scScore++;
    document.getElementById('sc-score').textContent = scScore;
  } else {
    rDiv.className   = 'sc-result-banner sc-result-lose';
    rDiv.textContent = `✗ The answer is "${canonical}"`;
    scWrong++;
    document.getElementById('sc-wrong').textContent = scWrong;
  }

  // Heard
  const hDiv = document.getElementById('sc-heard');
  if (heard) {
    hDiv.style.display = 'block';
    hDiv.textContent   = `Heard: "${heard}"`;
  }

  // Log entry
  scLog.push({
    num:       scLog.length + 1,
    sentence:  sentence.text,
    heard:     heard,
    canonical: canonical,
    correct:   correct,
    skipped:   false,
    ms:        speechMs,
  });
  appendLogRow(scLog[scLog.length - 1]);

  document.getElementById('sc-status').textContent = '';
  scProcessing = false;
  setBtnState('idle');

  // Auto-advance
  setTimeout(() => {
    advanceSentence();
  }, correct ? 1500 : 3000);
}

function advanceSentence() {
  scQueueIdx++;
  if (scQueueIdx >= scQueue.length) {
    // Reshuffle for another round
    buildQueue();
  }
  showSentence();
}

// ── Skip ──
function scSkip() {
  if (scListening) stopSC();
  const sentence = currentSentence();
  if (!sentence) return;
  scLog.push({
    num:       scLog.length + 1,
    sentence:  sentence.text,
    heard:     '—',
    canonical: '(skip)',
    correct:   false,
    skipped:   true,
    ms:        0,
  });
  appendLogRow(scLog[scLog.length - 1]);
  scSkipped++;
  document.getElementById('sc-skipped').textContent = scSkipped;
  advanceSentence();
}

// ── Log table ──
function appendLogRow(entry) {
  const tbody = document.getElementById('sc-log-body');
  document.getElementById('sc-log').style.display = 'block';

  const tr = document.createElement('tr');
  if (entry.skipped)  tr.className = 'sc-log-skip';
  else if (entry.correct)  tr.className = 'sc-log-ok';
  else tr.className = 'sc-log-err';

  const shortSentence = entry.sentence.replace(/___/, '___');
  tr.innerHTML = `
    <td>${entry.num}</td>
    <td class="sc-log-sent" title="${entry.sentence}">${shortSentence}</td>
    <td class="sc-log-heard">${entry.heard}</td>
    <td class="sc-log-answer">${entry.canonical}</td>
    <td class="sc-log-res">${entry.skipped ? '↷' : entry.correct ? '✓' : '✗'}</td>
    <td class="sc-log-ms">${entry.ms > 0 ? entry.ms : '—'}</td>
  `;
  tbody.appendChild(tr);
}
