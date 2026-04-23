'use strict';

const VF_SERVER = ''; // relative - same origin

// ── Categories ────────────────────────────────────────────────────────────────

const VF_CATEGORIES = [
  { id: 'fruits',      label: 'Fruits'      },
  { id: 'animals',     label: 'Animals'     },
  { id: 'vegetables',  label: 'Vegetables'  },
  { id: 'colors',      label: 'Colors'      },
  { id: 'furniture',   label: 'Furniture'   },
  { id: 'jobs',        label: 'Jobs'        },
  { id: 'letter:f',    label: 'Letter — F'  },
  { id: 'letter:a',    label: 'Letter — A'  },
  { id: 'letter:s',    label: 'Letter — S'  },
  { id: 'letter:p',    label: 'Letter — P'  },
  { id: 'letter:m',    label: 'Letter — M'  },
];

// ── State ─────────────────────────────────────────────────────────────────────

let gameActive   = false;
let timeLeft     = 60;
let totalTime    = 60;
let timerHandle  = null;

// word tracking
let wordOrder    = [];   // canonical words in order first said
let wordCount    = {};   // canonical → times said (including repeats)
let intrusionSet = [];   // unmatched words (intrusions)
let intrusionSeen = new Set();

// VAD
let audioCtx    = null;
let mediaStream = null;
let scriptNode  = null;
let sourceNode  = null;
let pcmChunks   = [];
let isSpeaking  = false;
let silenceTimer = null;
let speechStart  = 0;

const RMS_THRESH  = 0.015;
const SILENCE_MS  = 500;
const MIN_SPEECH  = 100;

// ── Init ──────────────────────────────────────────────────────────────────────

(function init() {
  const sel = document.getElementById('vf-category');
  VF_CATEGORIES.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = c.label;
    sel.appendChild(opt);
  });
  syncDurationDisplay();
  document.getElementById('vf-duration').addEventListener('change', syncDurationDisplay);
  initUsername();
})();

function syncDurationDisplay() {
  const t = +document.getElementById('vf-duration').value;
  if (!gameActive) {
    timeLeft  = t;
    totalTime = t;
    document.getElementById('vf-time-num').textContent = t;
    updateRing(1);
  }
}

function onCategoryChange() { /* live — no action needed */ }

// ── Game control ──────────────────────────────────────────────────────────────

async function toggleGame() {
  if (gameActive) {
    endGame(false);
  } else {
    await startGame();
  }
}

async function startGame() {
  // Reset state
  wordOrder    = [];
  wordCount    = {};
  intrusionSet = [];
  intrusionSeen = new Set();
  totalTime    = +document.getElementById('vf-duration').value;
  timeLeft     = totalTime;

  document.getElementById('vf-chips').innerHTML          = '';
  document.getElementById('vf-last-heard').textContent   = '';
  document.getElementById('vf-intrusions-row').style.display = 'none';
  document.getElementById('vf-summary').style.display        = 'none';
  document.getElementById('vf-category').disabled = true;
  document.getElementById('vf-duration').disabled = true;
  renderStats();
  updateRing(1);

  // Start mic
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch {
    setStatus('Microphone access denied.');
    document.getElementById('vf-category').disabled = false;
    document.getElementById('vf-duration').disabled = false;
    return;
  }

  audioCtx   = new AudioContext();
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
  pcmChunks  = [];
  isSpeaking = false;

  scriptNode.onaudioprocess = onAudio;
  sourceNode.connect(scriptNode);
  scriptNode.connect(audioCtx.destination);

  gameActive = true;
  const btn  = document.getElementById('vf-btn');
  btn.className = 'record-btn recording';
  document.getElementById('vf-btn-label').textContent = 'Stop';

  // Start countdown
  timerHandle = setInterval(tickTimer, 1000);
  setStatus('Listening…');
}

function endGame(timeout) {
  if (!gameActive) return;
  gameActive = false;

  clearInterval(timerHandle);
  timerHandle = null;
  if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }

  if (scriptNode)  { scriptNode.disconnect(); scriptNode  = null; }
  if (sourceNode)  { sourceNode.disconnect(); sourceNode  = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (audioCtx)    { audioCtx.close(); audioCtx = null; }

  const btn = document.getElementById('vf-btn');
  btn.className = 'record-btn idle';
  document.getElementById('vf-btn-label').textContent = 'Start';
  document.getElementById('vf-category').disabled = false;
  document.getElementById('vf-duration').disabled = false;

  if (timeout) {
    timeLeft = 0;
    document.getElementById('vf-time-num').textContent = '0';
    updateRing(0);
  }

  setStatus('');
  showSummary();
}

function tickTimer() {
  timeLeft = Math.max(0, timeLeft - 1);
  document.getElementById('vf-time-num').textContent = timeLeft;
  updateRing(timeLeft / totalTime);
  // Colour hint as time runs low
  const num = document.getElementById('vf-time-num');
  if (timeLeft <= 10)      num.style.color = '#e05c5c';
  else if (timeLeft <= 20) num.style.color = '#e0a85c';
  else                     num.style.color = '';
  if (timeLeft === 0) endGame(true);
}

// ── VAD ───────────────────────────────────────────────────────────────────────

function onAudio(e) {
  if (!gameActive) return;
  const f32 = e.inputBuffer.getChannelData(0);
  let sum = 0;
  for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
  const rms = Math.sqrt(sum / f32.length);

  const i16 = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) i16[i] = Math.max(-32768, Math.min(32767, f32[i] * 32768));

  if (rms > RMS_THRESH) {
    if (!isSpeaking) { isSpeaking = true; speechStart = Date.now(); pcmChunks = []; }
    if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
    pcmChunks.push(i16);
  } else if (isSpeaking) {
    pcmChunks.push(i16);
    if (!silenceTimer) {
      silenceTimer = setTimeout(() => {
        silenceTimer = null;
        isSpeaking   = false;
        if (Date.now() - speechStart >= MIN_SPEECH && pcmChunks.length) {
          const total  = pcmChunks.reduce((s, c) => s + c.length, 0);
          const merged = new Int16Array(total);
          let off = 0;
          pcmChunks.forEach(c => { merged.set(c, off); off += c.length; });
          pcmChunks = [];
          sendClip(merged, audioCtx.sampleRate);
        }
      }, SILENCE_MS);
    }
  }
}

// ── Server call ───────────────────────────────────────────────────────────────

function pcmToWav(pcm, sr) {
  const nc = 1, bps = 16, br = sr * bps / 8, ba = bps / 8;
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v   = new DataView(buf);
  const s   = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  s(0,'RIFF'); v.setUint32(4,36+pcm.length*2,true); s(8,'WAVE');
  s(12,'fmt '); v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,nc,true);
  v.setUint32(24,sr,true); v.setUint32(28,br,true); v.setUint16(32,ba,true); v.setUint16(34,bps,true);
  s(36,'data'); v.setUint32(40,pcm.length*2,true);
  for (let i = 0; i < pcm.length; i++) v.setInt16(44+i*2, pcm[i], true);
  return new Blob([buf], { type: 'audio/wav' });
}

async function sendClip(pcm, sr) {
  const category = document.getElementById('vf-category').value;
  const fd = new FormData();
  fd.append('file', pcmToWav(pcm, sr), 'clip.wav');
  fd.append('category', category);
  try {
    const res  = await fetch(`${VF_SERVER}/verbal-fluency/transcribe`, { method: 'POST', body: fd });
    const data = await res.json();
    processResult(data);
  } catch {
    setStatus('Server unreachable.');
  }
}

// ── Result processing ─────────────────────────────────────────────────────────

function processResult(data) {
  if (!gameActive) return;
  const { transcript, matched, unmatched } = data;
  if (transcript) {
    document.getElementById('vf-last-heard').textContent = 'Heard: "' + transcript.trim() + '"';
  }

  (matched || []).forEach(word => {
    if (!wordCount[word]) {
      wordCount[word] = 1;
      wordOrder.push(word);
    } else {
      wordCount[word]++;
    }
  });

  (unmatched || []).forEach(word => {
    const w = word.toLowerCase();
    if (!intrusionSeen.has(w) && word.length >= 3) {
      intrusionSeen.add(w);
      intrusionSet.push(word);
    }
  });

  renderChips();
  renderStats();
  renderIntrusions();
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function renderChips() {
  const el = document.getElementById('vf-chips');
  el.innerHTML = wordOrder.map(w => {
    const count = wordCount[w];
    const cls   = count > 1 ? 'vf-chip vf-chip-repeat' : 'vf-chip';
    const badge = count > 1 ? '<span class="vf-chip-badge">×' + count + '</span>' : '';
    return '<span class="' + cls + '">' + escHtml(w) + badge + '</span>';
  }).join('');
}

function renderStats() {
  const unique    = wordOrder.length;
  const repeats   = Object.values(wordCount).filter(c => c > 1).length;
  document.getElementById('vf-unique-count').textContent     = unique;
  document.getElementById('vf-repeat-count').textContent     = repeats;
  document.getElementById('vf-intrusion-count').textContent  = intrusionSet.length;
}

function renderIntrusions() {
  if (!intrusionSet.length) return;
  const row = document.getElementById('vf-intrusions-row');
  row.style.display = '';
  document.getElementById('vf-intrusions-list').textContent = intrusionSet.join(', ');
}

function showSummary() {
  const el   = document.getElementById('vf-summary');
  const body = document.getElementById('vf-summary-body');
  const cat  = VF_CATEGORIES.find(c => c.id === document.getElementById('vf-category').value);
  const unique  = wordOrder.length;
  const repeats = Object.values(wordCount).filter(c => c > 1).length;

  let html = '<div class="vf-summary-stats">';
  html += '<div class="vf-sum-stat"><span class="vf-sum-num">' + unique + '</span><span>unique words</span></div>';
  html += '<div class="vf-sum-stat"><span class="vf-sum-num">' + repeats + '</span><span>repeated</span></div>';
  html += '<div class="vf-sum-stat"><span class="vf-sum-num">' + intrusionSet.length + '</span><span>intrusions</span></div>';
  html += '</div>';

  if (wordOrder.length) {
    html += '<div class="vf-summary-list"><strong>Words said:</strong> ' +
      wordOrder.map(w => {
        const c = wordCount[w];
        return escHtml(w) + (c > 1 ? ' <em>(×' + c + ')</em>' : '');
      }).join(', ') + '</div>';
  }

  if (intrusionSet.length) {
    html += '<div class="vf-summary-list"><strong>Not matching ' + escHtml(cat ? cat.label : '') + ':</strong> ' +
      intrusionSet.map(escHtml).join(', ') + '</div>';
  }

  body.innerHTML = html;
  el.style.display = '';
}

function updateRing(fraction) {
  const r   = 52;
  const circ = 2 * Math.PI * r;
  const fg  = document.getElementById('vf-ring-fg');
  fg.style.strokeDashoffset = circ * (1 - fraction);
  // Color: green → yellow → red
  const hue = Math.round(fraction * 120);
  fg.style.stroke = `hsl(${hue},70%,52%)`;
}

function setStatus(msg) {
  document.getElementById('vf-status').textContent = msg;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Username ──────────────────────────────────────────────────────────────────

function initUsername() {
  fetch('/me').then(r => r.json()).then(data => {
    const name = data.username || '';
    if (!name) return;
    const el = document.getElementById('nav-user');
    if (el) el.textContent = 'Hi, ' + name;
  }).catch(() => {});
}
