/* ── Hangman game ── */

const SERVER_URL = 'https://98.87.120.209:8765';

const WORDS = [
  'APPLE','BEACH','BLACK','BRAVE','BREAD','BREAK','BRING','BROWN',
  'BRUSH','BUILD','CABIN','CANDY','CARRY','CATCH','CHAIN','CHAIR',
  'CHARM','CHEAP','CHECK','CHILD','CLEAN','CLEAR','CLIMB','CLOCK',
  'CLOSE','CLOUD','COAST','COUNT','COVER','CRACK','CRASH','CREAM',
  'CROSS','DANCE','DEPTH','DRAFT','DRAIN','DREAM','DRINK','DRIVE',
  'EAGLE','EARTH','EIGHT','EMPTY','EVENT','EXTRA','FAITH','FANCY',
  'FEAST','FIELD','FIGHT','FINAL','FIRST','FLAME','FLASH','FLOAT',
  'FLOOD','FLOOR','FOCUS','FORCE','FOUND','FRAME','FRESH','FRONT',
  'FROST','FRUIT','GIANT','GRACE','GRAIN','GRAND','GRAPE','GRASS',
  'GRAVE','GREAT','GREEN','GRIEF','GRIND','GROUP','GUESS','HAPPY',
  'HEART','HEAVY','HONEY','HONOR','HOTEL','HOUSE','HUMAN','IMAGE',
  'JUICE','KNIFE','KNOCK','LARGE','LAUGH','LEARN','LEMON','LEVEL',
  'LIGHT','LIMIT','LOCAL','LUCKY','MAGIC','MAJOR','MAPLE','MATCH',
  'MIGHT','MONEY','MONTH','MUSIC','NEVER','NIGHT','NOBLE','NORTH',
  'NOVEL','NURSE','OCEAN','OFFER','ORDER','PANEL','PARTY','PEACE',
  'PHONE','PIECE','PILOT','PLACE','PLAIN','PLANT','PLATE','POINT',
  'POWER','PRESS','PRICE','PRIDE','PRIME','PRINT','PRIZE','PROUD',
  'PROVE','QUEEN','QUICK','QUIET','QUOTE','RADIO','RAISE','RANGE',
  'RAPID','REACH','READY','REBEL','RELAX','REPLY','RIGHT','RIVER',
  'ROUND','ROYAL','SAINT','SAUCE','SCALE','SCENE','SCORE','SENSE',
  'SHAKE','SHAPE','SHARE','SHARK','SHARP','SHELF','SHIFT','SHIRT',
  'SHOCK','SHORE','SHORT','SHOUT','SIGHT','SINCE','SKILL','SLEEP',
  'SMALL','SMART','SMILE','SMOKE','SOLID','SOLVE','SOUTH','SPACE',
  'SPEAK','SPEED','SPEND','SPINE','SPORT','STACK','STAND','STARE',
  'START','STEAL','STEAM','STONE','STORE','STORM','STORY','STUDY',
  'STYLE','SUNNY','SWEET','SWIFT','SWING','SWORD','TABLE','TASTE',
  'TEACH','TEETH','THORN','THREE','THROW','TIGER','TIGHT','TITLE',
  'TOPIC','TOTAL','TOUCH','TOUGH','TOWER','TRACE','TRACK','TRADE',
  'TRAIL','TRAIN','TREND','TRIAL','TRICK','TRUST','TRUTH','TULIP',
  'UNION','UNTIL','UPPER','UPSET','VALID','VALUE','VERSE','VISIT',
  'VITAL','VOICE','WASTE','WATCH','WATER','WEIRD','WHITE','WHOLE',
  'WORLD','WORRY','WORTH','WRITE','YACHT','YIELD','YOUTH','ZEBRA',
];

const PHONETIC = {
  'alpha':'A','bravo':'B','charlie':'C','delta':'D','echo':'E',
  'foxtrot':'F','golf':'G','hotel':'H','india':'I','juliet':'J',
  'kilo':'K','lima':'L','mike':'M','november':'N','oscar':'O',
  'papa':'P','quebec':'Q','romeo':'R','sierra':'S','tango':'T',
  'uniform':'U','victor':'V','whiskey':'W','x-ray':'X','xray':'X',
  'yankee':'Y','zulu':'Z',
  'ay':'A','bee':'B','see':'C','sea':'C','dee':'D',
  'eff':'F','gee':'G','aitch':'H','jay':'J','kay':'K',
  'el':'L','em':'M','en':'N','oh':'O','pee':'P',
  'cue':'Q','queue':'Q','are':'R','ess':'S','tee':'T',
  'you':'U','vee':'V','ex':'X','why':'Y','wye':'Y',
  'zee':'Z','zed':'Z','double you':'W',
};

const BODY_PARTS = ['hm-head','hm-body','hm-larm','hm-rarm','hm-lleg','hm-rleg','hm-face'];
const MAX_WRONG = 7;

let currentWord = '';
let guessedLetters = new Set();
let wrongLetters = [];
let gameOver = false;
let listening = false;

// VAD state
let audioCtx = null;
let mediaStream = null;
let scriptNode = null;
let sourceNode = null;

let pcmChunks = [];
let isSpeaking = false;
let silenceTimer = null;
let speechStartTime = 0;
const RMS_THRESHOLD = 0.015;
const SILENCE_MS = 850;
const MIN_SPEECH_MS = 120;

function newGame() {
  stopListening();
  currentWord = WORDS[Math.floor(Math.random() * WORDS.length)];
  guessedLetters = new Set();
  wrongLetters = [];
  gameOver = false;

  BODY_PARTS.forEach(id => {
    document.getElementById(id).setAttribute('opacity', '0');
  });

  document.getElementById('hm-result-banner').style.display = 'none';
  document.getElementById('hm-transcript-last').textContent = '';
  document.getElementById('hm-status-msg').textContent = '';

  renderWord();
  renderWrong();
  updateChances();
}

function renderWord() {
  const el = document.getElementById('hm-word-display');
  el.innerHTML = currentWord.split('').map(ch => {
    const revealed = guessedLetters.has(ch);
    return `<span class="hm-letter${revealed ? ' revealed' : ''}">${revealed ? ch : ''}</span>`;
  }).join('');
}

function renderWrong() {
  const el = document.getElementById('hm-wrong-letters');
  if (wrongLetters.length === 0) {
    el.textContent = '—';
  } else {
    el.innerHTML = wrongLetters.map(l => `<span class="hm-wrong-letter">${l}</span>`).join('');
  }
}

function updateChances() {
  const remaining = MAX_WRONG - wrongLetters.length;
  document.getElementById('hm-chances-num').textContent = remaining;
}

function showBodyPart(index) {
  if (index < BODY_PARTS.length) {
    document.getElementById(BODY_PARTS[index]).setAttribute('opacity', '1');
  }
}

function checkWin() {
  return currentWord.split('').every(ch => guessedLetters.has(ch));
}

function handleGuess(letter) {
  if (gameOver) return;
  letter = letter.toUpperCase();
  if (!/^[A-Z]$/.test(letter)) return;
  if (guessedLetters.has(letter) || wrongLetters.includes(letter)) {
    setStatus(`Already guessed: ${letter}`);
    return;
  }

  if (currentWord.includes(letter)) {
    guessedLetters.add(letter);
    renderWord();
    if (checkWin()) {
      endGame(true);
    } else {
      setStatus(`✓ "${letter}" is in the word!`);
    }
  } else {
    wrongLetters.push(letter);
    showBodyPart(wrongLetters.length - 1);
    renderWrong();
    updateChances();
    if (wrongLetters.length >= MAX_WRONG) {
      endGame(false);
    } else {
      setStatus(`✗ "${letter}" is not in the word.`);
    }
  }
}

function handleWordGuess(word) {
  if (gameOver) return;
  if (word.toUpperCase() === currentWord) {
    // Reveal all letters and win
    currentWord.split('').forEach(ch => guessedLetters.add(ch));
    renderWord();
    endGame(true);
  } else {
    // Wrong word guess counts as one wrong guess
    wrongLetters.push('?');
    showBodyPart(wrongLetters.length - 1);
    updateChances();
    renderWrong();
    if (wrongLetters.length >= MAX_WRONG) {
      endGame(false);
    } else {
      setStatus(`✗ Not the word.`);
    }
  }
}

function endGame(won) {
  gameOver = true;
  stopListening();
  const banner = document.getElementById('hm-result-banner');
  banner.style.display = 'block';
  if (won) {
    banner.className = 'hm-result-win';
    banner.innerHTML = `<span>🎉</span> You got it — <strong>${currentWord}</strong>`;
  } else {
    banner.className = 'hm-result-lose';
    banner.innerHTML = `<span>💀</span> The word was <strong>${currentWord}</strong>`;
    // Show full word
    currentWord.split('').forEach(ch => guessedLetters.add(ch));
    renderWord();
  }
  setStatus('');
}

function setStatus(msg) {
  document.getElementById('hm-status-msg').textContent = msg;
}

function setTranscript(text) {
  document.getElementById('hm-transcript-last').textContent = text ? `Heard: "${text}"` : '';
}

// ── Transcript → game action ──────────────────────────────────────────────

function processTranscript(raw) {
  const t = raw.trim().toLowerCase().replace(/[.,!?;:'"]/g, '');
  if (!t) return;

  setTranscript(raw.trim());

  // Full word match
  if (t === currentWord.toLowerCase()) {
    handleWordGuess(currentWord);
    return;
  }

  // "the letter X" or "letter X"
  const letterPat = t.match(/\bletter\s+([a-z])\b/);
  if (letterPat) {
    handleGuess(letterPat[1]);
    return;
  }

  // Phonetic map — try longest key first
  const sorted = Object.keys(PHONETIC).sort((a, b) => b.length - a.length);
  for (const key of sorted) {
    if (t === key || t.startsWith(key + ' ') || t.endsWith(' ' + key) || t.includes(' ' + key + ' ')) {
      handleGuess(PHONETIC[key]);
      return;
    }
  }

  // Single letter utterance
  if (/^[a-z]$/.test(t)) {
    handleGuess(t);
    return;
  }

  // "the word is X" pattern
  const wordPat = t.match(/\b(?:the\s+word\s+is\s+|i\s+guess\s+|i\s+think\s+it(?:'s|\s+is)\s+)([a-z]+)/);
  if (wordPat) {
    const guess = wordPat[1].toUpperCase();
    if (guess.length > 1) { handleWordGuess(guess); return; }
    handleGuess(guess);
    return;
  }

  setStatus(`Didn't catch a letter — try again`);
}

// ── VAD / Recording ───────────────────────────────────────────────────────

function pcmToWav(pcm, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * bitsPerSample / 8;
  const blockAlign = numChannels * bitsPerSample / 8;
  const dataSize = pcm.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);
  for (let i = 0; i < pcm.length; i++) view.setInt16(44 + i * 2, pcm[i], true);
  return new Blob([buffer], { type: 'audio/wav' });
}

async function sendClip(pcm, sampleRate) {
  if (pcm.length === 0) return;
  const wav = pcmToWav(pcm, sampleRate);
  const fd = new FormData();
  fd.append('file', wav, 'clip.wav');
  try {
    const res = await fetch(`${SERVER_URL}/hangman/guess`, { method: 'POST', body: fd });
    const data = await res.json();
    if (data.transcript) processTranscript(data.transcript);
  } catch (e) {
    setStatus('Server error — is it running?');
  }
}

async function startListening() {
  if (listening) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch {
    setStatus('Microphone access denied.');
    return;
  }

  audioCtx = new AudioContext();
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);

  pcmChunks = [];
  isSpeaking = false;

  scriptNode.onaudioprocess = (e) => {
    if (!listening) return;
    const float32 = e.inputBuffer.getChannelData(0);
    // RMS
    let sumSq = 0;
    for (let i = 0; i < float32.length; i++) sumSq += float32[i] * float32[i];
    const rms = Math.sqrt(sumSq / float32.length);

    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));

    if (rms > RMS_THRESHOLD) {
      if (!isSpeaking) {
        isSpeaking = true;
        speechStartTime = Date.now();
        pcmChunks = [];
      }
      if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
      pcmChunks.push(int16);
    } else if (isSpeaking) {
      pcmChunks.push(int16);
      if (!silenceTimer) {
        silenceTimer = setTimeout(() => {
          silenceTimer = null;
          isSpeaking = false;
          const dur = Date.now() - speechStartTime;
          if (dur >= MIN_SPEECH_MS && pcmChunks.length > 0) {
            // Merge chunks
            const total = pcmChunks.reduce((s, c) => s + c.length, 0);
            const merged = new Int16Array(total);
            let off = 0;
            pcmChunks.forEach(c => { merged.set(c, off); off += c.length; });
            pcmChunks = [];
            sendClip(merged, audioCtx.sampleRate);
          }
        }, SILENCE_MS);
      }
    }
  };

  sourceNode.connect(scriptNode);
  scriptNode.connect(audioCtx.destination);
  listening = true;

  const btn = document.getElementById('hm-listen-btn');
  btn.className = 'record-btn recording';
  document.getElementById('hm-listen-label').textContent = 'Listening…';
}

function stopListening() {
  if (!listening) return;
  listening = false;
  if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
  if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
  if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  pcmChunks = [];
  isSpeaking = false;

  const btn = document.getElementById('hm-listen-btn');
  btn.className = 'record-btn idle';
  document.getElementById('hm-listen-label').textContent = 'Start Listening';
}

function toggleListening() {
  if (gameOver) return;
  if (listening) stopListening();
  else startListening();
}

// ── Init ──────────────────────────────────────────────────────────────────
newGame();
