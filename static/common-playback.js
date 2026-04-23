'use strict';

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
