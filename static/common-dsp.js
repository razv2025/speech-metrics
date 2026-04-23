'use strict';

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
   UTILITY — push helper
════════════════════════════════════════════ */
function push(arr, val, cap) {
  arr.push(val);
  if (arr.length > cap) arr.shift();
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
