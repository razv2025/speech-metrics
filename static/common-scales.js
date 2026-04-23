'use strict';

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
