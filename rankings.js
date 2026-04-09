'use strict';
// Usage:
//   node rankings.js           → valid files only  (default)
//   node rankings.js --all     → include invalid files too

const fs   = require('fs');
const path = require('path');

const includeAll = process.argv.includes('--all');

const tasks = [
  { name: 'amplitude', file: 'results_amplitude.tsv' },
  { name: 'pitch',     file: 'results_pitch.tsv'     },
  { name: 'reading',   file: 'results_reading.tsv'   },
];

const rows = ['task\tmetric\ttop_bottom\trank\tfile_id\tvalue'];

for (const { name, file } of tasks) {
  const lines   = fs.readFileSync(path.join(__dirname, file), 'utf8').trim().split('\n');
  const headers = lines[0].split('\t').map(h => h.trim());
  const validIdx = headers.indexOf('valid');

  // Skip 'filename' and 'valid' columns
  const metricIndices = headers
    .map((h, i) => ({ h, i }))
    .filter(({ h, i }) => i > 0 && h !== 'valid');

  const data = lines.slice(1).map(l => {
    const cols = l.split('\t');
    return {
      id:    cols[0],
      valid: validIdx >= 0 ? parseInt(cols[validIdx]) : 1,
      vals:  cols.map(v => parseFloat(v)),
    };
  });

  const filtered = includeAll ? data : data.filter(d => d.valid === 1);

  for (const { h: metric, i: mi } of metricIndices) {
    const entries = filtered
      .map(d => ({ id: d.id, v: d.vals[mi] }))
      .filter(e => !isNaN(e.v));

    entries.sort((a, b) => b.v - a.v);

    const top5    = entries.slice(0, 5);
    const bottom5 = entries.slice(-5).reverse();

    top5.forEach((e, i) => {
      rows.push(`${name}\t${metric}\ttop\t${i + 1}\t${e.id}\t${e.v}`);
    });
    bottom5.forEach((e, i) => {
      rows.push(`${name}\t${metric}\tbottom\t${i + 1}\t${e.id}\t${e.v}`);
    });
  }
}

const suffix = includeAll ? ' (all files including invalid)' : ' (valid files only)';
const out = path.join(__dirname, 'rankings.tsv');
fs.writeFileSync(out, rows.join('\n') + '\n');
console.log(`Written ${rows.length - 1} ranking rows to ${out}${suffix}`);
