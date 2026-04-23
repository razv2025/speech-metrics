'use strict';

const _sortState = { 1: null, 2: null, 3: null };

function sortLogTable(task, key) {
  const cur = _sortState[task];
  if (cur && cur.key === key) {
    _sortState[task] = cur.dir === 'asc' ? { key, dir: 'desc' } : null;
  } else {
    _sortState[task] = { key, dir: 'asc' };
  }
  renderLogTable(task);
}

function _thSort(task, key, label, title) {
  const s = _sortState[task];
  const active = s && s.key === key;
  const cls = active ? ' sort-' + s.dir : '';
  const ind = active ? (s.dir === 'asc' ? ' \u001e' : ' \u001f') : '';
  const tip = title ? ' title="' + _escHtml(title) + '"' : '';
  return '<th class="sortable' + cls + '"' + tip + ' onclick="sortLogTable(' + task + ',\'' + key + '\')">' + label + ind + '</th>';
}

function _fmtTs(iso) {
  if (!iso) return '-';
  const d = new Date(iso);
  const dd = String(d.getDate()).padStart(2, '0');
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  return dd + '/' + mm + ' ' + hh + ':' + mi;
}

function renderLogTable(task) {
  _getAllLogEntries(task).then(localEntries => {
    const wrap = document.getElementById('log-table-' + task);
    const delBtn = document.getElementById('log-delete-' + task);
    const expBtn = document.getElementById('log-export-' + task);
    if (!wrap) return;

    const myId = _getUserId();
    const filter = _pubFilter[task] || 'all';
    const published = _publishedEntries[task] || [];

    let display;
    if (filter === 'mine') {
      display = localEntries.filter(e => !e.published_id);
    } else if (filter === 'published') {
      display = published;
    } else {
      const myPubIds = new Set(localEntries.filter(e => e.published_id).map(e => e.published_id));
      display = [
        ...localEntries,
        ...published.filter(e => !myPubIds.has(e.id)),
      ];
    }

    if (display.length === 0) {
      wrap.innerHTML = '<p class="log-empty">No analyses recorded yet.</p>';
      if (delBtn) delBtn.disabled = true;
      if (expBtn) expBtn.disabled = true;
      return;
    }
    if (expBtn) expBtn.disabled = filter === 'published';

    const metrics = LOG_METRICS[task];
    const sort = _sortState[task];
    if (sort) {
      display = display.slice().sort((a, b) => {
        const av = a[sort.key], bv = b[sort.key];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        const an = Number(av), bn = Number(bv);
        const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : String(av).localeCompare(String(bv));
        return sort.dir === 'asc' ? cmp : -cmp;
      });
    } else {
      display = display.slice().reverse();
    }

    const filterBtns = ['all', 'mine', 'published'].map(f => {
      const labels = { all: '?? All', mine: '?? Local', published: '?? Published' };
      const active = f === filter ? ' log-filter-btn-active' : '';
      return '<button class="log-filter-btn' + active + '" onclick="setViewFilter(' + task + ',\'' + f + '\')">' + labels[f] + '</button>';
    }).join('');

    let html = '<div class="log-filter-bar">' + filterBtns + '</div>';
    html += '<div class="log-table-wrap"><table class="log-table"><thead><tr>';
    html += '<th class="log-chk-th"><input type="checkbox" onchange="logToggleAll(this,' + task + ')" title="Select all"></th>';
    html += '<th class="log-act-th" title="Re-analyze">\u0018</th>';
    html += '<th class="log-act-th" title="Download">\u0019</th>';
    html += '<th class="log-act-th" title="Publish">??</th>';
    html += _thSort(task, 'filename', 'File');
    html += _thSort(task, 'username', 'User');
    metrics.forEach(m => { html += _thSort(task, m.key, m.label, m.title); });
    html += _thSort(task, 'timestamp', 'Time');
    html += '</tr></thead><tbody>';

    display.forEach(e => {
      const isPub = !!e._published;
      const hasPub = !isPub && e.published_id != null;
      html += isPub ? '<tr class="log-row-published">' : '<tr>';

      if (isPub) {
        html += '<td class="log-chk-td"><input type="checkbox" class="log-chk" data-pub-id="' + e.id + '" data-task="' + task + '" onchange="logChkChange(' + task + ')"></td>';
      } else {
        html += '<td class="log-chk-td"><input type="checkbox" class="log-chk" data-id="' + e.id + '" data-task="' + task + '" onchange="logChkChange(' + task + ')"></td>';
      }

      if (isPub) {
        html += '<td class="log-act-td"><button class="log-play-btn" data-pub-id="' + e.id + '" onclick="replayPublishedEntry(' + task + ',' + e.id + ')" title="Re-analyze (creates new local entry)">\u0018</button></td>';
        html += '<td class="log-act-td"><button class="log-dl-btn" onclick="window.open(\'' + SERVER_URL + '/audio/' + e.id + '\')" title="Download audio">\u0019 wav</button></td>';
        html += '<td class="log-act-td"></td>';
      } else {
        const reBtn = e.audioData
          ? '<button class="log-play-btn" data-id="' + e.id + '" data-task="' + task + '" onclick="replayLogEntry(' + task + ',' + e.id + ')" title="Re-analyze">\u0018</button>'
          : '';
        const dlLabel = task === 3 ? '\u0019 zip' : '\u0019 wav';
        const dlTitle = task === 3 ? 'Download ZIP (audio + passage)' : 'Download WAV';
        const dlBtn = e.audioData
          ? '<button class="log-dl-btn" onclick="downloadLogAudio(event,' + task + ',' + e.id + ')" title="' + dlTitle + '">' + dlLabel + '</button>'
          : '';
        const pubBtn = (e.audioData && !hasPub)
          ? '<button class="log-pub-btn" data-id="' + e.id + '" data-task="' + task + '" onclick="publishEntry(' + task + ',' + e.id + ')" title="Publish">??</button>'
          : (hasPub ? '<span class="pub-badge-inline" title="Already published">??</span>' : '');
        html += '<td class="log-act-td">' + reBtn + '</td>';
        html += '<td class="log-act-td">' + dlBtn + '</td>';
        html += '<td class="log-act-td">' + pubBtn + '</td>';
      }

      const _ver = e.version ? ' � v' + e.version : '';
      if (isPub) {
        html += '<td class="log-filename log-filename-pub" title="' + _escHtml((e.filename || '') + _ver) + '">' + _escHtml(e.filename || '-') + '</td>';
      } else {
        const fname = _escHtml(e.filename || '-');
        html += '<td class="log-filename" title="' + _escHtml((e.filename || '') + _ver + ' (click to rename)') + '" onclick="renameLogEntry(this,' + task + ',' + e.id + ')">' + fname + '</td>';
      }

      html += '<td class="log-username">' + _escHtml(e.username || '-') + '</td>';
      metrics.forEach(m => {
        const v = e[m.key];
        html += '<td>' + (v != null ? fmt(parseFloat(v), m.dec) : '-') + '</td>';
      });

      html += '<td class="log-ts">' + _fmtTs(e.timestamp) + '</td>';
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

async function deleteSelected(task) {
  const checked = [...document.querySelectorAll('.log-chk[data-task="' + task + '"]:checked')];
  if (!checked.length) return;

  const localIds = checked.filter(c => c.dataset.id).map(c => +c.dataset.id);
  const pubIds = checked.filter(c => c.dataset.pubId).map(c => +c.dataset.pubId);

  if (localIds.length && logDB) {
    await new Promise(resolve => {
      const tx = logDB.transaction('task' + task, 'readwrite');
      const store = tx.objectStore('task' + task);
      localIds.forEach(id => store.delete(id));
      tx.oncomplete = resolve;
    });
  }

  for (const pubId of pubIds) {
    try {
      await deletePublishedEntry(task, pubId);
    } catch (err) {
      console.error('Delete published failed:', err);
    }
  }

  renderLogTable(task);
}

function exportLogTSV(task) {
  _getAllLogEntries(task).then(entries => {
    if (!entries.length) return;
    const metrics = LOG_METRICS[task];
    const headers = ['filename', 'timestamp', ...metrics.map(m => m.key), 'version'];
    const lines = [headers.join('\t')];
    entries.forEach(e => {
      lines.push(headers.map(h => { const v = e[h]; return v != null ? String(v) : ''; }).join('\t'));
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/tab-separated-values' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const now = new Date();
    const ts = now.getFullYear() + '-' + String(now.getMonth()+1).padStart(2,'0') + '-'
                  + String(now.getDate()).padStart(2,'0') + '_'
                  + String(now.getHours()).padStart(2,'0') + '-' + String(now.getMinutes()).padStart(2,'0');
    a.download = TASK_SLUG[task] + '_' + ts + '.tsv';
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
      blob = await zip.generateAsync({ type: 'blob' });
      filename = base + '.zip';
    } else {
      blob = new Blob([entry.audioData], { type: 'audio/wav' });
      filename = base + '.wav';
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
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
    const tx = logDB.transaction('task' + task, 'readwrite');
    const store = tx.objectStore('task' + task);
    const req = store.get(id);
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
