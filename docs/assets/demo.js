'use strict';

let index = 12;
const blue = '#2a6fbb', red = '#a33b3b', ink = '#1d1d1f', muted = '#6e6e73';
const gridMin = DATA.grid[0], gridMax = DATA.grid[DATA.grid.length - 1];
const posteriorYMax = Math.ceil(Math.max(...DATA.moves.flatMap(r => r.prior.concat(r.posterior))) * 100) / 100;
const moveEvidenceYMax = Math.ceil(Math.max(...DATA.moves.flatMap(r => r.rating_given_move)) * 100) / 100;
const byId = id => document.getElementById(id);
function esc(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function path(values, xs, xmin, xmax, ymin, ymax, w, h, m) {
  return values.map((y, i) => (i ? 'L' : 'M') + (m + (xs[i] - xmin) / (xmax - xmin) * (w - 2 * m)).toFixed(1) + ',' + (h - m - (y - ymin) / (ymax - ymin) * (h - 2 * m)).toFixed(1)).join(' ');
}
function chart(container, title, xs, series, ymax, yLabel, marker, secondary) {
  const w = Math.max(260, Math.round(byId(container).getBoundingClientRect().width));
  const h = 236, m = 45, parts = [];
  parts.push(`<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}">`);
  for (let i = 0; i < 4; i++) {
    const y = m + i * (h - 2 * m) / 3;
    const value = ymax * (1 - i / 3);
    parts.push(`<line x1="${m}" x2="${w - m}" y1="${y}" y2="${y}" stroke="#e8e8ed"/><text x="${m - 6}" y="${y + 4}" text-anchor="end" font-size="11" fill="${muted}">${value.toFixed(3)}</text>`);
  }
  const tickCount = w < 360 ? 3 : 5;
  for (let i = 0; i < tickCount; i++) {
    const x = m + i * (w - 2 * m) / (tickCount - 1);
    const value = gridMin + i * (gridMax - gridMin) / (tickCount - 1);
    parts.push(`<text x="${x}" y="${h - m + 18}" text-anchor="middle" font-size="12" fill="${muted}">${value.toFixed(0)}</text>`);
  }
  if (secondary) {
    parts.push(`<path d="${path(secondary, xs, gridMin, gridMax, 0, 1, w, h, m)}" fill="none" stroke="#92929a" stroke-width="1.5" opacity=".55"/>`);
    for (let i = 0; i < 3; i++) {
      const y = m + i * (h - 2 * m) / 2;
      parts.push(`<text x="${w - m + 7}" y="${y + 4}" font-size="11" fill="${muted}">${(1 - i / 2).toFixed(1)}</text>`);
    }
    parts.push(`<text x="${w - m}" y="22" text-anchor="end" font-size="11" fill="${muted}">P(move)</text>`);
  }
  series.forEach(s => parts.push(`<path d="${path(s.values, xs, gridMin, gridMax, 0, ymax, w, h, m)}" fill="none" stroke="${s.color}" stroke-width="${s.dash ? 1.8 : 2.4}" ${s.dash ? 'stroke-dasharray="5 4"' : ''} stroke-linejoin="round"/>`));
  if (marker !== null) {
    const x = m + (marker - gridMin) / (gridMax - gridMin) * (w - 2 * m);
    parts.push(`<line x1="${x}" x2="${x}" y1="${m}" y2="${h - m}" stroke="${ink}" opacity=".5" stroke-dasharray="3 4"/>`);
  }
  parts.push(`<text x="${m}" y="22" font-size="11" fill="${muted}">${esc(yLabel)}</text><text x="${w / 2}" y="${h - 4}" text-anchor="middle" font-size="12" fill="${muted}">Lichess rating</text></svg>`);
  return parts.join('');
}
function trajectorySvg(r) {
  const w = Math.max(260, Math.round(byId('trajectory').getBoundingClientRect().width));
  const h = 240, m = 43, lastMove = DATA.moves[DATA.moves.length - 1].move_number;
  const parts = [`<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="White and Black estimated ratings over all 26 moves. Shaded bands show approximately 50 percent posterior mass.">`];
  for (let i = 0; i < 5; i++) {
    const y = m + i * (h - 2 * m) / 4, v = gridMax - i * (gridMax - gridMin) / 4;
    parts.push(`<line x1="${m}" x2="${w - m}" y1="${y}" y2="${y}" stroke="#e8e8ed"/><text x="${m - 7}" y="${y + 4}" text-anchor="end" font-size="11" fill="${muted}">${v.toFixed(0)}</text>`);
  }
  function point(p, y) {
    return (m + (p.move_number - 1) / (lastMove - 1) * (w - 2 * m)).toFixed(1) + ',' + (h - m - (y - gridMin) / (gridMax - gridMin) * (h - 2 * m)).toFixed(1);
  }
  for (let move = 1; move <= lastMove; move += 5) {
    const x = m + (move - 1) / (lastMove - 1) * (w - 2 * m);
    parts.push(`<text x="${x}" y="${h - m + 18}" text-anchor="middle" font-size="11" fill="${muted}">${move}</text>`);
  }
  function band(items, color) {
    const upper = items.map(p => point(p, p.high)).join(' ');
    const lower = items.slice().reverse().map(p => point(p, p.low)).join(' ');
    return `<polygon points="${upper} ${lower}" fill="${color}" opacity=".13"/>`;
  }
  function draw(items, color) {
    const d = items.map((p, i) => (i ? 'L' : 'M') + point(p, p.mean)).join(' ');
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.3" stroke-linejoin="round"/>`;
  }
  parts.push(band(DATA.white, blue), band(DATA.black, red), draw(DATA.white, blue), draw(DATA.black, red));
  const x = m + (r.move_number - 1) / (lastMove - 1) * (w - 2 * m);
  parts.push(`<line x1="${x}" x2="${x}" y1="${m}" y2="${h - m}" stroke="${ink}" opacity=".55" stroke-dasharray="3 4"/>`);
  parts.push(`<circle cx="${x}" cy="${point(r, r.after_mean).split(',')[1]}" r="4.5" fill="${r.color}" stroke="white" stroke-width="1.5"/>`);
  if (index > 0) {
    const previous = DATA.moves[index - 1], previousPoint = point(previous, previous.after_mean).split(',');
    parts.push(`<circle cx="${previousPoint[0]}" cy="${previousPoint[1]}" r="3" fill="${previous.color}" opacity=".38"/>`);
  }
  parts.push(`<text x="${m}" y="22" font-size="11" fill="${muted}">Lichess rating</text><text x="${w / 2}" y="${h - 4}" text-anchor="middle" font-size="12" fill="${muted}">Move number</text></svg>`);
  return parts.join('');
}
function reportHeight() {
  if (window.parent !== window) {
    window.parent.postMessage({type: 'rym-demo-height', height: document.documentElement.scrollHeight}, location.origin === 'null' ? '*' : location.origin);
  }
}
function renderCharts() {
  const r = DATA.moves[index], grid = DATA.grid;
  byId('likelihood').innerHTML = chart('likelihood', `Rating evidence from ${r.side}'s ${r.san}`, grid, [{values: r.rating_given_move, color: r.color}], moveEvidenceYMax, 'Relative evidence', null, r.likelihood);
  byId('posterior').innerHTML = chart('posterior', `${r.side}'s estimated rating before and after ${r.san}`, grid, [{values: r.prior, color: muted, dash: true}, {values: r.posterior, color: r.color}], posteriorYMax, 'Relative weight', r.after_mean, null);
  byId('trajectory').innerHTML = trajectorySvg(r);
  requestAnimationFrame(reportHeight);
}
function render() {
  const r = DATA.moves[index];
  const focusedControl = document.activeElement;
  byId('start').disabled = byId('prev').disabled = index === 0;
  byId('next').disabled = byId('end').disabled = index === DATA.moves.length - 1;
  if (focusedControl instanceof HTMLButtonElement && focusedControl.disabled && focusedControl.closest('.controls')) {
    byId(index === 0 ? 'next' : 'prev').focus({preventScroll: true});
  }
  byId('move-label').textContent = `${r.move_number}${r.side === 'White' ? '.' : '…'} ${r.san}`;
  byId('move-label').setAttribute('aria-label', `Move ${r.move_number}, ${r.side}: ${r.san}`);
  byId('move-count').textContent = `${index + 1} / ${DATA.moves.length}`;
  byId('board').innerHTML = r.board_svg;
  byId('board').setAttribute('aria-label', `Position before move ${r.move_number}, ${r.side}: ${r.san}. Arrow from ${r.uci.slice(0, 2)} to ${r.uci.slice(2, 4)}.`);
  byId('estimate-label').textContent = `${r.side}’s estimated skill`;
  byId('summary').innerHTML = `<span class="before">${r.before_mean.toFixed(0)}</span><span class="arrow" aria-label="to">→</span>${r.after_mean.toFixed(0)}`;
  document.documentElement.style.setProperty('--selected', r.color);
  renderCharts();
}
byId('start').addEventListener('click', () => { index = 0; render(); });
byId('prev').addEventListener('click', () => { if (index > 0) { index--; render(); } });
byId('next').addEventListener('click', () => { if (index < DATA.moves.length - 1) { index++; render(); } });
byId('end').addEventListener('click', () => { index = DATA.moves.length - 1; render(); });
document.querySelector('.controls').addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    const nextIndex = Math.max(0, Math.min(DATA.moves.length - 1, index + (event.key === 'ArrowRight' ? 1 : -1)));
    if (nextIndex !== index) { index = nextIndex; render(); }
  }
});
let lastWidth = 0;
const resizeObserver = new ResizeObserver(() => {
  const width = byId('explorer').getBoundingClientRect().width;
  if (width > 0 && width !== lastWidth) { lastWidth = width; renderCharts(); }
  reportHeight();
});
resizeObserver.observe(byId('explorer'));
render();
