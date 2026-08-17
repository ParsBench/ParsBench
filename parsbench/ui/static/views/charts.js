// Optional diagrams (collapsed until opened): score-by-check bars and the
// app's score history. Hand-rolled SVG — no chart library, no dependencies.

import { html } from '../vendor/preact.standalone.module.js';
import { pct } from '../api.js';
import { checkMeans, relTime } from '../util.js';

function BarChart({ means }) {
  const entries = [...means.entries()];
  if (!entries.length) return html`<p class="muted">no scored checks yet.</p>`;
  const W = 420, RH = 26, LW = 140;
  return html`
    <svg class="chart" viewBox=${`0 0 ${W} ${entries.length * RH}`}
         role="img" aria-label="mean score per check">
      ${entries.map(([name, v], i) => {
        const w = Math.max(2, v * (W - LW - 52));
        const color = v >= 0.75 ? 'var(--teal)' : v >= 0.4 ? 'var(--amber)' : 'var(--red)';
        return html`<g transform=${`translate(0 ${i * RH})`}>
          <text x="0" y="17" class="chart-label">${name}</text>
          <rect x=${LW} y="6" width=${w} height="14" rx="3"
                fill=${color} opacity="0.85" />
          <text x=${LW + w + 6} y="17" class="chart-value">${pct(v)}</text>
        </g>`;
      })}
    </svg>`;
}

function HistoryChart({ run, runs }) {
  // oldest → newest, finished runs of the same app only
  const series = runs
    .filter(r => r.app_name === run.app_name && r.summary && r.summary.score != null)
    .slice().reverse();
  if (series.length < 2) {
    return html`<p class="muted">
      need at least two finished runs of ${run.app_name} to draw a trend.</p>`;
  }
  const W = 420, H = 130, P = 26;
  const x = i => P + i * (W - 2 * P) / (series.length - 1);
  const y = v => H - P - v * (H - 2 * P);
  const points = series.map((r, i) => `${x(i)},${y(r.summary.score)}`).join(' ');
  return html`
    <svg class="chart" viewBox=${`0 0 ${W} ${H}`}
         role="img" aria-label="score over runs">
      ${[0, 0.5, 1].map(v => html`
        <line x1=${P} x2=${W - P} y1=${y(v)} y2=${y(v)} class="chart-grid-line" />
        <text x="0" y=${y(v) + 4} class="chart-label">${pct(v)}</text>`)}
      <polyline points=${points} fill="none" stroke="var(--teal)" stroke-width="2" />
      ${series.map((r, i) => html`
        <a href=${'#/run/' + encodeURIComponent(r.run_id)}>
          <circle cx=${x(i)} cy=${y(r.summary.score)} r="4.5"
                  fill=${r.run_id === run.run_id ? 'var(--teal-bright)' : 'var(--teal)'}
                  stroke="var(--bg)" stroke-width="1.5" />
          <title>${relTime(r.started_at)} — ${pct(r.summary.score)}</title>
        </a>`)}
    </svg>`;
}

export function Charts({ run, events, runs }) {
  return html`
    <details class="card allchecks">
      <summary>charts — score by check · history</summary>
      <div class="card-body charts-grid">
        <div>
          <div class="chart-title">score by check</div>
          <${BarChart} means=${checkMeans(events)} />
        </div>
        <div>
          <div class="chart-title">score history — ${run.app_name}</div>
          <${HistoryChart} run=${run} runs=${runs} />
        </div>
      </div>
    </details>`;
}
