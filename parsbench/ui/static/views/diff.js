// Compare two runs: per-golden pass flips first (regressions on top),
// then per-check mean deltas — the UI twin of AppEvaluationResult.diff().

import { html, useEffect, useState } from '../vendor/preact.standalone.module.js';
import { fetchRun, pct } from '../api.js';
import { checkMeans, goldenPassed, groupByGolden, relTime } from '../util.js';

const goldenMap = events =>
  new Map(groupByGolden(events).map(g => [g.name, goldenPassed(g)]));

export function DiffView({ data, baseId }) {
  const [base, setBase] = useState(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    fetchRun(baseId).then(setBase).catch(() => setError(true));
  }, [baseId]);
  if (error) return html`<div class="pad">baseline run not found.</div>`;
  if (!base) return html`<div class="pad">loading baseline…</div>`;

  const cur = checkMeans(data.events);
  const old = checkMeans(base.events);
  const checks = [...new Set([...cur.keys(), ...old.keys()])].sort();
  const curG = goldenMap(data.events);
  const oldG = goldenMap(base.events);
  const names = [...new Set([...curG.keys(), ...oldG.keys()])];
  const regressions = names.filter(n => oldG.get(n) === true && curG.get(n) === false);
  const fixes = names.filter(n => oldG.get(n) === false && curG.get(n) === true);
  const added = names.filter(n => !oldG.has(n));
  const removed = names.filter(n => !curG.has(n));

  const verdict = regressions.length
    ? `✗ ${regressions.length} regression${regressions.length === 1 ? '' : 's'} since the baseline`
    : fixes.length
      ? `✓ no regressions — ${fixes.length} test case${fixes.length === 1 ? '' : 's'} fixed`
      : '✓ no test cases changed outcome';

  return html`
    <div class="page">
      <div class="card verdict-slim">
        <div class="eyebrow">this run vs ${base.run.app_name} — ${relTime(base.run.started_at)}</div>
        <h2 class=${regressions.length ? 'diff-bad' : 'diff-good'}>${verdict}</h2>
      </div>

      ${regressions.length > 0 && html`
        <div class="panel diff-block regressions">
          <h3>regressions (${regressions.length})</h3>
          ${regressions.map(n => html`<div class="fa flip fail" dir="auto">✗ ${n}</div>`)}
        </div>`}
      ${fixes.length > 0 && html`
        <div class="panel diff-block">
          <h3>fixed (${fixes.length})</h3>
          ${fixes.map(n => html`<div class="fa flip pass" dir="auto">✓ ${n}</div>`)}
        </div>`}
      ${(added.length > 0 || removed.length > 0) && html`
        <div class="panel diff-block">
          <h3>suite changes</h3>
          ${added.map(n => html`<div class="fa flip" dir="auto">+ ${n}</div>`)}
          ${removed.map(n => html`<div class="fa flip" dir="auto">− ${n}</div>`)}
        </div>`}
      <div class="panel">
      <h3>score per check</h3>
      <table class="matrix">
        <thead>
          <tr><th>check</th><th>baseline</th><th>this run</th><th>change</th></tr>
        </thead>
        <tbody>
          ${checks.map(c => {
            const a = old.get(c);
            const b = cur.get(c);
            const delta = a != null && b != null ? b - a : null;
            return html`<tr>
              <td>${c}</td>
              <td>${pct(a)}</td>
              <td>${pct(b)}</td>
              <td class=${delta > 0 ? 'delta-up' : delta < 0 ? 'delta-down' : ''}>
                ${delta == null ? '—'
                  : (delta >= 0 ? '+' : '−') + Math.abs(Math.round(delta * 100)) + '%'}</td>
            </tr>`;
          })}
        </tbody>
      </table>
      </div>
    </div>`;
}
