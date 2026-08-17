// Run page, failure-first: verdict card → failure cards with the judge's
// reason and the bot's answer inline → passed list → collapsed full table.

import { html, useState } from '../vendor/preact.standalone.module.js';
import { fmt, pct } from '../api.js';
import { cellFor, checkColumns, duration, failingChecks, goldenPassed,
         groupByGolden, overallScore, passHatK, runPassed,
         skippedChecks } from '../util.js';
import { Charts } from './charts.js';

const clip = (s, n = 220) =>
  s && s.length > n ? s.slice(0, n) + '…' : s;

function Zone({ title, children }) {
  return html`
    <section class="zone">
      <div class="zone-head">${title}</div>
      ${children}
    </section>`;
}

function Verdict({ run, groups, events }) {
  const failed = groups.filter(g => !goldenPassed(g)).length;
  const running = run.status === 'running';
  const crashed = run.status === 'crashed';
  const total = (run.n_goldens || 0) * (run.n_runs || 1);
  const score = run.summary && run.summary.score != null
    ? run.summary.score : overallScore(events);

  const icon = running ? '…' : crashed || failed ? '✗' : '✓';
  const cls = running ? 'running' : crashed || failed ? 'fail' : 'pass';
  const title = running
    ? `evaluating ${run.app_name}`
    : crashed ? `${run.app_name} crashed`
    : failed ? `${run.app_name} failed`
    : `${run.app_name} passed`;
  const sub = running
    ? `${events.length} of ${total} results in — they appear below as they finish`
    : failed
      ? `${failed} of ${groups.length} test case${groups.length === 1 ? '' : 's'} failed`
      : `all ${groups.length} test case${groups.length === 1 ? '' : 's'} passed`;

  return html`
    <div class="card">
      <div class="verdict">
        <div class="verdict-icon ${cls}">${icon}</div>
        <div class="verdict-text">
          <h1>${title}</h1>
          <div class="verdict-sub">${sub}
            ${run.kind === 'simulation' && html`
              <span class="dotsep">·</span> simulated conversations`}
            ${duration(run) && html`<span class="dotsep">·</span> took ${duration(run)}`}
          </div>
        </div>
        <div class="stats">
          <div class="stat">
            <div class="stat-value">${pct(score)}</div>
            <div class="stat-label">score</div>
          </div>
          ${(run.n_runs || 1) > 1 && html`
            <div class="stat">
              <div class="stat-value">${pct(passHatK(groups))}</div>
              <div class="stat-label">consistency (pass^${run.n_runs})</div>
            </div>`}
        </div>
      </div>
      ${run.status === 'running' && html`
        <div class="progress">
          <div class="progress-fill"
            style=${`width: ${total ? Math.min(100, events.length / total * 100) : 0}%`}>
          </div>
        </div>`}
    </div>`;
}

function FailureCard({ run, group }) {
  const checks = failingChecks(group);
  const flakyFails = group.runs.filter(e => !runPassed(e)).length;
  const link = `#/run/${encodeURIComponent(run.run_id)}/golden/${group.index}`;
  const isSim = run.kind === 'simulation';
  const answer = checks.find(c => c.answer)?.answer;
  return html`
    <div class="fail-card">
      <div class="fail-card-head">
        <span class="fa fail-name" dir="auto">✗ ${group.name}</span>
        ${group.runs.length > 1 && html`
          <span class="badge trap">failed ${flakyFails} of ${group.runs.length} runs</span>`}
      </div>
      <div class="fail-card-body">
        ${checks.map(c => html`
          <div class="fail-check">
            <span class="fail-check-name">${c.check}</span>
            ${c.reason && html`<div class="fa reason" dir="auto">${clip(c.reason)}</div>`}
          </div>`)}
        ${!isSim && answer && html`
          <div class="answer">
            <span class="answer-label">bot answered:</span>
            <span class="fa" dir="auto">«${clip(answer)}»</span>
          </div>`}
        <a class="trace-btn" href=${link}>
          ${isSim ? 'view conversation →' : 'view full trace →'}</a>
      </div>
    </div>`;
}

function AllChecks({ run, groups }) {
  const cols = checkColumns(groups);
  const [sel, setSel] = useState(null);
  const runLink = i => `#/run/${encodeURIComponent(run.run_id)}/golden/${i}`;
  return html`
    <details class="card allchecks">
      <summary>all checks — every test case × every check</summary>
      <div class="card-body">
        <table class="matrix">
          <thead>
            <tr><th>test case</th>${cols.map(c => html`<th>${c}</th>`)}</tr>
          </thead>
          <tbody>
            ${groups.map(g => html`
              <tr>
                <td><a class="fa golden-link" dir="auto" href=${runLink(g.index)}>
                  ${g.name}</a></td>
                ${cols.map(c => {
                  const cell = cellFor(g, c);
                  if (!cell) return html`<td></td>`;
                  const label = cell.state === 'skip' ? 'skip'
                    : cell.state === 'flaky' ? `${cell.passed}/${cell.total}`
                    : cell.state === 'pass' ? '✓' : '✗';
                  return html`<td>
                    <button class="chip ${cell.state}" title="click for details"
                      onClick=${() => setSel({ group: g, check: c, cell })}>
                      ${label}</button></td>`;
                })}
              </tr>`)}
          </tbody>
        </table>
      </div>
      ${sel && html`
        <div class="slideover-backdrop" onClick=${() => setSel(null)}>
          <div class="slideover" onClick=${e => e.stopPropagation()}>
            <div class="slideover-head">
              <strong class="fa" dir="auto">${sel.group.name} — ${sel.check}</strong>
              <button class="close" onClick=${() => setSel(null)}>×</button>
            </div>
            ${sel.cell.results.map((c, i) => html`
              <div class="slideover-run">
                <div class="slideover-run-head">
                  run ${i + 1} — ${c.skipped ? 'skipped' : c.passed ? 'passed' : 'failed'}
                  ${c.skipped ? '' : ` (${fmt(c.score)})`}
                </div>
                <div class="fa reason" dir="auto">${c.reason || '—'}</div>
              </div>`)}
          </div>
        </div>`}
    </details>`;
}

export function RunOverview({ data, runs }) {
  const { run, events } = data;
  const groups = groupByGolden(events);
  const failedGroups = groups.filter(g => !goldenPassed(g));
  const passedGroups = groups.filter(goldenPassed);
  const skipped = skippedChecks(events);

  return html`
    <div class="page">
      <${Verdict} run=${run} groups=${groups} events=${events} />

      ${run.status === 'crashed' && run.summary && run.summary.error && html`
        <div class="error-banner">${run.summary.error}</div>`}

      ${failedGroups.length > 0 && html`
        <${Zone} title=${`failures (${failedGroups.length})`}>
          ${failedGroups.map(g => html`
            <${FailureCard} key=${g.key} run=${run} group=${g} />`)}
        </${Zone}>`}

      ${passedGroups.length > 0 && html`
        <${Zone} title=${`passed (${passedGroups.length})`}>
          <div class="card">
            ${passedGroups.map(g => html`
              <a class="passed-line" key=${g.key}
                 href=${`#/run/${encodeURIComponent(run.run_id)}/golden/${g.index}`}>
                <span><span class="tick">✓</span>
                  <span class="fa" dir="auto">${g.name}</span></span>
                <span class="line-hint">
                  ${run.kind === 'simulation' ? 'view conversation →' : 'view trace →'}</span>
              </a>`)}
          </div>
        </${Zone}>`}

      ${skipped.length > 0 && html`
        <div class="skip-note">
          ${skipped.join(', ')} skipped — no judge configured.
          Set <code>PARSBENCH_JUDGE</code> (or pass <code>judge=</code>) to score them.
        </div>`}

      ${groups.length > 0 && html`
        <${Zone} title="details">
          <${Charts} run=${run} events=${events} runs=${runs || []} />
          <div class="zone-gap"></div>
          <${AllChecks} run=${run} groups=${groups} />
        </${Zone}>`}
    </div>`;
}
