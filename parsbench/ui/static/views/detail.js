// Golden detail: expectations + checks on the left, the trace (or the
// simulation replay) on the right; one tab per repeated run.

import { html, useState } from '../vendor/preact.standalone.module.js';
import { fmt } from '../api.js';
import { groupByGolden, runPassed } from '../util.js';
import { Replay } from './replay.js';

function GoldenFields({ golden }) {
  return html`
    <div class="panel">
      <h3>what was expected</h3>
      ${Object.entries(golden).map(([key, value]) => html`
        <div class="field">
          <div class="field-name">${key}</div>
          <div class="field-value fa" dir="auto">
            ${typeof value === 'string' ? value : JSON.stringify(value, null, 1)}
          </div>
        </div>`)}
    </div>`;
}

function ToolCard({ call }) {
  return html`
    <details class="toolcard ${call.error ? 'has-error' : ''}">
      <summary>${call.name}</summary>
      <div class="tool-body">
        <div class="field-name">arguments</div>
        <pre class="fa" dir="auto">${JSON.stringify(call.arguments, null, 2)}</pre>
        ${call.result != null && html`
          <div class="field-name">result</div>
          <pre class="fa" dir="auto">${typeof call.result === 'string'
            ? call.result : JSON.stringify(call.result, null, 2)}</pre>`}
        ${call.error && html`
          <div class="field-name">error</div>
          <pre class="tool-error">${call.error}</pre>`}
      </div>
    </details>`;
}

function TraceView({ trace }) {
  const steps = (trace.messages || []).filter(m => m.role === 'assistant').length;
  return html`
    <div class="trace">
      ${(trace.messages || []).map(m => html`
        <div class="msg role-${m.role}">
          <div class="msg-role">${m.role}</div>
          ${m.content && html`<div class="bubble fa" dir="auto">${m.content}</div>`}
          ${(m.tool_calls || []).map(call => html`<${ToolCard} call=${call} />`)}
        </div>`)}
      <div class="final">
        <div class="field-name">final output</div>
        <div class="bubble final-bubble fa" dir="auto">${trace.final_output}</div>
      </div>
      <div class="chips">
        ${trace.latency != null && html`
          <span class="statchip">${fmt(trace.latency)}s</span>`}
        ${trace.cost != null && html`
          <span class="statchip">$${fmt(trace.cost, 4)}</span>`}
        <span class="statchip">${steps} step${steps === 1 ? '' : 's'}</span>
      </div>
    </div>`;
}

function CheckList({ results }) {
  return html`
    <div class="panel">
      <h3>check results</h3>
      ${results.map(c => html`
        <div class="check-row">
          <span class="chip ${c.skipped ? 'skip' : c.passed ? 'pass' : 'fail'}">
            ${c.skipped ? 'skip' : c.passed ? '✓' : '✗'}</span>
          <span class="check-name">${c.check}</span>
          ${c.reason && html`<div class="fa reason" dir="auto">${c.reason}</div>`}
        </div>`)}
    </div>`;
}

export function GoldenDetail({ data, goldenIndex }) {
  const { run, events } = data;
  const group = groupByGolden(events).find(g => g.index === goldenIndex);
  const [tab, setTab] = useState(0);
  if (!group) {
    return html`<div class="pad">golden not evaluated yet (run still going?)</div>`;
  }
  const event = group.runs[Math.min(tab, group.runs.length - 1)];
  return html`
    <div class="page">
      <div class="detail-head">
        <h2 class="fa" dir="auto">
          <span class=${runPassed(event) ? 'tick' : 'cross'}>
            ${runPassed(event) ? '✓' : '✗'}</span> ${group.name}</h2>
        ${group.runs.length > 1 && html`
          <div class="tabs">
            ${group.runs.map((e, i) => html`
              <button class="tab ${i === tab ? 'active' : ''} ${runPassed(e) ? '' : 'fail'}"
                onClick=${() => setTab(i)}>
                run ${i + 1} ${runPassed(e) ? '✓' : '✗'}</button>`)}
          </div>`}
      </div>
      <div class="detail-grid">
        <div>
          <${GoldenFields} golden=${event.golden} />
          <${CheckList} results=${event.check_results} />
        </div>
        <div>
          ${run.kind === 'simulation'
            ? html`<${Replay} event=${event} run=${run} />`
            : event.trace
              ? html`<div class="panel">
                  <h3>what the bot did</h3>
                  <${TraceView} trace=${event.trace} />
                </div>`
              : html`<div class="panel">
                  <h3>what the bot did</h3>
                  no trace — the app crashed before answering</div>`}
        </div>
      </div>
    </div>`;
}
