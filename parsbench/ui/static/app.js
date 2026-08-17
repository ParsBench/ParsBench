// Shell: hash router, sidebar, SSE wiring. Views live in ./views/.

import { html, render, useEffect, useRef, useState }
  from './vendor/preact.standalone.module.js';
import { fetchRun, fetchRuns, subscribe } from './api.js';
import { downloadRun } from './export.js';
import { relTime } from './util.js';

// theme: dark is the default; "light" persists across sessions
const savedTheme = localStorage.getItem('parsbench-theme');
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
import { RunOverview } from './views/overview.js';
import { GoldenDetail } from './views/detail.js';
import { DiffView } from './views/diff.js';

function parseHash() {
  const parts = location.hash.replace(/^#\/?/, '').split('/')
    .filter(Boolean).map(decodeURIComponent);
  if (parts[0] === 'run' && parts[2] === 'golden')
    return { view: 'golden', runId: parts[1], goldenIndex: +parts[3] };
  if (parts[0] === 'run') return { view: 'run', runId: parts[1] };
  if (parts[0] === 'diff')
    return { view: 'diff', runId: parts[1], baseId: parts[2] };
  return { view: 'home' };
}

// the Pb mark inlined so the "P" follows the theme (white-on-dark logo.svg
// would vanish on the light sidebar); the "b" stays ParsBench teal
function Logo({ size = 26 }) {
  return html`
    <svg viewBox="0 0 512 512" width=${size} height=${size} aria-hidden="true">
      <path fill="var(--text)" d="m100.7 412l-96 7.2v-319.2h48l17.6 24q21.2-31.2 78.4-31.2 28.8 0 48.4 10.4 19.6 10.4 31.6 28 12.4 17.2 17.6 40.4 5.6 23.2 5.6 48.4 0 24.4-4.8 46-4.8 21.6-16 38.4-11.2 16.4-29.6 26-18.4 9.6-45.6 9.6h-55.2zm0-231.2v96.4h14.8q17.6 0 25.2-3.6 8-3.6 8-16.8v-94h-14.8q-10.4 0-16.8 0.8-6.4 0.8-11.2 5.2-4.8 4-5.2 12z"/>
      <path fill="var(--teal)" d="m260.2 400v-300l96-7.2v79.6q17.6-7.6 48-7.6 28.8 0 48.4 10.4 19.6 10.4 31.6 28 12.4 17.2 17.6 40.4 5.6 23.2 5.6 48.4 0 67.6-31.2 97.6-31.2 29.6-94.4 29.6-63.2 0-121.6-19.2zm96-147.2v96.4h14.8q17.6 0 25.2-3.6 8-3.6 8-16.8v-94h-14.8q-10.4 0-16.8 0.8-6.4 0.8-11.2 5.2-4.8 4-5.2 12z"/>
    </svg>`;
}

function StatusDot({ run }) {
  const cls =
    run.status === 'running' ? (run.stale ? 'stale' : 'running')
    : run.status === 'unreadable' ? 'stale' // grey, per spec §6 — not red
    : run.status === 'crashed' ? 'failed'
    : run.summary && run.summary.passed ? 'passed' : 'failed';
  return html`<span class="dot ${cls}"></span>`;
}

// what a row should say in plain words: "2/3 passed", "live", "crashed"
function runSummary(r) {
  if (r.status === 'running') return r.stale ? 'stalled' : 'running…';
  if (r.status === 'crashed') return 'crashed';
  if (r.status === 'unreadable') return 'unreadable';
  if (r.summary && r.summary.pass_rate != null && r.n_goldens) {
    return `${Math.round(r.summary.pass_rate * r.n_goldens)}/${r.n_goldens} passed`;
  }
  return '';
}

function groupRuns(runs) {
  const map = new Map();
  for (const r of runs) {
    const key = r.app_name || r.run_id;
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(r);
  }
  return [...map.entries()]; // newest app first — runs arrive newest-first
}

function Sidebar({ runs, route }) {
  return html`
    <aside class="sidebar">
      <a class="brand" href="#/">
        <${Logo} />
        <span>ParsBench</span>
      </a>
      ${groupRuns(runs).map(([app, group]) => html`
        <div class="sidebar-group">
          <div class="sidebar-app">${app}</div>
          ${group.map(r => html`
            <a key=${r.run_id} href=${'#/run/' + encodeURIComponent(r.run_id)}
               class="run-item ${route.runId === r.run_id ? 'active' : ''}">
              <${StatusDot} run=${r} />
              <span class="run-name">${runSummary(r)}</span>
              <span class="run-when">${relTime(r.started_at)}</span>
            </a>`)}
        </div>`)}
    </aside>`;
}

function StatusChip({ run }) {
  if (run.status === 'running') {
    return html`<span class="badge live">${run.stale ? 'stalled' : '● live'}</span>`;
  }
  if (run.status === 'crashed') return html`<span class="badge crash">crashed</span>`;
  const ok = run.summary && run.summary.passed;
  return html`<span class="badge ${ok ? 'ok' : 'crash'}">
    ${ok ? '✓ passed' : '✗ failed'}</span>`;
}

function ThemeToggle() {
  const [theme, setTheme] = useState(document.documentElement.dataset.theme || 'dark');
  const flip = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('parsbench-theme', next);
    setTheme(next);
  };
  return html`
    <button class="icon-btn" onClick=${flip}
      title=${theme === 'dark' ? 'switch to light theme' : 'switch to dark theme'}>
      ${theme === 'dark' ? '☀' : '☾'}</button>`;
}

function Topbar({ route, data, runs }) {
  const run = data && data.run;
  const runHref = run ? '#/run/' + encodeURIComponent(run.run_id) : '#/';
  return html`
    <div class="topbar">
      <nav class="crumbs">
        ${!run && html`<span class="crumb">runs</span>`}
        ${run && route.view === 'run' && html`
          <span class="crumb strong">${run.app_name}</span>
          <span class="crumb-sep">/</span>
          <span class="crumb">run — ${relTime(run.started_at)}</span>`}
        ${run && route.view === 'golden' && html`
          <a class="crumb strong" href=${runHref}>${run.app_name}</a>
          <span class="crumb-sep">/</span>
          <span class="crumb">${run.kind === 'simulation' ? 'conversation' : 'test case'}</span>`}
        ${run && route.view === 'diff' && html`
          <a class="crumb strong" href=${runHref}>${run.app_name}</a>
          <span class="crumb-sep">/</span>
          <span class="crumb">comparison</span>`}
      </nav>
      <div class="topbar-actions">
        ${run && html`<${StatusChip} run=${run} />`}
        ${run && route.view === 'run' && html`
          <select class="compare" onChange=${e => {
            if (e.target.value) {
              downloadRun(run, data.events, e.target.value);
              e.target.value = '';
            }
          }}>
            <option value="">download…</option>
            <option value="json">JSON — full run</option>
            <option value="csv">CSV — checks</option>
            <option value="md">Markdown — report</option>
          </select>`}
        ${run && route.view === 'run' && runs && runs.length > 1 && html`
          <select class="compare" onChange=${e => {
            if (e.target.value) {
              location.hash = `#/diff/${encodeURIComponent(run.run_id)}/` +
                encodeURIComponent(e.target.value);
            }
          }}>
            <option value="">compare with…</option>
            ${runs.filter(r => r.run_id !== run.run_id).map(r => html`
              <option value=${r.run_id}>
                ${r.app_name} — ${relTime(r.started_at)}
              </option>`)}
          </select>`}
        <${ThemeToggle} />
      </div>
    </div>`;
}

function EmptyState() {
  return html`
    <div class="empty">
      <${Logo} size="56" />
      <p>No evaluations yet. Run one and it appears here live:</p>
      <pre>from parsbench.appeval import AppEvaluator, Golden

AppEvaluator(goldens=[Golden(input="...", contains=["..."])]).evaluate(app)</pre>
    </div>`;
}

function App() {
  const [runs, setRuns] = useState([]);
  const [route, setRoute] = useState(parseHash());
  const [data, setData] = useState(null);
  const routeRef = useRef(route);
  routeRef.current = route;

  const refresh = () => fetchRuns().then(setRuns).catch(() => {});

  useEffect(() => {
    refresh();
    const onHash = () => setRoute(parseHash());
    addEventListener('hashchange', onHash);
    const es = subscribe({
      'run-started': () => refresh(),
      'run-finished': meta => {
        refresh();
        setData(d => d && d.run.run_id === meta.run_id ? { ...d, run: meta } : d);
      },
      'golden-finished': ({ run_id, event }) => {
        setData(d => {
          if (!d || d.run.run_id !== run_id) return d;
          if (d.events.some(e => e.seq === event.seq)) return d; // SSE/fetch race
          return { ...d, events: [...d.events, event] };
        });
      },
    });
    // EventSource auto-reconnects; events during the gap were never pushed,
    // so re-fetch on every (re)connect — spec §4.3
    es.onopen = () => {
      refresh();
      const r = routeRef.current;
      if (r.runId) fetchRun(r.runId).then(setData).catch(() => {});
    };
    return () => { removeEventListener('hashchange', onHash); es.close(); };
  }, []);

  // land on the newest run — never greet the user with an empty page
  useEffect(() => {
    if (route.view === 'home' && runs.length) {
      location.hash = '#/run/' + encodeURIComponent(runs[0].run_id);
    }
  }, [runs, route.view]);

  useEffect(() => {
    if (route.runId) fetchRun(route.runId).then(setData).catch(() => setData(null));
    else setData(null);
  }, [route.runId]);

  const main =
    !route.runId ? html`<${EmptyState} />`
    : !data ? html`<div class="pad">loading…</div>`
    : route.view === 'diff'
      ? html`<${DiffView} data=${data} baseId=${route.baseId} />`
    : route.view === 'golden'
      ? html`<${GoldenDetail} data=${data} goldenIndex=${route.goldenIndex} />`
    : html`<${RunOverview} data=${data} runs=${runs} />`;

  return html`
    <div class="layout">
      <${Sidebar} runs=${runs} route=${route} />
      <main class="main">
        <${Topbar} route=${route} data=${data} runs=${runs} />
        <div class="content">${main}</div>
      </main>
    </div>`;
}

render(html`<${App} />`, document.getElementById('app'));
