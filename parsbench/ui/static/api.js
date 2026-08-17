// Thin client for the parsbench view JSON API + SSE stream.

export async function fetchRuns() {
  const resp = await fetch('/api/runs');
  return resp.json();
}

export async function fetchRun(id) {
  const resp = await fetch(`/api/runs/${encodeURIComponent(id)}`);
  if (!resp.ok) throw new Error(`run ${id}: HTTP ${resp.status}`);
  return resp.json();
}

// handlers: {'run-started': fn, 'golden-finished': fn, 'run-finished': fn}
export function subscribe(handlers) {
  const es = new EventSource('/api/stream');
  for (const [name, fn] of Object.entries(handlers)) {
    es.addEventListener(name, e => fn(JSON.parse(e.data)));
  }
  return es;
}

export const fmt = (x, digits = 2) => (x == null ? '—' : Number(x).toFixed(digits));
export const pct = x => (x == null ? '—' : `${Math.round(x * 100)}%`);
