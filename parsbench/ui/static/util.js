// Client-side aggregation over recorded events. Grouping key is
// golden_index (falls back to name for stores written before the addenda).

export const runPassed = e =>
  e.check_results.every(c => c.skipped || c.passed);

export function groupByGolden(events) {
  const map = new Map();
  for (const e of events) {
    const key = e.golden_index ?? e.golden_name;
    if (!map.has(key)) {
      map.set(key, { key, name: e.golden_name, index: e.golden_index ?? 0, runs: [] });
    }
    map.get(key).runs.push(e);
  }
  for (const g of map.values()) g.runs.sort((a, b) => a.run_index - b.run_index);
  return [...map.values()].sort((a, b) => a.index - b.index);
}

export function checkColumns(groups) {
  const cols = [];
  for (const g of groups)
    for (const e of g.runs)
      for (const c of e.check_results)
        if (!cols.includes(c.check)) cols.push(c.check);
  return cols;
}

export function cellFor(group, check) {
  const results = group.runs
    .map(e => e.check_results.find(c => c.check === check))
    .filter(Boolean);
  if (!results.length) return null;
  const live = results.filter(c => !c.skipped);
  if (!live.length) return { state: 'skip', score: null, results };
  const passed = live.filter(c => c.passed).length;
  const score = live.reduce((s, c) => s + c.score, 0) / live.length;
  const state = passed === live.length ? 'pass' : passed === 0 ? 'fail' : 'flaky';
  return { state, score, passed, total: live.length, results };
}

export function overallScore(events) {
  const scores = [];
  for (const e of events)
    for (const c of e.check_results)
      if (!c.skipped) scores.push(c.score);
  return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
}

export const goldenPassed = group =>
  group.runs.length > 0 && group.runs.every(runPassed);

export const passHatK = groups =>
  groups.length ? groups.filter(goldenPassed).length / groups.length : null;

export function duration(run) {
  if (!run.started_at || !run.finished_at) return null;
  const s = (new Date(run.finished_at) - new Date(run.started_at)) / 1000;
  return s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${s.toFixed(1)}s`;
}

export function relTime(iso) {
  if (!iso) return '';
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// One entry per distinct failing check of a test case, with the bot's
// answer from the run where it failed — everything a failure card needs.
export function failingChecks(group) {
  const out = [];
  for (const e of group.runs) {
    for (const c of e.check_results) {
      if (!c.skipped && !c.passed && !out.some(o => o.check === c.check)) {
        out.push({ ...c, run_index: e.run_index,
                   answer: e.trace ? e.trace.final_output : null });
      }
    }
  }
  return out;
}

export function checkMeans(events) {
  const sums = new Map();
  for (const e of events) {
    for (const c of e.check_results) {
      if (c.skipped) continue;
      if (!sums.has(c.check)) sums.set(c.check, { total: 0, n: 0 });
      const s = sums.get(c.check);
      s.total += c.score;
      s.n += 1;
    }
  }
  return new Map([...sums].map(([k, v]) => [k, v.total / v.n]));
}

export function skippedChecks(events) {
  const names = new Set();
  for (const e of events)
    for (const c of e.check_results)
      if (c.skipped) names.add(c.check);
  return [...names];
}
