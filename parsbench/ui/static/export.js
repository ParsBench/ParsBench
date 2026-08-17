// Client-side downloads: full run as JSON, checks as CSV, summary as Markdown.

import { pct } from './api.js';
import { failingChecks, goldenPassed, groupByGolden, overallScore } from './util.js';

function save(name, mime, content) {
  const blob = new Blob([content], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

const cell = v => {
  const s = v == null ? '' : String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
};

function toCsv(events) {
  const rows = [['test_case', 'case_index', 'run', 'check', 'score', 'passed',
                 'skipped', 'reason'].join(',')];
  for (const e of events) {
    for (const c of e.check_results) {
      rows.push([cell(e.golden_name), e.golden_index ?? '', e.run_index,
                 cell(c.check), c.score, c.passed, c.skipped,
                 cell(c.reason)].join(','));
    }
  }
  return '\uFEFF' + rows.join('\n');  // BOM so Excel reads the Persian as UTF-8
}

function toMarkdown(run, events) {
  const groups = groupByGolden(events);
  const failed = groups.filter(g => !goldenPassed(g));
  const score = run.summary && run.summary.score != null
    ? run.summary.score : overallScore(events);
  const lines = [
    `# ParsBench — ${run.app_name} (${run.kind})`,
    '',
    `- run: \`${run.run_id}\``,
    `- started: ${run.started_at}`,
    `- score: ${pct(score)}`,
    `- result: ${failed.length
      ? `${failed.length} of ${groups.length} test cases failed`
      : `all ${groups.length} test cases passed`}`,
    '',
  ];
  if (failed.length) {
    lines.push('## Failures', '');
    for (const g of failed) {
      lines.push(`### ✗ ${g.name}`, '');
      for (const c of failingChecks(g)) {
        lines.push(`- **${c.check}**${c.reason ? ` — ${c.reason}` : ''}`);
        if (c.answer) lines.push(`  - bot answered: «${c.answer}»`);
      }
      lines.push('');
    }
  }
  const passed = groups.filter(goldenPassed);
  if (passed.length) {
    lines.push('## Passed', '');
    for (const g of passed) lines.push(`- ✓ ${g.name}`);
    lines.push('');
  }
  return lines.join('\n');
}

export function downloadRun(run, events, format) {
  const stem = `parsbench-${run.run_id}`;
  if (format === 'json') {
    save(`${stem}.json`, 'application/json',
         JSON.stringify({ run, events }, null, 2));
  } else if (format === 'csv') {
    save(`${stem}.csv`, 'text/csv;charset=utf-8', toCsv(events));
  } else if (format === 'md') {
    save(`${stem}.md`, 'text/markdown', toMarkdown(run, events));
  }
}
