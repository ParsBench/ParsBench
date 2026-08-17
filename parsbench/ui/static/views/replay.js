// Simulation chat replay: RTL bubbles + goal/traps + play-through.

import { html, useEffect, useState } from '../vendor/preact.standalone.module.js';

function messagesOf(event) {
  if (event.trace && event.trace.messages && event.trace.messages.length) {
    return event.trace.messages.filter(m => m.content);
  }
  // stores written without structured traces: parse the rendered transcript
  return (event.transcript || '').split('\n').filter(Boolean).map(line => ({
    role: line.startsWith('کاربر:') ? 'user' : 'assistant',
    content: line.replace(/^(کاربر|دستیار):\s*/, ''),
  }));
}

export function Replay({ event, run }) {
  const messages = messagesOf(event);
  const [shown, setShown] = useState(messages.length);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    if (shown >= messages.length) { setPlaying(false); return; }
    const t = setTimeout(() => setShown(s => s + 1), 900);
    return () => clearTimeout(t);
  }, [playing, shown]);

  const user = run.user || {};
  return html`
    <div class="panel replay">
      <h3>conversation</h3>
      <div class="replay-head">
        <div>
          ${event.golden.goal && html`
            <div class="fa" dir="auto"><strong>هدف:</strong> ${event.golden.goal}</div>`}
          ${event.golden.scenario && html`
            <div class="fa" dir="auto">${event.golden.scenario}</div>`}
          ${(user.traps || []).length > 0 && html`
            <div class="traps">
              ${user.traps.map(t => html`<span class="badge trap">${t}</span>`)}
            </div>`}
        </div>
        <button class="play" onClick=${() => { setShown(0); setPlaying(true); }}>
          ▶ replay</button>
      </div>
      <div class="chat" dir="rtl">
        ${messages.slice(0, shown).map(m => html`
          <div class="chat-msg ${m.role === 'user' ? 'from-user' : 'from-bot'}">
            <div class="bubble fa" dir="auto">${m.content}</div>
          </div>`)}
      </div>
      <div class="verdicts">
        ${event.converged != null && html`
          <span class="chip ${event.converged ? 'pass' : 'fail'}">
            ${event.converged ? '✓ user reached their goal'
                              : '✗ turn cap hit before the goal'}</span>`}
      </div>
    </div>`;
}
