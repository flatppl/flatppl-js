// @flatppl/viewer — syntax highlighting for the source text the viewer shows
// outside an editor: the header expression, the info bar and the graph
// tooltip.
//
// The highlighter is the canonical TextMate one from flatppl-grammars
// (codemirror/textmate-highlight.ts), which the host loads as a script that
// publishes window.FlatPPLTextmate. Its `tok-*` classes are the playground
// editor's. The element shows plain text at once, and the runs replace it
// when the grammar and the Oniguruma wasm have loaded, so the viewer's first
// paint never waits for the wasm. Without the script the text stays plain.

interface Run { text: string; cls: string | null }
interface Textmate {
  init(assets?: { grammarUrl?: string; wasmUrl?: string }): Promise<unknown>;
  highlightText?(text: string): Run[] | null;
}

const SOURCE = new WeakMap<HTMLElement, string>();

function textmate(): Textmate | null {
  const tm = typeof window !== 'undefined' ? (window as any).FlatPPLTextmate : null;
  return tm && typeof tm.highlightText === 'function' ? tm : null;
}

function fill(el: HTMLElement, runs: Run[]): void {
  el.textContent = '';
  for (const run of runs) {
    if (!run.cls) { el.appendChild(document.createTextNode(run.text)); continue; }
    const span = document.createElement('span');
    span.className = run.cls;
    span.textContent = run.text;
    el.appendChild(span);
  }
}

/** Show FlatPPL source `text` in `el`, highlighted as soon as the grammar is
 *  ready. A later call on the same element supersedes a pending one. */
export function paintCode(el: HTMLElement, text: string): void {
  el.textContent = text;
  el.classList.add('fp-highlight');
  SOURCE.set(el, text);
  const tm = textmate();
  if (!tm) return;
  const runs = tm.highlightText!(text);
  if (runs) { fill(el, runs); return; }
  const config = (typeof window !== 'undefined' && window.__FLATPPL_CONFIG__) || {};
  tm.init({ grammarUrl: config.textmateGrammarUrl, wasmUrl: config.textmateWasmUrl }).then(() => {
    if (SOURCE.get(el) !== text) return;
    const ready = tm.highlightText!(text);
    if (ready) fill(el, ready);
  }, () => {});
}
