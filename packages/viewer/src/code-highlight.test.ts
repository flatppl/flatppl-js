// @ts-nocheck — Node runs tests directly; the viewer build targets the browser.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installTestDom } from './test-dom.ts';
import { paintCode } from './code-highlight.ts';

/** A stand-in for window.FlatPPLTextmate whose grammar loads when the test says. */
function fakeTextmate() {
  let loaded = false;
  const ready = Promise.withResolvers();
  const inits = [];
  return {
    inits,
    load() { loaded = true; ready.resolve(); return ready.promise.then(() => new Promise(setImmediate)); },
    init(assets) { inits.push(assets); return ready.promise; },
    highlightText(text) {
      if (!loaded) return null;
      return text.split(/(\s+)/).filter(Boolean).map((t) => ({ text: t, cls: /^\s/.test(t) ? null : 'tok-' + t }));
    },
  };
}

const spans = (el) => el.children.map((c) => c.tagName ? c.className + ':' + c.textContent : JSON.stringify(c.textContent));

test('source shows as plain text first and highlighted once the grammar loads', async () => {
  const dom = installTestDom();
  try {
    const tm = fakeTextmate();
    dom.window.FlatPPLTextmate = tm;
    dom.window.__FLATPPL_CONFIG__ = { textmateGrammarUrl: 'g.json', textmateWasmUrl: 'o.wasm' };
    const el = dom.document.createElement('div');
    paintCode(el, 'a = b');
    assert.equal(el.textContent, 'a = b');
    assert.deepEqual(spans(el), []);
    assert.deepEqual(tm.inits, [{ grammarUrl: 'g.json', wasmUrl: 'o.wasm' }]);
    await tm.load();
    assert.deepEqual(spans(el), ['tok-a:a', '" "', 'tok-=:=', '" "', 'tok-b:b']);
    assert.ok(el.classList.contains('fp-highlight'));
    // Once loaded, a new paint highlights at once.
    paintCode(el, 'c');
    assert.deepEqual(spans(el), ['tok-c:c']);
  } finally { dom.restore(); }
});

test('a later paint supersedes one still waiting for the grammar', async () => {
  const dom = installTestDom();
  try {
    const tm = fakeTextmate();
    dom.window.FlatPPLTextmate = tm;
    const el = dom.document.createElement('div');
    paintCode(el, 'old');
    paintCode(el, 'new');
    await tm.load();
    assert.deepEqual(spans(el), ['tok-new:new']);
  } finally { dom.restore(); }
});

test('without the highlighter script the source stays plain text', () => {
  const dom = installTestDom();
  try {
    const el = dom.document.createElement('div');
    paintCode(el, '<b>x</b> = 1');
    assert.equal(el.textContent, '<b>x</b> = 1');
    assert.deepEqual(spans(el), []);
  } finally { dom.restore(); }
});
