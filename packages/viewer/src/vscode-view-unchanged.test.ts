// @ts-nocheck — whole-mount contract tests for the VS Code webview host.
//
// The web playground's pane breakout must leave the VS Code view unchanged.
// These tests mount the viewer the way the webview does (no opts.host,
// acquireVsCodeApi defined) and compare what it builds with a golden copy:
// the DOM, the injected stylesheet, the listeners, the graph model, what each
// chart was given, the posted messages and the saved state. The session
// covers the gestures the breakout touches: Plots on, a graph tap and a math
// row click. The golden was captured with this harness on the viewer source
// from before the breakout seams existed. Regenerate it
// (FLATPPL_UPDATE_GOLDEN=1) only when the VS Code view is meant to change,
// and say so in the pull request.

import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  },
});

const { installViewerEnv, graphModel } = await import('./test-viewer-env.ts');
const { settle } = await import('./test-worker.ts');
const { mount } = await import('./main.ts');
const { defaultVscodeHost } = await import('./host-adapter.ts');

const GOLDEN = new URL('./fixtures/vscode-view.golden.json', import.meta.url);

// host-adapter.ts caches the one-shot VS Code API for the whole process, so
// every test talks to one API object and swaps the sink it records into.
const sink = { posted: [], states: [] };
const vscodeApi = {
  postMessage: (m) => sink.posted.push(JSON.parse(JSON.stringify(m))),
  setState: (s) => sink.states.push(JSON.parse(JSON.stringify(s))),
  getState: () => undefined,
};
Object.defineProperty(globalThis, 'acquireVsCodeApi', { value: () => vscodeApi, configurable: true, writable: true });
function resetSink() { sink.posted = []; sink.states = []; return sink; }

const MODEL = [
  'mu = draw(Normal(mu = 0.0, sigma = 1.0))',
  'sigma = 1.5',
  'x = draw(Normal(mu = mu, sigma = sigma))',
  '',
].join('\n');

function deliver(dom, data) {
  dom.window.dispatchEvent(new MessageEvent('message', { data, source: null, origin: 'vscode-webview://panel' }));
}

/** Settle until sampling finished: no "Sampling…" left in the plot pane. */
async function settlePlots(dom) {
  for (let i = 0; i < 50; i++) {
    await settle();
    const plot = dom.document.getElementById('plot-content');
    if (!plot || !/Sampling/.test(JSON.stringify(dom.serialize(plot)))) return;
  }
  throw new Error('sampling did not finish');
}

/** A graph tap as cytoscape reports one; the position misses every glyph. */
function tap(cy, id, keys = {}) {
  cy.$id(id).emit({ type: 'tap', position: { x: -1e9, y: -1e9 },
    originalEvent: { ctrlKey: false, metaKey: false, shiftKey: false, ...keys } });
}

/** A click on a rendered math row. Node has no wasm renderer, so the row is
 *  the markup render_math emits for a binding. */
function clickMathRow(dom, binding) {
  const content = dom.document.getElementById('math-content');
  const row = dom.document.createElement('mtr');
  row.setAttribute('data-flatppl-binding', binding);
  content.appendChild(row);
  row.dispatchEvent(Object.assign(new Event('click', { bubbles: true, cancelable: true }),
    { ctrlKey: false, metaKey: false }));
  row.remove();
}

/** The VS Code session: boot, config push, a model, panel toggles, Plots on,
 *  a graph tap, a math row click, then a whole-module update. */
async function runSession() {
  const env = installViewerEnv();
  const { dom } = env;
  const posted = resetSink();
  let view = null;
  try {
    dom.document.body.innerHTML = '<div id="flatppl-viewer-root"></div>';
    const root = dom.document.getElementById('flatppl-viewer-root');
    view = mount(root);
    const steps = {};
    const take = (name) => { steps[name] = { dom: dom.serialize(root), graph: graphModel(env.graphs[0]) }; };
    await settle();
    take('boot');
    deliver(dom, { type: 'configUpdate', config: { sampleCount: 1000, marginalizationSampleCount: 50,
      dagNavigationHistoryCap: 20, truncateRejectionBudget: 500 } });
    deliver(dom, { type: 'sourceUpdate', source: MODEL, targetName: 'x' });
    await settle();
    take('model');
    dom.document.getElementById('math-toggle').click();
    await settle();
    take('mathOn');
    dom.document.getElementById('graph-toggle').click();
    dom.document.getElementById('graph-toggle').click();
    dom.document.getElementById('math-toggle').click();
    await settle();
    take('mathOff');
    dom.document.getElementById('plot-toggle').click();
    await settlePlots(dom);
    take('plotOn');
    tap(env.graphs[0], 'mu');
    await settlePlots(dom);
    take('tapMu');
    clickMathRow(dom, 'x');
    await settlePlots(dom);
    take('mathRowX');
    deliver(dom, { type: 'showModule', source: MODEL });
    await settlePlots(dom);
    take('module');
    return {
      steps,
      charts: env.charts.map((c) => c.options),
      head: dom.serialize(dom.document.head),
      rootStyle: dom.serialize(dom.document.documentElement).style || null,
      listeners: { window: [...dom.listeners.window].sort(), document: [...dom.listeners.document].sort() },
      posted: posted.posted,
      states: posted.states,
    };
  } finally {
    view?.dispose();
    await settle();
    env.restore();
  }
}

test('the VS Code webview mount matches the golden copy from before the pane breakout', async (t) => {
  const actual = await runSession();
  if (process.env.FLATPPL_UPDATE_GOLDEN === '1') {
    writeFileSync(GOLDEN, JSON.stringify(actual, null, 1) + '\n');
    t.diagnostic('golden rewritten: ' + GOLDEN.pathname);
    return;
  }
  const golden = JSON.parse(readFileSync(GOLDEN, 'utf8'));
  for (const step of Object.keys(golden.steps)) {
    assert.deepEqual(actual.steps[step], golden.steps[step], 'webview changed at step ' + step);
  }
  assert.deepEqual(Object.keys(actual.steps), Object.keys(golden.steps));
  assert.deepEqual(actual.charts, golden.charts, 'chart input changed');
  assert.deepEqual(actual.head, golden.head, 'injected stylesheet changed');
  assert.deepEqual(actual.rootStyle, golden.rootStyle, 'root CSS properties changed');
  assert.deepEqual(actual.listeners, golden.listeners, 'installed listeners changed');
  assert.deepEqual(actual.posted, golden.posted, 'messages to the extension host changed');
  assert.deepEqual(actual.states, golden.states, 'persisted webview state changed');
});

test('the default VS Code host adapter keeps its methods and message shapes', (t) => {
  const savedWindow = Object.getOwnPropertyDescriptor(globalThis, 'window');
  t.after(() => {
    if (savedWindow) Object.defineProperty(globalThis, 'window', savedWindow); else delete globalThis.window;
  });
  Object.defineProperty(globalThis, 'window', { value: {
    addEventListener: () => {}, removeEventListener: () => {},
  }, configurable: true, writable: true });
  const { posted } = resetSink();
  const host = defaultVscodeHost();
  assert.deepEqual(Object.keys(host).sort(), [
    'canPersist', 'editSource', 'loadState', 'openModule', 'promptForName',
    'revealSourceLine', 'saveState', 'setTitle', 'signalReady',
  ]);
  for (const hook of ['detachPanel', 'redockPanel', 'onSelect']) {
    assert.equal(hook in host, false, hook + ' must stay absent so VS Code shows no pane breakout');
  }
  host.revealSourceLine(4, 'x', { preserveFocus: true });
  host.revealSourceLine(5);
  host.setTitle('x');
  host.openModule('dep.flatppl', 'y');
  host.signalReady();
  host.promptForName({ suggested: 'p', existingNames: ['x'] });
  host.editSource({ range: null, newText: 'p = 1' });
  const shapes = posted.map((m) => m.type + ':' + Object.keys(m).sort().join(','));
  assert.deepEqual(shapes, [
    'navigateTo:line,preserveFocus,type',
    'navigateTo:line,preserveFocus,type',
    'updateTitle:name,type',
    'openModule:member,path,type',
    'webviewReady:type',
    'promptForName:existingNames,nonce,suggested,type',
    'editSource:newText,range,type',
  ]);
  assert.equal(host.canPersist(), true);
});

test('VS Code graph gestures post only the existing message types', async () => {
  const env = installViewerEnv();
  const { dom } = env;
  const posted = resetSink();
  let view = null;
  try {
    dom.document.body.innerHTML = '<div id="flatppl-viewer-root"></div>';
    view = mount(dom.document.getElementById('flatppl-viewer-root'));
    deliver(dom, { type: 'sourceUpdate', source: MODEL, targetName: 'x' });
    await settle();
    const cy = env.graphs[0];
    const mark = posted.posted.length;
    tap(cy, 'mu');
    await settle();
    assert.deepEqual(posted.posted.slice(mark), [], 'a plain tap stays inside the webview');
    tap(cy, 'mu', { ctrlKey: true });
    cy.$id('mu').emit({ type: 'dbltap' });
    await settle();
    const types = posted.posted.slice(mark).map((m) => m.type);
    assert.deepEqual(types, ['navigateTo', 'updateTitle']);
    const known = new Set(['webviewReady', 'navigateTo', 'updateTitle', 'openModule', 'promptForName', 'editSource']);
    for (const m of posted.posted) assert.ok(known.has(m.type), 'unexpected message ' + m.type);
    assert.equal(dom.document.getElementById('detach-btn'), null, 'no Pop out control in VS Code');
  } finally {
    view?.dispose();
    await settle();
    env.restore();
  }
});
