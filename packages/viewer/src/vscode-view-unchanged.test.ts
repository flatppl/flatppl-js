// @ts-nocheck — whole-mount contract tests for the VS Code webview host.
//
// The playground's pane breakout (flatppl-dev/playground-pane-breakout-design.md
// §7) must leave the VS Code view unchanged. These tests mount the viewer the
// way the webview does (no opts.host, acquireVsCodeApi defined) and compare
// the DOM it builds, the listeners it installs and the messages it posts with
// a golden copy. The golden was captured on origin/main before the breakout
// seams existed. Regenerate it (FLATPPL_UPDATE_GOLDEN=1) only when the VS Code
// view is meant to change, and say so in the pull request.

import { registerHooks, createRequire } from 'node:module';
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

const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');
const { default: realCytoscape } = await import('cytoscape');
const { default: dagre } = await import('cytoscape-dagre');
realCytoscape.use(dagre);
const { installTestDom } = await import('./test-dom.ts');
const { mount } = await import('./main.ts');
const { defaultVscodeHost } = await import('./host-adapter.ts');
const { InProcessWorker, settle } = await import('./test-worker.ts');

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

/** Install the browser surface the webview provides. */
function webview() {
  const dom = installTestDom();
  const saved = Object.fromEntries(['cytoscape', 'echarts', 'FlatPPLEngine', 'Worker']
    .map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  const define = (k, v) => Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
  // The webview renders into a canvas; headless cytoscape keeps the same model.
  const graphs = [];
  const cytoscape = (opts) => {
    const cy = realCytoscape({ ...opts, container: undefined, headless: true, styleEnabled: true });
    graphs.push(cy);
    return cy;
  };
  cytoscape.use = () => {};
  define('cytoscape', cytoscape);
  define('echarts', { getInstanceByDom: () => null, init: () => { throw new Error('no chart expected'); } });
  define('FlatPPLEngine', engine);
  define('Worker', InProcessWorker);
  dom.window.__FLATPPL_CONFIG__ = { samplerWorkerUrl: 'sampler-worker.min.js', wasmApiUrl: null };
  return {
    dom,
    graphs,
    sink: resetSink(),
    restore() {
      for (const [k, d] of Object.entries(saved)) {
        if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k];
      }
      dom.restore();
    },
  };
}

function deliver(dom, data) {
  dom.window.dispatchEvent(new MessageEvent('message', { data, source: null, origin: 'vscode-webview://panel' }));
}

/** The VS Code session: boot, config push, a model, then panel toggles. */
async function runSession() {
  const env = webview();
  const { dom } = env;
  let view = null;
  try {
    dom.document.body.innerHTML = '<div id="flatppl-viewer-root"></div>';
    const root = dom.document.getElementById('flatppl-viewer-root');
    view = mount(root);
    const snapshots = {};
    await settle();
    snapshots.boot = dom.serialize(root);
    deliver(dom, { type: 'configUpdate', config: { sampleCount: 1000, marginalizationSampleCount: 50,
      dagNavigationHistoryCap: 20, truncateRejectionBudget: 500 } });
    deliver(dom, { type: 'sourceUpdate', source: MODEL, targetName: 'x' });
    await settle();
    snapshots.model = dom.serialize(root);
    dom.document.getElementById('math-toggle').click();
    await settle();
    snapshots.mathOn = dom.serialize(root);
    dom.document.getElementById('graph-toggle').click();
    dom.document.getElementById('graph-toggle').click();
    dom.document.getElementById('math-toggle').click();
    await settle();
    snapshots.mathOff = dom.serialize(root);
    deliver(dom, { type: 'showModule', source: MODEL });
    await settle();
    snapshots.module = dom.serialize(root);
    return {
      snapshots,
      head: dom.serialize(dom.document.head),
      rootStyle: dom.serialize(dom.document.documentElement).style || null,
      listeners: { window: [...dom.listeners.window].sort(), document: [...dom.listeners.document].sort() },
      posted: env.sink.posted,
      states: env.sink.states,
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
  assert.deepEqual(actual.snapshots, golden.snapshots, 'webview DOM changed');
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
  const listeners = [];
  Object.defineProperty(globalThis, 'window', { value: {
    addEventListener: (type, fn) => listeners.push(fn), removeEventListener: () => {},
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
  const env = webview();
  const { dom } = env;
  let view = null;
  try {
    dom.document.body.innerHTML = '<div id="flatppl-viewer-root"></div>';
    view = mount(dom.document.getElementById('flatppl-viewer-root'));
    deliver(dom, { type: 'sourceUpdate', source: MODEL, targetName: 'x' });
    await settle();
    const cy = env.graphs[0];
    const mark = env.sink.posted.length;
    const tap = (id, originalEvent) => cy.$id(id).emit({ type: 'tap', originalEvent, position: { x: -1e9, y: -1e9 } });
    tap('mu', { ctrlKey: false, metaKey: false, shiftKey: false });
    await settle();
    assert.deepEqual(env.sink.posted.slice(mark), [], 'a plain tap stays inside the webview');
    tap('mu', { ctrlKey: true, metaKey: false, shiftKey: false });
    cy.$id('mu').emit({ type: 'dbltap' });
    await settle();
    const types = env.sink.posted.slice(mark).map((m) => m.type);
    assert.deepEqual(types, ['navigateTo', 'updateTitle']);
    const known = new Set(['webviewReady', 'navigateTo', 'updateTitle', 'openModule', 'promptForName', 'editSource']);
    for (const m of env.sink.posted) assert.ok(known.has(m.type), 'unexpected message ' + m.type);
    assert.equal(dom.document.getElementById('detach-btn'), null, 'no Pop out control in VS Code');
  } finally {
    view?.dispose();
    await settle();
    env.restore();
  }
});
