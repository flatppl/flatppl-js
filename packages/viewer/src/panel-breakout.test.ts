// @ts-nocheck — whole-mount tests of the pane-breakout seams for the web host.
// Design: flatppl-dev/playground-pane-breakout-design.md §5, §7, §8.

import { registerHooks, createRequire } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

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
const { InProcessWorker, settle } = await import('./test-worker.ts');
const { mount } = await import('./main.ts');

const MODEL = [
  'mu = draw(Normal(mu = 0.0, sigma = 1.0))',
  'sigma = 1.5',
  'x = draw(Normal(mu = mu, sigma = sigma))',
  '',
].join('\n');

/** Mount the viewer as the web playground would, with a recording host. */
async function webMount(t, opts = {}) {
  const dom = installTestDom();
  const saved = Object.fromEntries(['cytoscape', 'echarts', 'FlatPPLEngine', 'Worker']
    .map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  const define = (k, v) => Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
  const graphs = [];
  const cytoscape = (o) => {
    const cy = realCytoscape({ ...o, container: undefined, headless: true, styleEnabled: true });
    graphs.push(cy);
    return cy;
  };
  cytoscape.use = () => {};
  define('cytoscape', cytoscape);
  // Plots render into ECharts; these tests read plot state, not pixels.
  define('echarts', { getInstanceByDom: () => null, init: () => ({
    setOption() {}, resize() {}, dispose() {}, isDisposed: () => false, on() {}, off() {},
    getModel: () => ({ eachComponent() {}, eachSeries() {} }), getDom: () => null,
    getZr: () => ({ on() {}, off() {} }), dispatchAction() {}, getWidth: () => 400, getHeight: () => 300,
  }) });
  define('FlatPPLEngine', engine);
  define('Worker', InProcessWorker);
  dom.window.__FLATPPL_CONFIG__ = { samplerWorkerUrl: 'sampler-worker.min.js', wasmApiUrl: null };
  const calls = [];
  const host = opts.host === undefined ? {
    detachPanel: (p) => calls.push(['detachPanel', p]),
    redockPanel: (p) => calls.push(['redockPanel', p]),
    onSelect: (n) => calls.push(['onSelect', n]),
    setTarget: () => {},
  } : opts.host;
  function restore() {
    for (const [k, d] of Object.entries(saved)) {
      if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k];
    }
    dom.restore();
  }
  dom.document.body.innerHTML = '<div id="surface-root"></div>';
  let view;
  try {
    view = mount(dom.document.getElementById('surface-root'), { host, ...(opts.mount || {}) });
  } catch (err) {
    restore();
    throw err;
  }
  view.update(MODEL, 'x', {});
  await settle();
  t.after(async () => {
    view.dispose();
    await settle();
    restore();
  });
  const $ = (id) => dom.document.getElementById(id);
  return { dom, view, calls, cy: () => graphs[0], $ };
}

function key(el, k) {
  el.dispatchEvent(Object.assign(new Event('keydown', { bubbles: true, cancelable: true }), { key: k }));
}

test('a mount restricted to plots shows only plots and ignores the other toggles', async (t) => {
  const { $, calls } = await webMount(t, { mount: { panels: ['plot'] } });
  assert.equal($('graph-toggle').hidden, true);
  assert.equal($('math-toggle').hidden, true);
  assert.equal($('plot-toggle').hidden, false);
  assert.equal($('plot-panel').classList.contains('hidden'), false);
  assert.equal($('graph-panel').classList.contains('hidden'), true);
  assert.equal($('math-panel').classList.contains('hidden'), true);
  $('graph-toggle').click();
  $('math-toggle').click();
  assert.equal($('graph-panel').classList.contains('hidden'), true);
  assert.equal($('math-panel').classList.contains('hidden'), true);
  assert.deepEqual(calls, []);
});

test('mount refuses a panel list that names no panel', async (t) => {
  await assert.rejects(webMount(t, { mount: { panels: ['source'] } }), /names no panel/);
});

test('Pop out lists the docked panels and hands the choice to the host', async (t) => {
  const { $, calls, dom } = await webMount(t);
  const button = $('detach-btn');
  assert.ok(button, 'the host can detach, so the control exists');
  assert.equal(button.getAttribute('aria-haspopup'), 'menu');
  assert.ok(dom.document.head.querySelector('style[data-flatppl-breakout-css]'));
  key(button, 'ArrowDown');
  const menu = $('detach-menu');
  assert.equal(menu.hidden, false);
  assert.equal(button.getAttribute('aria-expanded'), 'true');
  const items = menu.querySelectorAll('[role="menuitem"]');
  assert.deepEqual(items.map((i) => i.textContent),
    ['Open Graph in a new window', 'Open Plots in a new window', 'Open Math in a new window']);
  assert.equal(dom.document.activeElement, items[0]);
  key(items[0], 'ArrowUp');
  assert.equal(dom.document.activeElement, items[2]);
  key(items[2], 'Escape');
  assert.equal(menu.hidden, true);
  assert.equal(dom.document.activeElement, button);
  button.click();
  menu.querySelectorAll('[role="menuitem"]')[1].click();
  assert.deepEqual(calls, [['detachPanel', 'plot']]);
  assert.equal(menu.hidden, true);
  assert.equal(dom.document.activeElement, $('plot-toggle'), 'focus moves to the panel toggle');
});

test('a detached panel is off here, marked on its toggle, and its toggle asks to re-dock', async (t) => {
  const { $, view, calls } = await webMount(t);
  view.setPanelPlacement('plot', 'detached');
  const toggle = $('plot-toggle');
  assert.equal($('plot-panel').classList.contains('hidden'), true);
  assert.equal(toggle.getAttribute('aria-pressed'), 'true');
  assert.ok(toggle.classList.contains('detached'));
  assert.equal(toggle.textContent, 'Plots ↗ (in a separate window)');
  assert.match(toggle.getAttribute('title'), /separate window/);
  $('detach-btn').click();
  const offered = $('detach-menu').querySelectorAll('[role="menuitem"]').map((i) => i.getAttribute('data-panel'));
  assert.deepEqual(offered, ['graph', 'math']);
  toggle.click();
  assert.deepEqual(calls, [['redockPanel', 'plot']]);
  assert.equal($('plot-panel').classList.contains('hidden'), true, 'the host decides when it is back');
  view.setPanelPlacement('plot', 'docked');
  assert.equal($('plot-panel').classList.contains('hidden'), false);
  assert.equal(toggle.classList.contains('detached'), false);
  assert.equal(toggle.textContent, 'Plots');
  assert.equal(toggle.getAttribute('title'), 'Toggle the plot panel');
});

test('with every panel detached the Pop out control is disabled', async (t) => {
  const { $, view } = await webMount(t);
  for (const p of ['graph', 'plot', 'math']) view.setPanelPlacement(p, 'detached');
  assert.equal($('detach-btn').disabled, true);
  assert.equal($('panels-hidden').hidden, false);
});

test('a graph tap reports the selection; a host select does not echo', async (t) => {
  const { view, calls, cy } = await webMount(t);
  cy().$id('mu').emit({ type: 'tap', originalEvent: { ctrlKey: false, metaKey: false, shiftKey: false },
    position: { x: -1e9, y: -1e9 } });
  await settle();
  assert.deepEqual(calls, [['onSelect', 'mu']]);
  view.select('x');
  await settle();
  assert.deepEqual(calls, [['onSelect', 'mu']]);
});

test('a plot snapshot is plain data and carries settings to another mount', async (t) => {
  const a = await webMount(t);
  const plot = a.view.snapshotPanel('plot').plot;
  assert.equal(plot.sampleCount, 50000);
  const changed = { plot: { ...plot, sampleCount: 1234, marginalizationCount: 33, rejectionBudget: 77,
    rootSeed: 9, inferenceOpts: { ...plot.inferenceOpts, backend: 'mh', chains: 3 },
    presetOverrides: [['p', { values: { a: 1 } }]], domainOverrides: [['d', { ranges: { a: { lo: 0, hi: 1 } } }]] } };
  a.view.restorePanel('plot', structuredClone(changed));
  const back = a.view.snapshotPanel('plot');
  assert.deepEqual(structuredClone(back), back, 'a snapshot survives structured cloning');
  for (const k of ['sampleCount', 'marginalizationCount', 'rejectionBudget', 'rootSeed', 'inferenceOpts',
    'presetOverrides', 'domainOverrides']) {
    assert.deepEqual(back.plot[k], changed.plot[k], k);
  }
});

test('a graph snapshot restores collapsed groups, history and viewport', async (t) => {
  const { view, cy } = await webMount(t);
  view.update(MODEL, 'mu', { pushHistory: true });
  await settle();
  cy().zoom(1.7);
  cy().pan({ x: 12, y: -4 });
  const snap = view.snapshotPanel('graph');
  assert.deepEqual(structuredClone(snap), snap);
  assert.deepEqual(snap.graph.history.map((h) => h.targetName), ['x']);
  assert.deepEqual(snap.graph.viewport, { zoom: 1.7, pan: { x: 12, y: -4 } });
  cy().zoom(1);
  cy().pan({ x: 0, y: 0 });
  view.restorePanel('graph', snap);
  assert.equal(cy().zoom(), 1.7);
  assert.deepEqual(cy().pan(), { x: 12, y: -4 });
  assert.deepEqual(view.snapshotPanel('graph').graph.history, snap.graph.history);
});

test('restorePanel refuses a snapshot for another panel', async (t) => {
  const { view } = await webMount(t);
  assert.throws(() => view.restorePanel('math', view.snapshotPanel('plot')), /no state for panel math/);
  assert.throws(() => view.setPanelPlacement('table', 'detached'), /unknown panel/);
});
