// @ts-nocheck — whole-mount tests of the pane-breakout seams for the web host,
// which will show graph, plots and math in separate browser windows.

import { registerHooks } from 'node:module';
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

const { installViewerEnv } = await import('./test-viewer-env.ts');
const { settle } = await import('./test-worker.ts');
const { mount } = await import('./main.ts');

const MODEL = [
  'mu = draw(Normal(mu = 0.0, sigma = 1.0))',
  'sigma = 1.5',
  'x = draw(Normal(mu = mu, sigma = sigma))',
  '',
].join('\n');

/** Mount the viewer as the web playground would, with a recording host.
 *  close() tears it down at once, so a test can mount a second window;
 *  otherwise it is torn down after the test. */
async function webMount(t, opts = {}) {
  const env = installViewerEnv();
  const { dom } = env;
  const calls = [];
  const host = opts.host === undefined ? {
    detachPanel: (p) => calls.push(['detachPanel', p]),
    redockPanel: (p) => calls.push(['redockPanel', p]),
    onSelect: (n) => calls.push(['onSelect', n]),
    setTarget: () => {},
  } : opts.host;
  dom.document.body.innerHTML = '<div id="surface-root"></div>';
  let view;
  try {
    view = mount(dom.document.getElementById('surface-root'), { host, ...(opts.mount || {}) });
  } catch (err) {
    env.restore();
    throw err;
  }
  if (opts.update !== false) {
    view.update(MODEL, opts.target || 'x', {});
    await settle();
  }
  let open = true;
  async function close() {
    if (!open) return;
    open = false;
    view.dispose();
    await settle();
    env.restore();
  }
  t.after(close);
  const $ = (id) => dom.document.getElementById(id);
  return { dom, view, calls, cy: () => env.graphs[0], $, close };
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

test('a plot snapshot carries every setting to a mount in another window', async (t) => {
  const a = await webMount(t);
  const base = a.view.snapshotPanel('plot').plot;
  const crafted = { plot: { ...base, sampleCount: 1234, marginalizationCount: 33, rejectionBudget: 77,
    rootSeed: 9, inferenceOpts: { ...base.inferenceOpts, backend: 'mh', chains: 3 },
    recordSelection: { bindingName: 'x', mode: 'marginals', selected: ['x'], marginalGroups: ['x'], genQuantities: [] },
    presetOverrides: [['p', { values: { a: 1 } }]],
    domainOverrides: [['d', { ranges: { a: { lo: 0, hi: 1 } } }]],
    planMemory: [['mu', { sweepKey: null, gridKey: null, mode: 'samples', outputKey: null, presetName: null,
      domainName: null, autoOverride: null, domainAutoOverride: null }]] } };
  a.view.restorePanel('plot', structuredClone(crafted));
  const fromA = a.view.snapshotPanel('plot');
  await a.close();

  const b = await webMount(t);
  b.view.restorePanel('plot', structuredClone(fromA));
  const fromB = b.view.snapshotPanel('plot');
  assert.deepEqual(fromB, fromA);
  for (const k of Object.keys(crafted.plot)) {
    if (k === 'planMemory') continue;
    assert.deepEqual(fromB.plot[k], crafted.plot[k], k);
  }
  const memory = new Map(fromB.plot.planMemory);
  assert.deepEqual(memory.get('mu'), crafted.plot.planMemory[0][1]);
});

test('a graph snapshot carries collapsed groups, history and viewport to another window', async (t) => {
  const a = await webMount(t);
  a.view.update(MODEL, 'mu', { pushHistory: true });
  await settle();
  a.view.restorePanel('graph', { graph: { collapsedReifications: ['grp'], viewport: null,
    history: a.view.snapshotPanel('graph').graph.history } });
  a.cy().zoom(1.7);
  a.cy().pan({ x: 12, y: -4 });
  const fromA = a.view.snapshotPanel('graph');
  assert.deepEqual(fromA.graph.history.map((h) => h.targetName), ['x']);
  await a.close();

  const b = await webMount(t, { target: 'mu' });
  b.view.restorePanel('graph', structuredClone(fromA));
  assert.deepEqual(b.view.snapshotPanel('graph'), fromA);
  assert.equal(b.cy().zoom(), 1.7);
  // Back works from the restored history: it returns to x's sub-DAG.
  const visible = () => b.cy().nodes().map((n) => n.id()).filter((id) => !id.includes(':')).sort();
  assert.deepEqual(visible(), ['mu']);
  b.$('back-btn').click();
  await settle();
  assert.deepEqual(b.view.snapshotPanel('graph').graph.history, []);
  assert.deepEqual(visible(), ['mu', 'sigma', 'x']);
});

test('restoring the graph before the first update is refused', async (t) => {
  const { view } = await webMount(t, { update: false });
  assert.throws(() => view.restorePanel('graph', { graph: { collapsedReifications: [], viewport: null,
    history: [{ targetName: 'x', path: null }] } }), /needs a model: call update\(\) first/);
});

test('restricted and detached mounts keep the stored panel flags', async (t) => {
  const store = { plotEnabled: true, graphEnabled: true, mathEnabled: true };
  const saved = [];
  const host = { saveState: (s) => saved.push(s), loadState: () => store, setTarget: () => {},
    detachPanel: () => {}, redockPanel: () => {} };
  const plots = await webMount(t, { host, mount: { panels: ['plot'] } });
  assert.ok(saved.length > 0);
  for (const s of saved) {
    assert.deepEqual([s.plotEnabled, s.graphEnabled, s.mathEnabled], [true, true, true]);
  }
  await plots.close();

  saved.length = 0;
  const main = await webMount(t, { host });
  main.view.setPanelPlacement('plot', 'detached');
  const last = saved[saved.length - 1];
  assert.equal(last.plotEnabled, true, 'the detached panel is not stored as off');
});

test('restorePanel refuses a snapshot for another panel', async (t) => {
  const { view } = await webMount(t);
  assert.throws(() => view.restorePanel('math', view.snapshotPanel('plot')), /no state for panel math/);
  assert.throws(() => view.setPanelPlacement('table', 'detached'), /unknown panel/);
});
