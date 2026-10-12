// @ts-nocheck — focused DOM event contracts; Chrome owns visual verification.
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
const { applyPanelLayout, installPanelDivider, setGraphEnabled } = await import('./render-frame.ts');
const { renderPlotForCurrent } = await import('./render-plot.ts');
const { buildInferenceControl } = await import('./render-controls.ts');
const { sourceErrorsFor } = await import('./diagnostics.ts');
const { enterModuleView } = await import('./dag.ts');
const { applySourceUpdate } = await import('./orchestration.ts');
const { VIEWER_BODY_HTML } = await import('./templates.ts');
const { renderRecordMarginals, renderGenQuantitiesDropdown } = await import('./render-record.ts');
const { listScalarAxes } = await import('./util.ts');
const { disposePlotCharts } = await import('./render-frame.ts');
const echarts = await import('echarts');
const engine = createRequire(import.meta.url)('@flatppl/engine');
const { default: cytoscape } = await import('cytoscape');
const { default: dagre } = await import('cytoscape-dagre');
cytoscape.use(dagre);

/** The viewer uses native events and a small DOM surface. Model that surface
 * here rather than add a DOM package solely for a few interaction contracts. */
function mountDOM(t) {
  const nodes = new Map();
  const doc = Object.assign(new EventTarget(), {
    getElementById: (id) => nodes.get(id),
    createElement: (tag) => element(tag),
    createDocumentFragment: () => element('fragment'),
    activeElement: null,
  });
  function element(tag = 'div') {
    const classes = new Set();
    const el = Object.assign(new EventTarget(), {
      tagName: tag.toUpperCase(), children: [], style: {}, attributes: {}, textContent: '',
      classList: {
        add: (name) => classes.add(name), contains: (name) => classes.has(name),
        remove: (name) => classes.delete(name),
        toggle: (name, on) => on ? classes.add(name) : classes.delete(name),
      },
      setAttribute(name, value) { this.attributes[name] = value; },
      getAttribute(name) { return this.attributes[name]; },
      closest(selector) { return selector.split(',').some((tag) => tag.trim().toUpperCase() === this.tagName) ? this : null; },
      appendChild(child) { this.children.push(child); return child; },
      append(...children) { this.children.push(...children); },
      contains(node) { return this === node || this.children.some((child) => child.contains(node)); },
      querySelectorAll(selector) {
        const tags = selector.split(',').map((s) => s.trim().toUpperCase());
        return this.children.flatMap((child) => [
          ...(tags.includes(child.tagName) ? [child] : []), ...child.querySelectorAll(selector),
        ]);
      },
      querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
      focus() { doc.activeElement = this; },
      setPointerCapture(pointer) { this.pointer = pointer; },
      hasPointerCapture(pointer) { return this.pointer === pointer; },
      releasePointerCapture() { this.pointer = null; },
      getBoundingClientRect() {
        return { height: Number(this.style.flex?.match(/([\d.]+)px$/)?.[1]) || this.height || 200,
          width: 200, top: 10, bottom: 38, left: 0, right: 200 };
      },
    });
    Object.defineProperty(el, 'id', {
      get() { return this._id; }, set(id) { this._id = id; nodes.set(id, this); },
    });
    if (tag === 'canvas') el.getContext = () => null; // ECharts SSR uses its fallback text metrics.
    Object.defineProperty(el, 'innerHTML', {
      get() { return this._html || ''; },
      set(html) {
        this._html = html; this.children = [];
        for (const match of html.matchAll(/<([a-z]+)[^>]*id="([^"]+)"/g)) {
          const child = element(match[1]); child.id = match[2]; this.appendChild(child);
        }
      },
    });
    return el;
  }
  doc.body = element('body');
  doc.body.innerHTML = VIEWER_BODY_HTML;
  const previous = Object.getOwnPropertyDescriptors(globalThis);
  globalThis.document = doc;
  globalThis.window = { innerWidth: 390, innerHeight: 844 };
  globalThis.requestAnimationFrame = (fn) => fn();
  globalThis.echarts = { getInstanceByDom: () => null };
  t.after(() => {
    for (const key of ['document', 'window', 'requestAnimationFrame', 'echarts']) {
      if (previous[key]) Object.defineProperty(globalThis, key, previous[key]); else delete globalThis[key];
    }
  });
  return doc;
}

function event(type, properties = {}) {
  return Object.assign(new Event(type, { cancelable: true }), properties);
}

test('pointer and keyboard resizing survive panel toggles without touching the plot', (t) => {
  const doc = mountDOM(t);
  const ctx = { graphEnabled: true, plotEnabled: true, mathEnabled: true, host: {},
    currentState: { data: { nodes: [{}], reifications: [] } },
    plotEchart: { resize() {}, dispose() { assert.fail('Graph toggles must keep the chart'); } },
    measureCache: new Map([['x', Float64Array.of(2, 4, 6)]]),
  };
  const graph = doc.getElementById('graph-panel'), plot = doc.getElementById('plot-panel');
  graph.height = 300;
  const divider = doc.getElementById('plot-divider');
  const lowerDivider = doc.getElementById('math-divider');
  const main = doc.getElementById('main');
  main.style.cursor = 'default'; main.style.userSelect = 'text';
  doc.body.style.cursor = 'crosshair'; doc.body.style.userSelect = 'auto';
  const cleanup = installPanelDivider(ctx, 'graph');
  t.after(cleanup);
  t.after(installPanelDivider(ctx, 'plot'));
  applyPanelLayout(ctx);
  assert.ok(parseFloat(plot.style.flex.slice(4)) > parseFloat(graph.style.flex.slice(4)), 'small graphs leave more room for plots');
  divider.dispatchEvent(event('pointerdown', { pointerId: 1, button: 0, clientY: 50 }));
  // A second finger must neither start another resize nor release the first.
  divider.dispatchEvent(event('pointerdown', { pointerId: 2, button: 0, clientY: 0 }));
  divider.dispatchEvent(event('pointerup', { pointerId: 2 }));
  lowerDivider.dispatchEvent(event('pointerdown', { pointerId: 2, button: 0, clientY: 0 }));
  divider.dispatchEvent(event('pointermove', { pointerId: 1, clientY: 90 }));
  divider.dispatchEvent(event('pointerup', { pointerId: 1 }));
  lowerDivider.dispatchEvent(event('pointerup', { pointerId: 2 }));
  assert.equal(main.style.cursor, 'default');
  assert.equal(main.style.userSelect, 'text');
  assert.equal(doc.body.style.cursor, 'crosshair');
  assert.equal(doc.body.style.userSelect, 'auto');
  assert.equal(graph.style.flex, '1 1 340px');
  assert.equal(plot.style.flex, '1 1 160px');
  divider.dispatchEvent(event('keydown', { key: 'ArrowUp', shiftKey: false }));
  assert.equal(graph.style.flex, '1 1 324px');
  assert.equal(plot.style.flex, '1 1 176px');
  const chart = ctx.plotEchart, samples = ctx.measureCache.get('x');
  setGraphEnabled(ctx, false);
  assert.equal(graph.classList.contains('hidden'), true);
  assert.equal(plot.classList.contains('hidden'), false);
  assert.equal(doc.getElementById('math-panel').classList.contains('hidden'), false);
  setGraphEnabled(ctx, true);
  assert.equal(graph.style.flex, '1 1 324px');
  assert.equal(plot.style.flex, '1 1 176px');
  assert.equal(ctx.plotEchart, chart);
  assert.equal(ctx.measureCache.get('x'), samples);
  cleanup();
  const split = plot.style.flex;
  divider.dispatchEvent(event('keydown', { key: 'ArrowDown' }));
  assert.equal(plot.style.flex, split);
});

test('an incomplete call shows its real diagnostic and jumps to its source line without sampling', (t) => {
  const doc = mountDOM(t);
  const result = engine.processSource('flatppl_compat = "0.1"\nx ~ Normal(');
  const reveals = [];
  const ctx = { sourceErrors: sourceErrorsFor(result), currentPlotBindingName: 'x',
    currentState: { data: engine.computeSubDAG(result.bindings, 'x') },
    currentPlotPlan: { mode: 'samples', name: 'x' },
    host: { revealSourceLine: (line) => reveals.push(line) }, plotEchart: null,
  };
  renderPlotForCurrent(ctx);
  const message = doc.getElementById('plot-empty');
  const text = message.children.map((child) => child.textContent).join(' ');
  assert.match(text, /Expected '\)' before the end of the source/);
  assert.match(text, /Line 2, column 12/);
  message.querySelector('button').dispatchEvent(event('click'));
  assert.deepEqual(reveals, [1]);
  const semantic = engine.processSource('flatppl_compat = "0.1"\nx = 1\ny = missing');
  assert.deepEqual(sourceErrorsFor(semantic), [], 'a separate invalid binding must not block x');
});

test('sampler settings label every field and return keyboard focus on Escape without drawing', (t) => {
  const doc = mountDOM(t);
  let draws = 0;
  const ctx = { inferenceOpts: { backend: 'mh', chains: 4, walkers: null, draws: 1000,
    warmup: 1000, seed: null, amisIters: 30, amisSamples: 300, smcParticles: 2000,
    smcSteps: 12, smcCESS: 0.7, nLive: 400, dlogz: 0.5 }, SAMPLE_COUNT: 1000, rootSeed: 42 };
  const control = buildInferenceControl(ctx, () => draws++);
  const gear = control.querySelectorAll('button').find((button) => button.getAttribute('aria-label') === 'Sampler settings');
  const panel = doc.getElementById(gear.getAttribute('aria-controls'));
  const labels = control.querySelectorAll('label');
  for (const input of control.querySelectorAll('input, select')) {
    assert.ok(labels.some((label) => label.htmlFor === input.id && label.textContent.length > 0));
  }
  gear.dispatchEvent(event('click'));
  assert.equal(doc.activeElement.tagName, 'INPUT');
  assert.equal(doc.activeElement.name, 'draws');
  doc.activeElement.value = '500';
  doc.activeElement.dispatchEvent(event('change'));
  assert.equal(draws, 0);
  panel.dispatchEvent(event('keydown', { key: 'Escape' }));
  assert.equal(doc.activeElement, gear);
  assert.equal(panel.style.display, 'none');
  assert.equal(gear.getAttribute('aria-expanded'), 'false');
  assert.equal(ctx.inferenceOpts.draws, 500);
});

test('an empty source clears the old graph and asks the user to add a binding', (t) => {
  const doc = mountDOM(t);
  const previous = globalThis.FlatPPLEngine;
  globalThis.FlatPPLEngine = engine;
  t.after(() => { globalThis.FlatPPLEngine = previous; });
  const cy = cytoscape({ headless: true, elements: [{ data: { id: 'old' } }] });
  t.after(() => cy.destroy());
  const ctx = { cy, host: {}, history: [], currentBindings: new Map(), currentSource: '',
    currentPlotPlan: null, graphEnabled: true, plotEnabled: true, mathEnabled: false,
    currentState: { targetName: 'old', data: { nodes: [{ id: 'old' }], reifications: [] } },
    collapsedReifications: new Set(), _reifSeen: new Set(), MODULE_TARGET: '__module__', HINT: '',
  };
  enterModuleView(ctx, false);
  assert.equal(cy.nodes().length, 0);
  assert.match(doc.getElementById('plot-content').innerHTML, /Add a binding/);
  assert.equal(doc.getElementById('graph-actions').hidden, true);
});

test('a failed source update clears the old plot and labels the retained graph', (t) => {
  const doc = mountDOM(t);
  const previous = globalThis.FlatPPLEngine;
  globalThis.FlatPPLEngine = { processSource() { throw new Error('Incomplete source'); } };
  t.after(() => { globalThis.FlatPPLEngine = previous; });
  t.mock.method(console, 'error', () => {});
  const graph = { data: { nodes: [{ id: 'x' }] } };
  let disposed = false;
  const ctx = { host: {}, currentSource: 'x = 1', currentBindings: new Map(),
    currentState: graph, currentPlotPlan: { name: 'x' }, plotEnabled: true,
    plotEchart: { dispose() { disposed = true; } }, mathEnabled: false };
  applySourceUpdate(ctx, { source: 'x = (', type: 'sourceUpdate' });
  assert.equal(ctx.currentState, graph);
  assert.equal(disposed, true);
  assert.equal(ctx.currentPlotPlan, null);
  assert.match(doc.getElementById('graph-source-notice').textContent, /last valid model/);
  assert.equal(doc.getElementById('graph-source-notice').hidden, false);
});

test('six-coordinate plots default to four and preserve uncapped selection, panning and data across rerenders', async (t) => {
  const doc = mountDOM(t);
  const priorEngine = globalThis.FlatPPLEngine, priorStyle = globalThis.getComputedStyle;
  globalThis.FlatPPLEngine = engine;
  globalThis.getComputedStyle = () => ({ color: '#333' });
  t.after(() => { globalThis.FlatPPLEngine = priorEngine; globalThis.getComputedStyle = priorStyle; });
  const { createWorkerHandler } = createRequire(import.meta.url)('@flatppl/engine/worker.ts');
  const source = 'flatppl_compat = "0.1"\nx ~ Normal(0, 1)\npoints = (x, 2*x, 3*x, 4*x, 5*x, 6*x)';
  const processed = engine.processSource(source);
  assert.deepEqual(processed.diagnostics, []);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const materialiser = { ...engine.orchestrator.buildDerivations(processed.bindings),
    sampleCount: 256, rootSeed: 42,
    getMeasure(name) {
      if (!cache.has(name)) cache.set(name, engine.materialiser.materialiseMeasure(name, materialiser));
      return cache.get(name);
    },
    sendWorker(message) {
      const reply = worker.handle(message);
      if (reply?.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
  };
  const measure = await materialiser.getMeasure('points');
  // A deterministic tail distinguishes the histogram's trimmed view from
  // the scatter range, without changing the known coordinate relationships.
  measure.samples.set([1000, 2000, 3000, 4000, 5000, 6000], measure.samples.length - 6);
  const axes = listScalarAxes(measure);
  const keys = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]'];
  assert.deepEqual(axes.map((axis) => axis.key), keys);
  for (let i = 0; i < axes.length; i++) {
    assert.deepEqual(axes[i].samples, axes[0].samples.map((x) => x * (i + 1)));
  }
  const originalSamples = measure.samples.slice();
  const mounted = new Map();
  let coordinateCalls = 0;
  globalThis.echarts = {
    init(element) {
      const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 220, height: 180 });
      const convertToPixel = chart.convertToPixel.bind(chart);
      chart.convertToPixel = (finder, value) => {
        const pixels = convertToPixel(finder, value);
        assert.ok((Array.isArray(pixels) ? pixels : [pixels]).every(Number.isFinite),
          'label formatters have initialized coordinates on first render and resize');
        coordinateCalls++;
        return pixels;
      };
      mounted.set(element, chart);
      return chart;
    },
    getInstanceByDom: (element) => {
      const chart = mounted.get(element);
      return chart && !chart.isDisposed() ? chart : undefined;
    },
  };
  t.after(() => { for (const chart of mounted.values()) if (!chart.isDisposed()) chart.dispose(); });
  const liveCharts = () => [...mounted.values()].filter((chart) => !chart.isDisposed());
  const ctx = { host: {}, SAMPLE_COUNT: 256, currentPlotBindingName: 'points',
    currentBindings: new Map(), TYPE_STYLE: { draw: { color: '#82b366' }, unknown: { color: '#82b366' } },
    CODICON_PATHS: { 'save-as': '' },
  };
  renderRecordMarginals(ctx, measure, 'points', null);
  assert.ok(coordinateCalls > 0, 'the first paint uses real chart geometry for inside labels');
  assert.deepEqual(ctx.recordSelection.selected, keys.slice(0, 4));
  assert.equal(liveCharts().length, 10, 'the first four variates are selected by default');
  const content = doc.getElementById('plot-content');
  const boxes = content.querySelectorAll('input');
  for (const index of [4, 5]) {
    boxes[index].checked = true;
    boxes[index].dispatchEvent(event('change'));
  }
  assert.deepEqual(ctx.recordSelection.selected, keys);
  assert.equal(liveCharts().length, 21, 'six marginals plus all fifteen coordinate pairs');
  const scatterX = new Map(), scatterY = new Map();
  for (const [element, chart] of mounted) {
    if (chart.isDisposed()) continue;
    const row = parseInt(element.style.gridRow) - 1, col = parseInt(element.style.gridColumn) - 2;
    const option = chart.getOption(), series = option.series[0];
    assert.equal(option.xAxis[0].axisLabel.show, row === col || row === 5);
    assert.equal(option.yAxis[0].axisLabel.show, row === col || col === 0);
    if (row === col) {
      assert.equal(series.type, 'custom');
      assert.ok(series.data.length > 0, 'each diagonal retains its histogram');
      assert.equal(option.yAxis[0].min, 0, 'density bars start at zero');
      assert.ok(chart.convertToPixel({ xAxisIndex: 0 }, 1000 * (col + 1)) > chart.getWidth(),
        'a histogram keeps its labeled trimmed range rather than flattening the bulk to an outlier');
    } else {
      assert.equal(series.type, 'scatter');
      assert.deepEqual(series.data, Array.from(axes[col].samples, (x, i) => [x, axes[row].samples[i]]));
    }
    const dataBeforeResize = option.series.map((s) => s.data);
    for (const [width, height] of [[52, 34], [12, 8], [56, 36], [96, 60], [180, 110], [220, 180]]) {
      chart.resize({ width, height });
      const points = row === col
        ? [[series.data[0].value[0], 0], [series.data.at(-1).value[0], Math.max(...series.data.map((bin) => bin.value[1]))]]
        : [[Math.min(...axes[col].samples), Math.min(...axes[row].samples)],
          [Math.max(...axes[col].samples), Math.max(...axes[row].samples)]];
      const pixels = points.map((point) => chart.convertToPixel({ seriesIndex: 0 }, point));
      for (const [x, y] of pixels) {
        assert.ok(Number.isFinite(x) && x >= 0 && x <= width, 'x stays inside the resized cell');
        assert.ok(Number.isFinite(y) && y >= 0 && y <= height, 'y stays inside the resized cell');
      }
      assert.ok(pixels[1][0] > pixels[0][0] && pixels[1][1] < pixels[0][1], 'tiny plot areas stay positive');
      assert.equal(option.xAxis[0].axisLabel.formatter(0), width < 56 ? '' : '0');
      if (row !== col) {
        const x = chart.convertToPixel({ xAxisIndex: 0 }, axes[col].samples[0]);
        const y = chart.convertToPixel({ yAxisIndex: 0 }, axes[row].samples[0]);
        const xKey = width + ':' + col, yKey = width + ':' + row;
        if (scatterX.has(xKey)) assert.ok(Math.abs(x - scatterX.get(xKey)) < 1e-8, 'scatter columns align');
        if (scatterY.has(yKey)) assert.ok(Math.abs(y - scatterY.get(yKey)) < 1e-8, 'scatter rows align');
        scatterX.set(xKey, x); scatterY.set(yKey, y);
      }
      const labels = [...chart.renderToSVGString().matchAll(/<text([^>]*)>([^<]*)<\/text>/g)].map(([, attributes, text]) => {
        const position = /transform="translate\(([-\d.]+) ([-\d.]+)\)"/.exec(attributes);
        const x = Number(position[1]), y = Number(position[2]) + Number(/\sy="([-\d.]+)"/.exec(attributes)?.[1] || 0);
        const fontSize = Number(/font-size:([\d.]+)px/.exec(attributes)[1]);
        const anchor = /text-anchor="([^"]+)"/.exec(attributes)[1];
        const align = { start: 'left', middle: 'center', end: 'right' }[anchor];
        const rect = echarts.format.getTextRect(text, fontSize + 'px sans-serif', align, 'middle');
        return { text, left: x + rect.x, right: x + rect.x + rect.width,
          top: y + rect.y, bottom: y + rect.y + rect.height };
      });
      for (let i = 0; i < labels.length; i++) {
        const a = labels[i];
        assert.ok(a.left >= 0 && a.right <= width && a.top >= 0 && a.bottom <= height,
          'numeric text stays inside the SVG cell');
        for (const b of labels.slice(i + 1)) {
          const overlaps = Math.min(a.right, b.right) > Math.max(a.left, b.left)
            && Math.min(a.bottom, b.bottom) > Math.max(a.top, b.top);
          assert.equal(overlaps, false, 'numeric labels ' + a.text + ' and ' + b.text + ' do not collide');
        }
      }
    }
    assert.deepEqual(chart.getOption().series.map((s) => s.data), dataBeforeResize);
  }
  const viewport = content.children.at(-1);
  Object.assign(viewport, { scrollWidth: 1300, clientWidth: 500, scrollHeight: 1000, clientHeight: 400,
    scrollLeft: 0, scrollTop: 0 });
  const chartsBeforePan = liveCharts();
  function pan(type, properties = {}, target = viewport) {
    const ev = event(type, { pointerId: 1, pointerType: 'mouse', button: 0, clientX: 150, clientY: 150, ...properties });
    Object.defineProperty(ev, 'target', { value: target });
    viewport.dispatchEvent(ev);
    return ev;
  }
  pan('pointerover');
  assert.equal(viewport.classList.contains('plot-pannable'), true);
  pan('pointerdown');
  pan('pointermove', { clientX: 152, clientY: 151 });
  pan('pointerup');
  assert.equal(pan('click').defaultPrevented, false, 'short clicks retain their normal behavior');
  assert.equal(viewport.scrollLeft, 0);
  pan('pointerdown');
  pan('pointermove', { clientX: 50, clientY: 50 });
  assert.equal(viewport.scrollLeft, 100);
  assert.equal(viewport.scrollTop, 100);
  assert.equal(viewport.hasPointerCapture(1), true);
  pan('pointermove', { clientX: -2000, clientY: -2000 });
  assert.equal(viewport.scrollLeft, 800);
  assert.equal(viewport.scrollTop, 600);
  pan('pointermove', { clientX: 2000, clientY: 2000 });
  assert.equal(viewport.scrollLeft, 0);
  assert.equal(viewport.scrollTop, 0);
  pan('pointerup');
  assert.equal(viewport.hasPointerCapture(1), false);
  assert.equal(pan('click').defaultPrevented, true, 'dragging must not also activate a click');
  assert.equal(viewport.classList.contains('plot-panning'), false);
  for (const end of ['pointercancel', 'lostpointercapture']) {
    pan('pointerdown');
    pan('pointermove', { clientX: 100, clientY: 100 });
    pan(end);
    const left = viewport.scrollLeft;
    pan('pointermove', { clientX: 0, clientY: 0 });
    assert.equal(viewport.scrollLeft, left);
    assert.equal(viewport.hasPointerCapture(1), false);
    assert.equal(viewport.classList.contains('plot-panning'), false);
  }
  for (const [properties, target] of [[{ pointerType: 'touch' }, viewport], [{ button: 2 }, viewport], [{}, doc.createElement('button')]]) {
    const left = viewport.scrollLeft;
    assert.equal(pan('pointerdown', properties, target).defaultPrevented, false);
    pan('pointermove', { ...properties, clientX: 0, clientY: 0 }, target);
    pan('pointerup', properties, target);
    assert.equal(viewport.scrollLeft, left, 'touch, secondary clicks and controls keep their interactions');
  }
  viewport.clientWidth = 1300; viewport.clientHeight = 1000;
  const unscrollableLeft = viewport.scrollLeft;
  pan('pointerdown'); pan('pointermove', { clientX: 0, clientY: 0 }); pan('pointerup');
  assert.equal(viewport.scrollLeft, unscrollableLeft, 'a grid that fits keeps normal mouse interactions');
  assert.equal(viewport.classList.contains('plot-pannable'), false);
  viewport.clientWidth = 500; viewport.clientHeight = 400;
  assert.deepEqual(liveCharts(), chartsBeforePan, 'panning keeps every chart instance');
  pan('pointerdown');
  pan('pointermove', { clientX: 100, clientY: 100 });
  for (const index of [5, 4, 4, 5]) {
    boxes[index].checked = !boxes[index].checked;
    boxes[index].dispatchEvent(event('change'));
    const n = ctx.recordSelection.selected.length;
    assert.equal(liveCharts().length, n * (n + 1) / 2, 'replaced cells must be disposed');
    assert.equal(viewport.hasPointerCapture(1), false, 'rerendering cancels an active drag');
    assert.equal(viewport.classList.contains('plot-panning'), false);
  }
  assert.deepEqual(ctx.recordSelection.selected, keys);
  const capture = t.mock.method(viewport, 'setPointerCapture');
  pan('pointerdown'); pan('pointermove', { clientX: 100, clientY: 100 }); pan('pointerup');
  assert.equal(capture.mock.callCount(), 1, 'reusing the viewport must not duplicate drag listeners');
  let mode = content.querySelector('select');
  mode.value = 'marginals'; mode.dispatchEvent(event('change'));
  assert.equal(liveCharts().length, 1);
  const oldLeft = viewport.scrollLeft;
  pan('pointerdown'); pan('pointermove', { clientX: 0, clientY: 0 });
  assert.equal(viewport.scrollLeft, oldLeft, 'the replaced corner host has no active pointer handlers');
  assert.equal(viewport.classList.contains('plot-pannable'), false);
  assert.deepEqual(liveCharts()[0].getOption().xAxis[0].data, keys);
  mode = content.querySelector('select');
  mode.value = 'correlations'; mode.dispatchEvent(event('change'));
  assert.deepEqual(ctx.recordSelection.selected, keys);
  assert.equal(liveCharts().length, 21);
  let generatedChanges = 0;
  const generated = renderGenQuantitiesDropdown(ctx, [{ name: 'derived', ir: null }], () => generatedChanges++);
  const box = generated.querySelector('input');
  box.checked = true; box.dispatchEvent(event('change'));
  assert.deepEqual(ctx.recordSelection.selected, [...keys, 'derived']);
  assert.ok(ctx.recordSelection.marginalGroups.includes('derived'));
  box.checked = false; box.dispatchEvent(event('change'));
  assert.deepEqual(ctx.recordSelection.selected, keys);
  assert.equal(generatedChanges, 2);
  assert.deepEqual(measure.samples, originalSamples, 'UI actions preserve all sample coordinates');
  disposePlotCharts(ctx);
  assert.equal(liveCharts().length, 0);
  ctx.recordSelection.selected = ['removed'];
  renderRecordMarginals(ctx, measure, 'points', null);
  assert.deepEqual(ctx.recordSelection.selected, keys.slice(0, 4), 'stale selection recovers the same default');
  ctx.recordSelection.selected = ['[1]'];
  renderRecordMarginals(ctx, measure, 'points', null);
  assert.equal(liveCharts().length, 1);
  const single = liveCharts()[0];
  single.resize({ width: 960, height: 540 });
  const baseline = single.convertToPixel({ yAxisIndex: 0 }, 0);
  assert.ok(baseline >= 530 && baseline <= 540, 'large histograms keep compact density-axis space');
  assert.equal(single.getOption().yAxis[0].axisLabel.formatter(0.5), '0.5');
  disposePlotCharts(ctx);
});
