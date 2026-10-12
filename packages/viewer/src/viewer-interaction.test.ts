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
const { applySourceUpdate, resizeAllEchartsInPlot } = await import('./orchestration.ts');
const { VIEWER_BODY_HTML } = await import('./templates.ts');
const { renderRecordMarginals, renderGenQuantitiesDropdown } = await import('./render-record.ts');
const { renderCornerGrid } = await import('./render-density.ts');
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
  const source = 'flatppl_compat = "0.1"\nx ~ Normal(0, 1000000)\npoints = (x, 2*x, 3*x, 4*x, 5*x, 6*x)';
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
  // A distant tail must not give the histogram and pairs different axes.
  measure.samples.set([1e9, 2e9, 3e9, 4e9, 5e9, 6e9], measure.samples.length - 6);
  const axes = listScalarAxes(measure);
  const keys = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]'];
  assert.deepEqual(axes.map((axis) => axis.key), keys);
  for (let i = 0; i < axes.length; i++) {
    assert.deepEqual(axes[i].samples, axes[0].samples.map((x) => x * (i + 1)));
  }
  const originalSamples = measure.samples.slice();
  const mounted = new Map();
  globalThis.echarts = {
    init(element) {
      const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 220, height: 180 });
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
  const columnX = new Map(), scatterY = new Map(), domains = new Map();
  for (const [element, chart] of mounted) {
    if (chart.isDisposed()) continue;
    const row = parseInt(element.style.gridRow) - 1, col = parseInt(element.style.gridColumn) - 2;
    if (row === col) domains.set(col, chart.getModel().getComponent('xAxis').axis.scale.getExtent());
  }
  for (const [element, chart] of mounted) {
    if (chart.isDisposed()) continue;
    const row = parseInt(element.style.gridRow) - 1, col = parseInt(element.style.gridColumn) - 2;
    const option = chart.getOption(), series = option.series[0];
    assert.deepEqual(chart.getModel().getComponent('xAxis').axis.scale.getExtent(), domains.get(col),
      'histograms and pairs share the column range');
    if (row !== col) assert.deepEqual(chart.getModel().getComponent('yAxis').axis.scale.getExtent(), domains.get(row),
      'pair rows share their variable range with the histogram');
    if (row === col) {
      assert.equal(series.type, 'custom');
      assert.ok(series.data.length > 0, 'each diagonal retains its histogram');
      assert.equal(option.yAxis[0].min, 0, 'density bars start at zero');
      assert.ok(chart.convertToPixel({ xAxisIndex: 0 }, 1e9 * (col + 1)) > chart.getWidth(),
        'the shared range retains the histogram bulk instead of stretching to an outlier');
    } else {
      assert.equal(series.type, 'scatter');
      const points = ArrayBuffer.isView(series.data)
        ? Array.from({ length: series.data.length / 2 }, (_, i) => [series.data[2 * i], series.data[2 * i + 1]])
        : series.data;
      assert.deepEqual(points, Array.from(axes[col].samples, (x, i) => [x, axes[row].samples[i]]));
    }
    const dataBeforeResize = option.series.map((s) => s.data);
    for (const [width, height] of [[52, 34], [12, 8], [56, 36], [80, 52], [96, 60], [180, 110], [220, 180]]) {
      chart.resize({ width, height });
      const points = row === col
        ? [[series.data[0].value[0], 0], [series.data.at(-1).value[0], Math.max(...series.data.map((bin) => bin.value[1]))]]
        : [[domains.get(col)[0], domains.get(row)[0]], [domains.get(col)[1], domains.get(row)[1]]];
      const pixels = points.map((point) => chart.convertToPixel({ seriesIndex: 0 }, point));
      for (const [x, y] of pixels) {
        assert.ok(Number.isFinite(x) && x >= 0 && x <= width, 'x stays inside the resized cell');
        assert.ok(Number.isFinite(y) && y >= 0 && y <= height, 'y stays inside the resized cell');
      }
      assert.ok(pixels[1][0] > pixels[0][0] && pixels[1][1] < pixels[0][1], 'tiny plot areas stay positive');
      const x = chart.convertToPixel({ xAxisIndex: 0 }, axes[col].samples[0]);
      const xKey = width + ':' + col;
      if (columnX.has(xKey)) assert.ok(Math.abs(x - columnX.get(xKey)) < 1e-8, 'histograms and pair columns align');
      columnX.set(xKey, x);
      if (row !== col) {
        const y = chart.convertToPixel({ yAxisIndex: 0 }, axes[row].samples[0]);
        const yKey = height + ':' + row;
        if (scatterY.has(yKey)) assert.ok(Math.abs(y - scatterY.get(yKey)) < 1e-8, 'scatter rows align');
        scatterY.set(yKey, y);
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
      if (width >= 72 && height >= 50 && (col === 0 || row === 5)) {
        assert.ok(labels.length > 0, 'outer ticks stay readable in mobile-sized cells: '
          + JSON.stringify({ row, col, width, height, grid: chart.getOption().grid }));
      }
      const frame = chart.getModel().getComponent('grid').coordinateSystem.getRect();
      for (let i = 0; i < labels.length; i++) {
        const a = labels[i];
        // SVG rounds transforms; allow subpixel serialization error.
        assert.ok(a.left >= -0.01 && a.right <= width + 0.01 && a.top >= -0.01 && a.bottom <= height + 0.01,
          'numeric text stays inside the SVG cell: ' + JSON.stringify({ row, col, width, height, ...a }));
        assert.ok(a.right <= frame.x || a.top >= frame.y + frame.height,
          'numeric text stays outside the data rectangle');
        for (const b of labels.slice(i + 1)) {
          const overlaps = Math.min(a.right, b.right) > Math.max(a.left, b.left)
            && Math.min(a.bottom, b.bottom) > Math.max(a.top, b.top);
          assert.equal(overlaps, false, 'numeric labels do not collide: ' + JSON.stringify({ row, col, width, height, a, b }));
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
  assert.ok(baseline >= 512 && baseline <= 524, 'large histograms leave a compact outside x-axis gutter');
  assert.equal(single.getOption().yAxis[0].axisLabel.formatter(0.5), '0.5');
  disposePlotCharts(ctx);
});

test('weighted corner scatters retain joint pairs and binary64 coordinates', (t) => {
  const doc = mountDOM(t);
  const priorEngine = globalThis.FlatPPLEngine, priorStyle = globalThis.getComputedStyle;
  globalThis.FlatPPLEngine = engine;
  globalThis.getComputedStyle = () => ({ color: '#333' });
  t.after(() => { globalThis.FlatPPLEngine = priorEngine; globalThis.getComputedStyle = priorStyle; });
  const axes = [
    { key: 'x', label: 'x', samples: Float64Array.of(1 + 2 ** -30, 2, 3, 4, 5, 6, 7, 8) },
    { key: 'y', label: 'y', samples: Float64Array.of(8, 7, 6, 5, 4, 3, 2, 1) },
  ];
  const measure = { logWeights: Float64Array.from([1, 1, 2, 4, 0, 0, 0, 0], Math.log) };
  const originals = axes.map(a => a.samples.slice());
  const mounted = new Map();
  globalThis.echarts = {
    init(element) {
      const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 180, height: 110 });
      mounted.set(element, chart);
      return chart;
    },
    getInstanceByDom: element => mounted.get(element),
  };
  t.after(() => { for (const chart of mounted.values()) if (!chart.isDisposed()) chart.dispose(); });
  const ctx = { recordSelection: { selected: ['x', 'y'] }, rootSeed: 42,
    TYPE_STYLE: { draw: { color: '#82b366' }, unknown: { color: '#82b366' } } };
  const host = doc.createElement('div');
  renderCornerGrid(ctx, host, measure, 'points', axes);
  const scatter = [...mounted.values()].find(c => c.getOption().series[0].type === 'scatter');
  const series = scatter.getOption().series[0];
  const points = ArrayBuffer.isView(series.data)
    ? Array.from({ length: series.data.length / 2 }, (_, i) => [series.data[2 * i], series.data[2 * i + 1]])
    : series.data;
  // Eight draws and integer masses give exact systematic-resampling counts.
  assert.deepEqual(points, [0, 1, 2, 2, 3, 3, 3, 3].map(i => [axes[0].samples[i], axes[1].samples[i]]));
  const plotted = scatter.getModel().getSeriesByIndex(0).getData();
  assert.equal(plotted.count(), points.length);
  for (let i = 0; i < plotted.count(); i++) assert.deepEqual([plotted.get('x', i), plotted.get('y', i)], points[i]);
  assert.equal(series.itemStyle.opacity, 0.6);
  assert.match(scatter.renderToSVGString(), /<path/);
  assert.deepEqual(axes.map(a => a.samples), originals);
  disposePlotCharts(ctx, host);
  assert.ok([...mounted.values()].every(c => c.isDisposed()));
});

test('plot resize skips unchanged content sizes and keeps root and padded charts responsive', (t) => {
  const doc = mountDOM(t), root = doc.getElementById('plot-content');
  const child = doc.createElement('div');
  root.appendChild(child);
  Object.assign(root, { clientWidth: 220, clientHeight: 180 });
  Object.assign(child, { clientWidth: 110, clientHeight: 80 });
  const priorStyle = globalThis.getComputedStyle;
  const childStyle = { width: '110px', height: '80px', paddingLeft: '3px', paddingRight: '7px', paddingTop: '5px', paddingBottom: '15px' };
  globalThis.getComputedStyle = node => node === child
    ? childStyle
    : { width: '220px', height: '180px' };
  t.after(() => { globalThis.getComputedStyle = priorStyle; });
  const mounted = new Map(), sizes = [];
  for (const [node, width, height] of [[root, 220, 180], [child, 100, 60]]) {
    const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width, height });
    const resize = chart.resize.bind(chart);
    chart.resize = opts => { sizes.push([opts.width, opts.height]); resize(opts); };
    mounted.set(node, chart);
  }
  globalThis.echarts = { getInstanceByDom: node => mounted.get(node) };
  t.after(() => { for (const chart of mounted.values()) chart.dispose(); });
  resizeAllEchartsInPlot({});
  assert.deepEqual(sizes, [], 'unchanged charts avoid the full ECharts update');
  child.clientWidth = 111;
  root.clientHeight = 190;
  resizeAllEchartsInPlot({});
  assert.deepEqual(sizes, [[220, 190], [101, 60]]);
  resizeAllEchartsInPlot({});
  assert.equal(sizes.length, 2);
  child.clientWidth = 0;
  child.clientHeight = 0;
  resizeAllEchartsInPlot({});
  assert.deepEqual(sizes.at(-1), [100, 60], 'hidden hosts retain ECharts CSS fallback sizing');
  childStyle.width = 'auto'; childStyle.height = 'auto';
  resizeAllEchartsInPlot({});
  assert.deepEqual(sizes.at(-1), [0, 0], 'an unsized hidden host has zero content size');
  child.clientWidth = 120;
  child.clientHeight = 90;
  resizeAllEchartsInPlot({});
  assert.equal(mounted.get(child).getWidth(), 110);
  assert.equal(mounted.get(child).getHeight(), 70);
});
