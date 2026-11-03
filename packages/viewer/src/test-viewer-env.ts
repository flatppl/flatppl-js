// @ts-nocheck — test support only; never bundled (index.ts does not import it).
//
// The browser globals a viewer mount needs in Node: the test DOM, headless
// cytoscape, the engine, the in-process sampler worker, and an ECharts stand-in
// that records what each chart was given, so a test can compare plots
// without pixels.

import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { installTestDom } from './test-dom.ts';
import { InProcessWorker } from './test-worker.ts';

const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');
const realCytoscape = require('cytoscape');
const dagre = require('cytoscape-dagre');
realCytoscape.use(dagre);

function digest(value) {
  const text = JSON.stringify(value, (_k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v));
  return createHash('sha256').update(text ?? 'undefined').digest('hex').slice(0, 16);
}

/** What a chart was given: titles, and per series its type, name, length and data digest. */
function summarize(option) {
  const titles = [].concat(option.title || []).map((t) => t && t.text).filter((t) => t != null);
  const series = [].concat(option.series || []).map((s) => ({
    type: s.type ?? null, name: s.name ?? null,
    length: Array.isArray(s.data) || ArrayBuffer.isView(s.data) ? s.data.length : null,
    data: digest(s.data),
  }));
  const out = {};
  if (titles.length) out.titles = titles;
  if (series.length) out.series = series;
  return out;
}

function fakeEcharts() {
  const charts = [];
  function init(dom) {
    let disposed = false;
    const record = { options: [] };
    charts.push(record);
    const chart = {
      _dom: dom,
      setOption(option) { if (option && (option.series || option.title)) record.options.push(summarize(option)); },
      resize() {}, dispose() { disposed = true; }, isDisposed: () => disposed,
      on() {}, off() {}, dispatchAction() {}, clear() {},
      getDom: () => dom, getWidth: () => 400, getHeight: () => 300,
      getZr: () => ({ on() {}, off() {}, setCursorStyle() {} }),
      convertFromPixel: () => [0, 0], convertToPixel: () => [0, 0], containPixel: () => false,
      getOption: () => ({}),
      getModel: () => ({ eachComponent() {}, eachSeries() {} }),
    };
    dom._chart = chart;
    return chart;
  }
  return {
    charts,
    api: {
      init,
      getInstanceByDom: (dom) => (dom && dom._chart && !dom._chart.isDisposed() ? dom._chart : null),
      format: { getTextRect: (text) => ({ width: String(text).length * 7, height: 12 }) },
    },
  };
}

/** Install the globals; returns handles plus restore(). */
export function installViewerEnv() {
  const dom = installTestDom();
  const names = ['cytoscape', 'echarts', 'FlatPPLEngine', 'Worker'];
  const saved = Object.fromEntries(names.map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  const define = (k, v) => Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
  const graphs = [];
  // The browser renders into a canvas; headless cytoscape keeps the same model.
  const cytoscape = (opts) => {
    const cy = realCytoscape({ ...opts, container: undefined, headless: true, styleEnabled: true });
    graphs.push(cy);
    return cy;
  };
  cytoscape.use = () => {};
  const charts = fakeEcharts();
  define('cytoscape', cytoscape);
  define('echarts', charts.api);
  define('FlatPPLEngine', engine);
  define('Worker', InProcessWorker);
  dom.window.__FLATPPL_CONFIG__ = { samplerWorkerUrl: 'sampler-worker.min.js', wasmApiUrl: null };
  return {
    dom, graphs, charts: charts.charts, engine,
    restore() {
      for (const [k, d] of Object.entries(saved)) {
        if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k];
      }
      dom.restore();
    },
  };
}

/** The graph model: node ids with classes and selection, and edges. */
export function graphModel(cy) {
  if (!cy) return null;
  return {
    nodes: cy.nodes().map((n) => {
      const out = { id: n.id() };
      const classes = n.classes();
      if (classes.length) out.classes = [...classes].sort();
      if (n.selected()) out.selected = true;
      return out;
    }).sort((a, b) => (a.id < b.id ? -1 : 1)),
    edges: cy.edges().map((e) => e.source().id() + '->' + e.target().id()).sort(),
    selected: cy.$(':selected').map((e) => e.id()).sort(),
  };
}
