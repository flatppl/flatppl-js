import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

const SID = 'ab'.repeat(16);

function source(name) {
  const file = new URL('../src/' + name, import.meta.url);
  return [stripTypeScriptTypes(readFileSync(file, 'utf8')), file.pathname];
}

/** In-memory BroadcastChannel shared by the panel window and the test's owner. */
function channelBus() {
  const open = [];
  class Channel {
    constructor(name) { this.name = name; this.onmessage = null; open.push(this); }
    postMessage(data) {
      const copy = structuredClone(data);
      for (const other of open) {
        if (other !== this && other.name === this.name) setImmediate(() => other.onmessage?.({ data: copy }));
      }
    }
    close() { open.splice(open.indexOf(this), 1); }
  }
  return Channel;
}

function element() {
  return { textContent: '', hidden: false, onclick: null, listeners: {},
    setAttribute() {}, focus() {}, addEventListener(type, fn) { this.listeners[type] = fn; } };
}

/** Boot pane-app.js for a plots window with a recording viewer stand-in.
 *  Timers are recorded, not run: the test fires them by hand. */
function bootPaneWindow(Channel) {
  const ids = ['pane-status', 'pane-banner', 'pane-banner-text', 'pane-banner-action', 'pane-title',
    'surface-root', 'pane-dock', 'pane-main'];
  const elements = Object.fromEntries(ids.map((id) => [id, element()]));
  const calls = [];
  const intervals = [];
  const view = new Proxy({}, {
    get: (_t, name) => (...args) => {
      calls.push([name, ...args.map((a) => structuredClone(a ?? null))]);
      if (name === 'snapshotPanel') return { plot: {} };
      if (name === 'exportMeasures') return null;
      return undefined;
    },
  });
  const context = {
    console, structuredClone, URLSearchParams, Date, crypto: globalThis.crypto,
    BroadcastChannel: Channel,
    location: { search: '?pane=plot', hash: '#sid=' + SID, origin: 'https://play.example' },
    document: { readyState: 'complete', title: '', documentElement: { dataset: {} },
      visibilityState: 'visible', getElementById: (id) => elements[id] },
    FlatPPLViewer: { mount: () => view },
    setInterval: (fn) => { intervals.push(fn); return intervals.length; },
    clearInterval: (id) => { intervals[id - 1] = null; },
    setTimeout: () => 0, clearTimeout() {},
    addEventListener() {}, close() {}, opener: null,
  };
  context.window = context;
  for (const name of ['pane-session.ts', 'pane-app.ts']) {
    const [code, filename] = source(name);
    vm.runInNewContext(code, context, { filename });
  }
  return { calls, intervals, elements };
}

const turn = () => new Promise((resolve) => setImmediate(resolve));

test('two hellos bring two welcomes; only the first applies and the parked samples survive', async () => {
  const Channel = channelBus();
  // The session module, loaded once more for the test's owner endpoint.
  const ownerScope = { crypto: globalThis.crypto, structuredClone, setInterval, clearInterval, BroadcastChannel: Channel };
  const [sessionCode, sessionFile] = source('pane-session.ts');
  vm.runInNewContext(sessionCode, ownerScope, { filename: sessionFile });
  const S = ownerScope.FlatPPLPaneSession;
  const handover = { binding: 'x', validity: { modelKey: 'm', settingsKey: 's' }, entries: [['x', { n_eff: 1 }]], bytes: 8 };
  const model = { path: 'examples/m.flatppl', type: 'flatppl', source: 'x = 1', preview: false, bundle: null };
  let hellos = 0;
  const owner = S.createSession({
    sid: SID, id: 'owner', transport: S.broadcastTransport(SID, Channel),
    onMessage(env) {
      if (env.type !== 'pane.hello') return;
      hellos++;
      // As pane-owner.ts does: the samples go with the first hello only.
      if (hellos === 1) owner.send('pane.measures', handover, env.from);
      owner.send('pane.welcome', { snapshot: { rev: 1, model, focus: 'x', select: null, settings: {} },
        panelState: hellos === 1 ? { plot: { sampleCount: 7 } } : null }, env.from);
    },
  });

  const pane = bootPaneWindow(Channel);
  // A second hello before the first welcome arrives: the retry timer fires.
  pane.intervals.filter(Boolean).forEach((fn) => fn());
  for (let i = 0; i < 6; i++) await turn();

  assert.equal(hellos, 2, 'the owner saw two hellos');
  const imports = pane.calls.filter(([name]) => name === 'importMeasures');
  assert.deepEqual(imports, [['importMeasures', handover]], 'one import, the parked samples, never null');
  assert.equal(pane.calls.filter(([name]) => name === 'update').length, 1, 'the model applied once');
  assert.equal(pane.elements['pane-status'].textContent, 'Connected to the main window.');
  owner.close();
});
