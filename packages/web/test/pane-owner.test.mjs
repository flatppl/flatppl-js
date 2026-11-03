import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

function source(name) {
  const file = new URL('../src/' + name, import.meta.url);
  return [stripTypeScriptTypes(readFileSync(file, 'utf8')), file.pathname];
}

/** Synchronous in-memory BroadcastChannel: each test step sees its effects at once. */
function channelBus() {
  const open = [];
  class Channel {
    constructor(name) { this.name = name; this.onmessage = null; open.push(this); }
    postMessage(data) {
      const copy = structuredClone(data);
      for (const other of open.slice()) if (other !== this && other.name === this.name) other.onmessage?.({ data: copy });
    }
    close() { open.splice(open.indexOf(this), 1); }
  }
  return Channel;
}

/** pane-owner.js with a recording viewer, hand-run timers and a hand-set clock. */
function bootOwner(Channel) {
  let now = 0;
  const intervals = new Map();
  const timeouts = new Map();
  let nextId = 0;
  const listeners = {};
  const calls = [];
  const view = {
    snapshotPanel: (p) => ({ [p]: { from: 'main' } }),
    exportMeasures: () => null,
    setPanelPlacement: (p, placement) => calls.push(['setPanelPlacement', p, placement]),
    restorePanel: (p, state) => calls.push(['restorePanel', p, structuredClone(state)]),
    importMeasures: (h) => calls.push(['importMeasures', h]),
    select() {},
  };
  const panelEl = { getBoundingClientRect: () => ({ width: 600, height: 400 }), setAttribute() {}, focus() {} };
  class FakeDate extends Date { static now() { return now; } }
  const storage = new Map();
  const context = {
    console, structuredClone, crypto: globalThis.crypto, BroadcastChannel: Channel, Date: FakeDate,
    MutationObserver: class { observe() {} },
    document: { documentElement: { dataset: {} }, getElementById: () => panelEl, hasFocus: () => false,
      visibilityState: 'visible' },
    sessionStorage: { getItem: (k) => storage.get(k) ?? null, setItem: (k, v) => storage.set(k, v), removeItem: (k) => storage.delete(k) },
    location: { origin: 'https://play.example' },
    screenX: 0, screenY: 0,
    open: () => ({ closed: false, close() { this.closed = true; }, postMessage() {} }),
    focus() {},
    addEventListener: (type, fn) => { (listeners[type] ||= []).push(fn); },
    setInterval: (fn, ms) => { const id = ++nextId; intervals.set(id, { fn, ms }); return id; },
    clearInterval: (id) => intervals.delete(id),
    setTimeout: (fn, ms) => { const id = ++nextId; timeouts.set(id, { fn, ms }); return id; },
    clearTimeout: (id) => timeouts.delete(id),
  };
  context.window = context;
  for (const name of ['pane-session.ts', 'pane-owner.ts']) {
    const [code, filename] = source(name);
    vm.runInNewContext(code, context, { filename });
  }
  const owner = context.FlatPPLPaneOwner.install({
    controller: { update() {}, viewerHandle: () => view, currentType: () => 'flatppl' },
    activePath: () => 'examples/m.flatppl',
    navigate() {},
    host: { canPersist: () => false, openModule() {}, revealSourceLine() {}, editSource() {} },
    announce() {},
  });
  return {
    owner, calls, context, listeners,
    sid: () => JSON.parse(storage.get('flatppl-pane-session-v1')).sid,
    setNow: (t) => { now = t; },
    /** Run every interval whose callback is still registered, once. */
    tickIntervals: () => { for (const { fn } of Array.from(intervals.values())) fn(); },
    /** Run every pending timeout, once. */
    runTimeouts: () => { const due = Array.from(timeouts.values()); timeouts.clear(); for (const { fn } of due) fn(); },
  };
}

/** The panel window's side, speaking the protocol over the shared channel. */
function replica(context, sid, pane, Channel) {
  const S = context.FlatPPLPaneSession;
  const got = [];
  const session = S.createSession({ sid, id: 'pane:' + pane, transport: S.broadcastTransport(sid, Channel),
    onMessage: (e) => got.push(e.type), isVisible: () => true });
  return { session, got };
}

test('Dock while a panel window reloads restores the state it sent with its goodbye', () => {
  const Channel = channelBus();
  const env = bootOwner(Channel);
  env.owner.detach('graph');
  const panel = replica(env.context, env.sid(), 'graph', Channel);
  panel.session.send('pane.hello', { pane: 'graph' }, 'owner');
  assert.ok(panel.got.includes('pane.welcome'));
  const sent = { graph: { collapsedReifications: ['g'], viewport: { zoom: 2, pan: { x: 1, y: 2 } }, history: [] } };
  panel.session.send('pane.bye', { reason: 'close', state: sent }, 'owner');
  // The user clicks the toggle before the goodbye grace ends.
  env.owner.redock('graph');
  assert.deepEqual(env.calls.filter(([n]) => n === 'restorePanel'), [['restorePanel', 'graph', sent]]);
  assert.deepEqual(env.calls.filter(([n]) => n === 'setPanelPlacement').pop(), ['setPanelPlacement', 'graph', 'docked']);
});

test('after a stay in the back/forward cache, live panel windows are not declared lost', () => {
  const Channel = channelBus();
  const env = bootOwner(Channel);
  env.owner.detach('plot');
  const panel = replica(env.context, env.sid(), 'plot', Channel);
  panel.session.send('pane.hello', { pane: 'plot' }, 'owner');
  // The page goes into the cache; its timers freeze while the clock moves 60 s.
  env.listeners.pagehide.forEach((fn) => fn({ persisted: true }));
  env.setNow(60000);
  env.listeners.pageshow.forEach((fn) => fn({ persisted: true }));
  env.tickIntervals();
  assert.deepEqual(env.calls.filter(([n, , placement]) => n === 'setPanelPlacement' && placement === 'docked'), [],
    'the live plot window stays detached');
  assert.deepEqual([...env.owner.detached()], ['plot']);
});
