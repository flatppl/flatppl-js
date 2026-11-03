import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

/** Load pane-session.ts as the page does: a plain script installing a global. */
function loadSession(globals = {}) {
  const file = new URL('../src/pane-session.ts', import.meta.url);
  const context = { crypto: globalThis.crypto, structuredClone, setInterval, clearInterval, ...globals };
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(file, 'utf8')), context, { filename: file.pathname });
  return context.FlatPPLPaneSession;
}

/** An in-memory BroadcastChannel: same-name instances, sender excluded,
 *  structured clone, delivery in a later task. */
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

const turn = () => new Promise((resolve) => setImmediate(resolve));

/** A transport the test drives: deliver(raw) is a message arriving. */
function testTransport() {
  const sent = [];
  let handler = null;
  return {
    sent,
    deliver: (raw, source = null) => handler(raw, source),
    post: (e) => sent.push(e),
    listen: (h) => { handler = h; },
    close() {},
  };
}


/** A clock the test advances by hand. */
function manualClock() {
  let now = 0;
  const intervals = new Map();
  let next = 0;
  return {
    now: () => now,
    setInterval(fn, ms) { const id = ++next; intervals.set(id, { fn, ms, due: now + ms }); return id; },
    clearInterval(id) { intervals.delete(id); },
    advance(ms) {
      const end = now + ms;
      for (;;) {
        let first = null;
        for (const [id, t] of intervals) if (t.due <= end && (!first || t.due < first[1].due)) first = [id, t];
        if (!first) break;
        now = first[1].due;
        first[1].due += first[1].ms;
        first[1].fn();
      }
      now = end;
    },
  };
}

function pair(S, Channel, extra = {}) {
  const sid = 'sid1';
  const got = { owner: [], replica: [] };
  const owner = S.createSession({ sid, id: 'owner', transport: S.broadcastTransport(sid, Channel),
    onMessage: (e) => got.owner.push(e), ...extra.owner });
  const replica = S.createSession({ sid, id: 'pane:plot', transport: S.broadcastTransport(sid, Channel),
    onMessage: (e) => got.replica.push(e), ...extra.replica });
  return { owner, replica, got };
}

test('a session pair exchanges addressed envelopes over a broadcast channel', async () => {
  const S = loadSession();
  const Channel = channelBus();
  const { owner, replica, got } = pair(S, Channel);
  const other = S.createSession({ sid: 'sid1', id: 'pane:graph', transport: S.broadcastTransport('sid1', Channel),
    onMessage: () => assert.fail('a message for pane:plot reached pane:graph') });
  replica.send('pane.hello', { pane: 'plot' });
  owner.send('pane.welcome', { snapshot: { rev: 1 } }, 'pane:plot');
  // Replicas ignore each other, addressed or not.
  other.send('intent.select', { binding: 'x' });
  other.send('intent.select', { binding: 'x' }, 'pane:plot');
  await turn(); await turn();
  assert.deepEqual(got.owner.map((e) => [e.type, e.from, e.seq, e.v, e.sid]),
    [['pane.hello', 'pane:plot', 1, 1, 'sid1'], ['intent.select', 'pane:graph', 1, 1, 'sid1']]);
  assert.deepEqual(got.replica.map((e) => [e.type, e.to]), [['pane.welcome', 'pane:plot']]);
  other.close();
});

test('another session id and foreign traffic are ignored', async () => {
  const S = loadSession();
  const Channel = channelBus();
  const { got } = pair(S, Channel);
  const stranger = S.createSession({ sid: 'sid2', id: 'pane:plot', transport: S.broadcastTransport('sid1', Channel),
    onMessage: () => {} });
  stranger.send('pane.hello', {});
  await turn(); await turn();
  assert.deepEqual(got.owner, []);
  const t = testTransport();
  const seen = [];
  S.createSession({ sid: 'sid1', id: 'owner', transport: t, onMessage: (e) => seen.push(e) });
  t.deliver({ hello: 'world' });
  t.deliver({ p: 'flatppl-pane', v: 1, sid: 'sid1', from: 'pane:plot', epoch: 'e', seq: 'x', type: 't', body: {} });
  assert.deepEqual(seen, []);
});

test('a duplicate or older seq is dropped; a new epoch starts over', () => {
  const S = loadSession();
  const Channel = channelBus();
  const epochs = [];
  const t = testTransport();
  const got = [];
  S.createSession({ sid: 'sid1', id: 'owner', transport: t, onMessage: (e) => got.push(e),
    onPeerEpoch: (from, e, prev) => epochs.push([from, e, prev ?? null]) });
  const env = (epoch, seq) => ({ p: 'flatppl-pane', v: 1, sid: 'sid1', from: 'pane:plot', epoch, seq, type: 'intent.select', body: { seq } });
  t.deliver(env('a', 1));
  t.deliver(env('a', 1));
  t.deliver(env('a', 3));
  t.deliver(env('a', 2));
  t.deliver(env('b', 1));
  assert.deepEqual(got.map((e) => e.epoch + e.body.seq), ['a1', 'a3', 'b1']);
  assert.deepEqual(epochs, [['pane:plot', 'a', null], ['pane:plot', 'b', 'a']]);
});

test('an unknown protocol version is reported and dropped', () => {
  const S = loadSession();
  const versions = [];
  const t = testTransport();
  const got = [];
  S.createSession({ sid: 'sid1', id: 'owner', transport: t, onMessage: (e) => got.push(e), onVersion: (v) => versions.push(v) });
  t.deliver({ p: 'flatppl-pane', v: 2, sid: 'sid1', from: 'pane:plot', epoch: 'a', seq: 1, type: 'pane.hello', body: {} });
  assert.deepEqual(versions, [2]);
  assert.deepEqual(got, []);
});

test('the revision tracker applies newer state, drops stale state and reports a new owner epoch', () => {
  const S = loadSession();
  const tracker = S.createRevisionTracker();
  const state = (epoch, rev) => ({ epoch, body: { rev } });
  assert.equal(tracker.accept(state('o1', 1)), 'apply');
  assert.equal(tracker.accept(state('o1', 3)), 'apply', 'a gap is not an error: each message is a full value');
  assert.equal(tracker.accept(state('o1', 2)), 'stale');
  assert.equal(tracker.accept(state('o1', 3)), 'stale');
  assert.equal(tracker.accept({ epoch: 'o1', body: {} }), 'stale');
  assert.equal(tracker.accept(state('o2', 1)), 'new-epoch');
  assert.deepEqual({ ...tracker.position() }, { epoch: 'o2', rev: 1 });
  assert.equal(tracker.accept(state('o2', 2)), 'apply');
  tracker.reset();
  assert.equal(tracker.accept(state('o3', 1)), 'apply');
});

test('the window fallback refuses a foreign origin and reaches its targets', () => {
  const S = loadSession();
  const self = new EventTarget();
  const seen = [];
  const sent = [];
  const target = { postMessage: (data, origin) => sent.push([data.type, origin]) };
  const transport = S.windowTransport({ self, origin: 'https://play.example', targets: () => [target, { postMessage() { throw new Error('closed'); } }] });
  const session = S.createSession({ sid: 's', id: 'owner', transport, onMessage: (e, source) => seen.push([e.type, source]) });
  const message = (origin, source) => Object.assign(new Event('message'), {
    origin, source, data: { p: 'flatppl-pane', v: 1, sid: 's', from: 'pane:math', epoch: 'e', seq: seen.length + 1, type: 'intent.focus', body: {} },
  });
  self.dispatchEvent(message('https://evil.example', target));
  self.dispatchEvent(message('null', target));
  self.dispatchEvent(message('https://play.example', target));
  assert.deepEqual(seen, [['intent.focus', target]]);
  session.send('state.focus', { rev: 1, target: 'x' });
  assert.deepEqual(sent, [['state.focus', 'https://play.example']]);
});

test('defaultTransport falls back to window messaging without BroadcastChannel', () => {
  const self = new EventTarget();
  self.location = { origin: 'https://play.example' };
  const S = loadSession({ addEventListener: self.addEventListener.bind(self),
    removeEventListener: self.removeEventListener.bind(self), location: self.location });
  const sent = [];
  const transport = S.defaultTransport('s', () => [{ postMessage: (d, o) => sent.push(o) }]);
  transport.post({ type: 'x' });
  assert.deepEqual(sent, ['https://play.example']);
  const withChannel = loadSession({ BroadcastChannel: channelBus() });
  assert.equal(typeof withChannel.defaultTransport('s', () => []).post, 'function');
});

test('a silent peer is lost after the timeout only while both sides are visible', async () => {
  const S = loadSession();
  const clock = manualClock();
  let ownerVisible = true;
  const lost = [];
  const t = testTransport();
  const sent = { filter: (f) => t.sent.map((e) => e.type).filter(f) };
  const owner = S.createSession({ sid: 's', id: 'owner', transport: t, clock, isVisible: () => ownerVisible,
    onMessage() {}, onLost: (p) => lost.push(p) });
  owner.watch('pane:plot', 5000, 15000);
  const pong = (seq, visible) => t.deliver({ p: 'flatppl-pane', v: 1, sid: 's', from: 'pane:plot', epoch: 'e', seq, type: 'pong', body: { visible } });

  clock.advance(10000);
  pong(1, true);
  clock.advance(14000);
  assert.deepEqual(lost, [], 'a reply resets the silence');
  assert.ok(sent.filter((t) => t === 'ping').length >= 4);

  // The replica reports itself hidden: its silence proves nothing.
  pong(2, false);
  clock.advance(60000);
  assert.deepEqual(lost, []);

  // Visible again on both sides, then silent past the timeout.
  pong(3, true);
  ownerVisible = false;
  clock.advance(60000);
  assert.deepEqual(lost, [], 'a hidden owner does not count silence');
  ownerVisible = true;
  clock.advance(14000);
  assert.deepEqual(lost, []);
  clock.advance(6000);
  assert.deepEqual(lost, ['pane:plot']);
  owner.close();
  assert.throws(() => owner.send('x'), /closed/);
});

test('a ping is answered with a pong and never reaches the handler', () => {
  const S = loadSession();
  const handled = [];
  const t = testTransport();
  S.createSession({ sid: 's', id: 'pane:graph', transport: t, isVisible: () => true,
    onMessage: (e) => handled.push(e.type) });
  t.deliver({ p: 'flatppl-pane', v: 1, sid: 's', from: 'owner', epoch: 'o', seq: 1, type: 'ping', body: { visible: true } });
  t.deliver({ p: 'flatppl-pane', v: 1, sid: 's', from: 'owner', epoch: 'o', seq: 2, type: 'pong', body: {} });
  assert.deepEqual(t.sent.map((e) => [e.type, e.to, e.body.visible]), [['pong', 'owner', true]]);
  assert.deepEqual(handled, []);
});

test('ids are 128-bit hex', () => {
  const S = loadSession();
  const a = S.randomId();
  assert.match(a, /^[0-9a-f]{32}$/);
  assert.notEqual(S.randomId(), a);
  assert.equal(S.channelName('abc'), 'flatppl-pane:abc');
});

test('watching a peer again restarts its silence count', () => {
  const S = loadSession();
  let now = 0;
  let tick = null;
  const clock = { now: () => now, setInterval: (fn) => { tick = fn; return 1; }, clearInterval() {} };
  const lost = [];
  const t = testTransport();
  const owner = S.createSession({ sid: 's', id: 'owner', transport: t, clock, isVisible: () => true,
    onMessage() {}, onLost: (p) => lost.push(p) });
  owner.watch('pane:plot', 5000, 15000);
  // Timers frozen (a cached page) while the clock moved on.
  now = 60000;
  owner.watch('pane:plot', 5000, 15000);
  tick();
  assert.deepEqual(lost, []);
  now = 76000;
  tick();
  assert.deepEqual(lost, ['pane:plot']);
});
