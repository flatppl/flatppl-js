'use strict';

// Tests for engine/worker.js (transport-agnostic handler) and the
// engine/worker-entry.js shim driven via Node worker_threads.
//
// Coverage:
//   - init / setEnv / setSeed / sample / density / evaluate / dispose
//     happy paths via direct handle() calls
//   - Reproducibility: same seed → same sample sequence
//   - State isolation: two handlers with the same seed yield identical
//     streams and don't interfere
//   - Per-request env overlay: caller-supplied env merges; worker-level
//     env wins on conflict
//   - transferablesOf returns the right ArrayBuffer set
//   - Error replies on unknown type / bad ir / non-positive count
//   - End-to-end through Node worker_threads: spawn the entry shim,
//     postMessage round-trips, verify a reply, close cleanly

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { Worker } = require('node:worker_threads');
const { join } = require('node:path');

const { createWorkerHandler, transferablesOf } = require('../worker');

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}

function distIR(op, kwargs) {
  const out = {};
  for (const [k, v] of Object.entries(kwargs)) {
    out[k] = { kind: 'lit', value: v, loc: synthLoc() };
  }
  return { kind: 'call', op, kwargs: out, loc: synthLoc() };
}

function refIR(name) {
  return { kind: 'ref', ns: 'self', name, loc: synthLoc() };
}

// =====================================================================
// Direct handler tests — these don't spawn a worker, they call the
// handler synchronously to keep tests fast and easy to debug.
// =====================================================================

test('init: returns ready, replaces seed and env', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'init', id: 1, seed: 42, env: { x: 3.14 } });
  assert.deepEqual(r, { type: 'ready', id: 1 });
  assert.equal(w._inspect().env.x, 3.14);
});

test('sample: count must be positive', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'sample', id: 7, ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 0 });
  assert.equal(r.type, 'error');
  assert.equal(r.id, 7);
  assert.match(r.message, /count must be positive/);
});

test('sample: returns Float64Array of requested length', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'sample', id: 1, ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 100 });
  assert.equal(r.type, 'samples');
  assert.equal(r.id, 1);
  assert.ok(r.samples instanceof Float64Array);
  assert.equal(r.samples.length, 100);
  for (let i = 0; i < 100; i++) assert.ok(Number.isFinite(r.samples[i]));
});

test('sample: reproducibility — same seed yields identical samples', () => {
  const irN = distIR('Normal', { mu: 0, sigma: 1 });
  const a = createWorkerHandler();
  const b = createWorkerHandler();
  a.handle({ type: 'init', seed: 12345 });
  b.handle({ type: 'init', seed: 12345 });
  const ra = a.handle({ type: 'sample', ir: irN, count: 50 });
  const rb = b.handle({ type: 'sample', ir: irN, count: 50 });
  for (let i = 0; i < 50; i++) assert.equal(ra.samples[i], rb.samples[i]);
});

test('sample: different seeds yield different sample sequences', () => {
  const irN = distIR('Normal', { mu: 0, sigma: 1 });
  const a = createWorkerHandler();
  const b = createWorkerHandler();
  a.handle({ type: 'init', seed: 1 });
  b.handle({ type: 'init', seed: 2 });
  const ra = a.handle({ type: 'sample', ir: irN, count: 20 });
  const rb = b.handle({ type: 'sample', ir: irN, count: 20 });
  let differing = 0;
  for (let i = 0; i < 20; i++) if (ra.samples[i] !== rb.samples[i]) differing++;
  assert.ok(differing > 15, `expected most samples to differ, got ${differing}/20`);
});

test('sample: RNG state advances across requests', () => {
  const irN = distIR('Normal', { mu: 0, sigma: 1 });
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 7 });
  const first = w.handle({ type: 'sample', ir: irN, count: 5 });
  const second = w.handle({ type: 'sample', ir: irN, count: 5 });
  // If state advanced, the two batches' samples should differ. (They could
  // accidentally match by floating-point coincidence, but with 5 values
  // from a continuous distribution the probability is effectively zero.)
  let same = 0;
  for (let i = 0; i < 5; i++) if (first.samples[i] === second.samples[i]) same++;
  assert.ok(same < 5, 'state did not advance between sample requests');
});

test('setSeed: re-seeds without touching env', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1, env: { foo: 99 } });
  w.handle({ type: 'setSeed', seed: 1 });
  assert.equal(w._inspect().env.foo, 99);
  // After setSeed back to 1, samples should match a fresh seed-1 handler.
  const fresh = createWorkerHandler();
  fresh.handle({ type: 'init', seed: 1 });
  const irN = distIR('Normal', { mu: 0, sigma: 1 });
  const ra = w.handle({ type: 'sample', ir: irN, count: 10 });
  const rb = fresh.handle({ type: 'sample', ir: irN, count: 10 });
  for (let i = 0; i < 10; i++) assert.equal(ra.samples[i], rb.samples[i]);
});

test('setEnv: merge=true keeps existing keys (default)', () => {
  const w = createWorkerHandler({ env: { a: 1, b: 2 } });
  w.handle({ type: 'setEnv', env: { b: 20, c: 3 } });
  const env = w._inspect().env;
  assert.equal(env.a, 1);
  assert.equal(env.b, 20);
  assert.equal(env.c, 3);
});

test('setEnv: merge=false replaces env entirely', () => {
  const w = createWorkerHandler({ env: { a: 1, b: 2 } });
  w.handle({ type: 'setEnv', env: { c: 3 }, merge: false });
  const env = w._inspect().env;
  assert.equal(env.a, undefined);
  assert.equal(env.b, undefined);
  assert.equal(env.c, 3);
});

test('sample: ref params resolve from worker env', () => {
  const w = createWorkerHandler({ env: { mu: 100, sigma: 0.001 } });
  w.handle({ type: 'init', seed: 42, env: { mu: 100, sigma: 0.001 } });
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: { mu: refIR('mu'), sigma: refIR('sigma') },
    loc: synthLoc(),
  };
  const r = w.handle({ type: 'sample', ir, count: 1000 });
  assert.equal(r.type, 'samples');
  // sigma is tiny, so all samples should be near mu=100.
  let mean = 0;
  for (let i = 0; i < 1000; i++) mean += r.samples[i];
  mean /= 1000;
  assert.ok(Math.abs(mean - 100) < 0.01, `mean ${mean} not near 100`);
});

test('sample: per-request env overlays (worker env wins on conflict)', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1, env: { mu: 50 } });
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: { mu: refIR('mu'), sigma: { kind: 'lit', value: 0.001, loc: synthLoc() } },
    loc: synthLoc(),
  };
  // Per-request env supplies sigma but tries to override mu — worker mu wins.
  const r = w.handle({ type: 'sample', ir, count: 50, env: { mu: 0 } });
  assert.equal(r.type, 'samples');
  let mean = 0;
  for (let i = 0; i < 50; i++) mean += r.samples[i];
  mean /= 50;
  assert.ok(Math.abs(mean - 50) < 0.5, `mean ${mean} should track worker-env mu=50, not request mu=0`);
});

test('density: continuous distribution returns Lebesgue ref + Float64Arrays', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'density', ir: distIR('Normal', { mu: 0, sigma: 1 }) });
  assert.equal(r.type, 'density');
  assert.equal(r.reference, 'lebesgue');
  assert.ok(r.xs instanceof Float64Array);
  assert.ok(r.ys instanceof Float64Array);
  assert.equal(r.xs.length, r.ys.length);
  assert.ok(r.xs.length > 50);
});

test('density: discrete distribution returns counting ref', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'density', ir: distIR('Poisson', { rate: 3 }) });
  assert.equal(r.reference, 'counting');
  // All x values are non-negative integers.
  for (let i = 0; i < r.xs.length; i++) {
    assert.ok(Number.isInteger(r.xs[i]));
    assert.ok(r.xs[i] >= 0);
  }
});

test('density: opts pass through (gridPoints)', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'density', ir: distIR('Normal', { mu: 0, sigma: 1 }), opts: { gridPoints: 50 } });
  assert.equal(r.xs.length, 50);
});

test('evaluate: literal returns its value', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'evaluate', id: 9, ir: { kind: 'lit', value: 7.5, loc: synthLoc() } });
  assert.deepEqual(r, { type: 'value', id: 9, value: 7.5 });
});

test('evaluate: ref looks up env', () => {
  const w = createWorkerHandler({ env: { x: 41 } });
  const r = w.handle({ type: 'evaluate', ir: refIR('x') });
  assert.equal(r.value, 41);
});

test('evaluate: per-request env overlay', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'evaluate', ir: refIR('y'), env: { y: 13 } });
  assert.equal(r.value, 13);
});

test('error: unknown message type', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'banana', id: 5 });
  assert.equal(r.type, 'error');
  assert.equal(r.id, 5);
  assert.match(r.message, /unknown message type/);
});

test('error: unknown distribution propagates from sampler', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'sample', ir: distIR('NotADistribution', {}), count: 1 });
  assert.equal(r.type, 'error');
});

test('dispose: returns null and clears state', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'dispose' });
  assert.equal(r, null);
  assert.equal(w._inspect().philox, null);
  assert.equal(w._inspect().env, null);
});

// ---------------------------------------------------------------------
// transferablesOf
// ---------------------------------------------------------------------

test('transferablesOf: samples reply lists the Float64Array buffer', () => {
  const buf = new Float64Array([1, 2, 3]);
  const t = transferablesOf({ type: 'samples', samples: buf });
  assert.equal(t.length, 1);
  assert.equal(t[0], buf.buffer);
});

test('transferablesOf: density reply lists xs and ys buffers', () => {
  const xs = new Float64Array([0, 1]);
  const ys = new Float64Array([0.5, 0.5]);
  const t = transferablesOf({ type: 'density', xs, ys });
  assert.equal(t.length, 2);
  assert.ok(t.includes(xs.buffer));
  assert.ok(t.includes(ys.buffer));
});

test('transferablesOf: other reply types yield empty list', () => {
  assert.deepEqual(transferablesOf({ type: 'ready' }), []);
  assert.deepEqual(transferablesOf({ type: 'value', value: 1 }), []);
  assert.deepEqual(transferablesOf(null), []);
});

// =====================================================================
// End-to-end: spawn the entry shim in a Node worker_thread and exchange
// real postMessage round-trips. This validates worker-entry.js's wiring.
// =====================================================================

test('entry shim: full round-trip via worker_threads (init, sample, density, dispose)', async () => {
  const entry = join(__dirname, '..', 'worker-entry.js');
  const worker = new Worker(entry);

  // Helper: send msg, await first reply matching id.
  function rpc(msg) {
    return new Promise((resolve, reject) => {
      const id = (msg.id != null) ? msg.id : Math.floor(Math.random() * 1e9);
      const wrapped = { ...msg, id };
      const onMsg = (reply) => {
        if (reply.id !== id) return;
        worker.off('message', onMsg);
        worker.off('error', onErr);
        resolve(reply);
      };
      const onErr = (e) => {
        worker.off('message', onMsg);
        worker.off('error', onErr);
        reject(e);
      };
      worker.on('message', onMsg);
      worker.on('error', onErr);
      worker.postMessage(wrapped);
    });
  }

  try {
    const ready = await rpc({ type: 'init', seed: 99 });
    assert.equal(ready.type, 'ready');

    const samples = await rpc({ type: 'sample', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 25 });
    assert.equal(samples.type, 'samples');
    assert.equal(samples.samples.length, 25);
    assert.ok(samples.samples instanceof Float64Array);

    const dens = await rpc({ type: 'density', ir: distIR('Exponential', { rate: 2 }) });
    assert.equal(dens.type, 'density');
    assert.equal(dens.reference, 'lebesgue');
    assert.ok(dens.xs.length > 0);

    // dispose: no reply, but worker should exit on its own (parentPort.close).
    worker.postMessage({ type: 'dispose' });
    const code = await new Promise((res) => worker.on('exit', res));
    assert.equal(code, 0);
  } finally {
    // If anything above threw, make sure the worker doesn't outlive the test.
    await worker.terminate().catch(() => {});
  }
});

// =====================================================================
// drawN — N-sample sampling with per-i ref env. The orchestrator's
// sample-step IRs go through this primitive on the main thread; tests
// directly exercise it via the handler.
// =====================================================================

test('drawN: count must be positive', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'drawN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 0, seed: 1 });
  assert.equal(r.type, 'error');
});

test('drawN: returns Float64Array of requested length', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'drawN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 100, seed: 1 });
  assert.equal(r.type, 'samples');
  assert.equal(r.samples.length, 100);
  for (let i = 0; i < 100; i++) assert.ok(Number.isFinite(r.samples[i]));
});

test('drawN: same (ir, seed) yields identical samples regardless of session state', () => {
  // Per-call seeding is the whole point of drawN: the main-thread
  // cache should get deterministic per-binding output, independent
  // of arrival order.
  const a = createWorkerHandler();
  const b = createWorkerHandler();
  a.handle({ type: 'init', seed: 999 });
  // Burn some session RNG on b — drawN with explicit seed should
  // ignore the session state.
  b.handle({ type: 'init', seed: 1 });
  b.handle({ type: 'sample', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 50 });
  const ra = a.handle({ type: 'drawN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 50, seed: 7 });
  const rb = b.handle({ type: 'drawN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 50, seed: 7 });
  for (let i = 0; i < 50; i++) assert.equal(ra.samples[i], rb.samples[i]);
});

test('drawN: refArrays supply per-i values for ref kwargs', () => {
  // A Normal whose mu is bound by a per-i array of values clustered
  // tightly around 100. Result samples should also cluster around 100.
  const w = createWorkerHandler();
  const muArr = new Float64Array(200);
  for (let i = 0; i < 200; i++) muArr[i] = 100;
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: {
      mu:    { kind: 'ref', ns: 'self', name: 'mu', loc: synthLoc() },
      sigma: { kind: 'lit', value: 0.001, loc: synthLoc() },
    },
    loc: synthLoc(),
  };
  const r = w.handle({ type: 'drawN', ir, count: 200, refArrays: { mu: muArr }, seed: 3 });
  assert.equal(r.type, 'samples');
  let mean = 0;
  for (let i = 0; i < 200; i++) mean += r.samples[i];
  mean /= 200;
  assert.ok(Math.abs(mean - 100) < 0.01);
});

test('drawN: without seed, advances session RNG', () => {
  // Two calls with no explicit seed should produce different output —
  // the session state advances between them.
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 5 });
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const r1 = w.handle({ type: 'drawN', ir, count: 5 });
  const r2 = w.handle({ type: 'drawN', ir, count: 5 });
  let same = 0;
  for (let i = 0; i < 5; i++) if (r1.samples[i] === r2.samples[i]) same++;
  assert.ok(same < 5, 'session RNG did not advance between unseeded drawN calls');
});

// =====================================================================
// evaluateN — element-wise deterministic compute over per-i refArrays.
// =====================================================================

test('evaluateN: count must be positive', () => {
  const w = createWorkerHandler();
  const ir = { kind: 'lit', value: 1, loc: synthLoc() };
  const r = w.handle({ type: 'evaluateN', ir, count: 0 });
  assert.equal(r.type, 'error');
});

test('evaluateN: literal IR yields constant array', () => {
  const w = createWorkerHandler();
  const r = w.handle({
    type: 'evaluateN',
    ir: { kind: 'lit', value: 7.5, loc: synthLoc() },
    count: 10,
  });
  for (let i = 0; i < 10; i++) assert.equal(r.samples[i], 7.5);
});

test('evaluateN: arithmetic on refArrays is element-wise', () => {
  const w = createWorkerHandler();
  const muArr = new Float64Array([1, 2, 3, 4, 5]);
  const ir = {
    kind: 'call', op: 'add',
    args: [
      { kind: 'ref', ns: 'self', name: 'mu', loc: synthLoc() },
      { kind: 'lit', value: 100, loc: synthLoc() },
    ],
    loc: synthLoc(),
  };
  const r = w.handle({ type: 'evaluateN', ir, count: 5, refArrays: { mu: muArr } });
  for (let i = 0; i < 5; i++) assert.equal(r.samples[i], muArr[i] + 100);
});

test('evaluateN: missing ref array → error from evaluator', () => {
  const w = createWorkerHandler();
  const ir = { kind: 'ref', ns: 'self', name: 'missing', loc: synthLoc() };
  const r = w.handle({ type: 'evaluateN', ir, count: 3, refArrays: {} });
  assert.equal(r.type, 'error');
});


// =====================================================================
// (continuing the entry-shim end-to-end tests from before)
// =====================================================================

test('entry shim: error replies survive postMessage', async () => {
  const entry = join(__dirname, '..', 'worker-entry.js');
  const worker = new Worker(entry);
  try {
    const reply = await new Promise((resolve, reject) => {
      worker.once('message', resolve);
      worker.once('error', reject);
      worker.postMessage({ type: 'banana', id: 42 });
    });
    assert.equal(reply.type, 'error');
    assert.equal(reply.id, 42);
    assert.match(reply.message, /unknown message type/);
  } finally {
    await worker.terminate();
  }
});
