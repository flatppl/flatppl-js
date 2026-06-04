'use strict';

// Tests for engine/worker.js (transport-agnostic handler) and the
// engine/worker-entry.ts shim driven via Node worker_threads.
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

const { createWorkerHandler, transferablesOf } = require('../worker.ts');

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}

function distIR(op: any, kwargs: any) {
  const out: any = {};
  for (const [k, v] of Object.entries(kwargs)) {
    out[k] = { kind: 'lit', value: v, loc: synthLoc() };
  }
  return { kind: 'call', op, kwargs: out, loc: synthLoc() };
}

function refIR(name: any) {
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
// real postMessage round-trips. This validates worker-entry.ts's wiring.
// =====================================================================

test('entry shim: full round-trip via worker_threads (init, sample, density, dispose)', async () => {
  const entry = join(__dirname, '..', 'worker-entry.ts');
  const worker = new Worker(entry);

  // Helper: send msg, await first reply matching id.
  function rpc(msg: any) {
    return new Promise((resolve, reject) => {
      const id = (msg.id != null) ? msg.id : Math.floor(Math.random() * 1e9);
      const wrapped = { ...msg, id };
      const onMsg = (reply: any) => {
        if (reply.id !== id) return;
        worker.off('message', onMsg);
        worker.off('error', onErr);
        resolve(reply);
      };
      const onErr = (e: any) => {
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
    const ready: any = await rpc({ type: 'init', seed: 99 });
    assert.equal(ready.type, 'ready');

    const samples: any = await rpc({ type: 'sample', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 25 });
    assert.equal(samples.type, 'samples');
    assert.equal(samples.samples.length, 25);
    assert.ok(samples.samples instanceof Float64Array);

    const dens: any = await rpc({ type: 'density', ir: distIR('Exponential', { rate: 2 }) });
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
// sampleN — N-sample sampling with per-i ref env. The orchestrator's
// sample-step IRs go through this primitive on the main thread; tests
// directly exercise it via the handler.
// =====================================================================

test('sampleN: count must be positive', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'sampleN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 0, seed: 1 });
  assert.equal(r.type, 'error');
});

test('sampleN: returns Float64Array of requested length', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'sampleN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 100, seed: 1 });
  assert.equal(r.type, 'samples');
  assert.equal(r.samples.length, 100);
  for (let i = 0; i < 100; i++) assert.ok(Number.isFinite(r.samples[i]));
});

test('sampleN: reply carries the EmpiricalMeasure shape (samples + logWeights)', () => {
  // Variates and i.i.d. draws are unweighted by construction. The
  // worker emits logWeights: null so the main-thread cache wraps it
  // as { samples, logWeights: null } directly. Once weighted ops land
  // this slot will hold an explicit Float64Array for those cases.
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'sampleN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 10, seed: 1 });
  assert.equal(r.type, 'samples');
  assert.ok('logWeights' in r, 'reply must include logWeights field');
  assert.equal(r.logWeights, null, 'unweighted draws → logWeights: null');
});

test('evaluateN: reply carries the EmpiricalMeasure shape', () => {
  const w = createWorkerHandler();
  const r = w.handle({
    type: 'evaluateN',
    ir: { kind: 'lit', value: 1, loc: synthLoc() },
    count: 5,
  });
  assert.equal(r.type, 'samples');
  assert.equal(r.logWeights, null);
});

test('sampleN: same (ir, seed) yields identical samples regardless of session state', () => {
  // Per-call seeding is the whole point of sampleN: the main-thread
  // cache should get deterministic per-binding output, independent
  // of arrival order.
  const a = createWorkerHandler();
  const b = createWorkerHandler();
  a.handle({ type: 'init', seed: 999 });
  // Burn some session RNG on b — sampleN with explicit seed should
  // ignore the session state.
  b.handle({ type: 'init', seed: 1 });
  b.handle({ type: 'sample', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 50 });
  const ra = a.handle({ type: 'sampleN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 50, seed: 7 });
  const rb = b.handle({ type: 'sampleN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 50, seed: 7 });
  for (let i = 0; i < 50; i++) assert.equal(ra.samples[i], rb.samples[i]);
});

test('sampleN: Dirac emits N copies of the value (static path)', () => {
  // Dirac is a degenerate distribution: every "draw" returns the
  // value parameter. The worker's static-params fast path builds
  // one sampler instance and calls it count times; for Dirac that
  // closure just returns the baked-in value.
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'sampleN', ir: distIR('Dirac', { value: 7.5 }), count: 64, seed: 1 });
  assert.equal(r.type, 'samples');
  assert.equal(r.samples.length, 64);
  for (let i = 0; i < 64; i++) assert.equal(r.samples[i], 7.5);
});

test('sampleN: Dirac with refArrays evaluates the value per atom', () => {
  // The per-i-params path: the value kwarg is a ref, refArrays
  // supplies a per-atom Float64Array. The Dirac sampler's parametric
  // factory returns each atom's value verbatim.
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const xs = new Float64Array([1.5, 2.5, 3.5, 4.5]);
  const ir = { kind: 'call', op: 'Dirac', kwargs: {
    value: refIR('x'),
  }, loc: synthLoc() };
  const r = w.handle({ type: 'sampleN', ir, count: 4, seed: 1, refArrays: { x: xs } });
  assert.equal(r.type, 'samples');
  assert.equal(r.samples.length, 4);
  for (let i = 0; i < 4; i++) assert.equal(r.samples[i], xs[i]);
});

test('sampleN: refArrays supply per-i values for ref kwargs', () => {
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
  const r = w.handle({ type: 'sampleN', ir, count: 200, refArrays: { mu: muArr }, seed: 3 });
  assert.equal(r.type, 'samples');
  let mean = 0;
  for (let i = 0; i < 200; i++) mean += r.samples[i];
  mean /= 200;
  assert.ok(Math.abs(mean - 100) < 0.01);
});

test('sampleN: without seed, advances session RNG', () => {
  // Two calls with no explicit seed should produce different output —
  // the session state advances between them.
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 5 });
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const r1 = w.handle({ type: 'sampleN', ir, count: 5 });
  const r2 = w.handle({ type: 'sampleN', ir, count: 5 });
  let same = 0;
  for (let i = 0; i < 5; i++) if (r1.samples[i] === r2.samples[i]) same++;
  assert.ok(same < 5, 'session RNG did not advance between unseeded sampleN calls');
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
// logDensityN — per-i scoring via density.ts (logDensityConsumeN).
// Builds on the density unit tests; here we verify the worker plumbing
// (refArrays per-i env, observed sharing across atoms, reply shape).
// =====================================================================

test('logDensityN: count must be positive', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'logDensityN', ir: distIR('Normal', { mu: 0, sigma: 1 }), count: 0 });
  assert.equal(r.type, 'error');
  assert.match(r.message, /count must be positive/);
});

test('logDensityN: leaf scoring with shared observation across atoms', () => {
  // Fixed leaf, fixed observation → every atom should produce the
  // same logpdf value.
  const w = createWorkerHandler();
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const r = w.handle({ type: 'logDensityN', ir, count: 5, observed: 0.7, tally: 'clamped' });
  assert.equal(r.type, 'samples');
  assert.equal(r.samples.length, 5);
  const lp = require('@stdlib/stats-base-dists-normal-logpdf');
  const expected = lp(0.7, 0, 1);
  for (const v of r.samples) assert.equal(v, expected);
});

test('logDensityN: refArrays parameterise distribution per atom', () => {
  // bayesupdate-style use: prior atoms supply mu_i; obs is shared.
  // Per-atom logp = logpdf(obs | mu_i, 1).
  const w = createWorkerHandler();
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: {
      mu:    { kind: 'ref', ns: 'self', name: 'mu', loc: synthLoc() },
      sigma: { kind: 'lit', value: 1, loc: synthLoc() },
    },
    loc: synthLoc(),
  };
  const muArr = new Float64Array([0, 1, 2, 3]);
  const obs = 1.5;
  const r = w.handle({
    type: 'logDensityN', ir, count: 4,
    refArrays: { mu: muArr }, observed: obs, tally: 'clamped',
  });
  const lp = require('@stdlib/stats-base-dists-normal-logpdf');
  for (let i = 0; i < 4; i++) {
    assert.equal(r.samples[i], lp(obs, muArr[i], 1));
  }
});

test('logDensityN: joint observed clamps per-field, sums logpdfs', () => {
  // Mirrors the bayesupdate end-to-end pattern: kernel body is a
  // joint over (theta_clamped_only_when_observed, obs); per-i theta
  // comes from refArrays, obs is shared.
  const w = createWorkerHandler();
  const innerNormalRefMu = {
    kind: 'call', op: 'Normal',
    kwargs: {
      mu:    refIR('theta'),
      sigma: { kind: 'lit', value: 1, loc: synthLoc() },
    },
    loc: synthLoc(),
  };
  const ir = {
    kind: 'call', op: 'joint',
    fields: [
      { name: 'a', value: innerNormalRefMu },
      { name: 'b', value: distIR('Normal', { mu: 0, sigma: 1 }) },
    ],
    loc: synthLoc(),
  };
  const thetaArr = new Float64Array([0, 1, 2]);
  const r = w.handle({
    type: 'logDensityN', ir, count: 3,
    refArrays: { theta: thetaArr },
    observed: { a: 0.5, b: -0.5 },
    tally: 'all',
  });
  const lp = require('@stdlib/stats-base-dists-normal-logpdf');
  for (let i = 0; i < 3; i++) {
    const expected = lp(0.5, thetaArr[i], 1) + lp(-0.5, 0, 1);
    assert.equal(r.samples[i], expected);
  }
});


// =====================================================================
// New scalar ops: comparisons, predicates, logic, conditionals.
// Exercised through evaluateN (the existing per-i evaluator) since
// the runtime evaluator is shared.
// =====================================================================

function unaryOpIR(op: any, x: any) {
  return {
    kind: 'call', op,
    args: [{ kind: 'lit', value: x, loc: synthLoc() }],
    loc: synthLoc(),
  };
}
function binaryOpIR(op: any, a: any, b: any) {
  return {
    kind: 'call', op,
    args: [
      { kind: 'lit', value: a, loc: synthLoc() },
      { kind: 'lit', value: b, loc: synthLoc() },
    ],
    loc: synthLoc(),
  };
}

test('eval ops: comparison ops produce booleans cast to numbers', () => {
  // evaluateN packs everything into a Float64Array, so booleans
  // round-trip as 0 / 1. The tests check the numerical encoding —
  // downstream code that uses the value as a boolean (ifelse,
  // logical ops) does the typecast back implicitly.
  const w = createWorkerHandler();
  const cases: any[] = [
    ['lt',      [1, 2], 1],  ['lt',      [2, 1], 0],
    ['le',      [2, 2], 1],  ['le',      [3, 2], 0],
    ['gt',      [3, 2], 1],  ['gt',      [2, 3], 0],
    ['ge',      [2, 2], 1],  ['ge',      [1, 2], 0],
    ['equal',   [2, 2], 1],  ['equal',   [2, 3], 0],
    ['unequal', [2, 3], 1],  ['unequal', [2, 2], 0],
  ];
  for (const [op, args, expected] of cases) {
    const r = w.handle({ type: 'evaluateN', ir: binaryOpIR(op, args[0], args[1]), count: 1 });
    assert.equal(r.samples[0], expected, op + '(' + args.join(',') + ')');
  }
});

test('eval ops: logic ops (land / lor / lxor / lnot)', () => {
  const w = createWorkerHandler();
  const cases: any[] = [
    ['land', [true,  true ], 1],  ['land', [true,  false], 0],
    ['land', [false, true ], 0],  ['land', [false, false], 0],
    ['lor',  [true,  false], 1],  ['lor',  [false, false], 0],
    ['lxor', [true,  true ], 0],  ['lxor', [true,  false], 1],
  ];
  for (const [op, args, expected] of cases) {
    const r = w.handle({ type: 'evaluateN', ir: binaryOpIR(op, args[0], args[1]), count: 1 });
    assert.equal(r.samples[0], expected, op);
  }
  // Unary lnot.
  const r1 = w.handle({ type: 'evaluateN', ir: unaryOpIR('lnot', true),  count: 1 });
  assert.equal(r1.samples[0], 0);
  const r2 = w.handle({ type: 'evaluateN', ir: unaryOpIR('lnot', false), count: 1 });
  assert.equal(r2.samples[0], 1);
});

test('eval ops: ifelse picks the right branch', () => {
  const w = createWorkerHandler();
  const ir = (cond: any, a: any, b: any) => ({
    kind: 'call', op: 'ifelse',
    args: [
      { kind: 'lit', value: cond, loc: synthLoc() },
      { kind: 'lit', value: a, loc: synthLoc() },
      { kind: 'lit', value: b, loc: synthLoc() },
    ],
    loc: synthLoc(),
  });
  assert.equal(w.handle({ type: 'evaluateN', ir: ir(true,  3.14, 2.72), count: 1 }).samples[0], 3.14);
  assert.equal(w.handle({ type: 'evaluateN', ir: ir(false, 3.14, 2.72), count: 1 }).samples[0], 2.72);
});

test('eval ops: predicates (isfinite / isinf / isnan / iszero)', () => {
  const w = createWorkerHandler();
  // isfinite
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('isfinite', 1.5), count: 1 }).samples[0], 1);
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('isfinite', Infinity), count: 1 }).samples[0], 0);
  // isinf
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('isinf', Infinity), count: 1 }).samples[0], 1);
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('isinf', 1.5), count: 1 }).samples[0], 0);
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('isinf', NaN), count: 1 }).samples[0], 0);
  // isnan
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('isnan', NaN), count: 1 }).samples[0], 1);
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('isnan', 1.5), count: 1 }).samples[0], 0);
  // iszero
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('iszero', 0), count: 1 }).samples[0], 1);
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('iszero', 1e-300), count: 1 }).samples[0], 0);
});

test('eval ops: mod / abs2', () => {
  const w = createWorkerHandler();
  assert.equal(w.handle({ type: 'evaluateN', ir: binaryOpIR('mod', 7, 3), count: 1 }).samples[0], 1);
  assert.equal(w.handle({ type: 'evaluateN', ir: unaryOpIR('abs2', -3), count: 1 }).samples[0], 9);
});

// =====================================================================
// Reductions over arrays. Runtime ops live in sampler.js; the
// orchestrator's static gate is conservative (vector isn't on
// EVALUABLE_OPS) so user-level bindings like `m = mean(arr)` only
// classify when arr is a kind:'array' derivation. These tests verify
// the runtime ops themselves — the IR shape they handle, the
// numerical correctness of var (sample, spec §07), and edge cases.
// =====================================================================

function vectorOfLits(values: any) {
  return {
    kind: 'call', op: 'vector',
    args: values.map((v: any) => ({ kind: 'lit', value: v, loc: synthLoc() })),
    loc: synthLoc(),
  };
}
function reductionIR(op: any, values: any) {
  return {
    kind: 'call', op,
    args: [vectorOfLits(values)],
    loc: synthLoc(),
  };
}

test('reductions: sum / mean / prod over a literal array', () => {
  const w = createWorkerHandler();
  const xs = [1, 2, 3, 4];
  assert.equal(w.handle({ type: 'evaluateN', ir: reductionIR('sum', xs),  count: 1 }).samples[0], 10);
  assert.equal(w.handle({ type: 'evaluateN', ir: reductionIR('mean', xs), count: 1 }).samples[0], 2.5);
  assert.equal(w.handle({ type: 'evaluateN', ir: reductionIR('prod', xs), count: 1 }).samples[0], 24);
});

test('reductions: lengthof over a literal array', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'evaluateN', ir: reductionIR('lengthof', [10, 20, 30, 40, 50]), count: 1 });
  assert.equal(r.samples[0], 5);
});

test('reductions: maximum / minimum over a literal array', () => {
  const w = createWorkerHandler();
  const xs = [3, -1, 4, 1, 5, 9, 2, 6];
  assert.equal(w.handle({ type: 'evaluateN', ir: reductionIR('maximum', xs), count: 1 }).samples[0], 9);
  assert.equal(w.handle({ type: 'evaluateN', ir: reductionIR('minimum', xs), count: 1 }).samples[0], -1);
});

test('reductions: var (sample, spec §07) matches the Bessel-corrected formula', () => {
  // Sample variance of [1,2,3,4,5]: mean=3, SS=4+1+0+1+4=10, /(n-1)=10/4=2.5.
  const w = createWorkerHandler();
  const r = w.handle({ type: 'evaluateN', ir: reductionIR('var', [1, 2, 3, 4, 5]), count: 1 });
  assert.equal(r.samples[0], 2.5);
});

test('reductions: var of empty array → 0 (degenerate)', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'evaluateN', ir: reductionIR('var', []), count: 1 });
  assert.equal(r.samples[0], 0);
});

test('reductions: var of single element → 0 (sample variance undefined for n=1)', () => {
  const w = createWorkerHandler();
  const r = w.handle({ type: 'evaluateN', ir: reductionIR('var', [42]), count: 1 });
  assert.equal(r.samples[0], 0);
});

test('reductions: maximum / minimum of length-1 array → that single value', () => {
  const w = createWorkerHandler();
  assert.equal(w.handle({ type: 'evaluateN', ir: reductionIR('maximum', [42]), count: 1 }).samples[0], 42);
  assert.equal(w.handle({ type: 'evaluateN', ir: reductionIR('minimum', [42]), count: 1 }).samples[0], 42);
});

// =====================================================================
// profileN — sweep one input across [lo, hi], hold others at fixed
// values. Drives the upcoming profile-plot UI for fn / functionof /
// kernelof / likelihoodof bindings.
// =====================================================================

test('profileN: count must be positive', () => {
  const w = createWorkerHandler();
  const r = w.handle({
    type: 'profileN', count: 0,
    ir: { kind: 'lit', value: 1, loc: synthLoc() },
    sweepName: 'x', range: [0, 1],
  });
  assert.equal(r.type, 'error');
  assert.match(r.message, /count must be positive/);
});

test('profileN: range must be finite', () => {
  const w = createWorkerHandler();
  const r = w.handle({
    type: 'profileN', count: 5,
    ir: { kind: 'lit', value: 1, loc: synthLoc() },
    sweepName: 'x', range: [0, NaN],
  });
  assert.equal(r.type, 'error');
  assert.match(r.message, /finite/);
});

test('profileN: sweepName is required', () => {
  const w = createWorkerHandler();
  const r = w.handle({
    type: 'profileN', count: 5,
    ir: { kind: 'lit', value: 1, loc: synthLoc() },
    range: [0, 1],
  });
  assert.equal(r.type, 'error');
  assert.match(r.message, /sweepName/);
});

test('profileN: function f(x) = x evaluates evenly across range', () => {
  // Identity body — output equals the swept input value at each i.
  // Verifies the linear-spacing and env injection path.
  const w = createWorkerHandler();
  const ir = { kind: 'ref', ns: '%local', name: 'x', loc: synthLoc() };
  const r = w.handle({
    type: 'profileN', count: 5,
    ir, sweepName: 'x', range: [0, 4], mode: 'function',
  });
  assert.equal(r.type, 'samples');
  assert.equal(r.samples.length, 5);
  for (let i = 0; i < 5; i++) assert.equal(r.samples[i], i);
});

test('profileN: function f(x) = x*x produces the quadratic', () => {
  // (mul x x) — sanity check that arithmetic IR works through
  // evaluateExpr with a swept env.
  const w = createWorkerHandler();
  const xRef = () => ({ kind: 'ref', ns: '%local', name: 'x', loc: synthLoc() });
  const ir = { kind: 'call', op: 'mul', args: [xRef(), xRef()], loc: synthLoc() };
  const r = w.handle({
    type: 'profileN', count: 5,
    ir, sweepName: 'x', range: [-2, 2], mode: 'function',
  });
  // x runs over [-2, -1, 0, 1, 2]; output is [4, 1, 0, 1, 4].
  assert.deepEqual(Array.from(r.samples), [4, 1, 0, 1, 4]);
});

test('profileN: function honours fixedEnv for non-swept inputs', () => {
  // f(x, c) = x + c, sweep x, hold c at 100.
  const w = createWorkerHandler();
  const ir = {
    kind: 'call', op: 'add',
    args: [
      { kind: 'ref', ns: '%local', name: 'x', loc: synthLoc() },
      { kind: 'ref', ns: '%local', name: 'c', loc: synthLoc() },
    ],
    loc: synthLoc(),
  };
  const r = w.handle({
    type: 'profileN', count: 4,
    ir, sweepName: 'x', range: [0, 3], mode: 'function',
    fixedEnv: { c: 100 },
  });
  assert.deepEqual(Array.from(r.samples), [100, 101, 102, 103]);
});

test('profileN: domain-of-definition error becomes NaN, not abort', () => {
  // log(x) is undefined for x ≤ 0. Sweep over [-1, 1] across 5 points
  // (so x = -1, -0.5, 0, 0.5, 1). The first three should be NaN /
  // -Infinity, the last two finite.
  const w = createWorkerHandler();
  const ir = {
    kind: 'call', op: 'log',
    args: [{ kind: 'ref', ns: '%local', name: 'x', loc: synthLoc() }],
    loc: synthLoc(),
  };
  const r = w.handle({
    type: 'profileN', count: 5,
    ir, sweepName: 'x', range: [-1, 1], mode: 'function',
  });
  assert.equal(r.type, 'samples');
  // log(-1), log(-0.5) → NaN; log(0) → -Infinity; log(0.5), log(1) → finite.
  assert.ok(Number.isNaN(r.samples[0]));
  assert.ok(Number.isNaN(r.samples[1]));
  assert.equal(r.samples[2], -Infinity);
  assert.ok(Number.isFinite(r.samples[3]));
  assert.equal(r.samples[4], 0); // log(1) = 0
});

test('profileN: logdensity mode evaluates Normal logpdf along mu axis', () => {
  // Sweep mu over [-2, 2] for a Normal(mu, 1) at observed = 0. Per
  // point the log-density is the Gaussian logpdf — peak at mu = 0,
  // symmetric. Verifies the density.ts wiring + observed plumbing.
  const w = createWorkerHandler();
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: {
      mu:    { kind: 'ref', ns: '%local', name: 'mu', loc: synthLoc() },
      sigma: { kind: 'lit', value: 1, loc: synthLoc() },
    },
    loc: synthLoc(),
  };
  const r = w.handle({
    type: 'profileN', count: 5,
    ir, sweepName: 'mu', range: [-2, 2], mode: 'logdensity',
    observed: 0, tally: 'clamped',
  });
  const lp = require('@stdlib/stats-base-dists-normal-logpdf');
  // mu runs over [-2, -1, 0, 1, 2].
  for (let i = 0; i < 5; i++) {
    const mu = -2 + i;
    assert.ok(Math.abs(r.samples[i] - lp(0, mu, 1)) < 1e-12);
  }
  // Symmetric around mu = 0 (the peak).
  assert.equal(r.samples[1], r.samples[3]);
  assert.equal(r.samples[0], r.samples[4]);
  assert.ok(r.samples[2] > r.samples[1]);
});

// =====================================================================
// (continuing the entry-shim end-to-end tests from before)
// =====================================================================

test('entry shim: error replies survive postMessage', async () => {
  const entry = join(__dirname, '..', 'worker-entry.ts');
  const worker = new Worker(entry);
  try {
    const reply: any = await new Promise((resolve, reject) => {
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

// =====================================================================
// Session env merging into per-call paths (fixed-phase bindings)
// =====================================================================

test('evaluateN: session env (setEnv) flows into per-atom callEnv', () => {
  // Push a fixed-phase array into session env. evaluateN's IR
  // references it; the reduction (mean) operates on the full array,
  // not on per-i slices.
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const arr = [1, 2, 3, 4, 5];
  w.handle({ type: 'setEnv', env: { random_data: arr } });
  // ir = mean(ref(random_data))
  const ir = {
    kind: 'call', op: 'mean',
    args: [{ kind: 'ref', ns: 'self', name: 'random_data' }],
  };
  const reply = w.handle({ type: 'evaluateN', ir, count: 4 });
  assert.equal(reply.type, 'samples');
  assert.equal(reply.samples.length, 4);
  // Every per-atom out is the same scalar mean of arr.
  const expected = arr.reduce((s, v) => s + v, 0) / arr.length;
  for (let i = 0; i < reply.samples.length; i++) {
    assert.equal(reply.samples[i], expected);
  }
});

test('evaluateN: per-atom refArrays override session env', () => {
  // If a name is both in session env AND in per-atom refArrays, the
  // per-atom value wins — it's the more specific layer.
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  w.handle({ type: 'setEnv', env: { x: 999 } });
  const refArrays = { x: new Float64Array([10, 20, 30]) };
  const ir = { kind: 'ref', ns: 'self', name: 'x' };
  const reply = w.handle({ type: 'evaluateN', ir, count: 3, refArrays });
  assert.deepEqual(Array.from(reply.samples), [10, 20, 30]);
});

// =====================================================================
// evaluateN — complex results (chunk 4 of the complex-values thread)
// =====================================================================
//
// Atom-dependent complex previously threw. evaluateN now carries the
// planar imaginary buffer through the reply (reply.imag), and
// _unwrapRefArrays keeps complex shape=[N] refs whole instead of
// dropping their .im.

const valueLib = require('../value.ts');

test('evaluateN: complex(re, im) over batched real refs → reply.imag', () => {
  const w = createWorkerHandler();
  const ir = {
    kind: 'call', op: 'complex',
    args: [
      { kind: 'ref', ns: 'self', name: 're', loc: synthLoc() },
      { kind: 'ref', ns: 'self', name: 'im', loc: synthLoc() },
    ],
    loc: synthLoc(),
  };
  const r = w.handle({
    type: 'evaluateN', ir, count: 3,
    refArrays: {
      re: new Float64Array([1, 2, 3]),
      im: new Float64Array([4, 5, 6]),
    },
  });
  assert.equal(r.type, 'samples');
  assert.deepEqual(Array.from(r.samples), [1, 2, 3]);
  assert.ok(r.imag, 'reply must carry the imaginary buffer');
  assert.deepEqual(Array.from(r.imag), [4, 5, 6]);
});

test('evaluateN: complex shape=[N] ref survives _unwrapRefArrays (conj)', () => {
  const w = createWorkerHandler();
  const z = valueLib.complexValue([1, 2, 3], [7, 8, 9], [3]);
  const ir = {
    kind: 'call', op: 'conj',
    args: [{ kind: 'ref', ns: 'self', name: 'z', loc: synthLoc() }],
    loc: synthLoc(),
  };
  const r = w.handle({ type: 'evaluateN', ir, count: 3, refArrays: { z } });
  assert.deepEqual(Array.from(r.samples), [1, 2, 3]);
  assert.deepEqual(Array.from(r.imag), [-7, -8, -9], 'conj negates imag');
});

test('evaluateN: abs2 of a complex ref → real reply (no imag)', () => {
  const w = createWorkerHandler();
  const z = valueLib.complexValue([3, 0], [4, 2], [2]);   // |.|² = 25, 4
  const ir = {
    kind: 'call', op: 'abs2',
    args: [{ kind: 'ref', ns: 'self', name: 'z', loc: synthLoc() }],
    loc: synthLoc(),
  };
  const r = w.handle({ type: 'evaluateN', ir, count: 2, refArrays: { z } });
  assert.deepEqual(Array.from(r.samples), [25, 4]);
  assert.equal(r.imag, undefined, 'real result must not carry imag');
});

test('evaluateN: spec §03 amplitude — abs2(A_sig*coupling + A_bkg)', () => {
  const w = createWorkerHandler();
  const ref = (n: any) => ({ kind: 'ref', ns: 'self', name: n, loc: synthLoc() });
  const ir = {
    kind: 'call', op: 'abs2',
    args: [{
      kind: 'call', op: 'add',
      args: [
        { kind: 'call', op: 'mul', args: [ref('A_sig'), ref('coupling')], loc: synthLoc() },
        ref('A_bkg'),
      ],
      loc: synthLoc(),
    }],
    loc: synthLoc(),
  };
  const r = w.handle({
    type: 'evaluateN', ir, count: 2,
    refArrays: {
      A_sig:    valueLib.complexValue([1, 2], [1, 0], [2]),
      coupling: valueLib.complexValue([0, 1], [1, 0], [2]),  // i, 1
      A_bkg:    valueLib.complexValue([1, 1], [0, 0], [2]),
    },
  });
  // atom0: (1+i)*i = -1+i ; +1 = i ; |.|²=1
  // atom1: 2*1 = 2 ; +1 = 3 ; |.|²=9
  assert.deepEqual(Array.from(r.samples), [1, 9]);
  assert.equal(r.imag, undefined);
});
