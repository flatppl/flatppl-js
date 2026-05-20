'use strict';

// Tests for the Phase 4a Value ↔ Measure bridges: `valueOf(m)` builds
// a Value view of a Measure (sharing storage with .samples), and
// `measureFromValue(v, extras)` is the reverse constructor.
//
// Phase 4a only introduces the helpers; handlers don't populate
// `.value` yet (that's 4b). valueOf falls back to constructing the
// Value from .samples + .dims for unmigrated handlers.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const materialiser = require('..').materialiser;
const valueLib     = require('..').value;
const { valueOf, measureFromValue } = materialiser;

// =====================================================================
// valueOf — Measure → Value view
// =====================================================================

test('valueOf: scalar-atom Measure → shape=[N] Value, storage shared', () => {
  const samples = new Float64Array([1, 2, 3, 4]);
  const m = { samples: samples, logWeights: null, logTotalmass: 0, n_eff: 4 };
  const v = valueOf(m);
  assert.deepEqual(v.shape, [4]);
  assert.equal(v.data, samples, 'data must SHARE storage with samples');
});

test('valueOf: vector-atom Measure (dims=[k]) → shape=[N, k] Value', () => {
  // 3 atoms × 2 components — atom-major flat storage.
  const samples = new Float64Array([1, 2, 3, 4, 5, 6]);
  const m = { samples: samples, dims: [2], logWeights: null,
              logTotalmass: 0, n_eff: 3 };
  const v = valueOf(m);
  assert.deepEqual(v.shape, [3, 2]);
  assert.equal(v.data, samples);
});

test('valueOf: matrix-atom Measure (dims=[m, n]) → shape=[N, m, n] Value', () => {
  // 2 atoms × 2x2 matrix = 8 elements.
  const samples = new Float64Array(8);
  const m = { samples: samples, dims: [2, 2], logWeights: null,
              logTotalmass: 0, n_eff: 2 };
  const v = valueOf(m);
  assert.deepEqual(v.shape, [2, 2, 2]);
});

test('valueOf: returns null for record-shaped Measure (no top-level .samples)', () => {
  const m = {
    fields: { a: { samples: new Float64Array(4) }, b: { samples: new Float64Array(4) } },
    logTotalmass: 0,
    n_eff: 4,
  };
  assert.equal(valueOf(m), null);
});

test('valueOf: prefers existing .value field (Phase 4b ready)', () => {
  // When a handler has already populated .value (Phase 4b migration),
  // valueOf returns it directly rather than rebuilding from .samples.
  const explicit = valueLib.scalar(42);  // shape=[]; intentionally weird
  const m = {
    samples: new Float64Array([42]),
    value: explicit,
    logWeights: null, logTotalmass: 0, n_eff: 1,
  };
  assert.equal(valueOf(m), explicit);
});

test('valueOf: null on missing measure / non-samples shapes', () => {
  assert.equal(valueOf(null), null);
  assert.equal(valueOf(undefined), null);
  assert.equal(valueOf({}), null);
});

// =====================================================================
// measureFromValue — Value → Measure
// =====================================================================

test('measureFromValue: scalar-atom Value shape=[N] → standard Measure', () => {
  const v = valueLib.batchedScalar(new Float64Array([1, 2, 3, 4]));
  const m = measureFromValue(v);
  assert.equal(m.samples, v.data, 'samples must share storage with value.data');
  assert.equal(m.value, v);
  assert.equal(m.logWeights, null);
  assert.equal(m.logTotalmass, 0);
  assert.equal(m.n_eff, 4);
  assert.equal(m.dims, undefined);
});

test('measureFromValue: vector-atom Value shape=[N, k] → dims=[k]', () => {
  const v = valueLib.batchedVector(new Float64Array([1, 2, 3, 4, 5, 6]), 2);
  const m = measureFromValue(v);
  assert.deepEqual(m.dims, [2]);
  assert.equal(m.samples.length, 6);
  assert.equal(m.value, v);
});

test('measureFromValue: matrix-atom shape=[N, m, n] → dims=[m, n]', () => {
  const v = { shape: [2, 3, 4], data: new Float64Array(24) };
  const m = measureFromValue(v);
  assert.deepEqual(m.dims, [3, 4]);
  assert.equal(m.samples.length, 24);
});

test('measureFromValue: passes through extras (logWeights, logTotalmass, n_eff)', () => {
  const v = valueLib.batchedScalar(new Float64Array([1, 2, 3]));
  const w = new Float64Array([-1, -1, -1]);
  const m = measureFromValue(v, {
    logWeights: w,
    logTotalmass: -2.5,
    n_eff: 2,
  });
  assert.equal(m.logWeights, w);
  assert.equal(m.logTotalmass, -2.5);
  assert.equal(m.n_eff, 2);
});

test('measureFromValue: rejects shape=[] (no atom axis)', () => {
  assert.throws(
    () => measureFromValue(valueLib.scalar(3)),
    /no atom axis/);
});

test('measureFromValue: rejects non-Value', () => {
  assert.throws(
    () => measureFromValue(new Float64Array([1, 2, 3])),
    /not a Value/);
});

// =====================================================================
// Round-trip: valueOf(measureFromValue(v)) ≡ v (modulo identity)
// =====================================================================

test('round-trip: measureFromValue then valueOf returns the same value', () => {
  const v = valueLib.batchedVector(new Float64Array([1, 2, 3, 4]), 2);
  const m = measureFromValue(v);
  const recovered = valueOf(m);
  // measureFromValue stashes the original Value in .value, so valueOf
  // returns it directly (same reference, free).
  assert.equal(recovered, v);
});

// =====================================================================
// Phase 4b regression — every handler that produces a scalar-leaf
// Measure must populate `.value` (the migration's atomic invariant).
// =====================================================================

const { processSource, orchestrator } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function makeCtx(source) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 0xC0DECAFE });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name) => {
      if (cache.has(name)) return cache.get(name);
      const materialiser = require('../materialiser.ts');
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg) => Promise.resolve(worker.handle(msg)),
    sampleCount: 256,
    rootSeed:    0xC0DECAFE,
  };
  return ctx;
}

test('Phase 4b: matSample populates .value (shape=[N])', async () => {
  const ctx = makeCtx(`x = Normal(mu=0.0, sigma=1.0)`);
  const m = await ctx.getMeasure('x');
  assert.ok(m.value, 'matSample result must have .value populated');
  assert.deepEqual(m.value.shape, [256]);
  assert.equal(m.value.data, m.samples, 'value.data must share storage with samples');
});

test('Phase 4b: matEvaluate populates .value', async () => {
  const ctx = makeCtx(`
x = Normal(mu=0.0, sigma=1.0)
y = 2.0 * x + 1.0
`);
  const m = await ctx.getMeasure('y');
  assert.ok(m.value);
  assert.deepEqual(m.value.shape, [256]);
});

test('Phase 4b: matIid populates .value (shape=[N, k])', async () => {
  const ctx = makeCtx(`xs = iid(Normal(mu=0.0, sigma=1.0), 5)`);
  const m = await ctx.getMeasure('xs');
  assert.ok(m.value, 'matIid result must have .value populated');
  assert.deepEqual(m.value.shape, [256, 5]);
  assert.equal(m.value.data, m.samples);
});

test('Phase 4b: matWeighted populates .value', async () => {
  const ctx = makeCtx(`
base = Normal(mu=0.0, sigma=1.0)
w = weighted(2.0, base)
`);
  const m = await ctx.getMeasure('w');
  assert.ok(m.value);
});

test('Phase 4b: matNormalize populates .value', async () => {
  const ctx = makeCtx(`
base = Normal(mu=0.0, sigma=1.0)
n = normalize(weighted(2.0, base))
`);
  const m = await ctx.getMeasure('n');
  assert.ok(m.value);
});

test('Phase 4b: matTotalmass populates .value', async () => {
  const ctx = makeCtx(`
base = Normal(mu=0.0, sigma=1.0)
w = weighted(2.0, base)
tm = totalmass(w)
`);
  const m = await ctx.getMeasure('tm');
  assert.ok(m.value);
});

test('Phase 4b: matSuperpose populates .value', async () => {
  const ctx = makeCtx(`
a = Normal(mu=0.0, sigma=1.0)
b = Normal(mu=5.0, sigma=1.0)
s = superpose(weighted(0.5, a), weighted(0.5, b))
`);
  const m = await ctx.getMeasure('s');
  assert.ok(m.value);
});

test('Phase 4b: matPushfwd populates .value', async () => {
  const ctx = makeCtx(`
base = Normal(mu=0.0, sigma=1.0)
ln = pushfwd(fn(exp(_)), base)
`);
  const m = await ctx.getMeasure('ln');
  assert.ok(m.value);
});

test('Phase 4b: matLogdensityof populates .value', async () => {
  const ctx = makeCtx(`
base = Normal(mu=0.0, sigma=1.0)
lp = logdensityof(base, 1.5)
`);
  const m = await ctx.getMeasure('lp');
  assert.ok(m.value);
});

test('Phase 4b: record fields have .value on the scalar leaves', async () => {
  const ctx = makeCtx(`
a = Normal(mu=0.0, sigma=1.0)
b = Normal(mu=5.0, sigma=1.0)
j = joint(a=a, b=b)
`);
  const m = await ctx.getMeasure('j');
  // The record itself has no top-level .value (no samples); each field
  // sub-measure has its own .value.
  assert.ok(m.fields);
  assert.ok(m.fields.a.value);
  assert.ok(m.fields.b.value);
  assert.deepEqual(m.fields.a.value.shape, [256]);
});
