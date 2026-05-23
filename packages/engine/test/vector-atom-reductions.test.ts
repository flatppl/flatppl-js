'use strict';

// Tests for Phase 7b: reductions on vector-atom Values. The Phase-4b
// `.value` field on matIid-produced Measures, plus the Phase-7b
// migration of `collectRefArrays` to surface Values for vector-atom
// parents, plus the Phase-7b accessor in `_perAtomFallback` /
// density.js's walkLeaf — together let reduction ops (sum, mean,
// l1norm, etc.) on vector-atom inputs produce per-atom scalar
// results through the per-atom fallback path.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 16;
const ROOT_SEED = 0xCAFE1234;

function makeCtx(source: any) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

// =====================================================================
// Direct: ARITH_OPS.sum/mean/etc. accept Values uniformly
// =====================================================================

const sampler = require('../sampler.ts');
const valueLib = require('..').value;
const { ARITH_OPS } = sampler._internal;

test('sum: works on Value just like on Float64Array', () => {
  const v = valueLib.vector([1, 2, 3, 4]);
  assert.equal(ARITH_OPS.sum(v), 10);
  assert.equal(ARITH_OPS.sum(new Float64Array([1, 2, 3, 4])), 10);
  assert.equal(ARITH_OPS.sum([1, 2, 3, 4]), 10);
});

test('mean / prod / lengthof on Value', () => {
  const v = valueLib.vector([2, 4, 6, 8]);
  assert.equal(ARITH_OPS.mean(v), 5);
  assert.equal(ARITH_OPS.prod(v), 384);
  assert.equal(ARITH_OPS.lengthof(v), 4);
});

test('maximum / minimum on Value', () => {
  const v = valueLib.vector([3, 1, 4, 1, 5, 9, 2, 6]);
  assert.equal(ARITH_OPS.maximum(v), 9);
  assert.equal(ARITH_OPS.minimum(v), 1);
});

test('var on Value (population variance)', () => {
  const v = valueLib.vector([2, 4, 6, 8]);
  // mean=5; deviations 9, 1, 1, 9; var = 20/4 = 5
  assert.equal(ARITH_OPS.var(v), 5);
});

test('l1norm / l2norm on Value', () => {
  const v = valueLib.vector([3, -4]);
  assert.equal(ARITH_OPS.l1norm(v), 7);
  assert.equal(ARITH_OPS.l2norm(v), 5);
});

test('logsumexp on Value', () => {
  const v = valueLib.vector([0, 1]);
  // log(1 + e) ≈ 1.31326
  assert.ok(Math.abs(ARITH_OPS.logsumexp(v) - Math.log(1 + Math.E)) < 1e-12);
});

test('l1unit / l2unit on Value (return JS arrays)', () => {
  const v = valueLib.vector([3, 4]);
  const u = ARITH_OPS.l2unit(v);
  assert.ok(Math.abs(u[0] - 0.6) < 1e-12);
  assert.ok(Math.abs(u[1] - 0.8) < 1e-12);
});

test('softmax / logsoftmax on Value', () => {
  const v = valueLib.vector([0, 0]);
  const s = ARITH_OPS.softmax(v);
  assert.ok(Math.abs(s[0] - 0.5) < 1e-12);
  assert.ok(Math.abs(s[1] - 0.5) < 1e-12);
});

// =====================================================================
// End-to-end: reduction on a vector-atom binding (e.g. iid of Normals)
// =====================================================================

test('Phase 7b: sum on vector-atom iid → per-atom scalar', async () => {
  // `xs ~ iid(Normal(0,1), 4)` produces shape=[N, 4] samples.
  // `s = sum(xs)` should compute per-atom sum: shape=[N], each entry
  // = sum of the corresponding 4-vector.
  const ctx = makeCtx(`
xs = iid(Normal(mu=0.0, sigma=1.0), 4)
s = sum(xs)
`);
  const xsM = await ctx.getMeasure('xs');
  const sM  = await ctx.getMeasure('s');
  assert.equal(sM.samples.length, SAMPLE_COUNT);
  // Verify: s[i] equals the sum of xs[i, 0..3].
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const expected = xsM.samples[i * 4] + xsM.samples[i * 4 + 1]
                   + xsM.samples[i * 4 + 2] + xsM.samples[i * 4 + 3];
    assert.ok(Math.abs(sM.samples[i] - expected) < 1e-10,
      'atom ' + i + ': got ' + sM.samples[i] + ', expected ' + expected);
  }
});

test('Phase 7b: mean on vector-atom iid → per-atom scalar', async () => {
  const ctx = makeCtx(`
xs = iid(Normal(mu=0.0, sigma=1.0), 4)
m = mean(xs)
`);
  const xsM = await ctx.getMeasure('xs');
  const mM  = await ctx.getMeasure('m');
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const expected = (xsM.samples[i * 4] + xsM.samples[i * 4 + 1]
                    + xsM.samples[i * 4 + 2] + xsM.samples[i * 4 + 3]) / 4;
    assert.ok(Math.abs(mM.samples[i] - expected) < 1e-10);
  }
});

test('Phase 7b: l2norm on vector-atom iid → per-atom scalar', async () => {
  const ctx = makeCtx(`
xs = iid(Normal(mu=0.0, sigma=1.0), 3)
n = l2norm(xs)
`);
  const xsM = await ctx.getMeasure('xs');
  const nM  = await ctx.getMeasure('n');
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    let s = 0;
    for (let j = 0; j < 3; j++) {
      const x = xsM.samples[i * 3 + j];
      s += x * x;
    }
    const expected = Math.sqrt(s);
    assert.ok(Math.abs(nM.samples[i] - expected) < 1e-10);
  }
});

test('Phase 7b: maximum on vector-atom iid → per-atom max', async () => {
  const ctx = makeCtx(`
xs = iid(Normal(mu=0.0, sigma=1.0), 5)
mx = maximum(xs)
`);
  const xsM = await ctx.getMeasure('xs');
  const mxM = await ctx.getMeasure('mx');
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    let m = -Infinity;
    for (let j = 0; j < 5; j++) {
      const x = xsM.samples[i * 5 + j];
      if (x > m) m = x;
    }
    assert.ok(Math.abs(mxM.samples[i] - m) < 1e-12);
  }
});

test('Phase 7b: lengthof on vector-atom returns dim per atom', async () => {
  // Each atom's intrinsic length is the dims size — 5 in this case.
  // The reduction gives a scalar per atom; broadcast to all atoms.
  const ctx = makeCtx(`
xs = iid(Normal(mu=0.0, sigma=1.0), 5)
n = lengthof(xs)
`);
  const m = await ctx.getMeasure('n');
  for (let i = 0; i < SAMPLE_COUNT; i++) assert.equal(m.samples[i], 5);
});

test('Phase 7b regression: sum of atom-indep array still scalar', async () => {
  // `arr = [1, 2, 3]; s = sum(arr)` — atom-indep fixed-phase array.
  // The pre-eval path computes s = 6 and broadcasts; should not
  // be affected by the Phase 7b changes.
  const ctx = makeCtx(`
arr = [1.0, 2.0, 3.0]
s = sum(arr)
`);
  const m = await ctx.getMeasure('s');
  for (let i = 0; i < SAMPLE_COUNT; i++) assert.equal(m.samples[i], 6);
});
