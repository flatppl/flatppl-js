'use strict';

// Tests for Phase 7c: per-atom vector reductions over vector-atom
// inputs. softmax / l1unit / l2unit / logsoftmax on a per-atom vector
// produce a per-atom vector (shape=[N, k]). The pipeline packs the
// per-atom JS-array results into a Value, the worker propagates dims,
// and matEvaluate produces a vector-atom Measure.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 16;
const ROOT_SEED = 0xC0FEEBED;

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

function close(a: any, b: any, tol: any) {
  tol = tol == null ? 1e-12 : tol;
  return Math.abs(a - b) <= tol;
}

// =====================================================================
// Per-atom-fallback Value packing — unit test through evaluateExprN
// =====================================================================

const sampler = require('../sampler.ts');
const valueLib = require('..').value;

test('evaluateExprN: per-atom softmax packs into Value shape=[N, k]', () => {
  // a = [0, 0, 0]; b = [0, 0, 0] — softmax of [0, 0] for each atom is
  // [0.5, 0.5]. Build vector(a, b) → shape=[N=3, 2], then softmax.
  const a = new Float64Array([0, 0, 0]);
  const b = new Float64Array([0, 0, 0]);
  const ir = {
    kind: 'call', op: 'softmax',
    args: [{ kind: 'call', op: 'vector',
             args: [{ kind: 'ref', ns: 'self', name: 'a' },
                    { kind: 'ref', ns: 'self', name: 'b' }] }],
  };
  const r = sampler.evaluateExprN(ir, { a, b }, 3, {});
  // Phase 7c: Value shape=[N=3, k=2] — each atom's softmax is [0.5, 0.5].
  assert.ok(r && r.shape && r.data);
  assert.deepEqual(r.shape, [3, 2]);
  for (let i = 0; i < 3; i++) {
    assert.ok(close(r.data[i * 2 + 0], 0.5));
    assert.ok(close(r.data[i * 2 + 1], 0.5));
  }
});

test('evaluateExprN: per-atom l2unit on vector(a, b)', () => {
  // a = [3, 4]; b = [4, 3] → vectors [3,4] and [4,3]; l2unit → [0.6, 0.8]
  // and [0.8, 0.6].
  const a = new Float64Array([3, 4]);
  const b = new Float64Array([4, 3]);
  const ir = {
    kind: 'call', op: 'l2unit',
    args: [{ kind: 'call', op: 'vector',
             args: [{ kind: 'ref', ns: 'self', name: 'a' },
                    { kind: 'ref', ns: 'self', name: 'b' }] }],
  };
  const r = sampler.evaluateExprN(ir, { a, b }, 2, {});
  assert.deepEqual(r.shape, [2, 2]);
  assert.ok(close(r.data[0], 0.6));
  assert.ok(close(r.data[1], 0.8));
  assert.ok(close(r.data[2], 0.8));
  assert.ok(close(r.data[3], 0.6));
});

// =====================================================================
// End-to-end via processSource — softmax / l1unit on iid(...) bindings
// =====================================================================

test('Phase 7c: softmax on iid(Normal, k) produces vector-atom Measure', async () => {
  const ctx = makeCtx(`
xs = iid(Normal(mu=0.0, sigma=1.0), 4)
sm = softmax(xs)
`);
  const xs = await ctx.getMeasure('xs');
  const sm = await ctx.getMeasure('sm');
  // sm should be vector-atom with dims=[4] (or .value shape=[N, 4]).
  assert.ok(sm.dims && sm.dims[0] === 4,
    'softmax(vector-atom) result should have dims=[4]');
  assert.ok(sm.value, 'should populate .value');
  assert.deepEqual(sm.value.shape, [SAMPLE_COUNT, 4]);
  // Verify per-atom softmax matches the expected formula.
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const slice = [
      xs.samples[i * 4 + 0],
      xs.samples[i * 4 + 1],
      xs.samples[i * 4 + 2],
      xs.samples[i * 4 + 3],
    ];
    let m = -Infinity;
    for (const v of slice) if (v > m) m = v;
    const exps = slice.map(v => Math.exp(v - m));
    const s = exps.reduce((a, b) => a + b, 0);
    const expected = exps.map(e => e / s);
    for (let j = 0; j < 4; j++) {
      assert.ok(close(sm.samples[i * 4 + j], expected[j], 1e-10),
        'atom ' + i + ' entry ' + j);
    }
    // softmax outputs sum to 1.
    let totalRow = 0;
    for (let j = 0; j < 4; j++) totalRow += sm.samples[i * 4 + j];
    assert.ok(close(totalRow, 1, 1e-10), 'softmax row sums to 1');
  }
});

test('Phase 7c: l1unit on iid(...) produces vector-atom Measure summing to 1 per atom', async () => {
  // Use positive variates via exp to ensure non-zero l1 norms.
  const ctx = makeCtx(`
xs = iid(Exponential(rate=1.0), 3)
u = l1unit(xs)
`);
  const u = await ctx.getMeasure('u');
  assert.deepEqual(u.dims, [3]);
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    let s = 0;
    for (let j = 0; j < 3; j++) s += u.samples[i * 3 + j];
    assert.ok(close(s, 1, 1e-10), 'atom ' + i + ' l1-unit row sums to 1, got ' + s);
  }
});
