'use strict';

// Spec §04 §sec:aggregate — the five normative aggregate examples must
// materialise as matrices, not {elems} tuples. Oracle values are
// closed-form (hand-computed integer matrix arithmetic), independent of
// the engine. A two-output-axis aggregate must yield the same flat-value
// + intrinsicShape representation `rowstack`/matmul matrices already use
// (see materialiser-shared.fixedValueToMeasure); rank-1 outputs stay
// plain rank-1 vectors.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function makeCtx(source: any) {
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [], 'unexpected diagnostics: ' + JSON.stringify(errs));
  const built = orchestrator.buildDerivations(lifted.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 1 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: 1, rootKey: 1, rootSeed: 1,
    getMeasure: (n: any) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => {
      const r = w.handle(m);
      if (r && r.type === 'error') return Promise.reject(new Error(r.message));
      return Promise.resolve(r);
    },
  };
  return ctx;
}

const A = 'A = rowstack([[1, 3, 5], [9, 5, 1]])\n';
const B = 'B = rowstack([[1, 0], [0, 1], [1, 1]])\n';

// A rank-2 aggregate result: flat row-major value.data + intrinsicShape, no elems.
function assertMatrix(m: any, data: number[], intrinsic: number[], label: any) {
  assert.equal(m.elems, undefined, `${label}: must not be an {elems} tuple`);
  assert.ok(m.value && Array.isArray(m.value.shape), `${label}: has a Value`);
  assert.deepEqual(Array.from(m.value.data), data, `${label}: value.data`);
  assert.deepEqual(m.intrinsicShape, intrinsic, `${label}: intrinsicShape`);
}
// A rank-1 aggregate result: plain vector value, no intrinsicShape rank-2 tag.
function assertVector(m: any, data: number[], label: any) {
  assert.equal(m.elems, undefined, `${label}: must not be an {elems} tuple`);
  assert.equal(m.intrinsicShape, undefined, `${label}: rank-1 result must not carry an intrinsicShape`);
  assert.deepEqual(Array.from(m.value.data), data, `${label}: value.data`);
}

test('C: matrix multiplication aggregate materialises as a matrix', async () => {
  const ctx = makeCtx(A + B + 'C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])\n');
  assertMatrix(await ctx.getMeasure('C'), [6, 8, 10, 6], [2, 2], 'C');
});

test('D: weighted squared-difference aggregate materialises as a matrix', async () => {
  const ctx = makeCtx(A + B + 'w = [1, 2, 1]\n'
    + 'D = aggregate(sum, [.i, .k], (A[.i, .j] - B[.j, .k])^2 * w[.j])\n');
  assertMatrix(await ctx.getMeasure('D'), [34, 25, 114, 113], [2, 2], 'D');
});

test('P: prod-reduction aggregate materialises as a matrix', async () => {
  const ctx = makeCtx(A + B + 'P = aggregate(prod, [.i, .k], A[.i, .j] + B[.j, .k])\n');
  assertMatrix(await ctx.getMeasure('P'), [36, 24, 100, 108], [2, 2], 'P');
});

test('V: column-variance aggregate stays a rank-1 vector', async () => {
  const ctx = makeCtx(A + 'V = aggregate(var, [.j], A[.i, .j])\n');
  assertVector(await ctx.getMeasure('V'), [32, 2, 8], 'V');
});

test('S: fixed-column row aggregate stays a rank-1 vector', async () => {
  const ctx = makeCtx(A + 'S = aggregate(sum, [.i], A[.i, 1])\n');
  assertVector(await ctx.getMeasure('S'), [1, 9], 'S');
});
