'use strict';

// =====================================================================
// functionof over an intermediate binding, applied by broadcast
// =====================================================================
//
//   x = elementof(reals)
//   y = x ^ 2
//   f = functionof(y)          % implicit boundary: the elementof leaf x
//   Y = f.(X)                  % broadcast f over X
//
// `functionof(y)` reifies the sub-DAG rooted at `y` down to its free
// `elementof` leaf `x` (the implicit boundary / parameter). The reified
// body must be expressed in terms of that boundary — the intermediate
// value binding `y` (= x^2) has to be inlined down to `x`, exactly as a
// reified kernel inlines derived distribution parameters to its
// boundaries (materialiser-shared.inlineBoundaryDerivations).
//
// Regression: before the fix the stored body was a bare `ref y`, so
//   - fixed X  → the fixed-phase folder could not resolve `y` under the
//     per-cell `x` binding → Y misclassified 'evaluate' → materialise crash;
//   - stochastic X → materialise threw "no derivation for 'y'".

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const { createWorkerHandler } = require('../worker.ts');

function build(src: string) {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, 'parses cleanly: ' + errs.map((d: any) => d.message).join('; '));
  return orchestrator.buildDerivations(lifted.bindings);
}

function asArray(v: any): number[] {
  // FixedValues stores the flat buffer as an array-like (plain object or
  // typed array); normalise to a JS number[] for comparison.
  if (!v) return v;
  const data = v.data != null ? v.data : v;
  const n = v.shape ? v.shape.reduce((a: number, b: number) => a * b, 1) : Object.keys(data).length;
  const out = [];
  for (let i = 0; i < n; i++) out.push(Number(data[i]));
  return out;
}

test('functionof(intermediate).(fixed array) folds at fixed phase to [1,4,9]', () => {
  const src = `x = elementof(reals)
y = x ^ 2
f = functionof(y)
X = [1, 2, 3]
Y = f.(X)`;
  const ds = build(src);
  assert.ok(ds.fixedValues.has('Y'),
    'Y is a fixed-phase value (all inputs fixed/deterministic)');
  const y = ds.fixedValues.get('Y');
  assert.deepEqual(y.shape, [3], 'Y shape [3]');
  assert.deepEqual(asArray(y), [1, 4, 9], 'Y = X .^ 2');
});

function meanOf(xs: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

test('functionof(intermediate).(stochastic iid) materialises to [N,k] of x^2', async () => {
  const N = 4000;
  const src = `x = elementof(reals)
y = x ^ 2
f = functionof(y)
X ~ iid(Normal(0, 1), 5)
Y = f.(X)`;
  const built = build(src);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 7 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
    sampleCount: N, rootKey: [7, 0],
  };
  const m = await ctx.getMeasure('Y');
  assert.deepEqual(m.value.shape, [N, 5], 'shape [N, 5]: N atoms × length-5 vector');
  // Every element is x^2 ≥ 0; E[x^2] = 1 for x ~ N(0,1).
  const data = m.value.data;
  let anyNeg = false;
  for (let i = 0; i < data.length; i++) if (data[i] < 0) { anyNeg = true; break; }
  assert.ok(!anyNeg, 'all outputs are squares (≥ 0)');
  assert.ok(Math.abs(meanOf(data) - 1.0) < 0.1, 'E[x^2] ≈ 1; got ' + meanOf(data).toFixed(4));
});
