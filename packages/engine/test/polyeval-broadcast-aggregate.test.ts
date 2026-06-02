'use strict';

// =====================================================================
// polyeval broadcast+aggregate — calibration oracle (Phase 8 forcing case)
// =====================================================================
//
// `test/fixtures/polyeval-iid-broadcast.flatppl`:
//   polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
//   C = [2.3, 1.5, 0.7];  X ~ iid(Normal(0,1), 10)
//   Y = polyeval.([C], X)            % per atom: length-10 vector
//
// Y layers the dissolution concepts the uniform broadcast-flatten
// (TODO Phase 8) must preserve: a fused broadcast+aggregate (polyeval's
// `sum(coeffs .* x .^ …)`) under a dotted broadcast over the length-10
// stochastic X. This test pins Y's calibration + shape so the
// batch-flatten refactor can prove "identical calibration", and pins
// the §03 invariant (Y is N atoms × a length-10 VECTOR — outerRank
// must keep the 10 distinct from the atom axis, not fold it in).
//
// Calibration: Y[i] = 2.3 + 1.5·X[i] + 0.7·X[i]²  with X[i] ~ N(0,1):
//   E[Y]   = 2.3 + 1.5·0 + 0.7·E[X²] = 2.3 + 0.7 = 3.0
//   Var[Y] = 1.5²·Var(X) + 0.7²·Var(X²) = 2.25·1 + 0.49·2 = 3.23 → std ≈ 1.797

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const { createWorkerHandler } = require('../worker.ts');

function materialiseBinding(src: string, name: string, N: number): Promise<any> {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, 'parses cleanly: ' + errs.map((d: any) => d.message).join('; '));
  const built = orchestrator.buildDerivations(lifted.bindings);
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
  return ctx.getMeasure(name);
}

function meanStd(xs: ArrayLike<number>): { mean: number; std: number } {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  const m = s / xs.length;
  let v = 0;
  for (let i = 0; i < xs.length; i++) v += (xs[i] - m) * (xs[i] - m);
  return { mean: m, std: Math.sqrt(v / xs.length) };
}

test('polyeval broadcast+aggregate: Y calibrates to E=3.0, std≈1.8 at shape [N,10]', async () => {
  const N = 4000;
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'polyeval-iid-broadcast.flatppl'), 'utf-8');
  const m = await materialiseBinding(src, 'Y', N);

  // §03: Y is N atoms each a length-10 VECTOR. Flat storage is [N,10]
  // with the 10 kept distinct from the atom axis by outerRank — the
  // batch-flatten must preserve this (never fold the 10 into the batch).
  assert.ok(m.value && Array.isArray(m.value.shape), 'Y is a shape-tagged Value');
  assert.deepEqual(m.value.shape, [N, 10], 'shape [N, 10]: N atoms × length-10 vector');
  assert.equal(m.value.data.length, N * 10, 'flat buffer of N·10');

  // Calibration of the fused broadcast+aggregate.
  const { mean, std } = meanStd(m.value.data);
  assert.ok(Math.abs(mean - 3.0) < 0.1,
    'E[Y] ≈ 3.0 (= 2.3 + 0.7·E[X²]); got ' + mean.toFixed(4));
  assert.ok(Math.abs(std - 1.797) < 0.15,
    'std[Y] ≈ 1.80; got ' + std.toFixed(4));
});
