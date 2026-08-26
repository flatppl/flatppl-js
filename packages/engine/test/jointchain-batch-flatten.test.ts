'use strict';

// =====================================================================
// jointchain-composite batch-flatten — calibration oracle (Phase 8 leg 4)
// =====================================================================
//
// `test/fixtures/jointchain-cat-chain.flatppl` (G=3 groups, C=3 steps):
//   x_0 ~ Normal(m_per_group[g], 1)
//   x_1 ~ Normal(x_0, 1)                 (one variate to the left)
//   x_2 ~ Normal(sum([x_0, x_1]), 1)     (the cat of two)
//   y = broadcast(group_chain, m = m_per_group)            → [N, 3, 3]
//
// A jointchain-bodied kernel-broadcast is a scan per cell. Phase 8 leg 4
// folds it: the cell axis K=3 folds into each step's sampleN (count = N·3,
// one call per step), and the steps run sequentially, each carrying every
// variate drawn so far — step k binds the `cat` of columns 0..k-1 (spec §06
// `c ~ K3([a, b])`), one variate binding whole, two or more as a `vector`.
// K·C per-cell calls collapse to C.
//
// The decisive check is the closed form, which separates the cat feed from
// a prev-only one. With x_1 = x_0 + e_1 and x_2 = 2·x_0 + e_1 + e_2:
//   E = {m, m, 2m}, Var = {1, 2, 6}, Cov(x_0,x_2) = 2, Cov(x_1,x_2) = 3
// A step 2 fed only x_1 would give E[x_2] = m and Var(x_2) = 3.
// Also pins the fold path (single [kernel_broadcast 3] ladder) and the shape.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const axisStackMod = require('../axis-stack.ts');
const { createWorkerHandler } = require('../worker.ts');

test('jointchain scan fold: the cat feed calibrates at shape [N,3,3]', async () => {
  const N = 40000;
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'jointchain-cat-chain.flatppl'), 'utf-8');
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  assert.ok(built.derivations && built.derivations.y, 'y has a derivation');
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 9 });
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
    sampleCount: N, rootKey: [9, 0],
  };

  // (a) Fold path: a jointchain body adds no inner axis → single ladder.
  const stack = axisStackMod.bindingAxisStack('y', ctx);
  assert.deepEqual(stack, [{ source: 'kernel_broadcast', size: 3, name: 'm_per_group' }],
    'y carries the single [kernel_broadcast 3] ladder (scan fold path)');

  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [N, 3, 3], 'shape [N, groups, chain length]');

  const G = 3, C = 3;
  const d = m.value.data;
  const mPer = [0.0, 1.0, 2.0];

  const col = (g: number, k: number) => {
    const out = new Float64Array(N);
    for (let i = 0; i < N; i++) out[i] = d[i * G * C + g * C + k];
    return out;
  };
  const mean = (xs: Float64Array) => {
    let s = 0;
    for (let i = 0; i < xs.length; i++) s += xs[i];
    return s / xs.length;
  };
  const cov = (xs: Float64Array, ys: Float64Array) => {
    const mx = mean(xs), my = mean(ys);
    let s = 0;
    for (let i = 0; i < xs.length; i++) s += (xs[i] - mx) * (ys[i] - my);
    return s / xs.length;
  };

  for (let g = 0; g < G; g++) {
    const x0 = col(g, 0), x1 = col(g, 1), x2 = col(g, 2);
    const mu = mPer[g];
    // (b) Means. E[x_2] = 2·m is the cat feed's signature: a prev-only
    // step 2 would centre on m.
    assert.ok(Math.abs(mean(x0) - mu) < 0.03, `group ${g} E[x_0] ≈ ${mu}; got ${mean(x0).toFixed(3)}`);
    assert.ok(Math.abs(mean(x1) - mu) < 0.04, `group ${g} E[x_1] ≈ ${mu}; got ${mean(x1).toFixed(3)}`);
    assert.ok(Math.abs(mean(x2) - 2 * mu) < 0.06,
      `group ${g} E[x_2] ≈ ${2 * mu} (CAT FEED: sum([x_0, x_1]) has mean 2m; `
      + `a prev-only step 2 would give ${mu}); got ${mean(x2).toFixed(3)}`);
    // (c) Variances {1, 2, 6}.
    for (const [k, want, xs] of [[0, 1, x0], [1, 2, x1], [2, 6, x2]] as any[]) {
      const got = cov(xs, xs);
      assert.ok(Math.abs(got - want) < 0.15 * want,
        `group ${g} Var(x_${k}) ≈ ${want}; got ${got.toFixed(3)}`);
    }
    // (d) Cross-covariances 2 and 3 — the carry, step by step.
    assert.ok(Math.abs(cov(x0, x2) - 2) < 0.15,
      `group ${g} Cov(x_0, x_2) ≈ 2; got ${cov(x0, x2).toFixed(3)}`);
    assert.ok(Math.abs(cov(x1, x2) - 3) < 0.2,
      `group ${g} Cov(x_1, x_2) ≈ 3; got ${cov(x1, x2).toFixed(3)}`);
  }
});
