'use strict';
// packages/engine/test/ram-correlated.test.ts
// RAM on an exactly-correlated 2-D Gaussian posterior. Prior a,b ~ N(0,3);
// observe y1 = a+b (sd 1) and y2 = b (sd 1). The posterior is Gaussian with
// precision Λ = (1/9)I + [[1,1],[1,1]] + [[0,0],[0,1]] = [[1+1/9, 1],[1, 2+1/9]],
// mean Λ⁻¹·[y1, y1+y2]. Oracle: exact 2×2 inverse (closed form). Asserts RAM
// recovers the posterior mean, marginal variances, and (crucially) the negative
// correlation the adaptive proposal must capture to mix well.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    moduleRegistry: lifted.loweredModule && lifted.loweredModule.moduleRegistry
      ? lifted.loweredModule.moduleRegistry : null,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n); const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p; },
    sendWorker: (m: any) => { const r = worker.handle(m); return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r); },
    sampleCount: N, rootSeed: 42, rootKey: 42,
  };
  return ctx;
}

const MODEL = `
a = draw(Normal(mu = 0.0, sigma = 3.0))
b = draw(Normal(mu = 0.0, sigma = 3.0))
prior = lawof(record(a = a, b = b))
obs = joint(y1 = Normal(mu = a + b, sigma = 1.0), y2 = Normal(mu = b, sigma = 1.0))
K = functionof(obs, a = a, b = b)
L = likelihoodof(K, record(y1 = 3.0, y2 = 1.0))
posterior = bayesupdate(L, prior)
`;

test('backend:ram recovers an exactly-correlated 2-D Gaussian posterior', async () => {
  // Closed-form oracle.
  const p = 1 / 9;
  const L11 = 1 + p, L12 = 1, L22 = 2 + p;
  const det = L11 * L22 - L12 * L12;
  const C11 = L22 / det, C12 = -L12 / det, C22 = L11 / det;   // cov = Λ⁻¹
  const bx = 3, by = 3 + 1;                                    // Xᵀy = [y1, y1+y2]
  const mAo = C11 * bx + C12 * by, mBo = C12 * bx + C22 * by;
  const rhoO = C12 / Math.sqrt(C11 * C22);

  const ctx = setupCtx(MODEL, 4000);
  const m = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'ram', chains: 4, warmup: 1500, draws: 1500, seed: 1,
  });
  const A = m.fields.a.samples, B = m.fields.b.samples, n = A.length;
  let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += A[i]; mb += B[i]; } ma /= n; mb /= n;
  let va = 0, vb = 0, cab = 0;
  for (let i = 0; i < n; i++) { va += (A[i] - ma) ** 2; vb += (B[i] - mb) ** 2; cab += (A[i] - ma) * (B[i] - mb); }
  va /= n; vb /= n; cab /= n;
  const rho = cab / Math.sqrt(va * vb);

  assert.ok(Math.abs(ma - mAo) < 0.25, `mean a ${ma.toFixed(3)} vs oracle ${mAo.toFixed(3)}`);
  assert.ok(Math.abs(mb - mBo) < 0.25, `mean b ${mb.toFixed(3)} vs oracle ${mBo.toFixed(3)}`);
  assert.ok(Math.abs(va - C11) < 0.35 * C11, `var a ${va.toFixed(3)} vs oracle ${C11.toFixed(3)}`);
  assert.ok(Math.abs(vb - C22) < 0.35 * C22, `var b ${vb.toFixed(3)} vs oracle ${C22.toFixed(3)}`);
  assert.ok(Math.abs(rho - rhoO) < 0.12, `corr ${rho.toFixed(3)} vs oracle ${rhoO.toFixed(3)}`);

  const acc = m.diagnostics.acceptRate;
  assert.ok(acc > 0.12 && acc < 0.45, `acceptRate ${acc.toFixed(3)} near target 0.234`);
});
