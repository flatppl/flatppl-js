'use strict';
// packages/engine/test/slice-funnel.test.ts
// Slice on harder geometries where it should shine (no covariance to break).
// (a) Exactly-correlated 2-D Gaussian: recovers the correlation (closed-form
//     2×2 inverse oracle). (b) A stiff/anisotropic posterior: slice must mix
//     (finite R̂, non-constant, stiff coord explored) — no freeze to collapse.

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
    derivations: built.derivations, bindings: built.bindings, fixedValues: built.fixedValues || new Map(),
    moduleRegistry: lifted.loweredModule && lifted.loweredModule.moduleRegistry ? lifted.loweredModule.moduleRegistry : null,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n); const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p; },
    sendWorker: (m: any) => { const r = worker.handle(m); return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r); },
    sampleCount: N, rootSeed: 42, rootKey: 42,
  };
  return ctx;
}
function stats(s: Float64Array) { let m = 0; for (let i = 0; i < s.length; i++) m += s[i]; m /= s.length; let v = 0; for (let i = 0; i < s.length; i++) v += (s[i] - m) ** 2; return { m, sd: Math.sqrt(v / s.length) }; }

const CORR = `
a = draw(Normal(mu = 0.0, sigma = 3.0))
b = draw(Normal(mu = 0.0, sigma = 3.0))
prior = lawof(record(a = a, b = b))
obs = joint(y1 = Normal(mu = a + b, sigma = 1.0), y2 = Normal(mu = b, sigma = 1.0))
K = functionof(obs, a = a, b = b)
L = likelihoodof(K, record(y1 = 3.0, y2 = 1.0))
posterior = bayesupdate(L, prior)
`;

test('backend:slice recovers an exactly-correlated 2-D Gaussian', async () => {
  const p = 1 / 9, L11 = 1 + p, L12 = 1, L22 = 2 + p, det = L11 * L22 - L12 * L12;
  const C11 = L22 / det, C12 = -L12 / det, C22 = L11 / det, rhoO = C12 / Math.sqrt(C11 * C22);
  const m = await materialiser.materialiseMeasure('posterior', setupCtx(CORR, 4000), { backend: 'slice', chains: 4, warmup: 500, draws: 1500, seed: 1 });
  const A = m.fields.a.samples, B = m.fields.b.samples, n = A.length;
  const sa = stats(A), sb = stats(B);
  let cab = 0; for (let i = 0; i < n; i++) cab += (A[i] - sa.m) * (B[i] - sb.m); cab /= n;
  const rho = cab / (sa.sd * sb.sd);
  assert.ok(Math.abs(sa.sd - Math.sqrt(C11)) < 0.3 * Math.sqrt(C11), `sd a ${sa.sd.toFixed(3)} vs ${Math.sqrt(C11).toFixed(3)}`);
  assert.ok(Math.abs(rho - rhoO) < 0.12, `corr ${rho.toFixed(3)} vs oracle ${rhoO.toFixed(3)}`);
});

const STIFF = `
a = draw(Normal(mu = 0.0, sigma = 5.0))
b = draw(Normal(mu = 0.0, sigma = 0.02))
prior = lawof(record(a = a, b = b))
obs = joint(y = Normal(mu = a + b, sigma = 1.0))
K = functionof(obs, a = a, b = b)
L = likelihoodof(K, record(y = 1.0))
posterior = bayesupdate(L, prior)
`;

test('backend:slice mixes on a stiff/anisotropic posterior', async () => {
  const m = await materialiser.materialiseMeasure('posterior', setupCtx(STIFF, 3000), { backend: 'slice', chains: 4, warmup: 500, draws: 1000, seed: 1 });
  const b = m.fields.b.samples;
  const sb = stats(b);
  assert.ok(Number.isFinite(m.diagnostics.perParam.b.rHat), 'b rHat finite');
  assert.ok(sb.sd > 0.003 && sb.sd < 0.05, `b std ${sb.sd.toFixed(5)} explores its ~0.02 scale (slice self-tunes width)`);
});
