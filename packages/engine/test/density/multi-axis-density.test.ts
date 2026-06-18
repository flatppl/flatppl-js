'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource, orchestrator, materialiser } = require('../..');
const { createWorkerHandler } = require('../../worker.ts');

const SEED = 0xFEED;
const TOL = 1e-9;

function materialise(src: string, target: string, N: number): Promise<any> {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: N,
    rootSeed: SEED,
    rootKey: SEED,
    marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m: any) => {
      const r = w.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message))
        : Promise.resolve(r);
    },
  };
  return ctx.getMeasure(target);
}

// =====================================================================
// Oracle-driven density tests: load points + expected from JSON
// =====================================================================

const CASES = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'multi-axis-density.json'), 'utf8')
) as Array<{ name: string; src: string; point: any; expected: number; oracle: string }>;

for (const c of CASES) {
  test('multi-axis density oracle: ' + c.name, async () => {
    const m = await materialise(c.src, '__score__', 1);
    const val: number = m.samples != null ? m.samples[0]
      : (m.value && m.value.data != null ? m.value.data[0] : NaN);
    assert.ok(
      Math.abs(val - c.expected) < TOL,
      `[${c.name}] got ${val}, expected ${c.expected} (Δ ${Math.abs(val - c.expected)})\n  oracle: ${c.oracle}`
    );
  });
}

// =====================================================================
// Sample ≡ density agreement: Normal [2,3] grid
// =====================================================================
// Materialise Normal.(mu, 1.0) for a [2,3] grid, then pick one cell
// (g=0, k=1 → mu=1.0). Draw many samples, compute the empirical log
// density ratio between two points (x1=1.0, x2=2.0), and compare it
// to the density difference at those points.
// logpdf(Normal(1,1),x1) - logpdf(Normal(1,1),x2) = (x2²-x1²)/2 + (x1-x2)*mu
// = (4-1)/2 + (1-2)*1 = 1.5 - 1 = 0.5
// Tolerance is 0.1 (sampling noise).

test('multi-axis: sample≡density agreement Normal [2,3] grid', async () => {
  const N = 80000;
  const m = await materialise(
    'mu = [[0.0,1.0,2.0],[3.0,4.0,5.0]]\n' +
    'r ~ Normal.(mu, 1.0)\n',
    'r', N);

  // m.value.shape = [N, 2, 3], row-major
  assert.deepEqual(m.value.shape, [N, 2, 3]);
  const data: Float64Array = m.value.data;
  const G = 2, K = 3;

  // Extract cell (g=0, k=1): mu=1.0
  // offset = i*G*K + g*K + k = i*6 + 0*3 + 1 = i*6 + 1
  const g = 0, k = 1;
  const mu = 1.0;
  const x1 = 1.0, x2 = 2.0;

  let n1 = 0, n2 = 0;
  const hw = 0.1;
  for (let i = 0; i < N; i++) {
    const s = data[i * G * K + g * K + k];
    if (s >= x1 - hw && s < x1 + hw) n1++;
    if (s >= x2 - hw && s < x2 + hw) n2++;
  }

  assert.ok(n1 > 100 && n2 > 100,
    `too few samples in windows (n1=${n1}, n2=${n2}); increase N`);

  // Empirical log density ratio (within same bin width → ratio cancels width)
  const empiricalLogRatio = Math.log(n1 / n2);
  // Analytical log density ratio: logpdf(Normal(mu,1),x1) - logpdf(Normal(mu,1),x2)
  //  = -0.5*(x1-mu)^2 + 0.5*(x2-mu)^2
  const analyticLogRatio = -0.5 * (x1 - mu) ** 2 + 0.5 * (x2 - mu) ** 2;

  assert.ok(
    Math.abs(empiricalLogRatio - analyticLogRatio) < 0.1,
    `sample≡density: empirical log ratio ${empiricalLogRatio.toFixed(4)}, ` +
    `analytic ${analyticLogRatio.toFixed(4)} (Δ ${Math.abs(empiricalLogRatio - analyticLogRatio).toFixed(4)})`
  );
});
