'use strict';

// kchain density, M1 (audit §3, H1 boundary-binding family). `kchain(M, K)`
// has marginal density ∫ κ(a, ·) dμ(a) over the PRIOR μ = M. The 2-step
// density path left the kernel's NAMED boundary params as self-refs that
// prepareDensityRefs resolved via getMeasure — the like-named DRAW, not the
// prior measure. When the prior DECOUPLES from those draws (a relabelled /
// transformed prior), the density scored the wrong measure: it contradicted
// the SAMPLE histogram of the very same measure by hundreds of nats.
//
// Fix: matLogdensityof materialises base.ref (the prior) and feeds its variate
// columns as the boundary inputs — mirroring the sample-side matJointchain.
//
// Counterexample: the boundary draw is `theta ~ Normal(0,1)`, but the prior
// given to the chain is `shifted = theta + 20` (mean ~20). Sampling honours
// the prior (mean ~20). The marginal density must therefore PEAK near 20, not
// near 0 — exact marginal at z=20 is Normal(20; 20, sqrt2) → -0.5·log(4π).

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const der = require(ENG + 'derivations.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function makeCtx(src: string, targets: string[], N: number) {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  for (const nm of targets) {
    if (built.bindings.has(nm) && !built.derivations[nm]) {
      const c = der.classifyDerivation(built.bindings.get(nm), built.bindings);
      if (c) built.derivations[nm] = c;
    }
  }
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N, rootKey: 42,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}
const scalar1 = (m: any) => (m.value ? m.value.data[0] : (m.samples ? m.samples[0] : m));

test('kchain density integrates over the PRIOR, not the like-named draws (M1)', async () => {
  const ctx = makeCtx(`
theta ~ Normal(0.0, 1.0)
shifted = theta + 20.0
M = lawof(shifted)
K = functionof(Normal(mu = shifted, sigma = 1.0), shifted = shifted)
ch = kchain(M, K)
ld20 = logdensityof(ch, 20.0)
ld0  = logdensityof(ch, 0.0)
`, ['ch', 'ld20', 'ld0', 'K', 'M'], 8000);

  // Sampling honours the prior: the chain variate concentrates near 20.
  const ch = await ctx.getMeasure('ch');
  const obs = ch.fields ? ch.fields[Object.keys(ch.fields)[0]] : ch;
  let mean = 0;
  for (let i = 0; i < obs.samples.length; i++) mean += obs.samples[i];
  mean /= obs.samples.length;
  assert.ok(mean > 18 && mean < 22, `chain sample mean ${mean.toFixed(2)} should be ~20 (prior honoured)`);

  const lp20 = +scalar1(await ctx.getMeasure('ld20'));
  const lp0 = +scalar1(await ctx.getMeasure('ld0'));
  // DENSITY must agree with the histogram: high near 20, far lower near 0.
  // (Pre-fix it was inverted — scored at the theta~N(0,1) draws.)
  assert.ok(lp20 > lp0 + 50,
    `logp(20)=${lp20.toFixed(2)} must dominate logp(0)=${lp0.toFixed(2)} (density must honour the prior)`);
  // In the bulk the MC marginal is exact: Normal(20; 20, sqrt2) → -0.5·log(4π).
  const exact = -0.5 * Math.log(4 * Math.PI);
  assert.ok(Math.abs(lp20 - exact) < 0.1,
    `logp(20)=${lp20.toFixed(4)} should match the exact marginal ${exact.toFixed(4)}`);
});

test('kchain density: hole-param scalar prior still integrates over the prior measure', async () => {
  // Regression: the already-correct hole-param path (single unmatched param
  // rewired to base.ref) must be unchanged. ∫ N(2;a,1) N(a;0,10) da =
  // N(2; 0, sqrt101) → logpdf ≈ -3.246.
  const ctx = makeCtx(`
a ~ Normal(0.0, 0.01)
K = functionof(Normal(mu = a, sigma = 1.0), a = a)
M = Normal(0.0, 10.0)
ch = kchain(M, K)
ld = logdensityof(ch, 2.0)
`, ['ch', 'ld', 'K'], 8000);
  const lp = +scalar1(await ctx.getMeasure('ld'));
  const exact = -0.5 * Math.log(2 * Math.PI * 101) - 4 / (2 * 101);
  assert.ok(Math.abs(lp - exact) < 0.1,
    `hole-param kchain logp(2)=${lp.toFixed(4)} should match the exact marginal ${exact.toFixed(4)}`);
});
