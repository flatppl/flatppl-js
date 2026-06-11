'use strict';

// Standalone likelihood density (spec §06, audit H2 / Phase 3):
// `logdensityof(L, θ)` with L = likelihoodof(K, obs) scores the kernel at
// the GIVEN θ against the FIXED obs — pdf(κ(θ), obs), the spec's primary
// likelihood extraction. Previously: no derivation at all (the classifier
// treated arg0 as a measure binding and never unwrapped the L→K chain).
//
// The new `likelihood_density` derivation reuses the bayesupdate chain
// resolution (_resolveLikelihood) and the CLM lowering (derived value
// bindings inline down to the kernel's parametric inputs, H5/H3), but
// feeds the inputs EXPLICITLY from θ via lowerMeasure's opts.boundaries
// path instead of from a prior's atoms. `densityof(L, θ)` rides for free
// (it lowers to exp(<anon logdensityof>)).
//
// All expectations are CLOSED FORM (no MC tolerance): the score of an
// iid-Normal kernel at a fixed point is exact.

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function buildCtx(src: string, N: number) {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

const logN = (x: number, mu: number, s: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(s) - ((x - mu) ** 2) / (2 * s * s);

test('logdensityof(L, θ) scores the kernel at the given scalar θ (closed form)', async () => {
  const ctx = buildCtx(`
theta ~ Normal(0, 1)
obs ~ iid(Normal(theta, 1), 5)
fk = kernelof(record(obs = obs), theta = theta)
observed_data = [1.0, 1.5, 0.5, 1.2, 0.8]
L = likelihoodof(fk, record(obs = observed_data))
ld = logdensityof(L, 1.0)
dd = densityof(L, 1.0)
`, 64);
  assert.ok(ctx.derivations['ld'], 'logdensityof(L, θ) gets a derivation (H2)');
  assert.strictEqual(ctx.derivations['ld'].kind, 'likelihood_density');

  const expect = [1.0, 1.5, 0.5, 1.2, 0.8]
    .reduce((acc, x) => acc + logN(x, 1.0, 1), 0);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - expect) < 1e-12,
    `ld ${m.samples[0]} = Σ logN(x_i | θ=1, 1) = ${expect}`);
  // θ and obs are fixed → the per-atom value is constant across atoms.
  assert.strictEqual(m.samples[0], m.samples[m.samples.length - 1]);

  // densityof(L, θ) = exp(logdensityof(L, θ)) — lowered form, rides free.
  const md = await ctx.getMeasure('dd');
  assert.ok(Math.abs(md.samples[0] - Math.exp(expect)) < 1e-15,
    `dd ${md.samples[0]} = exp(ld)`);
});

test('logdensityof(L, record θ) feeds multi-param kernels; derived params inline (H5/H3)', async () => {
  // `a = 2 * theta2` is a DERIVED value binding between the kernel input
  // and the leaf distribution — the CLM boundary inlining must close it
  // so the explicit θ feed reaches the leaf (σ = 2·θ₂ = 2).
  const ctx = buildCtx(`
theta1 ~ Normal(0, 1)
theta2 ~ Exponential(1)
a = 2 * theta2
obs ~ iid(Normal(mu = theta1, sigma = a), 4)
fk = kernelof(record(obs = obs), theta1 = theta1, theta2 = theta2)
observed_data = [0.5, 1.5, -0.5, 1.0]
L = likelihoodof(fk, record(obs = observed_data))
ld = logdensityof(L, record(theta1 = 0.5, theta2 = 1.0))
`, 32);
  assert.strictEqual(ctx.derivations['ld'].kind, 'likelihood_density');
  const expect = [0.5, 1.5, -0.5, 1.0]
    .reduce((acc, x) => acc + logN(x, 0.5, 2), 0);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - expect) < 1e-12,
    `ld ${m.samples[0]} = Σ logN(x_i | μ=0.5, σ=2) = ${expect}`);
});

test('logdensityof(L, θ) works with a free elementof boundary (decoupled prior)', async () => {
  // The spec's decoupled form (04: a reified boundary is a fresh
  // elementof input): no draw for theta at all. Previously this shape
  // hard-pruned ("no derivation").
  const ctx = buildCtx(`
theta = elementof(reals)
obs ~ iid(Normal(theta, 1), 3)
fk = kernelof(record(obs = obs), theta = theta)
observed_data = [0.0, 0.5, 1.0]
L = likelihoodof(fk, record(obs = observed_data))
ld = logdensityof(L, 0.5)
`, 32);
  assert.ok(ctx.derivations['ld'], 'elementof-boundary likelihood density classifies');
  assert.strictEqual(ctx.derivations['ld'].kind, 'likelihood_density');
  const expect = [0.0, 0.5, 1.0]
    .reduce((acc, x) => acc + logN(x, 0.5, 1), 0);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - expect) < 1e-12,
    `ld ${m.samples[0]} = Σ logN(x_i | θ=0.5, 1) = ${expect}`);
});
