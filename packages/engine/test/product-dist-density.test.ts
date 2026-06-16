'use strict';

// Shared-variate product_dist density scoring (spec §12, HS3 paper A.2).
//
// product_dist over the SAME observable lowers to the normalized pointwise
// density product `normalize(logweighted(x -> Σ logdensityof(Mᵢ, x), M0))`.
// Scoring `logdensityof(L, θ)` of such a product needs two engine pieces that
// previously threw / silently mis-normalized:
//   1. the `functionof`-variate logweighted weight — the log-weight is a
//      function of the SAME point being scored (walkLogWeighted);
//   2. the normalizer −logZ = −log ∫ ∏ᵢ gᵢ — closed-form for Normal factors
//      (the product of Gaussians is Gaussian), resolved at θ in mat-density.
//
// Oracle is INDEPENDENT closed form: the normalized product of N Normals is a
// Normal with 1/σ*² = Σ 1/σᵢ² and μ* = σ*² Σ μᵢ/σᵢ²; the per-point score is
// logpdf of that Gaussian. (Not ROOT, not the other engine — derived here.)

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function buildCtx(src: string, N: number) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
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

// Normalized product of Normals → Normal(μ*, σ*).
function combinedNormal(params: Array<[number, number]>): [number, number] {
  let tau = 0, wmu = 0;
  for (const [mu, s] of params) { const inv = 1 / (s * s); tau += inv; wmu += mu * inv; }
  return [wmu / tau, Math.sqrt(1 / tau)];
}

const MODEL = (likelihood: string) => `
mu1 = elementof(reals)
sigma1 = elementof(posreals)
mu2 = elementof(reals)
sigma2 = elementof(posreals)
g1 = Normal(mu = mu1, sigma = sigma1)
g2 = Normal(mu = mu2, sigma = sigma2)
prod = normalize(logweighted(x -> logdensityof(g2, x), g1))
${likelihood}
ld = logdensityof(L, record(mu1 = 0.0, sigma1 = 1.0, mu2 = 1.0, sigma2 = 2.0))
`;

test('product_dist: single-observation density matches the closed-form combined Gaussian', async () => {
  const ctx = buildCtx(MODEL('L = likelihoodof(prod, 0.83)'), 1);
  const [muStar, sStar] = combinedNormal([[0, 1], [1, 2]]);
  const expect = logN(0.83, muStar, sStar);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - expect) < 1e-12,
    `score ${m.samples[0]} = logN(0.83 | μ*=${muStar}, σ*=${sStar}) = ${expect}`);
});

test('product_dist: iid unbinned likelihood is the sum of per-entry closed-form scores', async () => {
  const toy = [-0.0285673, 0.83014143, 1.16860338, 2.29038875, 0.18297688];
  const lit = '[' + toy.join(', ') + ']';
  const ctx = buildCtx(MODEL(`L = likelihoodof(iid(prod, ${toy.length}), ${lit})`), 1);
  const [muStar, sStar] = combinedNormal([[0, 1], [1, 2]]);
  const expect = toy.reduce((acc, x) => acc + logN(x, muStar, sStar), 0);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - expect) < 1e-11,
    `Σ score ${m.samples[0]} = Σ logN(x_i | μ*, σ*) = ${expect}`);
});

test('product_dist: non-Gaussian factors fall back to numeric quadrature of the normalizer', async () => {
  // Normal × Cauchy over the SAME observable x: no closed-form product, so the
  // normalizer −logZ = −log ∫ N(x|0,1)·Cauchy(x|0.5,1) dx is computed by
  // quadrature over the variate's declared domain x ∈ [-40, 40].
  const src = `
mu = elementof(reals)
loc = elementof(reals)
g1 = Normal(mu = mu, sigma = 1.0)
g2 = Cauchy(location = loc, scale = 1.0)
prod = normalize(logweighted(x -> logdensityof(g2, x), g1))
dom = cartprod(x = interval(-40.0, 40.0))
L = likelihoodof(prod, 0.7)
ld = logdensityof(L, record(mu = 0.0, loc = 0.5))
`;
  // Independent oracle: scipy integral of the same integrand over the same
  // bounds — logN(0.7|0,1) + logCauchy(0.7|0.5,1) − logsumexp ∫ over [-40,40].
  // (python: norm.logpdf + cauchy.logpdf − logsumexp(grid) − log dx.)
  const SCIPY = -0.7221686750374490;
  const ctx = buildCtx(src, 1);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - SCIPY) < 1e-6,
    `Normal×Cauchy score ${m.samples[0]} ≈ scipy numeric ${SCIPY}`);
});

test('product_dist: three-factor product folds the log-weight and normalizer', async () => {
  const src = `
a = elementof(reals)
s = elementof(posreals)
g1 = Normal(mu = a, sigma = s)
g2 = Normal(mu = a, sigma = s)
g3 = Normal(mu = a, sigma = s)
prod = normalize(logweighted(x -> logdensityof(g2, x) + logdensityof(g3, x), g1))
L = likelihoodof(prod, 0.4)
ld = logdensityof(L, record(a = 0.0, s = 1.0))
`;
  const ctx = buildCtx(src, 1);
  const [muStar, sStar] = combinedNormal([[0, 1], [0, 1], [0, 1]]);
  const expect = logN(0.4, muStar, sStar);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - expect) < 1e-12,
    `3-factor score ${m.samples[0]} = logN(0.4 | μ*, σ*=${sStar}) = ${expect}`);
});
