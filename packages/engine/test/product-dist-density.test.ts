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

test('product_dist: a discrete product never uses a continuous domain integral', async () => {
  // Poisson(2)*Poisson(3) has Z = exp(-5) Σ 6^k/(k!)² and log p(2)
  // = -1.0173273100296354. Midpoint integration instead gives Infinity on
  // [0,40], or a finite wrong score on [-1,16383] (twice the even-count sum).
  // Neither an unrelated domain declaration nor a finite answer certifies Z.
  for (const bounds of ['0.0, 40.0', '-1.0, 16383.0']) {
    const ctx = buildCtx(`
mu = elementof(posreals)
nu = elementof(posreals)
g1 = Poisson(rate = mu)
g2 = Poisson(rate = nu)
prod = normalize(logweighted(x -> logdensityof(g2, x), g1))
dom = cartprod(x = interval(${bounds}))
L = likelihoodof(prod, 2)
ld = logdensityof(L, record(mu = 2.0, nu = 3.0))
`, 1);
    const m = await ctx.getMeasure('ld');
    assert.ok(Math.abs(m.samples[0] - (-1.0173273100296354)) < 1e-12);
  }
});

test('product_dist: finite support sums include endpoint atoms and fixed factors', async () => {
  // Bernoulli(0.25)*Bernoulli(0.6) has normalized mass 1/3 at one.
  const ctx = buildCtx(`
p = elementof(unitinterval)
g1 = Bernoulli(p)
g2 = Bernoulli(0.6)
prod = normalize(logweighted(x -> logdensityof(g2, x), g1))
L = likelihoodof(prod, 1)
ld = logdensityof(L, record(p=0.25))
`, 1);
  assert.ok(Math.abs((await ctx.getMeasure('ld')).samples[0] - Math.log(1/3)) < 1e-12);
});

test('product_dist: infinite geometric support normalizes without a plotting domain', async () => {
  // The product has geometric ratio (1-.2)*(1-.3)=.56.
  const ctx = buildCtx(`
g1 = Geometric(0.2)
g2 = Geometric(0.3)
prod = normalize(logweighted(x -> logdensityof(g2, x), g1))
ld = logdensityof(prod, 7)
`, 1);
  assert.ok(Math.abs((await ctx.getMeasure('ld')).samples[0] - (Math.log(.44)+7*Math.log(.56))) < 1e-12);
});

test('product_dist: counting support and tail sums match finite and tilted-law oracles', async () => {
  // Geometric weighting tilts NB(r,q) into NB(r,q*(1-p)). The r=.5
  // case has increasing adjacent ratios, so the last ratio is not a tail bound.
  const cases = [
    ['Categorical([0.2,0.3,0.5])', 'Categorical([0.6,0.3,0.1])', 2, Math.log(.09/.26)],
    ['Categorical0([0.2,0.3,0.5])', 'Categorical0([0.6,0.3,0.1])', 1, Math.log(.09/.26)],
    ['Binomial(2,0.25)', 'Binomial(3,0.5)', 1, Math.log(.6)],
    ['NegativeBinomial(0.5,2.0)', 'Geometric(0.4)', 3, Math.log((.5*1.5*2.5/6)*Math.sqrt(.8)*.2**3)],
    ['NegativeBinomial2(6.0,2.0)', 'Geometric(0.2)', 7, Math.log(8*.4**2*.6**7)],
    ['Poisson(0.0)', 'Poisson(3.0)', 0, 0],
    ['Bernoulli(1.0)', 'Poisson(2.0)', 1, 0],
    ['Bernoulli(0.5)', 'Normal(0.0,1.0)', 1, -Math.log1p(Math.exp(.5))],
    ['Categorical0([0.0,1.0])', 'Gamma(0.5,1.0)', 1, 0],
    ['Categorical0([0.0,1.0])', 'Gamma(0.5,1.0)', 0, -Infinity],
    // Independent mode-relative PMF-ratio sum. The tolerance covers the
    // primitive log-PMF roundoff, not omitted tail mass.
    ['Poisson(1000000.0)', 'Poisson(1000000.0)', 1000000, -7.480120451073515],
  ] as const;
  for (const [base, weight, x, want] of cases) {
    const ctx = buildCtx(`
g1=${base}
g2=${weight}
prod=normalize(logweighted(x -> logdensityof(g2,x),g1))
ld=logdensityof(prod,${x})
`, 1);
    const got = (await ctx.getMeasure('ld')).samples[0];
    assert.ok(got === want || Math.abs(got-want) < (x === 1000000 ? 1e-8 : 1e-12), `${base} * ${weight}: ${got} versus ${want}`);
  }
});

test('product_dist: runtime inputs must satisfy the tail-bound parameter domains', async () => {
  const ctx = buildCtx(`
a=elementof(posreals)
g1=NegativeBinomial(a,2.0)
g2=Bernoulli(0.5)
prod=normalize(logweighted(x -> logdensityof(g2,x),g1))
L=likelihoodof(prod,0)
ld=logdensityof(L,record(a=-0.5))
`, 1);
  await assert.rejects(() => ctx.getMeasure('ld'), /invalid NegativeBinomial parameters/);
});

test('product_dist: a transformed scoring point does not use the shared-variate normalizer', async () => {
  const ctx = buildCtx(`
g1=Normal(0.0,1.0)
g2=Normal(0.0,1.0)
prod=normalize(logweighted(x -> logdensityof(g2,2.0*x),g1))
ld=logdensityof(prod,0.0)
`, 1);
  // This general weight still lacks named-measure closure support. It must
  // not silently score the different product g1(x)*g2(x).
  await assert.rejects(() => ctx.getMeasure('ld'), /measure ref/);
});

test('product_dist: disjoint supports and continuous-base PMF weights have zero mass', async () => {
  for (const [base, weight] of [['Bernoulli(0.0)','Bernoulli(1.0)'], ['Normal(0.0,1.0)','Poisson(2.0)']]) {
    const ctx = buildCtx(`
g1=${base}
g2=${weight}
prod=normalize(logweighted(x -> logdensityof(g2,x),g1))
ld=logdensityof(prod,0)
`, 1);
    await assert.rejects(() => ctx.getMeasure('ld'), /mass is zero|zero mass/);
  }
});
