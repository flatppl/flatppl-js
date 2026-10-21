'use strict';

const assert = require('node:assert/strict');
const { inBothModes, inAllOptModes } = require('./_perf-helpers.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { logDensity, logDensityN } = require('../density.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');
const { orchestrator } = require('..');

const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
const binomial = () => ({ kind: 'call', op: 'Binomial', args: [ref('n'), ref('p')] });
const near = (actual: number, expected: number) => assert.ok(Math.abs(actual - expected) < 1e-10,
  `${actual} != ${expected}`);

// Finite products give an independent coefficient oracle without beta/gamma.
function mass(n: number, p: number, k: number): number {
  if (p === 0) return k === 0 ? 0 : -Infinity;
  if (p === 1) return k === n ? 0 : -Infinity;
  let choose = 1;
  for (let j = 1; j <= Math.min(k, n - k); j++) choose *= (n - j + 1) / j;
  return Math.log(choose) + k * Math.log(p) + (n - k) * Math.log1p(-p);
}

inBothModes('Binomial scoring keeps changed trials and observations live on shared IR', 'density.binomialCoefficients', () => {
  const ir = binomial();
  const env = { n: 8, p: .2 };
  for (const [n, p, k] of [[8, .2, 2], [12, .7, 4], [8, .2, 2]]) {
    Object.assign(env, { n, p });
    near(logDensity(ir, k, env), mass(n, p, k));
    near(logDensity(ir, 3, { n: 9, p: .4 }), mass(9, .4, 3));
    near(logDensity(ir, k, env), mass(n, p, k));
  }
  const n = Float64Array.of(8, 12, 8), p = Float64Array.of(.2, .7, .2), k = [2, 4, 2];
  logDensityN(ir, k, { n, p }, 3, { pointsBatched: true }).forEach((x: number, i: number) => near(x, mass(n[i], p[i], k[i])));
});

inBothModes('tilted Binomial coefficients keep changed trials and observations live on shared IR', 'density.binomialCoefficients', () => {
  const { ctx } = ctxFor(`
n = elementof(posintegers)
p = elementof(unitinterval)
theta = elementof(reals)
T = normalize(logweighted(fn(theta * _), Binomial(n, p)))
`, 1);
  const ir = orchestrator.expandMeasure('T', ctx);
  const env = { n: 8, p: .2, theta: 1000 };
  for (const [n, p, theta, k] of [[8, .2, 1000, 2], [12, .7, -1000, 2],
    [8, .2, 1000, 3], [8, .2, 1000, 2]]) {
    Object.assign(env, { n, p, theta });
    // Normalize all finite-support weights directly, without a log-odds rewrite.
    const weights = Array.from({ length: n + 1 }, (_, j) => mass(n, p, j) + theta * j);
    const max = Math.max(...weights);
    const expected = weights[k] - max - Math.log(weights.reduce((s, x) => s + Math.exp(x - max), 0));
    near(logDensity(ir, k, env), expected);
  }
});

inBothModes('Binomial scores survive coefficient-cache capacity and reuse', 'density.binomialCoefficients', () => {
  const ir = binomial();
  // More pairs than the bounded memo retains, with a simple quadratic oracle.
  const trials = Float64Array.from({ length: 1057 }, (_, i) => 100 + i);
  const observations = new Float64Array(trials.length).fill(2);
  for (let pass = 0; pass < 2; pass++) {
    const scores = logDensityN(ir, observations, { n: trials }, trials.length,
      { pointsBatched: true, baseEnv: { p: .3 } });
    scores.forEach((x: number, i: number) => near(x, mass(trials[i], .3, 2)));
    near(logDensity(ir, 2, { n: trials[0], p: .7 }), mass(trials[0], .7, 2));
  }
});

inAllOptModes('Binomial IID preparation retains endpoints, scalar conversion and invalid-input scores',
  ['density.binomialCoefficients', 'density.sharedParameters'], () => {
    const leaf = binomial();
    const iid = { kind: 'call', op: 'iid', args: [leaf, { kind: 'lit', value: 2 }] };
    near(logDensity(iid, [true, false], { n: 4, p: .3 }), mass(4, .3, 1) + mass(4, .3, 0));
    for (const [p, point, other] of [[0, 0, 1], [1, 4, 3]]) {
      assert.equal(logDensity(iid, [point, point], { n: 4, p }), 0);
      assert.equal(logDensity(iid, [point, other], { n: 4, p }), -Infinity);
    }
    assert.equal(logDensity(iid, [-1, 2], { n: 4, p: .3 }), -Infinity);
    assert.ok(Number.isNaN(logDensity(iid, [1, NaN], { n: 4, p: .3 })));
    assert.ok(Number.isNaN(logDensity(iid, [1, 2], { n: 2.5, p: .3 })));
    assert.ok(Number.isNaN(logDensity(iid, [1, 2], { n: 4, p: 1.1 })));
    assert.ok(Number.isNaN(logDensity(leaf, 1, { n: Infinity, p: .3 })));
  });

inAllOptModes('Binomial plates and mixtures share normalized constrained and transformed posterior scores',
  ['density.binomialCoefficients', 'density.sharedParameters'], async () => {
    const ys = [0, 2, 8], trials = [8, 12, 9];
    const { ctx } = ctxFor(`
p ~ Beta(2, 3)
A = Binomial(8, p)
B = Binomial(12, p)
f(t) = Binomial(t, p)
single ~ A
plate ~ Binomial.([8, 12, 9], p)
repeated ~ iid(A, 3)
callable ~ f.([8, 12, 9])
independent ~ iid(superpose(weighted(.25, A), weighted(.75, B)), 3)
shared ~ superpose(weighted(.25, iid(A, 3)), weighted(.75, iid(B, 3)))
P = lawof(record(p = p))
K = kernelof(record(single = single, plate = plate, repeated = repeated, callable = callable,
  independent = independent, shared = shared), p = p)
L = likelihoodof(pars -> K(p = pars.p), record(single = 2, plate = [0, 2, 8], repeated = [0, 2, 8],
  callable = [0, 2, 8], independent = [0, 2, 8], shared = [0, 2, 8]))
posterior = bayesupdate(L, P)
`, 1);
    const likelihood = (p: number) => {
      const a = ys.map(k => mass(8, p, k)), b = ys.map(k => mass(12, p, k));
      const mix = (x: number, y: number) => Math.log(.25 * Math.exp(x) + .75 * Math.exp(y));
      return mass(8, p, 2) + 2 * ys.reduce((s, k, i) => s + mass(trials[i], p, k), 0)
        + a.reduce((s, x) => s + x, 0) + a.reduce((s, x, i) => s + mix(x, b[i]), 0)
        + mix(a.reduce((s, x) => s + x, 0), b.reduce((s, x) => s + x, 0));
    };
    const points = [{ p: .2 }, { p: .7 }, { p: .2 }];
    const expected = points.map(({ p }) => likelihood(p));
    const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
    points.forEach((point, i) => near(scorer.likOf(point), expected[i]));
    scorer.priorLikBatch(points).lik.forEach((x: number, i: number) => near(x, expected[i]));
    const model = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
    const free = points.map(point => model.unconstrainAll(point));
    // Beta(2,3) normalization and the canonical unit-interval chart Jacobian.
    const posterior = points.map(({ p }, i) => expected[i] + Math.log(12) + 2 * Math.log(p) + 3 * Math.log1p(-p));
    free.forEach((z, i) => near(model.logPosterior(z), posterior[i]));
    model.logPosteriorBatch(free).forEach((x: number, i: number) => near(x, posterior[i]));
  });
