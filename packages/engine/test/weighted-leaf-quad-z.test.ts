'use strict';
// `normalize(weighted(w, <continuous scalar probability leaf>))` with a
// θ-independent `w` resolves its normalizer Z = ∫ w dB by DETERMINISTIC adaptive
// quadrature over the leaf's whole support, not by the seeded importance-sampling
// estimate the materialise fallback reads off the inner measure's tracked
// logTotalmass. Spec §06: "logdensityof(normalize(M), x) =
// logdensityof(M, x) − log Z, with Z = totalmass(M) finite and nonzero".
//
// The sibling of weighted-lebesgue-quad-z.test.ts, which pins the same property
// for a `Lebesgue(box)` base.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const H = 'flatppl_compat = "0.1"\n';

async function score(src: string, x: number, N: number, seed?: number): Promise<number> {
  const { ctx } = ctxFor(H + src + `\n__score__ = logdensityof(m, ${x})\n`, N);
  if (seed != null) { ctx.rootKey = seed; ctx.rootSeed = seed; }
  const mm = await ctx.getMeasure('__score__');
  return mm.value ? mm.value.data[0] : mm.samples[0];
}

// e^x·φ(x) = e^{1/2}·φ(x−1), so Z = e^{1/2} exactly and the normalized measure
// is Normal(1, 1). Oracle: mpmath, 40 dps.
const TILT = 'm = normalize(weighted(fn(exp(_)), Normal(mu = 0.0, sigma = 1.0)))';
const TILT_AT_HALF = -1.0439385332046727;

test('gaussian tilt: log-density matches the closed-form Z = e^{1/2}', async () => {
  const v = await score(TILT, 0.5, 1);
  assert.ok(Math.abs(v - TILT_AT_HALF) < 1e-11,
    `logdensityof = ${v}, want ${TILT_AT_HALF}`);
});

test('gaussian tilt: Z does not move with the seed or the sample count', async () => {
  const vals: number[] = [];
  for (const seed of [0xBA5E, 1, 12345, 99999]) {
    for (const N of [1, 64, 20000]) vals.push(await score(TILT, 0.5, N, seed));
  }
  const spread = Math.max(...vals) - Math.min(...vals);
  assert.equal(spread, 0, `Z must be deterministic; spread ${spread} over ${vals.length} runs`);
});

test('semi-infinite support: weighted Exponential normalizes to Gamma(2, 2)', async () => {
  // w(x) = x over Exponential(rate = 2): Z = E[X] = 1/2, so the result is
  // Gamma(shape = 2, rate = 2). logpdf at 1 = log 1 + log 2 − 2 − log(1/2).
  const v = await score('m = normalize(weighted(fn(_), Exponential(rate = 2.0)))', 1.0, 1);
  assert.ok(Math.abs(v - -0.6137056388801094) < 1e-10, `logdensityof = ${v}`);
});

test('unit-interval support: weighted Beta(2, 3) normalizes by Z = 2/5', async () => {
  const v = await score('m = normalize(weighted(fn(_), Beta(alpha = 2.0, beta = 3.0)))', 0.3, 1);
  assert.ok(Math.abs(v - 0.2799018851328186) < 1e-12, `logdensityof = ${v}`);
});

test('positive support: weighted LogNormal normalizes by its own mean', async () => {
  // w(x) = x over LogNormal(0, 1): Z = E[X] = e^{1/2}, and x·p(x)/Z is
  // LogNormal(1, 1). logpdf at 2 = −log 2 − log√(2π) − (log 2 − 1)²/2.
  const want = -Math.log(2) - 0.5 * Math.log(2 * Math.PI)
    - Math.pow(Math.log(2) - 1, 2) / 2;
  const v = await score('m = normalize(weighted(fn(_), LogNormal(mu = 0.0, sigma = 1.0)))', 2.0, 1);
  assert.ok(Math.abs(v - want) < 1e-10, `logdensityof = ${v}, want ${want}`);
});

test('the unnormalized measure keeps its own level (no divisor applies)', async () => {
  const v = await score('m = weighted(fn(exp(_)), Normal(mu = 0.0, sigma = 1.0))', 0.5, 1);
  // log(e^{0.5}·φ(0.5)) = 0.5 + log φ(0.5) = TILT_AT_HALF + log Z.
  assert.ok(Math.abs(v - (TILT_AT_HALF + 0.5)) < 1e-12, `logdensityof = ${v}`);
});

test('a heavy-tailed base is normalized, not mistaken for a divergence', async () => {
  // Cauchy has no mean, so `Z = ∫ e^{−x²} dCauchy` is the case that separates a
  // real divergence from a base whose own tail is heavy: the inverse-CDF map
  // cancels the base density, so the integrand is the weight alone and decays.
  const v = await score(
    'm = normalize(weighted(x -> exp(0.0 - x * x), Cauchy(location = 0.0, scale = 1.0)))',
    0.5, 1);
  assert.ok(Number.isFinite(v), `expected a finite log-density, got ${v}`);
});

test('a non-integrable weight is refused, not silently normalized', async () => {
  // e^{x²}·φ(x) grows without bound, so Z = ∫ w dB does not exist. Before the
  // endpoint test the quadrature reported Ẑ = 4.97e12 at rel-err 1.0e-10.
  await assert.rejects(
    score('m = normalize(weighted(x -> exp(x * x), Normal(mu = 0.0, sigma = 1.0)))', 0.5, 1),
    /does not decay against this base measure/);
});

test('a weight the base cannot damp is refused on a heavy-tailed base too', async () => {
  // |x| has no Cauchy expectation. The adaptive routine alone called this
  // converged at Ẑ = 21.9.
  await assert.rejects(
    score('m = normalize(weighted(x -> abs(x), Cauchy(location = 0.0, scale = 1.0)))', 0.5, 1),
    /does not decay against this base measure/);
});

test('an everywhere-zero weight is refused (spec §06 leaves Z = 0 undefined)', async () => {
  await assert.rejects(
    score('m = normalize(weighted(x -> 0.0 * x, Normal(mu = 0.0, sigma = 1.0)))', 0.5, 1),
    /Z = ∫ w dB is 0/);
});

test('a discrete base declines the arm and keeps the existing fallback', async () => {
  // Poisson has no entry in the inverse-CDF ladder, so the arm declines; the
  // query must still produce a number by the pre-existing route.
  const v = await score('m = normalize(weighted(fn(_), Poisson(rate = 3.0)))', 2.0, 4000);
  assert.ok(Number.isFinite(v), `expected a finite log-density, got ${v}`);
});

test('a θ-dependent weight goes to the per-θ arm, at the exact Z(θ)', async () => {
  // The weight references a latent, so Z is per-θ and this arm must not bake one
  // constant. leaf-mass-quad emits Z(θ) as an expression instead, and the value
  // is the closed-form e^{θ²/2}: `normalize(weighted(fn(exp(θ·_)), Normal(0,1)))`
  // is Normal(θ, 1), so the likelihood at 0.5 is log N(0.5; 1, 1) at θ = 1. The
  // Monte-Carlo route this replaced returned −1.885 at N = 1 and −1.654 at
  // N = 4000. leaf-mass-quad.test.ts owns the arm's own witnesses.
  const { buildLogPi } = require('../mcmc-density.ts');
  const src = H
    + 'theta ~ Uniform(interval(0.5, 2.0))\n'
    + 'm = normalize(weighted(x -> exp(theta * x), Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = 0.5))\n'
    + 'posterior = bayesupdate(L, lawof(theta))\n';
  const at = async (N: number) => {
    const { ctx } = ctxFor(src, N);
    let deriv: any = null;
    for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
      if (v && (v as any).kind === 'bayesupdate') deriv = v;
    }
    const { likOf } = await buildLogPi(ctx, deriv);
    return likOf({ theta: 1.0 });
  };
  const a = await at(1), b = await at(4000);
  assert.equal(a, b, 'the per-θ arm must not move with the sample count');
  assert.ok(Math.abs(a - TILT_AT_HALF) < 1e-7,
    `likelihoodof at θ = 1 is ${a}, closed form ${TILT_AT_HALF}`);
});

test('the arm itself returns null for a θ-dependent weight', () => {
  // The end-to-end θ-dependent shape above never reaches `resolveNormalizeMasses`
  // (the MH route builds its own log-π), so pin the decline on the arm directly:
  // a weight referencing a binding that is neither a constant nor a function is
  // per-θ, and the arm must hand back null rather than bake one Z.
  const { weightedLeafQuadLogZ } = require('../mat-density.ts');
  const leaf = {
    kind: 'call', op: 'Normal',
    kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } },
  };
  const node = (body: any) => ({
    kind: 'call', op: 'normalize',
    args: [{
      kind: 'call', op: 'weighted',
      args: [{ kind: 'call', op: 'functionof', params: ['x'], body }, leaf],
    }],
  });
  const tilt = { kind: 'call', op: 'exp',
    args: [{ kind: 'call', op: 'mul', args: [
      { kind: 'ref', ns: 'self', name: 'theta' },
      { kind: 'ref', ns: '%local', name: 'x' }] }] };
  // `theta` is a plain module binding, so it is a latent: decline.
  const latent = { bindings: new Map([['theta', { ir: { kind: 'lit', value: 1 } }]]),
    fixedValues: new Map() };
  assert.equal(weightedLeafQuadLogZ(node(tilt), latent), null);
  // The SAME body with `theta` resolved to a constant is θ-independent, so the
  // arm takes it and reproduces the e^{1/2} normalizer.
  const fixed = { bindings: new Map(), fixedValues: new Map([['theta', 1]]) };
  const logZ = weightedLeafQuadLogZ(node(tilt), fixed);
  assert.ok(Math.abs(logZ - 0.5) < 1e-11, `log Z = ${logZ}, want 0.5`);
});

test('a normalize whose inner is not a weighted declines the arm', async () => {
  // normalize(truncate(...)) reaches the same resolver; the truncate arm must
  // still own it.
  const v = await score(
    'm = normalize(truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, 1.0)))', 0.5, 1);
  assert.ok(Number.isFinite(v), `expected a finite log-density, got ${v}`);
});

test('a multi-parameter weight declines the arm (§06 arity is per-axis)', async () => {
  // A 2-parameter weight over a scalar leaf is not the shape this arm reads, so
  // it must decline rather than bind the variate to the first parameter.
  const { ctx } = ctxFor(H
    + 'm = normalize(weighted(2.0, Normal(mu = 0.0, sigma = 1.0)))\n'
    + '__score__ = logdensityof(m, 0.5)\n', 1);
  const mm = await ctx.getMeasure('__score__');
  const v = mm.value ? mm.value.data[0] : mm.samples[0];
  // A constant weight cancels exactly: normalize(weighted(2, B)) = B.
  const want = -0.5 * Math.log(2 * Math.PI) - 0.125;
  assert.ok(Math.abs(v - want) < 1e-12, `logdensityof = ${v}, want ${want}`);
});
