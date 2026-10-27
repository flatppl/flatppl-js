'use strict';
// §06: each parameter slice divides by its own integral. The exponential
// oracles below come from elementary integration, independently confirmed by
// scipy quad/dblquad. Sampling and scoring must use the same normalizer.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const H = 'flatppl_compat = "0.1"\n';

function weightedModel(weight: string, support: string, observed: string): string {
  return H + 'theta ~ Normal(0.0, 2.0)\n'
    + `m = normalize(weighted((x,t) -> ${weight}, Lebesgue(${support}), theta))\n`
    + `y ~ m\nK = kernelof((y=y))\nL = likelihoodof(K,(y=${observed}))\n`
    + 'posterior = bayesupdate(L, lawof(theta))\n';
}

test('polynomial box integrals preserve mixed terms and axis order', async () => {
  // ∫[0,1]×[0,2] (1+t²u+2v+3uv) = 9+t².
  const source = weightedModel('1.0+t^2*x[1]+2.0*x[2]+3.0*x[1]*x[2]',
    'cartprod(interval(0.0,1.0),interval(0.0,2.0))', '[0.2,0.7]');
  for (const theta of [0.3, 1.4]) {
    const expected = Math.log(1 + theta ** 2 * 0.2 + 1.4 + 0.42) - Math.log(9 + theta ** 2);
    assert.ok(Math.abs(await likAt(source, theta) - expected) < 2e-12);
  }
});

test('the polynomial rule covers its highest exact degree', async () => {
  const source = weightedModel('1.0+t^2*x^9', 'interval(0.0,1.0)', '0.3');
  const expected = Math.log1p(1.5 ** 2 * 0.3 ** 9) - Math.log1p(1.5 ** 2 / 10);
  assert.ok(Math.abs(await likAt(source, 1.5) - expected) < 2e-12);
});

test('exponential integrals handle zero and tiny slopes', async () => {
  const source = weightedModel('exp(t*x)', 'interval(0.0,1.0)', '0.2');
  for (const theta of [0, -1e-12, 1e-12]) {
    // log Z = t/2 + t²/24 + O(t⁴); the omitted term is below 1e-49.
    const expected = 0.2 * theta - theta / 2 - theta ** 2 / 24;
    assert.ok(Math.abs(await likAt(source, theta) - expected) < 1e-14);
  }
});

test('exponential integrals keep the normalizer in log space', async () => {
  // Z ≈ exp(710)/|t| is finite for t=-1e308, although exp(710) and |t|*2
  // overflow. The observation stays in the finite range of exp as well.
  const source = weightedModel('exp(710.0+t*x)', 'interval(0.0,2.0)', '7.1e-306');
  const expected = (710 - 1e308 * 7.1e-306) - (710 - Math.log(1e308));
  assert.ok(Math.abs(await likAt(source, -1e308) - expected) < 2e-12);
  assert.ok(Math.abs(await likAt(source.replace('y=7.1e-306', 'y=0.0'), -1e308)
    - Math.log(1e308)) < 2e-12);
});

test('a steep exponential retains its small log normalizer', async () => {
  for (const weight of ['exp(t*(x-1.0))', 'exp(t*x)']) {
    const source = weightedModel(weight, 'interval(0.0,1.0)', '1.0');
    assert.ok(Math.abs(await likAt(source, 1e17) - Math.log(1e17)) < 2e-12);
  }
});

test('polynomial quadrature refuses a box whose interior nodes cannot be represented', async () => {
  const source = weightedModel('1.0+t*(x-1e16)', 'interval(1e16,10000000000000002.0)', '1e16');
  await assert.rejects(() => likAt(source, 1), { code: 'ENGINE_LIMITATION' });
});

test('a nonfinite computed normalizer stops runtime scoring instead of rejecting proposals', async () => {
  // Both integrals are positive and finite in real arithmetic. These scales
  // exceed the representable range of the polynomial evaluation route.
  for (const [weight, theta] of [['t*x^2', 5e-324], ['t*(x^8+x^9)', 1.6e308]] as const) {
    const source = weightedModel(weight, 'interval(0.0,1.0)', '0.1')
      .replace('Normal(0.0, 2.0)', 'Exponential(1.0)');
    await assert.rejects(() => likAt(source, theta), { undefinedNormalize: true });
  }
});

test('an unresolved narrow peak refuses scoring and sampling', async () => {
  // The exact mass is 1+0.1*sqrt(2π), to negligible boundary-tail error.
  // Fixed sampling and paired quadrature can both miss this peak entirely.
  const source = weightedModel('1.0+100.0*exp(-0.5*((x-t)/0.001)^2)',
    'interval(0.0,1.0)', '0.9');
  await assert.rejects(() => likAt(source, 0.5), { code: 'ENGINE_LIMITATION' });
  const { ctx } = ctxFor(source, 16);
  await assert.rejects(() => ctx.getMeasure('y'), { code: 'ENGINE_LIMITATION' });
});

const M1D = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'm = normalize(weighted((x, t) -> exp(0.0 - x / t), Lebesgue(support = interval(0.0, 1.0)), theta))\n'
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = 0.5))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

test('scalar and batched scoring reject proposals outside the prior before checking their normalizer', async () => {
  const { ctx } = ctxFor(M1D, 16);
  const scorer = await buildLogPi(ctx, postDeriv(ctx));
  const points = [{ theta: 0 }, { theta: 1.25 }];
  const scalar = points.map(scorer.logPi);
  assert.equal(scalar[0], -Infinity);
  assert.deepEqual(Array.from(scorer.logPiBatch(points)), scalar);
  const pair = scorer.priorLikBatch(points);
  assert.equal(pair.prior[0], -Infinity);
  assert.equal(pair.lik[0], -Infinity);
  assert.ok(Number.isFinite(pair.lik[1]));
});

const M2D = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'm = normalize(weighted((x, t) -> exp(0.0 - (x[1] + x[2]) / t), '
  + 'Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 2.0))), theta))\n'
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = [0.3, 0.7]))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

const M2D_ASYM = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'm = normalize(weighted((x, t) -> exp(0.0 - (x[1] + 2.0 * x[2]) / t), '
  + 'Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 2.0))), theta))\n'
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = [0.3, 0.7]))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

// A named weight with a record parameter receives the same whole 2-D variate
// as M2D. The record changes the parameter's representation, not the measure.
const M2D_VARIATE = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'weight = (w, p) -> exp(0.0 - (w[1] + w[2]) / p.theta)\n'
  + 'm = normalize(weighted(weight, '
  + 'Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 2.0))), record(theta = theta)))\n'
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = [0.3, 0.7]))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

// The SAME measure as M1D, written over a one-component cartprod. §03/§07 make
// that variate a length-1 VECTOR, so §06's "receives the variate whole" gives
// the weight an array to index — `w[1]`, not a bare `w`. Sharing M1D's oracle
// is the point: the two spellings denote one measure, so the whole-variate
// binding at k = 1 must reproduce the scalar-interval numbers exactly.
const M1D_BOX = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'm = normalize(weighted((w, t) -> exp(0.0 - w[1] / t), '
  + 'Lebesgue(support = cartprod(interval(0.0, 1.0))), theta))\n'
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = [0.5]))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

// scipy-confirmed closed forms (see the header).
const ORACLE_1D: Record<string, number> = {
  '0.6': -0.11317386031459675, '1.25': -0.026525872125234218, '1.9': -0.011515485860314256,
};
const ORACLE_2D: Record<string, number> = {
  '0.6': -0.3993557095150331, '1.25': -0.42415241019824657, '1.9': -0.4872538708836366,
};
const ORACLE_2D_ASYM: Record<string, number> = {
  '0.6': -0.9079276117012174, '1.25': -0.47490597052663985, '1.9': -0.4619047158890312,
};
// What a TRANSPOSED axis→parameter binding would score for M2D_ASYM. The two
// oracles are 0.53 nats apart at θ = 0.6 and only 0.046 apart at θ = 1.9, so
// only the θ = 0.6 point discriminates.
const ORACLE_2D_TRANSPOSED: Record<string, number> = {
  '0.6': -0.37921651784092636, '1.25': -0.3421058955860784, '1.9': -0.4162243263288049,
};

function postDeriv(ctx: any): any {
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') return v;
  }
  return null;
}

// Per-θ log-likelihood: the normalized density of `m` at the observed point,
// evaluated at exactly θ. This is the value the oracles predict.
async function likAt(src: string, theta: number): Promise<number> {
  const { ctx, proc } = ctxFor(src, 1);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const d = postDeriv(ctx);
  assert.ok(d, 'the model must produce a bayesupdate derivation');
  const { likOf } = await buildLogPi(ctx, d);
  return likOf({ theta });
}

// =====================================================================
// End-to-end density against the independent oracles
// =====================================================================

test('1-D: a θ-dependent normalizer tracks the closed-form Z(θ)', async () => {
  for (const t of ['0.6', '1.25', '1.9']) {
    const got = await likAt(M1D, +t);
    assert.ok(Math.abs(got - ORACLE_1D[t]) < 2e-12,
      `theta=${t}: got ${got}, oracle ${ORACLE_1D[t]}`);
  }
});

test('1-D: the θ-DEPENDENCE is real, not a constant shift', async () => {
  // The discriminating assertion. A baked constant Z gives lik(θ) = −x/θ − C,
  // so the difference between two θ misses Δ log Z — 0.468 nats between these
  // two points. Checking the DIFFERENCE cannot be satisfied by any constant.
  const lo = await likAt(M1D, 0.6);
  const hi = await likAt(M1D, 1.9);
  const want = ORACLE_1D['1.9'] - ORACLE_1D['0.6'];
  assert.ok(Math.abs((hi - lo) - want) < 2e-12,
    `Δlik across θ: got ${hi - lo}, oracle ${want}`);
});

test('2-D: a θ-dependent normalizer over a box tracks Z(θ)', async () => {
  for (const t of ['0.6', '1.25', '1.9']) {
    const got = await likAt(M2D, +t);
    assert.ok(Math.abs(got - ORACLE_2D[t]) < 2e-12,
      `theta=${t}: got ${got}, oracle ${ORACLE_2D[t]}`);
  }
});

test('2-D: variate index i binds to axis i, not the transpose', async () => {
  for (const t of ['0.6', '1.25', '1.9']) {
    const got = await likAt(M2D_ASYM, +t);
    assert.ok(Math.abs(got - ORACLE_2D_ASYM[t]) < 2e-12,
      `theta=${t}: got ${got}, oracle ${ORACLE_2D_ASYM[t]}`);
  }
  // The transposed box is a different measure, 0.53 nats away here.
  const at06 = await likAt(M2D_ASYM, 0.6);
  assert.ok(Math.abs(at06 - ORACLE_2D_TRANSPOSED['0.6']) > 0.3,
    `got ${at06}, which is the TRANSPOSED oracle ${ORACLE_2D_TRANSPOSED['0.6']}`);
});

test('a named weight with a record parameter receives the variate whole', async () => {
  // Same measure as M2D, so the same oracle. Binding only the first coordinate
  // would score a different function without erroring.
  for (const t of ['0.6', '1.9']) {
    const got = await likAt(M2D_VARIATE, +t);
    assert.ok(Math.abs(got - ORACLE_2D[t]) < 2e-12,
      `theta=${t}: got ${got}, oracle ${ORACLE_2D[t]}`);
  }
});

test('1-axis box: the whole-variate weight reproduces the scalar-interval numbers', async () => {
  // Same measure as M1D, so the SAME oracle — Z(θ) = θ(1 − e^{−1/θ}), scipy
  // quad to 1e-15. This is the θ-dependent witness for the k = 1 whole-variate
  // binding: the normalizer's integration points must feed the weight a
  // length-1 vector, exactly as the density route does, or the two disagree.
  for (const t of ['0.6', '1.25', '1.9']) {
    const got = await likAt(M1D_BOX, +t);
    assert.ok(Math.abs(got - ORACLE_1D[t]) < 2e-12,
      `theta=${t}: got ${got}, oracle ${ORACLE_1D[t]}`);
  }
});

test('1-axis box: the IS route and the MH route score the same measure', async () => {
  // The two-routes-one-measure gate, on the k = 1 whole-variate binding. Both
  // routes build their own Z(θ) expression; a binding that differed between
  // them would show up as a per-atom disagreement here.
  const { ctx } = ctxFor(M1D_BOX, 8);
  const post = await ctx.getMeasure('posterior');
  const th = await ctx.getMeasure('theta');
  const ts = Array.from(th.samples) as number[];
  const lw = Array.from(post.logWeights) as number[];
  const { ctx: ctx2 } = ctxFor(M1D_BOX, 1);
  const { likOf } = await buildLogPi(ctx2, postDeriv(ctx2));
  for (let i = 1; i < Math.min(5, ts.length); i++) {
    const isDelta = lw[i] - lw[0];
    const mhDelta = likOf({ theta: ts[i] }) - likOf({ theta: ts[0] });
    assert.ok(Math.abs(isDelta - mhDelta) < 1e-9,
      `atom ${i}: IS Δ ${isDelta} vs MH Δ ${mhDelta}`);
  }
});

// =====================================================================
// Determinism — independence from inference settings (§06 "Reproducibility")
// =====================================================================

test('the normalizer does not depend on the sample count', async () => {
  // Normalization must not depend on the inference sample count or seed.
  const a = await likAt(M1D, 1.25);
  const b = await likAt(M1D, 1.25);
  assert.equal(a, b, 'repeated scoring must be bit-identical');
  const { ctx: c1 } = ctxFor(M1D, 1);
  const { ctx: c2 } = ctxFor(M1D, 4096);
  const l1 = (await buildLogPi(c1, postDeriv(c1))).likOf({ theta: 1.25 });
  const l2 = (await buildLogPi(c2, postDeriv(c2))).likOf({ theta: 1.25 });
  assert.equal(l1, l2, `N=1 gave ${l1}, N=4096 gave ${l2}`);
});

test('the IS route and the MH route score the same measure', async () => {
  // Two routes, one measure. mat-density and mcmc-density each rewrite the
  // normalize independently, and the audit's whole subject is the two
  // disagreeing. The prior is Uniform over an interval of width 1.5, and the IS
  // posterior weight carries a flat −log N offset, so compare the DIFFERENCE
  // between two atoms — which cancels every θ-independent term.
  const N = 8;
  const { ctx } = ctxFor(M1D, N);
  const post = await ctx.getMeasure('posterior');
  const th = await ctx.getMeasure('theta');
  assert.ok(post.logWeights, 'the IS posterior must carry per-atom logWeights');
  const ts = Array.from(th.samples) as number[];
  const lw = Array.from(post.logWeights) as number[];
  const { ctx: ctx2 } = ctxFor(M1D, 1);
  const { likOf } = await buildLogPi(ctx2, postDeriv(ctx2));
  for (let i = 1; i < Math.min(5, ts.length); i++) {
    const isDelta = lw[i] - lw[0];
    const mhDelta = likOf({ theta: ts[i] }) - likOf({ theta: ts[0] });
    assert.ok(Math.abs(isDelta - mhDelta) < 1e-9,
      `atom ${i}: IS Δ ${isDelta} vs MH Δ ${mhDelta}`);
  }
});
