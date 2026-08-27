'use strict';
// A θ-DEPENDENT normalizer for `normalize(weighted(f, Lebesgue(support = S)))`,
// estimated by reweighting ONE fixed sample of the box (common random numbers).
//
// THE DEFECT. Spec §06 "Density of composed measures" requires the shift
// "logdensityof(normalize(M), x) = logdensityof(M, x) − log Z, with Z =
// totalmass(M)". When `f` references a latent, Z is a function of θ, and both
// density routes fell through to materialising the inner measure once and baking
// a CONSTANT −log Z. Measured on the base for the witness below (a rate inside
// an exponential — the non-bilinear case the integral-matrix trick cannot
// reach), the baked constant is wrong by 0.29 / 0.056 / 0.18 nats at θ =
// 0.6 / 1.25 / 1.9, and reports a POSITIVE log-density at two of the three
// points where the true value is negative. With the fixed-sample estimator the
// same three points land within 2.3e-4 / 1.5e-5 / 1.1e-5.
//
// ORACLES, all independent of the engine:
//   1-D  Z(θ) = ∫₀¹ e^{−x/θ} dx = θ(1 − e^{−1/θ}), hand-derived, and confirmed
//        by scipy.integrate.quad to 1e-16.
//   2-D  Z(θ) = θ(1 − e^{−1/θ}) · θ(1 − e^{−2/θ}) over [0,1]×[0,2],
//        hand-derived, confirmed by scipy.integrate.dblquad to 1e-16.
//   2-D asymmetric  f(u,v) = e^{−(u+2v)/θ} over [0,1]×[0,2]:
//        Z(θ) = θ(1 − e^{−1/θ}) · (θ/2)(1 − e^{−4/θ}), hand-derived, confirmed
//        by dblquad exactly. A TRANSPOSED axis→parameter binding scores a
//        different function — 0.53 nats away at θ = 0.6 — so this is what pins
//        the convention. (Note for anyone re-deriving these: dblquad's integrand
//        takes (inner, outer), and calling it with the arguments the other way
//        round yields the transposed box's integral, which agrees with the
//        symmetric case by commutativity and so hides the error.)
//
// TOLERANCES. The estimator is a Monte-Carlo integral over a fixed sample, so
// the tolerances below are the accuracy MEASURED at each test's point count,
// not a target. In 1-D at M = 128 the error is ≤ 2.3e-4. In 2-D it is ~1e-2 at
// M = 128 and ~2e-3 at M = 8192 for a smooth integrand; the asymmetric weight
// decays over 0.3 of a 2-wide axis, and a peaked integrand plateaus near 2e-2
// — a uniform point set spends most of its points where the weight is
// negligible. Every tolerance is still one to two orders of magnitude tighter
// than the 0.06–0.29 nat constant-bake error it has to exclude.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const crn = require('../crn-normalize.ts');

const H = 'flatppl_compat = "0.1"\n';

const M1D = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'm = normalize(weighted(x -> exp(0.0 - x / theta), Lebesgue(support = interval(0.0, 1.0))))\n'
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = 0.5))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

const M2D = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'm = normalize(weighted((u, v) -> exp(0.0 - (u + v) / theta), '
  + 'Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 2.0)))))\n'
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = [0.3, 0.7]))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

const M2D_ASYM = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'm = normalize(weighted((u, v) -> exp(0.0 - (u + 2.0 * v) / theta), '
  + 'Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 2.0)))))\n'
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = [0.3, 0.7]))\n'
  + 'posterior = bayesupdate(L, lawof(theta))\n';

// One-parameter weight over the same 2-D box: §06 says it "receives the variate
// whole", so it indexes the array. It denotes the SAME measure as M2D.
const M2D_VARIATE = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'm = normalize(weighted(w -> exp(0.0 - (w[1] + w[2]) / theta), '
  + 'Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 2.0)))))\n'
  + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = [0.3, 0.7]))\n'
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
async function likAt(src: string, theta: number, points?: number): Promise<number> {
  const { ctx, proc } = ctxFor(src, 1);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  if (points) ctx.crnNormalizePoints = points;
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
    assert.ok(Math.abs(got - ORACLE_1D[t]) < 1e-3,
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
  assert.ok(Math.abs((hi - lo) - want) < 1e-3,
    `Δlik across θ: got ${hi - lo}, oracle ${want}`);
});

test('2-D: a θ-dependent normalizer over a box tracks Z(θ)', async () => {
  for (const t of ['0.6', '1.25', '1.9']) {
    const got = await likAt(M2D, +t);
    assert.ok(Math.abs(got - ORACLE_2D[t]) < 3e-2,
      `theta=${t}: got ${got}, oracle ${ORACLE_2D[t]}`);
  }
});

test('2-D: parameter i binds to axis i, not the transpose', async () => {
  for (const t of ['0.6', '1.25', '1.9']) {
    const got = await likAt(M2D_ASYM, +t, 2048);
    assert.ok(Math.abs(got - ORACLE_2D_ASYM[t]) < 5e-2,
      `theta=${t}: got ${got}, oracle ${ORACLE_2D_ASYM[t]}`);
  }
  // θ = 0.6 is the discriminating point: the two conventions are 0.53 nats
  // apart there, against a 2e-2 estimator error.
  const at06 = await likAt(M2D_ASYM, 0.6, 2048);
  assert.ok(Math.abs(at06 - ORACLE_2D_TRANSPOSED['0.6']) > 0.3,
    `got ${at06}, which is the TRANSPOSED oracle ${ORACLE_2D_TRANSPOSED['0.6']}`);
});

test('a one-parameter weight over a box receives the variate whole (§06)', async () => {
  // Same measure as M2D by §06's arity rule, so the same oracle. Binding only
  // the first coordinate would score a different function without erroring.
  for (const t of ['0.6', '1.9']) {
    const got = await likAt(M2D_VARIATE, +t, 2048);
    assert.ok(Math.abs(got - ORACLE_2D[t]) < 2e-2,
      `theta=${t}: got ${got}, oracle ${ORACLE_2D[t]}`);
  }
});

// =====================================================================
// Determinism — the reason the sample is fixed (§06 "Reproducibility")
// =====================================================================

test('the estimate does not depend on the sample count', async () => {
  // The point set is seeded from the node's own content, never from the session
  // seed or N. Bit-for-bit, not approximately: a leak of ctx state into the
  // point set would show up here as a small difference.
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

test('the estimate improves with the point count', async () => {
  // Not a convergence-rate claim: a different M is a different fixed sample, so
  // the errors are independent draws rather than a monotone sequence. What must
  // hold is that a large M is far closer than a small one.
  const coarse = Math.abs(await likAt(M1D, 1.25, 16) - ORACLE_1D['1.25']);
  const fine = Math.abs(await likAt(M1D, 1.25, 8192) - ORACLE_1D['1.25']);
  assert.ok(fine < coarse / 10, `M=16 err ${coarse}, M=8192 err ${fine}`);
});

test('a large point count does not overflow the IR walkers', async () => {
  // The terms are folded into a BALANCED tree. A left-leaning chain of M terms
  // blew the recursive walkers' stack at M = 2048, and `likWith` swallowed the
  // RangeError to −∞ — a constant chain with no diagnostic.
  const v = await likAt(M1D, 1.25, 4096);
  assert.ok(Number.isFinite(v), `M=4096 scored ${v}`);
  assert.ok(Math.abs(v - ORACLE_1D['1.25']) < 1e-3, `M=4096 scored ${v}`);
});

// =====================================================================
// Recognition boundaries — what the estimator declines
// =====================================================================

function normalizeIR(inner: any) {
  return { kind: 'call', op: 'normalize', args: [inner] };
}
function weightedOver(refIR: any, params: string[], body: any) {
  return {
    kind: 'call', op: 'weighted',
    args: [{ kind: 'call', op: 'functionof', params, body }, refIR],
  };
}
function interval(lo: number, hi: number) {
  return {
    kind: 'call', op: 'interval',
    args: [{ kind: 'lit', value: lo }, { kind: 'lit', value: hi }],
  };
}
function lebesgue(supp: any) {
  return { kind: 'call', op: 'Lebesgue', kwargs: { support: supp } };
}
const XREF = { kind: 'ref', name: 'x' };

test('an unbounded axis is declined (no uniform sample over an infinite box)', () => {
  const unbounded = lebesgue({
    kind: 'call', op: 'interval',
    args: [{ kind: 'lit', value: 0 }, { kind: 'const', name: 'inf' }],
  });
  assert.equal(crn.crnRecognize(normalizeIR(weightedOver(unbounded, ['x'], XREF))), null);
});

test('a NON-unit bare Uniform base is declined (its density is 1/(b−a))', () => {
  const uni = { kind: 'call', op: 'Uniform', args: [interval(0, 3)] };
  assert.equal(crn.lebesgueBoxSupport(uni), null);
  // The unit interval IS the Lebesgue reading — density 1 either way.
  assert.ok(crn.lebesgueBoxSupport({ kind: 'call', op: 'Uniform', args: [interval(0, 1)] }));
  // ... and so is the #307 Lebesgue-restoring shift over any interval.
  assert.ok(crn.lebesgueBoxSupport({
    kind: 'call', op: 'logweighted',
    args: [{ kind: 'lit', value: Math.log(3) },
      { kind: 'call', op: 'Uniform', args: [interval(0, 3)] }],
  }));
});

test('a weight arity that is neither 1 nor the axis count is declined (§06)', () => {
  const box = lebesgue({ kind: 'call', op: 'cartprod', args: [interval(0, 1), interval(0, 1)] });
  assert.ok(crn.crnRecognize(normalizeIR(weightedOver(box, ['a', 'b'], XREF))), '2 params, 2 axes');
  assert.ok(crn.crnRecognize(normalizeIR(weightedOver(box, ['a'], XREF))), '1 param, 2 axes');
  assert.equal(crn.crnRecognize(normalizeIR(weightedOver(box, ['a', 'b', 'c'], XREF))), null,
    '3 params over 2 axes is an arity error, not a guess');
});

test('a constant-weight `weighted` is declined (it is a mass shift, not a density)', () => {
  const node = normalizeIR({
    kind: 'call', op: 'weighted',
    args: [{ kind: 'lit', value: 2 }, lebesgue(interval(0, 1))],
  });
  assert.equal(crn.crnRecognize(node), null);
});

test('a weight body over the node budget is declined rather than shrunk', () => {
  // A body of `nodes` nodes inlined M times must not silently drop to a smaller
  // M: that would change the measure being scored without saying so.
  let body: any = XREF;
  while (crn._internal.nodeCount(body) * 128 <= crn.CRN_NODE_BUDGET) {
    body = { kind: 'call', op: 'add', args: [body, { kind: 'lit', value: 1 }] };
  }
  const node = normalizeIR(weightedOver(lebesgue(interval(0, 1)), ['x'], body));
  assert.ok(crn._internal.nodeCount(body) * 128 > crn.CRN_NODE_BUDGET, 'the body must exceed the budget');
  assert.equal(crn.crnNormalizeMassExpr(node), null);
});

// =====================================================================
// The fixed point set itself
// =====================================================================

test('the point set is deterministic in the seed alone', () => {
  const axes = [{ lo: 0, hi: 1 }, { lo: -2, hi: 3 }];
  const a = crn.crnFixedPoints(axes, 64, 12345);
  const b = crn.crnFixedPoints(axes, 64, 12345);
  assert.deepEqual(a, b);
  const c = crn.crnFixedPoints(axes, 64, 12346);
  assert.notDeepEqual(a, c, 'a different seed must give a different sample');
});

test('every stratum of every axis holds exactly one point', () => {
  // The latin-hypercube property. Without it the sample is a plain MC draw and
  // the variance claim in the module header is wrong.
  const M = 64;
  const axes = [{ lo: 0, hi: 1 }, { lo: -2, hi: 3 }];
  const pts = crn.crnFixedPoints(axes, M, 999);
  assert.equal(pts.length, M);
  for (let j = 0; j < axes.length; j++) {
    const width = axes[j].hi - axes[j].lo;
    const hit = new Array(M).fill(0);
    for (const p of pts) {
      const u = (p[j] - axes[j].lo) / width;
      assert.ok(u > 0 && u < 1, `axis ${j}: point ${p[j]} outside the box`);
      hit[Math.min(M - 1, Math.floor(u * M))]++;
    }
    assert.deepEqual(hit, new Array(M).fill(1), `axis ${j} strata: ${hit.join(',')}`);
  }
});

test('the seed names the base measure only, not the weight', () => {
  // The sample is a sample of the BASE, so two spellings of one measure must
  // draw the same points. Seeding off the weight body broke the reified-vs-
  // lambda bit-for-bit agreement, because the two bodies differ in parameter
  // names. It also keeps a route-local annotation or source span out of the
  // seed, which is what makes the IS and MH routes agree.
  const axes = [{ lo: 0, hi: 1 }, { lo: -2, hi: 3 }];
  assert.equal(crn.seedString(axes, 128), crn.seedString(axes.map((a) => ({ ...a })), 128));
  assert.notEqual(crn.seedString(axes, 128), crn.seedString(axes, 256));
  assert.notEqual(crn.seedString(axes, 128), crn.seedString([{ lo: 0, hi: 2 }], 128));
});

test('two spellings of one measure score bit-for-bit alike', async () => {
  // A lambda weight and the equivalent `functionof` reification denote the same
  // measure (§06 "weight … a constant or a function of the variate"), so the
  // normalizer must not distinguish them. Not "within tolerance": equal.
  const lam = M1D;
  const reified = H
    + 'theta ~ Uniform(interval(0.5, 2.0))\n'
    + 'xv = elementof(interval(0.0, 1.0))\n'
    + 'wbody = exp(0.0 - xv / theta)\n'
    + 'wfn = functionof(wbody, xv = xv)\n'
    + 'm = normalize(weighted(wfn, Lebesgue(support = interval(0.0, 1.0))))\n'
    + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = 0.5))\n'
    + 'posterior = bayesupdate(L, lawof(theta))\n';
  const a = await likAt(lam, 1.25);
  const b = await likAt(reified, 1.25);
  assert.equal(a, b, `lambda ${a} vs reified ${b}`);
});

test('the terms fold into a balanced tree, not a chain', () => {
  const terms = new Array(1024).fill(0).map((_, i) => ({ kind: 'lit', value: i }));
  const tree = crn._internal.balancedAdd(terms);
  let depth = 0;
  for (let n = tree; n && n.args; n = n.args[0]) depth++;
  assert.equal(depth, 10, `depth ${depth} for 1024 terms (log2 = 10)`);
  const single = crn._internal.balancedAdd([{ kind: 'lit', value: 7 }]);
  assert.deepEqual(single, { kind: 'lit', value: 7 }, 'one term folds to itself');
});
