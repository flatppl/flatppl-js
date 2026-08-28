'use strict';
// SAMPLER/DENSITY AGREEMENT for `normalize(weighted(f, Lebesgue(box)))` with a
// STOCHASTIC θ — the surface the θ-dependent normalizer (crn-normalize.ts) left
// unverified.
//
// THE DEFECT. `matNormalize` divided the parent's weights by their POOLED sum.
// That sum is the parent's mass averaged over the atom ensemble, so when `f`
// references a latent — where the parent's mass is Z(θ_i), a different number on
// every atom — atom i kept the residue Z(θ_i)/E[Z]. Spec §06 "Normalization"
// makes `normalize(M)` a probability measure, so the θ-marginal of the sampled
// joint must be the prior unchanged; the residue tilted it towards large-Z θ
// instead, while `logdensityof` of the same measure divided by Ẑ(θ) per θ. Two
// routes, one measure, two answers — the class measure-algebra-audit.md exists
// for.
//
// ORACLES, independent of the engine (scipy.integrate.quad, and the exact
// prior mean where the property is exact):
//   1-D  f(x;θ) = e^{θx} over [0,1], θ ~ Uniform(0,4). Z(θ) = (e^θ − 1)/θ.
//        CORRECT   E[θ] = 2 exactly, E[y] = 0.6488050479, E[y²] = 0.4943055580.
//        Z-TILTED  E[θ] = 2.8073315742, E[y] = 0.7018328935, E[y²] = 0.5546754391.
//        Measured before the fix: E[θ] = 2.8087, E[y] = 0.7018, E[y²] = 0.5546 —
//        the TILTED oracle, to 3-4 digits.
//   2-D  f(u,v) = e^{−(u+v)/θ} over [0,1]×[0,2], θ ~ Uniform(0.5,2).
//        CORRECT   E[θ] = 1.25 exactly, E[u] = 0.4246374, E[v] = 0.7152499.
//        Z-TILTED  E[θ] = 1.3989707, E[u] = 0.4346912, E[v] = 0.7495397.
//        Measured before the fix: E[θ] = 1.40009, E[u] = 0.43311, E[v] = 0.75081.
//   θ-independent control  f(x) = e^{−x} over [0,1]: E[y] = (1−2e^{−1})/(1−e^{−1})
//        = 0.4180232931. The pooled divisor is CORRECT here, and this path must
//        leave it bit-for-bit alone.
//
// WHY E[θ] IS THE DISCRIMINATING MOMENT. It is exact under the spec — no
// quadrature error — and the two oracles are 0.81 (1-D) and 0.15 (2-D) apart,
// against a Monte-Carlo error of ~5e-3 at these sample counts. The y moments
// separate the two hypotheses by only 0.010–0.053, so they confirm rather than
// discriminate.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const H = 'flatppl_compat = "0.1"\n';

// A rate INSIDE an exponential: Z(θ) is not algebraic in θ, so this is the
// shape the closed-form mass routes decline and the fixed-sample estimator owns.
const S1D = H
  + 'theta ~ Uniform(interval(0.0, 4.0))\n'
  + 'm = normalize(weighted(x -> exp(theta * x), Lebesgue(support = interval(0.0, 1.0))))\n'
  + 'y ~ m\n';

const S2D = H
  + 'theta ~ Uniform(interval(0.5, 2.0))\n'
  + 'm = normalize(weighted((u, v) -> exp(0.0 - (u + v) / theta), '
  + 'Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 2.0)))))\n'
  + 'y ~ m\n';

// The same box and the same base measure, with a weight that does NOT reference
// a latent. Z is a constant, the pooled divisor is exactly right, and this path
// must not touch it.
const S_FIXED = H
  + 'theta ~ Uniform(interval(0.0, 4.0))\n'
  + 'm = normalize(weighted(x -> exp(0.0 - x), Lebesgue(support = interval(0.0, 1.0))))\n'
  + 'y ~ m\n';

const N = 60000;

function logSumExp(a: number[]): number {
  let mx = -Infinity;
  for (const v of a) if (v > mx) mx = v;
  let s = 0;
  for (const v of a) s += Math.exp(v - mx);
  return mx + Math.log(s);
}

// Weighted moments of the sampled (θ, y) ensemble. `k` is the variate width, so
// `ey[j]` is E[y_j] over axis j of a k-axis box.
async function jointMoments(src: string, k: number, n = N) {
  const { proc, ctx } = ctxFor(src, n);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const y = await ctx.getMeasure('y');
  const th = await ctx.getMeasure('theta');
  const ts = Array.from(th.samples) as number[];
  const lw: number[] = y.logWeights
    ? Array.from(y.logWeights) as number[]
    : ts.map(() => -Math.log(n));
  const norm = logSumExp(lw);
  const ey = new Array(k).fill(0);
  let et = 0;
  let ey2 = 0;
  for (let i = 0; i < n; i++) {
    const w = Math.exp(lw[i] - norm);
    for (let j = 0; j < k; j++) ey[j] += w * y.samples[i * k + j];
    ey2 += w * y.samples[i * k] * y.samples[i * k];
    et += w * ts[i];
  }
  return { et, ey, ey2, dims: y.dims, n_eff: y.n_eff, samples: y.samples, ts, lw };
}

// =====================================================================
// The witness
// =====================================================================

test('1-D: the sampled θ-marginal is the prior, not the prior tilted by Z(θ)', async () => {
  const m = await jointMoments(S1D, 1);
  // θ ~ Uniform(0,4), and every m_θ is a probability measure by §06, so the
  // θ-marginal of the joint is the prior exactly.
  assert.ok(Math.abs(m.et - 2.0) < 0.02, `E[θ] = ${m.et}, prior mean 2.0`);
  // The failing hypothesis, excluded by 0.7 against a 0.02 tolerance.
  assert.ok(Math.abs(m.et - 2.8073315742) > 0.5,
    `E[θ] = ${m.et} is the Z-TILTED oracle 2.8073315742`);
});

test('1-D: the sampled y moments match the quadrature oracle', async () => {
  const m = await jointMoments(S1D, 1);
  assert.ok(Math.abs(m.ey[0] - 0.6488050479) < 5e-3,
    `E[y] = ${m.ey[0]}, oracle 0.6488050479`);
  assert.ok(Math.abs(m.ey2 - 0.4943055580) < 5e-3,
    `E[y²] = ${m.ey2}, oracle 0.4943055580`);
  assert.ok(Math.abs(m.ey[0] - 0.7018328935) > 2e-2,
    `E[y] = ${m.ey[0]} is the Z-TILTED oracle 0.7018328935`);
});

test('2-D: a θ-dependent normalizer over a box samples the untilted joint', async () => {
  const m = await jointMoments(S2D, 2);
  assert.deepEqual(m.dims, [2], `normalize must keep the vector-atom shape, got ${m.dims}`);
  assert.ok(Math.abs(m.et - 1.25) < 0.02, `E[θ] = ${m.et}, prior mean 1.25`);
  assert.ok(Math.abs(m.et - 1.3989707) > 0.1,
    `E[θ] = ${m.et} is the Z-TILTED oracle 1.3989707`);
  assert.ok(Math.abs(m.ey[0] - 0.4246374) < 5e-3, `E[u] = ${m.ey[0]}, oracle 0.4246374`);
  assert.ok(Math.abs(m.ey[1] - 0.7152499) < 5e-3, `E[v] = ${m.ey[1]}, oracle 0.7152499`);
});

test('the sampler divides by the SAME Ẑ(θ) the density subtracts', async () => {
  // The two-routes-one-measure check, with no quadrature and no oracle in it.
  // The sampler's weight on atom i is log f(y_i; θ_i) − log Ẑ(θ_i) − c, and the
  // density route at a fixed y* scores log f(y*; θ) − log Ẑ(θ). Subtracting the
  // known log f = θ·y from each leaves −log Ẑ(θ) on both sides, up to the
  // sampler's one constant — so the DIFFERENCES between atoms must agree. They
  // are equal to 1e-9, not merely close: the point set is seeded from the box
  // axes alone, so both routes build the same expression.
  const YSTAR = 0.5;
  const m = await jointMoments(S1D, 1, 512);
  const score = H
    + 'theta ~ Uniform(interval(0.0, 4.0))\n'
    + 'm = normalize(weighted(x -> exp(theta * x), Lebesgue(support = interval(0.0, 1.0))))\n'
    + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = ' + YSTAR + '))\n'
    + 'posterior = bayesupdate(L, lawof(theta))\n';
  const { ctx } = ctxFor(score, 1);
  let post: any = null;
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') post = v;
  }
  assert.ok(post, 'the scoring model must produce a bayesupdate derivation');
  const { likOf } = await buildLogPi(ctx, post);
  const minusLogZhat = (i: number) => m.lw[i] - m.ts[i] * m.samples[i];
  const density = (i: number) => likOf({ theta: m.ts[i] }) - m.ts[i] * YSTAR;
  for (let i = 1; i < 12; i++) {
    const sDelta = minusLogZhat(i) - minusLogZhat(0);
    const dDelta = density(i) - density(0);
    assert.ok(Math.abs(sDelta - dDelta) < 1e-9,
      `atom ${i} (θ = ${m.ts[i]}): sampler Δ(−log Ẑ) ${sDelta} vs density Δ ${dDelta}`);
  }
});

test('2-D: the sampler divides by the SAME Ẑ(θ) the density subtracts', async () => {
  // The same identity over a 2-axis box, where the weight is
  // log f(u, v; θ) = −(u + v)/θ. The 1-D case cannot see an axis-ordering or
  // volume-factor difference between the two routes; this one can.
  const m = await jointMoments(S2D, 2, 512);
  const score = H
    + 'theta ~ Uniform(interval(0.5, 2.0))\n'
    + 'm = normalize(weighted((u, v) -> exp(0.0 - (u + v) / theta), '
    + 'Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 2.0)))))\n'
    + 'y ~ m\nK = kernelof(record(y = y))\nL = likelihoodof(K, record(y = [0.3, 0.7]))\n'
    + 'posterior = bayesupdate(L, lawof(theta))\n';
  const { ctx } = ctxFor(score, 1);
  let post: any = null;
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') post = v;
  }
  assert.ok(post, 'the scoring model must produce a bayesupdate derivation');
  const { likOf } = await buildLogPi(ctx, post);
  for (let i = 1; i < 12; i++) {
    const sAt = (j: number) =>
      m.lw[j] + (m.samples[j * 2] + m.samples[j * 2 + 1]) / m.ts[j];
    const dAt = (j: number) => likOf({ theta: m.ts[j] }) + 1.0 / m.ts[j];
    const sDelta = sAt(i) - sAt(0);
    const dDelta = dAt(i) - dAt(0);
    assert.ok(Math.abs(sDelta - dDelta) < 1e-9,
      `atom ${i} (θ = ${m.ts[i]}): sampler Δ(−log Ẑ) ${sDelta} vs density Δ ${dDelta}`);
  }
});

test('a weight whose mass estimates to zero is refused, not sampled', async () => {
  // §06 "normalize": "If Z = 0 or Z = ∞, the result is undefined." The density
  // route already refuses a non-finite −log Ẑ shift; the sampler must refuse the
  // same measure rather than divide by zero and hand back a garbage ensemble.
  const zero = H
    + 'theta ~ Uniform(interval(0.5, 2.0))\n'
    + 'm = normalize(weighted(x -> 0.0 * theta * x, Lebesgue(support = interval(0.0, 1.0))))\n'
    + 'y ~ m\n';
  const { ctx } = ctxFor(zero, 16);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('y')),
    /estimated total mass .* is 0.*spec §06/s);
});

test('over the node budget the sampler refuses rather than pooling the divisor',
  async () => {
    // crn-normalize.ts declines to inline a weight body M times past its node
    // budget. It used to return null there and let the sampler reinstate the
    // POOLED divisor behind a `console.warn`, which is the tilted measure this
    // whole path exists to remove: measured at this point count, E[θ] = 2.8124
    // against the prior's 2.0 and the Z-tilted 2.8073. §06 makes every θ-slice a
    // probability measure and its engine contract says an engine "does not
    // silently substitute heuristics", so the budget is a refusal.
    //
    // Deliberateness is preserved the same way the pooled pin preserved it: the
    // assertion names the budget and the shape, so a quiet drop to a smaller M —
    // or a quiet return of the tilted ensemble — fails here rather than passing.
    // Forced through the point count, which multiplies the same budget.
    const n = 20000;
    const { proc, ctx } = ctxFor(S1D, n);
    assert.equal(proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    ctx.crnNormalizePoints = 400000;
    await assert.rejects(() => Promise.resolve(ctx.getMeasure('y')),
      /the weight depends on a latent.*over the budget of 400000.*spec §06 normalize/s);
  });

test('over the node budget the density route refuses too', async () => {
  // The same builder feeds `logdensityof`, where the fallback was a baked
  // constant −log Z rather than a pooled weight sum — the same pooled number by
  // another name. One measure, one verdict.
  const src = S1D + 'K = kernelof(record(y = y))\n'
    + 'L = likelihoodof(K, record(y = 0.5))\n'
    + 'posterior = bayesupdate(L, lawof(theta))\n';
  const { proc, ctx } = ctxFor(src, 64);
  assert.equal(proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  ctx.crnNormalizePoints = 400000;
  let post: any = null;
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') post = v;
  }
  assert.ok(post, 'the scoring model must produce a bayesupdate derivation');
  await assert.rejects(() => Promise.resolve(buildLogPi(ctx, post)),
    /the weight depends on a latent.*over the budget of 400000.*spec §06 normalize/s);
});

test('over the node budget a θ-INDEPENDENT weight still keeps the pooled divisor',
  async () => {
    // The gate on the refusal. Z is one constant here, so the pooled mass is
    // exactly it and refusing would reject a correct answer. Same point count,
    // same budget overrun, opposite verdict — and the number must still be the
    // one the pooled path produces, bit-for-bit.
    const n = 60000;
    const { proc, ctx } = ctxFor(S_FIXED, n);
    assert.equal(proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    ctx.crnNormalizePoints = 400000;
    const y = await ctx.getMeasure('y');
    const th = await ctx.getMeasure('theta');
    const lw = Array.from(y.logWeights) as number[];
    const norm = logSumExp(lw);
    let ey = 0;
    let et = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.exp(lw[i] - norm);
      ey += w * y.samples[i];
      et += w * th.samples[i];
    }
    assert.equal(ey, 0.4170387508194253, `E[y] = ${ey} moved`);
    assert.equal(et, 2.0013148946121326, `E[θ] = ${et} moved`);
  });

test('a θ-INDEPENDENT weight keeps the pooled divisor untouched', async () => {
  // The control that pins the per-atom divisor to a θ-DEPENDENT weight. Here Z
  // is a constant, the pooled sum is exactly right, and the numbers must be the
  // ones the pooled path already produced — bit-for-bit, which is what the
  // hard-coded value below records (measured on the unmodified checkout).
  const m = await jointMoments(S_FIXED, 1);
  assert.equal(m.ey[0], 0.4170387508194253, `E[y] = ${m.ey[0]} moved`);
  assert.equal(m.et, 2.0013148946121326, `E[θ] = ${m.et} moved`);
  // ... and it is the right number: f = e^{−x} over [0,1] has
  // E[x] = (1 − 2e^{−1})/(1 − e^{−1}).
  assert.ok(Math.abs(m.ey[0] - 0.4180232931) < 5e-3,
    `E[y] = ${m.ey[0]}, closed form 0.4180232931`);
});
