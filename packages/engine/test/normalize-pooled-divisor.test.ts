'use strict';
// THE POOLED DIVISOR, swept across every `normalize` SAMPLING shape.
//
// THE DEFECT. `matNormalize` divides the parent's weights by their POOLED sum,
// which is the parent's mass averaged over the atom ensemble. Where the mass
// moves with a latent that is NOT the variate, the parent's mass is Z(θ_i) — a
// different number on every atom — and the pooled divisor leaves the residue
// Z(θ_i)/E[Z] on atom i.
//
// THE ORACLE, and it is exact. Spec §06 `normalize`: "given a measure M with
// finite total mass Z = totalmass(M) > 0, returns the probability measure M / Z
// … On a non-nullary kernel, normalizes the output measures." Every θ-slice is
// therefore a probability measure, so the θ-MARGINAL of the sampled joint is
// the prior UNCHANGED — no quadrature enters, and the failing hypothesis has
// its own closed form (the prior tilted by Z(θ)). Each witness below pins the
// correct value AND excludes the tilted one, so a green test cannot mean "the
// number moved somewhere plausible".
//
// #216 wired the fixed-sample Ẑ(θ) for `weighted(f, Lebesgue(box))`. This file
// covers the rest of the surface: the closed-form `totalMassExpr` shapes (a
// scalar mass factor over a probability measure, in log space too, over a
// record/tuple variate), the shapes with no per-θ expression (refused), and
// the shapes where the pooled divisor is right (pinned bit-for-bit).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const H = 'flatppl_compat = "0.1"\n';
const N = 60000;

function logSumExp(a: number[]): number {
  let mx = -Infinity;
  for (const v of a) if (v > mx) mx = v;
  let s = 0;
  for (const v of a) s += Math.exp(v - mx);
  return mx + Math.log(s);
}

// Weighted moments of the sampled (θ, y) ensemble, plus the covariance that
// the mixing-proportion witnesses need.
async function joint(src: string, n = N, latent = 'theta', target = 'y') {
  const { proc, ctx } = ctxFor(src, n);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const y = await ctx.getMeasure(target);
  const th = await ctx.getMeasure(latent);
  const ts = Array.from(th.samples) as number[];
  const lw: number[] = y.logWeights
    ? Array.from(y.logWeights) as number[]
    : ts.map(() => -Math.log(n));
  const norm = logSumExp(lw);
  // A record/tuple measure carries its variate in `fields`/`elems`; take the
  // first component, which is all these witnesses read.
  const sv: Float64Array | null = y.samples
    ? y.samples
    : (y.fields ? y.fields[Object.keys(y.fields)[0]].samples
                : (y.elems ? y.elems[0].samples : null));
  let et = 0; let ey = 0; let ety = 0; let et2 = 0;
  for (let i = 0; i < n; i++) {
    const w = Math.exp(lw[i] - norm);
    et += w * ts[i];
    et2 += w * ts[i] * ts[i];
    if (sv) { ey += w * sv[i]; ety += w * ts[i] * sv[i]; }
  }
  return { et, ey, cov: ety - et * ey, varT: et2 - et * et, m: y, n_eff: y.n_eff };
}

// =====================================================================
// FIXED — a closed-form θ-dependent mass, reused from `totalMassExpr`
// =====================================================================

// θ ~ Uniform(1, 5) and Z(θ) = θ, so the tilted marginal is
// E[θ²]/E[θ] = (124/12)/3 = 3.4444444444 — 0.44 from the prior's 3.0, against a
// Monte-Carlo error of ~4e-3 at this sample count.
const B = H
  + 'theta ~ Uniform(interval(1.0, 5.0))\n'
  + 'm = normalize(weighted(theta, Normal(mu = 0.0, sigma = 1.0)))\n'
  + 'y ~ m\n';

const TILT_B = 3.4444444444444444;

test('a scalar mass factor in a latent: the θ-marginal is the prior, not the '
  + 'prior tilted by Z(θ)', async () => {
  const r = await joint(B);
  assert.ok(Math.abs(r.et - 3.0) < 0.02, `E[θ] = ${r.et}, prior mean 3.0`);
  assert.ok(Math.abs(r.et - TILT_B) > 0.3, `E[θ] = ${r.et} is the Z-TILTED ${TILT_B}`);
  // normalize(weighted(θ, Normal(0,1))) IS Normal(0,1) at every θ — the factor
  // divides straight out — so the y-marginal is standard normal and the atoms
  // come back equally weighted.
  assert.ok(Math.abs(r.ey) < 0.02, `E[y] = ${r.ey}, oracle 0`);
  assert.ok(Math.abs(r.n_eff - N) < 1e-6,
    `the divisor must leave the atoms equally weighted, n_eff ${r.n_eff}`);
});

test('the same mass factor in LOG space is divided out per atom', async () => {
  // Z(θ) = e^θ over θ ~ Uniform(1, 5). The tilted marginal is
  // ∫θe^θ / ∫e^θ = 4e⁵/(e⁵−e) = 4.0746294415 (scipy.integrate.quad agrees to
  // 10 digits), a full nat above the prior.
  const r = await joint(H
    + 'theta ~ Uniform(interval(1.0, 5.0))\n'
    + 'm = normalize(logweighted(theta, Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'y ~ m\n');
  assert.ok(Math.abs(r.et - 3.0) < 0.02, `E[θ] = ${r.et}, prior mean 3.0`);
  assert.ok(Math.abs(r.et - 4.0746294415) > 0.5,
    `E[θ] = ${r.et} is the Z-TILTED 4.0746294415`);
});

test('a RECORD-variate parent takes the same per-atom divisor', async () => {
  // The structured branch of matNormalize renormalises the TOP-LEVEL weights
  // and leaves the fields alone, so it had the identical pooled divisor and the
  // identical tilt — E[θ] was 3.4463720282, the scalar witness's number to 4
  // digits. §06 "Joint composition" gives the mass of the inner joint:
  // (M1⊗M2)(A×B) = M1(A)·M2(B) = 1, so Z(θ) = θ exactly as above.
  const r = await joint(H
    + 'theta ~ Uniform(interval(1.0, 5.0))\n'
    + 'm = normalize(weighted(theta, joint(a = Normal(mu = 0.0, sigma = 1.0), '
    + 'b = Normal(mu = 3.0, sigma = 1.0))))\n'
    + 'y ~ m\n');
  assert.equal(r.m.shape, 'record', 'normalize must keep the record variate');
  assert.deepEqual(Object.keys(r.m.fields), ['a', 'b']);
  assert.ok(Math.abs(r.et - 3.0) < 0.02, `E[θ] = ${r.et}, prior mean 3.0`);
  assert.ok(Math.abs(r.et - TILT_B) > 0.3, `E[θ] = ${r.et} is the Z-TILTED ${TILT_B}`);
});

test('a TUPLE-variate parent takes the same per-atom divisor', async () => {
  const r = await joint(H
    + 'theta ~ Uniform(interval(1.0, 5.0))\n'
    + 'm = normalize(weighted(theta, joint(Normal(mu = 0.0, sigma = 1.0), '
    + 'Normal(mu = 3.0, sigma = 1.0))))\n'
    + 'y ~ m\n');
  assert.equal(r.m.shape, 'tuple', 'normalize must keep the tuple variate');
  assert.ok(Math.abs(r.et - 3.0) < 0.02, `E[θ] = ${r.et}, prior mean 3.0`);
  assert.ok(Math.abs(r.et - TILT_B) > 0.3, `E[θ] = ${r.et} is the Z-TILTED ${TILT_B}`);
});

test('the density route divides by the SAME Z(θ) the sampler now does', async () => {
  // Two routes, one measure. Both call `totalMassExpr` on the same inner IR, so
  // the sampler's per-atom divisor is the density route's −log Z shift. At a
  // FIXED θ the density has a closed form to check it against:
  // normalize(weighted(θ, Normal(0,1))) is Normal(0,1) for EVERY θ, so
  // logdensityof at 0.5 is log N(0.5; 0, 1) = −1.0439385332046727 — the same
  // number at two very different θ, which is what "the shift tracks θ" means.
  // A pooled or baked Z would leave a log(θ/Z̄) residue and move with θ.
  const ORACLE = -1.0439385332046727;
  for (const th of [0.4, 7.3]) {
    const src = H
      + 'theta = ' + th + '\n'
      + 'm = normalize(weighted(theta, Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'ld = logdensityof(m, 0.5)\n';
    const { proc, ctx } = ctxFor(src, 1);
    assert.equal(proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    const ld = await ctx.getMeasure('ld');
    assert.ok(Math.abs(ld.samples[0] - ORACLE) < 1e-9,
      `θ = ${th}: logdensityof = ${ld.samples[0]}, closed form ${ORACLE}`);
  }
});

// =====================================================================
// REFUSED — a θ-dependent mass with no per-θ expression
// =====================================================================

test('a θ-dependent mass with no closed form is refused, not sampled tilted',
  async () => {
    // `weighted(x -> exp(θx), Normal(0,1))` has Z(θ) = e^{θ²/2}: not a scalar
    // factor (the weight is a function of the variate) and not an integral over
    // a Lebesgue box (the base is a probability measure), so neither expression
    // builder covers it. The pooled divisor returned E[θ] = 1.3557 against the
    // prior's 1.0, matching the Z-tilted 1.3510637950 — the caller can act on a
    // refusal and cannot see a tilted ensemble.
    const src = H
      + 'theta ~ Uniform(interval(0.0, 2.0))\n'
      + 'm = normalize(weighted(x -> exp(theta * x), Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'y ~ m\n';
    const { ctx } = ctxFor(src, 64);
    await assert.rejects(() => Promise.resolve(ctx.getMeasure('y')),
      /moves with a latent[\s\S]*no per-θ expression[\s\S]*§06/);
  });

test('a joint whose component mass is not 1 gets no expression, so it is refused',
  async () => {
    // §06's independent product gives mass(joint) = ∏ mass(Mᵢ), which is only
    // closed-form when every component's is. A `truncate` component's mass is
    // M(S), which `totalMassExpr` does not express — so the product declines
    // and the θ-dependent factor in front of it has nothing to divide by.
    const src = H
      + 'theta ~ Uniform(interval(1.0, 5.0))\n'
      + 'm = normalize(weighted(theta, joint(a = Normal(mu = 0.0, sigma = 1.0), '
      + 'b = truncate(Normal(mu = 0.0, sigma = 1.0), interval(-1.0, 1.0)))))\n'
      + 'y ~ m\n';
    const { ctx } = ctxFor(src, 64);
    await assert.rejects(() => Promise.resolve(ctx.getMeasure('y')),
      /moves with a latent[\s\S]*no per-θ expression/);
  });

// =====================================================================
// UNTOUCHED — the shapes whose pooled divisor is already right
// =====================================================================

test('a θ-INDEPENDENT scalar mass factor keeps the pooled divisor bit-for-bit',
  async () => {
    // Z is one number, the pooled sum is exactly it, and this sweep must not
    // move a digit. Both values were measured on the origin/main checkout and
    // are asserted with `equal`, not a tolerance.
    const r = await joint(H
      + 'theta ~ Uniform(interval(1.0, 5.0))\n'
      + 'm = normalize(weighted(2.0, Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'y ~ m\n');
    assert.equal(r.et, 3.003582380503686, `E[θ] = ${r.et} moved`);
    assert.equal(r.ey, -0.004265923637528699, `E[y] = ${r.ey} moved`);
  });

test('a θ-INDEPENDENT weight that IS a function of the variate keeps the pooled '
  + 'divisor bit-for-bit', async () => {
  // The importance-weighted case the pooled divisor exists for:
  // normalize(weighted(x -> e^x, Normal(0,1))) is Normal(1,1), and the pooled
  // sum is the exact self-normalized estimator of its Z. Bit-for-bit against
  // origin/main, then against the closed-form mean 1.
  const r = await joint(H
    + 'theta ~ Uniform(interval(1.0, 5.0))\n'
    + 'm = normalize(weighted(x -> exp(x), Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'y ~ m\n');
  assert.equal(r.et, 3.0104291975648607, `E[θ] = ${r.et} moved`);
  assert.equal(r.ey, 1.0013316886774164, `E[y] = ${r.ey} moved`);
  assert.ok(Math.abs(r.ey - 1.0) < 5e-3, `E[y] = ${r.ey}, closed form 1.0`);
});

test('normalize(truncate(…)) with a θ-dependent mass is already exact', async () => {
  // The truncate parent never carries the mass in its weights: matTruncate
  // REJECTION-samples at each atom's own θ, so its atoms are already draws from
  // the normalized θ-slice and normalize is the identity. Pinned so a future
  // pass cannot quietly route this shape through a pooled divisor.
  //   y | θ ~ Normal(θ, 1) restricted to [−1, 1], θ ~ Uniform(0, 2).
  //   correct  E[y] = 0.2660743629   (scipy: ∫ mean of the truncated slice)
  //   tilted   E[y] = 0.2135518357, E[θ] = 0.7864481643 (Z(θ) falls with θ)
  const r = await joint(H
    + 'theta ~ Uniform(interval(0.0, 2.0))\n'
    + 'm = normalize(truncate(Normal(mu = theta, sigma = 1.0), interval(-1.0, 1.0)))\n'
    + 'y ~ m\n');
  assert.ok(Math.abs(r.et - 1.0) < 0.02, `E[θ] = ${r.et}, prior mean 1.0`);
  assert.ok(Math.abs(r.ey - 0.2660743629) < 5e-3,
    `E[y] = ${r.ey}, quadrature oracle 0.2660743629`);
  assert.ok(Math.abs(r.ey - 0.2135518357) > 2e-2,
    `E[y] = ${r.ey} is the Z-TILTED 0.2135518357`);
});

test('iid over normalize(truncate(…)) keeps the untilted θ-marginal', async () => {
  // `_resolveIidLeaf` peels the normalize and hands the truncate leaf to the
  // worker's `truncateSampleN`, so the k draws are rejection draws at atom i's
  // own θ. The tilted marginal would be 0.7864481643.
  const r = await joint(H
    + 'theta ~ Uniform(interval(0.0, 2.0))\n'
    + 'm = normalize(truncate(Normal(mu = theta, sigma = 1.0), interval(-1.0, 1.0)))\n'
    + 'y ~ iid(m, 3)\n');
  assert.deepEqual(r.m.dims, [3]);
  assert.ok(Math.abs(r.et - 1.0) < 0.02, `E[θ] = ${r.et}, prior mean 1.0`);
  assert.ok(Math.abs(r.et - 0.7864481643) > 0.1,
    `E[θ] = ${r.et} is the Z-TILTED 0.7864481643`);
});

// =====================================================================
// WILL-FLIP — pooled masses one operator away from `normalize`
// =====================================================================
// Both are asserted TIED TO THE FAILURE MODE, so each goes red the moment its
// defect is fixed. Neither is fixable by a divisor at `matNormalize`: the
// parent's atoms reach it already pooled. Tracked in
// flatppl-dev/TODO-flatppl-js.md and measure-algebra-audit.md.

test('[WILL-FLIP] a LATENT mixing weight is pooled by the superpose lift',
  async () => {
    // The canonical latent-weight mixture, §06 normalize: "To build a
    // normalized mixture distribution, use normalize(superpose(weighted(w1, M1),
    // weighted(w2, M2)))". Here Z(p) = p + (1 − p) = 1, so the MASS is constant
    // and only the per-atom mixing proportion is at stake: atom i must mix at
    // p_i, giving E[y | p] = 10(1 − p) and therefore
    //   cov(p, y) = −10 · Var(p) = −0.2551020408   (Beta(2,5): Var = 10/392)
    // matSuperpose instead SIR-lifts a parent whose per-atom weights are
    // non-uniform (materialiser.ts, the `lifted` map), which replaces p_i with
    // the parent's pooled total mass — so every atom mixes at E[p] and p comes
    // out INDEPENDENT of y. The y-marginal is right (it is linear in p), which
    // is why only the covariance sees this.
    const r = await joint(H
      + 'p ~ Beta(alpha = 2.0, beta = 5.0)\n'
      + 'q = 1.0 - p\n'
      + 'm = normalize(superpose(weighted(p, Normal(mu = 0.0, sigma = 1.0)), '
      + 'weighted(q, Normal(mu = 10.0, sigma = 1.0))))\n'
      + 'y ~ m\n', N, 'p');
    assert.ok(Math.abs(r.varT - 0.0255102041) < 2e-3,
      `Var(p) = ${r.varT}, Beta(2,5) variance 0.0255102041`);
    assert.ok(Math.abs(r.cov) < 0.02,
      `THE DEFECT IS FIXED — cov(p, y) = ${r.cov} is no longer 0. Re-tag this test `
      + 'GREEN and assert cov ≈ −10·Var(p) = −0.2551020408.');
  });

test('[WILL-FLIP] iid over a normalize with importance weights drops them',
  async () => {
    // normalize(weighted(x -> e^x, Normal(0,1))) is exactly Normal(1,1), so
    // every coordinate of `iid(m, 3)` has mean 1 — and sampling `m` DIRECTLY
    // gives 1.0013 (pinned above). Through iid the composite fallback
    // re-materialises at the inflated count and then packs the atoms into an
    // array measure, discarding the per-atom logWeights that carry the whole
    // reweighting: the result is a plain draw from the UNNORMALIZED base,
    // Normal(0,1), a full sigma out. `rand` refuses this exact weight-drop
    // loudly (materialiser.ts's matRandSample normalize gate); the iid path
    // does it silently.
    const r = await joint(H
      + 'theta ~ Uniform(interval(1.0, 5.0))\n'
      + 'm = normalize(weighted(x -> exp(x), Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'y ~ iid(m, 3)\n');
    assert.deepEqual(r.m.dims, [3]);
    let mean = 0;
    for (let i = 0; i < r.m.samples.length; i++) mean += r.m.samples[i];
    mean /= r.m.samples.length;
    assert.ok(Math.abs(mean) < 0.05,
      `THE DEFECT IS FIXED — the iid coordinates mean ${mean}, no longer the `
      + 'unnormalized base\'s 0. Re-tag this test GREEN and assert mean ≈ 1.');
  });
