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
// FIXED — iid over a normalize keeps the importance weights
// =====================================================================

// Weighted per-coordinate moments of an `iid(M, k)` output: k coordinates per
// atom, atom-major, all sharing the atom's folded importance weight.
async function iidCoords(src: string, k: number, n = N) {
  const { proc, ctx } = ctxFor(src, n);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const y = await ctx.getMeasure('y');
  const lw: number[] = y.logWeights
    ? Array.from(y.logWeights) as number[]
    : new Array(n).fill(-Math.log(n));
  const norm = logSumExp(lw);
  const mean = new Array(k).fill(0);
  const sq = new Array(k).fill(0);
  let cross = 0;
  for (let b = 0; b < n; b++) {
    const w = Math.exp(lw[b] - norm);
    for (let j = 0; j < k; j++) {
      mean[j] += w * y.samples[b * k + j];
      sq[j]   += w * y.samples[b * k + j] * y.samples[b * k + j];
    }
    cross += w * y.samples[b * k] * y.samples[b * k + 1];
  }
  return {
    m: y,
    mean,
    variance: mean.map((mu, j) => sq[j] - mu * mu),
    cov01: cross - mean[0] * mean[1],
  };
}

test('iid over a normalize FOLDS the importance weights into one per atom',
  async () => {
    // §06 `normalize`: "returns the probability measure M / Z". With
    // f(x) = e^x over Normal(0, 1) the tilt is conjugate and exact:
    //   e^x · e^(−x²/2) = e^(−(x−1)²/2) · e^(1/2)
    // so normalize(weighted(x -> exp(x), Normal(0, 1))) IS Normal(1, 1) and
    // Z = e^(1/2). §06 `iid` is "the product measure M^⊗N", so `iid(m, 3)` is
    // Normal(1, 1)^⊗3: every coordinate has mean 1 and variance 1, and distinct
    // coordinates are independent (covariance 0).
    //
    // The fallback packs the k inner draws of atom b at the BASE measure's
    // positions and the whole reweighting rides in the per-position logWeights;
    // discarding them was a draw from the UNNORMALIZED Normal(0, 1), a full
    // sigma low on every coordinate. `_foldIidBlockLogWeights` sums the k log
    // weights into the atom's weight — the product-measure ratio — so all three
    // coordinate means come back at 1 together. Reading ONE position's weight
    // instead of the block's would put coordinate 0 at 1 and the rest at 0,
    // which is why every coordinate is asserted, not just the pooled mean.
    const r = await iidCoords(H
      + 'theta ~ Uniform(interval(1.0, 5.0))\n'
      + 'm = normalize(weighted(x -> exp(x), Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'y ~ iid(m, 3)\n', 3);
    assert.deepEqual(r.m.dims, [3]);
    assert.ok(r.m.logWeights, 'the folded importance weights must survive on the '
      + 'iid output — without them the draw is from the unnormalized base');
    // n_eff at 60000 atoms is ~4.1e3: the atom weight is a product of three
    // lognormal(0, 1) factors, so it is far from uniform. Pins that the weights
    // are the real thing and not an all-equal array.
    assert.ok(r.m.n_eff < 0.5 * N,
      `three importance-weighted coordinates cannot leave n_eff ${r.m.n_eff} `
      + `near ${N}`);
    for (let j = 0; j < 3; j++) {
      assert.ok(Math.abs(r.mean[j] - 1.0) < 0.08,
        `coordinate ${j} mean = ${r.mean[j]}, Normal(1, 1) oracle 1`);
      assert.ok(Math.abs(r.variance[j] - 1.0) < 0.10,
        `coordinate ${j} variance = ${r.variance[j]}, Normal(1, 1) oracle 1`);
    }
    assert.ok(Math.abs(r.cov01) < 0.10,
      `cov(y1, y2) = ${r.cov01}, product measure oracle 0`);
  });

test('iid over a RECORD normalize with importance weights is refused, not '
  + 'sampled unnormalized', async () => {
  // The record arm of the fallback emits ONE k-row table and forces N to 1, so
  // there is no ensemble to carry the reweighting — a single atom's importance
  // weight renormalises to one. Before the fold this arm assembled the table
  // from the per-field samples and dropped the weights, which is a draw from
  // the UNNORMALIZED joint. §06 `normalize` defines the result as "the
  // probability measure M / Z", so refuse.
  for (const op of ['weighted(v -> exp(v.a)', 'logweighted(v -> v.a']) {
    const { proc, ctx } = ctxFor(H
      + 'm = normalize(' + op + ', joint(a = Normal(mu = 0.0, sigma = 1.0), '
      + 'b = Normal(mu = 3.0, sigma = 1.0))))\n'
      + 'y ~ iid(m, 3)\n', 1);
    assert.equal(proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    await assert.rejects(() => ctx.getMeasure('y'), (e: any) => {
      assert.match(e.message, /importance weights/);
      assert.match(e.message, /§06/);
      return true;
    }, op);
  }
});

test('a weighted PARAMETER ancestor does not block the fold', async () => {
  // The inner measure's location is itself an importance-weighted draw, so the
  // repeat axis tiles a value measure that carries its own weights. Those never
  // reach `innerM.logWeights` — only a measure handler introduces per-position
  // weights, and a value binding in measure position is a reified variate, which
  // the tiling exempts — so the k-block fold stays a product of k INDEPENDENT
  // coordinate weights and must not refuse here.
  //
  // No moment is asserted: the joint log weight is θ + Σⱼ yⱼ with variance 19,
  // so the ensemble is importance-degenerate at any reachable N. `n_eff` is the
  // engine's honest readout of exactly that, and the point of the assertion is
  // that it collapses instead of reporting a confident N.
  const r = await joint(H
    + 'tm = normalize(weighted(x -> exp(x), Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'theta ~ tm\n'
    + 'm = normalize(weighted(x -> exp(x), Normal(mu = theta, sigma = 1.0)))\n'
    + 'y ~ iid(m, 3)\n');
  assert.deepEqual(r.m.dims, [3]);
  assert.ok(r.m.logWeights, 'the folded weights must survive');
  assert.ok(r.m.n_eff < 0.01 * N,
    `a log-weight variance of 19 cannot leave n_eff ${r.m.n_eff} anywhere near `
    + `${N} — an n_eff of N would mean the weights were dropped`);
});

test('the folded weights leave the θ-marginal untilted', async () => {
  // The same witness read through the θ lens: Z = e^(1/2) is θ-INDEPENDENT, so
  // the folded atom weights must not move E[θ] off the Uniform(1, 5) prior mean.
  const r = await joint(H
    + 'theta ~ Uniform(interval(1.0, 5.0))\n'
    + 'm = normalize(weighted(x -> exp(x), Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'y ~ iid(m, 3)\n');
  assert.ok(Math.abs(r.et - 3.0) < 0.1, `E[θ] = ${r.et}, prior mean 3.0`);
});

// =====================================================================
// A LATENT MIXING WEIGHT — the superpose lift's pooled mass
// =====================================================================

// Beta(2,5) variance 10/392, and the mixture's covariance oracle.
const VAR_P = 0.025510204081632654;
const COV_ORACLE = -10 * VAR_P;   // −0.2551020408
// Sampling error on cov at N = 60000: 400 replicate ensembles of the closed-form
// generative model (p ~ Beta(2,5); component 0 with probability p) gave
// sd = 0.00313, so 0.012 is a ~3.8σ band.
const COV_BAND = 0.012;

test('a LATENT mixing weight tilts each atom at its own p_i', async () => {
  // The canonical latent-weight mixture, §06 normalize: "To build a normalized
  // mixture distribution, use normalize(superpose(weighted(w1, M1),
  // weighted(w2, M2)))". Here Z(p) = p + (1 − p) = 1, so the MASS is constant
  // and only the per-atom mixing proportion is at stake: atom i must mix at
  // p_i, giving E[y | p] = 10(1 − p) and therefore cov(p, y) = −10 · Var(p).
  //
  // THE DEFECT this pins. matSuperpose SIR-lifted every parent whose per-atom
  // weights were non-uniform, which replaced p_i with the parent's POOLED total
  // mass — every atom mixed at E[p] and p came out INDEPENDENT of y
  // (cov = −0.003, three sampling errors from zero). The y-marginal is right
  // either way (it is linear in p), which is why only the covariance sees this.
  const r = await joint(H
    + 'p ~ Beta(alpha = 2.0, beta = 5.0)\n'
    + 'q = 1.0 - p\n'
    + 'm = normalize(superpose(weighted(p, Normal(mu = 0.0, sigma = 1.0)), '
    + 'weighted(q, Normal(mu = 10.0, sigma = 1.0))))\n'
    + 'y ~ m\n', N, 'p');
  assert.ok(Math.abs(r.varT - VAR_P) < 2e-3,
    `Var(p) = ${r.varT}, Beta(2,5) variance ${VAR_P}`);
  assert.ok(Math.abs(r.cov - COV_ORACLE) < COV_BAND,
    `cov(p, y) = ${r.cov}, oracle −10·Var(p) = ${COV_ORACLE}`);
  assert.ok(Math.abs(r.cov) > 0.1,
    `cov(p, y) = ${r.cov} is the POOLED value — p is independent of y again`);
});

test('the mixing weight leaves the p- and y-marginals where they were',
  async () => {
    // The moments that must NOT move, and the reason this defect stayed hidden:
    // E[p] = 2/7 is the Beta prior, and E[y] = 10 · E[1 − p] = 50/7 is LINEAR in
    // p, so the pooled lift got both right. A fix validated on means alone
    // proves nothing here.
    const r = await joint(H
      + 'p ~ Beta(alpha = 2.0, beta = 5.0)\n'
      + 'q = 1.0 - p\n'
      + 'm = normalize(superpose(weighted(p, Normal(mu = 0.0, sigma = 1.0)), '
      + 'weighted(q, Normal(mu = 10.0, sigma = 1.0))))\n'
      + 'y ~ m\n', N, 'p');
    assert.ok(Math.abs(r.et - 2 / 7) < 0.005, `E[p] = ${r.et}, prior mean ${2 / 7}`);
    assert.ok(Math.abs(r.ey - 50 / 7) < 0.05, `E[y] = ${r.ey}, oracle ${50 / 7}`);
    // Z(p) = 1 at every atom, so the mixture's per-atom masses are equal and
    // `normalize` leaves the ensemble unweighted.
    assert.ok(Math.abs(r.n_eff - N) < 1e-6,
      `the atoms must stay equally weighted, n_eff ${r.n_eff}`);
  });

test('a θ-dependent MASS inside the superpose gets a per-atom divisor',
  async () => {
    // The same shape with the mixture's MASS moving too: components weighted θ
    // and 1 over θ ~ Uniform(0, 4), so Z(θ) = θ + 1 and the mixing proportion
    // on N(0,1) is θ/(θ+1). §06 normalize makes every θ-slice a probability
    // measure, so
    //   E[y] = 10 · E[1/(1 + θ)] = (10/4)·ln 5 = 4.0235947811
    // and the θ-marginal stays the prior mean 2.0. Both closed form.
    //
    // Two failure modes are excluded. Pooling atom i's slice mass into the
    // ensemble average gives the mixing proportion at E[θ], so E[y] = 10/3 —
    // measured 3.3478 before this fix. Carrying the per-atom mass without a
    // per-atom divisor at `normalize` instead leaves the residue Z(θ_i)/E[Z] on
    // the atom weights and tilts E[θ] to 2.4477. Sampling errors at this count
    // (400 replicate ensembles of the closed-form generative model) are 0.020 on
    // E[y] and 0.0047 on E[θ], so both bands below are ~4σ.
    const r = await joint(H
      + 'theta ~ Uniform(support = interval(0.0, 4.0))\n'
      + 'm = normalize(superpose(weighted(theta, Normal(mu = 0.0, sigma = 1.0)), '
      + 'weighted(1.0, Normal(mu = 10.0, sigma = 1.0))))\n'
      + 'y ~ m\n');
    assert.ok(Math.abs(r.ey - 4.0235947811) < 0.08,
      `E[y] = ${r.ey}, oracle 10·E[1/(1+θ)] = 4.0235947811`);
    assert.ok(Math.abs(r.ey - 10 / 3) > 0.3,
      `E[y] = ${r.ey} is the POOLED 10/3 — the mixing proportion sits at E[θ]`);
    assert.ok(Math.abs(r.et - 2.0) < 0.02, `E[θ] = ${r.et}, prior mean 2.0`);
    assert.ok(Math.abs(r.n_eff - N) < 1e-6,
      `the per-atom divisor must leave the atoms equally weighted, n_eff ${r.n_eff}`);
  });

test('REIFIED components inline to their leaves, so the divisor is exact',
  async () => {
    // The same mixture over `lawof(u)` / `lawof(w)`. `expandMeasureIR` inlines a
    // reified law to the leaf it reifies, so `totalMassExpr` still sees two
    // probability components and Z(p) = p + (1 − p) is exact. The covariance
    // oracle is unchanged, which is the point: reifying the components must not
    // move the mixture.
    const r = await joint(H
      + 'p ~ Beta(alpha = 2.0, beta = 5.0)\n'
      + 'q = 1.0 - p\n'
      + 'u ~ Normal(mu = 0.0, sigma = 1.0)\n'
      + 'w ~ Normal(mu = 10.0, sigma = 1.0)\n'
      + 'm = normalize(superpose(weighted(p, lawof(u)), weighted(q, lawof(w))))\n'
      + 'y ~ m\n', N, 'p');
    assert.ok(Math.abs(r.cov - COV_ORACLE) < COV_BAND,
      `cov(p, y) = ${r.cov}, oracle −10·Var(p) = ${COV_ORACLE}`);
    assert.ok(Math.abs(r.ey - 50 / 7) < 0.05, `E[y] = ${r.ey}, oracle ${50 / 7}`);
  });

// =====================================================================
// A COMPONENT'S OWN MASS — the superposition counts Σ w_i·Z_i, not Σ w_i
// =====================================================================
// A SEPARATE and OLDER defect than this file's subject: it was in the
// superposition's own mass accounting, not in any divisor, and it hit the
// density and sampling routes CONSISTENTLY — they agreed with each other and
// both disagreed with §06.
//
// §06 `truncate`: "restricts the support of measure M to the set S:
// ν(A) = M(A ∩ S). Does not normalize automatically." So
//   Z_t = totalmass(truncate(Normal(0,1), [−1,1])) = 2Φ(1) − 1 = 0.6826894921,
// and §06 `superpose`'s "ν(A) = M₁(A) + M₂(A) + …" makes the superposition's mass
// Σ_i w_i·Z_i, not Σ_i w_i.
//
// THE CAUSE. `matSuperpose` read each component's per-atom `logWeights`, which
// carry only the weighting events introduced along that component's own chain.
// `matTruncate` keeps uniform weights and records the accept rate on
// `logTotalmass` alone, so Z_t never reached the mixture. Fixed by
// `_superposeComponentWeights`, which shifts a component's weights by the part
// of `logTotalmass` they do not already carry.

const Z_TRUNC = 0.6826894921370859;          // 2Φ(1) − 1
const TRUNC = 'truncate(Normal(mu = 0.0, sigma = 1.0), interval(-1.0, 1.0))';

test('a superposition carries each component\'s own mass, on BOTH routes',
  async () => {
    // CONSTANT weights, so the density side is exact and no band enters.
    //   Z = 0.3·Z_t + 0.7 = 0.9048068476
    //   density(y) = [0.3·φ(y)·1(|y| ≤ 1) + 0.7·φ(y − 10)] / Z
    //     at y = 0.5   → −2.147877551448541
    //     at y = 10.0  → −1.1755796910613374
    //   P(component 0) = 0.3·Z_t / Z = 0.2263542193, so E[y] = 10·0.7/Z = 7.7364578067
    // Before the fix the engine scored −2.2479113375299518 and
    // −1.2756134771427479 — each exactly log Z = −0.1000337860820677 BELOW the
    // oracle, the same offset at both points, so it was a missing divisor and not
    // a wrong shape: `normalize` divided by 0.3 + 0.7 = 1. The sampler mixed at
    // 0.3 : 0.7 for the same reason and reported E[y] = 6.9938.
    //
    // The MASS-IGNORED value is excluded on both routes, so a green test cannot
    // mean "the number moved somewhere plausible". Sampling error on E[y] at this
    // count (400 replicate ensembles of the closed-form mixture) is 0.0172, so
    // the band below is ~4σ.
    const { proc, ctx } = ctxFor(H
      + 'm = normalize(superpose(weighted(0.3, ' + TRUNC + '), '
      + 'weighted(0.7, Normal(mu = 10.0, sigma = 1.0))))\n'
      + 'y ~ m\n'
      + 'lp = logdensityof(m, 0.5)\n'
      + 'lp2 = logdensityof(m, 10.0)\n', N);
    assert.equal(proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);

    const logZ = Math.log(0.3 * Z_TRUNC + 0.7);
    const scored = [];
    for (const nm of ['lp', 'lp2']) {
      const v: any = await ctx.getMeasure(nm);
      scored.push(v.samples ? v.samples[0] : v.value.data[0]);
    }
    for (const [i, want] of [[0, -2.147877551448541], [1, -1.1755796910613374]] as [number, number][]) {
      // 1e-9, not exact: Z reaches the density route through the materialised
      // superpose's `logTotalmass`, a logsumexp over the components' analytic
      // masses, so the last few digits are accumulation noise.
      assert.ok(Math.abs(scored[i] - want) < 1e-9,
        `logdensityof #${i} = ${scored[i]}, closed form ${want}`);
      assert.ok(Math.abs(scored[i] - (want + logZ)) > 1e-3,
        `logdensityof #${i} = ${scored[i]} is the MASS-IGNORED ${want + logZ}`);
    }

    const y: any = await ctx.getMeasure('y');
    let s = 0;
    for (let i = 0; i < N; i++) s += y.samples[i];
    const ey = s / N;
    assert.ok(Math.abs(ey - 7.7364578067) < 0.07,
      `E[y] = ${ey}, mixing oracle 10·0.7/Z = 7.7364578067`);
    assert.ok(Math.abs(ey - 7.0) > 0.3,
      `E[y] = ${ey} is the MASS-IGNORED 7.0 — the component's Z_t is dropped again`);
  });

test('an ALREADY-NORMALIZED component does not get its mass applied twice',
  async () => {
    // The same mixture with each component's mass divided out inside it, so
    // Z_i = 1 and Z = 0.3 + 0.7 = 1. §06 `normalize` gives the inner truncate
    // "the probability measure M / Z", so the outer divisor must be 1 and the
    // mixing proportions must be 0.3 : 0.7 — the numbers the DEFECT produced for
    // the un-normalized spelling above. Applying a component's `logTotalmass` on
    // top of weights that already carry it would divide by Z_t twice: both scores
    // would come back log Z = 0.1000337861 HIGH, at −1.7661624051 and
    // −1.1755796911, and the mixture would shift to 0.3·Z_t : 0.7 (E[y] = 7.7365).
    //   density(0.5)  = log[0.3·φ(0.5)/Z_t + 0.7·φ(−9.5)] = −1.8661961912284826
    //   density(10.0) = log[0.7·φ(0)]                     = −1.2756134771434051
    //   E[y] = 0.3·0 + 0.7·10 = 7.0   (the truncated slice is symmetric, mean 0)
    const { proc, ctx } = ctxFor(H
      + 'm = normalize(superpose(weighted(0.3, normalize(' + TRUNC + ')), '
      + 'weighted(0.7, Normal(mu = 10.0, sigma = 1.0))))\n'
      + 'y ~ m\n'
      + 'lp = logdensityof(m, 0.5)\n'
      + 'lp2 = logdensityof(m, 10.0)\n', N);
    assert.equal(proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    const want = [-1.8661961912284826, -1.2756134771434051];
    for (const [i, nm] of ['lp', 'lp2'].entries()) {
      const v: any = await ctx.getMeasure(nm);
      const got = v.samples ? v.samples[0] : v.value.data[0];
      assert.ok(Math.abs(got - want[i]) < 1e-9,
        `logdensityof #${i} = ${got}, closed form ${want[i]}`);
    }
    const y: any = await ctx.getMeasure('y');
    let s = 0;
    for (let i = 0; i < N; i++) s += y.samples[i];
    const ey = s / N;
    assert.ok(Math.abs(ey - 7.0) < 0.07, `E[y] = ${ey}, oracle 7.0`);
  });

test('a truncate component reaches the pooled-divisor fallback without refusing',
  async () => {
    // The one spelling that reaches `_perAtomMassExpr`'s "no closed-form Σ_i w_i"
    // arm with θ-DEPENDENT weights: `totalMassExpr` has no `truncate` arm, so
    // there is no expression to divide by. It must return the POOLED divisor —
    // the number this shape already had — and NOT refuse.
    //
    // The component mass now reaches the SAMPLE POSITIONS, so the unweighted
    // positions are the exact mixture: E[y] = 7.7783296397 (scipy.quad over the
    // Beta(2,5) prior; replicate sd 0.0180 at this count). The `normalize` step
    // still leaves the residue Z(p_i)/E[Z] on the atom WEIGHTS, so the weighted
    // read stays the pooled one — E[y] = 7.8549918431, cov(p, y) = −0.2106127697,
    // E[p] tilted to 0.2768126020 off the prior's 2/7 (replicate sds 0.0169 and
    // 0.00289). Both readings are asserted, and the GAP BETWEEN THEM is asserted
    // too: the two reads come off one ensemble, so their difference is far
    // sharper than either band. cov discriminates poorly here (the mass-aware
    // −0.2196964652 is only 3.1 sampling errors from the pooled value), which is
    // why E[y] carries the exclusion. This row pins the remaining θ-dependent
    // gap, recorded in flatppl-dev/measure-algebra-audit.md, so closing it flips
    // this row loudly.
    const src = H
      + 'p ~ Beta(alpha = 2.0, beta = 5.0)\n'
      + 'q = 1.0 - p\n'
      + 'm = normalize(superpose(weighted(p, ' + TRUNC + '), '
      + 'weighted(q, Normal(mu = 10.0, sigma = 1.0))))\n'
      + 'y ~ m\n';
    const r = await joint(src, N, 'p');
    assert.ok(Number.isFinite(r.ey), 'the shape must sample, not refuse');
    let s = 0;
    for (let i = 0; i < N; i++) s += r.m.samples[i];
    const pos = s / N;
    assert.ok(Math.abs(pos - 7.7783296397) < 0.08,
      `unweighted E[y] = ${pos}, mass-aware oracle 7.7783296397`);
    assert.ok(Math.abs(r.ey - 7.8549918431) < 0.07,
      `weighted E[y] = ${r.ey}, pooled-divisor value 7.8549918431`);
    assert.ok(Math.abs(r.cov - -0.2106127697) < 0.012,
      `cov(p, y) = ${r.cov}, pooled-divisor value −0.2106127697`);
    assert.ok(r.ey - pos > 0.03,
      `the weighted read ${r.ey} no longer sits ABOVE the positions ${pos} — the `
      + 'pooled residue Z(p)/E[Z] is gone, flip this row to the exact 7.7783296397');
  });
