'use strict';
// THE MASS OF THE TWO SELECT SHAPES #293 LEFT UNSWEPT.
//
// (a) `ksuperpose(kernel, weights)` with explicit, unequal, non-unit weights.
//     §06: "applied to a parameter family it yields ν = Σᵢ wᵢ κ(θᵢ) … the
//     weights must be non-negative but need not sum to one", so its mass is
//     Σ wᵢ. Measured correct in all four shapes before this change — bare,
//     under `normalize`, as a chain base, and as a field of a dependent record
//     law — so these are PINNING tests, not repairs.
//
// (b) A selector-driven select over branches of DIFFERENT mass. §06 makes
//     `ifelse(c, A, B)` with `c ~ Bernoulli(p)` the marginal over c, so the
//     mass is p·mass(A) + (1−p)·mass(B), which the engine now RECORDS in closed
//     form (see `certified-mass.test.ts`). The gather read the branches as
//     parents of one PRODUCT, multiplying every branch's weight into every
//     atom: `totalmass` reported 5.9999999999999964 — the product 2·3 —
//     against the exact 0.25·2 + 0.75·3 = 2.75. A constant weight cannot
//     re-weight anything, so the weighted mean collapsed to the UNWEIGHTED
//     gather mean, 3.74338 against the exact 4.09091, and the variance read
//     5.69997 against 4.71901. The DENSITY was right throughout, which is the
//     signature this thread has seen five times now: mass and density
//     disagreeing on one node.
//
// ORACLES from Distributions.jl. For (a) the mixture at 0.5 is
// 2·pdf(N(0,1),0.5) + 3·pdf(N(5,1),0.5), whose log is -0.35072325506877616.
// For (b) the normalised mixture weights are 0.5/2.75 and 2.25/2.75, giving
// mean 4.090909090909091 and variance 4.719008264462811.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const KSUP_DENS = -0.35072325506877616;        // log(2·φ(0.5) + 3·φ₅(0.5))
const KSUP_NORM_DENS = -1.9601611675028767;    // the same over 5
const KSUP_CHAIN_DENS = -1.2696617882734489;   // × pdf(N(0.5,1), 0.5)
const SEL_DENS = -1.7368814349470252;          // log(0.25·2·φ(0.5) + 0.75·3·φ₅(0.5))

const KSUP = `
w = [2.0, 3.0]
mus = [0.0, 5.0]
sigs = [1.0, 1.0]
`;

// =====================================================================
// (a) ksuperpose — weights that need not sum to one
// =====================================================================

test('ksuperpose mass: explicit unequal weights give their sum', async () => {
  const { ctx } = ctxFor(KSUP + `
A = ksuperpose(Normal, w)(mu = mus, sigma = sigs)
tm = totalmass(A)
lp = logdensityof(A, 0.5)
`, 8192);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 5) < 1e-9,
    `ksuperpose totalmass: got ${tm.samples[0]}, expected 5`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - KSUP_DENS) < 1e-12,
    `ksuperpose density: got ${lp.samples[0]}, expected ${KSUP_DENS}`);
});

test('ksuperpose mass: under normalize it is 1', async () => {
  const { ctx } = ctxFor(KSUP + `
A = normalize(ksuperpose(Normal, w)(mu = mus, sigma = sigs))
tm = totalmass(A)
lp = logdensityof(A, 0.5)
`, 8192);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 1) < 1e-11,
    `normalized ksuperpose totalmass: got ${tm.samples[0]}, expected 1`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - KSUP_NORM_DENS) < 1e-12,
    `normalized ksuperpose density: got ${lp.samples[0]}, expected ${KSUP_NORM_DENS}`);
});

test('ksuperpose mass: as a chain base it carries 5 into the chain', async () => {
  const { ctx } = ctxFor(KSUP + `
S = ksuperpose(Normal, w)(mu = mus, sigma = sigs)
A = jointchain(aa = S, bb = fn(Normal(_, 1.0)))
tm = totalmass(A)
lp = logdensityof(A, record(aa = 0.5, bb = 0.5))
`, 8192);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 5) < 1e-9,
    `ksuperpose chain-base totalmass: got ${tm.samples[0]}, expected 5`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - KSUP_CHAIN_DENS) < 1e-12,
    `ksuperpose chain-base density: got ${lp.samples[0]}, expected ${KSUP_CHAIN_DENS}`);
});

test('ksuperpose mass: as a dependent record-law field it is counted once', async () => {
  const { ctx } = ctxFor(KSUP + `
theta ~ ksuperpose(Normal, w)(mu = mus, sigma = sigs)
x ~ Normal(theta, 1.0)
A = lawof(record(t = theta, xx = x))
tm = totalmass(A)
lp = logdensityof(A, record(t = 0.5, xx = 0.5))
`, 8192);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 5) < 1e-9,
    `ksuperpose record-law totalmass: got ${tm.samples[0]}, expected 5`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - KSUP_CHAIN_DENS) < 1e-12,
    `ksuperpose record-law density: got ${lp.samples[0]}, expected ${KSUP_CHAIN_DENS}`);
});

// =====================================================================
// (b) a selector-driven select over branches of DIFFERENT mass
// =====================================================================

const UNEVEN = `
A = weighted(2.0, Normal(mu = 0.0, sigma = 1.0))
B = weighted(3.0, Normal(mu = 5.0, sigma = 1.0))
c = draw(Bernoulli(p = 0.25))
M = ifelse(c, A, B)
`;

test('select mass: unequal branch masses give the selector-weighted sum', async () => {
  // The engine reported the PRODUCT 6 before this rule landed. The mass was
  // then an importance estimate for a while, and is now the recorded closed
  // form Σ pᵢ Zᵢ — hence `assert.equal` rather than the band this test first
  // carried. The `certified-mass` file sweeps N to pin that exactness.
  const { ctx } = ctxFor(UNEVEN + `
tm = totalmass(M)
lp = logdensityof(M, 0.5)
`, 32768);
  const tm = await ctx.getMeasure('tm');
  assert.equal(tm.samples[0], 2.75,
    `uneven-branch select totalmass: got ${tm.samples[0]}, expected exactly 2.75`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - SEL_DENS) < 1e-12,
    `uneven-branch select density: got ${lp.samples[0]}, expected ${SEL_DENS}`);
});

test('select mass: each atom carries its OWN branch weight', async () => {
  // The sharpest witness. With every atom carrying the constant log 6 the
  // weights could not re-weight anything and the mean collapsed to the
  // unweighted gather mean 3.74338; the exact mixture mean is 4.09091 and the
  // variance 4.71901. Bands are ~4 standard errors at this N.
  const N = 200000;
  const { ctx } = ctxFor(UNEVEN, N);
  const M = await ctx.getMeasure('M');
  assert.ok(M.logWeights, 'the gather must carry per-atom weights');
  let mx = -Infinity;
  for (let i = 0; i < N; i++) if (M.logWeights[i] > mx) mx = M.logWeights[i];
  let tot = 0; let mean = 0;
  const w = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    w[i] = Math.exp(M.logWeights[i] - mx); tot += w[i]; mean += w[i] * M.samples[i];
  }
  mean /= tot;
  let variance = 0;
  for (let i = 0; i < N; i++) variance += w[i] * (M.samples[i] - mean) ** 2;
  variance /= tot;
  assert.ok(Math.abs(mean - 4.090909090909091) < 0.05,
    `uneven-branch select mean: got ${mean}, expected 4.090909090909091 `
    + '(3.74338 = every atom on one constant weight)');
  assert.ok(Math.abs(variance - 4.719008264462811) < 0.1,
    `uneven-branch select variance: got ${variance}, expected 4.719008264462811`);
});

test('select mass: EQUAL branch masses keep the exact path', async () => {
  // Two probability branches: the masses agree, so the parents' weight array
  // passes through untouched and the mixture's mass is exactly 1 rather than an
  // estimate. This is the case the per-atom gather must NOT take over.
  const { ctx } = ctxFor(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 1.0)
c = draw(Bernoulli(p = 0.25))
M = ifelse(c, A, B)
tm = totalmass(M)
`, 8192);
  const tm = await ctx.getMeasure('tm');
  assert.equal(tm.samples[0], 1,
    `equal-mass select totalmass: got ${tm.samples[0]}, expected exactly 1`);
});
