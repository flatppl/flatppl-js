'use strict';
// A CERTIFIED CLOSED-FORM MASS, DISTINGUISHED FROM AN ATOM-DERIVED RESIDUE.
//
// §06 makes a selector mixture the marginal over its selector, so its mass is
// Σᵢ pᵢ Zᵢ — a CLOSED FORM whenever the branch probabilities are constants.
// The gather realises the selector, so the atoms' own `logSumExp` is only an
// importance estimate of that same number, and it moved with N: measured on the
// parent commit, 2.7578125000000075 at N = 512, 2.7414550781249636 at 4096 and
// 2.7467956542971845 at 32768, against the exact 2.75.
//
// matSelect now records the closed form and marks the gathered weighting event
// with it, and every reader that derives one measure's mass from another
// prefers a certified event's share over the atoms' sum. So the product rule
// states two terms explicitly rather than inferring one:
//
//   - ACCOUNTED: what a factor's own weighting events account for. A certified
//     event states that exactly; without one it is the atoms' `logSumExp`.
//   - RESIDUE: the recorded mass beyond what the events account for — zero in
//     the certified case, and otherwise the genuine atom-derived remainder,
//     which is where a `truncate` keeps its accept rate.
//
// Reading the residue from the gap between recorded mass and atom sum, as the
// rule used to, invents one out of the estimator's error the moment a mass is
// certified, and a dependent product then contributes the certificate through
// one factor and the estimate through its descendant.
//
// ORACLES from Distributions.jl. With `A = weighted(2.0, Normal(0,1))`,
// `B = weighted(3.0, Normal(5,1))`, `c ~ Bernoulli(0.25)`: mass 2.75, density at
// 0.5 = -1.7368814349470252, and the dependent record law over (theta, x) with
// `x ~ Normal(theta, 1.0)` has mass 2.75 and density -2.655819968151698 at
// (0.5, 0.5). The truncate control's mass is cdf(N(),1) - cdf(N(),-1) =
// 0.6826894921370861.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const SELECT_DENS = -1.7368814349470252;
const PRODUCT_DENS = -2.655819968151698;
const PHI_BAND = 0.6826894921370861;

const UNEVEN = `
A = weighted(2.0, Normal(mu = 0.0, sigma = 1.0))
B = weighted(3.0, Normal(mu = 5.0, sigma = 1.0))
c = draw(Bernoulli(p = 0.25))
S = ifelse(c, A, B)
`;

test('certified mass: the select records 2.75 at every N, not an estimate', async () => {
  // The N sweep is the point: the estimate moved with N and no tolerance
  // admits all three rows at once.
  for (const N of [512, 4096, 32768]) {
    const { ctx } = ctxFor(UNEVEN + `
M = S
tm = totalmass(M)
`, N);
    const tm = await ctx.getMeasure('tm');
    assert.equal(tm.samples[0], 2.75,
      `N = ${N}: totalmass ${tm.samples[0]}, expected exactly 2.75`);
  }
});

test('certified mass: the density is unchanged by certifying the mass', async () => {
  const { ctx } = ctxFor(UNEVEN + `
M = S
lp = logdensityof(M, 0.5)
`, 32768);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - SELECT_DENS) < 1e-12,
    `select density: got ${lp.samples[0]}, expected ${SELECT_DENS}`);
});

test('certified mass: the per-atom weights are unchanged', async () => {
  // The certificate is recorded mass, not a re-weighting: the atoms still carry
  // their own branch's weight, so the weighted mean is the mixture's
  // 4.090909090909091 rather than the unweighted gather mean 3.74338.
  const N = 200000;
  const { ctx } = ctxFor(UNEVEN + 'M = S\n', N);
  const M = await ctx.getMeasure('M');
  assert.ok(M.logWeights, 'the gather still carries per-atom weights');
  let mx = -Infinity;
  for (let i = 0; i < N; i++) if (M.logWeights[i] > mx) mx = M.logWeights[i];
  let tot = 0; let mean = 0;
  for (let i = 0; i < N; i++) {
    const wi = Math.exp(M.logWeights[i] - mx); tot += wi; mean += wi * M.samples[i];
  }
  mean /= tot;
  assert.ok(Math.abs(mean - 4.090909090909091) < 0.05,
    `weighted mean: got ${mean}, expected 4.090909090909091`);
});

test('certified mass: as a DIRECT factor of an independent product', async () => {
  // Was 2.7414550781249636 at N = 4096 and 2.7467956542971845 at 32768.
  for (const N of [4096, 32768]) {
    const { ctx } = ctxFor(UNEVEN + `
M = joint(aa = S, bb = Normal(mu = 0.0, sigma = 1.0))
tm = totalmass(M)
`, N);
    const tm = await ctx.getMeasure('tm');
    assert.equal(tm.samples[0], 2.75,
      `N = ${N}: independent product totalmass ${tm.samples[0]}, expected 2.75`);
  }
});

test('certified mass: inside a DEPENDENT product it is not double-counted', async () => {
  // The witness the ruling asks for. `x` is drawn AT theta, so it inherits
  // theta's certified event; one factor must contribute the certificate and its
  // descendant nothing, or the product drifts by the estimator's error. Was
  // 2.7414550781249636 / 2.7467956542971845 against the exact 2.75.
  for (const N of [4096, 32768]) {
    const { ctx } = ctxFor(UNEVEN + `
theta ~ S
x ~ Normal(theta, 1.0)
M = lawof(record(t = theta, xx = x))
tm = totalmass(M)
lp = logdensityof(M, record(t = 0.5, xx = 0.5))
`, N);
    const tm = await ctx.getMeasure('tm');
    assert.equal(tm.samples[0], 2.75,
      `N = ${N}: dependent product totalmass ${tm.samples[0]}, expected 2.75`);
    const lp = await ctx.getMeasure('lp');
    assert.ok(Math.abs(lp.samples[0] - PRODUCT_DENS) < 1e-12,
      `N = ${N}: dependent product density ${lp.samples[0]}, expected ${PRODUCT_DENS}`);
  }
});

test('certified mass: a constant wrapper multiplies the certificate exactly', async () => {
  // Was 5.482910156249927 / 5.49359130859437 against the exact 2 * 2.75.
  for (const N of [4096, 32768]) {
    const { ctx } = ctxFor(UNEVEN + `
M = weighted(2.0, S)
tm = totalmass(M)
`, N);
    const tm = await ctx.getMeasure('tm');
    assert.equal(tm.samples[0], 5.5,
      `N = ${N}: weighted-over-select totalmass ${tm.samples[0]}, expected 5.5`);
  }
});

test('certified mass: normalize over the certified select stays exactly 1', async () => {
  const { ctx } = ctxFor(UNEVEN + `
M = normalize(S)
tm = totalmass(M)
`, 8192);
  const tm = await ctx.getMeasure('tm');
  assert.equal(tm.samples[0], 1,
    `normalize over a certified select: got ${tm.samples[0]}, expected exactly 1`);
});

test('certified mass: a truncate accept rate is still an atom-derived residue', async () => {
  // The one legitimate residue, and the reason the two terms are named rather
  // than merged. A truncation keeps uniform weights and records its accept rate
  // on the mass, so nothing certifies it and the product must still pick it up.
  const { ctx } = ctxFor(`
M = jointchain(aa = truncate(Normal(mu = 0.0, sigma = 1.0), interval(-1.0, 1.0)),
               bb = fn(Normal(_, 1.0)))
tm = totalmass(M)
`, 8192);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - PHI_BAND) < 1e-11,
    `truncate residue through a chain: got ${tm.samples[0]}, expected ${PHI_BAND}`);
});

test('certified mass: a NON-closed-form selector keeps the estimate', async () => {
  // `u > 0` gives per-atom indicator branch weights, not a closed-form p, so
  // there is nothing to certify and the atom-derived estimate stands. The
  // certificate is claimed only where it exists: exact 0.5*2 + 0.5*3 = 2.5,
  // reached as an estimate, so this asserts a band and NOT equality.
  const N = 32768;
  const { ctx } = ctxFor(`
A = weighted(2.0, Normal(mu = 0.0, sigma = 1.0))
B = weighted(3.0, Normal(mu = 5.0, sigma = 1.0))
u ~ Normal(mu = 0.0, sigma = 1.0)
c = u > 0.0
S = ifelse(c, A, B)
M = S
tm = totalmass(M)
`, N);
  const tm = await ctx.getMeasure('tm');
  assert.notEqual(tm.samples[0], 2.5,
    'a non-closed-form selector must not claim a certificate');
  assert.ok(Math.abs(tm.samples[0] - 2.5) < 0.05,
    `indicator-selector totalmass: got ${tm.samples[0]}, expected about 2.5`);
});

test('certified mass: a certified event beside an UNCERTIFIED one claims nothing', async () => {
  // A function-of-the-variate weight over the certified select leaves the array
  // carrying the certificate AND a per-atom event nothing certifies, so the
  // events cannot account for the array on their own and no certificate is
  // claimed. This pins the DECLINE, not the mass: that shape's mass is a
  // separate pre-existing defect (7.515575945377148 before this change,
  // 7.539001464843649 after, against the exact 2.75 for a weight that is
  // identically 1), carded under the variate-weight path.
  const { ctx } = ctxFor(`
A = weighted(2.0, Normal(mu = 0.0, sigma = 1.0))
B = weighted(3.0, Normal(mu = 5.0, sigma = 1.0))
c = draw(Bernoulli(p = 0.25))
S = ifelse(c, A, B)
M = weighted(fn(exp(0.0 * _)), S)
`, 4096);
  const M = await ctx.getMeasure('M');
  const lineage = require('../weight-lineage.ts');
  const events = lineage.lineageOf(M.logWeights).events;
  assert.ok(events.some((e: any) => e.mass != null),
    'the certified select event survives the wrapper');
  assert.ok(events.some((e: any) => e.values && e.mass == null),
    'the variate weight adds an uncertified per-atom event');
  // And the rule declines to certify that array: asked directly, it says it
  // cannot account for the mass from the events alone. The recorded mass on
  // this path is composed by `matWeighted`'s function-weight arm as
  // `massOf(parent) + logSumExp(w)`, not read off the array, so it is not the
  // thing to assert here — and its value is the carded defect above.
  const { _massFromEventsForTest } = require('../materialiser.ts');
  assert.equal(_massFromEventsForTest(M.logWeights), null,
    'a certified event mixed with an uncertified per-atom one certifies nothing');
});

test('certified mass: a constant event beside the certificate splits exactly', async () => {
  // `theta ~ weighted(2.0, S)` leaves theta's array carrying the baseline, the
  // certified select event AND a constant log 2, and a descendant drawn at
  // theta inherits all three. The events then account for the whole mass —
  // certificate plus offset — so the dependent product is exactly 2 * 2.75 at
  // any N rather than the estimate 5.482910156249927 / 5.49359130859437.
  for (const N of [4096, 32768]) {
    const { ctx } = ctxFor(`
A = weighted(2.0, Normal(mu = 0.0, sigma = 1.0))
B = weighted(3.0, Normal(mu = 5.0, sigma = 1.0))
c = draw(Bernoulli(p = 0.25))
S = ifelse(c, A, B)
theta ~ weighted(2.0, S)
x ~ Normal(theta, 1.0)
M = lawof(record(t = theta, xx = x))
tm = totalmass(M)
`, N);
    const tm = await ctx.getMeasure('tm');
    assert.equal(tm.samples[0], 5.5,
      `N = ${N}: scaled certified product totalmass ${tm.samples[0]}, expected 5.5`);
  }
});
