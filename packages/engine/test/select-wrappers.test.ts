'use strict';
// THE MEASURE WRAPPERS OVER A CERTIFIED SELECT, AND THE CHAIN BASE.
//
// The certified-mass work recorded a selector mixture's exact `Σᵢ pᵢ Zᵢ`. This
// file sweeps what sits ON TOP of such a select — `weighted`, `truncate`,
// `pushfwd` — and the `jointchain` base that was fenced off while its mass was
// wrong.
//
// THE CHAIN DEFECT, and it was a bridge bug rather than a mass bug.
// `_bridgeDerivation`'s select arm PEELED each branch's constant weight into
// `synthWeights`, which matSelect ignores whenever an external `selectorName`
// decides the branch — so the branches arrived MASSLESS, the gather took its
// equal-mass path, and `jointchain(aa = S, bb = fn(Normal(_, 1.0)))` answered
// totalmass 1 against the exact 2.75. Keeping the branch whole then needed
// matSelect to materialise a COMPOSITE inline branch through the measure path,
// since the worker samples leaf kernels only and reported
// `'logweighted' is not a known distribution`.
//
// ORACLES from Distributions.jl, with `A = weighted(2.0, Normal(0,1))`,
// `B = weighted(3.0, Normal(5,1))`, `c ~ Bernoulli(0.25)` (mass 2.75, density
// at 0.5 = -1.7368814349470252):
//   weighted(2.0, S)                 mass 5.5,   density log 2 + that
//   truncate(S, interval(-1,1))      mass 0.5·(Φ(1)-Φ(-1)) + 2.25·(Φ(-4)-Φ(-6))
//                                          = 0.34141600414284534, density
//                                          unchanged inside the set
//   pushfwd(exp bijection, S)        mass 2.75 (§06 pushfwd preserves mass),
//                                          density sel(log y) - log y
//   jointchain(aa = S, bb = fn(N(_,1)))  mass 2.75, density -2.655819968151698

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const UNEVEN = `
A = weighted(2.0, Normal(mu = 0.0, sigma = 1.0))
B = weighted(3.0, Normal(mu = 5.0, sigma = 1.0))
c = draw(Bernoulli(p = 0.25))
S = ifelse(c, A, B)
`;

test('select wrappers: weighted over a certified select is exact', async () => {
  const { ctx } = ctxFor(UNEVEN + `
M = weighted(2.0, S)
tm = totalmass(M)
lp = logdensityof(M, 0.5)
`, 32768);
  const tm = await ctx.getMeasure('tm');
  assert.equal(tm.samples[0], 5.5,
    `weighted-over-select totalmass: got ${tm.samples[0]}, expected 5.5`);
  const lp = await ctx.getMeasure('lp');
  const want = -1.04373425438708;
  assert.ok(Math.abs(lp.samples[0] - want) < 1e-12,
    `weighted-over-select density: got ${lp.samples[0]}, expected ${want}`);
});

test('select wrappers: pushfwd over a certified select preserves the mass', async () => {
  // §06's pushfwd is mass-preserving whatever the forward map, so the exact
  // 2.75 survives the change of variables and the density picks up -log y.
  const { ctx } = ctxFor(UNEVEN + `
eb = bijection(fn(exp(_)), fn(log(_)), fn(_))
M = pushfwd(eb, S)
tm = totalmass(M)
lp = logdensityof(M, 1.5)
`, 32768);
  const tm = await ctx.getMeasure('tm');
  assert.equal(tm.samples[0], 2.75,
    `pushfwd-over-select totalmass: got ${tm.samples[0]}, expected 2.75`);
  const lp = await ctx.getMeasure('lp');
  const want = -2.0996244602009067;
  assert.ok(Math.abs(lp.samples[0] - want) < 1e-12,
    `pushfwd-over-select density: got ${lp.samples[0]}, expected ${want}`);
});

test('select wrappers: truncate keeps its density exact and its mass an estimate', async () => {
  // A truncation's mass is an accept rate, and no certificate exists for the
  // RESTRICTED mass of a mixture — `truncateMassLit` needs a constant-parameter
  // leaf. So the mass is an unbiased 1/sqrt(N) estimate, which the band admits
  // and which converges: 0.359025737 at N = 4096, 0.345312031 at 32768,
  // 0.342248458 at 262144. The DENSITY inside the set is exact.
  const { ctx } = ctxFor(UNEVEN + `
M = truncate(S, interval(-1.0, 1.0))
tm = totalmass(M)
lp = logdensityof(M, 0.5)
`, 32768);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 0.34141600414284534) < 0.02,
    `truncate-over-select totalmass: got ${tm.samples[0]}, expected about 0.34141600414284534`);
  const lp = await ctx.getMeasure('lp');
  const want = -1.7368814349470252;
  assert.ok(Math.abs(lp.samples[0] - want) < 1e-12,
    `truncate-over-select density: got ${lp.samples[0]}, expected ${want}`);
});

test('select wrappers: a chain over a certified select is exact at every N', async () => {
  // Answered totalmass 1 before, at any N. The N sweep is the check: the exact
  // 2.75 does not move, and the wrong 1 did not either, so only equality
  // separates them from an estimate.
  for (const N of [4096, 32768]) {
    const { ctx } = ctxFor(UNEVEN + `
M = jointchain(aa = S, bb = fn(Normal(_, 1.0)))
tm = totalmass(M)
lp = logdensityof(M, record(aa = 0.5, bb = 0.5))
`, N);
    const tm = await ctx.getMeasure('tm');
    assert.equal(tm.samples[0], 2.75,
      `N = ${N}: chain-over-select totalmass ${tm.samples[0]}, expected 2.75`);
    const lp = await ctx.getMeasure('lp');
    const want = -2.655819968151698;
    assert.ok(Math.abs(lp.samples[0] - want) < 1e-12,
      `N = ${N}: chain-over-select density ${lp.samples[0]}, expected ${want}`);
  }
});

test('select wrappers: a constant-weight mixture with NO selector still peels', async () => {
  // The peel is right where the weight IS the selector. Keeping it gated on the
  // absence of a `selectorName` is what preserves that: `superpose` of weighted
  // probability measures is additive and its mass is the weight sum, 3.
  const { ctx } = ctxFor(`
M = superpose(weighted(1.5, Normal(mu = 0.0, sigma = 1.0)),
              weighted(1.5, Normal(mu = 5.0, sigma = 1.0)))
tm = totalmass(M)
`, 8192);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 3) < 1e-11,
    `constant-weight superposition totalmass: got ${tm.samples[0]}, expected 3`);
});
