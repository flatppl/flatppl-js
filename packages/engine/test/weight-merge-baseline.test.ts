'use strict';
// ONE EMPIRICAL BASELINE PER MERGED WEIGHT ARRAY.
//
// THE DEFECT. A measure's per-atom log-weights are a sum of weighting events,
// one of which is the `-log(N)` empirical baseline. Merging k parents' arrays
// took the UNION of their events and so kept k baselines, leaving the merged
// array's absolute normalisation short by (k-1)·log(N). A measure derived from
// k weighted ancestors then reported its mass divided by N^(k-1) — and it took
// no measure algebra to hit: two `weighted` priors and one child do it.
//
// THE ORACLE, closed form. `weighted(3.0, Normal(0,1))` and
// `weighted(4.0, Normal(0,1))` are unnormalised measures of mass 3 and 4, and
// `theta + phi` pushes them forward to mass 12 (§06 pushfwd is
// mass-preserving), so `x ~ Normal(theta + phi, 1.0)` has a law of mass 12
// whatever N is. That N-INDEPENDENCE is the point: the defect's answer moved
// with the atom count, so a sweep over N distinguishes the fix from any
// tolerance.
//
// The joint law of all three variates is closed form too and pins the DENSITY,
// which a wrong normaliser usually rides along with:
//   log 12 + logpdf(N(0,1), 0.5) + logpdf(N(0,1), 0.5) + logpdf(N(1,1), 1.0)
//   = -0.5219089498260178   (Distributions.jl)
//
// Spec §06 "Joint composition": `(M₁ ⊗ M₂)(A × B) = M₁(A)·M₂(B)`, so the two
// independent priors' masses multiply, and §06 pushfwd carries that mass
// through the sum into x's law.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const TWO_ANCESTORS = `
theta ~ weighted(3.0, Normal(0.0, 1.0))
phi ~ weighted(4.0, Normal(0.0, 1.0))
x ~ Normal(theta + phi, 1.0)
A = lawof(record(xx = x))
tm = totalmass(A)
`;

const THREE_ANCESTORS = `
theta ~ weighted(3.0, Normal(0.0, 1.0))
phi ~ weighted(4.0, Normal(0.0, 1.0))
psi ~ weighted(2.0, Normal(0.0, 1.0))
x ~ Normal(theta + phi + psi, 1.0)
A = lawof(record(xx = x))
tm = totalmass(A)
`;

test('weight merge: two weighted ancestors give mass 12 at every N', async () => {
  // Reported 12/N before: 0.023437499999999993 at N = 512 and
  // 0.002929687500000001 at N = 4096.
  for (const N of [512, 4096, 16384]) {
    const { ctx } = ctxFor(TWO_ANCESTORS, N);
    const tm = await ctx.getMeasure('tm');
    assert.ok(Math.abs(tm.samples[0] - 12) < 1e-9,
      `N = ${N}: totalmass ${tm.samples[0]}, expected 12`);
  }
});

test('weight merge: three weighted ancestors give mass 24, not 24/N²', async () => {
  // Reported 24/N² before: 1.4305114746093742e-6 at N = 4096.
  for (const N of [512, 4096]) {
    const { ctx } = ctxFor(THREE_ANCESTORS, N);
    const tm = await ctx.getMeasure('tm');
    assert.ok(Math.abs(tm.samples[0] - 24) < 1e-8,
      `N = ${N}: totalmass ${tm.samples[0]}, expected 24`);
  }
});

test('weight merge: ONE ancestor was already right and stays right', async () => {
  for (const N of [512, 4096]) {
    const { ctx } = ctxFor(`
theta ~ weighted(3.0, Normal(0.0, 1.0))
x ~ Normal(theta, 1.0)
A = lawof(record(xx = x))
tm = totalmass(A)
`, N);
    const tm = await ctx.getMeasure('tm');
    assert.ok(Math.abs(tm.samples[0] - 3) < 1e-11,
      `N = ${N}: totalmass ${tm.samples[0]}, expected 3`);
  }
});

test('weight merge: the joint law scores its closed-form density', async () => {
  // The retained joint over (theta, phi, x) needs no marginal integral, so it
  // is exact: mass 12 and the density oracle above.
  const { ctx } = ctxFor(`
theta ~ weighted(3.0, Normal(0.0, 1.0))
phi ~ weighted(4.0, Normal(0.0, 1.0))
x ~ Normal(theta + phi, 1.0)
A = lawof(record(t = theta, p = phi, xx = x))
lp = logdensityof(A, record(t = 0.5, p = 0.5, xx = 1.0))
tm = totalmass(A)
`, 2048);
  const want = -0.5219089498260178;
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - want) < 1e-12,
    `joint law density: got ${lp.samples[0]}, expected ${want}`);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 12) < 1e-9,
    `joint law totalmass: got ${tm.samples[0]}, expected 12`);
});

test('weight merge: a probability measure over two tilted parents has mass 1', async () => {
  // `a` and `b` are `normalize(weighted(…))`, so both are probability measures
  // and so is y's law. With two baselines it read 1/N — the same defect on a
  // shape whose exact mass needs no arithmetic at all. The residual here is the
  // tilts' own importance-sampling error, not the baseline.
  const { ctx } = ctxFor('flatppl_compat = "0.1"\n'
    + 't1 = x -> exp(x)\n'
    + 't2 = x -> exp(0.5 * x)\n'
    + 'ma = normalize(weighted(t1, Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'mb = normalize(weighted(t2, Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'a ~ ma\n'
    + 'b ~ mb\n'
    + 'sb = 1.0 + 0.0 * b\n'
    + 'y ~ Normal(mu = a, sigma = sb)\n'
    + 'Ly = lawof(y)\n'
    + 'tmy = totalmass(Ly)\n', 16384);
  const tm = await ctx.getMeasure('tmy');
  assert.ok(Math.abs(tm.samples[0] - 1) < 0.02,
    `probability law over two tilted parents: got ${tm.samples[0]}, expected 1`);
});
