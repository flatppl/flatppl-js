'use strict';
// AN INLINE SUPERPOSITION IN A LOWERED BODY IS MEASURE ADDITION, NOT A MIXTURE.
//
// THE DEFECT. A `jointchain` materialises through its `clm` body, where the
// base appears as the INLINE IR an expanded `superpose` derivation takes: a
// `select` with no selector and no explicit logweights. The bridge routed that
// to `matSelect`, which synthesises a selector over the NORMALISED branch
// weights — the mixture reading — and so divided the superposition's total mass
// away. The named `superpose` derivation, reached when the same measure is a
// field of an independent `joint`, was correct all along, so the two spellings
// of one measure disagreed.
//
// §06 makes `superpose` measure ADDITION: "ν(A) = M₁(A) + M₂(A) + …", so
// `superpose(weighted(1.5, ·), weighted(1.5, ·))` has mass 3. §06's normalized
// MIXTURE is `normalize(superpose(…))`, and that `normalize` is its own
// derivation — so the bare shape is additive and the wrapper still divides,
// which is what the second witness checks.
//
// `normalize-mass.totalMassExpr` already read this exact shape as an additive
// superpose when computing a mass ("superpose / logweights-null select"), so
// before the fix sampling and mass disagreed on one node.
//
// ORACLES, closed form and from Distributions.jl. The mixture density at 0.5 is
// 1.5·pdf(N(0,1),0.5) + 1.5·pdf(N(5,1),0.5) = 0.5281219657581097, so the chain
// density at (0.5, 0.5) is log of that plus logpdf(N(0.5,1), 0.5). Every mass
// assertion below is paired with the density at the same point, because a lost
// normaliser shows up in both or in neither.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const N = 8192;

const SUPERPOSE_CHAIN_DENSITY = -1.5573665594019641;   // log 3-mass mixture × N(0.5,1)
const NORMALIZED_CHAIN_DENSITY = -2.655978848070074;   // the same minus log 3
const ASYM_CHAIN_DENSITY = -1.2696617882734489;        // weights 2 and 3, mass 5

test('chain body: a superposition base keeps its additive mass', async () => {
  // Reported totalmass 1 against the exact 3: matSelect normalised the branch
  // weights, so the superposition's total was divided away.
  const { ctx } = ctxFor(`
S = superpose(weighted(1.5, Normal(0.0, 1.0)), weighted(1.5, Normal(5.0, 1.0)))
A = jointchain(aa = S, bb = fn(Normal(_, 1.0)))
tm = totalmass(A)
lp = logdensityof(A, record(aa = 0.5, bb = 0.5))
`, N);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 3) < 1e-11,
    `superposition-base chain totalmass: got ${tm.samples[0]}, expected 3`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - SUPERPOSE_CHAIN_DENSITY) < 1e-12,
    `superposition-base chain density: got ${lp.samples[0]}, `
    + `expected ${SUPERPOSE_CHAIN_DENSITY}`);
});

test('chain body: a NORMALIZED superposition base has mass 1', async () => {
  // Reported 0.33333333333333354 against the exact 1: the normalize's -log 3
  // landed and the superposition's +log 3 never did.
  const { ctx } = ctxFor(`
S = normalize(superpose(weighted(1.5, Normal(0.0, 1.0)), weighted(1.5, Normal(5.0, 1.0))))
A = jointchain(aa = S, bb = fn(Normal(_, 1.0)))
tm = totalmass(A)
lp = logdensityof(A, record(aa = 0.5, bb = 0.5))
`, N);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 1) < 1e-11,
    `normalized superposition-base chain totalmass: got ${tm.samples[0]}, expected 1`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - NORMALIZED_CHAIN_DENSITY) < 1e-12,
    `normalized superposition-base chain density: got ${lp.samples[0]}, `
    + `expected ${NORMALIZED_CHAIN_DENSITY}`);
});

test('chain body: asymmetric branch weights give their sum', async () => {
  const { ctx } = ctxFor(`
S = superpose(weighted(2.0, Normal(0.0, 1.0)), weighted(3.0, Normal(5.0, 1.0)))
A = jointchain(aa = S, bb = fn(Normal(_, 1.0)))
tm = totalmass(A)
lp = logdensityof(A, record(aa = 0.5, bb = 0.5))
`, N);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 5) < 1e-10,
    `asymmetric superposition-base chain totalmass: got ${tm.samples[0]}, expected 5`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - ASYM_CHAIN_DENSITY) < 1e-12,
    `asymmetric superposition-base chain density: got ${lp.samples[0]}, `
    + `expected ${ASYM_CHAIN_DENSITY}`);
});

test('chain body: the named superpose spelling agreed all along and still does', async () => {
  // The same measure as a field of an INDEPENDENT joint reaches matRecord and
  // so the named `superpose` derivation, which was never wrong. Both spellings
  // must now report 3.
  const { ctx } = ctxFor(`
S = superpose(weighted(1.5, Normal(0.0, 1.0)), weighted(1.5, Normal(5.0, 1.0)))
A = joint(aa = S, bb = Normal(0.0, 1.0))
tm = totalmass(A)
lp = logdensityof(A, record(aa = 0.5, bb = 0.5))
`, N);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 3) < 1e-11,
    `named-superpose joint totalmass: got ${tm.samples[0]}, expected 3`);
  const lp = await ctx.getMeasure('lp');
  // The independent joint's second field is a standard normal at 0.5, so the
  // density is the mixture's log plus logpdf(N(0,1), 0.5).
  const want = Math.log(0.5281219657581097) + -1.0439385332046727;
  assert.ok(Math.abs(lp.samples[0] - want) < 1e-12,
    `named-superpose joint density: got ${lp.samples[0]}, expected ${want}`);
});

test('chain body: a probability superposition base stays at mass 1', async () => {
  // Equal weights summing to one: additive and mixture readings agree here, so
  // this pins that the fix moved nothing where nothing was wrong.
  const { ctx } = ctxFor(`
S = superpose(weighted(0.5, Normal(0.0, 1.0)), weighted(0.5, Normal(5.0, 1.0)))
A = jointchain(aa = S, bb = fn(Normal(_, 1.0)))
tm = totalmass(A)
lp = logdensityof(A, record(aa = 0.5, bb = 0.5))
`, N);
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 1) < 1e-11,
    `probability superposition-base chain totalmass: got ${tm.samples[0]}, expected 1`);
  const lp = await ctx.getMeasure('lp');
  // A third of the mass-3 mixture, so log 3 below the first witness.
  const want = SUPERPOSE_CHAIN_DENSITY - Math.log(3);
  assert.ok(Math.abs(lp.samples[0] - want) < 1e-12,
    `probability superposition-base chain density: got ${lp.samples[0]}, expected ${want}`);
});
