'use strict';
// `normalize` OVER A SELECTOR-DRIVEN SELECT, AND WHY THE CHAIN BASE STILL SAYS NO.
//
// THE DEFECT. `analyzer.isMeasureExpr` had no arm for `ifelse`, so
// `S = ifelse(c, A, B)` was not a measure EXPRESSION even though it classifies
// to a `select` derivation and materialises fine. `resolveMeasureBaseName` then
// declined it, `classifyNormalize` returned null, and `normalize(S)` produced
// NO derivation at all — the query died as "no derivation for 'tm'", so the
// exact mass 1 was unreachable.
//
// `ifelse` is DUAL the way `broadcast` is: `ifelse(c, 1.0, 2.0)` is a value,
// `ifelse(c, A, B)` over measure-typed branches is the discrete-selector
// mixture, a measure. So the arm is conditional on the branches, and `ifelse`
// stays OUT of the unconditional `MEASURE_PRODUCING` set.
//
// THE CHAIN BASE IS A DELIBERATE DECLINE, not an oversight. With `ifelse` a
// measure expression the chain classifies too, and then its body loses the
// base's weights entirely: `jointchain(aa = S, bb = fn(Normal(_, 1.0)))` over
// the mass-2.75 witness answers totalmass 1. A loud decline beats a silent
// wrong number, so `classifyJointchain` refuses an `ifelse` base until the
// chain-body mass loss is fixed (carded).
//
// ORACLES. `A = weighted(2.0, Normal(0,1))`, `B = weighted(3.0, Normal(5,1))`,
// `c ~ Bernoulli(0.25)`: the select's mass is 0.25·2 + 0.75·3 = 2.75, so
// `normalize` of it is exactly 1 and its density at 0.5 is the select's minus
// log 2.75 — from Distributions.jl, −1.7368814349470252 − log(2.75) =
// −2.748482346625505.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const UNEVEN = `
A = weighted(2.0, Normal(mu = 0.0, sigma = 1.0))
B = weighted(3.0, Normal(mu = 5.0, sigma = 1.0))
c = draw(Bernoulli(p = 0.25))
S = ifelse(c, A, B)
`;

test('select classify: normalize over a selector-driven select has mass 1', async () => {
  // Was "no derivation for 'tm'".
  const { ctx } = ctxFor(UNEVEN + `
M = normalize(S)
tm = totalmass(M)
`, 8192);
  assert.ok(ctx.derivations.M, 'the normalize must classify');
  assert.equal(ctx.derivations.M.kind, 'normalize');
  const tm = await ctx.getMeasure('tm');
  assert.ok(Math.abs(tm.samples[0] - 1) < 1e-11,
    `normalize over a select: totalmass ${tm.samples[0]}, expected 1`);
});

test('select classify: the normalized select scores its shifted density', async () => {
  // `normalize` subtracts log Z (§06), and Z here is the select's own mass, so
  // the density is the select's minus log 2.75. The mass estimate rides in the
  // shift, hence the band rather than 1e-12.
  const { ctx } = ctxFor(UNEVEN + `
M = normalize(S)
lp = logdensityof(M, 0.5)
`, 32768);
  const lp = await ctx.getMeasure('lp');
  const want = -1.7368814349470252 - Math.log(2.75);
  assert.ok(Math.abs(lp.samples[0] - want) < 0.02,
    `normalized select density: got ${lp.samples[0]}, expected about ${want}`);
});

test('select classify: EQUAL-mass branches under normalize stay exactly 1', async () => {
  // Two probability branches: the select's own mass is exactly 1 and the
  // normalize is a no-op, so nothing here is an estimate.
  const { ctx } = ctxFor(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 1.0)
c = draw(Bernoulli(p = 0.25))
S = ifelse(c, A, B)
M = normalize(S)
tm = totalmass(M)
`, 8192);
  const tm = await ctx.getMeasure('tm');
  assert.equal(tm.samples[0], 1,
    `normalize over an equal-mass select: got ${tm.samples[0]}, expected exactly 1`);
});

test('select classify: a VALUE ifelse is still not a measure expression', async () => {
  // The arm is conditional on the branches. `ifelse` over numbers must stay a
  // value, or `MEASURE_PRODUCING`'s consumers would mis-route it.
  const { isMeasureExpr } = require('../analyzer.ts');
  const { processSource } = require('..');
  const proc = processSource(`
c = draw(Bernoulli(p = 0.25))
v = ifelse(c, 1.0, 2.0)
`);
  const b = proc.bindings.get('v');
  assert.ok(b && b.node && b.node.value, 'the value binding has an AST');
  assert.equal(isMeasureExpr(b.node.value, proc.bindings), false,
    'ifelse over numbers is a value, not a measure');
});

test('select classify: a selector-driven select as a chain base DECLINES', async () => {
  // Deliberate. With the measure-expression arm the chain would classify and
  // then answer totalmass 1 against the exact 2.75, because the chain body
  // loses the base's weights. The decline keeps the loud failure until that is
  // fixed; see TODO-flatppl-js.md.
  const { ctx } = ctxFor(UNEVEN + `
M = jointchain(aa = S, bb = fn(Normal(_, 1.0)))
tm = totalmass(M)
`, 4096);
  assert.ok(!ctx.derivations.M,
    'the chain over an ifelse base must not classify while its mass is lost');
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('tm')),
    (e: any) => /no derivation/.test(e.message));
});
