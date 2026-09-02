'use strict';

// =====================================================================
// §06 `weighted` weight arity over ANY k-element-array variate.
// =====================================================================
//
// Spec §06 "Weight arity": "A one-parameter weight receives the variate whole.
// If the variate is a k-element array with k >= 2, a weight of exactly k scalar
// parameters instead receives one component per parameter, in order; any other
// arity is an error."
//
// The rule is stated over THE VARIATE, not over `Lebesgue`. The engine had it
// implemented for a Lebesgue box alone, so over `iid(M, k)` — the same
// k-element variate — the k-parameter spelling got no derivation at all and the
// one-parameter spelling bound only the FIRST coordinate, which made an
// indexing body throw "get index target is not an array (got number)".
//
// Both spellings denote one measure, so both are checked against the same
// closed form here. The base `iid(Uniform(interval(0, 1)), k)` has density 1 on
// the unit cube, so the weighted log-density at a point is log of the weight
// there exactly.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { makeMatCtx } = require('./_materialise-helpers.ts');

async function scoreScalar(ctx: any, name: string): Promise<number> {
  const m = await ctx.getMeasure(name);
  return m.value ? m.value.data[0] : m.samples[0];
}

test('both weight-arity spellings over an iid vector variate score the closed form',
  async () => {
    const { ctx } = makeMatCtx(`
square = iid(Uniform(interval(0.0, 1.0)), 2)

w2 = (x, y) -> x * y^2
m2 = weighted(w2, square)
lp2 = logdensityof(m2, [0.5, 0.8])

w1 = v -> v[1] * v[2]^2
m1 = weighted(w1, square)
lp1 = logdensityof(m1, [0.5, 0.8])
`, { sampleCount: 8 });
    const want = Math.log(0.5 * 0.8 ** 2);
    assert.ok(Math.abs(await scoreScalar(ctx, 'lp1') - want) < 1e-12, 'one-parameter form');
    assert.ok(Math.abs(await scoreScalar(ctx, 'lp2') - want) < 1e-12, 'k-parameter form');
  });

test('the k-parameter form classifies over an iid base', () => {
  const { built } = makeMatCtx(`
cube = iid(Uniform(interval(0.0, 1.0)), 3)
w = (x, y, z) -> x + y + z
m = weighted(w, cube)
`, { sampleCount: 8 });
  const d = built.derivations.m;
  assert.equal(d.kind, 'weighted');
  assert.equal(d.boxAxes, 3, 'one component per parameter, in axis order');
  assert.equal(d.isVariateWeight, true);
});

test('a k-parameter weight whose arity mismatches the variate is refused', () => {
  // §06: "any other arity is an error". Three parameters over a 2-element
  // variate has no coordinate for the third, so it must not classify.
  const { built } = makeMatCtx(`
square = iid(Uniform(interval(0.0, 1.0)), 2)
w = (x, y, z) -> x + y + z
m = weighted(w, square)
`, { sampleCount: 8 });
  assert.equal(built.derivations.m, undefined);
});

test('a k-parameter weight over an iid variate materialises as a weighted measure',
  async () => {
    const { ctx } = makeMatCtx(`
square = iid(Uniform(interval(0.0, 1.0)), 2)
w = (x, y) -> x * y^2
m = weighted(w, square)
`, { sampleCount: 4096 });
    const m = await ctx.getMeasure('m');
    assert.equal(m.shape, 'array', 'the vector atom shape survives reweighting');
    assert.deepEqual(m.dims, [2]);
    assert.ok(m.logWeights, 'per-atom weights are attached');
    // totalmass = ∫ x y² dx dy over the unit square = (1/2)(1/3) = 1/6. The
    // measure tracks it in log space; Monte Carlo at 4096 atoms holds ~1%.
    const got = Math.exp(m.logTotalmass);
    assert.ok(Math.abs(got - 1 / 6) < 0.01, `totalmass ${got}, closed form ${1 / 6}`);
  });

test('a SCALAR-variate base still binds the one-parameter weight as a scalar',
  async () => {
    // The whole-variate rule reads the variate, so a scalar base is unchanged:
    // the weight's parameter takes the scalar, not a length-1 vector.
    const { ctx } = makeMatCtx(`
base = Uniform(interval(0.0, 1.0))
w = x -> x^2
m = weighted(w, base)
lp = logdensityof(m, 0.5)
`, { sampleCount: 8 });
    const want = Math.log(0.25);   // density 1 on [0,1], so log w(0.5)
    assert.ok(Math.abs(await scoreScalar(ctx, 'lp') - want) < 1e-12);
  });

test('logweighted takes the same arity rule over an iid variate', async () => {
  const { ctx } = makeMatCtx(`
square = iid(Uniform(interval(0.0, 1.0)), 2)
g = (x, y) -> x + y
m = logweighted(g, square)
lp = logdensityof(m, [0.25, 0.5])
`, { sampleCount: 8 });
  // log-space weight, so the log-density is the weight itself.
  assert.ok(Math.abs(await scoreScalar(ctx, 'lp') - 0.75) < 1e-12);
});
