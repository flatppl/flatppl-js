'use strict';

// §06: callbacks receive the whole variate. Coordinate extraction is explicit
// even when composing with a function that takes several scalar arguments.
// The iid Uniform base has density 1, so its weighted density is exactly w(x).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { makeMatCtx } = require('./_materialise-helpers.ts');

async function scoreScalar(ctx: any, name: string): Promise<number> {
  const m = await ctx.getMeasure(name);
  return m.value ? m.value.data[0] : m.samples[0];
}

test('explicit coordinate composition and a whole-vector callback score the closed form',
  async () => {
    const { ctx } = makeMatCtx(`
square = iid(Uniform(interval(0.0, 1.0)), 2)

w2 = (x, y) -> x * y^2
m2 = weighted(v -> w2(v[1], v[2]), square)
lp2 = logdensityof(m2, [0.5, 0.8])

w1 = v -> v[1] * v[2]^2
m1 = weighted(w1, square)
lp1 = logdensityof(m1, [0.5, 0.8])
`, { sampleCount: 8 });
    const want = Math.log(0.5 * 0.8 ** 2);
    assert.ok(Math.abs(await scoreScalar(ctx, 'lp1') - want) < 1e-12, 'one-parameter form');
    assert.ok(Math.abs(await scoreScalar(ctx, 'lp2') - want) < 1e-12, 'explicit composition');
  });

test('explicit coordinate composition scores all axes of an iid cube', async () => {
  const { ctx } = makeMatCtx(`
cube = iid(Uniform(interval(0.0, 1.0)), 3)
w = (x, y, z) -> x + y + z
m = weighted(v -> w(v[1], v[2], v[3]), cube)
lp = logdensityof(m, [0.2, 0.3, 0.4])
`, { sampleCount: 8 });
  assert.ok(Math.abs(await scoreScalar(ctx, 'lp') - Math.log(0.9)) < 1e-12);
});

test('weight callbacks never implicitly unpack array coordinates', () => {
  for (const params of ['x, y', 'x, y, z']) assert.throws(() => makeMatCtx(`
square = iid(Uniform(interval(0.0, 1.0)), 2)
w = (${params}) -> x + y
m = weighted(w, square)
`, { sampleCount: 8 }), /callback receives 1 whole input/);
});

test('a composed weight over an iid variate materialises with the analytic mass',
  async () => {
    const { ctx } = makeMatCtx(`
square = iid(Uniform(interval(0.0, 1.0)), 2)
w = (x, y) -> x * y^2
m = weighted(v -> w(v[1], v[2]), square)
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
g = v -> v[1] + v[2]
m = logweighted(g, square)
lp = logdensityof(m, [0.25, 0.5])
`, { sampleCount: 8 });
  // log-space weight, so the log-density is the weight itself.
  assert.ok(Math.abs(await scoreScalar(ctx, 'lp') - 0.75) < 1e-12);
});
