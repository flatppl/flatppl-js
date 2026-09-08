'use strict';

// Spec §06 iid is a product measure: a fixed inner mass Z contributes Z^k.
// All mass/density oracles below are algebraic, not sampled estimates.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { ENGINE_LIMITATION } = require('../limitations.ts');
const shared = require('../materialiser-shared.ts');

function close(actual: number, expected: number) {
  assert.ok(Math.abs(actual - expected) < 1e-11,
    `${actual} differs from the exact oracle ${expected}`);
}

for (const [label, source] of [
  ['constructor', 'q = weighted(2.0, Normal(0.0, 1.0))'],
  ['reified law', 'x ~ Normal(0.0, 1.0)\nq = weighted(2.0, lawof(x))'],
]) {
  test(`iid retains fixed mass over a ${label} and agrees with its density`, async () => {
    const { ctx } = makeMatCtx(`${source}
M = iid(q, 2)
mass = totalmass(M)
lp = logdensityof(M, [0.0, 0.0])
`, { sampleCount: 128, rootSeed: 818 });
    close((await ctx.getMeasure('mass')).samples[0], 4);
    // Product of two densities 2*phi(0) integrates to 2².
    close((await ctx.getMeasure('lp')).samples[0], Math.log(4) - Math.log(2 * Math.PI));
  });
}

for (const [label, base, size, expected] of [
  ['nested constant factors', 'weighted(3.0, weighted(2.0, Normal(0.0, 1.0)))', '2', 36],
  ['additive component masses',
    'superpose(weighted(2.0, Normal(0.0, 1.0)), weighted(3.0, Normal(1.0, 1.0)))', '2', 25],
  ['normalization', 'normalize(weighted(2.0, Normal(0.0, 1.0)))', '2', 1],
  ['multiple dimensions', 'weighted(2.0, Normal(0.0, 1.0))', '[2, 3]', 64],
] as Array<[string, string, string, number]>) {
  test(`iid fixed mass composes through ${label}`, async () => {
    const { ctx } = makeMatCtx(`q = ${base}\nM = iid(q, ${size})\nmass = totalmass(M)`,
      { sampleCount: 128, rootSeed: 818 });
    close((await ctx.getMeasure('mass')).samples[0], expected);
  });
}

test('record iid retains its fixed product mass on the table result', async () => {
  const { ctx } = makeMatCtx(`
q = joint(a = weighted(2.0, Normal(0.0, 1.0)), b = weighted(3.0, Normal(1.0, 1.0)))
M = iid(q, 2)
mass = totalmass(M)
`, { sampleCount: 1, rootSeed: 818 });
  const table = await ctx.getMeasure('M');
  assert.equal(table.nrows, 2);
  close((await ctx.getMeasure('mass')).samples[0], 36);
});

test('a repeated constructor remains a unit-mass product beneath a scalar weight', async () => {
  const { ctx } = makeMatCtx(`
q = Normal(0.0, 1.0)
M = iid(weighted(2.0, joint(a = q, b = q)), 2)
mass = totalmass(M)
`, { sampleCount: 1, rootSeed: 818 });
  // The product of normalized factors has mass one. Two weighted copies have 2².
  close((await ctx.getMeasure('mass')).samples[0], 4);
});

test('iid retains translated pushforward mass and its exact density', async () => {
  const { ctx } = makeMatCtx(`
M = iid(pushfwd(fn(_ + 1.0), weighted(2.0, Normal(0.0, 1.0))), 2)
mass = totalmass(M)
lp = logdensityof(M, [1.0, 1.0])
P = normalize(M)
normalizedMass = totalmass(P)
normalizedLp = logdensityof(P, [1.0, 1.0])
`, { sampleCount: 128, rootSeed: 818 });
  close((await ctx.getMeasure('mass')).samples[0], 4);
  close((await ctx.getMeasure('lp')).samples[0], Math.log(4) - Math.log(2 * Math.PI));
  close((await ctx.getMeasure('normalizedMass')).samples[0], 1);
  close((await ctx.getMeasure('normalizedLp')).samples[0], -Math.log(2 * Math.PI));
});

test('a non-injective pushforward also preserves fixed iid mass', async () => {
  const { ctx } = makeMatCtx(`
M = iid(pushfwd(fn(abs(_)), weighted(2.0, Normal(0.0, 1.0))), 2)
mass = totalmass(M)
`, { sampleCount: 128, rootSeed: 818 });
  close((await ctx.getMeasure('mass')).samples[0], 4);
  assert.ok((await ctx.getMeasure('M')).samples.every((x: number) => x >= 0));
});

test('fixed conditional mass preserves the shared stochastic parameter', async () => {
  const n = 30000;
  const { ctx } = makeMatCtx(`
z ~ Normal(0.0, 2.0)
q = weighted(2.0, Normal(z, 0.6))
M = iid(q, 2)
mass = totalmass(M)
`, { sampleCount: n, rootSeed: 818 });
  close((await ctx.getMeasure('mass')).samples[0], 4);
  const flat = (await ctx.getMeasure('M')).samples;
  const z = (await ctx.getMeasure('z')).samples;
  const a = Float64Array.from({ length: n }, (_, i) => flat[2 * i]);
  const b = Float64Array.from({ length: n }, (_, i) => flat[2 * i + 1]);
  const cov = (x: Float64Array, y: Float64Array) => {
    const mx = x.reduce((s, v) => s + v, 0) / n;
    const my = y.reduce((s, v) => s + v, 0) / n;
    return x.reduce((s, v, i) => s + (v - mx) * (y[i] - my), 0) / n;
  };
  // Conditional X_j=z+epsilon_j: Cov(X_1,X_2)=Var(z)=4.
  assert.ok(Math.abs(cov(a, b) - 4) < 0.24);
  for (const col of [a, b]) {
    const noise = Float64Array.from(col, (x, i) => x - z[i]);
    assert.ok(Math.abs(cov(noise, noise) - 0.36) < 0.025);
  }
});

// ---------------------------------------------------------------------
// An UNCERTIFIED product mass is unknown, not 1 (see `limitations.ts`)
// ---------------------------------------------------------------------

test('an uncertified composite iid refuses totalmass instead of answering 1', async () => {
  // `truncate` is outside `_iidFixedLogTotalmass`'s certified algebra, and its
  // real mass is NOT one: the inner measure has 2·P(|X| < 1) = 1.36537898…, so
  // the product over k = 2 is 1.86425977…. Answering 1 was a wrong scalar, not
  // a missing one, which is exactly what the marker now prevents.
  const { ctx } = makeMatCtx(`
q = truncate(weighted(2.0, Normal(0.0, 1.0)), interval(-1.0, 1.0))
M = iid(q, 2)
mass = totalmass(M)
`, { sampleCount: 64, rootSeed: 818 });
  // Sampling still works — the refusal costs the mass query alone.
  const m = await ctx.getMeasure('M');
  assert.equal(m.logTotalmass, null, 'an uncertified mass is null, not 0');
  assert.equal(m.samples.length, 128);
  await assert.rejects(() => ctx.getMeasure('mass'), (e: any) => {
    assert.equal(e.code, ENGINE_LIMITATION, e.message);
    assert.deepEqual(e.limitation,
      { construct: 'totalmass', route: 'composite iid' });
    assert.match(e.message, /carries an uncertified mass/);
    return true;
  });
});

test('a certified composite iid still reports k times the inner mass', async () => {
  // The guard must not fire on the certified path: `weighted` IS in the
  // algebra, so the mass stays k · log 2 = log 4 and never reaches the refusal.
  const { ctx } = makeMatCtx(`
q = weighted(2.0, Normal(0.0, 1.0))
x ~ Normal(0.0, 1.0)
r = weighted(2.0, lawof(x))
M = iid(r, 2)
mass = totalmass(M)
`, { sampleCount: 64, rootSeed: 818 });
  const m = await ctx.getMeasure('M');
  // k · inner = 2 · log 2, the certified product mass, and never null.
  assert.equal(m.logTotalmass, 2 * Math.log(2));
  close((await ctx.getMeasure('mass')).samples[0], 4);
});

test('an expansion error during certification is not swallowed', async () => {
  // Mass certification is OPTIONAL, but it must not hide a fault to stay
  // optional. A measure with no expanded density IR RETURNS null rather than
  // throwing, so a catch here could only ever swallow a real error — and the
  // one throw reachable from this call is `jointchain`'s duplicate-label
  // MODEL error, which the user has to see. Inject a fault and require it out.
  const derivations = require('../derivations.ts');
  const original = derivations.expandMeasureIR;
  derivations.expandMeasureIR = () => {
    throw new Error('expandMeasureIR: injected expansion fault');
  };
  try {
    const { ctx } = makeMatCtx(`
x ~ Normal(0.0, 1.0)
r = weighted(2.0, lawof(x))
M = iid(r, 2)
`, { sampleCount: 64, rootSeed: 818 });
    await assert.rejects(() => ctx.getMeasure('M'), /injected expansion fault/);
  } finally {
    derivations.expandMeasureIR = original;
  }
});

test('an uncertified mass propagates through a wrapper instead of becoming 1', async () => {
  // Every derived mass goes through `massOf` / `addMass`, so a wrapper around
  // an uncertified iid stays uncertified. Reading the null as 0 here would
  // reintroduce the wrong answer one level out.
  for (const wrapper of ['weighted(3.0, M)', 'pushfwd(fn(_ + 1.0), M)',
                         'joint(a = M, b = Normal(0.0, 1.0))']) {
    const { ctx } = makeMatCtx(`
q = truncate(weighted(2.0, Normal(0.0, 1.0)), interval(-1.0, 1.0))
M = iid(q, 2)
W = ${wrapper}
mass = totalmass(W)
`, { sampleCount: 64, rootSeed: 818 });
    const w = await ctx.getMeasure('W');
    assert.equal(shared.massOf(w), null, wrapper);
    await assert.rejects(() => ctx.getMeasure('mass'), (e: any) => {
      assert.equal(e.code, ENGINE_LIMITATION, wrapper + ': ' + e.message);
      return true;
    });
  }
});

test('superpose refuses a component whose mass is uncertified', async () => {
  // Unlike the mass sums, §06's mixture weights are BUILT from the component
  // masses, so this reader needs a number for sampling and cannot carry the
  // unknown forward. `null - x` is `-x` in JavaScript, so defaulting would
  // have produced a finite, wrong offset and silently mis-weighted draws.
  const { ctx } = makeMatCtx(`
q = truncate(weighted(2.0, Normal(0.0, 1.0)), interval(-1.0, 1.0))
A = iid(q, 2)
B = iid(weighted(2.0, Normal(0.0, 1.0)), 2)
S = superpose(A, B)
`, { sampleCount: 32, rootSeed: 818 });
  await assert.rejects(() => ctx.getMeasure('S'), (e: any) => {
    assert.equal(e.code, ENGINE_LIMITATION, e.message);
    assert.deepEqual(e.limitation, {
      construct: 'superpose over a component with an uncertified mass',
      route: 'sampling',
    });
    return true;
  });
});
