'use strict';

// Sampling-side counterpart to #187 (engine: enforce non-negative
// superposition weights). #187 split the DENSITY path's per-atom weight
// accumulator in two (density.ts):
//
//   - addLogWOfVariate — a weight that is a FUNCTION of the base's variate
//     (spec §06: "f is a non-negative weight (a constant or a function of
//     the variate x of M)"). A negative sample there is legitimate
//     off-support behaviour ("contributes nothing rather than corrupting
//     the quadrature"), so it collapses to -Infinity and is not refused.
//   - addLogW — anything else (a constant, or a closed-form expression the
//     static passes could not fold). A negative sample there makes
//     weighted(w, M) a signed measure at that atom, with no density to
//     report, so it throws (tagged `modelRefusal`).
//
// materialiser.ts's three matWeighted call sites (record/tuple base, plain
// scalar base, N-D box base) never drew this line for SAMPLING: every one
// treated a negative per-atom weight sample leniently (console.warn +
// zero mass), including the variate-INDEPENDENT case where §06 gives no
// legitimate reading. `_addWeightedLogSamples` (materialiser.ts) now takes
// the same `isVariateWeight` split, fed by a new `DerivationWeighted`
// field derivations.ts's `_classifyWeightedByFunction` sets on the
// parameter-substituted weightIR it builds (the substitution already
// erases the raw `functionof` shape density.ts's walker keys off, so the
// derivation needs its own flag).
//
// ORACLE for the lenient retained-mass case, mirroring the density side's
// own x-1 example: `weighted(x -> x - 1, Lebesgue(interval(0, 2)))` has
// mass ∫₀² max(x-1, 0) dx = ∫₁² (x-1) dx = 1/2.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { makeMatCtx } = require('./_materialise-helpers.ts');

function makeCtx(source: string, sampleCount = 8192) {
  return makeMatCtx(source, { sampleCount, rootSeed: 0xBEEF01 }).ctx;
}

// =====================================================================
// Scalar base (site: matWeighted's plain, non-record/tuple/box branch)
// =====================================================================

test('§06: a valid function-of-variate weight over a scalar Lebesgue base is '
  + 'unaffected — regression floor for the split', async () => {
  // f(x) = x + 1 is strictly positive on [0, 2], so nothing here ever
  // touches the negative branch; this pins that the split moved no legal
  // number for the ordinary case. Z = ∫₀²(x+1)dx = 2 + 2 = 4.
  const ctx = makeCtx(`
f = x -> x + 1.0
M = weighted(f, Lebesgue(support = interval(0.0, 2.0)))
`);
  const m: any = await ctx.getMeasure('M');
  const got = Math.exp(m.logTotalmass);
  assert.ok(Math.abs(got - 4) / 4 < 0.02,
    `totalmass ${got} should be ≈ 4 (rel err ${Math.abs(got - 4) / 4})`);
});

test('§06: a variate-independent negative weight sample refuses, tagged '
  + 'modelRefusal, instead of silently zeroing mass', async () => {
  // min(0.2, -0.3) is evaluable but not a `functionof` weight and not
  // statically foldable to a literal, so it reaches matWeighted's
  // weightIR branch same as a function weight would — but it does not vary
  // with the base's variate, so §06 gives it no legitimate negative reading.
  const ctx = makeCtx(`
M = weighted(min(0.2, -0.3), Normal(0.0, 1.0))
`);
  await assert.rejects(() => ctx.getMeasure('M'), (e: any) => {
    assert.match(e.message, /non-negative/);
    assert.match(e.message, /§06/);
    assert.match(e.message, /-0\.3/);
    assert.equal(e.modelRefusal, true,
      'must be tagged modelRefusal so MCMC re-raises rather than treats it as a rejected proposal');
    return true;
  });
});

test('§06: a variate-dependent weight retains only the positive-region mass, '
  + 'pinned against the closed form (x - 1 on [0, 2] ⇒ 1/2)', async () => {
  const { ctx } = makeMatCtx(`
f = x -> x - 1.0
M = weighted(f, Lebesgue(support = interval(0.0, 2.0)))
`, { sampleCount: 300000, rootSeed: 0xF00D });
  const m: any = await ctx.getMeasure('M');
  const got = Math.exp(m.logTotalmass);
  assert.ok(Math.abs(got - 0.5) < 0.02,
    `retained mass ${got} should be ≈ 0.5 (the negative half of [0,2] contributes zero)`);
});

// =====================================================================
// Record/tuple base (site: matWeighted's isRecord/isTuple branch)
// =====================================================================

test('§06: a variate-dependent weight over a record base collapses negative '
  + 'atoms to -Infinity, exact per atom (no MC noise)', async () => {
  const ctx = makeCtx(`
m = joint(a = Normal(0.0, 1.0), b = Normal(2.0, 1.0))
g = r -> r.a
W = weighted(g, m)
`);
  const [W, M]: [any, any] = await Promise.all(
    [ctx.getMeasure('W'), ctx.getMeasure('m')]);
  const N = M.fields.a.samples.length;
  const baseline = -Math.log(N);
  let sawNegative = false, sawPositive = false;
  for (let i = 0; i < N; i++) {
    const a = M.fields.a.samples[i];
    if (a > 0) {
      sawPositive = true;
      assert.ok(Math.abs((W.logWeights[i] - baseline) - Math.log(a)) < 1e-9,
        `atom ${i}: positive weight sample must be log(a)`);
    } else {
      sawNegative = true;
      assert.equal(W.logWeights[i], -Infinity,
        `atom ${i}: negative weight sample (a=${a}) must collapse to -Infinity`);
    }
  }
  assert.ok(sawPositive && sawNegative, 'the fixture must exercise both signs');
});

test('§06: a variate-independent negative weight over a record base refuses', async () => {
  const ctx = makeCtx(`
m = joint(a = Normal(0.0, 1.0), b = Normal(2.0, 1.0))
W = weighted(min(0.2, -0.3), m)
`);
  await assert.rejects(() => ctx.getMeasure('W'), (e: any) => {
    assert.match(e.message, /non-negative/);
    assert.equal(e.modelRefusal, true);
    return true;
  });
});

// =====================================================================
// N-D box base (site: _matWeightedOverBox)
// =====================================================================

test('§06: a k-parameter variate-dependent weight over a box retains only '
  + 'the positive-region mass (same x-1 oracle, over [0,2]×[0,1] ⇒ 1/2)', async () => {
  const { ctx } = makeMatCtx(`
L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 1.0)))
f(x, y) = x - 1.0
M = weighted(f, L)
`, { sampleCount: 300000, rootSeed: 0xF00D });
  const m: any = await ctx.getMeasure('M');
  const got = Math.exp(m.logTotalmass);
  assert.ok(Math.abs(got - 0.5) < 0.02,
    `retained mass ${got} should be ≈ 0.5`);
});

test('§06: a variate-independent negative weight over a box base refuses', async () => {
  const { ctx } = makeMatCtx(`
L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 1.0)))
M = weighted(min(0.2, -0.3), L)
`, { sampleCount: 64 });
  await assert.rejects(() => ctx.getMeasure('M'), (e: any) => {
    assert.match(e.message, /non-negative/);
    assert.equal(e.modelRefusal, true);
    return true;
  });
});
