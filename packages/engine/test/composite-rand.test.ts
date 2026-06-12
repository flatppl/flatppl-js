'use strict';

// =====================================================================
// Demand-driven composite `rand` (engine-concepts §17.4 stage 2).
// =====================================================================
//
// `samples, _ = rand(state, iid(M, count))` for a COMPOSITE M — a
// forward measure the per-draw measure walker (sampler.walk) can't sample (here
// `Y_dist = lawof(Y)` with `Y = polyeval.([C], X)`, a deterministic
// transform of a stochastic iid vector). The classifier routes the
// DRAW half (`tuple_get(<rand>, 0)`) to a `randsample` derivation; the
// materialiser draws `count` independent realizations of M in a child
// ctx at sampleCount = count, seeded off the rand state.
//
// Two properties pinned here:
//   1. Y_samples materialises (was a hard "no resolveMeasureRef" error)
//      to [count, variate] with the right distribution (mean ≈ 3.0).
//   2. It is a FIXED value — independent of the session sample count
//      (the iid count is baked into the binding, not the display N).
//
// The leaf-rand gate (a single known-distribution inner stays on the
// batched `sampleLeafN` path, preserving the bit-for-bit
// `builtin_sample ≡ rand+iid` invariant) is pinned in the third test.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { makeMatCtx } = require('./_materialise-helpers.ts');
const { processSource, orchestrator } = require('..');

const FIXTURE = path.join(__dirname, 'fixtures', 'polyeval-iid-broadcast.flatppl');

// polyeval(C, x) = 2.3 + 1.5x + 0.7x² with x ~ Normal(0,1):
//   E = 2.3 + 1.5·0 + 0.7·E[x²] = 2.3 + 0.7 = 3.0
const EXPECTED_MEAN = 3.0;

function meanOf(data: any): number {
  let s = 0;
  for (let i = 0; i < data.length; i++) s += data[i];
  return s / data.length;
}

test('composite rand: Y_samples materialises to [1000,10] with E≈3.0', async () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { ctx, built } = makeMatCtx(src, { sampleCount: 64 });

  // The draw half classifies as `randsample`, not `evaluate` (which
  // would route through evaluateRand → the measure walker, which can't
  // sample this composite).
  const d = built.derivations.Y_samples;
  assert.equal(d.kind, 'randsample', 'Y_samples should be a randsample derivation');
  assert.equal(d.from, 'Y_dist', 'randsample draws from the iid inner measure');
  assert.equal(d.count, 1000, 'count = the iid size');

  const m = await ctx.getMeasure('Y_samples');
  assert.ok(m && m.value, 'Y_samples should materialise with a .value');
  // [count, variate] — 1000 iid draws, each a length-10 vector.
  assert.deepEqual(m.value.shape, [1000, 10], 'shape is [iid count, variate]');
  assert.equal(m.value.outerRank, 1, 'leading axis tagged as the sample axis');
  assert.equal(m.value.data.length, 10000, 'prod(shape) data');

  // Distribution check (deterministic given rstate, but asserted as a
  // calibration tolerance — split-seeding may change exact draws).
  const mean = meanOf(m.value.data);
  assert.ok(Math.abs(mean - EXPECTED_MEAN) < 0.15,
    `sample mean ${mean.toFixed(4)} should be ≈ ${EXPECTED_MEAN}`);

  // The draws are genuinely independent realizations, not one draw
  // tiled: the first 50 values are all distinct.
  const distinct = new Set(
    Array.from(m.value.data.slice(0, 50)).map((x: any) => x.toFixed(8)));
  assert.equal(distinct.size, 50, '1000 iid draws are distinct, not repeated');
});

test('composite rand: Y_samples is fixed — independent of session N', async () => {
  // A composite-rand draw is a FIXED value (deterministic given the
  // rng state); its count is the iid size, not the display sample
  // count. Materialising at two different session Ns yields the SAME
  // [1000,10] array, bit-for-bit.
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const a = makeMatCtx(src, { sampleCount: 16, rootSeed: 1 });
  const b = makeMatCtx(src, { sampleCount: 512, rootSeed: 999 });

  const ma = await a.ctx.getMeasure('Y_samples');
  const mb = await b.ctx.getMeasure('Y_samples');
  assert.deepEqual(ma.value.shape, mb.value.shape, 'shape independent of session N');
  assert.equal(ma.value.data.length, mb.value.data.length);
  // Identical despite different session N AND different rootSeed —
  // the draw is seeded off `rstate`, not the session key.
  let maxDiff = 0;
  for (let i = 0; i < ma.value.data.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(ma.value.data[i] - mb.value.data[i]));
  }
  assert.equal(maxDiff, 0, 'composite-rand draw is seeded off rstate, not the session');
});

test('composite rand: leaf-distribution inner stays off the randsample path', () => {
  // `A, _ = rand(rstate, iid(Normal(0,1), [3,3]))` — a single known
  // distribution inner. The leaf gate keeps it on the existing batched
  // path (pre-eval / sampleLeafN, bit-for-bit `builtin_sample ≡
  // rand+iid`), NOT `randsample`.
  const src = `
flatppl_compat = "0.1"
rstate = rnginit([9, 8, 7, 6])
A, _ = rand(rstate, iid(Normal(mu = 0, sigma = 1), [3, 3]))
`;
  const r = processSource(src);
  const errs = r.diagnostics.filter((dd: any) => dd.severity === 'error');
  assert.equal(errs.length, 0, 'leaf-rand source parses cleanly');
  const built = orchestrator.buildDerivations(r.bindings);
  assert.notEqual(built.derivations.A && built.derivations.A.kind, 'randsample',
    'leaf-rand must NOT route through randsample');
  // It is computed eagerly into fixedValues by pre-eval (the bit-for-bit
  // batched-leaf path), so the materialiser short-circuits it.
  assert.ok(built.fixedValues.has('A'),
    'leaf-rand is pre-evaluated into fixedValues');
});

// =====================================================================
// rand of a NAMED non-iid composite measure (the applied-kernel /
// lawof-binding draw — `sim_data, _ = rand(rstate, model_dist)` in the
// simple-transport fixtures). The shared classifyRandTuple gate used to
// overwrite the measure REF with its resolved IR while probing for the
// iid shape, so the no-iid fallback handed a `lawof` CALL to the
// must-be-a-ref check → null → kind='evaluate' → the per-draw evaluator
// crashed with "no resolveMeasureRef was supplied". The fallback now
// keeps the original ref → count-1 randsample via matRandSample. Two
// halves pinned through the SAME gate: the draw half classifies +
// materialises; the state half rewrites to rand_succ.
// =====================================================================

test('composite rand: named non-iid measure ref draws via randsample (count 1)', async () => {
  const src = `
mu0 = elementof(reals)
x ~ Normal(mu = mu0, sigma = 0.1)
ys ~ iid(Normal(mu = x, sigma = 0.1), 4)
K = kernelof(ys, mu0 = mu0)
model_dist = K(1.5)
rstate = rnginit([1, 2, 3, 4])
sim, rs2 = rand(rstate, model_dist)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 64 });
  const d = built.derivations.sim;
  assert.ok(d, 'sim must carry a derivation');
  assert.equal(d.kind, 'randsample', 'named composite-measure rand → randsample');
  assert.equal(d.count, 1, 'no-iid draw is a single realization');
  const m = await ctx.getMeasure('sim');
  const vals = m.value ? m.value.data : m.samples;
  assert.ok(vals && vals.length >= 4, 'one realization of the length-4 variate');
  for (let i = 0; i < vals.length; i++) {
    assert.ok(Number.isFinite(vals[i]), 'draw values are finite');
    assert.ok(Math.abs(vals[i] - 1.5) < 1.5, 'draws cluster near mu0=1.5');
  }
  // State half rides the SAME gate: composite successor → rand_succ.
  const rs2b = built.bindings.get('rs2');
  assert.equal(rs2b.ir && rs2b.ir.op, 'rand_succ',
    'composite state half rewrites to the value-domain successor');
});
