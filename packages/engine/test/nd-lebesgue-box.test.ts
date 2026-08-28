'use strict';

// Lebesgue over an N-D box support — spec §06 "Fundamental measures":
// "`Lebesgue(support = S)` — the canonical continuous reference measure on
// the support set `S`, restricted to `S`. For full-dimensional subsets of
// Euclidean or product spaces this is the ordinary Lebesgue measure on the
// ambient space", and "`S` may be any FlatPPL set: one-dimensional …, a
// Cartesian power (e.g. `cartpow(reals, n)`), a record-structured product
// … and so on".
//
// The positional `cartprod(interval, …)` spelling carries an ARRAY variate
// (spec §03: "The resulting set is a set of arrays, not a set of tuples"),
// so the box measure's atoms are k-vectors and its total mass is the
// product of the side lengths.
//
// ORACLES. Every expected number is derived by hand in closed form (the
// derivation sits in each test), never read off an engine run. Two classes
// of assertion, kept apart deliberately:
//   - EXACT (tolerance ~1e-12): the box mass, the reference density ≡ 1,
//     and the uniform-on-box density — all closed forms the engine also
//     computes in closed form.
//   - MONTE CARLO (stated relative tolerance): anything whose normalizer is
//     Z = ∫ f dLebesgue for a genuine function weight. The engine estimates
//     that Z from the box's own atoms, so the comparison is 1/√N-limited.
//     Tolerances below are set well inside the error a DROPPED weight would
//     produce (a flat density, off by a factor, not by permille).
//
// The record-form (keyword) `cartprod(a = S1, …)` spelling stays out of
// scope and is asserted to refuse rather than mislower.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function derivOf(src: string, name: string) {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const built = orchestrator.buildDerivations(lifted.bindings);
  return { d: built.derivations[name], built, lifted };
}

function buildCtx(src: string, N: number, seed = 7) {
  const proc = processSource(src);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    // rootSeed only: the materialiser derives the two-lane PhiloxKey itself.
    // Passing `rootKey: seed` (a bare number) made every seed sample the same
    // stream, which is what made this file's Monte-Carlo masses look
    // suspiciously seed-independent.
    rootSeed: seed, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

// Score `logdensityof(<measure>, <point>)` through a named binding — the
// real extraction route. The point is an array literal (the box variate).
async function scoreAt(src: string, measure: string, point: number[], N = 4096, seed = 7) {
  const lit = '[' + point.map((v) => v.toFixed(6)).join(', ') + ']';
  const ctx = buildCtx(src + '\nld = logdensityof(' + measure + ', ' + lit + ')\n', N, seed);
  const m = await ctx.getMeasure('ld');
  return m.samples[0];
}

// =====================================================================
// Classification
// =====================================================================

test('Lebesgue over a positional cartprod of intervals classifies as a box', () => {
  const { d } = derivOf(
    'L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))', 'L');
  assert.ok(d, 'expected a derivation for the 2-D box Lebesgue');
  assert.equal(d.kind, 'lebesguebox');
  assert.deepEqual(d.axes.map((a: any) => [a.lo, a.hi]), [[0, 2], [0, 3]]);
  // logTotalmass = log(side₁ · side₂) = log(2 · 3) = log 6.
  assert.ok(Math.abs(d.logTotalmass - Math.log(6)) < 1e-15,
    'logTotalmass ' + d.logTotalmass + ' ≠ log 6 = ' + Math.log(6));
});

test('Lebesgue over cartpow(interval, n) classifies as an n-axis box', () => {
  const { d } = derivOf('L = Lebesgue(support = cartpow(interval(-1.0, 1.0), 3))', 'L');
  assert.ok(d, 'expected a derivation for cartpow(interval, 3)');
  assert.equal(d.kind, 'lebesguebox');
  assert.equal(d.axes.length, 3);
  // Each side is 1 − (−1) = 2, so logTotalmass = log(2³) = 3 log 2.
  assert.ok(Math.abs(d.logTotalmass - 3 * Math.log(2)) < 1e-15,
    'logTotalmass ' + d.logTotalmass + ' ≠ 3·log 2');
});

test('1-D Lebesgue(interval) keeps its existing scalar classification', () => {
  // Regression guard: the 1-D path must not be rerouted through the box kind.
  const { d } = derivOf('L = Lebesgue(support = interval(0.0, 4.0))', 'L');
  assert.equal(d.kind, 'sample');
  assert.ok(Math.abs(d.logTotalmass - Math.log(4)) < 1e-15);
});

test('a single-component positional cartprod is a ONE-axis box, not the scalar case', () => {
  // §03 makes a member the `cat` of one element per component and §07 `cat(x)`
  // is `vector(x)` for a scalar, so the support is a set of length-1 VECTORS.
  // It classifies as a 1-axis box; only the bare `interval(a,b)` spelling keeps
  // the scalar variate. The mass is the same either way (log 4).
  const { d } = derivOf('L = Lebesgue(support = cartprod(interval(0.0, 4.0)))', 'L');
  assert.equal(d.kind, 'lebesguebox');
  assert.equal(d.axes.length, 1);
  assert.ok(Math.abs(d.logTotalmass - Math.log(4)) < 1e-15,
    'logTotalmass ' + d.logTotalmass + ' ≠ log 4');
});

test('an unbounded box axis is refused, not silently given finite mass', () => {
  // Spec §06: normalize / totalmass of an infinite-mass measure is
  // undefined. An unbounded axis makes the box mass infinite, so the
  // derivation must decline rather than invent a side length.
  const { d } = derivOf('L = Lebesgue(support = cartprod(interval(0.0, 1.0), reals))', 'L');
  assert.equal(d, undefined, 'expected no box derivation for an unbounded axis');
});

test('an INFINITE box axis bound is refused', () => {
  // Distinct from the `reals` factor above: this factor IS an interval, but a
  // semi-infinite one, so the box volume diverges and §06 leaves
  // normalize/totalmass undefined. Refuse rather than emit an infinite mass.
  assert.equal(derivOf(
    'L = Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, inf)))', 'L').d,
    undefined);
  // A degenerate axis (hi == lo) is zero-volume, likewise refused.
  assert.equal(derivOf(
    'L = Lebesgue(support = cartprod(interval(0.0, 1.0), interval(2.0, 2.0)))', 'L').d,
    undefined);
});

test('cartpow box dimension runs from 1 up to the axis cap', () => {
  // k = 1 is a ONE-axis box, matching the single-component cartprod above.
  // `cartpow(reals, 1)` is R¹, a one-element ARRAY, so it carries a length-1
  // vector variate and never collapses to the scalar interval case.
  const one = derivOf('L = Lebesgue(support = cartpow(interval(0.0, 1.0), 1))', 'L').d;
  assert.equal(one?.kind, 'lebesguebox', 'cartpow(interval, 1) is a 1-axis box');
  assert.equal(one.axes.length, 1);
  assert.equal(derivOf('L = Lebesgue(support = cartpow(interval(0.0, 1.0), 9))', 'L').d,
    undefined, 'cartpow(interval, 9) exceeds the 8-axis cap');
  assert.equal(derivOf('L = Lebesgue(support = cartpow(interval(0.0, 1.0), 8))', 'L').d?.kind,
    'lebesguebox', 'cartpow(interval, 8) is exactly at the cap');
});

test('the record-form cartprod spelling does not classify as a box', () => {
  // Out of scope: a record variate has no positional axis order. Refuse
  // rather than mislower it to the positional reading.
  const { d } = derivOf(
    'L = Lebesgue(support = cartprod(a = interval(0.0, 1.0), b = interval(0.0, 1.0)))', 'L');
  assert.equal(d, undefined, 'record-form cartprod must not classify as a positional box');
});

test('a set-binding support resolves to the box (the Dalitz spelling)', () => {
  // flatppl-examples/examples/dminus-to-3pi-amplitude.flatppl names its box
  // first (`square = cartprod(...)`) and then passes the NAME as the
  // support. That must classify identically to the inline spelling.
  const { d } = derivOf(`square = cartprod(interval(0.0, 2.0), interval(0.0, 3.0))
L = Lebesgue(support = square)
`, 'L');
  assert.ok(d, 'expected a box derivation through a named set binding');
  assert.equal(d.kind, 'lebesguebox');
  assert.ok(Math.abs(d.logTotalmass - Math.log(6)) < 1e-15);
});

// =====================================================================
// Sampling — uniform over the box
// =====================================================================

test('box Lebesgue materialises k-vector atoms with the box mass', async () => {
  const { ctx } = makeMatCtx(
    'L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))',
    { sampleCount: 4096 });
  const m: any = await ctx.getMeasure('L');
  assert.deepEqual(m.dims, [2], 'atoms must be 2-vectors, got dims=' + JSON.stringify(m.dims));
  assert.equal(m.samples.length, 4096 * 2);
  assert.ok(Math.abs(m.logTotalmass - Math.log(6)) < 1e-12,
    'logTotalmass ' + m.logTotalmass + ' ≠ log 6');
});

test('box Lebesgue samples uniformly: coordinate means hit the midpoints', async () => {
  // Oracle: for U ~ Uniform(box), E[xᵢ] = (loᵢ+hiᵢ)/2 exactly. Axis 1 on
  // [0,2] → 1.0; axis 2 on [0,3] → 1.5. The per-axis standard error is
  // side/√(12N) = 0.0045 and 0.0068 at N = 16384, so the 0.05 tolerance is
  // a ~7σ shape check, not a tight statistical claim.
  const N = 16384;
  const { ctx } = makeMatCtx(
    'L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))',
    { sampleCount: N });
  const m: any = await ctx.getMeasure('L');
  let s0 = 0, s1 = 0;
  for (let i = 0; i < N; i++) { s0 += m.samples[2 * i]; s1 += m.samples[2 * i + 1]; }
  assert.ok(Math.abs(s0 / N - 1.0) < 0.05, 'axis-1 mean ' + (s0 / N) + ' ≉ 1.0');
  assert.ok(Math.abs(s1 / N - 1.5) < 0.05, 'axis-2 mean ' + (s1 / N) + ' ≉ 1.5');
  // Every atom must lie inside the box (support restriction, spec §06).
  for (let i = 0; i < N; i++) {
    const x = m.samples[2 * i], y = m.samples[2 * i + 1];
    assert.ok(x >= 0 && x <= 2 && y >= 0 && y <= 3,
      'atom ' + i + ' = [' + x + ', ' + y + '] outside the box');
  }
});

// =====================================================================
// Density — EXACT closed forms
// =====================================================================

test('logdensityof(Lebesgue(box), x) is 0 inside and −∞ outside', async () => {
  // Spec §06: the Lebesgue density w.r.t. itself is ≡ 1, and "density is
  // zero outside" the support. Both are exact, no normalizer involved.
  const src = 'L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))';
  assert.equal(await scoreAt(src, 'L', [1.0, 1.5]), 0);
  assert.equal(await scoreAt(src, 'L', [2.5, 1.5]), -Infinity);
  assert.equal(await scoreAt(src, 'L', [1.0, -0.5]), -Infinity);
});

test('the uniform probability measure on a box scores −log(area) exactly', async () => {
  // normalize(Lebesgue(box)) is the uniform distribution on the box: density
  // 1/area inside, area = 2·3 = 6. The normalizer is the box mass, which the
  // derivation carries in closed form, so this must be exact — a Monte-Carlo
  // Z here would be a regression.
  const src = `L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))
P = normalize(L)
`;
  const got = await scoreAt(src, 'P', [1.0, 1.5]);
  assert.ok(Math.abs(got + Math.log(6)) < 1e-12,
    'got ' + got + ', closed form ' + (-Math.log(6)));
});

// =====================================================================
// Density — function weights (Monte-Carlo normalizer)
// =====================================================================

// Z = ∫ f dLebesgue over the box is estimated from the box's own atoms, so
// these comparisons carry a 1/√N error. At N = 2·10⁵ the observed relative
// error is a few 1e-3; 2e-2 leaves margin for RNG variance across seeds
// while staying far inside the factor-level error a dropped weight gives.
const MC_N = 200000;
const MC_TOL = 2e-2;

function assertRel(got: number, want: number, tol: number, what: string) {
  const gotD = Math.exp(got), rel = Math.abs(gotD - want) / Math.abs(want);
  assert.ok(rel < tol,
    what + ': density ' + gotD + ' vs closed form ' + want + ' (rel ' + rel + ')');
}

test('normalize(weighted(x·y, Lebesgue([0,1]²))) has density 4xy', async () => {
  // Closed form: Z = ∫₀¹∫₀¹ xy dx dy = (1/2)(1/2) = 1/4, so the normalized
  // density is xy / (1/4) = 4xy.
  const src = `L = Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 1.0)))
f(x, y) = x * y
P = normalize(weighted(f, L))
`;
  for (const [pt, want] of [
    [[0.5, 0.5], 1.0],
    [[0.25, 0.8], 0.8],
    [[0.125, 0.5], 0.25],
  ] as [number[], number][]) {
    assertRel(await scoreAt(src, 'P', pt, MC_N), want, MC_TOL, 'P at [' + pt + ']');
  }
});

test('a k-param weight over a k-axis box binds parameters to axes in order', async () => {
  // Axis order must not be transposable: f(x, y) = x over a box whose axes
  // have different extents pins which coordinate feeds which parameter.
  // Z = ∫₀¹∫₀³ x dy dx = 3 · (1/2) = 3/2, so the density is x/(3/2) = 2x/3.
  // At x = 0.5 that is 1/3. Had the axes been swapped the weight would read
  // the [0,3] coordinate and the density would be a different function
  // entirely, so this assertion is the axis-order oracle.
  const src = `L = Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 3.0)))
f(x, y) = x
P = normalize(weighted(f, L))
`;
  assertRel(await scoreAt(src, 'P', [0.5, 2.0], MC_N), 2 * 0.5 / 3, MC_TOL, 'axis order');
});

test('a single-param weight receives the whole array variate', async () => {
  // Spec §06 states the weight is "a function of the variate x of M"; for a
  // box that variate is the array itself, so a 1-param weight indexes it.
  // Z = ∫₀¹∫₀¹ (x + y) dx dy = 1, so the normalized density is x + y.
  const src = `L = Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 1.0)))
f(v) = v[1] + v[2]
P = normalize(weighted(f, L))
`;
  assertRel(await scoreAt(src, 'P', [0.25, 0.5], MC_N), 0.75, MC_TOL, 'array-variate weight');
});

// =====================================================================
// One-axis box: the length-1-vector variate, and the density identity
// against the scalar spelling. Spec §03 makes a cartprod member the `cat`
// of one element per component and §07 `cat(x)` is `vector(x)` for a
// scalar, so `cartprod(interval(a,b))` has a length-1 VECTOR variate
// while bare `interval(a,b)` stays scalar. The two carry the SAME mass and
// the same density value — the reference measure over a length-1 box is
// the scalar one composed with the trivial R ≅ R¹ identification — so the
// typing fix must not move any number.
// =====================================================================

test('the 1-axis box and the scalar interval agree on mass and density', async () => {
  // Closed forms, both spellings: total mass = 4 − 0 = 4, and the
  // normalized (uniform) density is 1/4 everywhere inside, so
  // logdensityof = −log 4. Exact on both paths, no Monte Carlo.
  const boxDeriv = derivOf('L = Lebesgue(support = cartprod(interval(0.0, 4.0)))', 'L').d;
  const scalarDeriv = derivOf('L = Lebesgue(support = interval(0.0, 4.0))', 'L').d;
  assert.equal(boxDeriv.kind, 'lebesguebox', 'the 1-cartprod spelling is a box');
  assert.equal(scalarDeriv.kind, 'sample', 'the bare interval keeps the scalar kind');
  for (const d of [boxDeriv, scalarDeriv]) {
    assert.ok(Math.abs(d.logTotalmass - Math.log(4)) < 1e-15,
      'logTotalmass ' + d.logTotalmass + ' ≠ log 4');
  }
  // The density agrees too — scored through each spelling's own variate
  // (a length-1 array for the box, a bare scalar for the interval).
  const boxLd = await scoreAt(
    'P = normalize(Lebesgue(support = cartprod(interval(0.0, 4.0))))', 'P', [1.5], 64);
  const scalarCtx = buildCtx(
    'P = normalize(Lebesgue(support = interval(0.0, 4.0)))\nld = logdensityof(P, 1.5)\n', 64);
  const scalarLd = (await scalarCtx.getMeasure('ld')).samples[0];
  const want = -Math.log(4);
  assert.ok(Math.abs(boxLd - want) < 1e-12, '1-axis box logpdf ' + boxLd + ' ≠ −log 4');
  assert.ok(Math.abs(scalarLd - want) < 1e-12, 'scalar logpdf ' + scalarLd + ' ≠ −log 4');
  assert.ok(Math.abs(boxLd - scalarLd) < 1e-12,
    'the two spellings must score identically: ' + boxLd + ' vs ' + scalarLd);
});

// =====================================================================
// The WHOLE-VARIATE weight reading, at one axis as much as above it.
//
// §06 `weighted` states the arity rule: "A one-parameter weight receives the
// variate whole. If the variate is a k-element array with k >= 2, a weight of
// exactly k scalar parameters instead receives one component per parameter, in
// order". The per-axis form therefore needs k >= 2, and at ONE axis only the
// whole-variate reading applies — the variate there being the length-1 VECTOR
// §03/§07 give a one-component cartprod, per the identity test above.
//
// So `f(v) = v[1] * v[1]` is the spelling of a quadratic weight over a 1-axis
// box, and `f(x) = x * x` is refused rather than read as its element: §06 also
// requires the weight to be "a non-negative weight", and `x * x` with `x` the
// vector is vector-valued, a proven violation typeinfer can see.
//
// ORACLES. scipy.integrate.quad / dblquad, to 1e-13, agreeing with the hand
// derivations stated at each test:
//   1-axis [0,2],   f = v₁²        Z = 8/3,  logpdf([1.5])      = -0.16989903679539747
//   2-axis [0,2]×[0,3], f = v₁²v₂²  Z = 24,   logpdf([1.5, 2.0]) = -0.9808292530117262
// =====================================================================

test('a weighted 1-axis box integrates to its closed form', async () => {
  // Z = ∫₀² v₁² dv = 8/3, so the normalized density at v = [1.5] is
  // 1.5² / (8/3) = 2.25 · 3/8 = 0.84375. Monte Carlo over the box atoms.
  const src = `L = Lebesgue(support = cartprod(interval(0.0, 2.0)))
f(v) = v[1] * v[1]
P = normalize(weighted(f, L))
`;
  assertRel(await scoreAt(src, 'P', [1.5], MC_N), 0.84375, MC_TOL, '1-axis weighted box');
});

test('a weighted 1-axis box carries the closed-form total mass', async () => {
  // ∫₀² v₁² dv = 8/3 (scipy.integrate.quad: 2.6666666666666665). The estimator
  // is volume · Ê[f] over the box's own atoms, whose relative standard error
  // here is 0.30 % at N = 65536, so 2 % is a ~6σ band.
  const { ctx } = makeMatCtx(`L = Lebesgue(support = cartprod(interval(0.0, 2.0)))
f(v) = v[1] * v[1]
M = weighted(f, L)
`, { sampleCount: 65536 });
  const m: any = await ctx.getMeasure('M');
  const Z = Math.exp(m.logTotalmass);
  assert.ok(Math.abs(Z - 8 / 3) / (8 / 3) < 0.02, 'totalmass ' + Z + ' ≉ 8/3');
  assert.deepEqual(m.dims, [1], 'the 1-axis box keeps its length-1 vector atoms');
});

test('a SCALAR-form weight over a 1-axis box is refused, naming the vector reading', () => {
  // `f(x) = x * x` denotes a vector-valued function once `x` is the length-1
  // vector §06 hands a one-parameter weight, and §06 requires a non-negative
  // real weight. Refuse rather than silently read `x` as the element: that
  // scores a different function, and the two readings part company the moment
  // the body stops being elementwise.
  const proc = processSource(`L = Lebesgue(support = cartprod(interval(0.0, 2.0)))
f(x) = x * x
P = normalize(weighted(f, L))
`);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1, 'expected exactly one error, got '
    + JSON.stringify(proc.diagnostics));
  assert.match(errs[0].message, /variate WHOLE/);
  assert.match(errs[0].message, /array of real \(length 1\)/,
    'the message must name the variate type it is talking about');
  assert.match(errs[0].message, /v\[1\]/, 'the message must name the vector spelling');
  assert.match(errs[0].message, /bare `interval\(\.\.\.\)`/,
    'the message must name the scalar-variate alternative');
  assert.ok(errs[0].loc, 'the refusal must be located');
});

test('the scalar-form refusal fires above one axis too, without the interval hint', () => {
  // Nothing about the refusal is special to k = 1 — a one-parameter weight
  // receives the whole variate at every k. Above one axis there is no scalar
  // spelling of the same measure, so the message offers only the vector form.
  const proc = processSource(`L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))
f(x) = x * x
P = normalize(weighted(f, L))
`);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1, JSON.stringify(proc.diagnostics));
  assert.match(errs[0].message, /variate WHOLE/);
  assert.doesNotMatch(errs[0].message, /bare `interval/,
    'the scalar-variate hint is only correct at one axis');
});

test('the same weight over a BARE interval keeps its scalar reading', () => {
  // The refusal above is about the VARIATE, not about the spelling `x * x`:
  // `Lebesgue(support = interval(...))` has a scalar variate, so the same
  // weight body is a scalar function of it and passes untouched.
  const proc = processSource(`L = Lebesgue(support = interval(0.0, 2.0))
f(x) = x * x
P = normalize(weighted(f, L))
`);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
});

test('a WRAPPED scalar base still binds the weight parameter to a scalar', async () => {
  // The box search descends through the reweighting chain, so it has to say
  // "no box" for a wrapper over a scalar Lebesgue as surely as for a bare one.
  // Exact, no Monte Carlo: logdensityof = log f(1.5) + log 2 + log 1 =
  // log(2.25 · 2) = log 4.5, the base being Lebesgue (density ≡ 1).
  const ctx = buildCtx(`L = Lebesgue(support = interval(0.0, 2.0))
S = logweighted(0.6931471805599453, L)
f(x) = x * x
M = weighted(f, S)
ld = logdensityof(M, 1.5)
`, 16);
  const got = (await ctx.getMeasure('ld')).samples[0];
  assert.ok(Math.abs(got - Math.log(4.5)) < 1e-12, 'logdensityof ' + got + ' ≠ log 4.5');
});

test('a WRAPPED box base is still found through the reweighting chain', async () => {
  // The twin of the test above: the same wrapper, a box underneath. The search
  // has to descend past the log-weight argument and report the box's 2 axes, so
  // the weight parameter binds to the whole 2-vector. Exact: log f([1.5, 2.0])
  // + log 2 = log(3 · 2) = log 6.
  const ctx = buildCtx(`L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))
S = logweighted(0.6931471805599453, L)
f(v) = v[1] * v[2]
M = weighted(f, S)
ld = logdensityof(M, [1.5, 2.0])
`, 16);
  const got = (await ctx.getMeasure('ld')).samples[0];
  assert.ok(Math.abs(got - Math.log(6)) < 1e-12, 'logdensityof ' + got + ' ≠ log 6');
});

test('a one-parameter weight over a 2-axis box scores its closed form', async () => {
  // The k >= 2 twin of the 1-axis case, same reading: ∫₀²∫₀³ v₁²v₂² = 24
  // (scipy.integrate.dblquad: 24.0), so the density at [1.5, 2.0] is
  // 2.25 · 4 / 24 = 0.375.
  const src = `L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))
f(v) = v[1] * v[1] * v[2] * v[2]
P = normalize(weighted(f, L))
`;
  assertRel(await scoreAt(src, 'P', [1.5, 2.0], MC_N), 0.375, MC_TOL, '2-axis 1-param weight');
});

// Self-normalized weighted mean of axis `j` over a weighted box measure — the
// first moment of the measure `normalize(M)` denotes, read off M's own atoms.
function axisMean(m: any, j: number, k: number): number {
  const N = m.logWeights.length;
  let sw = 0, sx = 0;
  for (let i = 0; i < N; i++) {
    const w = Math.exp(m.logWeights[i]);
    sw += w; sx += w * m.samples[i * k + j];
  }
  return sx / sw;
}

test('the sampled moments of a weighted box match scipy quadrature', async () => {
  // The SAMPLING route, against the same oracles the density route is pinned
  // to. First moments of the normalized measure, by scipy.integrate:
  //   1-axis [0,2],       f = v₁²    → E[v₁] = 1.5
  //   2-axis [0,2]×[0,3], f = v₁²v₂² → E[v₁] = 1.5, E[v₂] = 2.25
  // Tolerances are 6σ of the self-normalized estimator, measured at
  // N = 65536 over 200 replicates: 0.011 / 0.013 / 0.021.
  const { ctx: c1 } = makeMatCtx(`L = Lebesgue(support = cartprod(interval(0.0, 2.0)))
f(v) = v[1] * v[1]
M = weighted(f, L)
`, { sampleCount: 65536 });
  const m1: any = await c1.getMeasure('M');
  assert.ok(Math.abs(axisMean(m1, 0, 1) - 1.5) < 0.011,
    '1-axis E[v1] ' + axisMean(m1, 0, 1) + ' ≉ 1.5');

  const { ctx: c2 } = makeMatCtx(`L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))
f(v) = v[1] * v[1] * v[2] * v[2]
M = weighted(f, L)
`, { sampleCount: 65536 });
  const m2: any = await c2.getMeasure('M');
  assert.ok(Math.abs(axisMean(m2, 0, 2) - 1.5) < 0.013,
    '2-axis E[v1] ' + axisMean(m2, 0, 2) + ' ≉ 1.5');
  assert.ok(Math.abs(axisMean(m2, 1, 2) - 2.25) < 0.021,
    '2-axis E[v2] ' + axisMean(m2, 1, 2) + ' ≉ 2.25');
});

test('weighted(f, Lebesgue(box)) totalmass is ∫ f over the box', async () => {
  // Closed form: ∫₀²∫₀³ xy dy dx = (2²/2)·(3²/2) = 2 · 4.5 = 9. The engine
  // reports area · Ê[f], a Monte-Carlo mass whose relative standard error
  // over this box is ~0.4 % at N = 65536; 3 % is a loose ~7σ band.
  const { ctx } = makeMatCtx(`L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))
f(x, y) = x * y
M = weighted(f, L)
`, { sampleCount: 65536 });
  const m: any = await ctx.getMeasure('M');
  const rel = Math.abs(Math.exp(m.logTotalmass) - 9) / 9;
  assert.ok(rel < 0.03,
    'totalmass ' + Math.exp(m.logTotalmass) + ' ≉ 9 (rel ' + rel + ')');
  assert.deepEqual(m.dims, [2], 'weighted must preserve the vector-atom shape');
});

// The three closed forms below over the SAME box [0,2]×[0,3], all derived by
// hand and confirmed symbolically, with the exact Monte-Carlo relative
// standard error of `volume · Ê[f]` at N = 65536 (σ(f)/E[f]/√N):
//
//   f          Z    E[f]   Var[f]     rel-se @ 65536
//   x·y        9    3/2    7/4        0.34 %
//   x²·y²     24    4      896/25     0.58 %
//   (x+y)²    44    22/3   1288/45    0.28 %
//
// Tolerances are 6·rel-se, so each is a genuine ~6σ band on the estimator: it
// passes on an unbiased estimator and fails on a bias of ~2 % or worse. This is
// what the estimator IS — a plain Monte-Carlo mean over the box's own atoms —
// not a quadrature, so a tighter band would flake.
const BOX_2x3 = 'L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))\n';

async function boxMass(weightDecl: string, N: number, seed?: number) {
  const { ctx } = makeMatCtx(BOX_2x3 + weightDecl + 'M = weighted(f, L)\n',
    seed == null ? { sampleCount: N } : { sampleCount: N, rootSeed: seed });
  const m: any = await ctx.getMeasure('M');
  return Math.exp(m.logTotalmass);
}

test('a SEPARABLE quadratic weight over a box integrates to its closed form', async () => {
  // ∫₀²∫₀³ x²y² dy dx = (2³/3)·(3³/3) = (8/3)·9 = 24.
  const Z = await boxMass('f(x, y) = x * x * y * y\n', 65536);
  const rel = Math.abs(Z - 24) / 24;
  assert.ok(rel < 0.035, 'totalmass ' + Z + ' ≉ 24 (rel ' + rel + ')');
});

test('a NON-SEPARABLE weight over a box integrates to its closed form', async () => {
  // ∫₀³ (x+y)² dy = ((x+3)³ − x³)/3, so
  // ∫₀² ((x+3)³ − x³)/3 dx = [(x+3)⁴ − x⁴]/12 |₀² = (625−16−81)/12 = 528/12 = 44.
  // (x+y)² is not a product of per-axis factors, so it exercises the estimator
  // rather than a per-axis rule that a tensor product would integrate exactly.
  const Z = await boxMass('f(x, y) = (x + y) * (x + y)\n', 65536);
  const rel = Math.abs(Z - 44) / 44;
  assert.ok(rel < 0.017, 'totalmass ' + Z + ' ≉ 44 (rel ' + rel + ')');
});

test('the box mass estimator is UNBIASED — its seed ensemble mean is the closed form', async () => {
  // The single-seed tolerances above cannot separate "noisy" from "biased".
  // Average 24 independent seeds at N = 16384: the per-replicate relative
  // standard error for x·y is 0.34 %·2 = 0.688 %, so the ensemble mean's is
  // 0.688 %/√24 = 0.14 %. A 0.5 % band is 3.6σ — it passes on an unbiased
  // estimator and FAILS the ~0.5 % systematic bias this test was written to
  // rule out. Every replicate uses its own seed, so a numeric-rootKey
  // regression (which collapses all seeds onto one stream) shows up here as a
  // zero-variance ensemble and trips the spread assertion below.
  const S = 24, N = 16384;
  const zs: number[] = [];
  for (let s = 0; s < S; s++) zs.push(await boxMass('f(x, y) = x * y\n', N, 1000 + s * 7919));
  const mean = zs.reduce((a, b) => a + b, 0) / S;
  const sd = Math.sqrt(zs.reduce((a, z) => a + (z - mean) ** 2, 0) / (S - 1));
  assert.ok(Math.abs(mean - 9) / 9 < 0.005,
    'ensemble mean ' + mean + ' is more than 0.5 % from the closed form 9');
  // The replicate spread must be the genuine 1/√N Monte-Carlo error (0.688 %
  // of 9 = 0.062 at this N), not zero (one collapsed stream) and not wild.
  assert.ok(sd > 0.02 && sd < 0.15,
    'replicate sd ' + sd + ' is not the expected ~0.062 Monte-Carlo spread');
});

test('a ctx that passes rootKey as a bare number still varies with its seed', async () => {
  // `foldIn` indexes its key as key[0]/key[1], so a NUMBER rootKey folds in
  // [NaN, NaN] → [0, 0]: the seed is discarded and every seed draws the same
  // stream. `_ensureRootKey` normalises a numeric rootKey through
  // `keyFromSeed`. Without that, two different seeds give bit-identical masses
  // and any seed-robustness claim in a ctx of this shape is vacuous — which is
  // how a plain Monte-Carlo mass got mistaken for a biased deterministic one.
  const src = BOX_2x3 + 'f(x, y) = x * y\nM = weighted(f, L)\n';
  const massAt = async (seed: number) => {
    const proc = processSource(src);
    const built = orchestrator.buildDerivations(proc.bindings);
    const w = createWorkerHandler();
    w.handle({ type: 'init', seed });
    const cache = new Map();
    const ctx: any = {
      derivations: built.derivations, bindings: built.bindings,
      fixedValues: built.fixedValues || new Map(), sampleCount: 4096,
      rootKey: seed, rootSeed: seed,
      getMeasure: (n: string) => {
        if (cache.has(n)) return cache.get(n);
        const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
      },
      sendWorker: (m: any) => Promise.resolve(w.handle(m)),
    };
    return (await ctx.getMeasure('M')).logTotalmass;
  };
  assert.notEqual(await massAt(7), await massAt(1),
    'a numeric rootKey collapsed both seeds onto one stream');
});

test('a CONSTANT weight over a box scales the mass exactly', async () => {
  // Spec §06: dν = f · dM, so a constant f gives totalmass = f · volume with
  // no Monte-Carlo error at all — the estimator never enters. 2 · (2·3) = 12,
  // i.e. logTotalmass = log 2 + log 6 = log 12.
  const { ctx } = makeMatCtx(`L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))
M = weighted(2.0, L)
`, { sampleCount: 256 });
  const m: any = await ctx.getMeasure('M');
  assert.ok(Math.abs(m.logTotalmass - Math.log(12)) < 1e-12,
    'logTotalmass ' + m.logTotalmass + ' ≠ log 12');
  assert.deepEqual(m.dims, [2], 'a constant reweighting must preserve the vector-atom shape');
});

test('logweighted over a box adds the log-weight in log space', async () => {
  // Same measure by two spellings (spec §06: dν = exp(g) · dM against
  // dν = f · dM), so logweighted(log 2, L) and weighted(2, L) must agree
  // exactly. Oracle is the closed form log 12 either way.
  const { ctx } = makeMatCtx(`L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))
g(x, y) = 0.6931471805599453
M = logweighted(g, L)
`, { sampleCount: 4096 });
  const m: any = await ctx.getMeasure('M');
  // A constant log-weight makes the empirical mean exact: logSumExp of N
  // copies of (−log N + log 2) is exactly log 2, so no MC error enters.
  assert.ok(Math.abs(m.logTotalmass - Math.log(12)) < 1e-9,
    'logTotalmass ' + m.logTotalmass + ' ≠ log 12');
  assert.deepEqual(m.dims, [2]);
});

test('a weight that goes negative over the box contributes zero mass there', async () => {
  // Spec §06 requires a NON-NEGATIVE weight. Where f goes negative the engine
  // treats the atom as zero mass (−∞ log-weight) rather than corrupting the
  // total, and warns. f(x, y) = x − 1 over [0,2]×[0,1] is negative on exactly
  // half the box, so the retained mass is ∫₁²∫₀¹ (x−1) dy dx = 1/2.
  const warnings: string[] = [];
  const realWarn = console.warn;
  console.warn = (...a: any[]) => { warnings.push(a.join(' ')); };
  let m: any;
  try {
    const { ctx } = makeMatCtx(`L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 1.0)))
f(x, y) = x - 1.0
M = weighted(f, L)
`, { sampleCount: 65536 });
    m = await ctx.getMeasure('M');
  } finally {
    console.warn = realWarn;
  }
  const rel = Math.abs(Math.exp(m.logTotalmass) - 0.5) / 0.5;
  assert.ok(rel < 0.03, 'retained mass ' + Math.exp(m.logTotalmass) + ' ≉ 0.5 (rel ' + rel + ')');
  assert.ok(warnings.some((w) => /negative off-support weight/.test(w)),
    'expected a negative-weight warning, got ' + JSON.stringify(warnings));
});

test('a weight-function arity that mismatches the box dimension is refused', () => {
  // Refuse-don't-mislower: a 3-param weight over a 2-axis box has no
  // coordinate for its third parameter.
  const { d } = derivOf(`L = Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 1.0)))
f(x, y, z) = x * y * z
M = weighted(f, L)
`, 'M');
  assert.equal(d, undefined, 'arity mismatch must not classify');
});
