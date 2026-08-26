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
    rootKey: seed, rootSeed: seed, marginalizationCount: 32,
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

test('a single-component positional cartprod is the 1-D scalar case', () => {
  // §03: "single-component cartprod is the component itself" — so it must
  // NOT become a 1-axis vector-variate box.
  const { d } = derivOf('L = Lebesgue(support = cartprod(interval(0.0, 4.0)))', 'L');
  assert.equal(d.kind, 'sample');
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

test('cartpow box dimension is bounded below by 2 and above by the axis cap', () => {
  // k = 1 declines. Note the ASYMMETRY with cartprod, which is deliberate:
  // §03 collapses a one-component cartprod ("single-component cartprod is the
  // component itself") but says no such thing for cartpow — `cartpow(reals, 1)`
  // is R¹, a one-element ARRAY, not a scalar. So this is left underivable
  // rather than quietly treated as the scalar interval case.
  assert.equal(derivOf('L = Lebesgue(support = cartpow(interval(0.0, 1.0), 1))', 'L').d,
    undefined, 'cartpow(interval, 1) is a 1-array, not a scalar — declines');
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
  assert.ok(warnings.some((w) => /negative weight sample/.test(w)),
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
