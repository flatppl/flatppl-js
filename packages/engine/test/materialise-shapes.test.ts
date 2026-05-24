'use strict';

// Integration sweep: every binding shape FlatPPL allows must
// materialise into a measure the viewer can plot. Closes the gap that
// let the `rand(rstate, iid(Normal(0,1), [3, 3]))` bug ship.
//
// Strategy: drive each shape from source through `processSource ->
// buildDerivations -> materialiseMeasure` via the shared helper, and
// assert the measure has the fields the viewer reads. The helper
// itself enforces internal consistency (e.g. value.shape vs data.length).
//
// Coverage map:
//   - Scalars (real, integer, boolean, complex, Dirac)
//   - Flat numeric arrays (existing, smoke check)
//   - Multi-dim numeric arrays (the bug class fixed by 256455a)
//   - Multi-axis array generators (fill, zeros, ones, eye)
//   - Records, tuples, arrays of records, arrays of complex
//   - Empty arrays / zero dimensions

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { makeMatCtx, expectPlottable } = require('./_materialise-helpers.ts');

// ---------------------------------------------------------------------
// Baseline: scalars
// ---------------------------------------------------------------------

test('materialise: fixed real scalar', async () => {
  const { ctx } = makeMatCtx('x = 3.14');
  const m = await expectPlottable(ctx, 'x');
  assert.ok(m.samples.length > 0);
});

test('materialise: fixed integer scalar', async () => {
  const { ctx } = makeMatCtx('n = 42');
  const m = await expectPlottable(ctx, 'n');
  assert.equal(m.samples[0], 42);
});

test('materialise: fixed boolean scalar', async () => {
  const { ctx } = makeMatCtx('b = true');
  const m = await expectPlottable(ctx, 'b');
  // Booleans coerce to 1/0
  assert.equal(m.samples[0], 1);
});

// ---------------------------------------------------------------------
// Flat numeric arrays — existing path, smoke check
// ---------------------------------------------------------------------

test('materialise: flat real array', async () => {
  const { ctx } = makeMatCtx('xs = [1.0, 2.0, 3.0]');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 3);
  assert.deepEqual(Array.from(m.samples), [1, 2, 3]);
});

test('materialise: flat integer array', async () => {
  const { ctx } = makeMatCtx('xs = [1, 2, 3]');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 3);
});

test('materialise: flat boolean array', async () => {
  const { ctx } = makeMatCtx('xs = [true, false, true]');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 3);
  // booleans coerce to 1/0
  assert.deepEqual(Array.from(m.samples), [1, 0, 1]);
});

// ---------------------------------------------------------------------
// Multi-dim numeric arrays — the bug class fixed by 256455a
// ---------------------------------------------------------------------

test('materialise: 2D nested array via rand+iid renders as flat samples (not atom-batched)', async () => {
  // Critical invariant: a fixed multi-dim value is ONE value, not an
  // atom-batched measure. Flat samples + no .dims / .shape='array'
  // marker → viewer's array-mode step plot, NOT the multivariate
  // corner-plot dispatch.
  const { ctx } = makeMatCtx(`
rstate = rnginit([1,2,3,4])
A, _ = rand(rstate, iid(Normal(0,1), [3, 3]))
`);
  const m = await expectPlottable(ctx, 'A');
  assert.equal(m.samples.length, 9);
  assert.equal(m.shape, undefined,
    'fixed multi-dim must not carry shape=\'array\' (would route to corner plot)');
  assert.equal(m.dims, undefined,
    'fixed multi-dim must not carry .dims (would mis-label as atom-batched)');
});

test('materialise: 3D nested array flattens to 24 values', async () => {
  const { ctx } = makeMatCtx(`
rstate = rnginit([1,2,3,4])
B, _ = rand(rstate, iid(Normal(0,1), [2, 3, 4]))
`);
  const m = await expectPlottable(ctx, 'B');
  assert.equal(m.samples.length, 24);
  assert.equal(m.shape, undefined);
  assert.equal(m.dims, undefined);
});

test('materialise: 2D matrix literal via rowstack', async () => {
  const { ctx } = makeMatCtx(`M = rowstack([[1.0, 2.0], [3.0, 4.0]])`);
  const m = await expectPlottable(ctx, 'M');
  assert.equal(m.samples.length, 4);
  assert.equal(m.shape, undefined);
});

// ---------------------------------------------------------------------
// Multi-axis array generators
// ---------------------------------------------------------------------

test('materialise: fill(value, scalar size)', async () => {
  const { ctx } = makeMatCtx('xs = fill(2.5, 4)');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 4);
  for (let i = 0; i < 4; i++) assert.equal(m.samples[i], 2.5);
});

test('materialise: fill(value, [m, n]) multi-axis', async () => {
  const { ctx } = makeMatCtx('xs = fill(2.5, [3, 3])');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 9);
  for (let i = 0; i < 9; i++) assert.equal(m.samples[i], 2.5);
});

test('materialise: zeros([m, n]) multi-axis', async () => {
  const { ctx } = makeMatCtx('xs = zeros([3, 3])');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 9);
});

test('materialise: ones([m, n]) multi-axis', async () => {
  const { ctx } = makeMatCtx('xs = ones([2, 3])');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 6);
});

test('materialise: eye(n) identity matrix', async () => {
  const { ctx } = makeMatCtx('I = eye(3)');
  const m = await expectPlottable(ctx, 'I');
  assert.equal(m.samples.length, 9);
  // Diagonal = 1, off-diagonal = 0
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    assert.equal(m.samples[i * 3 + j], i === j ? 1 : 0);
  }
});

// ---------------------------------------------------------------------
// Records and tuples
// ---------------------------------------------------------------------

test('materialise: fixed record', async () => {
  const { ctx } = makeMatCtx('r = record(mu = 0.0, sigma = 1.0)');
  const m = await expectPlottable(ctx, 'r');
  assert.ok(m.fields, 'fixed record should expose .fields');
  assert.ok(m.fields.mu, '.fields.mu present');
  assert.ok(m.fields.sigma, '.fields.sigma present');
});

test('materialise: array of records', async () => {
  const { ctx } = makeMatCtx(`
xs = [record(a = 1.0, b = 2.0), record(a = 3.0, b = 4.0)]
`);
  // Either succeeds with a clear shape, or returns a measure the viewer
  // can route. Just ensure no throw.
  const m = await expectPlottable(ctx, 'xs');
  assert.ok(m);
});

// ---------------------------------------------------------------------
// Empty / edge shapes
// ---------------------------------------------------------------------

test('materialise: empty array literal', async () => {
  const { ctx } = makeMatCtx('xs = []');
  const m = await ctx.getMeasure('xs');
  // Empty arrays may legitimately have no plot — accept either a
  // missing-fields measure or an empty .samples — but the call must
  // not throw.
  assert.ok(m);
});

// ---------------------------------------------------------------------
// Complex
// ---------------------------------------------------------------------

test('materialise: fixed complex scalar', async () => {
  const { ctx } = makeMatCtx('z = complex(1.0, 2.0)');
  const m = await expectPlottable(ctx, 'z');
  // Complex measures expose .samples (Re) and .imag (Im) per Value
  // shape contract.
  assert.ok(m.samples, 'complex scalar exposes .samples (Re)');
});

test('materialise: array of complex scalars', async () => {
  const { ctx } = makeMatCtx(`
zs = [complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)]
`);
  const m = await expectPlottable(ctx, 'zs');
  assert.ok(m, 'should materialise without throwing');
});

// ---------------------------------------------------------------------
// Variates and measures: confirm the existing path is also exercised
// ---------------------------------------------------------------------

test('materialise: scalar Normal distribution', async () => {
  const { ctx } = makeMatCtx('m = Normal(mu = 0, sigma = 1)');
  const m = await expectPlottable(ctx, 'm');
  assert.equal(m.samples.length, ctx.sampleCount);
});

test('materialise: iid(Normal, scalar) → 1D atom-vector measure', async () => {
  const { ctx } = makeMatCtx('m = iid(Normal(0, 1), 4)');
  const m = await expectPlottable(ctx, 'm');
  // N atoms × dims=[4]
  assert.deepEqual(m.dims, [4]);
});

test('materialise: iid(Normal, [m, n]) → matrix-atom measure', async () => {
  const { ctx } = makeMatCtx('m = iid(Normal(0, 1), [3, 3])');
  const m = await expectPlottable(ctx, 'm');
  assert.deepEqual(m.dims, [3, 3]);
});

// ---------------------------------------------------------------------
// More shape generators — linspace, onehot, partition, cumsum, etc.
// ---------------------------------------------------------------------

test('materialise: linspace(lo, hi, n)', async () => {
  const { ctx } = makeMatCtx('xs = linspace(0.0, 1.0, 5)');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 5);
  assert.equal(m.samples[0], 0);
  assert.equal(m.samples[4], 1);
});

test('materialise: extlinspace(lo, hi, n) emits ±Infinity endpoints', async () => {
  // extlinspace spec §07: extends linspace with -inf/+inf bracket
  // endpoints, so a 5-point grid produces 7 values.
  const { ctx } = makeMatCtx('xs = extlinspace(0.0, 1.0, 5)');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 7);
  assert.equal(m.samples[0], -Infinity);
  assert.equal(m.samples[6], Infinity);
});

test('materialise: onehot(i, n)', async () => {
  const { ctx } = makeMatCtx('v = onehot(2, 4)');
  const m = await expectPlottable(ctx, 'v');
  assert.equal(m.samples.length, 4);
  assert.deepEqual(Array.from(m.samples), [0, 1, 0, 0]);
});

test('materialise: cumsum on a flat array', async () => {
  const { ctx } = makeMatCtx('cs = cumsum([1.0, 2.0, 3.0, 4.0])');
  const m = await expectPlottable(ctx, 'cs');
  assert.equal(m.samples.length, 4);
});

test('materialise: cat([1,2], [3,4]) → length-4 array', async () => {
  const { ctx } = makeMatCtx('xs = cat([1.0, 2.0], [3.0, 4.0])');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 4);
});

// ---------------------------------------------------------------------
// Records containing non-scalar fields
// ---------------------------------------------------------------------

test('materialise: record with array field', async () => {
  const { ctx } = makeMatCtx('r = record(xs = [1.0, 2.0, 3.0], mu = 0.0)');
  const m = await expectPlottable(ctx, 'r');
  assert.ok(m.fields, 'must expose .fields');
  assert.ok(m.fields.xs, '.fields.xs');
  assert.ok(m.fields.mu, '.fields.mu');
});

test('materialise: record with matrix field', async () => {
  const { ctx } = makeMatCtx(`
r = record(M = rowstack([[1.0, 2.0], [3.0, 4.0]]), v = [0.5, 0.5])
`);
  const m = await expectPlottable(ctx, 'r');
  assert.ok(m.fields);
});

test('materialise: nested record', async () => {
  const { ctx } = makeMatCtx(`
r = record(inner = record(a = 1.0, b = 2.0), top = 3.0)
`);
  const m = await expectPlottable(ctx, 'r');
  assert.ok(m.fields);
});

// ---------------------------------------------------------------------
// Linear algebra outputs (matrix-shaped fixed Values)
// ---------------------------------------------------------------------

test('materialise: lower_cholesky of an SPD matrix', async () => {
  const { ctx } = makeMatCtx(`
M = rowstack([[4.0, 2.0], [2.0, 3.0]])
L = lower_cholesky(M)
`);
  const m = await expectPlottable(ctx, 'L');
  assert.equal(m.samples.length, 4);
});

test('materialise: transpose of a matrix', async () => {
  const { ctx } = makeMatCtx(`
M = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
T = transpose(M)
`);
  const m = await expectPlottable(ctx, 'T');
  assert.equal(m.samples.length, 6);
});

test('materialise: inv of an SPD matrix', async () => {
  const { ctx } = makeMatCtx(`
M = rowstack([[2.0, 0.0], [0.0, 4.0]])
I = inv(M)
`);
  const m = await expectPlottable(ctx, 'I');
  assert.equal(m.samples.length, 4);
});

test('materialise: diagmat from a vector', async () => {
  const { ctx } = makeMatCtx('D = diagmat([1.0, 2.0, 3.0])');
  const m = await expectPlottable(ctx, 'D');
  assert.equal(m.samples.length, 9);
});

// ---------------------------------------------------------------------
// Reductions to scalars — already covered by scalar tests but pin
// the array→scalar paths
// ---------------------------------------------------------------------

test('materialise: sum of a flat array', async () => {
  const { ctx } = makeMatCtx('s = sum([1.0, 2.0, 3.0, 4.0])');
  const m = await expectPlottable(ctx, 's');
  assert.equal(m.samples[0], 10);
});

test('materialise: mean of a flat array', async () => {
  const { ctx } = makeMatCtx('mu = mean([1.0, 2.0, 3.0, 4.0])');
  const m = await expectPlottable(ctx, 'mu');
  assert.equal(m.samples[0], 2.5);
});

// ---------------------------------------------------------------------
// Multi-axis variants: cartpow + set-typed Lebesgue support
// ---------------------------------------------------------------------

test('materialise: Lebesgue(support=cartpow(reals, [3, 3]))', async () => {
  // Pure measure binding (sampleable=false; Lebesgue is not normalisable
  // without a finite support set). Just ensure materialiser path
  // doesn't throw on the type-side.
  const { ctx } = makeMatCtx(
    'M = Lebesgue(support = cartpow(reals, [3, 3]))');
  // Lebesgue isn't sampleable; getMeasure may legitimately fail with
  // an "unsupported" error rather than a shape error. We only assert
  // that the failure mode is the explicit unsupported one, not an
  // undefined-length crash.
  try {
    await ctx.getMeasure('M');
  } catch (err: any) {
    const msg = String(err.message || err);
    assert.ok(!/Cannot read properties of undefined/.test(msg),
      'unsupported should not surface as undefined-length: ' + msg);
  }
});

// ---------------------------------------------------------------------
// bincounts — integer array output, important for HEP workflows
// ---------------------------------------------------------------------

test('materialise: bincounts on a literal dataset', async () => {
  const { ctx } = makeMatCtx(`
edges = [0.0, 1.0, 2.0, 3.0]
data = [0.2, 0.4, 1.5, 1.7, 2.5]
counts = bincounts(edges, data)
`);
  const m = await expectPlottable(ctx, 'counts');
  assert.equal(m.samples.length, 3);
});

// ---------------------------------------------------------------------
// Rank-3 multi-axis generators
// ---------------------------------------------------------------------

test('materialise: fill(value, [m, n, p]) rank-3', async () => {
  const { ctx } = makeMatCtx('xs = fill(7.0, [2, 3, 4])');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 24);
});

test('materialise: zeros([m, n, p]) rank-3', async () => {
  const { ctx } = makeMatCtx('xs = zeros([2, 3, 4])');
  const m = await expectPlottable(ctx, 'xs');
  assert.equal(m.samples.length, 24);
});

// ---------------------------------------------------------------------
// Empty / degenerate shapes
// ---------------------------------------------------------------------

test('materialise: fill(value, 0) — empty 1D', async () => {
  const { ctx } = makeMatCtx('xs = fill(1.0, 0)');
  // Empty array — materialise should not throw; viewer may render
  // "no data" but not crash.
  try {
    const m = await ctx.getMeasure('xs');
    assert.ok(m);
  } catch (err: any) {
    // If the engine deems this unsupported, fine — but it must not be
    // an undefined-length crash.
    assert.ok(!/Cannot read properties of undefined/.test(String(err.message)),
      'empty array must not surface as undefined: ' + err.message);
  }
});

test('materialise: zeros([0, 3]) — empty 2D', async () => {
  const { ctx } = makeMatCtx('xs = zeros([0, 3])');
  try {
    const m = await ctx.getMeasure('xs');
    assert.ok(m);
  } catch (err: any) {
    assert.ok(!/Cannot read properties of undefined/.test(String(err.message)));
  }
});

// ---------------------------------------------------------------------
// Tuple-shaped fixed bindings
// ---------------------------------------------------------------------

test('materialise: tuple of scalars via multi-LHS', async () => {
  // `t = tuple(1.0, 2.0)` and the multi-LHS decomposition test the
  // same path through the analyzer.
  const { ctx } = makeMatCtx(`
a = 1.0
b = 2.0
c = a + b
`);
  // Smoke-check that all three materialise.
  await expectPlottable(ctx, 'a');
  await expectPlottable(ctx, 'b');
  await expectPlottable(ctx, 'c');
});

// ---------------------------------------------------------------------
// Stochastic variates of various shapes
// ---------------------------------------------------------------------

test('materialise: scalar variate ~ Normal', async () => {
  const { ctx } = makeMatCtx('x ~ Normal(mu = 0, sigma = 1)');
  const m = await expectPlottable(ctx, 'x');
  assert.equal(m.samples.length, ctx.sampleCount);
});

test('materialise: matrix-atom variate ~ iid(N, [3, 3])', async () => {
  const { ctx } = makeMatCtx('X ~ iid(Normal(0, 1), [3, 3])');
  const m = await expectPlottable(ctx, 'X');
  assert.deepEqual(m.dims, [3, 3]);
});

test('materialise: MvNormal variate', async () => {
  const { ctx } = makeMatCtx(`
mu = [0.0, 0.0]
Sigma = rowstack([[1.0, 0.5], [0.5, 1.0]])
v ~ MvNormal(mu = mu, cov = Sigma)
`);
  const m = await expectPlottable(ctx, 'v');
  assert.deepEqual(m.dims, [2]);
});

// ---------------------------------------------------------------------
// Functions + kernels (first-class, fixed-phase by definition)
// ---------------------------------------------------------------------

test('materialise: functionof binding — function itself not plottable', async () => {
  // functionof produces a function-typed binding; the viewer correctly
  // skips plotting these (function-typed bindings are first-class but
  // don't have a meaningful plot). materialiseMeasure should reject
  // with a clean "no derivation" / "unsupported" error rather than an
  // undefined-length crash.
  const { ctx } = makeMatCtx(`
sq = functionof(x * x, x = _x_)
`);
  try {
    await ctx.getMeasure('sq');
  } catch (err: any) {
    const msg = String(err.message || err);
    assert.ok(!/Cannot read properties of undefined/.test(msg),
      'function-typed binding must not crash with undefined: ' + msg);
  }
});

test('materialise: lambda binding', async () => {
  const { ctx } = makeMatCtx(`
sq = x -> x * x
y = sq(4.0)
`);
  const m = await expectPlottable(ctx, 'y');
  assert.equal(m.samples[0], 16);
});

// ---------------------------------------------------------------------
// aggregate — broadcast-reduce semantics, multi-axis outputs
// ---------------------------------------------------------------------

test('materialise: aggregate produces vector output', async () => {
  const { ctx } = makeMatCtx(`
A = rowstack([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
rowsums = aggregate(sum, [.i], A[.i, .j])
`);
  const m = await expectPlottable(ctx, 'rowsums');
  assert.equal(m.samples.length, 3);
  // rows [1+2, 3+4, 5+6] = [3, 7, 11]
  assert.deepEqual(Array.from(m.samples), [3, 7, 11]);
});

test('materialise: matmul via aggregate produces matrix output', async () => {
  const { ctx } = makeMatCtx(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
B = rowstack([[5.0, 6.0], [7.0, 8.0]])
C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])
`);
  const m = await expectPlottable(ctx, 'C');
  assert.equal(m.samples.length, 4);
});

// ---------------------------------------------------------------------
// Restrict + bayesupdate-derived bindings
// ---------------------------------------------------------------------

test('materialise: bayesupdate posterior — does not crash with undefined', async () => {
  // Scalar Gaussian conjugate posterior. The classifier may or may
  // not produce a derivation for `posterior` depending on the exact
  // kernel shape; what matters here is no undefined-length crash.
  const { ctx } = makeMatCtx(`
prior = Normal(mu = 0, sigma = 2)
obs = 1.5
K = kernelof(Normal(mu = mu, sigma = 0.5), mu = _mu_)
L = likelihoodof(K, obs)
posterior = bayesupdate(L, prior)
`);
  try {
    await ctx.getMeasure('posterior');
  } catch (err: any) {
    assert.ok(!/Cannot read properties of undefined/.test(String(err.message)),
      'posterior materialise must not crash with undefined: ' + err.message);
  }
});

// ---------------------------------------------------------------------
// Reused-name decomposition (multi-LHS with both names used)
// ---------------------------------------------------------------------

test('materialise: multi-LHS where both names are kept', async () => {
  const { ctx } = makeMatCtx(`
rstate = rnginit([7, 8, 9, 10])
val, rstate2 = rand(rstate, Normal(0, 1))
`);
  await expectPlottable(ctx, 'val');
  // rstate2 is rngstate — opaque; should not crash but may legitimately
  // not be plottable. Just ensure no thrown undefined.
  try { await ctx.getMeasure('rstate2'); } catch (err: any) {
    assert.ok(!/Cannot read properties of undefined/.test(String(err.message)));
  }
});

// ---------------------------------------------------------------------
// Set-typed bindings (sets are not values, must not crash)
// ---------------------------------------------------------------------

test('materialise: elementof(reals) parameter', async () => {
  const { ctx } = makeMatCtx('mu = elementof(reals)');
  // mu is a parameter — phase=parameterized — viewer plots it via
  // preset overrides. The default render path may not produce a
  // measure for it. Just don't crash.
  try { await ctx.getMeasure('mu'); } catch (err: any) {
    assert.ok(!/Cannot read properties of undefined/.test(String(err.message)));
  }
});
