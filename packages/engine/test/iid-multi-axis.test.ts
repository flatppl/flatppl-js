'use strict';

// Spec §06: iid(M, size) — `size` is a positive integer (1-D length)
// or a vector of positive integers (multi-axis shape). Spec example:
// `iid(Normal(mu = 0, sigma = 1), [3, 3])` produces 3×3 matrices.
//
// Three engine entry points handle `iid` size:
//   1. classifyIid  (derivations) — unpacks `vector(...)` → dims[]
//   2. inferIid     (typeinfer)   — unpacks `vector(...)` → shape
//   3. matIid       (materialiser)— takes dims[] from the derivation
//   4. walkIid      (sampler)     — must compute prod(size) draws
//                                    and reshape into nested array
//   5. walkIid      (density)     — must compute prod(size) for the
//                                    reduce loop over the footprint
//
// (1)-(3) already handle the vector form. (4) and (5) were
// scalar-only until fixed alongside this test file.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const sampler = require('../sampler.ts');
const rng = require('../rng.ts');
const { processSource } = require('../index.ts');

function makeIid(sizeIR: any) {
  return {
    kind: 'call', op: 'iid',
    args: [
      { kind: 'call', op: 'Normal',
        kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } } },
      sizeIR,
    ],
  };
}
function vec(...vs: number[]) {
  return { kind: 'call', op: 'vector',
    args: vs.map((v: number) => ({ kind: 'lit', value: v })) };
}
function lit(v: number) { return { kind: 'lit', value: v }; }

// ---------------------------------------------------------------------
// traceeval.walk (used by `rand`) — multi-axis shape
// ---------------------------------------------------------------------

test('iid multi-axis: traceeval walk produces a rank-2 matrix Value', () => {
  // Per spec §03 — `iid(Normal, [3, 3])` produces a rank-2 array
  // (matrix), not a vector-of-vectors. The engine returns a shape-
  // explicit Value so downstream `*`/`+`/`-` route through the
  // value-ops shape dispatch (matrix multiplication for `*`).
  const state = rng.stateFromKey([1, 2, 3, 4, 5, 6, 7, 8]);
  const r = sampler.walk(state, makeIid(vec(3, 3)), {}, {});
  assert.deepEqual(r.value.shape, [3, 3], 'matrix Value with shape [3, 3]');
  assert.equal(r.value.data.length, 9, 'flat Float64Array of length 9');
  for (let i = 0; i < 9; i++) {
    assert.equal(typeof r.value.data[i], 'number');
  }
});

test('iid multi-axis: rank-3 shape produces rank-3 Value', () => {
  const state = rng.stateFromKey([1, 2, 3, 4, 5, 6, 7, 8]);
  const r = sampler.walk(state, makeIid(vec(2, 3, 4)), {}, {});
  assert.deepEqual(r.value.shape, [2, 3, 4]);
  assert.equal(r.value.data.length, 24);
});

test('iid multi-axis: scalar size still produces a flat array (back-compat)', () => {
  const state = rng.stateFromKey([1, 2, 3, 4, 5, 6, 7, 8]);
  const r = sampler.walk(state, makeIid(lit(5)), {}, {});
  assert.equal(r.value.length, 5);
  assert.equal(typeof r.value[0], 'number',
    'scalar-size form remains flat (not wrapped in an extra dim)');
});

test('iid multi-axis: zero-prod size produces an empty Value', () => {
  const state = rng.stateFromKey([1, 2, 3, 4, 5, 6, 7, 8]);
  const r = sampler.walk(state, makeIid(vec(0, 3)), {}, {});
  // prod = 0 — empty matrix. Still a shape-explicit Value.
  assert.deepEqual(r.value.shape, [0, 3]);
  assert.equal(r.value.data.length, 0);
});

test('iid multi-axis: end-to-end `rand(state, iid(N, [3, 3]))` parses + analyses', () => {
  const src = `
rstate = rnginit([1,2,3,4])
A, _ = rand(rstate, iid(Normal(0,1), [3, 3]))
`;
  const ctx = processSource(src);
  const errs = ctx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, 'parses cleanly: ' +
    errs.map((d: any) => d.message).join('; '));
});

test('iid multi-axis: fixed 2D A produces a flat-samples measure (viewer plot path)', async () => {
  // Regression for the bug where the viewer threw
  // "Cannot read properties of undefined (reading 'length')" when
  // plotting A from `A, _ = rand(rstate, iid(Normal(0,1), [3, 3]))`.
  // Root cause: fixedValueToMeasure treated the nested 3x3 numeric
  // array as a 3-tuple of inner arrays (elems-shape) instead of a
  // flat numeric matrix. The viewer's array-mode renderer then read
  // measure.samples (undefined) and blew up on .length.
  const src = `
rstate = rnginit([1,2,3,4])
A, _ = rand(rstate, iid(Normal(0,1), [3, 3]))
`;
  const { orchestrator, materialiser } = require('..');
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: 1024,
    getMeasure:  (n: any) => materialiser.materialiseMeasure(n, ctx),
    sendWorker:  (_: any) => { throw new Error('should not reach worker'); },
  };
  const m = await ctx.getMeasure('A');
  assert.ok(m.samples, 'A measure must expose .samples (viewer reads this)');
  assert.equal(m.samples.length, 9,
    'flat samples of length prod(shape) = 9; got ' + m.samples.length);
  // Fixed-phase multi-dim values flatten to a 1D scalar measure (no
  // atom axis), so the viewer's array-mode step plot renders them
  // honestly rather than mis-dispatching to the corner-plot path.
  assert.equal(m.shape, undefined);
  assert.equal(m.dims, undefined);
});

test('iid multi-axis: fixed 3D A flattens to length-24 samples', async () => {
  const src = `
rstate = rnginit([1,2,3,4])
B, _ = rand(rstate, iid(Normal(0,1), [2, 3, 4]))
`;
  const { orchestrator, materialiser } = require('..');
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: 1024,
    getMeasure:  (n: any) => materialiser.materialiseMeasure(n, ctx),
    sendWorker:  (_: any) => { throw new Error('should not reach worker'); },
  };
  const m = await ctx.getMeasure('B');
  assert.equal(m.samples.length, 24);
  assert.equal(m.shape, undefined);
});

// ---------------------------------------------------------------------
// density.walk — multi-axis size in iid logdensity
// ---------------------------------------------------------------------

test('iid multi-axis: density(iid(N, [3, 3]), x) reduces over prod(size) cells', () => {
  // Sum-of-logpdf over 9 standard-normal cells, all zero. The
  // consume/rest density walker operates on the flattened
  // footprint, so the value here is a 9-element array — what the
  // walker would receive after the surrounding measure-handler
  // (matLogdensityof / matBayesupdate / etc.) flattens the
  // user-facing nested input.
  const density = require('../density.ts');
  const ir = makeIid(vec(3, 3));
  const x = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  // 9 × log(1/sqrt(2π)) ≈ 9 × -0.9189385332
  const expected = 9 * Math.log(1 / Math.sqrt(2 * Math.PI));
  const got = density.logDensity(ir, x, {}, {});
  assert.ok(Math.abs(got - expected) < 1e-10,
    `expected ~${expected}, got ${got}`);
});

// ---------------------------------------------------------------------
// inferIid — multi-axis shape is preserved in the inferred type
// ---------------------------------------------------------------------

test('iid multi-axis: inferred type carries the full rank + per-axis lengths', () => {
  const ctx = processSource('M = iid(Normal(0,1), [3, 3])\n');
  const m = ctx.loweredModule.bindings.get('M');
  assert.ok(m && m.inferredType, 'inferred type present');
  const t = m.inferredType;
  // measure(array(2, [3, 3], real))
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'array');
  assert.equal(t.domain.rank, 2);
  assert.equal(t.domain.shape[0], 3);
  assert.equal(t.domain.shape[1], 3);
});

// ---------------------------------------------------------------------
// cartpow — sister rule from spec §03 sets
// ---------------------------------------------------------------------

test('cartpow multi-axis: cartpow(reals, [3, 3]) type-infers as array(2, [3,3], real)', () => {
  // Use it via Lebesgue(support = cartpow(...)) so the engine flows
  // the set's value type through to a measure binding.
  const ctx = processSource(
    'M = Lebesgue(support = cartpow(reals, [3, 3]))\n');
  const m = ctx.loweredModule.bindings.get('M');
  assert.ok(m && m.inferredType);
  const t = m.inferredType;
  assert.equal(t.kind, 'measure', 'measure type');
  assert.equal(t.domain.kind, 'array');
  assert.equal(t.domain.rank, 2);
  assert.equal(t.domain.shape[0], 3);
  assert.equal(t.domain.shape[1], 3);
});

test('cartpow scalar still infers as rank-1 with that length', () => {
  const ctx = processSource(
    'M = Lebesgue(support = cartpow(reals, 5))\n');
  const m = ctx.loweredModule.bindings.get('M');
  const t = m.inferredType;
  assert.equal(t.domain.rank, 1);
  assert.equal(t.domain.shape[0], 5);
});

// =====================================================================
// classifyIid: size argument can be a binding ref to a fixed-phase
// expression of arbitrary shape, not just a literal or a fold-able
// arithmetic tree. resolveConstant now consults the orchestrator's
// pre-eval fixedValues cache, and buildDerivations re-classifies
// unclassified bindings after pre-eval has populated it. Before this
// fix, `n = lengthof(data)` followed by `iid(Dist, n)` silently
// produced no derivation for the iid binding, cascading to drop any
// downstream bayesupdate / likelihood path.
// =====================================================================

const { orchestrator } = require('../index.ts');

// Trace through alias chains to the underlying iid derivation (the
// lift pass introduces synthetic anonymous bindings for inline
// `iid(...)` calls, so `y` typically alias-points at an anon).
function resolveIidVia(derivations: any, name: any): any {
  let cur = derivations[name];
  let hops = 0;
  while (cur && cur.kind === 'alias' && hops < 16) {
    cur = derivations[cur.from];
    hops += 1;
  }
  return cur;
}

test('classifyIid: size given as binding ref to lengthof() resolves through pre-eval', () => {
  const src = `
counts_data = [2, 3, 7, 6, 4]
n = lengthof(counts_data)
lambda = draw(Gamma(shape = 2.0, rate = 1.0))
y = draw(iid(Poisson(rate = lambda), n))
`;
  const r = processSource(src);
  const built = orchestrator.buildDerivations(r.bindings);
  assert.ok(built.derivations.y, 'y derivation present (missing before n-ref fix)');
  const yDer = resolveIidVia(built.derivations, 'y');
  assert.ok(yDer, 'y traces to an iid derivation via alias hops');
  assert.equal(yDer.kind, 'iid');
  assert.deepEqual(yDer.dims, [5]);
});

test('classifyIid: size given as binding ref to literal integer', () => {
  const src = `
sz = 7
lambda = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(iid(Normal(mu = lambda, sigma = 1.0), sz))
`;
  const r = processSource(src);
  const built = orchestrator.buildDerivations(r.bindings);
  assert.ok(built.derivations.y, 'y derivation present');
  const yDer = resolveIidVia(built.derivations, 'y');
  assert.ok(yDer, 'y derivation present for literal-int binding ref');
  assert.equal(yDer.kind, 'iid');
  assert.deepEqual(yDer.dims, [7]);
});

test('classifyIid: bayesupdate(L, prior) derives end-to-end with n=lengthof(data)', () => {
  // Gamma-Poisson conjugate model. The full pipeline (derivation →
  // pre-eval → cascade-prune) succeeds and the posterior binding
  // emerges as a bayesupdate derivation, ready for the materialiser
  // to score with importance weights.
  const src = `
counts_data = [2, 3, 7, 6, 4]
n = lengthof(counts_data)
lambda = draw(Gamma(shape = 2.0, rate = 1.0))
y = draw(iid(Poisson(rate = lambda), n))
prior = lawof(record(lambda = lambda))
forward_kernel = kernelof(record(y = y), lambda = lambda)
L = likelihoodof(forward_kernel, record(y = counts_data))
posterior = bayesupdate(L, prior)
`;
  const r = processSource(src);
  const built = orchestrator.buildDerivations(r.bindings);
  assert.ok(built.derivations.posterior,
    'posterior derivation present (was missing because iid(...,n) didn\'t classify)');
  assert.equal(built.derivations.posterior.kind, 'bayesupdate');
  assert.equal(built.derivations.posterior.from, 'prior');
});
