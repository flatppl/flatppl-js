'use strict';

// =====================================================================
// Type-mode consume/rest walker (engine-concepts §17.3)
// =====================================================================
//
// `density-prims.staticConsume` and its strict wrapper
// `staticDensityShapeCheck` are the type-mode counterpart to
// `density.ts`'s value-mode consume/rest walker. Given a measure IR
// + a variate type, they return whether the measure's footprint
// matches statically. v0.1 scope: scalar leaves, weighted/normalize
// passthroughs, joint(fields), iid (literal n). Select / positional
// joint / pushfwd stay deferred.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const densityPrims = require('../density-prims.ts');
const T = require('../types.ts');

// Convenience IR builders.
function lit(v: any) { return { kind: 'lit', value: v }; }
function Normal(mu: any, sigma: any) {
  return { kind: 'call', op: 'Normal',
    kwargs: { mu: lit(mu), sigma: lit(sigma) } };
}

// =====================================================================
// Scalar leaves
// =====================================================================

test('static: Normal vs scalar variate type → empty rest', () => {
  const r = densityPrims.staticConsume(Normal(0, 1), T.REAL);
  assert.equal(r.rest, null);
  assert.ok(!r.error);
});

test('static: Normal vs length-1 vector → empty rest', () => {
  const r = densityPrims.staticConsume(Normal(0, 1),
    T.array(1, [1], T.REAL));
  assert.equal(r.rest, null);
});

test('static: Normal vs length-3 vector → rest is length-2', () => {
  const r = densityPrims.staticConsume(Normal(0, 1),
    T.array(1, [3], T.REAL));
  assert.ok(r.rest && r.rest.kind === 'array');
  assert.deepEqual(r.rest.shape, [2]);
});

test('static: Normal vs record → static error', () => {
  const r = densityPrims.staticConsume(Normal(0, 1),
    { kind: 'record', fields: { x: T.REAL } });
  assert.ok(r.error && /scalar leaf/.test(r.error));
});

// =====================================================================
// joint(fields)
// =====================================================================

test('static: joint(x = Normal, y = Normal) vs matching record → empty rest', () => {
  const ir = { kind: 'call', op: 'joint',
    fields: [
      { name: 'x', value: Normal(0, 1) },
      { name: 'y', value: Normal(2, 3) },
    ] };
  const variate = { kind: 'record', fields: { x: T.REAL, y: T.REAL } };
  const r = densityPrims.staticConsume(ir, variate);
  assert.equal(r.rest, null);
  assert.ok(!r.error);
});

test('static: joint(x = Normal) vs record missing x → static error', () => {
  const ir = { kind: 'call', op: 'joint',
    fields: [{ name: 'x', value: Normal(0, 1) }] };
  const variate = { kind: 'record', fields: { y: T.REAL } };
  const r = densityPrims.staticConsume(ir, variate);
  assert.ok(r.error && /missing field/.test(r.error));
});

test('static: joint(fields) vs scalar variate → static error', () => {
  const ir = { kind: 'call', op: 'joint',
    fields: [{ name: 'x', value: Normal(0, 1) }] };
  const r = densityPrims.staticConsume(ir, T.REAL);
  assert.ok(r.error && /record variate/.test(r.error));
});

// =====================================================================
// iid (literal n)
// =====================================================================

test('static: iid(Normal, 3) vs array(3, real) → empty rest', () => {
  const ir = { kind: 'call', op: 'iid', args: [Normal(0, 1), lit(3)] };
  const variate = T.array(1, [3], T.REAL);
  const r = densityPrims.staticConsume(ir, variate);
  assert.equal(r.rest, null);
  assert.ok(!r.error);
});

test('static: iid(Normal, 3) vs array(4, real) → static error (length mismatch)', () => {
  const ir = { kind: 'call', op: 'iid', args: [Normal(0, 1), lit(3)] };
  const variate = T.array(1, [4], T.REAL);
  const r = densityPrims.staticConsume(ir, variate);
  assert.ok(r.error && /3 elements.*4/.test(r.error),
    `expected length-mismatch error, got ${r.error}`);
});

test('static: iid(Normal, 3) vs scalar → static error', () => {
  const ir = { kind: 'call', op: 'iid', args: [Normal(0, 1), lit(3)] };
  const r = densityPrims.staticConsume(ir, T.REAL);
  assert.ok(r.error && /array variate/.test(r.error));
});

test('static: iid with dynamic n stays deferred (passes static check)', () => {
  // Non-literal size (e.g. a ref) → can't pin static length; defer
  // the empty-rest check to runtime, but report no error.
  const ir = { kind: 'call', op: 'iid', args: [Normal(0, 1),
    { kind: 'ref', ns: 'self', name: 'n' }] };
  const variate = T.array(1, ['%dynamic'], T.REAL);
  const err = densityPrims.staticDensityShapeCheck(ir, variate);
  assert.equal(err, null);
});

// =====================================================================
// Positional joint
// =====================================================================

test('static: positional joint(Normal, Normal) vs array(2, real) → empty rest', () => {
  const ir = { kind: 'call', op: 'joint',
    args: [Normal(0, 1), Normal(2, 3)] };
  const variate = T.array(1, [2], T.REAL);
  const r = densityPrims.staticConsume(ir, variate);
  assert.equal(r.rest, null);
  assert.ok(!r.error);
});

test('static: positional joint(Normal, Normal) vs array(3, real) → 1-element rest', () => {
  const ir = { kind: 'call', op: 'joint',
    args: [Normal(0, 1), Normal(2, 3)] };
  const variate = T.array(1, [3], T.REAL);
  const r = densityPrims.staticConsume(ir, variate);
  assert.ok(r.rest && r.rest.kind === 'array');
  assert.deepEqual(r.rest.shape, [1]);
});

test('static: positional joint(Normal, Normal, Normal) vs array(2, real) → static error', () => {
  const ir = { kind: 'call', op: 'joint',
    args: [Normal(0, 1), Normal(2, 3), Normal(4, 5)] };
  const variate = T.array(1, [2], T.REAL);
  const r = densityPrims.staticConsume(ir, variate);
  assert.ok(r.error && /variate exhausted/.test(r.error),
    `expected exhaustion error, got: ${r.error}`);
});

// =====================================================================
// select (mixture / superpose / measure-valued ifelse)
// =====================================================================

test('static: select with matching-footprint branches → empty rest', () => {
  const ir = { kind: 'call', op: 'select',
    branches: [{ ir: Normal(0, 1) }, { ir: Normal(2, 3) }],
    logweights: null };
  const r = densityPrims.staticConsume(ir, T.REAL);
  assert.equal(r.rest, null);
  assert.ok(!r.error);
});

test('static: select with disagreeing footprints → static error', () => {
  // Branch 0: scalar leaf consumes 1 entry off a length-2 vector → rest [1].
  // Branch 1: iid(Normal, 2) consumes both entries → rest null.
  // Disagree: static error.
  const ir = { kind: 'call', op: 'select',
    branches: [
      { ir: Normal(0, 1) },
      { ir: { kind: 'call', op: 'iid', args: [Normal(0, 1), lit(2)] } },
    ],
    logweights: null };
  const variate = T.array(1, [2], T.REAL);
  const r = densityPrims.staticConsume(ir, variate);
  assert.ok(r.error && /different footprint/.test(r.error),
    `expected different-footprint error, got: ${r.error}`);
});

test('static: select with ref-form branches stays deferred', () => {
  // The walker can't statically resolve a `ref` branch (would need
  // binding lookup). Defer rather than error — runtime will check.
  const ir = { kind: 'call', op: 'select',
    branches: [{ ref: 'm1' }, { ref: 'm2' }],
    logweights: null };
  const err = densityPrims.staticDensityShapeCheck(ir, T.REAL);
  assert.equal(err, null);   // deferred → no error reported
});

// =====================================================================
// Passthroughs (weighted / logweighted / truncate / normalize)
// =====================================================================

test('static: weighted(w, M) inherits M\'s shape check', () => {
  const M = Normal(0, 1);
  const ir = { kind: 'call', op: 'weighted', args: [lit(2.0), M] };
  // Scalar variate: passes.
  let r = densityPrims.staticConsume(ir, T.REAL);
  assert.equal(r.rest, null);
  // Record variate: same scalar-leaf rejection through the passthrough.
  r = densityPrims.staticConsume(ir, { kind: 'record', fields: { x: T.REAL } });
  assert.ok(r.error);
});

test('static: normalize(M) and truncate(M, S) pass through', () => {
  const M = Normal(0, 1);
  const norm = { kind: 'call', op: 'normalize', args: [M] };
  const trunc = { kind: 'call', op: 'truncate', args: [M, lit(null)] };
  assert.equal(densityPrims.staticConsume(norm, T.REAL).rest, null);
  assert.equal(densityPrims.staticConsume(trunc, T.REAL).rest, null);
});

// =====================================================================
// pushfwd
// =====================================================================

test('static: pushfwd over scalar Normal vs scalar variate → empty rest', () => {
  // pushfwd(f, M) with scalar↔scalar f: recurse into M with the
  // same variate type (rank preserved).
  const ir: any = { kind: 'call', op: 'pushfwd',
    args: [
      { kind: 'ref', ns: 'self', name: 'f' },   // bijection ref (ignored without annotation)
      Normal(0, 1),
    ],
    // Synthetic bijection annotation: scalar forward fn body type.
    bijection: {
      f: { body: { meta: { type: { kind: 'scalar', prim: 'real' } } } },
      fInv: { body: { meta: { type: { kind: 'scalar', prim: 'real' } } } },
    },
  };
  const r = densityPrims.staticConsume(ir, T.REAL);
  assert.equal(r.rest, null);
  assert.ok(!r.error);
});

test('static: pushfwd without bijection annotation → falls through to M', () => {
  // Without an annotated f, defer to recursing into M with the
  // surface variate — preserves the old behaviour for unannotated
  // bijections.
  const ir: any = { kind: 'call', op: 'pushfwd',
    args: [{ kind: 'lit', value: 'f' }, Normal(0, 1)] };
  const r = densityPrims.staticConsume(ir, T.REAL);
  // Recurses into Normal → scalar leaf consumes the scalar variate.
  assert.equal(r.rest, null);
});

// =====================================================================
// Multivariate kernels
// =====================================================================

test('static: MvNormal vs rank-1 vector variate → empty rest (deferred length)', () => {
  const ir = { kind: 'call', op: 'MvNormal', kwargs: {} };
  const r = densityPrims.staticConsume(ir, T.array(1, [3], T.REAL));
  assert.equal(r.rest, null);
  assert.ok(!r.error);
});

test('static: MvNormal vs scalar variate → static error', () => {
  const ir = { kind: 'call', op: 'MvNormal', kwargs: {} };
  const r = densityPrims.staticConsume(ir, T.REAL);
  assert.ok(r.error && /array variate|rank-≥1 variate/.test(r.error));
});

test('static: Wishart vs rank-1 vector → static error (needs rank-2)', () => {
  const ir = { kind: 'call', op: 'Wishart', kwargs: {} };
  const r = densityPrims.staticConsume(ir, T.array(1, [3], T.REAL));
  assert.ok(r.error && /rank-≥2/.test(r.error));
});

test('static: Wishart vs rank-2 matrix → empty rest (deferred shape)', () => {
  const ir = { kind: 'call', op: 'Wishart', kwargs: {} };
  const r = densityPrims.staticConsume(ir, T.array(2, [3, 3], T.REAL));
  assert.equal(r.rest, null);
  assert.ok(!r.error);
});

// =====================================================================
// staticDensityShapeCheck strict wrapper
// =====================================================================

test('staticDensityShapeCheck: passes on matching shapes, fails on mismatch', () => {
  const ok = densityPrims.staticDensityShapeCheck(
    Normal(0, 1), T.REAL);
  assert.equal(ok, null);
  const bad = densityPrims.staticDensityShapeCheck(
    { kind: 'call', op: 'iid', args: [Normal(0, 1), lit(3)] },
    T.array(1, [5], T.REAL));
  assert.ok(bad && /3 elements/.test(bad));
});
