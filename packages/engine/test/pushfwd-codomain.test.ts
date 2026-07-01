'use strict';

// shape-contract S2 — `pushfwd(f, M)` codomain. The result measure's variate
// type is the type `f` PRODUCES when applied to a value of M's variate type.
// Before S2, `inferPushfwd` hard-coded `measure(real)`, discarding f's codomain
// — because `f.result` is the *definition-time* type, inferred with the reified
// param bound to `any` (so `fn(2.0 .* _).result` is a scalar even for a vector
// base). S2 re-infers f's single-param body with the param bound to M's domain
// (the FlatPIR call-site-specialization / polymorphic path), so the advertised
// type matches what the sampler draws and what the density (for a registered /
// annotated bijection) already scores. Falls back to `measure(real)` for the
// cases the re-inference can't specialize (bare builtin symbol, multi-param,
// unresolved) — never a regression.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const T = require('../types.ts');

function inferredDomain(src: string, name: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, 'unexpected errors: ' + errs.map((e: any) => e.message).join(' | '));
  const t = r.bindings.get(name).inferredType;
  assert.equal(t.kind, 'measure', 'expected a measure, got ' + T.show(t));
  return t.domain;
}

test('pushfwd of a scalar transform over a scalar measure → measure over a scalar', () => {
  const d = inferredDomain('m = pushfwd(fn(exp(_)), Normal(0.0, 1.0))', 'm');
  assert.equal(d.kind, 'scalar', 'got ' + T.show(d));
});

test('pushfwd of an elementwise transform over a vector measure → measure over the vector', () => {
  const d = inferredDomain(
    'm = pushfwd(fn(2.0 .* _), MvNormal(mu = [0.0, 0.0], cov = eye(2)))', 'm');
  assert.equal(d.kind, 'array', 'got ' + T.show(d));
  assert.deepEqual(d.shape, [2]);
});

test('pushfwd codomain specializes a self-ref function too', () => {
  const d = inferredDomain(
    'f = fn(2.0 .* _)\nm = pushfwd(f, MvNormal(mu = [0.0, 0.0], cov = eye(2)))', 'm');
  assert.equal(d.kind, 'array', 'got ' + T.show(d));
  assert.deepEqual(d.shape, [2]);
});

test('pushfwd projection narrows the codomain (get subset over a vector base)', () => {
  // fn(get(_, [1, 2])) selects a 2-element sub-vector of a 3-vector base.
  const d = inferredDomain(
    'm = pushfwd(fn(get(_, [1, 2])), MvNormal(mu = [0.0, 0.0, 0.0], cov = eye(3)))', 'm');
  assert.equal(d.kind, 'array', 'got ' + T.show(d));
  assert.deepEqual(d.shape, [2]);
});

test('pushfwd of a bare builtin symbol falls back to a scalar codomain (no regression)', () => {
  // `pushfwd(exp, Normal)` — exp is a bare builtin, not a reification we can
  // re-infer; the scalar default is still correct here.
  const d = inferredDomain('m = pushfwd(exp, Normal(0.0, 1.0))', 'm');
  assert.equal(d.kind, 'scalar', 'got ' + T.show(d));
});
