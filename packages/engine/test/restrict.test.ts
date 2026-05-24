'use strict';

// Spec §06 "Measure restriction":
//
//   restrict(M, x)  ≡
//     kernel, marginal = disintegrate([..field-names-of-x..], M)
//     bayesupdate(likelihoodof(kernel, x), marginal)
//
// The engine expands `restrict(...)` to that chain in the analyzer's
// pre-pass — every op it produces is already classified and
// materialised, so this is a pure structural rewrite.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

function errors(src: string) {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

test('restrict: kwarg form parses cleanly', () => {
  const src = `
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
restricted = restrict(prior, sigma = 0.8)
`;
  assert.equal(errors(src).length, 0);
});

test('restrict: positional record-literal form parses cleanly', () => {
  const src = `
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
obs = record(sigma = 0.8)
restricted = restrict(prior, obs)
`;
  // The positional case requires the inline record literal OR a
  // record-typed binding — both should be accepted.
  // Inline-literal form:
  const src2 = `
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
restricted = restrict(prior, record(sigma = 0.8))
`;
  assert.equal(errors(src2).length, 0);
});

test('restrict: expands to bayesupdate', () => {
  const src = `
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
restricted = restrict(prior, sigma = 0.8)
`;
  const ctx = processSource(src);
  const b = ctx.bindings.get('restricted');
  assert.ok(b, 'restricted binding present');
  // After expansion, the binding's classification is bayesupdate.
  assert.equal(b.type, 'bayesupdate',
    `expected bayesupdate, got ${b.type}`);
});

test('restrict: synthesizes __restrict_kernel/__restrict_marginal anons', () => {
  const src = `
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
restricted = restrict(prior, sigma = 0.8)
`;
  const ctx = processSource(src);
  const anonNames = [...ctx.bindings.keys()].filter(
    (n: string) => n.startsWith('__restrict_'));
  // Exactly one kernel + one marginal anon.
  const kernels = anonNames.filter((n: string) => n.startsWith('__restrict_kernel'));
  const margs   = anonNames.filter((n: string) => n.startsWith('__restrict_marginal'));
  assert.equal(kernels.length, 1, `expected 1 __restrict_kernel, got ${kernels.length}`);
  assert.equal(margs.length,   1, `expected 1 __restrict_marginal, got ${margs.length}`);
});

test('restrict: matches the spec equivalence (against manual expansion)', () => {
  // Same posterior built two ways:
  //   A: restrict(prior, sigma = 0.8)
  //   B: kernel, marginal = disintegrate(["sigma"], prior)
  //      bayesupdate(likelihoodof(kernel, record(sigma = 0.8)), marginal)
  const ctxA = processSource(`
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
posterior = restrict(prior, sigma = 0.8)
`);
  const ctxB = processSource(`
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
kernel, marginal = disintegrate(["sigma"], prior)
posterior = bayesupdate(likelihoodof(kernel, record(sigma = 0.8)), marginal)
`);
  // Both should parse cleanly.
  assert.equal(ctxA.diagnostics.filter(
    (d: any) => d.severity === 'error').length, 0);
  assert.equal(ctxB.diagnostics.filter(
    (d: any) => d.severity === 'error').length, 0);
  // posterior classifies as bayesupdate in both.
  assert.equal(ctxA.bindings.get('posterior').type, 'bayesupdate');
  assert.equal(ctxB.bindings.get('posterior').type, 'bayesupdate');
});

test('restrict: posterior-construction example from spec parses', () => {
  // Spec §06: posterior built from joint model + observation.
  const src = `
mu = Normal(mu = 0, sigma = 1)
sigma = Exponential(rate = 1)
prior = joint(mu = mu, sigma = sigma)
likelihood_kernel = functionof(iid(Normal(mu = mu, sigma = sigma), 5), mu = mu, sigma = sigma)
joint_model = jointchain(prior, likelihood_kernel)
posterior = restrict(joint_model, obs = [0.9, 0.7, -1.2, 0.3, -0.5])
`;
  // This one stresses disintegration too — may or may not succeed
  // structurally depending on the joint's shape. We assert the
  // restrict expansion itself doesn't crash; downstream
  // disintegrate-admissibility may or may not be admissible.
  const errs = errors(src);
  // The expansion must run; the only acceptable errors are
  // disintegrate-admissibility / unsupported diagnostics — NOT
  // "restrict() ..." errors.
  for (const e of errs) {
    assert.ok(!/restrict\(\)/.test(e.message),
      `unexpected restrict error: ${e.message}`);
  }
});

test('restrict: missing measure arg → error', () => {
  const errs = errors('x = restrict()\n');
  assert.ok(errs.some((d: any) => /restrict\(\) requires/.test(d.message)));
});

test('restrict: measure must be an identifier (not a literal)', () => {
  const errs = errors('x = restrict(42, sigma = 0.8)\n');
  assert.ok(errs.some((d: any) => /measure argument must be a binding reference/.test(d.message)));
});

test('restrict: positional non-record observation → error', () => {
  const errs = errors(`
prior = joint(mu = Normal(mu = 0, sigma = 1))
x = restrict(prior, 42)
`);
  assert.ok(errs.some((d: any) => /field names could not be determined statically/.test(d.message)));
});

// ---------------------------------------------------------------------
// restrict(M, x_ident) where x is a binding ref to a record literal
// — the analyzer follows the identifier to its defining `record(...)`
// and extracts field names. Previously this dead-ended with
// "field names could not be determined statically" (TODO line 173).
// ---------------------------------------------------------------------

test('restrict: positional identifier referencing a record literal resolves', () => {
  const src = `
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
obs = record(sigma = 0.8)
restricted = restrict(prior, obs)
`;
  const errs = errors(src);
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'identifier-typed observation should resolve via the defining record literal');
});

test('restrict: identifier-resolution expansion produces the same shape as inline', () => {
  // Build the model two ways and confirm the analyzer reaches the
  // same binding structure for `restricted`.
  const A = processSource(`
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
obs = record(sigma = 0.8)
restricted = restrict(prior, obs)
`);
  const B = processSource(`
prior = joint(mu = Normal(mu = 0, sigma = 1), sigma = Exponential(rate = 1))
restricted = restrict(prior, record(sigma = 0.8))
`);
  assert.deepEqual(A.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.deepEqual(B.diagnostics.filter((d: any) => d.severity === 'error'), []);
  // Both should produce a bayesupdate-typed `restricted` binding.
  assert.equal(A.bindings.get('restricted').type, 'bayesupdate');
  assert.equal(B.bindings.get('restricted').type, 'bayesupdate');
});

test('restrict: cycle in identifier resolution (r = r) does not infinite-loop', () => {
  // Pathological self-reference: the resolver's seen-set guards against
  // infinite recursion; the analyzer reports the field-name diagnostic.
  const src = `
prior = joint(mu = Normal(mu = 0, sigma = 1))
r = r
x = restrict(prior, r)
`;
  // We just need this to terminate (not infinite-loop) with some error.
  const errs = errors(src);
  // The classifier surfaces an error one way or another; cycle in r,
  // unknown record fields, etc. — what we pin is "doesn't hang".
  assert.ok(errs.length > 0, 'self-referential identifier should surface an error');
});

test('restrict: identifier referencing a non-record binding → clean diagnostic', () => {
  const src = `
prior = joint(mu = Normal(mu = 0, sigma = 1))
xs = [1.0, 2.0]   # not a record
x = restrict(prior, xs)
`;
  const errs = errors(src);
  assert.ok(errs.some((d: any) =>
    /field names could not be determined statically/.test(d.message)),
    'non-record identifier should produce the static field-names diagnostic');
});
