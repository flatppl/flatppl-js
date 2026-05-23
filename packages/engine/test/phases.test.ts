'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, computePhases } = require('../index.ts');
const { computePhasesForScope } = require('../analyzer.ts');
const { computeSubDAG } = require('../dag.ts');

function phasesOf(src: any) {
  const { bindings } = processSource(src);
  const result = {};
  for (const [name, b] of bindings) result[name] = b.phase;
  return result;
}

// --- Per-spec direct rules ---

test('phase: draw is stochastic', () => {
  const p = phasesOf('x = draw(Normal(mu = 0, sigma = 1))\n');
  assert.equal(p.x, 'stochastic');
});

test('phase: elementof is parameterized', () => {
  const p = phasesOf('x = elementof(reals)\n');
  assert.equal(p.x, 'parameterized');
});

test('phase: external is fixed', () => {
  const p = phasesOf('x = external(reals)\n');
  assert.equal(p.x, 'fixed');
});

test('phase: literal is fixed', () => {
  const p = phasesOf('x = 1.5\narr = [1, 2, 3]\n');
  assert.equal(p.x, 'fixed');
  assert.equal(p.arr, 'fixed');
});

// --- Propagation through ancestors ---

test('phase: propagates stochastic through deterministic operations', () => {
  // a = f(theta1) where theta1 is draw → a is stochastic
  const p = phasesOf(`
theta1 = draw(Normal(mu = 0, sigma = 1))
a = 2 * theta1 + 5
b = a + 1
`);
  assert.equal(p.theta1, 'stochastic');
  assert.equal(p.a, 'stochastic');
  assert.equal(p.b, 'stochastic');
});

test('phase: parameterized propagates through deterministic ops', () => {
  const p = phasesOf(`
mu_p = elementof(reals)
a = mu_p + 1
b = a * 2
`);
  assert.equal(p.mu_p, 'parameterized');
  assert.equal(p.a, 'parameterized');
  assert.equal(p.b, 'parameterized');
});

test('phase: stochastic dominates over parameterized', () => {
  const p = phasesOf(`
mu_p = elementof(reals)
theta1 = draw(Normal(mu = mu_p, sigma = 1))
a = mu_p + theta1
`);
  assert.equal(p.theta1, 'stochastic');
  assert.equal(p.a, 'stochastic');
});

test('phase: parameterized dominates over fixed', () => {
  const p = phasesOf(`
n = external(integers)
mu_p = elementof(reals)
a = mu_p + n
`);
  assert.equal(p.n, 'fixed');
  assert.equal(p.mu_p, 'parameterized');
  assert.equal(p.a, 'parameterized');
});

// --- Bayesian inference example ---

test('phase: bayesian_inference_2 fixture has correct phases', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'bayesian_inference_2.flatppl'), 'utf8');
  const p = phasesOf(src);

  // Stochastic chain
  assert.equal(p.theta1, 'stochastic');
  assert.equal(p.theta2, 'stochastic');
  assert.equal(p.a, 'stochastic', "a depends on theta1 (stochastic)");
  assert.equal(p.b, 'stochastic', "b depends on theta1, theta2 (stochastic)");
  assert.equal(p.obs, 'stochastic');

  // joint_model = lawof(record(...)). Per spec §sec:lawof line 309-314,
  // lawof absorbs stochasticity into the reified measure — the result
  // is fixed unless an elementof remains in the ancestor closure. Here
  // theta1/theta2/obs are all draws of literal-kwarg distributions
  // (no elementof anywhere), so joint_model is fixed.
  assert.equal(p.joint_model, 'fixed');

  // observed_data is a literal
  assert.equal(p.observed_data, 'fixed');
});

// --- computePhases as a standalone function ---

test('phase: computePhases works on already-built bindings', () => {
  const { bindings } = processSource(`
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
y = 2 * x
`);
  const phases = computePhases(bindings);
  assert.equal(phases.get('mu'), 'parameterized');
  assert.equal(phases.get('x'), 'stochastic');
  assert.equal(phases.get('y'), 'stochastic');
});

// --- computePhasesForScope: scope-local phase under boundaries ---

test('phase: computePhasesForScope cuts the chain at boundary names', () => {
  // Globally beta1 is stochastic (depends on draw via theta1).
  // With theta1 declared as a boundary input, beta1's phase walk
  // stops at theta1 → 'parameterized', so beta1 itself reads as
  // 'parameterized' too.
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
beta1 = 2 * theta1
`);
  const global = computePhases(bindings);
  assert.equal(global.get('theta1'), 'stochastic');
  assert.equal(global.get('beta1'),  'stochastic');

  const scoped = computePhasesForScope(bindings, new Set(['theta1']));
  assert.equal(scoped.get('theta1'), 'parameterized');
  assert.equal(scoped.get('beta1'),  'parameterized');
});

test('phase: computePhasesForScope with empty boundaries === computePhases', () => {
  const { bindings } = processSource(`
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
y = 2 * x
`);
  const a = computePhases(bindings);
  const b = computePhasesForScope(bindings, new Set());
  for (const k of a.keys()) assert.equal(b.get(k), a.get(k));
});

// --- DAG: scope-local phase override applied to in-bubble nodes ---

test('phase: DAG nodes inside a kernel bubble carry scope-local phase', () => {
  // forward_kernel's body has theta1/theta2 as boundary inputs.
  // Inside the bubble, both they and beta1 (= 2*theta2) read as
  // 'parameterized'; outside, they're stochastic.
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
beta1  = 2 * theta2
obs    = draw(Normal(mu = beta1, sigma = 1))
fk     = functionof(obs, theta1 = theta1, theta2 = theta2)
`);
  const dag = computeSubDAG(bindings, 'fk');
  const byId = new Map(dag.nodes.map((n: any) => [n.id, n]));
  // Inside the kernel: theta2 and beta1 cut by the boundary.
  assert.equal(byId.get('theta2').phase, 'parameterized');
  assert.equal(byId.get('beta1').phase,  'parameterized');
  // Global view of the same bindings still has them stochastic.
  const global = computePhases(bindings);
  assert.equal(global.get('theta2'), 'stochastic');
  assert.equal(global.get('beta1'),  'stochastic');
});

// --- Phase-check on reified-callable boundary kwargs ---
// Per spec §04 sec:functionof: "Boundary inputs themselves may be of
// parametric or stochastic phase, but not fixed phase." A fixed-phase
// boundary is closed over by the reified callable, not lifted into
// its signature — the user's intent is malformed. The analyzer emits
// an error so the silent-wrong-result path is short-circuited.

function diagsFor(src: any) {
  const { diagnostics } = processSource(src);
  return diagnostics.filter((d: any) => /Boundary input/.test(d.message));
}

test('boundary-phase: literal binding as boundary kwarg → error', () => {
  // Without this diagnostic, inlineUserCall silently produces a wrong
  // substitution because `c`'s value is already baked into the body's
  // lifted form, so the call-arg replacement reaches no live ref.
  const diags = diagsFor(`
c = 2.0
f = functionof(c + 1, theta = c)
`);
  assert.equal(diags.length, 1);
  assert.equal(diags[0].severity, 'error');
  assert.match(diags[0].message, /references 'c'/);
  assert.match(diags[0].message, /fixed phase/);
});

test('boundary-phase: external() as boundary kwarg → error (also fixed)', () => {
  const diags = diagsFor(`
ext = external("data.dat")
f = functionof(ext + 1, theta = ext)
`);
  assert.equal(diags.length, 1);
  assert.match(diags[0].message, /references 'ext'/);
});

test('boundary-phase: parameterized binding (elementof) → no error', () => {
  const diags = diagsFor(`
mu = elementof(reals)
f = functionof(mu + 1, theta = mu)
`);
  assert.equal(diags.length, 0);
});

test('boundary-phase: stochastic binding (draw) → no error', () => {
  const diags = diagsFor(`
y = draw(Normal(mu = 0, sigma = 1))
f = functionof(y + 1, theta = y)
`);
  assert.equal(diags.length, 0);
});

test('boundary-phase: kernelof obeys the same rule', () => {
  // kernelof shares the boundary-kwarg semantics — the spec rule
  // applies uniformly. Catches the parallel mistake on the kernel side.
  const diags = diagsFor(`
c = 2.0
K = kernelof(Normal(mu = c, sigma = 1), theta = c)
`);
  assert.equal(diags.length, 1);
  assert.match(diags[0].message, /of kernelof/);
});

test('boundary-phase: placeholder boundary is unchecked (always parametric)', () => {
  // `par = _par_` synthesizes the boundary as elementof(valueset(par))
  // at lower time, so placeholders are inherently parametric. The
  // diagnostic only inspects Identifier-form values; placeholders
  // skip the check cleanly.
  const diags = diagsFor(`
c = 2.0
f = functionof(c + _par_, par = _par_)
`);
  assert.equal(diags.length, 0);
});

test('boundary-phase: multiple kwargs reported independently', () => {
  // Each fixed boundary is its own error so the user can see all of
  // them at once instead of fixing one and re-running.
  const diags = diagsFor(`
c1 = 1.0
c2 = 2.0
mu = elementof(reals)
f = functionof(c1 + c2 + mu, a = c1, b = c2, m = mu)
`);
  // Two errors: on c1 and c2. mu is parametric, no error.
  assert.equal(diags.length, 2);
  const targets = diags.map((d: any) => d.message.match(/references '(\w+)'/)[1]).sort();
  assert.deepEqual(targets, ['c1', 'c2']);
});

test('boundary-phase: no-kwargs functionof (auto-promote) is unchecked', () => {
  // canonicalizeImplicitBoundaries synthesizes kwargs whose values are
  // already parametric elementof leaves (it explicitly filters fixed-
  // phase out at promote time), so the diagnostic finds nothing to
  // complain about on auto-promoted bindings — and it shouldn't run
  // against synthesized AST anyway. Smoke-test: the canonical case
  // emits no boundary-phase errors.
  const diags = diagsFor(`
mu = elementof(reals)
f = functionof(mu * 2)
`);
  assert.equal(diags.length, 0);
});
