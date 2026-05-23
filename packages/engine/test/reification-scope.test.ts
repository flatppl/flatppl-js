'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const path  = require('node:path');
const fs    = require('node:fs');
const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');

// =====================================================================
// Reification scope: bodyDeps / paramSourceDeps + absorbedPhaseOf path
// =====================================================================
//
// These tests pin the engine's two-scope reification model (engine-
// concepts §8) at the analyzer surface: a `functionof` / `kernelof`
// binding's deps split into
//   - bodyDeps:        body refs to outer-scope bindings (closure captures)
//                      — body refs to formal parameters are %local and
//                      DO NOT appear here.
//   - paramSourceDeps: kwarg RHS refs (boundary value-set declarations)
// `deps` remains the union (preserved for backwards compatibility
// with the topological sort and DAG-viewer edge rendering).
//
// The patterns are taken from packages/web/demo/feature-test1.flatppl,
// the demo that exposed the original `rand` / kernel-call-result
// regression (`matEvaluate: parent 'expected_obs' has neither .value
// nor .samples`). Each test names what the case is checking.

function bindingsOf(src: string) {
  const { bindings, diagnostics } = processSource(src);
  return { bindings, diagnostics };
}

function depsOf(b: any) {
  return {
    deps:             [...(b?.deps             ?? [])].sort(),
    bodyDeps:         [...(b?.bodyDeps         ?? [])].sort(),
    paramSourceDeps:  [...(b?.paramSourceDeps  ?? [])].sort(),
  };
}

// ---------------------------------------------------------------------
// collectDeps: body / paramSource partitioning
// ---------------------------------------------------------------------

test('reif/deps: functionof body closure captures vs kwarg paramSources', () => {
  // f_a's body uses `c` (closure capture) and `_par` (formal). The
  // kwarg RHS `par = _par` references _par as a value-set declaration.
  //   - bodyDeps must be just {c}: the body's `_par` lowers to %local
  //     and doesn't count as an outer dep.
  //   - paramSourceDeps must be {_par}: the kwarg RHS is a real ref.
  //   - deps is the union for back-compat.
  const { bindings } = bindingsOf(`
    c = 2.5
    _par = elementof(reals)
    f_a = functionof(c * _par, par = _par)
  `);
  const d = depsOf(bindings.get('f_a'));
  assert.deepEqual(d.bodyDeps, ['c']);
  assert.deepEqual(d.paramSourceDeps, ['_par']);
  assert.deepEqual(d.deps, ['_par', 'c'].sort());
});

test('reif/deps: forward_kernel with two paramSources', () => {
  // forward_kernel = functionof(obs_dist, theta1 = theta1, theta2 = theta2).
  //   - bodyDeps = {obs_dist} only — theta1 / theta2 inside obs_dist
  //     would resolve through obs_dist's own deps, not through the
  //     functionof's surface refs.
  //   - paramSourceDeps = {theta1, theta2}.
  const { bindings } = bindingsOf(`
    theta1 = draw(Normal(mu = 0, sigma = 1))
    theta2 = draw(Exponential(rate = 1))
    obs_dist = joint(obs = iid(Normal(mu = theta1, sigma = theta2), 10))
    forward_kernel = functionof(obs_dist, theta1 = theta1, theta2 = theta2)
  `);
  const d = depsOf(bindings.get('forward_kernel'));
  assert.deepEqual(d.bodyDeps, ['obs_dist']);
  assert.deepEqual(d.paramSourceDeps, ['theta1', 'theta2']);
});

test('reif/deps: kernelof partitions like functionof (kernelof lowers to functionof(lawof(...)))', () => {
  // Same split rule applies to kernelof per spec §sec:kernelof.
  const { bindings } = bindingsOf(`
    theta1 = draw(Normal(mu = 0, sigma = 1))
    theta2 = draw(Exponential(rate = 1))
    a = 5.0 * theta1
    obs = iid(Normal(mu = a, sigma = theta2), 10)
    K = kernelof(obs, theta1 = theta1, theta2 = theta2)
  `);
  const d = depsOf(bindings.get('K'));
  assert.deepEqual(d.bodyDeps, ['obs']);
  assert.deepEqual(d.paramSourceDeps, ['theta1', 'theta2']);
});

test('reif/deps: fn (no boundary kwargs) — body uses placeholder, no paramSources', () => {
  // fn(...) takes positional `_` holes (lowered to numbered placeholders
  // via the parser); body refs to closure-captured outer bindings still
  // count as bodyDeps. No kwarg RHS, so paramSourceDeps is empty.
  const { bindings } = bindingsOf(`
    theta = draw(Normal(mu = 0, sigma = 1))
    f = fn(abs(_) * theta)
  `);
  const d = depsOf(bindings.get('f'));
  // theta is the closure capture; the hole isn't a binding ref.
  assert.deepEqual(d.bodyDeps, ['theta']);
  assert.deepEqual(d.paramSourceDeps, []);
});

test('reif/deps: non-reification bindings have empty paramSourceDeps; bodyDeps mirrors deps', () => {
  // Sanity: for ordinary bindings (literals, arithmetic, joints), the
  // body bucket carries everything and paramSource is empty. This is
  // the invariant that lets callers use bodyDeps uniformly without
  // checking the binding kind first.
  const { bindings } = bindingsOf(`
    a = 1.0
    b = a + 2.0
    M = Normal(mu = a, sigma = b)
    J = joint(x = M, y = Exponential(rate = a))
  `);
  for (const name of ['b', 'M', 'J']) {
    const d = depsOf(bindings.get(name));
    assert.deepEqual(d.paramSourceDeps, [], `${name} has empty paramSourceDeps`);
    assert.deepEqual(d.bodyDeps, d.deps, `${name} bodyDeps mirrors deps`);
  }
});

// ---------------------------------------------------------------------
// Phase: absorbedPhaseOf must not propagate paramSource phases
// ---------------------------------------------------------------------
//
// These pin the original regression: `rand(state, expected_obs)` where
// expected_obs is the result of a kernel application — the kernel's
// internal formal parameters (e.g. theta1/theta2 bound to user-side
// theta1/theta2 of `draw` phase) used to leak `parameterized` through
// the absorbed-phase walker, preventing pre-eval. After the dep split,
// the walker reads bodyDeps and naturally skips kwarg-RHS refs.

test('reif/phase: rand of kernel-applied measure is fixed (no elementof leak)', () => {
  // The minimal repro of the feature-test1.flatppl regression. The
  // value-set chain runs through f_a's kwarg RHS (_par = elementof);
  // beta1 (the actual call arg) is fixed (no elementof), so the rand
  // result must be fixed.
  const { bindings } = bindingsOf(`
    _par = elementof(reals)
    c = 2.5
    f_a = functionof(c * _par, par = _par)
    beta1 = 2.0
    a = f_a(par = beta1)
    M = Normal(mu = a, sigma = 1)
    rstate0 = rnginit([1,2,3,4])
    sample, _ = rand(rstate0, M)
  `);
  const sample = bindings.get('sample');
  assert.equal(sample.phase, 'fixed',
    'rand result must be fixed when call args carry no elementof');
});

test('reif/phase: kernel-applied measure used as rand argument is fixed', () => {
  // feature-test1.flatppl shape: `rand_obs_data, _ = rand(rstate2, forward_kernel(rand_pars))`.
  const { bindings } = bindingsOf(`
    theta1 = draw(Normal(mu = 0, sigma = 1))
    theta2 = draw(Exponential(rate = 1))
    obs_dist = joint(obs = iid(Normal(mu = theta1, sigma = theta2), 10))
    prior = lawof(record(theta1 = theta1, theta2 = theta2))
    forward_kernel = functionof(obs_dist, theta1 = theta1, theta2 = theta2)
    rs0 = rnginit([1,2,3,4])
    rp, rs1 = rand(rs0, prior)
    ro, _ = rand(rs1, forward_kernel(rp))
  `);
  const ro = bindings.get('ro');
  assert.equal(ro.phase, 'fixed',
    'ro = rand(state, forward_kernel(rp)) is fixed end-to-end');
});

test('reif/phase: rand under elementof closure remains parameterized', () => {
  // Counterpart: when there IS a non-formal elementof in the call's
  // ancestor closure, the result IS parameterized — the bodyDeps walk
  // reaches it through the body, not the kwargs.
  const { bindings } = bindingsOf(`
    mu = elementof(reals)
    M = Normal(mu = mu, sigma = 1)
    rstate0 = rnginit([1,2,3,4])
    sample, _ = rand(rstate0, M)
  `);
  assert.equal(bindings.get('sample').phase, 'parameterized',
    'rand under elementof remains parameterized');
});

test('reif/phase: forward_kernel definition is fixed even with elementof formals', () => {
  // The function value itself is fixed (spec §sec:functionof: "the
  // function value itself is %fixed"). Mark this explicitly so a
  // future change that disturbs absorbedPhaseOf's body/kwarg split
  // shows up here.
  const { bindings } = bindingsOf(`
    theta = elementof(reals)
    sigma = elementof(posreals)
    K = functionof(Normal(mu = theta, sigma = sigma), theta = theta, sigma = sigma)
  `);
  assert.equal(bindings.get('K').phase, 'fixed');
});

// ---------------------------------------------------------------------
// End-to-end: feature-test1.flatppl as a whole compiles + pre-evaluates
// ---------------------------------------------------------------------

const featureTest1Path = path.join(
  __dirname, '..', '..', 'web', 'demo', 'feature-test1.flatppl');

test('e2e: feature-test1.flatppl analyzes with the expected invalid-* diagnostics', () => {
  if (!fs.existsSync(featureTest1Path)) {
    // Test inert if the demo file isn't reachable from the build root
    // (e.g. flatppl-js cloned standalone without sibling packages).
    return;
  }
  const src = fs.readFileSync(featureTest1Path, 'utf8');
  const { diagnostics } = processSource(src);
  // The file intentionally contains 3 invalid bindings (named with the
  // `invalid*` prefix). The valid bindings must produce no errors.
  // We accept the file's documented invalid forms (recorded as
  // type/arity errors) and reject anything else.
  const errs = diagnostics.filter((d: any) => d.severity === 'error');
  // Each of the three invalid_* bindings should yield exactly one
  // diagnostic each. We don't pin exact messages (they may improve
  // over time) but we pin the count: any new error coming from a
  // *valid* binding would show up here as a regression.
  assert.equal(errs.length, 3,
    `expected exactly 3 errors from the file's three invalid bindings, got ${errs.length}:\n` +
    errs.map((d: any) => `  - ${d.message}`).join('\n'));
});

test('e2e: feature-test1.flatppl rand-chain bindings reach fixedValues', () => {
  if (!fs.existsSync(featureTest1Path)) return;
  const src = fs.readFileSync(featureTest1Path, 'utf8');
  const ctx = processSource(src);
  const { fixedValues } = buildDerivations(ctx.bindings, ctx.diagnostics);
  // The rand-chain bindings of the demo must pre-evaluate. This is
  // the contract the original regression broke. We pin each step
  // separately so a future failure points at the right one.
  for (const name of ['rand_pars', 'rstate2', 'rand_obs_data', 'rand_obs_values']) {
    assert.ok(fixedValues.has(name),
      `${name} should pre-evaluate to a concrete value`);
  }
  const rod = fixedValues.get('rand_obs_data');
  assert.ok(rod && typeof rod === 'object',
    'rand_obs_data is a record-shaped concrete value');
  assert.ok(Array.isArray(rod.obs) && rod.obs.length === 10,
    'rand_obs_data.obs is a length-10 array of concrete samples');
  // rand_obs_values mirrors rand_obs_data.obs via field access.
  const rov = fixedValues.get('rand_obs_values');
  assert.ok(Array.isArray(rov) && rov.length === 10,
    'rand_obs_values is a length-10 concrete array');
});

test('e2e: feature-test1.flatppl variant binding phases match the spec rules', () => {
  if (!fs.existsSync(featureTest1Path)) return;
  const src = fs.readFileSync(featureTest1Path, 'utf8');
  const { bindings } = processSource(src);
  // Spot-check the phase of each binding shape in the demo.
  // Phase classifications follow spec §04 §phases + §sec:lawof.
  const expected: Record<string, string> = {
    // Inputs / constants
    c: 'fixed',
    _par: 'parameterized',
    // draw → stochastic
    theta1: 'stochastic',
    theta2: 'stochastic',
    // Function/kernel values themselves are fixed
    f_a: 'fixed',
    f_b: 'fixed',
    forward_kernel: 'fixed',
    // lawof / rand absorb stochasticity; the demo has no remaining
    // elementof in their bodyDep closures, so result is fixed.
    prior: 'fixed',
    joint_model: 'fixed',
    rand_pars: 'fixed',
    expected_obs: 'fixed',
    rand_obs_data: 'fixed',
    rand_obs_values: 'fixed',
    // observed_data is a literal array
    observed_data: 'fixed',
  };
  for (const [name, phase] of Object.entries(expected)) {
    const b = bindings.get(name);
    assert.ok(b, `binding ${name} should exist`);
    assert.equal(b.phase, phase, `phase of ${name}`);
  }
});
