'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, computeSubDAG } = require('../index');

test('disintegrate: simple kernel/prior split from lawof(record(...))', () => {
  const src = `
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
theta2 = draw(Normal(mu = 0.0, sigma = 1.0))
obs = draw(Normal(mu = theta1, sigma = theta2))
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
forward_kernel, prior = disintegrate(["obs"], joint_model)
`;
  const { bindings, diagnostics } = processSource(src);
  assert.equal(diagnostics.filter(d => d.severity === 'error').length, 0);

  // Both should be classified as lawof (they are kernels/measures)
  assert.equal(bindings.get('forward_kernel').type, 'lawof');
  assert.equal(bindings.get('prior').type, 'lawof');

  // forward_kernel sub-DAG: target=obs, boundaries=theta1, theta2
  const fk = computeSubDAG(bindings, 'forward_kernel');
  const fkIds = new Set(fk.nodes.map(n => n.id));
  assert.ok(fkIds.has('forward_kernel'));
  assert.ok(fkIds.has('obs'));
  assert.ok(fkIds.has('theta1'));
  assert.ok(fkIds.has('theta2'));
  // theta1, theta2 must be boundaries
  assert.equal(fk.nodes.find(n => n.id === 'theta1').isBoundary, true);
  assert.equal(fk.nodes.find(n => n.id === 'theta2').isBoundary, true);
  // forward_kernel must be the target
  assert.equal(fk.nodes.find(n => n.id === 'forward_kernel').isTarget, true);

  // prior sub-DAG: targets=theta1, theta2, no boundaries
  const pr = computeSubDAG(bindings, 'prior');
  const prIds = new Set(pr.nodes.map(n => n.id));
  assert.ok(prIds.has('prior'));
  assert.ok(prIds.has('theta1'));
  assert.ok(prIds.has('theta2'));
  // theta1, theta2 in prior should NOT be boundaries
  assert.equal(pr.nodes.find(n => n.id === 'theta1').isBoundary, false);
  assert.equal(pr.nodes.find(n => n.id === 'theta2').isBoundary, false);
  // prior should NOT contain obs (it was the kernel target, marginalized away)
  assert.ok(!prIds.has('obs'));
});

test('disintegrate: string selector "obs" works like ["obs"]', () => {
  const src = `
a = draw(Normal(mu = 0, sigma = 1))
b = draw(Normal(mu = a, sigma = 1))
joint_model = lawof(record(a = a, b = b))
fk, pr = disintegrate("b", joint_model)
`;
  const { bindings, diagnostics } = processSource(src);
  assert.equal(diagnostics.filter(d => d.severity === 'error').length, 0);

  const fk = computeSubDAG(bindings, 'fk');
  const fkIds = new Set(fk.nodes.map(n => n.id));
  assert.ok(fkIds.has('b'));
  assert.equal(fk.nodes.find(n => n.id === 'a').isBoundary, true);

  const pr = computeSubDAG(bindings, 'pr');
  const prIds = new Set(pr.nodes.map(n => n.id));
  assert.ok(prIds.has('a'));
  assert.ok(!prIds.has('b'));
});

test('disintegrate: inherited boundaries from joint lawof carry through', () => {
  // joint has its own boundary input mu — both kernel and prior inherit it.
  const src = `
mu = elementof(reals)
gamma = draw(Normal(mu = 0, sigma = 1))
obs = draw(Normal(mu = gamma * mu, sigma = 1))
joint_model = lawof(record(gamma = gamma, obs = obs), mu = mu)
fk, pr = disintegrate("obs", joint_model)
`;
  const { bindings, diagnostics } = processSource(src);
  assert.equal(diagnostics.filter(d => d.severity === 'error').length, 0);

  // forward_kernel: target=obs; boundaries=gamma (unselected), mu (inherited)
  const fk = computeSubDAG(bindings, 'fk');
  const fkIds = new Set(fk.nodes.map(n => n.id));
  assert.ok(fkIds.has('obs'));
  assert.equal(fk.nodes.find(n => n.id === 'gamma').isBoundary, true);
  assert.equal(fk.nodes.find(n => n.id === 'mu').isBoundary, true);

  // prior: target=gamma; mu is inherited boundary
  const pr = computeSubDAG(bindings, 'pr');
  const prIds = new Set(pr.nodes.map(n => n.id));
  assert.ok(prIds.has('gamma'));
  assert.equal(pr.nodes.find(n => n.id === 'gamma').isBoundary, false);
  assert.equal(pr.nodes.find(n => n.id === 'mu').isBoundary, true);
});

test('disintegrate: fall back to plain trace when joint is non-lawof', () => {
  // joint is the result of chain(...) — structure not statically resolvable.
  // We don't lower disintegration; fk and pr just trace through joint as deps.
  const src = `
prior_m = Normal(mu = 0, sigma = 1)
fwd_k = Normal(mu = 0, sigma = 1)
joint_model = chain(prior_m, fwd_k)
fk, pr = disintegrate("x", joint_model)
`;
  const { bindings, diagnostics } = processSource(src);
  // No errors expected (semantic intractability is a runtime concern, not a static error)
  assert.equal(diagnostics.filter(d => d.severity === 'error').length, 0);

  // fk should at least include itself and the joint binding (plain dep trace)
  const fk = computeSubDAG(bindings, 'fk');
  const fkIds = new Set(fk.nodes.map(n => n.id));
  assert.ok(fkIds.has('fk'));
  assert.ok(fkIds.has('joint_model'));
});

test('disintegrate: multi-field selector', () => {
  // disintegrate(["a", "b"], joint) extracts a kernel for {a, b} jointly,
  // leaving {c} as the marginal.
  const src = `
a = draw(Normal(mu = 0, sigma = 1))
b = draw(Normal(mu = 0, sigma = 1))
c = draw(Normal(mu = 0, sigma = 1))
joint_model = lawof(record(a = a, b = b, c = c))
fk, pr = disintegrate(["a", "b"], joint_model)
`;
  const { bindings, diagnostics } = processSource(src);
  assert.equal(diagnostics.filter(d => d.severity === 'error').length, 0);

  // fk: targets a, b; boundary c
  const fk = computeSubDAG(bindings, 'fk');
  const fkIds = new Set(fk.nodes.map(n => n.id));
  assert.ok(fkIds.has('a'));
  assert.ok(fkIds.has('b'));
  assert.equal(fk.nodes.find(n => n.id === 'c').isBoundary, true);

  // pr: target c
  const pr = computeSubDAG(bindings, 'pr');
  const prIds = new Set(pr.nodes.map(n => n.id));
  assert.ok(prIds.has('c'));
  assert.equal(pr.nodes.find(n => n.id === 'c').isBoundary, false);
  // a, b should not be in pr at all
  assert.ok(!prIds.has('a'));
  assert.ok(!prIds.has('b'));
});

test('disintegrate: invalid selector field is a diagnostic', () => {
  // Selector "z" doesn't appear in joint's record — should be flagged
  const src = `
a = draw(Normal(mu = 0, sigma = 1))
joint_model = lawof(record(a = a))
fk, pr = disintegrate("z", joint_model)
`;
  const { diagnostics } = processSource(src);
  assert.ok(diagnostics.some(d => /selector|field|'z'/i.test(d.message)));
});

// --- Fixture-based: complex multi-disintegration scenarios ---

const fs = require('node:fs');
const path = require('node:path');
const FIXTURE = path.join(__dirname, 'fixtures', 'disintegrate-complex.flatppl');

test('disintegrate (complex fixture): parses without errors', () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { diagnostics } = processSource(src);
  const errors = diagnostics.filter(d => d.severity === 'error');
  assert.equal(errors.length, 0, `errors: ${JSON.stringify(errors)}`);
});

test('disintegrate (complex fixture): single-field kernel inherits joint boundaries', () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { bindings } = processSource(src);
  const fk = computeSubDAG(bindings, 'fk_a');
  const ids = new Set(fk.nodes.map(n => n.id));

  // Target obs1 is reachable; theta1, theta2, obs2 become boundaries (from unselected fields);
  // mu, sigma remain boundaries (inherited from joint).
  assert.ok(ids.has('obs1'));
  for (const b of ['theta1', 'theta2', 'obs2', 'mu', 'sigma']) {
    const node = fk.nodes.find(n => n.id === b);
    assert.ok(node, `missing boundary ${b}`);
    assert.equal(node.isBoundary, true, `${b} should be boundary`);
  }
});

test('disintegrate (complex fixture): multi-field kernel and matching prior', () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { bindings } = processSource(src);

  const fk = computeSubDAG(bindings, 'fk_b');
  const fkIds = new Set(fk.nodes.map(n => n.id));
  // Both selected fields appear as targets (their nodes traced through)
  for (const t of ['obs1', 'obs2']) assert.ok(fkIds.has(t));
  // Unselected fields become boundaries
  for (const b of ['theta1', 'theta2']) {
    assert.equal(fk.nodes.find(n => n.id === b).isBoundary, true);
  }

  const pr = computeSubDAG(bindings, 'pr_b');
  const prIds = new Set(pr.nodes.map(n => n.id));
  // Prior has theta1, theta2 as targets; obs1, obs2 should not appear at all
  for (const t of ['theta1', 'theta2']) assert.ok(prIds.has(t));
  for (const x of ['obs1', 'obs2']) assert.ok(!prIds.has(x));
  // Inherited boundaries mu, sigma preserved
  for (const b of ['mu', 'sigma']) {
    assert.equal(pr.nodes.find(n => n.id === b).isBoundary, true);
  }
});

test('disintegrate (complex fixture): no-input joint produces clean prior', () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { bindings } = processSource(src);
  const pr = computeSubDAG(bindings, 'pr_c');
  const ids = new Set(pr.nodes.map(n => n.id));
  // theta1 is the unselected field -> target; obs1 is excluded.
  assert.ok(ids.has('theta1'));
  assert.ok(!ids.has('obs1'));
});

test('disintegrate (complex fixture): chain-based joint falls back to plain trace', () => {
  // joint_chained = chain(...) — non-statically-resolvable.
  // fk_d, pr_d should NOT be tagged with disintegrateRole and just trace deps.
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { bindings } = processSource(src);

  const fk_d = bindings.get('fk_d');
  const pr_d = bindings.get('pr_d');
  // Neither should have a disintegrate role — fall-back regime.
  assert.equal(fk_d.disintegrateRole, undefined);
  assert.equal(pr_d.disintegrateRole, undefined);
  // Their type should still be 'deterministic' (decomposition default)
  assert.equal(fk_d.type, 'deterministic');
  assert.equal(pr_d.type, 'deterministic');

  // Sub-DAG just walks through joint_chained as a dep.
  const dag = computeSubDAG(bindings, 'fk_d');
  const ids = new Set(dag.nodes.map(n => n.id));
  assert.ok(ids.has('joint_chained'));
});

// --- Tier 2: joint(name = M, ...) keyword form ---

test('disintegrate Tier 2: joint(...) keyword form is recognised', () => {
  const src = `
mu_p = elementof(reals)
joint_indep = joint(
    theta1 = Normal(mu = mu_p, sigma = 1.0),
    theta2 = Exponential(rate = 1.0)
)
fk, pr = disintegrate("theta1", joint_indep)
`;
  const { bindings, diagnostics } = processSource(src);
  assert.equal(diagnostics.filter(d => d.severity === 'error').length, 0);
  // Both decomposed names should be classified as lawof (kernel/measure)
  assert.equal(bindings.get('fk').type, 'lawof');
  assert.equal(bindings.get('pr').type, 'lawof');
  assert.equal(bindings.get('fk').disintegrateRole.jointKind, 'joint');
});

test('disintegrate Tier 2: kernel ancestors come from selected component expressions', () => {
  // theta1 = Normal(mu = mu_p, sigma = 1) -> kernel depends on mu_p
  // theta2 = Exponential(rate = 1) -> not selected, mu_p NOT involved
  const src = `
mu_p = elementof(reals)
joint_indep = joint(
    theta1 = Normal(mu = mu_p, sigma = 1.0),
    theta2 = Exponential(rate = 1.0)
)
fk, pr = disintegrate("theta1", joint_indep)
`;
  const { bindings } = processSource(src);
  const fk = computeSubDAG(bindings, 'fk');
  const ids = new Set(fk.nodes.map(n => n.id));
  // mu_p is the only module-level node referenced by Normal(mu = mu_p, sigma = 1)
  assert.ok(ids.has('mu_p'), 'kernel should include mu_p as ancestor');
  // Ensure NO "boundary" annotation — joint components are independent
  for (const n of fk.nodes) {
    if (n.id !== 'fk') assert.equal(n.isBoundary, false);
  }
});

test('disintegrate Tier 2: prior ancestors come from unselected component expressions', () => {
  const src = `
mu_p = elementof(reals)
rate_p = elementof(posreals)
joint_indep = joint(
    theta1 = Normal(mu = mu_p, sigma = 1.0),
    theta2 = Exponential(rate = rate_p)
)
fk, pr = disintegrate("theta1", joint_indep)
`;
  const { bindings } = processSource(src);
  const pr = computeSubDAG(bindings, 'pr');
  const ids = new Set(pr.nodes.map(n => n.id));
  // theta2's M depends only on rate_p; mu_p should NOT be in the prior's DAG
  assert.ok(ids.has('rate_p'));
  assert.ok(!ids.has('mu_p'), 'prior should not contain selected-only deps');
});

test('disintegrate Tier 2: multi-field selector', () => {
  const src = `
a = elementof(reals)
b = elementof(reals)
c = elementof(reals)
m = joint(
    f1 = Normal(mu = a, sigma = 1.0),
    f2 = Normal(mu = b, sigma = 1.0),
    f3 = Normal(mu = c, sigma = 1.0)
)
fk, pr = disintegrate(["f1", "f2"], m)
`;
  const { bindings } = processSource(src);
  const fk = computeSubDAG(bindings, 'fk');
  const fkIds = new Set(fk.nodes.map(n => n.id));
  // Kernel has deps from f1 and f2 (a, b) but NOT c
  assert.ok(fkIds.has('a'));
  assert.ok(fkIds.has('b'));
  assert.ok(!fkIds.has('c'));

  const pr = computeSubDAG(bindings, 'pr');
  const prIds = new Set(pr.nodes.map(n => n.id));
  assert.ok(prIds.has('c'));
  assert.ok(!prIds.has('a'));
  assert.ok(!prIds.has('b'));
});

// --- Tier 2: jointchain(name = M, ...) keyword form ---

test('disintegrate Tier 2 (jointchain): keyword form is recognised', () => {
  const src = `
mu_p = elementof(reals)
m = jointchain(
    a = Normal(mu = 0.0, sigma = 1.0),
    b = Normal(mu = mu_p, sigma = 1.0)
)
fk, pr = disintegrate("b", m)
`;
  const { bindings, diagnostics } = processSource(src);
  assert.equal(diagnostics.filter(d => d.severity === 'error').length, 0);
  assert.equal(bindings.get('fk').disintegrateRole.jointKind, 'jointchain');
  assert.equal(bindings.get('fk').type, 'lawof');
});

test('disintegrate Tier 2 (jointchain): trailing selection adds synthetic boundary for earlier field', () => {
  // selected ["b"] — kernel needs synthetic 'a' boundary (chain-earlier).
  const src = `
mu_p = elementof(reals)
m = jointchain(
    a = Normal(mu = 0.0, sigma = 1.0),
    b = Normal(mu = mu_p, sigma = 1.0)
)
fk, pr = disintegrate("b", m)
`;
  const { bindings } = processSource(src);
  const fk = computeSubDAG(bindings, 'fk');
  const ids = new Set(fk.nodes.map(n => n.id));
  // Synthetic boundary 'a' for the kernel
  const aBoundary = fk.nodes.find(n => n.id === 'fk:a');
  assert.ok(aBoundary, 'expected synthetic boundary node fk:a');
  assert.equal(aBoundary.label, 'a');
  assert.equal(aBoundary.isBoundary, true);
  // The kernel also picks up mu_p as ancestor (used in b's expression)
  assert.ok(ids.has('mu_p'));
});

test('disintegrate Tier 2 (jointchain): leading selection has no synthetic boundary', () => {
  // selected ["a"] — kernel is the chain head, no preceding fields.
  const src = `
mu_p = elementof(reals)
m = jointchain(
    a = Normal(mu = mu_p, sigma = 1.0),
    b = Normal(mu = 0.0, sigma = 1.0)
)
fk, pr = disintegrate("a", m)
`;
  const { bindings } = processSource(src);
  const fk = computeSubDAG(bindings, 'fk');
  // No synthetic field-name boundary
  for (const n of fk.nodes) {
    if (n.id !== 'fk') assert.equal(n.isBoundary, false);
  }
});
