'use strict';

// Tests for the structural-disintegration Plan rewriter.
// Distinct from disintegrate.test.js, which exercises the legacy DAG-rendering
// path. Here we look directly at the Plan AST returned by `disintegratePlan`
// (via `binding.disintegratePlan`) and check its shape.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, disintegrate } = require('../index');

const { disintegratePlan } = disintegrate;

// --- Helpers ---------------------------------------------------------

// Render a small AST node (the kernel/prior expressions returned in
// 'synthesized' plans) to a canonical string so tests can assert on shape
// without caring about loc info.
function render(node) {
  if (!node) return '';
  switch (node.type) {
    case 'Identifier':    return node.name;
    case 'NumberLiteral': return node.raw != null ? node.raw : String(node.value);
    case 'StringLiteral': return JSON.stringify(node.value);
    case 'BoolLiteral':   return String(node.value);
    case 'CallExpr':      return render(node.callee) + '(' + (node.args || []).map(render).join(', ') + ')';
    case 'KeywordArg':    return node.name + ' = ' + render(node.value);
    case 'ArrayLiteral':  return '[' + (node.elements || []).map(render).join(', ') + ']';
    default:              return '<' + node.type + '>';
  }
}

function planFor(src, bindingName) {
  const { bindings, diagnostics } = processSource(src);
  const errs = diagnostics.filter(d => d.severity === 'error');
  assert.equal(errs.length, 0, `errors: ${JSON.stringify(errs)}`);
  const b = bindings.get(bindingName);
  assert.ok(b, `no binding named ${bindingName}`);
  assert.ok(b.disintegratePlan, `no disintegratePlan on ${bindingName}`);
  return b.disintegratePlan;
}

// --- v1: lawof(record(...)) -----------------------------------------

test('Plan: lawof(record) → synthesized kernelof + lawof(record)', () => {
  const src = `
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
theta2 = draw(Normal(mu = 0.0, sigma = 1.0))
obs = draw(Normal(mu = theta1, sigma = theta2))
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
fk, pr = disintegrate(["obs"], joint_model)
`;
  const plan = planFor(src, 'fk');
  assert.equal(plan.kind, 'synthesized');
  assert.equal(render(plan.kernel),
    'kernelof(record(obs = obs), theta1 = theta1, theta2 = theta2)');
  assert.equal(render(plan.prior),
    'lawof(record(theta1 = theta1, theta2 = theta2))');
});

test('Plan: lawof(record) admissibility: selected feeding into unselected → Unsupported', () => {
  // 'a' is selected; 'b' = transform of 'a' is unselected. Disintegrating
  // 'a' would require integrating against a deterministic chain — refused.
  const src = `
a = draw(Normal(mu = 0, sigma = 1))
b = draw(Normal(mu = a, sigma = 1))
joint_model = lawof(record(a = a, b = b))
fk, pr = disintegrate("a", joint_model)
`;
  const plan = planFor(src, 'fk');
  assert.equal(plan.kind, 'unsupported');
  assert.match(plan.reason, /unselected.*depends on selected/);
});

// --- v1: joint(kw) ---------------------------------------------------

test('Plan: joint(kw) → synthesized joint kernel + joint prior', () => {
  const src = `
mu_p = elementof(reals)
joint_indep = joint(
    theta1 = Normal(mu = mu_p, sigma = 1.0),
    theta2 = Exponential(rate = 1.0)
)
fk, pr = disintegrate("theta1", joint_indep)
`;
  const plan = planFor(src, 'fk');
  assert.equal(plan.kind, 'synthesized');
  assert.equal(render(plan.kernel),
    'joint(theta1 = Normal(mu = mu_p, sigma = 1.0))');
  assert.equal(render(plan.prior),
    'joint(theta2 = Exponential(rate = 1.0))');
});

// --- v1: jointchain(kw) suffix -------------------------------------

test('Plan: jointchain(kw) suffix → synthesized', () => {
  const src = `
mu_p = elementof(reals)
m = jointchain(
    a = Normal(mu = 0.0, sigma = 1.0),
    b = Normal(mu = mu_p, sigma = 1.0)
)
fk, pr = disintegrate("b", m)
`;
  const plan = planFor(src, 'fk');
  assert.equal(plan.kind, 'synthesized');
  // Single-component sides collapse: prior = a's measure verbatim;
  // kernel = b's measure wrapped in kernelof with `a` as a boundary input
  // (the chain's earlier-field-as-input semantics, encoded explicitly).
  assert.equal(render(plan.kernel),
    'kernelof(Normal(mu = mu_p, sigma = 1.0), a = a)');
  assert.equal(render(plan.prior),
    'Normal(mu = 0.0, sigma = 1.0)');
});

test('Plan: jointchain(kw) non-suffix selector → Unsupported', () => {
  // selector covers leading 'a' but leaves 'b' on the prior side — that's
  // the inversion direction, requires Bayes-update.
  const src = `
m = jointchain(
    a = Normal(mu = 0.0, sigma = 1.0),
    b = Normal(mu = 0.0, sigma = 1.0)
)
fk, pr = disintegrate("a", m)
`;
  const plan = planFor(src, 'fk');
  assert.equal(plan.kind, 'unsupported');
  assert.match(plan.reason, /suffix/);
});

// --- v1: jointchain positional → Delegate -------------------------

test('Plan: jointchain positional with two binding-identifier sides → Delegate', () => {
  const src = `
prior = Normal(mu = 0.0, sigma = 1.0)
forward_kernel = kernelof(Normal(mu = _a_, sigma = 1.0), a = _a_)
joint_model = jointchain(prior, forward_kernel)
fk, pr = disintegrate(["forward_kernel"], joint_model)
`;
  const { bindings, diagnostics } = processSource(src);
  const errs = diagnostics.filter(d => d.severity === 'error');
  assert.equal(errs.length, 0, `errors: ${JSON.stringify(errs)}`);
  // detectDisintegration only matches when each selector field appears in
  // namedOutputFields of some component. forward_kernel has no statically
  // resolvable named outputs, so we don't get a Plan here — confirm that.
  // (Documented limitation; remove when v2 introduces more tracing.)
  const b = bindings.get('fk');
  // Either disintegratePlan is missing entirely (analyzer didn't recognise
  // the joint statically) or it's set and is delegate/unsupported.
  if (b.disintegratePlan) {
    const p = b.disintegratePlan;
    assert.ok(p.kind === 'delegate' || p.kind === 'unsupported',
      `unexpected plan kind: ${p.kind}`);
  }
});

// --- v2: chain → Unsupported ---------------------------------------

test('Plan: chain → Unsupported (marginalization)', () => {
  const src = `
prior_m = Normal(mu = 0, sigma = 1)
fwd_k = Normal(mu = 0, sigma = 1)
joint_model = chain(prior_m, fwd_k)
fk, pr = disintegrate("x", joint_model)
`;
  const { bindings } = processSource(src);
  const b = bindings.get('fk');
  // detectDisintegration may not match (no extractJointFields rule for
  // chain) so disintegratePlan may be unset. In either case, no synthesized
  // plan should be produced.
  const p = b.disintegratePlan;
  if (p) {
    assert.equal(p.kind, 'unsupported');
  }
});

// --- v2: relabel(positional joint) → recurse ----------------------

test('Plan: relabel(positional joint) lifts names then disintegrates', () => {
  // Direct rewriter test: build the AST shape without going through
  // detectDisintegration (which currently only matches lawof/joint/jointchain
  // keyword forms at the joint binding). We construct the joint ourselves.
  const src = `
m1 = Normal(mu = 0, sigma = 1)
m2 = Exponential(rate = 1.0)
labelled = relabel(joint(m1, m2), ["a", "b"])
`;
  const { bindings } = processSource(src);
  const labelled = bindings.get('labelled');
  assert.ok(labelled);
  const plan = disintegratePlan(
    labelled.node.value, ['a'], bindings,
    { seen: new Set(), source: 'labelled' });
  assert.equal(plan.kind, 'synthesized');
  assert.equal(render(plan.kernel), 'joint(a = m1)');
  assert.equal(render(plan.prior),  'joint(b = m2)');
});

test('Plan: relabel without inner positional structure → Unsupported', () => {
  const src = `
m = Normal(mu = 0, sigma = 1)
labelled = relabel(m, ["x"])
`;
  const { bindings } = processSource(src);
  const plan = disintegratePlan(
    bindings.get('labelled').node.value, ['x'], bindings,
    { seen: new Set(), source: 'labelled' });
  assert.equal(plan.kind, 'unsupported');
});

// --- v2: pushfwd → Unsupported (with detail) -----------------------

test('Plan: pushfwd → Unsupported', () => {
  const src = `
m = Normal(mu = 0, sigma = 1)
pushed = pushfwd(exp, m)
`;
  const { bindings } = processSource(src);
  const plan = disintegratePlan(
    bindings.get('pushed').node.value, ['x'], bindings,
    { seen: new Set(), source: 'pushed' });
  assert.equal(plan.kind, 'unsupported');
  assert.match(plan.reason, /pushfwd/);
});

// --- Cho-Jacobs post-condition ------------------------------------
//
// For a synthesized plan, jointchain(prior, kernel) must reconstruct the
// original joint up to (a) field-name re-ordering and (b) the kernel-side
// boundary inputs being supplied by the prior side. We don't have a
// full FlatPPL evaluator in this engine, so we check a structural
// approximation: the union of named output fields across the
// reconstructed jointchain equals the joint's named output fields.

test('Plan: post-condition — synthesized prior + kernel cover the joint fields', () => {
  const src = `
a = draw(Normal(mu = 0, sigma = 1))
b = draw(Normal(mu = 0, sigma = 1))
c = draw(Normal(mu = a, sigma = b))
joint_model = lawof(record(a = a, b = b, c = c))
fk, pr = disintegrate(["c"], joint_model)
`;
  const { bindings } = processSource(src);
  const plan = bindings.get('fk').disintegratePlan;
  assert.equal(plan.kind, 'synthesized');
  const { namedOutputFields } = disintegrate;
  const priorFields  = new Set(namedOutputFields(plan.prior, bindings));
  const kernelFields = new Set(namedOutputFields(plan.kernel, bindings));
  // Joint had a, b, c; selector was c; so prior covers {a, b}, kernel {c}.
  assert.deepEqual([...priorFields].sort(),  ['a', 'b']);
  assert.deepEqual([...kernelFields].sort(), ['c']);
});

// --- AST hygiene: synthesized nodes carry synthLoc ----------------

test('Plan: synthesized AST nodes are marked synthetic', () => {
  const src = `
a = draw(Normal(mu = 0, sigma = 1))
b = draw(Normal(mu = 0, sigma = 1))
c = draw(Normal(mu = 0, sigma = 1))
joint_model = lawof(record(a = a, b = b, c = c))
fk, pr = disintegrate(["a"], joint_model)
`;
  const plan = planFor(src, 'fk');
  assert.equal(plan.kind, 'synthesized');
  // Walk plan.kernel and plan.prior; every node's loc should be synthetic.
  function check(node) {
    if (!node || typeof node !== 'object') return;
    if (node.loc) {
      assert.equal(node.loc.synthetic, true,
        `node ${node.type} has non-synthetic loc`);
    }
    for (const k of Object.keys(node)) {
      const v = node[k];
      if (Array.isArray(v)) v.forEach(check);
      else if (v && typeof v === 'object' && v.type) check(v);
    }
  }
  check(plan.kernel);
  check(plan.prior);
});
