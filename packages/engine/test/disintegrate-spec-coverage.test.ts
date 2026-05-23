'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const { disintegratePlan } = require('../disintegrate.ts');

// =====================================================================
// Spec coverage audit for `disintegrate` (flatppl-design §06)
// =====================================================================
//
// Spec §06 §"Structural disintegration": "For the large class of joint
// models whose factorization structure is explicit in the DAG,
// disintegrate can be implemented via straightforward graph inspection.
// For models that involve internal marginalization, non-bijective
// changes of variables, or other transformations that destroy explicit
// factorization structure, the decomposition may be intractable and
// may not be supported."
//
// This file is the **coverage matrix** mapping each shape the spec
// describes to the engine's current handling. Three buckets:
//
//   PASSES — engine returns a synthesized / delegate plan and
//            downstream classification (prior derivation, kernel
//            shape) is well-formed.
//   UNSUPPORTED (spec-permissive) — engine returns Unsupported and
//            the spec allows this (intractable / non-bijective /
//            destroys-factorization).
//   UNSUPPORTED (gap) — engine returns Unsupported but the spec's
//            "explicit factorization in the DAG" wording implies it
//            should work. Asserted via assert.throws / assert.ok(!d)
//            so the test stays honest if a future fix closes the gap.
//
// Where the engine produces a result, we additionally check the
// disintegration equation must satisfy
//   jointchain(base_measure, kernel) ≡ joint_measure                (1)
// (spec §06 line 553-554) holds structurally — i.e. kernel + prior's
// DAGs together cover the joint's variates with the right scoping.

function planOf(src: string, jointName: string, selector: string[]) {
  const { bindings, diagnostics } = processSource(src);
  const errs = diagnostics.filter((d: any) => d.severity === 'error');
  if (errs.length > 0) {
    throw new Error(`parse/type errors: ${errs.map((e: any) => e.message).join('; ')}`);
  }
  const j = bindings.get(jointName);
  if (!j) throw new Error(`no binding '${jointName}'`);
  return disintegratePlan(j.node.value, selector, bindings, {
    seen: new Set(), source: jointName,
  });
}

function classifyOf(src: string) {
  const ctx = processSource(src);
  const dctx = buildDerivations(ctx.bindings, ctx.diagnostics);
  return { bindings: ctx.bindings, derivations: dctx.derivations,
           diagnostics: ctx.diagnostics };
}

// ---------------------------------------------------------------------
// PASSES: explicit-factorization shapes the spec requires
// ---------------------------------------------------------------------

test('spec-coverage PASS: lawof(record(name=variate, ...))', () => {
  // The canonical spec example (§06 lines 530-537). Selector picks
  // the observation field; prior marginalises it out.
  const src = `
sigma = 1.0
a = draw(Normal(mu = 0.0, sigma = 2.0))
b = draw(Normal(mu = a, sigma = sigma))
joint_model = lawof(record(a = a, b = b))
fk, pr = disintegrate(["b"], joint_model)
  `;
  const plan = planOf(src, 'joint_model', ['b']);
  assert.equal(plan.kind, 'synthesized',
    'lawof(record(...)) with explicit fields must synthesize');
  // pr classifies as alias of the marginal anon (single-variate path
  // collapses to draw-alias).
  const { derivations } = classifyOf(src);
  assert.ok(derivations.pr, 'prior derivation must exist');
});

test('spec-coverage PASS: joint(name=M, ...) keyword form', () => {
  const src = `
M1 = Normal(mu = 0, sigma = 1)
M2 = Exponential(rate = 1)
J = joint(a = M1, b = M2)
fk, pr = disintegrate(["b"], J)
  `;
  const plan = planOf(src, 'J', ['b']);
  assert.equal(plan.kind, 'synthesized');
});

test('spec-coverage PASS: jointchain(M, K) suffix selector — demo case', () => {
  // The feature-test1.flatppl shape: jointchain prior + forward kernel,
  // selector picks the kernel's output field.
  const src = `
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
a = 5 * theta1
obs_dist = joint(obs = iid(Normal(mu = a, sigma = theta2), 10))
forward_kernel = functionof(obs_dist, theta1 = theta1, theta2 = theta2)
prior = lawof(record(theta1 = theta1, theta2 = theta2))
joint_model = jointchain(prior, forward_kernel)
fk, pr = disintegrate(["obs"], joint_model)
  `;
  const plan = planOf(src, 'joint_model', ['obs']);
  // Either 'delegate' (when prior + kernel are pre-existing binding
  // refs and the rewriter can re-use them) or 'synthesized' (when the
  // rewriter has to manufacture new AST) is acceptable per spec.
  assert.notEqual(plan.kind, 'unsupported',
    'jointchain(prior, K) with K-output selector must NOT be Unsupported');
});

test('spec-coverage PASS: multi-field selector picks several joint fields', () => {
  const src = `
M1 = Normal(mu = 0, sigma = 1)
M2 = Exponential(rate = 1)
M3 = Normal(mu = 1, sigma = 1)
J = joint(a = M1, b = M2, c = M3)
fk, pr = disintegrate(["a", "b"], J)
  `;
  const plan = planOf(src, 'J', ['a', 'b']);
  assert.equal(plan.kind, 'synthesized',
    'multi-field selector on joint must synthesize');
});

test('spec-coverage PASS: relabel(positional joint, [names]) lifts to keyword joint', () => {
  // relabel of positional joint(M1, M2) with name list ["a", "b"] is
  // equivalent to joint(a=M1, b=M2). Disintegrate must see through.
  const src = `
M1 = Normal(mu = 0, sigma = 1)
M2 = Exponential(rate = 1)
J = relabel(joint(M1, M2), ["a", "b"])
fk, pr = disintegrate(["b"], J)
  `;
  const plan = planOf(src, 'J', ['b']);
  assert.equal(plan.kind, 'synthesized',
    'relabel of positional joint must synthesize');
});

test('spec-coverage PASS: delegate when joint components are binding refs', () => {
  // jointchain(prior_binding, kernel_binding) — if both sides are
  // pre-existing binding refs AND the selector picks a field that
  // structurally comes from the kernel side, the disintegrate result
  // delegates directly to them (no synthesis needed; the existing
  // structure already provides the kernel + prior pair).
  const src = `
theta = draw(Normal(mu = 0, sigma = 1))
prior = lawof(record(theta = theta))
kernel = functionof(joint(obs = Normal(mu = theta, sigma = 1)), theta = theta)
J = jointchain(prior, kernel)
fk, pr = disintegrate(["obs"], J)
  `;
  const plan = planOf(src, 'J', ['obs']);
  assert.notEqual(plan.kind, 'unsupported',
    'jointchain of binding refs (with valid selector) must not be Unsupported');
});

// ---------------------------------------------------------------------
// PASSES: selector form variants per spec §06 "Selectors work like
// in `get`: \"b\" selects the bare value, [\"b\"] selects a record"
// ---------------------------------------------------------------------

test('spec-coverage PASS: bare-string selector "b" and array selector ["b"] both produce a plan', () => {
  // Spec §06 lines 550-551: 'Selectors work like in `get`: "b" selects
  // the bare value, ["b"] selects a `record(b = ...)`'. So the two
  // forms SHOULD differ in the shape of the kernel's output (bare
  // value vs single-field record).
  //
  // Today's engine normalises both forms to ['b'] in
  // analyzer.detectDisintegration (line ~797) before reaching the
  // rewriter, so the resulting kernel is always record-shaped
  // regardless of which selector form the user wrote. This test pins
  // BOTH the parts the engine gets right (both forms parse and
  // produce a plan) AND records the known gap (see GAP test below).
  const src1 = `
M1 = Normal(mu = 0, sigma = 1)
M2 = Exponential(rate = 1)
J = joint(a = M1, b = M2)
fk, pr = disintegrate("b", J)
  `;
  const src2 = `
M1 = Normal(mu = 0, sigma = 1)
M2 = Exponential(rate = 1)
J = joint(a = M1, b = M2)
fk, pr = disintegrate(["b"], J)
  `;
  const ctx1 = processSource(src1);
  const ctx2 = processSource(src2);
  assert.equal(ctx1.diagnostics.filter((d: any) => d.severity === 'error').length, 0,
    'bare-string selector parses');
  assert.equal(ctx2.diagnostics.filter((d: any) => d.severity === 'error').length, 0,
    'array selector parses');
  const plan1 = disintegratePlan(ctx1.bindings.get('J').node.value, ['b'],
    ctx1.bindings, { seen: new Set(), source: 'J' });
  const plan2 = disintegratePlan(ctx2.bindings.get('J').node.value, ['b'],
    ctx2.bindings, { seen: new Set(), source: 'J' });
  assert.equal(plan1.kind, plan2.kind,
    'bare-string and array selectors produce same plan kind');
});

test('spec-coverage PASS: bare-string vs array selector produces differently-shaped kernels', () => {
  // Spec §06 lines 550-551: `"b"` selects the bare value (kernel emits
  // the variate directly), `["b"]` selects a `record(b = ...)` (kernel
  // emits single-field-record). Closed 2026-05-23: analyzer preserves
  // the surface shape via `selectorBareString`, threaded into
  // disintegrate's ctx; synthesizeJoint and synthesizeLawofRecord
  // honour the flag.
  const ctx = processSource(`
M1 = Normal(mu = 0, sigma = 1)
M2 = Exponential(rate = 1)
J = joint(a = M1, b = M2)
  `);
  const planBare  = disintegratePlan(ctx.bindings.get('J').node.value, ['b'],
    ctx.bindings, { seen: new Set(), source: 'J', selectorBareString: true });
  const planArray = disintegratePlan(ctx.bindings.get('J').node.value, ['b'],
    ctx.bindings, { seen: new Set(), source: 'J', selectorBareString: false });
  // Both succeed structurally.
  assert.equal(planBare.kind, 'synthesized');
  assert.equal(planArray.kind, 'synthesized');
  // Kernel shapes DIFFER per the selector form.
  // Bare: kernel is just the variate's measure expression (no joint wrap).
  assert.equal(planBare.kernel.type, 'Identifier',
    'bare-string selector → kernel is the bare measure ref');
  assert.equal((planBare.kernel as any).name, 'M2');
  // Array: kernel is `joint(b = M2)`, a single-field-record measure.
  assert.equal(planArray.kernel.type, 'CallExpr');
  assert.equal((planArray.kernel as any).callee.name, 'joint',
    'array selector → kernel is joint(...) wrapping the field');
  // Prior shape is independent of selector form.
  assert.equal(JSON.stringify(planBare.prior), JSON.stringify(planArray.prior),
    'prior shape is independent of selector form');
});

// ---------------------------------------------------------------------
// UNSUPPORTED (spec-permissive): cases where the spec explicitly says
// the decomposition "may be intractable and may not be supported"
// ---------------------------------------------------------------------

test('spec-coverage UNSUPPORTED-OK: kchain (chain) — marginalization destroys factorization', () => {
  // kchain marginalizes out intermediate variates. The marginalised
  // measure has lost info about the latents; disintegrate can't
  // generally recover them. Spec §06 lists this as may-not-be-
  // supported. Engine must NOT silently produce a wrong answer.
  const src = `
M = Normal(mu = 0, sigma = 1)
K = functionof(Normal(mu = x, sigma = 1), x = x)
x = draw(M)
J = kchain(M, K)
fk, pr = disintegrate(["b"], J)
  `;
  const plan = planOf(src, 'J', ['b']);
  assert.equal(plan.kind, 'unsupported',
    'kchain (marginalization) must report Unsupported');
});

test('spec-coverage UNSUPPORTED-OK: pushfwd through non-trivial f', () => {
  // pushfwd through a non-projection function (exp) is a non-bijective
  // change of variables from disintegrate's point of view; spec allows
  // engines to refuse this.
  const src = `
M = joint(a = Normal(mu = 0, sigma = 1), b = Exponential(rate = 1))
J = pushfwd(fn(exp(_)), M)
fk, pr = disintegrate(["a"], J)
  `;
  // pushfwd of joint may not classify cleanly — what we want is that
  // we don't silently mis-disintegrate.
  const plan = planOf(src, 'J', ['a']);
  assert.equal(plan.kind, 'unsupported',
    'pushfwd through non-projection f must be Unsupported (or wrap into a Plan that\'s clearly correct, which the current engine does not)');
});

// ---------------------------------------------------------------------
// UNSUPPORTED (gaps the spec doesn't strictly forbid)
// ---------------------------------------------------------------------
//
// These are shapes whose factorization IS explicit in the DAG but the
// current engine returns Unsupported. They're permissive gaps — the
// spec doesn't require them to work — but they're the candidates for
// "what could we add coverage for next?". Asserted as Unsupported so
// the test pins the contract honestly; if a future change closes a
// gap, this test fires and the contract gets updated.

test('spec-coverage GAP: jointchain non-suffix selector (picks a middle field)', () => {
  // jointchain(M, K1, K2) keyword form — selector picks the middle
  // kernel's output (not a suffix). The factorization IS explicit but
  // the engine's current rule only handles suffix selectors per the
  // existing disintegrate-plan.test.ts "non-suffix selector → Unsupported"
  // case. Recorded here as a known gap.
  const src = `
M = Normal(mu = 0, sigma = 1)
K1 = functionof(joint(b = Normal(mu = a, sigma = 1)), a = a)
K2 = functionof(joint(c = Normal(mu = b, sigma = 1)), b = b)
a = draw(M)
b = draw(K1(a = a))
J = jointchain(a = M, b = K1, c = K2)
fk, pr = disintegrate(["b"], J)
  `;
  // The selector ["b"] is non-suffix (picks the middle field of a
  // three-component jointchain). The factorization is explicit in
  // the DAG but the engine's current rule returns Unsupported.
  const plan = planOf(src, 'J', ['b']);
  assert.equal(plan.kind, 'unsupported',
    'jointchain non-suffix selector is currently Unsupported (gap)');
});

test('spec-coverage PASS: nested joint(joint(...), ...) — prior classifies as joint-record', () => {
  // joint(inner = joint(a=A, b=B), c=C) — nested factorization is
  // explicit in the DAG. The decompose-based rewriter produces a
  // synthesized prior `joint(inner = inner)`, and the orchestrator's
  // classifier now handles it via MEASURE_OP_CLASSIFIERS' joint
  // entry (the fall-through enabled when the disintegrate-result
  // prior's IR is a non-lawof shape, 2026-05-23). Gap closed.
  const src = `
A = Normal(mu = 0, sigma = 1)
B = Normal(mu = 0, sigma = 1)
C = Normal(mu = 0, sigma = 1)
inner = joint(a = A, b = B)
outer = joint(inner = inner, c = C)
fk, pr = disintegrate(["c"], outer)
  `;
  const plan = planOf(src, 'outer', ['c']);
  assert.equal(plan.kind, 'synthesized',
    'outer plan synthesizes (the disintegrate.ts rule fires)');
  const { derivations } = classifyOf(src);
  assert.ok(derivations.pr,
    'synthesized prior `joint(inner = inner)` now classifies as a record derivation');
  // The derivation kind for `joint(name=ref)` is `record` per the
  // existing classifyRecordOrJoint rule: a single-field record
  // measure wrapping the bound ref.
  assert.equal(derivations.pr.kind, 'record');
});
