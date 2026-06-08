'use strict';

// =====================================================================
// arrow-kernel-round2.test.ts
// =====================================================================
// TDD (red-first) for the SECOND-round review findings on PR #16
// (fix/arrow-kernel-broadcast). These express the DESIRED post-fix
// behaviour; most FAIL today. Each test's trailing comment tags the
// finding and the observed pre-fix failure.
//
// Findings covered:
//   [H1] generative-bodied ARROW kernel broadcast must MATERIALISE
//        (today: classifies as kernelbroadcast but throws
//        "unbound %local reference '_x_'" at materialise).
//   [M2] an arrow lambda whose body is a measure must infer
//        inferredType.kind === 'kernel' (iid/joint/jointchain/nested/
//        broadcast-bodied), and a deterministic arrow must STAY
//        'function'.
//   [M1] arrow kernels over the wider measure-constructor set
//        (kchain) must be recognised as kernel heads; the
//        signatures.bodyImpliesKernel predicate must return true for
//        joint/jointchain/broadcast-of-dist/kchain bodies and false for
//        a plain value body.
//   [L1] the inner-dist positional+keyword collision error must NOT
//        claim spec authority ("spec §04").
//
// Syntax notes (mirrored from the passing arrow tests):
//   - single-arg paren lambda `(x) -> …` is a PARSE ERROR; single-arg
//     kernels must be bare `x -> …`.
//   - the dotted call `K.(args)` lowers to `broadcast(K, args)`.
//   - the generative draw primitive is `draw(...)`; `Uniform(interval(a,b))`.
//
// NOTE: fixtures and module resolution only work when run from
// packages/engine (`cd packages/engine && node --test test/...`).

const { test } = require('node:test');
const assert = require('node:assert');
const { makeMatCtx, expectPlottable } = require('./_materialise-helpers.ts');
const { processSource } = require('..');
const orchestrator = require('../orchestrator.ts');
const derivations = require('../derivations.ts');
const signatures = require('../signatures.ts');

// --- shared helpers (mirror the author's + verify suites) -------------

// The dotted call `K.(args)` lowers to an anonymous `broadcast(K, args)`
// binding. Find it and return its classify `kind`.
function classifyKernelBroadcastKind(src: string, kernelName: string): any {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  let bcBinding: any = null;
  for (const [, v] of built.bindings) {
    if (v.ir && v.ir.op === 'broadcast' && Array.isArray(v.ir.args)
        && v.ir.args[0] && v.ir.args[0].name === kernelName) {
      bcBinding = v;
      break;
    }
  }
  if (!bcBinding) return { found: false, kind: null };
  const d = derivations.classifyDerivation(
    bcBinding, built.bindings, built.fixedValues);
  return { found: true, kind: d ? d.kind : null, distOp: d && d.distOp };
}

// inferredType.kind for a binding, as produced by typeinfer at
// processSource time (this is the field the M2 finding is about). No
// orchestrator pass: the type kind must be settled BEFORE derivations.
function inferredKind(src: string, bindingName: string): any {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `source should have no errors; got: ${errs.map((d: any) => d.message).join('; ')}`);
  const b = lifted.bindings.get(bindingName);
  assert.ok(b, `binding ${bindingName} should exist`);
  return b.inferredType ? b.inferredType.kind : null;
}

// Resolve a built (post-orchestrator) binding's functionof IR body so a
// bodyImpliesKernel unit test can feed it the SAME node shape the
// fallback path sees in production.
function builtBodyOf(src: string, bindingName: string): { body: any; bindings: any } {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const b = built.bindings.get(bindingName);
  if (!b || !b.ir) return { body: null, bindings: built.bindings };
  return { body: b.ir.body, bindings: built.bindings };
}

// =====================================================================
// [H1] Generative-bodied ARROW kernel broadcast must MATERIALISE.
//
// PRE-FIX (RED): classifies as kernelbroadcast (the recogniser peels the
// lawof) but materialise throws
//   "evaluateExprN: unbound %local reference '_x_' — env must provide
//    values for all upstream-resolved names"
// The arrow lambda's surface param is an internal `%local` (`_x_`) that
// the generative executor fails to re-root onto the broadcast arg.
// =====================================================================

test('[H1] generative ARROW kernel (1 param) materialises + per-cell mean tracks a[i]+1', async () => {
  const src = `
a = [0.5, 1.0, 1.5]
gen_K = x -> lawof(x + 2.0 * draw(Uniform(interval(0.0, 1.0))))
y ~ gen_K.(a)
`;
  // Recogniser already accepts it; the open question is whether it RUNS.
  const r = classifyKernelBroadcastKind(src, 'gen_K');
  assert.equal(r.kind, 'kernelbroadcast',
    'generative arrow form is recognised as a kernel broadcast');
  const { ctx } = makeMatCtx(src, { sampleCount: 4000 });
  const m = await expectPlottable(ctx, 'y');
  const sh = m.value.shape;
  const Ns = sh[0];
  const Kc = sh[sh.length - 1];
  assert.equal(Kc, 3, 'K = 3 cells (one per element of a)');
  // Per cell j: mean ≈ a[j] + 2*E[Uniform(0,1)] = a[j] + 1.
  const a = [0.5, 1.0, 1.5];
  const dd = m.value.data;
  for (let j = 0; j < Kc; j++) {
    let s = 0;
    for (let i = 0; i < Ns; i++) s += dd[i * Kc + j];
    assert.ok(Math.abs(s / Ns - (a[j] + 1.0)) < 0.15,
      `cell ${j} mean ≈ ${a[j] + 1.0} (a[${j}] + E[2U]); got ${(s / Ns).toFixed(3)}`);
  }
});

test('[H1] generative ARROW kernel (2 params) materialises + per-cell mean tracks a[i]+b[i]+1', async () => {
  const src = `
a = [0.5, 1.0, 1.5]
b = [10.0, 20.0, 30.0]
gen_K = (x, c) -> lawof(x + c + 2.0 * draw(Uniform(interval(0.0, 1.0))))
y ~ gen_K.(a, b)
`;
  const r = classifyKernelBroadcastKind(src, 'gen_K');
  assert.equal(r.kind, 'kernelbroadcast',
    '2-param generative arrow form is recognised as a kernel broadcast');
  const { ctx } = makeMatCtx(src, { sampleCount: 4000 });
  const m = await expectPlottable(ctx, 'y');
  const sh = m.value.shape;
  const Ns = sh[0];
  const Kc = sh[sh.length - 1];
  assert.equal(Kc, 3, 'K = 3 cells (one per outer element)');
  const a = [0.5, 1.0, 1.5];
  const b = [10.0, 20.0, 30.0];
  const dd = m.value.data;
  for (let j = 0; j < Kc; j++) {
    let s = 0;
    for (let i = 0; i < Ns; i++) s += dd[i * Kc + j];
    const want = a[j] + b[j] + 1.0;
    assert.ok(Math.abs(s / Ns - want) < 0.2,
      `cell ${j} mean ≈ ${want}; got ${(s / Ns).toFixed(3)}`);
  }
});

// =====================================================================
// [M2] An arrow lambda whose body is a MEASURE must infer
// inferredType.kind === 'kernel'.
//
// OBSERVED today (probed): iid/joint/jointchain arrow bodies ALREADY
// infer 'kernel'; the BROADCAST-of-dist arrow body still infers
// 'function' (RED), and a kchain arrow body infers 'failed' (RED). A
// deterministic arrow stays 'function' (the negative guard, GREEN).
// All five are pinned here so the boundary is unambiguous after the fix.
// =====================================================================

test('[M2] iid-bodied arrow infers kind=kernel', () => {
  const src = `
N = 4
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
`;
  assert.equal(inferredKind(src, 'beta_row_K'), 'kernel',
    'iid-bodied arrow lambda is a kernel');
});

test('[M2] joint-bodied arrow infers kind=kernel', () => {
  const src = `
joint_K = (a, b) -> joint(x = Normal(mu = a, sigma = 1.0), y = Exponential(rate = b))
`;
  assert.equal(inferredKind(src, 'joint_K'), 'kernel',
    'joint-bodied arrow lambda is a kernel');
});

test('[M2] jointchain-bodied arrow infers kind=kernel', () => {
  const src = `
step_K = kernelof(Normal(mu = prev, sigma = 0.5), prev = prev)
chain_K = x0 -> jointchain(Normal(mu = x0, sigma = 0.1), step_K, step_K)
`;
  assert.equal(inferredKind(src, 'chain_K'), 'kernel',
    'jointchain-bodied arrow lambda is a kernel');
});

test('[M2] nested broadcast-of-dist-bodied arrow infers kind=kernel', () => {
  // RED today: a `broadcast(Normal, …)` body infers kind='function'.
  const src = `
mus = [1.0, 2.0, 3.0]
patient_K = sigma_g -> broadcast(Normal, mu = mus, sigma = sigma_g)
`;
  assert.equal(inferredKind(src, 'patient_K'), 'kernel',
    'broadcast-of-distribution-bodied arrow lambda is a kernel');
});

test('[M2] (negative guard) deterministic arrow STAYS kind=function', () => {
  const src = `
f = (m, s) -> m + s
`;
  assert.equal(inferredKind(src, 'f'), 'function',
    'deterministic arrow lambda is a value function, not a kernel');
});

// =====================================================================
// [M1] Arrow kernels over the wider measure-constructor set must be
// recognised as kernel heads.
//
// `markovchain` is NOT a recognised engine op (no parser/typeinfer/
// executor entry — see report); the measure-algebra chain constructor
// that DOES exist and mirrors valid usage in test/fixtures is `kchain`
// (test/fixtures/disintegrate-complex.flatppl). So M1 is exercised via
// `kchain`. If kchain has no broadcast executor yet, the inferredType
// kind assertion (not full materialise) is the contract.
//
// OBSERVED today (probed): a kchain-bodied arrow infers kind='failed'
// (RED). DESIRED: kind === 'kernel'.
// =====================================================================

test('[M1] kchain-bodied arrow infers kind=kernel (not misclassified as value)', () => {
  // A kchain whose first arg is a distribution and whose subsequent args
  // are kernels — mirrors the valid measure-algebra usage in the
  // disintegrate fixtures, lifted into arrow form.
  const src = `
fwd_k = kernelof(Normal(mu = prev, sigma = 1.0), prev = prev)
chain_K = x -> kchain(Normal(mu = x, sigma = 1.0), fwd_k)
`;
  assert.equal(inferredKind(src, 'chain_K'), 'kernel',
    'kchain-bodied arrow lambda is a kernel head, not a value');
});

test('[M1] signatures.bodyImpliesKernel: true for joint body', () => {
  const { body, bindings } = builtBodyOf(
    `K = (a, b) -> joint(x = Normal(mu = a, sigma = 1.0), y = Exponential(rate = b))`,
    'K');
  assert.equal(signatures.bodyImpliesKernel(body, bindings), true,
    'joint body implies a kernel');
});

test('[M1] signatures.bodyImpliesKernel: true for jointchain body', () => {
  // RED today: jointchain is not in KNOWN_MEASURE_OPS.
  const { body, bindings } = builtBodyOf(
    `step_K = kernelof(Normal(mu = prev, sigma = 0.5), prev = prev)
K = x0 -> jointchain(Normal(mu = x0, sigma = 0.1), step_K, step_K)`,
    'K');
  assert.equal(signatures.bodyImpliesKernel(body, bindings), true,
    'jointchain body implies a kernel');
});

test('[M1] signatures.bodyImpliesKernel: true for broadcast-of-dist body', () => {
  // RED today: broadcast is not in KNOWN_MEASURE_OPS.
  const { body, bindings } = builtBodyOf(
    `mus = [1.0, 2.0]
K = s -> broadcast(Normal, mu = mus, sigma = s)`,
    'K');
  assert.equal(signatures.bodyImpliesKernel(body, bindings), true,
    'broadcast-of-distribution body implies a kernel');
});

test('[M1] signatures.bodyImpliesKernel: true for kchain body', () => {
  // RED today: kchain is not in KNOWN_MEASURE_OPS.
  const { body, bindings } = builtBodyOf(
    `fwd_k = kernelof(Normal(mu = prev, sigma = 1.0), prev = prev)
K = x -> kchain(Normal(mu = x, sigma = 1.0), fwd_k)`,
    'K');
  assert.equal(signatures.bodyImpliesKernel(body, bindings), true,
    'kchain body implies a kernel');
});

test('[M1] (negative guard) signatures.bodyImpliesKernel: false for a plain value body', () => {
  const { body, bindings } = builtBodyOf(`K = (m, s) -> m + s`, 'K');
  assert.equal(signatures.bodyImpliesKernel(body, bindings), false,
    'a plain arithmetic value body does NOT imply a kernel');
});

// =====================================================================
// [L1] The inner-dist positional+keyword collision error message must
// NOT claim spec authority. The collision IS an engine policy (the
// kernel-broadcast executor's kwargs-only path), not a spec mandate, so
// the message must not cite "spec §04".
//
// PRE-FIX (RED): the thrown message reads
//   "…a parameter may be bound at most once per call (spec §04); remove
//    one binding of 'alpha'."
// DESIRED: the SAME actionable, param-naming error WITHOUT the bogus
// "spec §04" citation (and ideally signalling it is an engine policy).
// =====================================================================

test('[L1] inner-dist collision error names the param but does NOT cite "spec §04"', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
bb = [5.0, 4.0]
K = (a_g, b_g) -> iid(Beta(a_g, alpha = 2.0), N)
p ~ K.(a, b_g = bb)
`;
  let caught: any = null;
  try {
    const { ctx } = makeMatCtx(src, { sampleCount: 64 });
    await ctx.getMeasure('p');
  } catch (e) {
    caught = e;
  }
  assert.ok(caught, 'inner-dist collision must still throw a clear error');
  const msg = String(caught.message || caught);
  // Still names the doubly-bound param (the diagnostic stays actionable).
  assert.match(msg, /alpha/,
    'collision error must still name the doubly-bound param `alpha`');
  // But must NOT claim spec authority for what is an engine policy.
  assert.doesNotMatch(msg, /spec §04|spec §4|§04/,
    'collision error must NOT cite "spec §04" (this is an engine policy, not a spec mandate)');
  // Ideally it signals the engine-policy origin.
  assert.match(msg, /engine|kernel-broadcast/i,
    'collision error should signal it is an engine policy');
});
