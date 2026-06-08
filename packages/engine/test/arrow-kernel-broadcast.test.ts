'use strict';

// Regression: a measure-bodied ARROW lambda `(args) -> <measure>` is a
// kernel (spec §04: a lambda is `functionof` sugar; the engine treats
// "functionof on a measure" as a kernel, analyzer.ts). Broadcasting it
// must classify as a kernel-broadcast, NOT a deterministic value
// broadcast.
//
// Before the fix the arrow form lowered to `functionof(<measure>, …)`
// WITHOUT the `lawof` wrapper that `kernelof` inserts
// (`kernelof(x) ≡ functionof(lawof(x))`), and every composite
// kernel-broadcast detector in kernel-broadcast-shape.ts hard-required
// `body.op === 'lawof'`. So `p ~ beta_row_K.(a, b)` classified as
// kind=evaluate, the measure body was inlined into the value evaluator,
// and the worker threw "call op 'iid' not evaluable in sampler
// context". Three sub-gaps, all fixed:
//   1. detectors peel an optional `lawof` so the arrow form is
//      recognised (kernel-broadcast-shape.peelKernelBody).
//   2. positional inner-dist args (`Beta(a_g, b_g)`) normalise to named
//      kwargs for the executor (distKwargsWithPositional).
//   3. positional OUTER broadcast args (`K.(a, b)`) bind to the
//      kernel's surface params by position
//      (mat-broadcast._normalizeCompositeBroadcastArgs).

const { test } = require('node:test');
const assert = require('node:assert');
const { makeMatCtx, expectPlottable } = require('./_materialise-helpers.ts');
const { processSource } = require('..');
const orchestrator = require('../orchestrator.ts');
const derivations = require('../derivations.ts');

// --- shared helpers for the post-fix expectation tests below ----------
//
// The dot-call `K.(args)` lowers to an anonymous `broadcast(K, args)`
// binding (spec §05). Find it and return its classify `kind`.
// `classifyKind` walks the lowered bindings (mirrors the pattern in
// generative-kernel-broadcast.test.ts) so the negative/positive
// classification guards do not depend on materialisation.
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

// Classify a NAMED binding directly (for the deterministic case, whose
// dotted broadcast inlines into the value binding `y` rather than an
// anon broadcast binding).
function classifyNamedKind(src: string, bindingName: string): any {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const b = built.bindings.get(bindingName);
  if (!b) return null;
  const d = derivations.classifyDerivation(b, built.bindings, built.fixedValues);
  return d ? d.kind : null;
}

test('arrow-form iid-composite kernel broadcast (positional dist args) materialises', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 256 });
  await expectPlottable(ctx, 'p');
});

test('arrow-form iid-composite kernel broadcast (kwarg dist args) materialises', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
beta_row_K = (a_g, b_g) -> iid(Beta(alpha = a_g, beta = b_g), N)
p ~ beta_row_K.(a, b)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 256 });
  await expectPlottable(ctx, 'p');
});

// The litter hierarchical Beta-Binomial (the reported repro). The arrow
// kernel `beta_row_K` now samples and the prior (a record over it)
// materialises — the originally-reported "iid not evaluable" crash is
// gone. The full posterior additionally needs broadcast over 2D [G, N]
// collection args (the binomial likelihood) on BOTH the sample and
// density sides, which is a separate v1 broadcast-shape limitation
// (kernel/dist broadcast "supports scalar / [K] / [N] / [N, K]"), not
// the arrow-kernel recognition bug fixed here.
test('litter Beta-Binomial: arrow kernel + prior materialise (reported crash fixed)', async () => {
  const src = `
G = 2
N = 16
n_data = [[13,12,9,9,8,8,13,12,10,10,9,13,5,7,10,10],[12,11,10,9,11,10,10,9,9,5,9,7,10,6,10,7]]
pareto = pushfwd(fn(0.1 * exp(_)), Exponential(1.5))
a_plus_b ~ iid(pareto, G)
mu ~ iid(Beta(1, 1), G)
a = mu .* a_plus_b
b = (1 .- mu) .* a_plus_b
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
prior = lawof(record(a_plus_b = a_plus_b, mu = mu, p = p))
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 128 });
  // The exact thing that used to throw "iid not evaluable in sampler
  // context" — both now succeed.
  await expectPlottable(ctx, 'p');
  const prior = await ctx.getMeasure('prior');
  assert.ok(prior, 'prior should materialise');
});

// =====================================================================
// POST-FIX EXPECTATIONS (TDD). The tests below express the DESIRED
// behaviour after the PR #16 follow-up review. Most are RED today; the
// inline comment on each tags the finding and the observed pre-fix
// failure. A couple are GREEN regression guards (so marked).
// =====================================================================

// ---------------------------------------------------------------------
// [H1] Arrow-form JOINT-bodied kernel broadcast must materialise.
//
// Mirrors joint-composite-batch-flatten.test.ts but in ARROW form:
//   K = (a, b) -> joint(x = Normal(mu = a, sigma = 1), y = Exponential(rate = b))
// The kernelof control materialises to [N, cells, components]; the arrow
// form denotes the SAME kernel and must too.
//
// PRE-FIX (RED): throws
//   "evaluateExpr: call op 'joint' not evaluable in sampler context —
//    the orchestrator should pre-resolve this"
// The arrow body (no `lawof` wrap) is recognised as a joint kernel by
// peelKernelBody, but the joint composite executor is not actually
// reached / the body leaks into the value evaluator.
// ---------------------------------------------------------------------

test('[H1] arrow JOINT kernel (keyword components, positional outer) materialises', async () => {
  const src = `
arrA = [1.0, 2.0]
arrB = [0.5, 1.0]
K = (a, b) -> joint(x = Normal(mu = a, sigma = 1.0), y = Exponential(rate = b))
z ~ K.(arrA, arrB)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 4000 });
  const m = await expectPlottable(ctx, 'z');
  const [Ns, K, C] = m.value.shape;
  assert.equal(K, 2, 'two cells (one per outer arg element)');
  assert.equal(C, 2, 'two joint components (x, y)');
  // Calibration: component x ~ Normal(arrA[j], 1); y ~ Exponential(arrB[j]).
  const arrA = [1.0, 2.0];
  const arrB = [0.5, 1.0];
  const d = m.value.data;
  for (let j = 0; j < 2; j++) {
    let sx = 0; let sy = 0;
    for (let i = 0; i < Ns; i++) {
      sx += d[i * K * C + j * C + 0];
      sy += d[i * K * C + j * C + 1];
    }
    assert.ok(Math.abs(sx / Ns - arrA[j]) < 0.12,
      `cell ${j} x mean ≈ ${arrA[j]}; got ${(sx / Ns).toFixed(3)}`);
    assert.ok(Math.abs(sy / Ns - 1 / arrB[j]) < 0.2,
      `cell ${j} y mean ≈ ${(1 / arrB[j]).toFixed(3)} (Exp mean); got ${(sy / Ns).toFixed(3)}`);
  }
});

test('[H1] arrow JOINT kernel (positional components) materialises', async () => {
  const src = `
arrA = [1.0, 2.0]
arrB = [0.5, 1.0]
K = (a, b) -> joint(Normal(mu = a, sigma = 1.0), Exponential(rate = b))
z ~ K.(arrA, arrB)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 256 });
  const m = await expectPlottable(ctx, 'z');
  assert.equal(m.value.shape[m.value.shape.length - 1], 2, 'two joint components');
});

test('[H1] arrow JOINT kernel (keyword OUTER broadcast args) materialises', async () => {
  // Outer args bound by KEYWORD to the kernel's surface params a, b.
  const src = `
arrA = [1.0, 2.0]
arrB = [0.5, 1.0]
K = (a, b) -> joint(x = Normal(mu = a, sigma = 1.0), y = Exponential(rate = b))
z ~ K.(a = arrA, b = arrB)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 256 });
  await expectPlottable(ctx, 'z');
});

// ---------------------------------------------------------------------
// [H1] Arrow-form JOINTCHAIN-bodied kernel broadcast must materialise.
//
// Mirror of test/fixtures/hierarchical-state-space.flatppl
// (jointchain-batch-flatten.test.ts), in ARROW form. NOTE the single-arg
// lambda MUST be written `x0 -> …` (bare name); the paren form
// `(x0) -> …` is a parse error (spec §05: paren-form lambdas need ≥ 2
// names), which is a separate parser limitation, not the recognition bug.
//
// PRE-FIX (RED): throws
//   "evaluateExpr: call op 'jointchain' not evaluable in sampler context"
// ---------------------------------------------------------------------

test('[H1] arrow JOINTCHAIN kernel broadcast materialises (AR-1 carry, shape [N,3,4])', async () => {
  const src = `
x0_per_group = [0.0, 0.5, 1.0]
step_kernel = kernelof(Normal(mu = prev, sigma = 0.5), prev = prev)
group_chain = x0 -> jointchain(Normal(mu = x0, sigma = 0.1), step_kernel, step_kernel, step_kernel)
y ~ group_chain.(x0_per_group)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 4000 });
  const m = await expectPlottable(ctx, 'y');
  assert.deepEqual(m.value.shape, [4000, 3, 4], 'shape [N, groups, chain length]');
  // AR-1 carry: increments x_k - x_{k-1} ~ Normal(0, 0.5).
  const G = 3; const C = 4; const d = m.value.data; const N = 4000;
  let si = 0; let sq = 0; let ni = 0;
  for (let g = 0; g < G; g++) {
    for (let k = 1; k < C; k++) {
      for (let i = 0; i < N; i++) {
        const inc = d[i * G * C + g * C + k] - d[i * G * C + g * C + (k - 1)];
        si += inc; sq += inc * inc; ni++;
      }
    }
  }
  const incMean = si / ni;
  const incStd = Math.sqrt(sq / ni - incMean * incMean);
  assert.ok(Math.abs(incMean) < 0.05, `increment mean ≈ 0; got ${incMean.toFixed(4)}`);
  assert.ok(Math.abs(incStd - 0.5) < 0.06,
    `increment std ≈ 0.5 (carry threads x_{k-1}); got ${incStd.toFixed(4)}`);
});

// ---------------------------------------------------------------------
// [H1] Arrow-form NESTED broadcast kernel must materialise.
//
// Mirror of test/fixtures/nested-broadcast.flatppl
// (nested-broadcast-batch-flatten.test.ts), in ARROW form.
//
// PRE-FIX (RED): throws
//   "broadcast: nested-broadcast 'y' has a non-static axis ladder
//    (outer=null, inner=3) — folding a symbolic nested ladder is not yet
//    supported …"
// The kernelof control (kernelof(broadcast(Normal,…),…)) materialises to
// [N,3,4]; the arrow form leaves the OUTER axis size unresolved (null).
// ---------------------------------------------------------------------

test('[H1] arrow NESTED-broadcast kernel materialises (shape [N,3,4])', async () => {
  const src = `
visit_means = [10.0, 20.0, 30.0, 40.0]
sigmas_per_patient = [0.5, 1.0, 1.5]
patient_kernel = sigma_g -> broadcast(Normal, mu = visit_means, sigma = sigma_g)
y = broadcast(patient_kernel, sigma_g = sigmas_per_patient)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 4000 });
  const m = await expectPlottable(ctx, 'y');
  assert.deepEqual(m.value.shape, [4000, 3, 4], 'shape [N, patients, visits]');
  // Per visit column: mean ≈ visit_means[v]; per patient row: σ ≈ sigmas[p].
  const visitMeans = [10, 20, 30, 40];
  const sigmas = [0.5, 1.0, 1.5];
  const P = 3; const Vn = 4; const d = m.value.data; const N = 4000;
  for (let p = 0; p < P; p++) {
    for (let v = 0; v < Vn; v++) {
      let s = 0;
      for (let i = 0; i < N; i++) s += d[i * P * Vn + p * Vn + v];
      assert.ok(Math.abs(s / N - visitMeans[v]) < 0.3 * (sigmas[p] + 0.2),
        `cell (p${p},v${v}) mean ≈ ${visitMeans[v]}; got ${(s / N).toFixed(3)}`);
    }
  }
});

// ---------------------------------------------------------------------
// [H2] Classification guards — deterministic arrow lambda is NOT a kernel
// broadcast; the iid arrow form IS. These are GREEN regression guards
// (they already classify correctly today) — they pin that the recogniser
// stays discriminating and a future change cannot blur the boundary.
// ---------------------------------------------------------------------

test('[H2] (regression GREEN) deterministic arrow lambda f.(a,b) classifies as evaluate, NOT kernelbroadcast', () => {
  const src = `
a = [2.0, 3.0]
b = [5.0, 4.0]
f = (m, s) -> m + s
y = f.(a, b)
`;
  // The deterministic dotted broadcast inlines into the value binding `y`.
  const kind = classifyNamedKind(src, 'y');
  assert.equal(kind, 'evaluate',
    'deterministic arrow broadcast is a value-evaluate, not a kernel broadcast');
});

test('[H2] (regression GREEN) iid arrow kernel K.(a,b) classifies as kernelbroadcast', () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
`;
  const r = classifyKernelBroadcastKind(src, 'beta_row_K');
  assert.ok(r.found, 'found the anon broadcast binding for the kernel');
  assert.equal(r.kind, 'kernelbroadcast', 'iid arrow kernel is a kernel broadcast');
});

// ---------------------------------------------------------------------
// [H3] A positional+keyword COLLISION on the SAME inner-dist param must
// raise a CLEAR static error naming the param — not silently override.
//
// `Beta(a_g, alpha = 2.0)` targets `alpha` both positionally (first
// positional arg of Beta is `alpha`) and by keyword. distKwargsWithPositional
// keeps the explicit kwarg and silently drops the positional (kwargs win),
// so the per-cell `a_g` boundary is silently lost.
//
// PRE-FIX (RED): no static diagnostic; materialise throws an UNCLEAR
//   "evaluateExprN: unbound self reference 'bb' …" rather than naming
//   the doubly-bound param.
// DESIRED: a clear error mentioning the param name `alpha` (raised either
//   as an analyze-time diagnostic OR a materialise-time throw).
// ---------------------------------------------------------------------

test('[H3] inner-dist positional+keyword collision on alpha errors clearly (names the param)', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
bb = [5.0, 4.0]
K = (a_g, b_g) -> iid(Beta(a_g, alpha = 2.0), N)
p ~ K.(a, b_g = bb)
`;
  // Accept either a static diagnostic (caught by makeMatCtx) or a
  // materialise-time throw; in BOTH cases the message must name `alpha`.
  await assert.rejects(
    (async () => {
      const { ctx } = makeMatCtx(src, { sampleCount: 64 });
      await ctx.getMeasure('p');
    })(),
    /alpha/,
    'collision error should name the doubly-bound param `alpha`',
  );
});

test('[H3] outer broadcast positional+keyword collision (K.(a, a_g=b)) errors clearly (names a_g)', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ K.(a, a_g = b)
`;
  // `a` (positional → a_g) and `a_g = b` (keyword) both target a_g.
  await assert.rejects(
    (async () => {
      const { ctx } = makeMatCtx(src, { sampleCount: 64 });
      await ctx.getMeasure('p');
    })(),
    /a_g/,
    'collision error should name the doubly-bound surface param `a_g`',
  );
});

// ---------------------------------------------------------------------
// [H6] MIXED positional + keyword OUTER broadcast args that do NOT
// collide must materialise correctly: `K.(a, b_g = b)` — positional `a`
// binds to the first surface param `a_g`, keyword `b_g` binds the second.
//
// PRE-FIX (RED): throws
//   "evaluateExprN: unbound %local reference '_a_g_' …" —
// _normalizeCompositeBroadcastArgs only maps positional args when there
// are NO kwargs at all (`hasKw || argIRs.length === 0 → return d`), so a
// mixed call drops the positional `a` and leaves `a_g` unbound.
// DESIRED: positional `a` binds (a_g), keyword `b` binds (b_g); the
// result calibrates to Beta(a, b) per group — identical to the all-
// positional / all-keyword forms.
// ---------------------------------------------------------------------

test('[H6] mixed positional+keyword outer args (K.(a, b_g=b)) bind correctly and calibrate', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ K.(a, b_g = b)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 8000 });
  const m = await expectPlottable(ctx, 'p');
  // Per-atom shape [G, N] = [2, 4]; positional `a` MUST have bound a_g.
  const sh = m.value.shape;
  const [Ns, G, Nn] = sh;
  assert.equal(G, 2, 'two groups (a_g binding kept the positional arg)');
  assert.equal(Nn, 4, 'iid count N = 4');
  // Group 0 Beta(2,5) mean ≈ 0.2857; group 1 Beta(3,4) mean ≈ 0.4286.
  const d = m.value.data;
  const expect = [2 / 7, 3 / 7];
  for (let g = 0; g < G; g++) {
    let s = 0;
    for (let i = 0; i < Ns; i++) for (let j = 0; j < Nn; j++) s += d[i * G * Nn + g * Nn + j];
    assert.ok(Math.abs(s / (Ns * Nn) - expect[g]) < 0.02,
      `group ${g} mean ≈ ${expect[g].toFixed(4)} (positional a bound, not dropped); `
      + `got ${(s / (Ns * Nn)).toFixed(4)}`);
  }
});

// ---------------------------------------------------------------------
// [M3] VALUE/SHAPE assertions for the iid arrow case already exercised
// above. Pins the per-atom shape [G, N] = [2, 4] and per-cell empirical
// means tracking DISTINCT params (Beta(2,5) mean ≈ 0.286 vs Beta(3,4)
// mean ≈ 0.429). A positional→param swap or a per-group collapse would
// move the means and be caught here. This GREEN today (the iid path is
// what PR #16 fixed); it is the value-level regression guard.
// ---------------------------------------------------------------------

test('[M3] (regression GREEN) iid arrow case: per-atom shape [2,4] and per-group means track distinct params', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 8000 });
  const m = await expectPlottable(ctx, 'p');
  const sh = m.value.shape;
  assert.equal(sh.length, 3, 'value shape is [Nsamp, G, N]');
  const [Ns, G, Nn] = sh;
  assert.equal(G, 2, 'per-atom outer dim G = 2');
  assert.equal(Nn, 4, 'per-atom inner dim N = 4');
  const d = m.value.data;
  const expect = [2 / 7, 3 / 7];   // 0.2857, 0.4286 — DISTINCT
  const means: number[] = [];
  for (let g = 0; g < G; g++) {
    let s = 0;
    for (let i = 0; i < Ns; i++) for (let j = 0; j < Nn; j++) s += d[i * G * Nn + g * Nn + j];
    means.push(s / (Ns * Nn));
    assert.ok(Math.abs(means[g] - expect[g]) < 0.02,
      `group ${g} mean ≈ ${expect[g].toFixed(4)}; got ${means[g].toFixed(4)}`);
  }
  assert.ok(Math.abs(means[0] - means[1]) > 0.1,
    'group means are DISTINCT (a swap/collapse would equalise them)');
});

// ---------------------------------------------------------------------
// [M4] Arity / edge cases. The correctness review says bad arity already
// errors loudly — these pin the ACTUAL clear-error behaviour so a future
// change cannot regress it into a silent miscompute. These are GREEN
// guards (they already throw clearly today).
// ---------------------------------------------------------------------

test('[M4] (regression GREEN) too few outer args (2-param kernel called with 1) errors loudly', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ K.(a)
`;
  await assert.rejects(
    (async () => {
      const { ctx } = makeMatCtx(src, { sampleCount: 64 });
      await ctx.getMeasure('p');
    })(),
    /b_g|positional|expected 2|arity|unbound/i,
    'under-applying the kernel must error loudly (no silent miscompute)',
  );
});

test('[M4] (RED desired-behaviour) too many outer args (2-param kernel called with 3) errors loudly', async () => {
  // NOTE (ambiguity flagged to fixers): pre-fix this does NOT throw — the
  // extra positional arg is silently dropped and a [N,2,4] measure
  // materialises. DESIRED behaviour is a loud arity error. This test is
  // therefore RED today (it asserts a throw that does not happen).
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
c = [1.0, 1.0]
K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ K.(a, b, c)
`;
  await assert.rejects(
    (async () => {
      const { ctx } = makeMatCtx(src, { sampleCount: 64 });
      await ctx.getMeasure('p');
    })(),
    /expected 2|too many|arity|3 positional|extra/i,
    'over-applying the kernel must error loudly (extra arg must not be silently dropped)',
  );
});

test('[M4] (regression GREEN) empty outer args (K.()) errors loudly', async () => {
  const src = `
N = 4
K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ K.()
`;
  await assert.rejects(
    (async () => {
      const { ctx } = makeMatCtx(src, { sampleCount: 64 });
      await ctx.getMeasure('p');
    })(),
    /expected 2 positional arrays, got 0|arity|positional/i,
    'calling a kernel with no args must error loudly',
  );
});
