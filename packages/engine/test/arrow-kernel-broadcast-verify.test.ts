'use strict';

// =====================================================================
// arrow-kernel-broadcast-verify.test.ts
// =====================================================================
// INDEPENDENT adversarial verification of PR #16 (fix/arrow-kernel-
// broadcast). Written WITHOUT trusting the author's own suite: these
// cases probe edges the author's arrow-kernel-broadcast.test.ts does not
// cover, and actively try to break the iid/joint/jointchain/nested/
// generative arrow-kernel fixes and the positional/keyword binding +
// collision/arity logic.
//
// Syntax notes (mirrored from passing tests):
//   - single-arg paren lambda `(x) -> …` is a PARSE ERROR; single-arg
//     kernels must be bare `x -> …`.
//   - the dotted call `K.(args)` lowers to `broadcast(K, args)`.
//   - generative draw primitive is `draw(...)` (NOT `sample(...)`).
//
// Each test's trailing comment records what it PROVES or what GAP it
// exposes; see the reviewer's summary table for the consolidated verdict.

const { test } = require('node:test');
const assert = require('node:assert');
const { makeMatCtx, expectPlottable } = require('./_materialise-helpers.ts');
const { processSource } = require('..');
const orchestrator = require('../orchestrator.ts');
const derivations = require('../derivations.ts');

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

function classifyNamedKind(src: string, bindingName: string): any {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const b = built.bindings.get(bindingName);
  if (!b) return null;
  const d = derivations.classifyDerivation(b, built.bindings, built.fixedValues);
  return d ? d.kind : null;
}

// ---------------------------------------------------------------------
// V1. JOINT arrow kernel, 3+ components, MIXED non-colliding
//     positional+keyword OUTER args. Probes that the 3rd component is
//     not dropped AND mixed-arg binding extends past 2 params.
// ---------------------------------------------------------------------
test('V1 joint 3-component kernel, mixed positional+keyword outer args, correct shape+values', async () => {
  const src = `
arrA = [1.0, 2.0]
arrB = [0.5, 1.0]
arrC = [3.0, 4.0]
K = (a, b, c) -> joint(x = Normal(mu = a, sigma = 1.0), y = Exponential(rate = b), w = Normal(mu = c, sigma = 0.5))
z ~ K.(arrA, b = arrB, c = arrC)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 8000 });
  const m = await expectPlottable(ctx, 'z');
  const [Ns, Kc, C] = m.value.shape;
  assert.equal(Kc, 2, 'two cells (one per outer element)');
  assert.equal(C, 3, 'three joint components (x,y,w) — none dropped');
  const arrA = [1.0, 2.0]; const arrB = [0.5, 1.0]; const arrC = [3.0, 4.0];
  const dd = m.value.data;
  for (let j = 0; j < 2; j++) {
    let sx = 0; let sy = 0; let sw = 0;
    for (let i = 0; i < Ns; i++) {
      sx += dd[i * Kc * C + j * C + 0];
      sy += dd[i * Kc * C + j * C + 1];
      sw += dd[i * Kc * C + j * C + 2];
    }
    assert.ok(Math.abs(sx / Ns - arrA[j]) < 0.1,
      `cell ${j} x mean ≈ ${arrA[j]}; got ${(sx / Ns).toFixed(3)}`);
    assert.ok(Math.abs(sy / Ns - 1 / arrB[j]) < 0.2,
      `cell ${j} y mean ≈ ${(1 / arrB[j]).toFixed(3)}; got ${(sy / Ns).toFixed(3)}`);
    assert.ok(Math.abs(sw / Ns - arrC[j]) < 0.1,
      `cell ${j} w mean ≈ ${arrC[j]} (3rd component, keyword-bound); got ${(sw / Ns).toFixed(3)}`);
  }
});

// ---------------------------------------------------------------------
// V2. JOINTCHAIN (AR-style) arrow kernel — independent re-derivation of
//     the AR-1 carry statistics with a DIFFERENT step sigma + chain
//     length than the author's test, to confirm the carry is genuine.
// ---------------------------------------------------------------------
test('V2 jointchain arrow kernel: AR-1 carry holds with distinct sigma/length, shape [N,2,5]', async () => {
  const src = `
x0_per_group = [0.0, 2.0]
step_kernel = kernelof(Normal(mu = prev, sigma = 0.3), prev = prev)
chain = x0 -> jointchain(Normal(mu = x0, sigma = 0.05), step_kernel, step_kernel, step_kernel, step_kernel)
y ~ chain.(x0_per_group)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 6000 });
  const m = await expectPlottable(ctx, 'y');
  assert.deepEqual(m.value.shape, [6000, 2, 5], 'shape [N, groups, chain length]');
  const G = 2; const C = 5; const dd = m.value.data; const N = 6000;
  // First column tracks its group's x0 (mean ≈ x0_per_group[g]).
  const x0 = [0.0, 2.0];
  for (let g = 0; g < G; g++) {
    let s0 = 0;
    for (let i = 0; i < N; i++) s0 += dd[i * G * C + g * C + 0];
    assert.ok(Math.abs(s0 / N - x0[g]) < 0.05,
      `group ${g} col0 mean ≈ ${x0[g]}; got ${(s0 / N).toFixed(4)}`);
  }
  // AR-1 increments x_k - x_{k-1} ~ Normal(0, 0.3) across both groups.
  let si = 0; let sq = 0; let ni = 0;
  for (let g = 0; g < G; g++) {
    for (let k = 1; k < C; k++) {
      for (let i = 0; i < N; i++) {
        const inc = dd[i * G * C + g * C + k] - dd[i * G * C + g * C + (k - 1)];
        si += inc; sq += inc * inc; ni++;
      }
    }
  }
  const incMean = si / ni;
  const incStd = Math.sqrt(sq / ni - incMean * incMean);
  assert.ok(Math.abs(incMean) < 0.04, `increment mean ≈ 0; got ${incMean.toFixed(4)}`);
  assert.ok(Math.abs(incStd - 0.3) < 0.04,
    `increment std ≈ 0.3 (carry threads x_{k-1}); got ${incStd.toFixed(4)}`);
});

// ---------------------------------------------------------------------
// V3. NESTED arrow kernel where the OUTER broadcast arg is a NON-literal
//     COMPUTED binding (`sigmas = sa .* sb`). Probes the dissolver
//     `_argSizeIdentifier` numeric-shape path: a computed elementwise
//     array still carries a STATICALLY-inferred leading dim, so the
//     outer axis size resolves numerically (NOT via the symbolic-name
//     fallback). Must materialise to the same [N,3,4] as a literal outer.
// ---------------------------------------------------------------------
test('V3 nested arrow kernel with computed (.*) outer arg materialises [N,3,4] and calibrates', async () => {
  const src = `
visit_means = [10.0, 20.0, 30.0, 40.0]
sa = [0.5, 1.0, 1.5]
sb = [1.0, 1.0, 1.0]
sigmas = sa .* sb
patient_kernel = sigma_g -> broadcast(Normal, mu = visit_means, sigma = sigma_g)
y = broadcast(patient_kernel, sigma_g = sigmas)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 6000 });
  const m = await expectPlottable(ctx, 'y');
  assert.deepEqual(m.value.shape, [6000, 3, 4],
    'computed outer arg resolves to static leading dim 3 (NOT a null/symbolic axis)');
  const visitMeans = [10, 20, 30, 40]; const sigmas = [0.5, 1.0, 1.5];
  const P = 3; const Vn = 4; const dd = m.value.data; const N = 6000;
  for (let p = 0; p < P; p++) {
    for (let v = 0; v < Vn; v++) {
      let s = 0;
      for (let i = 0; i < N; i++) s += dd[i * P * Vn + p * Vn + v];
      assert.ok(Math.abs(s / N - visitMeans[v]) < 0.3 * (sigmas[p] + 0.2),
        `cell (p${p},v${v}) mean ≈ ${visitMeans[v]}; got ${(s / N).toFixed(3)}`);
    }
  }
});

// ---------------------------------------------------------------------
// V4. GENERATIVE-bodied arrow kernel (the claimed "M1" fix). The
//     CANONICAL kernelof form materialises; the ARROW form denotes the
//     SAME kernel and SHOULD too. This is the adversarial probe most
//     likely to expose a gap: the arrow lambda's surface param is an
//     internal `%local` (`_x_`) that the generative executor may fail to
//     re-root onto the broadcast arg.
//
//     Control (kernelof form) and the arrow form are BOTH asserted so
//     the table can attribute any divergence to the arrow path alone.
// ---------------------------------------------------------------------
test('V4a generative kernelof form (control) materialises [N,K]', async () => {
  const src = `
xs = [0.5, 1.0, 1.5]
x_in = elementof(reals)
yv = x_in + 2 * draw(Uniform(interval(0, 1)))
gen_K = kernelof(yv, x_in = x_in)
ys ~ gen_K.(xs)
`;
  const r = classifyKernelBroadcastKind(src, 'gen_K');
  assert.equal(r.kind, 'kernelbroadcast', 'generative kernelof form classifies as kernel broadcast');
  const { ctx } = makeMatCtx(src, { sampleCount: 2000 });
  const m = await expectPlottable(ctx, 'ys');
  assert.equal(m.value.shape[m.value.shape.length - 1], 3, 'K = 3 cells (one per xs element)');
});

test('V4b generative ARROW form materialises (claimed M1 fix)', async () => {
  const src = `
xs = [0.5, 1.0, 1.5]
gen_K = x -> lawof(x + 2 * draw(Uniform(interval(0, 1))))
ys ~ gen_K.(xs)
`;
  // It DOES classify as a kernel broadcast (recogniser peels the lawof);
  // the open question this test answers is whether it EXECUTES.
  const r = classifyKernelBroadcastKind(src, 'gen_K');
  assert.equal(r.kind, 'kernelbroadcast',
    'generative arrow form is recognised as a kernel broadcast');
  const { ctx } = makeMatCtx(src, { sampleCount: 2000 });
  const m = await expectPlottable(ctx, 'ys');
  assert.equal(m.value.shape[m.value.shape.length - 1], 3, 'K = 3 cells');
  // Calibration: per cell, mean ≈ xs[j] + 2*E[Uniform(0,1)] = xs[j] + 1.
  const xs = [0.5, 1.0, 1.5];
  const sh = m.value.shape; const Ns = sh[0]; const Kc = sh[1];
  const dd = m.value.data;
  for (let j = 0; j < Kc; j++) {
    let s = 0;
    for (let i = 0; i < Ns; i++) s += dd[i * Kc + j];
    assert.ok(Math.abs(s / Ns - (xs[j] + 1)) < 0.15,
      `cell ${j} mean ≈ ${xs[j] + 1}; got ${(s / Ns).toFixed(3)}`);
  }
});

test('V4c generative ARROW form, 2 params, materialises + per-cell calibrates', async () => {
  // ROUND-2 [H1]: the generative-arrow materialise gap is not specific to
  // arity. A 2-param generative arrow closes over BOTH surface params plus
  // an internal draw; per cell j: mean ≈ xs[j] + bs[j] + 2*E[U] = +1.
  const src = `
xs = [0.5, 1.0, 1.5]
bs = [10.0, 20.0, 30.0]
gen_K = (x, b) -> lawof(x + b + 2.0 * draw(Uniform(interval(0.0, 1.0))))
ys ~ gen_K.(xs, bs)
`;
  const r = classifyKernelBroadcastKind(src, 'gen_K');
  assert.equal(r.kind, 'kernelbroadcast',
    '2-param generative arrow is recognised as a kernel broadcast');
  const { ctx } = makeMatCtx(src, { sampleCount: 3000 });
  const m = await expectPlottable(ctx, 'ys');
  const sh = m.value.shape; const Ns = sh[0]; const Kc = sh[1];
  assert.equal(Kc, 3, 'K = 3 cells (one per outer element)');
  const xs = [0.5, 1.0, 1.5]; const bs = [10.0, 20.0, 30.0];
  const dd = m.value.data;
  for (let j = 0; j < Kc; j++) {
    let s = 0;
    for (let i = 0; i < Ns; i++) s += dd[i * Kc + j];
    const want = xs[j] + bs[j] + 1;
    assert.ok(Math.abs(s / Ns - want) < 0.2,
      `cell ${j} mean ≈ ${want}; got ${(s / Ns).toFixed(3)}`);
  }
});

// ---------------------------------------------------------------------
// V5. COLLISIONS — both layers must throw clear errors naming the param.
// ---------------------------------------------------------------------
test('V5a inner-dist positional+keyword collision (Beta(a_g, alpha=…)) names alpha', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
bb = [5.0, 4.0]
K = (a_g, b_g) -> iid(Beta(a_g, alpha = 2.0), N)
p ~ K.(a, b_g = bb)
`;
  await assert.rejects(
    (async () => { const { ctx } = makeMatCtx(src, { sampleCount: 64 }); await ctx.getMeasure('p'); })(),
    /alpha/,
    'inner-dist collision must name alpha');
});

test('V5b outer broadcast positional+keyword collision (K.(a, a_g=b)) names a_g', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ K.(a, a_g = b)
`;
  await assert.rejects(
    (async () => { const { ctx } = makeMatCtx(src, { sampleCount: 64 }); await ctx.getMeasure('p'); })(),
    /a_g/,
    'outer collision must name a_g');
});

test('V5c outer collision on the SECOND param (K.(a, b, b_g=c)) names b_g', async () => {
  // Adversarial: collision not on the first param. positional b -> b_g,
  // and keyword b_g both target b_g.
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
c = [1.0, 1.0]
K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ K.(a, b, b_g = c)
`;
  await assert.rejects(
    (async () => { const { ctx } = makeMatCtx(src, { sampleCount: 64 }); await ctx.getMeasure('p'); })(),
    /b_g/,
    'second-param outer collision must name b_g');
});

// ---------------------------------------------------------------------
// V6. NEGATIVE classification guards — the recogniser must stay
//     discriminating: deterministic arrow broadcasts and value-broadcast
//     functionof lambdas are NOT kernel broadcasts.
// ---------------------------------------------------------------------
test('V6a deterministic arrow lambda f.(a,b) classifies as evaluate', () => {
  const src = `
a = [2.0, 3.0]
b = [5.0, 4.0]
f = (m, s) -> m + s
y = f.(a, b)
`;
  assert.equal(classifyNamedKind(src, 'y'), 'evaluate',
    'deterministic arrow broadcast is value-evaluate, not kernel');
});

test('V6b functionof over a VALUE broadcast is NOT a kernel broadcast', () => {
  // `m -> functionof(m .+ g)` — a function over a VALUE expression, not a
  // measure. Must NOT be misclassified as a kernel head.
  const src = `
a = [1.0, 2.0]
g = [3.0, 4.0]
K = m -> functionof(m .+ g)
y = broadcast(K, m = a)
`;
  const r = classifyKernelBroadcastKind(src, 'K');
  assert.notEqual(r.kind, 'kernelbroadcast',
    'functionof over a value is NOT a kernel broadcast');
});

test('V6c deterministic arrow lambda with a measure-NAME-looking body stays evaluate', () => {
  // Body is arithmetic only; no measure construction. Even with a param
  // named like a dist, it must stay evaluate.
  const src = `
Normalish = [1.0, 2.0]
b = [3.0, 4.0]
f = (Normalish_p, s) -> Normalish_p * s
y = f.(Normalish, b)
`;
  assert.equal(classifyNamedKind(src, 'y'), 'evaluate',
    'arithmetic-bodied arrow stays evaluate');
});

// ---------------------------------------------------------------------
// V7. DOUBLE-lawof body. peelKernelBody is deliberately UNARY/one-level.
//     `m -> lawof(lawof(Normal(m,1)))` either errors clearly or is NOT
//     silently treated as a clean kernel that miscomputes. Pin the
//     observed behaviour (must not silently produce a wrong measure).
// ---------------------------------------------------------------------
test('V7 double-lawof arrow body does not silently miscompute (errors or non-kernelbroadcast)', async () => {
  const src = `
a = [1.0, 2.0]
K = m -> lawof(lawof(Normal(m, 1.0)))
p ~ K.(a)
`;
  const r = classifyKernelBroadcastKind(src, 'K');
  // peelKernelBody peels only ONE lawof; the residual inner lawof is not a
  // measure construction the detectors accept. Acceptable outcomes:
  //   (i)  not classified as kernelbroadcast (inner lawof not recognised), OR
  //   (ii) classified but materialise throws a CLEAR error.
  // Unacceptable: silently materialises a plausible-but-wrong measure.
  if (r.found && r.kind === 'kernelbroadcast') {
    await assert.rejects(
      (async () => { const { ctx } = makeMatCtx(src, { sampleCount: 64 }); await ctx.getMeasure('p'); })(),
      /.+/,
      'a double-lawof body, if classified as a kernel, must error rather than silently miscompute');
  } else {
    // Recogniser declined — that is a correct (conservative) outcome.
    assert.ok(true, `double-lawof not treated as kernel broadcast (kind=${r.kind})`);
  }
});

// ---------------------------------------------------------------------
// V8. ARITY edges — independent of the author's M4 cases.
// ---------------------------------------------------------------------
test('V8a too few outer args (3-param joint kernel called with 2) errors loudly', async () => {
  const src = `
arrA = [1.0, 2.0]
arrB = [0.5, 1.0]
K = (a, b, c) -> joint(x = Normal(mu = a, sigma = 1.0), y = Exponential(rate = b), w = Normal(mu = c, sigma = 0.5))
z ~ K.(arrA, arrB)
`;
  await assert.rejects(
    (async () => { const { ctx } = makeMatCtx(src, { sampleCount: 64 }); await ctx.getMeasure('z'); })(),
    /c|expected 3|arity|positional|unbound/i,
    'under-applying a 3-param kernel must error (no silent component drop)');
});

test('V8b too many outer args (2-param kernel called with 3) errors loudly with arity message', async () => {
  const src = `
N = 4
a = [2.0, 3.0]
b = [5.0, 4.0]
c = [1.0, 1.0]
K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ K.(a, b, c)
`;
  await assert.rejects(
    (async () => { const { ctx } = makeMatCtx(src, { sampleCount: 64 }); await ctx.getMeasure('p'); })(),
    /expected 2|too many|arity|positional/i,
    'over-applying must throw an arity error (extra arg not silently dropped)');
});

// ---------------------------------------------------------------------
// V9. VALUE-LEVEL: joint kernel per-cell components must track DISTINCT
//     params (a swap or per-cell collapse would equalise them). Pins
//     that the positional-component joint binds the right surface arg to
//     the right component.
// ---------------------------------------------------------------------
test('V9 joint positional-component kernel: per-cell component means distinct + correct', async () => {
  const src = `
arrA = [-3.0, 3.0]
arrB = [10.0, 40.0]
K = (a, b) -> joint(Normal(mu = a, sigma = 0.5), Normal(mu = b, sigma = 0.5))
z ~ K.(arrA, arrB)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 6000 });
  const m = await expectPlottable(ctx, 'z');
  const [Ns, Kc, C] = m.value.shape;
  assert.equal(Kc, 2); assert.equal(C, 2);
  const arrA = [-3.0, 3.0]; const arrB = [10.0, 40.0]; const dd = m.value.data;
  for (let j = 0; j < 2; j++) {
    let sa = 0; let sb = 0;
    for (let i = 0; i < Ns; i++) {
      sa += dd[i * Kc * C + j * C + 0];
      sb += dd[i * Kc * C + j * C + 1];
    }
    assert.ok(Math.abs(sa / Ns - arrA[j]) < 0.06,
      `cell ${j} comp0 ≈ ${arrA[j]}; got ${(sa / Ns).toFixed(3)}`);
    assert.ok(Math.abs(sb / Ns - arrB[j]) < 0.06,
      `cell ${j} comp1 ≈ ${arrB[j]}; got ${(sb / Ns).toFixed(3)}`);
    assert.ok(Math.abs(sa / Ns - sb / Ns) > 5,
      'the two components are DISTINCT (no swap/collapse)');
  }
});
