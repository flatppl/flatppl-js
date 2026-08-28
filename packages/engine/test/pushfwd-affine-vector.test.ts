'use strict';

// The EXPLICIT matrix-vector affine pushforward spelling over a vector base:
//
//   X = pushfwd(x -> L * x + b, iid(Normal(0, 1), D))
//
// Spec §06 "Engine contract for `pushfwd` density evaluation" case 1 lists
// "matrix-vector affine maps such as `mu + lower_cholesky(cov) * _`" among the
// built-in bijections every conforming engine must recognize by name, so this
// spelling owes an analytic density — and, because the affine registry's
// forward consumes the base's `[N, D]` atom batch whole, a sampling path that
// keeps the atom axis. Before the fix the lift left the call as a plain
// pushfwd: density consumed ONE scalar from a length-D variate ("value
// exhausted") and sampling handed `mul` the flat `[N*D]` buffer
// ("matrix×vector dimension mismatch ([2,2] × [128])").
//
// Oracle (INDEPENDENT — Distributions.jl, not the other engine):
//   L = [2.0 0.0; 0.5 1.5];  b = [1.0, -1.0];  Σ = L*L' = [4.0 1.0; 1.0 2.5]
//   d = MvNormal(b, Σ)
//   logpdf(d, [1.5, -0.5]) = -2.9989893550774553
//   logpdf(d, [1.0, -1.0]) = -2.9364893550774553
//   logpdf(d, [0.0,  0.0]) = -3.4087115772996777
// L is deliberately NON-diagonal: a diagonal scale cannot tell a genuine
// matrix-vector product apart from an elementwise rescale.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const ROOT_SEED = 0x10C5CA1E;

const L_SRC = `L = rowstack([[2.0, 0.0], [0.5, 1.5]])\n`;
const B_SRC = `b = [1.0, -1.0]\n`;
const PRE = L_SRC + B_SRC;
const SIGMA = [[4.0, 1.0], [1.0, 2.5]];
const ORACLE: any[] = [
  ['[1.5, -0.5]', -2.9989893550774553],
  ['[1.0, -1.0]', -2.9364893550774553],
  ['[0.0, 0.0]', -3.4087115772996777],
];

function makeCtx(source: any, opts?: any) {
  opts = opts || {};
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((e: any) => e.message), [],
    'unexpected diagnostics');
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx); cache.set(name, p); return p;
    },
    sendWorker: (msg: any) => {
      const r = worker.handle(msg);
      if (r && r.type === 'error') return Promise.reject(new Error(r.message));
      return Promise.resolve(r);
    },
    sampleCount: opts.sampleCount != null ? opts.sampleCount : 256,
    rootSeed: ROOT_SEED,
  };
  return ctx;
}

// The synthetic bijection binding the affine lift emits, or null when the
// call was not routed to the registry.
function affineBijection(ctx: any) {
  for (const b of ctx.bindings.values()) {
    if (b && b.bijection && b.bijection.registryName === 'affine') return b.bijection;
  }
  return null;
}

// Per-component mean and covariance of an atom-major [N, 2] sample batch.
function moments2(value: any) {
  const N = value.shape[0], d = value.data;
  let m0 = 0, m1 = 0;
  for (let i = 0; i < N; i++) { m0 += d[2 * i]; m1 += d[2 * i + 1]; }
  m0 /= N; m1 /= N;
  let c00 = 0, c01 = 0, c11 = 0;
  for (let i = 0; i < N; i++) {
    const a = d[2 * i] - m0, c = d[2 * i + 1] - m1;
    c00 += a * a; c01 += a * c; c11 += c * c;
  }
  return { mean: [m0, m1], cov: [[c00 / N, c01 / N], [c01 / N, c11 / N]] };
}

test('the affine lambda spelling routes to the registry-affine pushfwd', () => {
  const ctx = makeCtx(PRE
    + `X = pushfwd(x -> L * x + b, iid(Normal(0.0, 1.0), 2))\n`);
  assert.equal(ctx.derivations['X'].kind, 'pushfwd');
  const bij = affineBijection(ctx);
  assert.ok(bij, 'a bijection binding marked registryName=affine exists');
  assert.ok(bij.paramIRs && bij.paramIRs.L && bij.paramIRs.b,
    'paramIRs {L, b} present');
});

test('affine lambda density matches the Distributions.jl MvNormal oracle', async () => {
  for (const [pt, expected] of ORACLE) {
    const ctx = makeCtx(PRE
      + `X = pushfwd(x -> L * x + b, iid(Normal(0.0, 1.0), 2))\n`
      + `lp = logdensityof(X, ${pt})\n`);
    const lp = await ctx.getMeasure('lp');
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-12,
      `pt=${pt}: got ${lp.samples[0]}, expected ${expected}`);
  }
});

test('affine lambda sampling keeps the [N, D] atom batch and matches L·Lᵀ', async () => {
  const ctx = makeCtx(PRE
    + `X = pushfwd(x -> L * x + b, iid(Normal(0.0, 1.0), 2))\n`,
    { sampleCount: 20000 });
  const X = await ctx.getMeasure('X');
  // The reported symptom was a throw here; the atom axis must survive.
  assert.deepEqual(X.value.shape, [20000, 2]);
  assert.equal(X.value.outerRank, 1);
  const { mean, cov } = moments2(X.value);
  // Closed-form moments of the pushforward: E[X] = b, Cov[X] = L·Lᵀ. At
  // N=20000 the standard error of a mean is ≈0.014 and of a covariance entry
  // ≈0.04, so these tolerances are ≈7σ and ≈6σ.
  assert.ok(Math.abs(mean[0] - 1.0) < 0.1, `mean[0] = ${mean[0]}`);
  assert.ok(Math.abs(mean[1] - (-1.0)) < 0.1, `mean[1] = ${mean[1]}`);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      assert.ok(Math.abs(cov[i][j] - SIGMA[i][j]) < 0.25,
        `cov[${i}][${j}] = ${cov[i][j]}, expected ${SIGMA[i][j]}`);
    }
  }
});

test('the shift may sit on either side of the +', async () => {
  const ctx = makeCtx(PRE
    + `X = pushfwd(x -> b + L * x, iid(Normal(0.0, 1.0), 2))\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`);
  assert.ok(affineBijection(ctx), 'b + L * x routes to the affine registry');
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-2.9989893550774553)) < 1e-12,
    `got ${lp.samples[0]}`);
});

test('a NAMED affine function ref routes and scores the same', async () => {
  const ctx = makeCtx(PRE
    + `f = x -> L * x + b\n`
    + `X = pushfwd(f, iid(Normal(0.0, 1.0), 2))\n`
    + `lp = logdensityof(X, [1.0, -1.0])\n`);
  assert.ok(affineBijection(ctx), 'a named fn ref routes to the affine registry');
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-2.9364893550774553)) < 1e-12,
    `got ${lp.samples[0]}`);
});

test('a named lower_cholesky scale routes (the spec MvNormal idiom)', async () => {
  // Lc = lower_cholesky(Σ) is exactly L, so the same oracle applies. The op's
  // own result type is %dynamic, so D is discovered from its argument.
  const ctx = makeCtx(
    `cov = rowstack([[4.0, 1.0], [1.0, 2.5]])\n`
    + `Lc = lower_cholesky(cov)\n`
    + B_SRC
    + `X = pushfwd(x -> Lc * x + b, iid(Normal(0.0, 1.0), 2))\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`);
  assert.ok(affineBijection(ctx), 'lower_cholesky scale routes');
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-2.9989893550774553)) < 1e-12,
    `got ${lp.samples[0]}`);
});

test('an INLINE matrix-literal scale routes and scores the same', async () => {
  // Written inline rather than as a named ref, so the scale resolves through
  // the literal-structure square-confirm rather than a ref inferredType.
  const ctx = makeCtx(B_SRC
    + `X = pushfwd(x -> rowstack([[2.0, 0.0], [0.5, 1.5]]) * x + b, `
    + `iid(Normal(0.0, 1.0), 2))\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`);
  assert.ok(affineBijection(ctx), 'an inline literal scale routes');
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-2.9989893550774553)) < 1e-12,
    `got ${lp.samples[0]}`);
});

// ── Gate negatives. Each is NOT a matrix-vector affine map in the parameter,
// so none may be routed to the affine registry — routing one would score a
// different measure than the user wrote.

const NOT_AFFINE: any[] = [
  // The parameter on the LEFT of the product is a vector×matrix map, xᵀL,
  // not the registry's L·x.
  ['parameter left of the product', `pushfwd(x -> x * L + b, iid(Normal(0.0, 1.0), 2))`],
  // A second occurrence of the parameter makes the map non-affine in it.
  ['parameter also in the shift', `pushfwd(x -> L * x + x, iid(Normal(0.0, 1.0), 2))`],
  // Non-square scale: L·x is not an endomorphism, so it is not a bijection.
  ['non-square scale', `pushfwd(x -> Lns * x + b, iid(Normal(0.0, 1.0), 2))`],
  // A base of the wrong length cannot feed a [2,2] scale.
  ['base length mismatched with D', `pushfwd(x -> L * x + b, iid(Normal(0.0, 1.0), 3))`],
  // The map is affine in exp(x), not in x.
  ['scale applied to a transform of the parameter',
    `pushfwd(x -> L * exp(x) + b, iid(Normal(0.0, 1.0), 2))`],
  // An inline NON-SQUARE literal scale fails the literal square-confirm.
  ['inline non-square literal scale',
    `pushfwd(x -> rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * x + b, `
    + `iid(Normal(0.0, 1.0), 2))`],
  // An inline square-op call carries no static D: the op's own result type is
  // %dynamic, and only a NAMED ref lets the lift reach the op's argument.
  ['inline square-op scale',
    `pushfwd(x -> lower_cholesky(rowstack([[4.0, 1.0], [1.0, 2.5]])) * x + b, `
    + `iid(Normal(0.0, 1.0), 2))`],
  // The registry density path needs a base that scores as iid(<scalar>, D).
  ['non-iid base', `pushfwd(x -> L * x + b, Normal(0.0, 1.0))`],
];

for (const [label, call] of NOT_AFFINE) {
  test(`not routed to the affine registry: ${label}`, () => {
    const src = PRE
      + `Lns = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n`
      + `X = ${call}\n`;
    const lifted = processSource(src);
    const built = orchestrator.buildDerivations(lifted.bindings);
    let routed = false;
    for (const b of built.bindings.values()) {
      if (b && b.bijection && b.bijection.registryName === 'affine') routed = true;
    }
    assert.equal(routed, false, `${label} must not route to the affine registry`);
  });
}

// ── The plain-pushfwd AST path over a vector base.

test('a shift-free matrix map samples through the atom batch (no registry)', async () => {
  // `L * x` is not the gated `L * x + b` form, so this rides the AST path —
  // which must bind the parameter to the base's [N, D] Value. Previously the
  // flat [N*D] buffer reached `mul` and threw the reported dimension error.
  // Closed-form moments: E[X] = 0, Cov[X] = L·Lᵀ.
  const ctx = makeCtx(L_SRC
    + `X = pushfwd(x -> L * x, iid(Normal(0.0, 1.0), 2))\n`,
    { sampleCount: 20000 });
  assert.equal(affineBijection(ctx), null, 'no registry route for this form');
  const X = await ctx.getMeasure('X');
  assert.deepEqual(X.value.shape, [20000, 2]);
  const { mean, cov } = moments2(X.value);
  assert.ok(Math.abs(mean[0]) < 0.1, `mean[0] = ${mean[0]}`);
  assert.ok(Math.abs(mean[1]) < 0.1, `mean[1] = ${mean[1]}`);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      assert.ok(Math.abs(cov[i][j] - SIGMA[i][j]) < 0.25,
        `cov[${i}][${j}] = ${cov[i][j]}, expected ${SIGMA[i][j]}`);
    }
  }
});

test('an elementwise map over a vector variate keeps the atom axis', async () => {
  // This case used to refuse: elementwise scalar primitives over a whole
  // D-vector atom were not wired into the batched evaluator, and reading the
  // flat atom buffer as N scalars produced a shape-[N] measure of NaNs past
  // the first N/D entries. `broadcastN` now maps the primitive over each
  // atom's cell — see broadcastn-vector-atom.test.ts for the oracle legs.
  const N = 8;
  const ctx = makeCtx(`Z = iid(Normal(0.0, 1.0), 2)\n`
    + `X = pushfwd(x -> exp.(x), Z)\n`, { sampleCount: N });
  const Z = await ctx.getMeasure('Z');
  const X = await ctx.getMeasure('X');
  assert.deepEqual(X.value.shape, [N, 2]);
  for (let i = 0; i < N * 2; i++) {
    assert.equal(X.value.data[i], Math.exp(Z.value.data[i]),
      `cell ${i} is not exp of the base cell`);
  }
});

test('a product of two vector atoms still reports the unsupported shape', async () => {
  // The residual the batched evaluator genuinely does not cover: `x * x`
  // over two [N,D] operands is neither elementwise-scalar nor one of
  // value-ops' shape-aware kernels.
  const ctx = makeCtx(`X = pushfwd(x -> x * x, iid(Normal(0.0, 1.0), 2))\n`,
    { sampleCount: 8 });
  await assert.rejects(() => ctx.getMeasure('X'), (err: any) => {
    assert.match(err.message, /vector-valued variate/,
      `expected a vector-variate diagnostic, got: ${err.message}`);
    assert.match(err.message, /product of two vector atoms/, `got: ${err.message}`);
    return true;
  });
});
