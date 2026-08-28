'use strict';

// The SCALAR-affine pushforward spelling over a vector base:
//
//   X = pushfwd(x -> 2.0 * x + b, iid(Normal(0, 1), D))
//
// Spec §06 "Engine contract for `pushfwd` density evaluation" case 1 requires
// every conforming engine to recognize "affine maps composed from
// `add`/`sub`/`neg`/`mul`/`divide`" by name, so this spelling owes an analytic
// density. Before the fix the lift left it as a plain pushfwd: a rescale threw
// "scalar leaf has no entry to consume (value exhausted)" (density consumed ONE
// scalar from the length-D variate) and a vector shift threw "per-atom or
// vector-valued bijections not yet supported here". The lift now synthesises
// `L = s * eye(D)` and `b = zeros(D)` and routes to the same affine-registry
// entry the matrix-vector spelling uses (pushfwd-affine-vector.test.ts).
//
// Oracle (INDEPENDENT — Distributions.jl, not the other engine). The
// pushforward of `iid(Normal(mu0, 1), 2)` through `x -> s * x + b` is the pair
// of independent `Normal(s * mu0 + b[i], |s|)`, so every leg below is a
// closed-form product of scalar normal logpdfs, evaluated at y = [1.5, -0.5]
// with b = [1.0, -1.0].

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const ROOT_SEED = 0x10C5CA1E;

const B_SRC = `b = [1.0, -1.0]\n`;
const POINT = '[1.5, -0.5]';

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

// The synthetic bijection binding the affine lift emits, or null when the call
// was not routed to the registry.
function affineBijection(ctx: any) {
  for (const b of ctx.bindings.values()) {
    if (b && b.bijection && b.bijection.registryName === 'affine') return b.bijection;
  }
  return null;
}

// ── Every routed spelling, scored against the Distributions.jl oracle.
//
// Base `iid(Normal(0, 1), 2)`. The shift legs pin the SHIFT (their means differ
// from the shift-free legs); the scale legs pin the log-volume element (|s| = 2
// moves the score by -2*log(2) relative to |s| = 1).

const ROUTED: any[] = [
  // Rescale only. `-2.0 * x` parses as `(-2.0) * x`, so its minus sits on the
  // literal factor; `-(2.0 * x)` puts it on the whole product instead.
  ['s * x',        `2.0 * x`,       -3.536671427529236],
  ['x * s',        `x * 2.0`,       -3.536671427529236],
  ['(-s) * x',     `-2.0 * x`,      -3.536671427529236],
  ['-(s * x)',     `-(2.0 * x)`,    -3.536671427529236],
  // Unscaled: the identity and the pure negation.
  ['x',            `x`,             -3.0878770664093453],
  ['-x',           `-x`,            -3.0878770664093453],
  // Shift only — the four ways the additive layer can sit.
  ['x + b',        `x + b`,         -2.0878770664093453],
  ['b + x',        `b + x`,         -2.0878770664093453],
  ['x - b',        `x - b`,         -6.087877066409346],
  ['b - x',        `b - x`,         -2.0878770664093453],
  // Rescale and shift together.
  ['s * x + b',    `2.0 * x + b`,   -3.286671427529236],
  ['b + s * x',    `b + 2.0 * x`,   -3.286671427529236],
  ['s * x - b',    `2.0 * x - b`,   -4.286671427529236],
  ['b - s * x',    `b - 2.0 * x`,   -3.286671427529236],
];

for (const [label, body, expected] of ROUTED) {
  test(`scalar-affine density matches the oracle: ${label}`, async () => {
    const ctx = makeCtx(B_SRC
      + `X = pushfwd(x -> ${body}, iid(Normal(0.0, 1.0), 2))\n`
      + `lp = logdensityof(X, ${POINT})\n`);
    assert.ok(affineBijection(ctx), `${label} routes to the affine registry`);
    const lp = await ctx.getMeasure('lp');
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-12,
      `${label}: got ${lp.samples[0]}, expected ${expected}`);
  });
}

test('the routed bijection carries registry paramIRs {L, b}', () => {
  const ctx = makeCtx(B_SRC
    + `X = pushfwd(x -> 2.0 * x + b, iid(Normal(0.0, 1.0), 2))\n`);
  assert.equal(ctx.derivations['X'].kind, 'pushfwd');
  const bij = affineBijection(ctx);
  assert.ok(bij, 'a bijection binding marked registryName=affine exists');
  assert.ok(bij.paramIRs && bij.paramIRs.L && bij.paramIRs.b,
    'paramIRs {L, b} present');
});

test('a NAMED scalar scale ref routes and scores the same', async () => {
  // The factor is an Identifier rather than a literal, so it is confirmed
  // scalar through its inferredType instead of structurally.
  const ctx = makeCtx(`s = 2.0\n`
    + `X = pushfwd(x -> s * x, iid(Normal(0.0, 1.0), 2))\n`
    + `lp = logdensityof(X, ${POINT})\n`);
  assert.ok(affineBijection(ctx), 'a named scalar scale routes');
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-3.536671427529236)) < 1e-12,
    `got ${lp.samples[0]}`);
});

test('a NAMED affine function ref routes and scores the same', async () => {
  const ctx = makeCtx(B_SRC
    + `f = x -> 2.0 * x + b\n`
    + `X = pushfwd(f, iid(Normal(0.0, 1.0), 2))\n`
    + `lp = logdensityof(X, ${POINT})\n`);
  assert.ok(affineBijection(ctx), 'a named fn ref routes');
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-3.286671427529236)) < 1e-12,
    `got ${lp.samples[0]}`);
});

test('a length-3 base routes at its own D', async () => {
  // D comes from the base's iid count, not from a matrix, so a non-2 base must
  // synthesise eye(3) / zeros(3). Oracle: three Normal(0, 2) legs.
  const ctx = makeCtx(
    `X = pushfwd(x -> 2.0 * x, iid(Normal(0.0, 1.0), 3))\n`
    + `lp = logdensityof(X, [1.5, -0.5, 0.25])\n`);
  assert.ok(affineBijection(ctx), 'a length-3 base routes');
  const lp = await ctx.getMeasure('lp');
  // logpdf(Normal(0,2), 1.5) + logpdf(Normal(0,2), -0.5) + logpdf(Normal(0,2), 0.25)
  const expected = -5.156569641293855;
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-12,
    `got ${lp.samples[0]}, expected ${expected}`);
});

// ── Sign observability. A symmetric zero-mean base scores a negated map
// identically to the un-negated one, so a dropped or doubled sign is invisible
// above. Over `iid(Normal(1, 1), 2)` the pushforward mean moves with the sign.

const SIGNED: any[] = [
  ['x',              `x`,             -3.0878770664093453],
  ['-x',             `-x`,            -5.087877066409346],
  ['s * x + b',      `2.0 * x + b`,   -3.786671427529236],
  ['(-s) * x + b',   `-2.0 * x + b`,  -4.786671427529236],
  ['b - s * x',      `b - 2.0 * x`,   -4.786671427529236],
  ['x - b',          `x - b`,         -6.087877066409346],
];

for (const [label, body, expected] of SIGNED) {
  test(`the sign survives over an off-centre base: ${label}`, async () => {
    const ctx = makeCtx(B_SRC
      + `X = pushfwd(x -> ${body}, iid(Normal(1.0, 1.0), 2))\n`
      + `lp = logdensityof(X, ${POINT})\n`);
    const lp = await ctx.getMeasure('lp');
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-12,
      `${label}: got ${lp.samples[0]}, expected ${expected}`);
  });
}

// ── Sampling keeps the atom axis. The affine registry's forward consumes the
// base's [N, D] batch whole, so the routed sample path must not flatten it.

test('scalar-affine sampling keeps the [N, D] atom batch', async () => {
  const N = 20000;
  const ctx = makeCtx(B_SRC
    + `X = pushfwd(x -> 2.0 * x + b, iid(Normal(0.0, 1.0), 2))\n`,
    { sampleCount: N });
  const X = await ctx.getMeasure('X');
  assert.deepEqual(X.value.shape, [N, 2]);
  assert.equal(X.value.outerRank, 1);
  const d = X.value.data;
  let m0 = 0, m1 = 0;
  for (let i = 0; i < N; i++) { m0 += d[2 * i]; m1 += d[2 * i + 1]; }
  m0 /= N; m1 /= N;
  let v0 = 0, v1 = 0;
  for (let i = 0; i < N; i++) {
    v0 += (d[2 * i] - m0) ** 2; v1 += (d[2 * i + 1] - m1) ** 2;
  }
  // Closed-form moments: E[X] = b, sd = |s| = 2. At N = 20000 a mean's
  // standard error is 2/sqrt(N) ≈ 0.014, so 0.1 is ≈7σ.
  assert.ok(Math.abs(m0 - 1.0) < 0.1, `mean[0] = ${m0}`);
  assert.ok(Math.abs(m1 - (-1.0)) < 0.1, `mean[1] = ${m1}`);
  assert.ok(Math.abs(Math.sqrt(v0 / N) - 2.0) < 0.1, `sd[0] = ${Math.sqrt(v0 / N)}`);
  assert.ok(Math.abs(Math.sqrt(v1 / N) - 2.0) < 0.1, `sd[1] = ${Math.sqrt(v1 / N)}`);
});

test('a negated shift samples with the negated mean', async () => {
  // `x - b` lowers the shift as `zeros(D) - b` rather than `-b`, because
  // resolveIRToValue's `neg` cannot fold a vector-valued binding ref. This
  // pins that the resolved shift really is -b at materialise time.
  const N = 20000;
  const ctx = makeCtx(B_SRC
    + `X = pushfwd(x -> x - b, iid(Normal(0.0, 1.0), 2))\n`,
    { sampleCount: N });
  const X = await ctx.getMeasure('X');
  assert.deepEqual(X.value.shape, [N, 2]);
  const d = X.value.data;
  let m0 = 0, m1 = 0;
  for (let i = 0; i < N; i++) { m0 += d[2 * i]; m1 += d[2 * i + 1]; }
  m0 /= N; m1 /= N;
  assert.ok(Math.abs(m0 - (-1.0)) < 0.1, `mean[0] = ${m0}`);
  assert.ok(Math.abs(m1 - 1.0) < 0.1, `mean[1] = ${m1}`);
});

// ── Gate negatives. Each body is NOT a scalar-affine map over the base, so
// none may reach the affine registry — routing one would score a different
// measure than the user wrote. Spec §06 case 1 covers affine maps only; the
// remaining bodies fall to case 3 (a static error unless the user annotates the
// function with `bijection`), which is the spec-CORRECT outcome, not a bug.

const NOT_SCALAR_AFFINE: any[] = [
  // Elementwise `exp` is a DIAGONAL bijection. §06 case 1 names `exp`/`log`
  // among the scalar built-ins and matrix-vector AFFINE maps, but not a
  // nonlinear diagonal map over a vector variate, so refusing is correct.
  ['elementwise exp', `pushfwd(x -> exp(x), iid(Normal(0.0, 1.0), 2))`],
  // A per-component scale is a diagonal matrix the user spelled as a vector;
  // `v * x` is not the scalar-vector product §07 `mul` gives a scalar factor.
  ['per-component vector scale',
    `pushfwd(x -> v * x, iid(Normal(0.0, 1.0), 2))`],
  // §07 `add` takes "scalars or arrays of same shape", so a scalar shift over
  // a vector base is not a legal map at all.
  ['scalar shift over a vector base',
    `pushfwd(x -> x + 1.0, iid(Normal(0.0, 1.0), 2))`],
  // A shift of the wrong length cannot be added to a length-2 variate.
  ['shift length mismatched with D',
    `pushfwd(x -> x + b3, iid(Normal(0.0, 1.0), 2))`],
  // A zero factor is a singular map, so it is not a bijection and has no
  // change-of-variables density.
  ['zero scale', `pushfwd(x -> 0.0 * x, iid(Normal(0.0, 1.0), 2))`],
  // A second occurrence of the parameter makes the map non-affine in it.
  ['parameter on both sides of the +',
    `pushfwd(x -> 2.0 * x + x, iid(Normal(0.0, 1.0), 2))`],
  // Neither operand of the product is a bare parameter reference.
  ['product of two parameter occurrences',
    `pushfwd(x -> x * x, iid(Normal(0.0, 1.0), 2))`],
  // §06 case 1 admits `divide`, but typeinfer's `divide` is scalars-only
  // (absent from its BINARY_ARITH_OPS), so `x / s` over a vector types as a
  // scalar and the query is refused before the lift is consulted. Routing it
  // would pair a correct number with a static error.
  ['scalar divide', `pushfwd(x -> x / 2.0, iid(Normal(0.0, 1.0), 2))`],
  // The parameter in the DIVISOR is not affine in it.
  ['parameter as divisor', `pushfwd(x -> 2.0 / x, iid(Normal(0.0, 1.0), 2))`],
  // D is synthesised into `eye(D)`, so a base whose count is not statically an
  // integer cannot route.
  ['base count not static',
    `pushfwd(x -> 2.0 * x, iid(Normal(0.0, 1.0), n))`],
  // The registry density path needs a base that scores as iid(<scalar>, D).
  ['non-iid base', `pushfwd(x -> 2.0 * x, Normal(0.0, 1.0))`],
  // Dotted operators lower to `broadcast(...)`, a different map from the
  // scalar-array arithmetic §07 gives `add` and `mul`. One case per position:
  // the body root, the inner product, and a peeled unary minus.
  ['dotted product at the body root',
    `pushfwd(x -> 2.0 .* x, iid(Normal(0.0, 1.0), 2))`],
  ['dotted product under a shift',
    `pushfwd(x -> 2.0 .* x + b, iid(Normal(0.0, 1.0), 2))`],
  ['dotted unary minus under a shift',
    `pushfwd(x -> .-x + b, iid(Normal(0.0, 1.0), 2))`],
];

for (const [label, call] of NOT_SCALAR_AFFINE) {
  test(`not routed to the affine registry: ${label}`, () => {
    const src = B_SRC
      + `b3 = [1.0, -1.0, 0.5]\n`
      + `v = [2.0, 3.0]\n`
      + `n = 2\n`
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

// The matrix-vector gate matches `2.0 * x + b` syntactically and only its shape
// checks can tell a scalar factor from a matrix one, so a body it matches but
// cannot confirm must fall through to the scalar route rather than abandoning
// the call. This pins that fall-through directly: the matrix gate's own
// negatives (pushfwd-affine-vector.test.ts) must still refuse afterwards.
test('a matrix-gate match with a scalar scale falls through to the scalar route',
  async () => {
    const ctx = makeCtx(B_SRC
      + `X = pushfwd(x -> 2.0 * x + b, iid(Normal(0.0, 1.0), 2))\n`
      + `lp = logdensityof(X, ${POINT})\n`);
    const bij = affineBijection(ctx);
    assert.ok(bij, 'the scalar route claimed the body the matrix gate matched');
    const lp = await ctx.getMeasure('lp');
    assert.ok(Math.abs(lp.samples[0] - (-3.286671427529236)) < 1e-12,
      `got ${lp.samples[0]}`);
  });
