'use strict';

// The INLINE spelling of the W1 five-row table (kernel-joint-w1-refusal.test.ts),
// with the second joint component written verbatim rather than bound to a name
// first: `joint(p = K1, q = kernelof(u, z = z))` instead of
// `K2 = kernelof(u, z = z); joint(p = K1, q = K2)`.
//
// Before this file, the inline spelling silently dropped the `q` component:
// zero diagnostics at typeinfer, and `logdensityof` refused with a bare
// `no derivation for 'ld'` that gave no reason. Root cause was in lift.ts's
// AST-hoisting pass, not in typeinfer: `joint`'s kwarg-field lift routed an
// inline `kernelof`/`functionof` component through the same `liftMeasure`
// hoist as an ordinary inline measure, replacing it with a ref to a
// synthetic anon binding. Synthetic bindings never get an `inferredType`
// (typeinfer runs over the pre-lift module, before any synthetic binding
// exists), so `_jointComponentAsMeasure`'s kernel/measure test — which reads
// `inferredType.kind` off the resolved binding — read the untyped synthetic
// binding as a bare measure. The component's kernel-ness, and its boundary
// input, both vanished before the fan-out hoist ever ran.
//
// The fix keeps a `joint` component that is written as an inline
// `kernelof(...)` / `functionof(...)` CallExpr in place (`liftJointComponent`
// in lift.ts) instead of hoisting it, so `_jointComponentAsMeasure` reads the
// CallExpr head directly — exactly the shape a named binding gives it, minus
// the lookup. The same widening extends to typeinfer's W1 detector
// (`reifiedBoundaryInfo`), which previously resolved a component's boundary
// only through a `ref` to a `functionof`-shaped binding; kernelof lowers to
// functionof in the IR before typeinfer runs, so an inline kernel component's
// IR already has that same `{kind:'call', op:'functionof', ...}` shape and
// needed only a second branch, not a lookup, to be seen. Both directions
// were needed: without the typeinfer half, ROW1's illegal inline shape would
// pass typeinfer clean and refuse only for the unrelated fallback reason
// ("no derivation"), not the located §06 ancestry error.
//
// Oracles are unchanged from kernel-joint-w1-refusal.test.ts — same shapes,
// same closed-form + quadrature derivations, just spelled inline.

const test = require('node:test');
const assert = require('node:assert');
const { processSource, materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const AT_0 = -2.0878770664093453;
const CLOSED_U_AT_5 = -27.087877066409344;
const F64_TOL = 1e-13;

const infer = (src: string) => (processSource(src).diagnostics || [])
  .filter((d: any) => d.severity === 'error');

const scoreOf = async (src: string) => {
  const { ctx } = ctxFor(src, 1);
  return (await ctx.getMeasure('ld')).samples[0];
};

const SHARED_ANCESTRY_ERROR
  = /share the stochastic node 'u'.*ancestor 'z' is bound as input 'z'.*not bound at all by a measure component, which binds nothing/s;

// ── row 1: the illegal shape refuses, spelled inline ────────────────────────

const ROW1_ILLEGAL_INLINE = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
KJ = joint(p = K1, q = lawof(u))
`;

test('row 1 (inline) — an inline measure component sharing a boundary-descended '
  + 'node is a STATIC error, same as the named spelling', () => {
  const errors = infer(ROW1_ILLEGAL_INLINE);
  assert.ok(errors.some((e: any) => SHARED_ANCESTRY_ERROR.test(e.message)),
    'got: ' + errors.map((e: any) => e.message).join(' | '));
});

test('row 1 (inline) — the density query refuses instead of silently dropping '
  + 'the component (zero diagnostics, no number)', async () => {
  const src = ROW1_ILLEGAL_INLINE
    + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = 0.5))\n';
  const { ctx } = ctxFor(src, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'), /no derivation for 'ld'/);
});

// ── rows 2-3: reading E, `q` written inline as `kernelof(u, z = z)` ─────────

const ROWS23_INLINE = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
KJ = joint(p = K1, q = kernelof(u, z = z))
`;

test('row 2 (inline) — the inline kernelof component types clean and scores '
  + 'reading E at z = 0 (the shape that used to silently drop)', async () => {
  assert.deepEqual(infer(ROWS23_INLINE).map((e: any) => e.message), []);
  const got = await scoreOf(ROWS23_INLINE
    + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = 0.5))\n');
  assert.ok(Math.abs(got - AT_0) < F64_TOL, `got ${got}, oracle ${AT_0}`);
});

test('row 3 (inline) — the inline kernelof component tracks the fed input in '
  + 'BOTH coordinates, same as the named spelling', async () => {
  const got = await scoreOf(ROWS23_INLINE
    + 'ld = logdensityof(KJ(z = 5.0), record(p = 6.0, q = 5.5))\n');
  assert.ok(Math.abs(got - AT_0) < F64_TOL, `got ${got}, oracle ${AT_0}`);
});

test('row 2/3 (inline, positional) — the same inline component resolves in '
  + 'the positional spelling too', async () => {
  const src = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
KJ = joint(K1, kernelof(u, z = z))
ld = logdensityof(KJ(z = 0.0), [1.0, 0.5])
`;
  assert.deepEqual(infer(src).map((e: any) => e.message), []);
  const got = await scoreOf(src);
  assert.ok(Math.abs(got - AT_0) < F64_TOL, `got ${got}, oracle ${AT_0}`);
});

// ── rows 4-5: the closed-`u` variant, `q` written inline as `lawof(u)` ──────

const ROWS45_INLINE = `
z = elementof(reals)
u ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = u + z, sigma = 1.0)
K1 = kernelof(a1, z = z)
KJ = joint(p = K1, q = lawof(u))
`;

test('row 4 (inline) — a CLOSED shared node keeps the inline measure component '
  + 'legal and scores reading E at z = 0', async () => {
  assert.deepEqual(infer(ROWS45_INLINE).map((e: any) => e.message), []);
  const got = await scoreOf(ROWS45_INLINE
    + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = 0.5))\n');
  assert.ok(Math.abs(got - AT_0) < F64_TOL, `got ${got}, oracle ${AT_0}`);
});

test('row 5 (inline) — the closed-`u` measure coordinate IGNORES the input, '
  + 'inline exactly as it does named', async () => {
  const got = await scoreOf(ROWS45_INLINE
    + 'ld = logdensityof(KJ(z = 5.0), record(p = 6.0, q = 5.5))\n');
  assert.ok(Math.abs(got - CLOSED_U_AT_5) < F64_TOL, `got ${got}, oracle ${CLOSED_U_AT_5}`);
});

// ── sampling must not escape the fix for the illegal shape ──────────────────

test('the illegal inline shape refuses SAMPLING too', async () => {
  const { ctx } = ctxFor(ROW1_ILLEGAL_INLINE + 'S = KJ(z = 0.0)\n', 8);
  await assert.rejects(async () => materialiser.materialiseMeasure('S', ctx),
    /no derivation for 'S'/);
});
