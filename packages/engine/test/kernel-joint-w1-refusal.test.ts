'use strict';

// W1: a MEASURE component of a kernel `joint` sharing a stochastic node whose
// ancestor another component binds as a boundary input.
//
// The engine used to SCORE this shape — the fed input reached inside the measure
// component through the shared node, rewriting the measure into a kernel. The
// ruling (flatppl-dev/kernel-joint-w1-maths.md, VERDICT, FORCED) makes it a
// static error: "A measure component may share a stochastic node with a kernel
// component only if no ancestor of the shared node is any component's boundary
// input."
//
// SPEC ANCHOR — flatppl-design docs/06-measure-algebra.md, `joint` entry
// (flatppl-design#85, head 291949a), quoted verbatim:
//
//   "Components that share a stochastic node must agree on that node's
//   ancestry: every ancestor of the shared node that any component binds as a
//   boundary input must be bound by every sharing component, under the same
//   input name. A `joint` in which a sharing component binds such an ancestor
//   under a different name, or does not bind it at all — in particular a
//   measure component, which binds nothing — is a static error. Measure
//   components are permitted and are the nullary case: they ignore the input. A
//   measure component may be parameterized and may share stochastic nodes with
//   kernel components; only a shared node with a boundary-bound ancestor is
//   excluded, by the naming clause above."
//
// §04 "Specifying reification boundaries": "A specified boundary node `a` can be
// thought of as being substituted with a new node, generated via
// `elementof(valueset(a))`, in the reified graph." — the decoupling the old
// behavior repealed.
//
// ORACLES ARE INDEPENDENT of this engine and of flatppl-rust (which evaluates no
// density at all). Each value below was computed twice, by routes that share no
// algebra: scipy's `multivariate_normal(...).logpdf`, and a closed form written
// as the chain `logpdf(N(mq,1), q) + logpdf(N(q + (mp-mq), 1), p)` that never
// forms a covariance matrix. The record law of (a1, u) is
// MvNormal([mp, mq], [[2, 1], [1, 1]]).
//
//   MvNormal([0,0], S) at (1, 0.5)     →  -2.0878770664093453
//   MvNormal([5,5], S) at (6, 5.5)     →  -2.0878770664093453
//   MvNormal([5,0], S) at (6, 5.5)     → -27.087877066409344
//
// The last two DISCRIMINATE the ruling: the both-bind spelling tracks the fed
// input in both coordinates, the closed-`u` variant holds its measure coordinate
// still while the kernel coordinate moves.

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

// The repro: `u` is shared, and its ancestor `z` is K1's boundary input while
// `M` binds nothing.
const W1 = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = lawof(u)
KJ = joint(p = K1, q = M)
`;

// §4 reading E: the same law, spelled with both components binding `z`. Written
// in the NAMED spelling — an inline `kernelof` joint component silently drops
// (filed separately), so an inline spelling would pass for the wrong reason.
const BOTH_BIND = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
K2 = kernelof(u, z = z)
KJ = joint(p = K1, q = K2)
`;

// §5 first bullet: the shared node is CLOSED — no ancestor of `u` is anyone's
// boundary — so the measure coordinate genuinely ignores the input.
const CLOSED_U = `
z = elementof(reals)
u ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = u + z, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = lawof(u)
KJ = joint(p = K1, q = M)
`;

// §5 third bullet: NO shared stochastic node. `w` is parameterized by the
// ambient `z` and belongs to no kernel component's trace.
const NON_SHARING = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
w ~ Normal(mu = z, sigma = 2.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = lawof(w)
KJ = joint(p = K1, q = M)
`;

const SHARED_ANCESTRY_ERROR
  = /share the stochastic node 'u'.*ancestor 'z' is bound as input 'z'.*not bound at all by a measure component, which binds nothing/s;

// ── row 1: the illegal shape refuses ────────────────────────────────────────

test('W1 — a measure component sharing a boundary-descended node is a STATIC '
  + 'error', () => {
  const errors = infer(W1);
  assert.ok(errors.some((e: any) => SHARED_ANCESTRY_ERROR.test(e.message)),
    'got: ' + errors.map((e: any) => e.message).join(' | '));
});

test('W1 — the error does not depend on component ORDER', () => {
  // The measure component first, so the non-binder is the left-hand side of the
  // pair. Reporting only one direction would leave half the shape scoring.
  const errors = infer(`
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = lawof(u)
KJ = joint(p = M, q = K1)
`);
  assert.ok(errors.some((e: any) => SHARED_ANCESTRY_ERROR.test(e.message)),
    'got: ' + errors.map((e: any) => e.message).join(' | '));
});

test('W1 — the density query refuses instead of answering -2.0878770664093453',
  async () => {
    // The pre-fix engine returned reading A's value here: the log pdf of
    // MvNormal([0,0], [[2,1],[1,1]]) at (1, 0.5). That number is not
    // mis-evaluated, it answers a rewritten question (§8, "wrong by
    // illegality"), so the test pins the REFUSAL, never a value.
    const src = W1
      + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = 0.5))\n';
    const { ctx } = ctxFor(src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'),
      /no derivation for 'ld'/);
  });

test('W1 — SAMPLING refuses too, so no moments from the rewritten shape escape',
  async () => {
    // Pre-fix this drew a correlated (p, q) record from the substituted DAG.
    const { ctx } = ctxFor(W1 + 'S = KJ(z = 0.0)\n', 64);
    await assert.rejects(async () => materialiser.materialiseMeasure('S', ctx),
      /no derivation for 'S'/);
  });

// ── rows 2-3: reading E, the both-bind spelling, is legal and tracks the input

test('both components binding `z` is legal and scores reading E at z = 0',
  async () => {
    assert.deepEqual(infer(BOTH_BIND).map((e: any) => e.message), []);
    const got = await scoreOf(BOTH_BIND
      + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = 0.5))\n');
    assert.ok(Math.abs(got - AT_0) < F64_TOL, `got ${got}, oracle ${AT_0}`);
  });

test('the both-bind spelling tracks the fed input in BOTH coordinates', async () => {
  // At z = 5 the mean shifts to [5, 5], so the same residual gives the same
  // number — the measure-side coordinate is a kernel here by declaration.
  const got = await scoreOf(BOTH_BIND
    + 'ld = logdensityof(KJ(z = 5.0), record(p = 6.0, q = 5.5))\n');
  assert.ok(Math.abs(got - AT_0) < F64_TOL, `got ${got}, oracle ${AT_0}`);
});

// ── rows 4-5: the closed-`u` variant is legal and holds its measure still ────

test('a CLOSED shared node keeps the measure component legal (§5, first bullet)',
  async () => {
    assert.deepEqual(infer(CLOSED_U).map((e: any) => e.message), []);
    const got = await scoreOf(CLOSED_U
      + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = 0.5))\n');
    assert.ok(Math.abs(got - AT_0) < F64_TOL, `got ${got}, oracle ${AT_0}`);
  });

test('the closed-`u` measure coordinate IGNORES the input — Q3 verbatim',
  async () => {
    // The discriminator: at z = 5 the kernel coordinate moves to mean 5 while
    // the q marginal stays Normal(0, 1), which costs 25 nats against the
    // both-bind law. Reading A cannot produce this number.
    const got = await scoreOf(CLOSED_U
      + 'ld = logdensityof(KJ(z = 5.0), record(p = 6.0, q = 5.5))\n');
    assert.ok(Math.abs(got - CLOSED_U_AT_5) < F64_TOL,
      `got ${got}, oracle ${CLOSED_U_AT_5}`);
  });

// ── §5 third bullet: no shared node, so no static error — and no substitution

test('a NON-SHARING parameterized measure component is not a static error',
  () => {
    // Decoupling already makes this coherent (§5, third bullet: "no retained
    // node forces the two views onto one parent slot"), so the clause must not
    // reach it.
    assert.deepEqual(infer(NON_SHARING).map((e: any) => e.message), []);
  });

test('the hoist does not substitute the fed input into a non-sharing measure '
  + 'component', async () => {
  // Pre-fix, `q`'s law was Normal(fed z, 2): at z = 5, (p, q) = (6, 0.5) scored
  // -5.658847837249263 = logpdf(N(5,√2), 6) + logpdf(N(5,2), 0.5), the
  // substituted reading, against -3.1588478372492634 for the ambient z = 0.
  // The correct coordinate reads the AMBIENT `z`, which is unbound here, so the
  // query refuses — kernel-joint-w1-maths.md §9 item 2: "refusing when the
  // ambient parameter is unbound is correct behavior".
  for (const feed of ['z = 0.0', 'z = 5.0']) {
    const { ctx } = ctxFor(NON_SHARING
      + `ld = logdensityof(KJ(${feed}), record(p = 6.0, q = 0.5))\n`, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'),
      /no derivation for 'ld'/, `KJ(${feed}) must refuse, not substitute`);
  }
});

test('a measure component over a node with NO boundary ancestor still scores '
  + 'with a kernel component present', async () => {
  // The guard fires only on a measure component that actually reaches a hoisted
  // boundary node. `w` here is closed, so the joint stays scoreable.
  //   logpdf(N(5,√2), 6) + logpdf(N(0,2), 5.5) = -6.908847837249263
  const oracle = -6.908847837249263;
  const got = await scoreOf(`
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
w ~ Normal(mu = 0.0, sigma = 2.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = lawof(w)
KJ = joint(p = K1, q = M)
ld = logdensityof(KJ(z = 5.0), record(p = 6.0, q = 5.5))
`);
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
});
