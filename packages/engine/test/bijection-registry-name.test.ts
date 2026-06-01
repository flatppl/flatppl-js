'use strict';

// =====================================================================
// bijection-registry-name.test.ts — Phase 5.1 Session 5c
// =====================================================================
//
// Pins the additive contract extension that lets a bijection binding
// signal "I'm a recognised closed-form bijection — consult the open
// registry's atom-batched fast paths" via an optional
// `registryName: string` field on `binding.bijection`.
//
// Engine-concepts §22 architectural reframe: multivariate distributions
// decompose as `pushfwd(known_bijection, iid(scalar, D))`. Producers
// that recognise a closed-form bijection mark `binding.bijection.
// registryName` with the registry-entry name (e.g. 'affine'); consumers
// (matPushfwd / walkPushfwd vector-base — Session 5d+) read that
// marker to dispatch through the bijection-registry's atom-batched
// fast paths rather than the generic AST-eval cold path.
//
// **Invariant (load-bearing).** registryName is PURELY ADDITIVE. When
// it is set, the binding MUST still carry valid fName / fInvName /
// logVolume — the registry path is an OPTIMISATION over the AST
// path, not a REPLACEMENT. matPushfwd's existing resolveFnBody →
// fName → callable body walk continues to find a body for every
// bijection binding regardless of registryName presence. This
// eliminates the "degenerate binding" risk surface: callers that
// don't opt into the registry path work identically to pre-5c.
//
// Session 5c lands ONLY the contract extension at
// `derivations.resolveBijectionMeta` — registryName flows from
// `binding.bijection.registryName` to `ir.bijection.registryName` via
// expandMeasureIR. The downstream consumers (matPushfwd vector-base,
// walkPushfwd vector-base) opt in during Session 5d+.
//
// Tests:
//   1. Backward compat — bindings WITHOUT registryName work identically;
//      ir.bijection.registryName is undefined.
//   2. Registry-name round-trip — bindings WITH registryName have it
//      forwarded onto ir.bijection.registryName end-to-end; the rest of
//      the IR remains identical; density evaluation produces the SAME
//      numeric result (5c doesn't yet branch on registryName).
//   3. resolveFnBody invariant — a binding with registryName still
//      resolves a callable body (the AST path is untouched).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const density = require('../density.ts');

const SAMPLE_COUNT = 256;
const ROOT_SEED    = 0xB1737CFD;

function makeCtx(source: any) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => Promise.resolve(worker.handle(msg)),
    sampleCount: SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

// =====================================================================
// 1. Backward compat — no registryName → ir.bijection.registryName undef
// =====================================================================
//
// Pin: an unmodified `bijection(f, f_inv, logvolume)` produces an
// ir.bijection whose registryName field is undefined. Density
// evaluation runs the existing scalar AST path; the result is
// unchanged from pre-5c behavior.

test('bijection contract (5c): unmarked binding has ir.bijection.registryName undefined', () => {
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
`);
  const expanded = orchestrator.expandMeasure(
    'LN', { derivations: ctx.derivations, bindings: ctx.bindings });
  assert.ok(expanded, 'pushfwd LN expands');
  assert.ok(expanded.bijection,
    'unmarked bijection still attaches bijection metadata');
  assert.equal(expanded.bijection.registryName, undefined,
    'unmarked binding produces ir.bijection.registryName === undefined');
  // Existing fields remain present.
  assert.ok(expanded.bijection.fInv && expanded.bijection.fInv.body,
    'fInv.body present');
  assert.ok(expanded.bijection.logVolume,
    'logVolume present');
});

// =====================================================================
// 2. Registry-name round-trip — synthetic mark on binding.bijection
// =====================================================================
//
// Pin: after buildDerivations runs, we hand-modify the bijection
// binding's `bijection.registryName` and re-expand. The new field
// flows onto ir.bijection.registryName intact. Existing fields
// (fInv.body, logVolume) remain byte-equivalent; density evaluation
// produces the SAME numeric result (5c does NOT yet branch on
// registryName — Session 5d+ adds the consumer path).

test('bijection contract (5c): registryName round-trips onto ir.bijection.registryName', () => {
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
`);
  // Synthetic producer: mark the bijection binding with a registryName.
  // Session 5d+'s lift-time MvNormal lowering will perform this kind
  // of post-pass automatically; here we do it by hand to verify the
  // contract surface.
  const bBinding = ctx.bindings.get('b');
  assert.ok(bBinding && bBinding.bijection,
    'b is a bijection binding with metadata');
  bBinding.bijection.registryName = 'exp';

  const expanded = orchestrator.expandMeasure(
    'LN', { derivations: ctx.derivations, bindings: ctx.bindings });
  assert.ok(expanded.bijection, 'bijection metadata still attached');
  assert.equal(expanded.bijection.registryName, 'exp',
    'registryName forwarded onto ir.bijection.registryName');
  // Other fields unchanged — registry marker is purely additive.
  assert.ok(expanded.bijection.fInv && expanded.bijection.fInv.body,
    'fInv.body still present (additive invariant)');
  assert.ok(expanded.bijection.logVolume,
    'logVolume still present (additive invariant)');
});

// =====================================================================
// 3. Numerical equivalence — registryName presence doesn't alter density
// =====================================================================
//
// Pin: density of `pushfwd(b_marked, M)` matches density of
// `pushfwd(b_unmarked, M)` byte-for-byte. The registry path is
// opt-in; consumers in 5c don't branch on the marker.

test('bijection contract (5c): registryName does NOT alter density numerically', () => {
  // Two parallel ctxes — same model, one with the marker, one without.
  const sourceUnmarked = `
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
`;
  const sourceMarked = sourceUnmarked;     // identical surface

  const ctxUnmarked = makeCtx(sourceUnmarked);
  const ctxMarked = makeCtx(sourceMarked);
  ctxMarked.bindings.get('b').bijection.registryName = 'exp';

  const obs = 2.5;
  const irUnmarked = orchestrator.expandMeasure(
    'LN', { derivations: ctxUnmarked.derivations, bindings: ctxUnmarked.bindings });
  const irMarked = orchestrator.expandMeasure(
    'LN', { derivations: ctxMarked.derivations, bindings: ctxMarked.bindings });

  const lpUnmarked = density.logDensity(irUnmarked, obs, {}, {});
  const lpMarked   = density.logDensity(irMarked, obs, {}, {});

  assert.ok(Number.isFinite(lpUnmarked),
    'unmarked density is finite');
  assert.ok(Number.isFinite(lpMarked),
    'marked density is finite');
  assert.equal(lpMarked, lpUnmarked,
    'registryName presence does not alter density (5c is plumbing only; '
    + '5d+ adds the consumer branch)');
});

// =====================================================================
// 4. resolveFnBody invariant — marked binding still resolves to a body
// =====================================================================
//
// Direct response to the scout's HIGH-RISK concern: with the
// additive invariant, resolveFnBody never spuriously returns null
// because of registryName presence. The fName walk continues to
// find the forward function's body.

test('bijection contract (5c): resolveFnBody returns a body for a marked binding', () => {
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
`);
  const bBinding = ctx.bindings.get('b');
  bBinding.bijection.registryName = 'exp';

  const shared = require('../materialiser-shared.ts');
  const fnInfo = shared.resolveFnBody(bBinding, ctx.bindings);
  assert.ok(fnInfo,
    'resolveFnBody returns a callable body for a registryName-marked '
    + 'binding (additive invariant: fName still present)');
  assert.ok(fnInfo.body, 'fnInfo.body is the forward function body');
  assert.equal(typeof fnInfo.paramName, 'string',
    'fnInfo.paramName is a binding-input name');
});

// =====================================================================
// 5. matPushfwd smoke — marked binding samples identically
// =====================================================================
//
// Driving matPushfwd with a marked binding: the AST path runs
// unchanged because registryName presence doesn't divert the
// resolveFnBody walk. Density check in test 3 + sample check here
// jointly verify the seam holds in both directions.

// =====================================================================
// 6. Session 5d commit 1 — paramIRs round-trip
// =====================================================================
//
// Pin: a binding marked with both registryName AND paramIRs has BOTH
// fields forwarded onto ir.bijection. The downstream consumers
// (matPushfwd / walkPushfwd vector-base — Session 5d commits 2-3) read
// these together to dispatch through the registry. Producer contract:
// when registryName is set, paramIRs MUST also be set.

test('bijection contract (5d): paramIRs round-trip alongside registryName', () => {
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
`);
  const bBinding = ctx.bindings.get('b');
  // Synthetic producer marker — paramIRs alongside registryName.
  bBinding.bijection.registryName = 'affine';
  bBinding.bijection.paramIRs = {
    L: { kind: 'lit', value: [[2.0]] },
    b: { kind: 'lit', value: [3.0] },
  };

  const expanded = orchestrator.expandMeasure(
    'LN', { derivations: ctx.derivations, bindings: ctx.bindings });
  assert.equal(expanded.bijection.registryName, 'affine',
    'registryName still forwards');
  assert.ok(expanded.bijection.paramIRs,
    'paramIRs forwarded onto ir.bijection');
  assert.deepEqual(expanded.bijection.paramIRs.L,
    { kind: 'lit', value: [[2.0]] },
    'paramIRs.L round-trips intact');
  assert.deepEqual(expanded.bijection.paramIRs.b,
    { kind: 'lit', value: [3.0] },
    'paramIRs.b round-trips intact');
  // Additive invariant: AST fields still present.
  assert.ok(expanded.bijection.fInv && expanded.bijection.fInv.body);
  assert.ok(expanded.bijection.logVolume);
});

test('bijection contract (5d): paramIRs without registryName also rides through', () => {
  // Lenient on the producer side: forwarding is independent. Consumers
  // are responsible for checking the pair together.
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
`);
  ctx.bindings.get('b').bijection.paramIRs = { foo: { kind: 'lit', value: 1 } };
  const expanded = orchestrator.expandMeasure(
    'LN', { derivations: ctx.derivations, bindings: ctx.bindings });
  assert.equal(expanded.bijection.registryName, undefined,
    'registryName absent');
  assert.deepEqual(expanded.bijection.paramIRs,
    { foo: { kind: 'lit', value: 1 } },
    'paramIRs forwards even without registryName');
});

test('bijection contract (5c): matPushfwd samples a marked binding via AST path', async () => {
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
`);
  ctx.bindings.get('b').bijection.registryName = 'exp';

  // Materialise LN; expect a measure with finite samples (LogNormal-
  // distributed via the forward exp transform).
  const m = await ctx.getMeasure('LN');
  assert.ok(m && m.samples,
    'materialised measure has samples');
  for (let i = 0; i < m.samples.length; i++) {
    assert.ok(Number.isFinite(m.samples[i]),
      `sample ${i} is finite (= ${m.samples[i]})`);
    assert.ok(m.samples[i] > 0,
      `sample ${i} = ${m.samples[i]} should be positive (exp output)`);
  }
});
