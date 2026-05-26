'use strict';

// Regression tests for the profile-plot self-ref classifier
// (materialiser.classifyProfileSelfRefs) and the density-level
// invariant it protects.
//
// Background. The viewer's profile-plot path (render-profile.ts)
// inlines a likelihood / function body, then needs a `fixedEnv` map
// of single representative values for every value-typed self-ref in
// the inlined IR (since profileN sweeps ONE axis and holds the rest
// fixed). The "what counts as a self-ref needing a representative"
// question has three subtleties:
//
//   1. Built-in distribution names (`Normal`, `MvNormal`, …) appear
//      as `(ref self <name>)` in expanded measure IRs but are not
//      bindings; they must be skipped.
//   2. Function-like bindings (`fn` / `functionof` / `kernelof` /
//      `bijection`) are consulted by name at sample/density dispatch
//      and don't materialise as values; they must be skipped.
//   3. Fixed-phase bindings (literal arrays, external inputs, fixed
//      reductions) already live in the worker session env via the
//      `setEnv merge:false` push that derivations.ts emits with the
//      `fixedValues` map. Overriding them via samples[0] of a
//      materialised constant-atom measure collapses, e.g., an array
//      literal `x_data = [1.1, 1.5, 1.3, 1.4]` down to the scalar
//      1.1 — which then silently breaks every downstream broadcast
//      that consumes the array.
//
// The third bug was the empty-likelihood-plot symptom on linear-
// regression: `L = likelihoodof(forward_kernel, record(y=y_data))`
// where the kernel body is `lawof(record(y = y))`, `y ~ Normal.(
// alpha .+ beta .* x_data, sigma)`. profileN over alpha showed an
// empty plot at every range because `x_data` was being clobbered to
// 1.1 in the fixedEnv override layer.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser, density } =
  require('..');

const LINREG = [
  'x_data = [1.1, 1.5, 1.3, 1.4]',
  'y_data = [3.2, 4.1, 3.4, 3.9]',
  'sigma2 ~ InverseGamma(5, 5)',
  'sigma = sqrt(sigma2)',
  'alpha ~ Normal(0, sigma * 3)',
  'beta ~ Normal(0, sigma * 3)',
  'means = alpha .+ beta .* x_data',
  'y ~ Normal.(means, sigma)',
  'forward_kernel = kernelof(record(y = y),' +
    '  alpha = alpha, beta = beta, sigma = sigma)',
  'L = likelihoodof(forward_kernel, record(y = y_data))',
].join('\n');

function makeProfileIR() {
  const r = processSource(LINREG);
  const bindings = orchestrator.liftInlineSubexpressions(r.bindings);
  const built = orchestrator.buildDerivations(bindings);
  const sig = orchestrator.signatureOf('L', bindings);
  assert.ok(sig, 'signature for L');
  assert.equal(sig.kind, 'likelihood');
  let ir = sig.body;
  ir = orchestrator.expandMeasureRefsInIR(ir, built.derivations);
  const paramNames = sig.inputs.map((i: any) => i.paramName);
  ir = orchestrator.inlineForProfile(
    ir, paramNames, bindings, built.derivations);
  const observed = orchestrator.resolveIRToValue(
    sig.obsIR, bindings, built.fixedValues);
  return { ir, observed, built, bindings };
}

test('classifyProfileSelfRefs drops fixed-phase array bindings', () => {
  const { ir, built, bindings } = makeProfileIR();

  // Bare collectSelfRefs returns both `Normal` (built-in) and
  // `x_data` (fixed-phase array). The viewer must drop both before
  // overriding fixedEnv.
  const raw = Array.from(orchestrator.collectSelfRefs(ir));
  assert.ok(raw.indexOf('Normal') !== -1,
    'sanity: raw self-refs include built-in Normal');
  assert.ok(raw.indexOf('x_data') !== -1,
    'sanity: raw self-refs include fixed-phase x_data');

  const { perAtomNames } = materialiser.classifyProfileSelfRefs(
    ir, bindings, built.fixedValues);

  assert.ok(perAtomNames.indexOf('Normal') === -1,
    'built-in distribution name must not need a per-atom representative');
  assert.ok(perAtomNames.indexOf('x_data') === -1,
    'fixed-phase x_data lives in session env via setEnv, not in fixedEnv');
});

test('classifyProfileSelfRefs filter logic across kinds', () => {
  // Construct a small synthetic IR + bindings/fixedValues map to
  // pin down each branch of the classifier independently of the
  // linreg fixture.
  const ir = {
    kind: 'call', op: 'record', fields: [
      { name: 'a', value: { kind: 'ref', ns: 'self', name: 'Normal' } },
      { name: 'b', value: { kind: 'ref', ns: 'self', name: 'x_data' } },
      { name: 'c', value: { kind: 'ref', ns: 'self', name: 'fn_thing' } },
      { name: 'd', value: { kind: 'ref', ns: 'self', name: 'r_var' } },
    ],
  };
  const bindings = new Map<string, any>([
    ['x_data',   { name: 'x_data',   type: 'literal',   ir: {} }],
    ['fn_thing', { name: 'fn_thing', type: 'functionof', ir: {} }],
    ['r_var',    { name: 'r_var',    type: 'draw',      ir: {} }],
  ]);
  const fixedValues = new Map([['x_data', [1.0, 2.0]]]);

  const { perAtomNames } = materialiser.classifyProfileSelfRefs(
    ir, bindings, fixedValues);

  // Normal: not a binding (built-in)         → skipped.
  // x_data: binding, but in fixedValues      → skipped.
  // fn_thing: function-like binding          → skipped.
  // r_var:  random-phase value binding       → kept.
  assert.deepEqual(perAtomNames.sort(), ['r_var']);
});

test('isFunctionLikeBinding predicate is exposed and recognises all callable-layer producer tags', () => {
  // The viewer's `overrides.computeAutoValues` and `render-kernel`'s
  // bindingSourceLookups filter use this predicate to avoid the
  // array-collapse failure mode (engine-concepts §19; flatppl-js
  // commit e9984f3 fixed the analogous bug in render-profile).
  // Pin the surface: the predicate is exposed from `materialiser`,
  // and recognises every callable-layer producer tag including the
  // Phase 1 addition `'fchain'`.
  assert.equal(typeof materialiser.isFunctionLikeBinding, 'function',
    'materialiser.isFunctionLikeBinding is exposed');
  assert.equal(typeof materialiser.isCallableLayerBinding, 'function',
    'materialiser.isCallableLayerBinding is exposed');
  for (const t of ['fn', 'functionof', 'kernelof', 'bijection', 'fchain']) {
    assert.equal(materialiser.isFunctionLikeBinding({ type: t }), true,
      'producer tag ' + t + ' is callable-layer');
  }
  for (const t of ['draw', 'lawof', 'literal', 'call', 'input']) {
    assert.equal(materialiser.isFunctionLikeBinding({ type: t }), false,
      'producer tag ' + t + ' is NOT callable-layer');
  }
  // The type-driven predicate keys on inferredType.kind ∈ {function,
  // kernel} — a kernel-typed binding whose producer tag is the
  // catch-all 'call' (e.g. a kernel-first jointchain).
  const kernelTypedCall = {
    type: 'call',
    inferredType: { kind: 'kernel', inputs: [{ name: 'theta', type: {} }], result: {} },
  };
  assert.equal(materialiser.isFunctionLikeBinding(kernelTypedCall), false,
    'producer-tag predicate misses kernelType-via-call bindings (tag is "call")');
  assert.equal(materialiser.isCallableLayerBinding(kernelTypedCall), true,
    'type-driven predicate catches kernelType-via-call bindings');
});

test('linear-regression likelihood: density.logDensityN is finite ' +
     'at MLE when fixed-phase x_data is in baseEnv', () => {
  // Pin the density-level invariant the classifier protects.
  // `density.logDensityN` walks the broadcast IR per-atom via
  // `_broadcastLogical` (ops-declarations.ts), which evaluates source
  // args against ctx.env — that env MUST contain fixed-phase parents
  // like `x_data`. The worker layers session env + per-call fixedEnv
  // before delegating, so as long as x_data flows via setEnv (and
  // isn't overwritten by a samples[0] scalar from a constant-atom
  // measure), the batched path succeeds.
  const { ir, observed, built } = makeProfileIR();

  // What setEnv pushes for fixed-phase bindings: the Value form
  // produced by orchestrator.fixedValues.
  const x_data = built.fixedValues.get('x_data');
  assert.ok(x_data, 'fixedValues has x_data');

  // Sweep alpha across [0, 1] at 5 points; the OLS MLE is ≈ 0.55.
  const sweep = new Float64Array(5);
  for (let i = 0; i < 5; i++) sweep[i] = i * 0.25;
  const refArrays = { alpha: sweep };
  const baseEnv = {
    beta:   2.3429,
    sigma:  0.1115,
    x_data: x_data,
  };
  const logps = density.logDensityN(
    ir, observed, refArrays, 5, { baseEnv });
  for (let i = 0; i < 5; i++) {
    assert.ok(Number.isFinite(logps[i]),
      'log-density at alpha=' + sweep[i] + ' must be finite (got '
      + logps[i] + ')');
  }
  // Peak should be at the third grid point (α=0.5, nearest to the
  // OLS MLE 0.55).
  let argmax = 0;
  for (let i = 1; i < 5; i++) {
    if (logps[i] > logps[argmax]) argmax = i;
  }
  assert.equal(argmax, 2,
    'peak should be at the grid point closest to the MLE');
});
