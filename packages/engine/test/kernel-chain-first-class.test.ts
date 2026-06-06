'use strict';

// Integration tests for kernel-first jointchain / kchain (Phase 2 of
// the callable-layer work; engine-concepts §19). Pins:
//
//   1. Kernel-first chain bindings carry kernelType inferredType with
//      step_0's inputs as the residual.
//   2. Closed-first chain bindings carry measureType (existing
//      behaviour, unchanged).
//   3. The kernel↔measure collapse (spec §06 line 86-91) falls out:
//      a kernel-first chain whose residual is empty (no kernel-first
//      kernel inputs) collapses to measure.
//   4. The materialiser entry-point gate refuses direct
//      materialisation of kernel-first chains with a clear diagnostic.
//   5. Step-boundary type mismatches surface as typeinfer diagnostics
//      anchored at the failing step.
//   6. The predicates (isFunctionLikeBinding, isCallableLayerBinding)
//      both recognise kernel-typed chain bindings.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function infer(src: string) {
  return processSource(src);
}

test('jointchain closed-first: inferredType is measureType', () => {
  const r = infer(`
prior = joint(theta = Normal(0, 1))
mu = elementof(reals)
k_inner = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)
joint_dep = jointchain(prior, k_inner)
`);
  const b = r.loweredModule.bindings.get('joint_dep');
  assert.ok(b.inferredType,
    'joint_dep should have an inferredType');
  assert.equal(b.inferredType.kind, 'measure',
    'closed-first chain ⇒ measureType (got '
    + JSON.stringify(b.inferredType) + ')');
});

test('jointchain kernel-first: inferredType is kernelType with residual inputs', () => {
  // K0 is a kernel taking theta → measure<real>. K1 is a kernel
  // taking real → measure<real>. The chain `jointchain(K0, K1)` is
  // itself a kernel: its inputs are K0's inputs (theta), its variate
  // retains both K0's and K1's outputs.
  const r = infer(`
theta = elementof(reals)
K0 = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
mu = elementof(reals)
K1 = functionof(Normal(mu = mu, sigma = 0.5), mu = mu)
chain = jointchain(K0, K1)
`);
  const b = r.loweredModule.bindings.get('chain');
  assert.ok(b.inferredType, 'chain should have an inferredType');
  assert.equal(b.inferredType.kind, 'kernel',
    'kernel-first jointchain ⇒ kernelType (got '
    + JSON.stringify(b.inferredType).slice(0, 200) + ')');
  // Residual inputs are K0's inputs — `theta` survives because no
  // later step bound it. The exact input name comes from K0's
  // signature; we check structurally.
  assert.ok(Array.isArray(b.inferredType.inputs) && b.inferredType.inputs.length >= 1,
    'kernel-first chain residual inputs preserved');
  // The result must be a measure (kernel returns a measure).
  assert.equal(b.inferredType.result.kind, 'measure');
});

test('kchain kernel-first: inferredType is kernelType', () => {
  const r = infer(`
theta = elementof(reals)
K0 = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
mu = elementof(reals)
K1 = functionof(Normal(mu = mu, sigma = 0.5), mu = mu)
chain = kchain(K0, K1)
`);
  const b = r.loweredModule.bindings.get('chain');
  assert.equal(b.inferredType.kind, 'kernel',
    'kernel-first kchain ⇒ kernelType');
});

test('predicates recognise kernel-first chain bindings', () => {
  const r = infer(`
theta = elementof(reals)
K0 = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
mu = elementof(reals)
K1 = functionof(Normal(mu = mu, sigma = 0.5), mu = mu)
chain = jointchain(K0, K1)
`);
  const lowered = r.loweredModule.bindings.get('chain');
  // chain binding.type is 'call' (the existing analyzer doesn't tag
  // jointchain bindings specifically — they classify via the
  // jointchain derivation). The type-driven predicate sees the
  // kernelType inferredType.
  const matShared = require('../materialiser-shared.ts');
  const synthetic = { type: 'call', inferredType: lowered.inferredType };
  assert.equal(matShared.isCallableLayerBinding(synthetic), true,
    'isCallableLayerBinding should recognise kernelType chain binding');
});

test('materialiser entry-point gate refuses kernel-first chain materialisation', () => {
  const r = infer(`
theta = elementof(reals)
K0 = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
mu = elementof(reals)
K1 = functionof(Normal(mu = mu, sigma = 0.5), mu = mu)
chain = jointchain(K0, K1)
`);
  const lifted = orchestrator.liftInlineSubexpressions(r.bindings);
  const built = orchestrator.buildDerivations(lifted);
  // Synthesize ctx and attempt materialisation.
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 12345 });
  // Mirror the analyzer's bindings.type onto the lowered binding for
  // the gate's tag-based check; also attach inferredType for the
  // type-based check.
  const bindings = new Map();
  for (const [name, b] of lifted.entries()) {
    const lb = r.loweredModule.bindings.get(name);
    const merged = Object.assign({}, b);
    if (lb && lb.inferredType) merged.inferredType = lb.inferredType;
    bindings.set(name, merged);
  }
  const ctx = {
    derivations: built.derivations,
    bindings:    bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (n: any) => materialiser.materialiseMeasure(n, ctx),
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: 32,
    rootSeed:    12345,
  };
  return materialiser.materialiseMeasure('chain', ctx).then(
    () => { throw new Error('expected rejection'); },
    (err: any) => {
      assert.match(err.message, /callable-layer binding/i);
      assert.match(err.message, /chain/);
    }
  );
});

test('jointchain step-boundary type-mismatch surfaces typeinfer diagnostic', () => {
  // prior has scalar variate; K1 has TWO inputs. Multi-input next-
  // step boundary requires a record-typed prev variate (auto-splat);
  // a scalar against 2 inputs is a static mismatch. Structural shape
  // mismatch ⇒ caught by _matchChainBoundary even when individual
  // types are loose.
  const r = infer(`
prior_var ~ Normal(0.0, 1.0)
prior = lawof(prior_var)
mu1 = elementof(reals)
mu2 = elementof(reals)
K1 = functionof(Normal(mu = mu1 + mu2, sigma = 1.0),
                mu1 = mu1, mu2 = mu2)
chain = jointchain(prior, K1)
`);
  const d = r.diagnostics.find((d: any) =>
    /jointchain-retain.*step boundary/i.test(d.message));
  assert.ok(d, 'should have step-boundary diagnostic; got: '
    + JSON.stringify(r.diagnostics.map((x: any) => x.message)));
});

test('jointchain auto-splatting at multi-input boundary (record → kwargs)', () => {
  // Spec §04 + §06 (recently added): record-shaped variates auto-
  // splat into next-step keyword inputs. K0's variate is record(a, b);
  // K1 takes both a and b as kwargs.
  const r = infer(`
theta = elementof(reals)
K0 = functionof(lawof(record(a = theta, b = 2.0 * theta)), theta = theta)
mu1 = elementof(reals)
mu2 = elementof(reals)
K1 = functionof(Normal(mu = mu1 + mu2, sigma = 1.0),
                a = mu1, b = mu2)
chain = jointchain(K0, K1)
`);
  const stepDiag = r.diagnostics.find((d: any) =>
    /step boundary/i.test(d.message));
  assert.equal(stepDiag, undefined,
    'record→kwargs auto-splat should compose cleanly; got: '
    + JSON.stringify(r.diagnostics.map((d: any) => d.message)));
  const chain = r.loweredModule.bindings.get('chain');
  assert.equal(chain.inferredType.kind, 'kernel');
});

test('jointchain keyword form: retained record variate has kwarg names', () => {
  // Keyword form `jointchain(name1 = M1, name2 = K1)` produces a
  // record-shaped variate per spec §06 line 230-232.
  const r = infer(`
mu_dist = Normal(0, 1)
mu = elementof(reals)
K_inner = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)
chain = jointchain(mu = mu_dist, x = K_inner)
`);
  const b = r.loweredModule.bindings.get('chain');
  assert.equal(b.inferredType.kind, 'measure',
    'closed-first keyword jointchain ⇒ measure of record');
  assert.equal(b.inferredType.domain.kind, 'record',
    'variate of keyword jointchain is a record');
});

test('nested chain: kernel-first chain consumed as step of outer chain (classifier-level)', () => {
  // outer = jointchain(prior, inner) where inner = jointchain(K0, K1)
  // is a kernel-first chain. The classifier must recognise inner as
  // kernel-typed (via inferredType.kind === 'kernel') even though
  // inner.type is 'call' (not 'functionof'/'kernelof'). Engine-
  // concepts §19: kernel-typed bindings produced by ordinary calls
  // (jointchain, kchain) are first-class kernel components.
  const r = infer(`
theta = elementof(reals)
K0 = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
mu = elementof(reals)
K1 = functionof(Normal(mu = mu, sigma = 0.5), mu = mu)
inner = jointchain(K0, K1)
prior = Normal(0.0, 1.0)
outer = jointchain(prior, inner)
`);
  const lifted = orchestrator.liftInlineSubexpressions(r.bindings);
  const built = orchestrator.buildDerivations(lifted);
  assert.ok(built.derivations.outer,
    'outer should have a jointchain derivation (classifier recognises inner as kernel)');
  assert.equal(built.derivations.outer.kind, 'jointchain');
  assert.equal(built.derivations.outer.steps.length, 2);
  assert.equal(built.derivations.outer.steps[0].ref, 'prior');
  assert.equal(built.derivations.outer.steps[1].ref, 'inner');
  assert.equal(built.derivations.outer.steps[1].role, 'kernel',
    'inner step is classified as a kernel');
  // outer's inferredType: closed-first chain → measureType (kernel↔
  // measure collapse at empty residual).
  const outer = r.loweredModule.bindings.get('outer');
  assert.equal(outer.inferredType.kind, 'measure');
});

test('nested chain: materialisation flattens inner into outer step list', () => {
  // outer = jointchain(prior, inner) where inner = jointchain(K0, K1)
  // materialises end-to-end via the splice-at-materialise-time path
  // in `_flattenNestedChainSteps`. The retain interpretation flattens
  // inner's variates into outer's flat retain list (spec §06's
  // `cat(variates)` semantics applied to nested chains; the nested-
  // retention alternative is spec-ambiguous and harder to compose).
  // Resulting outer variate is a 3-tuple [prior, inner_K0, inner_K1].
  const r = infer(`
theta = elementof(reals)
K0 = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
mu = elementof(reals)
K1 = functionof(Normal(mu = mu, sigma = 0.5), mu = mu)
inner = jointchain(K0, K1)
prior = Normal(0.0, 1.0)
outer = jointchain(prior, inner)
`);
  const lifted = orchestrator.liftInlineSubexpressions(r.bindings);
  const built = orchestrator.buildDerivations(lifted);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 12345 });
  const bindings = new Map();
  for (const [name, b] of lifted.entries()) {
    const lb = r.loweredModule.bindings.get(name);
    const merged = Object.assign({}, b);
    if (lb && lb.inferredType) merged.inferredType = lb.inferredType;
    bindings.set(name, merged);
  }
  const ctx: any = {
    derivations: built.derivations,
    bindings:    bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (n: any) => materialiser.materialiseMeasure(n, ctx),
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: 64,
    rootSeed:    12345,
  };
  return materialiser.materialiseMeasure('outer', ctx).then((m: any) => {
    assert.ok(m, 'outer materialises');
    // outer.variate (flat retain) has 3 components: prior, inner_K0, inner_K1.
    // The materialiser yields a tuple-shaped measure (no labels in either
    // outer or inner positional form).
    assert.ok(m.elems || m.fields,
      'outer measure has structured shape (tuple or record)');
    const components = m.elems || Object.values(m.fields || {});
    assert.equal(components.length, 3,
      'flat retain: outer.variate has 3 components [prior, K0, K1]');
    for (const c of components) {
      assert.ok(c.samples && c.samples.length === 64,
        'each component has 64 samples aligned per-atom');
    }
  });
});

test('nested chain: outer=kchain + inner=jointchain mismatch rejected with clear diagnostic', () => {
  // Mixed-marginalize nested chains corrupt semantics under flat
  // splicing (outer=kchain drops everything except the last step,
  // including inner's retain-set; or outer=jointchain exposes
  // inner's kchain-marginalised intermediate). The materialiser
  // rejects with a clear "needs per-atom delegation" message; the
  // delegation runtime is a deferred follow-up. Workaround: spell
  // out the chain explicitly.
  const r = infer(`
theta = elementof(reals)
K0 = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
mu = elementof(reals)
K1 = functionof(Normal(mu = mu, sigma = 0.5), mu = mu)
inner = jointchain(K0, K1)
prior = Normal(0.0, 1.0)
outer = kchain(prior, inner)
`);
  const lifted = orchestrator.liftInlineSubexpressions(r.bindings);
  const built = orchestrator.buildDerivations(lifted);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 12345 });
  const bindings = new Map();
  for (const [name, b] of lifted.entries()) {
    const lb = r.loweredModule.bindings.get(name);
    const merged = Object.assign({}, b);
    if (lb && lb.inferredType) merged.inferredType = lb.inferredType;
    bindings.set(name, merged);
  }
  const ctx: any = {
    derivations: built.derivations,
    bindings:    bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (n: any) => materialiser.materialiseMeasure(n, ctx),
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: 32,
    rootSeed:    12345,
  };
  return materialiser.materialiseMeasure('outer', ctx).then(
    () => { throw new Error('expected rejection — mixed-marginalize nested chain'); },
    (err: any) => {
      assert.match(err.message, /marginalize/i);
      assert.match(err.message, /per-atom delegation/i);
    }
  );
});

test('applied-via-draw: y ~ chain(theta=value) materialises through kernel-first chain', () => {
  // Spec §06 "uniform kernel extension" (line 86-91): applying a
  // kernel binds its inputs and yields a measure. Engine-concepts
  // §19: a kernel-first jointchain/kchain binding applied to its
  // residual inputs materialises as a closed measure. The classifier
  // (derivations.ts classifyAppliedChain) substitutes the bound
  // args into the chain's first step's kernel body and synthesises a
  // closed-first jointchain derivation that matJointchain handles.
  const r = infer(`
theta = elementof(reals)
K0 = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
mu = elementof(reals)
K1 = functionof(Normal(mu = mu, sigma = 0.5), mu = mu)
chain = jointchain(K0, K1)
y ~ chain(theta = 0.5)
`);
  const lifted = orchestrator.liftInlineSubexpressions(r.bindings);
  const built = orchestrator.buildDerivations(lifted);
  // y is an alias to the chain-apply anon (lifted) — both should
  // classify.
  assert.ok(built.derivations.y, 'y should have a derivation');
  // The lifted anon should classify as a jointchain (the synthesised
  // closed-first form).
  const anonNames = Object.keys(built.derivations)
    .filter(n => n.startsWith('__'));
  assert.ok(anonNames.length > 0,
    'expected an anon binding for chain(theta=0.5)');
  assert.equal(built.derivations[anonNames[0]].kind, 'jointchain',
    'anon should classify as a jointchain (closed-first via applied-chain substitution)');
  // Step 0 should be a closed base (kernel: false) — the substituted
  // K0 body. Step 1 should be the original K1.
  const anonDeriv = built.derivations[anonNames[0]];
  assert.equal(anonDeriv.steps[0].role, 'base');
  assert.equal(anonDeriv.steps[0].kernel, false,
    'first step is now a closed measure after substitution');
  assert.equal(anonDeriv.steps[1].ref, 'K1');

  // Materialise + verify shape.
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 12345 });
  const ctx: any = {
    derivations: built.derivations,
    bindings:    lifted,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (n: any) => materialiser.materialiseMeasure(n, ctx),
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: 256,
    rootSeed:    12345,
  };
  return materialiser.materialiseMeasure('y', ctx).then((m: any) => {
    assert.ok(m, 'y materialises');
    assert.ok(m.elems || m.fields,
      'y is a structured measure (positional jointchain ⇒ tuple)');
    const components = m.elems || Object.values(m.fields || {});
    assert.equal(components.length, 2, 'two retained variates from K0 and K1');
    // K0 samples ~ Normal(0.5, 1.0). The empirical mean should be
    // near 0.5 with a tolerance for 256 atoms.
    const k0 = components[0] as { samples: ArrayLike<number> };
    assert.ok(k0.samples && k0.samples.length === 256);
    const mean0 = Array.from(k0.samples).reduce((a: number, b: number) => a + b, 0) / k0.samples.length;
    assert.ok(Math.abs(mean0 - 0.5) < 0.25,
      'K0 sample mean ≈ 0.5 (got ' + mean0 + ')');
    // K1 samples ~ Normal(K0_sample, 0.5). Expected mean ≈ E[K0] ≈ 0.5.
    const k1 = components[1] as { samples: ArrayLike<number> };
    const mean1 = Array.from(k1.samples).reduce((a: number, b: number) => a + b, 0) / k1.samples.length;
    assert.ok(Math.abs(mean1 - 0.5) < 0.3,
      'K1 sample mean ≈ 0.5 (got ' + mean1 + ')');
  });
});

test('applied-via-draw: applied-chain classifier rejects closed-first chain', () => {
  // A closed-first chain has no inputs to bind — `closed_chain(args)`
  // is semantically ill-defined. The applied-chain classifier returns
  // null; the call falls through to no derivation.
  const r = infer(`
prior = Normal(0, 1)
mu = elementof(reals)
K = functionof(Normal(mu = mu, sigma = 0.5), mu = mu)
closed_chain = jointchain(prior, K)
applied = closed_chain(theta = 0.5)
`);
  const lifted = orchestrator.liftInlineSubexpressions(r.bindings);
  const built = orchestrator.buildDerivations(lifted);
  // closed_chain has a derivation (closed-first works); applied does
  // not (no inputs to bind).
  assert.ok(built.derivations.closed_chain);
  assert.ok(!built.derivations.applied,
    'closed_chain(theta=0.5) should NOT classify (no inputs to bind)');
});

test('kernel↔measure collapse: closed-first chain ⇒ measureType (not kernelType[∅])', () => {
  // Pin spec §06 line 86-91 "kernel with empty interface IS a measure"
  // at the engine level: a chain with NO residual inputs surfaces as
  // measureType, not kernelType with empty inputs.
  const r = infer(`
prior = joint(theta = Normal(0, 1))
mu = elementof(reals)
K1 = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)
chain = jointchain(prior, K1)
`);
  const b = r.loweredModule.bindings.get('chain');
  assert.equal(b.inferredType.kind, 'measure',
    'kernel↔measure collapse: empty residual ⇒ measureType (not kernelType[∅])');
  assert.notEqual(b.inferredType.kind, 'kernel');
});
