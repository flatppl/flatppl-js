'use strict';

// Tests for engine/orchestrator.js — building a sample chain from an
// analyzed bindings map.
//
// Coverage:
//   - Single distribution binding → single sample step
//   - draw(distribution-with-literal-params) → sample step on inner dist
//   - Stochastic chain (mu ~ Normal; y ~ Normal(mu, 1)) → ordered steps
//   - Deterministic intermediates (e.g. s = mu + 1) → evaluate step
//   - Discrete leaf flagged correctly
//   - Unsupported cases short-circuit cleanly:
//       * unknown binding name
//       * distribution not in SAMPLEABLE list
//       * reified scope (lawof / functionof)
//       * deterministic call with non-evaluable op
//   - Cycle detection
//   - End-to-end: parse + analyze + buildSampleChain on a real source

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');
const {
  buildSampleChain, buildDerivations, collectSelfRefs, leafSampleIR,
  signatureOf, distributeAxes, inlineForProfile, substituteLocals,
  resolveAxisBaseSet, fourSigmaQuantileRange,
  findMatchingPresets, findMatchingDomains,
  expandMeasureIR, implicitKernelSignature, implicitFunctionSignature,
  canonicalizeImplicitBoundaries,
  _internal: { isEvaluable, classifyForChain },
} = require('../orchestrator');

function chainOf(source, target) {
  const { bindings } = processSource(source);
  return buildSampleChain(target, bindings);
}

// =====================================================================
// Single-binding shapes
// =====================================================================

test('chain: single draw of Normal with literal params', () => {
  const r = chainOf('y = draw(Normal(mu = 0, sigma = 1))', 'y');
  assert.equal(r.unsupported, undefined);
  assert.equal(r.chain.length, 1);
  assert.equal(r.chain[0].name, 'y');
  assert.equal(r.chain[0].kind, 'sample');
  assert.equal(r.chain[0].ir.kind, 'call');
  assert.equal(r.chain[0].ir.op, 'Normal');
  assert.equal(r.discrete, false);
});

test('classifyForChain: both elementof and external skip (boundary inputs)', () => {
  // Per orchestrator's contract, type='input' bindings get no chain
  // step regardless of phase: parameterized-phase elementof is
  // supplied via env at chain-eval time, fixed-phase external /
  // load_data is supplied at module-init time. Either way the caller
  // pre-binds the value, so the chain doesn't compute it.
  //
  // Directly drives classifyForChain so the assertion isn't muddled
  // by downstream chain-promotion logic.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
mu = elementof(reals)
ext = external("data.dat")
`);
  const lifted = liftInlineSubexpressions(bindings);
  const muBinding = lifted.get('mu');
  const extBinding = lifted.get('ext');
  assert.equal(muBinding.phase, 'parameterized');
  assert.equal(extBinding.phase, 'fixed');
  // Both classify to null (no chain step).
  assert.equal(classifyForChain(muBinding, muBinding.ir, lifted), null,
    'elementof must skip the chain step');
  assert.equal(classifyForChain(extBinding, extBinding.ir, lifted), null,
    'external must skip the chain step too — same input-boundary path');
});

test('chain: measure-alias as target → single sample step on resolved IR', () => {
  // theta1_dist is a measure construction (analyzer type='call'). When
  // it's a transitive dep, classifyForChain marks it 'skip'; when it's
  // the target the orchestrator must promote it to a sample step so
  // there's a value to draw.
  const r = chainOf('theta1_dist = Normal(mu = 0, sigma = 1)', 'theta1_dist');
  assert.equal(r.unsupported, undefined);
  assert.equal(r.chain.length, 1);
  assert.equal(r.chain[0].name, 'theta1_dist');
  assert.equal(r.chain[0].kind, 'sample');
  assert.equal(r.chain[0].ir.op, 'Normal');
});

test('chain: draw of measure alias resolves through the alias', () => {
  // Same model as bayesian_inference_3.flatppl. theta1's chain step's
  // IR is the resolved Normal call (not a (ref self theta1_dist) IR).
  const src = `
theta1_dist = Normal(mu = 0, sigma = 1)
theta1      = draw(theta1_dist)
`;
  const r = chainOf(src, 'theta1');
  assert.equal(r.unsupported, undefined);
  // theta1_dist is 'skip' (alias), so the chain has only theta1's step.
  assert.equal(r.chain.length, 1);
  assert.equal(r.chain[0].name, 'theta1');
  assert.equal(r.chain[0].ir.op, 'Normal');
});

test('chain: discrete leaf is flagged', () => {
  const r = chainOf('k = draw(Poisson(rate = 3))', 'k');
  assert.equal(r.unsupported, undefined);
  assert.equal(r.discrete, true);
});

test('chain: literal binding is an evaluate step', () => {
  // Pure-literal numeric binding: lowers to a `lit` IR.
  const r = chainOf('c = 3.14', 'c');
  assert.equal(r.unsupported, undefined);
  assert.equal(r.chain.length, 1);
  assert.equal(r.chain[0].kind, 'evaluate');
  assert.equal(r.chain[0].ir.kind, 'lit');
});

// =====================================================================
// Multi-binding chains
// =====================================================================

test('chain: stochastic dependency (mu → y) topologically ordered', () => {
  const src = `
mu = draw(Normal(mu = 0, sigma = 1))
y  = draw(Normal(mu = mu, sigma = 1))
`;
  const r = chainOf(src, 'y');
  assert.equal(r.unsupported, undefined);
  assert.equal(r.chain.length, 2);
  assert.equal(r.chain[0].name, 'mu');
  assert.equal(r.chain[1].name, 'y');
  assert.equal(r.chain[0].kind, 'sample');
  assert.equal(r.chain[1].kind, 'sample');
});

test('chain: deterministic intermediate gets an evaluate step', () => {
  const src = `
mu = draw(Normal(mu = 0, sigma = 1))
s  = mu + 1
y  = draw(Normal(mu = s, sigma = 1))
`;
  const r = chainOf(src, 'y');
  assert.equal(r.unsupported, undefined);
  assert.equal(r.chain.length, 3);
  const names = r.chain.map(s => s.name);
  assert.deepEqual(names, ['mu', 's', 'y']);
  assert.equal(r.chain[0].kind, 'sample');
  assert.equal(r.chain[1].kind, 'evaluate');
  assert.equal(r.chain[2].kind, 'sample');
});

test('chain: target is the only step when it has no deps', () => {
  // No dependencies → chain has just the target.
  const src = `
unused = draw(Normal(mu = 0, sigma = 1))
y      = draw(Exponential(rate = 2))
`;
  const r = chainOf(src, 'y');
  assert.equal(r.chain.length, 1);
  assert.equal(r.chain[0].name, 'y');
});

test('chain: shared dep visited once', () => {
  const src = `
mu = draw(Normal(mu = 0, sigma = 1))
a  = draw(Normal(mu = mu, sigma = 1))
b  = draw(Normal(mu = mu, sigma = 2))
c  = a + b
`;
  const r = chainOf(src, 'c');
  // Even though both a and b depend on mu, mu must appear exactly once.
  const muCount = r.chain.filter(s => s.name === 'mu').length;
  assert.equal(muCount, 1);
  // c depends on both a and b — both should precede c.
  const idx = name => r.chain.findIndex(s => s.name === name);
  assert.ok(idx('mu') < idx('a'));
  assert.ok(idx('mu') < idx('b'));
  assert.ok(idx('a') < idx('c'));
  assert.ok(idx('b') < idx('c'));
});

// =====================================================================
// Unsupported cases
// =====================================================================

test('chain: unknown target → unsupported', () => {
  const r = chainOf('x = 1', 'nonexistent');
  assert.ok(r.unsupported);
  assert.match(r.unsupported.reason, /unknown binding/);
});

test('chain: distribution not in SAMPLEABLE list → unsupported', () => {
  const r = chainOf('x = draw(Categorical(p = [0.5, 0.5]))', 'x');
  assert.ok(r.unsupported);
});

test('chain: lawof binding → unsupported', () => {
  const r = chainOf('m = lawof(draw(Normal(mu = 0, sigma = 1)))', 'm');
  assert.ok(r.unsupported);
});

test('chain: deterministic call with non-evaluable op → unsupported', () => {
  // Custom user function call — sampler can't evaluate it.
  const src = `
f = functionof(_x_ + 1, _x_ = _y_)
y = f(2)
`;
  const r = chainOf(src, 'y');
  assert.ok(r.unsupported);
});

test('chain: dist with non-literal kwarg only succeeds when the ref is a chained binding', () => {
  // Direct ref to a known binding works — the chain picks it up.
  const ok = chainOf(`
mu = draw(Normal(mu = 0, sigma = 1))
y  = draw(Normal(mu = mu, sigma = 1))
`, 'y');
  assert.equal(ok.unsupported, undefined);

  // Ref to something we can't sample (a lawof binding) propagates as
  // unsupported through the recursive walker.
  const bad = chainOf(`
m = lawof(draw(Normal(mu = 0, sigma = 1)))
y = draw(Normal(mu = m, sigma = 1))
`, 'y');
  assert.ok(bad.unsupported);
});

// =====================================================================
// isEvaluable predicate
// =====================================================================

test('isEvaluable: literals, refs, consts, evaluable ops', () => {
  assert.equal(isEvaluable({ kind: 'lit', value: 1 }), true);
  assert.equal(isEvaluable({ kind: 'lit', value: true }), true);
  assert.equal(isEvaluable({ kind: 'lit', value: 'hi' }), false);
  assert.equal(isEvaluable({ kind: 'const', name: 'pi' }), true);
  assert.equal(isEvaluable({ kind: 'ref', ns: 'self', name: 'x' }), true);
  assert.equal(isEvaluable({
    kind: 'call', op: 'add',
    args: [{ kind: 'lit', value: 1 }, { kind: 'lit', value: 2 }],
  }), true);
  assert.equal(isEvaluable({
    kind: 'call', op: 'log',  // log is now evaluable (added with abs/exp/sqrt/...)
    args: [{ kind: 'lit', value: 1 }],
  }), true);
  assert.equal(isEvaluable({
    kind: 'call', op: 'unknown_op_xyz',
    args: [{ kind: 'lit', value: 1 }],
  }), false);
  assert.equal(isEvaluable({ kind: 'call', op: 'add', args: [{ kind: 'lit', value: 1 }, { kind: 'hole' }] }), false);
});

// =====================================================================
// Cycle detection (defensive)
// =====================================================================

// =====================================================================
// buildDerivations — main-thread sample-cache key/derivation map
// =====================================================================

function derivationsOf(source) {
  const { bindings } = processSource(source);
  return buildDerivations(bindings);
}

test('derivations: variate aliases its underlying measure', () => {
  // theta1 = draw(theta1_dist) means theta1's samples ARE theta1_dist's
  // samples — so the derivation must be an alias, not a fresh sample step.
  // The main-thread cache then gives both names the same Float64Array
  // when materialised.
  const { derivations, discrete } = derivationsOf(`
theta1_dist = Normal(mu = 0, sigma = 1)
theta1      = draw(theta1_dist)
`);
  assert.deepEqual(derivations.theta1, { kind: 'alias', from: 'theta1_dist' });
  assert.equal(derivations.theta1_dist.kind, 'sample');
  assert.equal(derivations.theta1_dist.distIR.op, 'Normal');
  assert.equal(discrete.theta1, false);
  assert.equal(discrete.theta1_dist, false);
});

test('derivations: lawof(<ref>) aliases the ref', () => {
  const { derivations } = derivationsOf(`
y = draw(Normal(mu = 0, sigma = 1))
m = lawof(y)
`);
  assert.deepEqual(derivations.m, { kind: 'alias', from: 'y' });
});

test('derivations: inline draw(Dist(...)) lifts the dist to a synthetic anon, y aliases it', () => {
  // After the lift pass, every measure-arg position is a bare ref.
  // Inline `Normal(0, 1)` in draw's measure slot becomes a synthetic
  // anonymous binding (with a 'sample' derivation), and y aliases to it.
  // Equivalent computation as the previous shape; uniform classifier.
  const { derivations } = derivationsOf(`
y = draw(Normal(mu = 0, sigma = 1))
`);
  assert.equal(derivations.y.kind, 'alias');
  const anon = derivations.y.from;
  assert.match(anon, /^__anon\d+$/);
  assert.equal(derivations[anon].kind, 'sample');
  assert.equal(derivations[anon].distIR.op, 'Normal');
});

test('derivations: deterministic arithmetic becomes evaluate', () => {
  const { derivations } = derivationsOf(`
mu = draw(Normal(mu = 0, sigma = 1))
s  = mu + 1
`);
  assert.equal(derivations.s.kind, 'evaluate');
  // The evaluate IR is the lowered RHS: (call add (ref mu) (lit 1))
  assert.equal(derivations.s.ir.op, 'add');
});

// --- weighted / logweighted / normalize derivations ---

test('derivations: weighted(<lit>, <ref>) → kind=weighted with precomputed logShift', () => {
  const { derivations } = derivationsOf(`
m = Normal(mu = 0, sigma = 1)
mw = weighted(2.0, m)
`);
  assert.equal(derivations.mw.kind, 'weighted');
  assert.equal(derivations.mw.from, 'm');
  assert.ok(Math.abs(derivations.mw.logShift - Math.log(2)) < 1e-12);
});

test('derivations: weighted(<binding-of-constant>, <ref>) resolves through', () => {
  // Weight = a binding that itself reduces to a literal.
  const { derivations } = derivationsOf(`
c = 0.5
m = Normal(mu = 0, sigma = 1)
mw = weighted(c, m)
`);
  assert.equal(derivations.mw.kind, 'weighted');
  assert.ok(Math.abs(derivations.mw.logShift - Math.log(0.5)) < 1e-12);
});

test('derivations: weighted with non-positive scalar is unsupported', () => {
  // weight must be > 0 for a valid measure reweighting.
  const { derivations } = derivationsOf(`
m = Normal(mu = 0, sigma = 1)
mw = weighted(0.0, m)
`);
  assert.ok(!('mw' in derivations));
});

test('derivations: logweighted(<lit>, <ref>) → kind=weighted with logShift = lit', () => {
  const { derivations } = derivationsOf(`
m = Normal(mu = 0, sigma = 1)
mw = logweighted(-3.5, m)
`);
  assert.equal(derivations.mw.kind, 'weighted');
  assert.equal(derivations.mw.from, 'm');
  assert.ok(Math.abs(derivations.mw.logShift - (-3.5)) < 1e-12);
});

test('derivations: normalize(<ref>) → kind=normalize', () => {
  const { derivations } = derivationsOf(`
m = Normal(mu = 0, sigma = 1)
mn = normalize(m)
`);
  assert.equal(derivations.mn.kind, 'normalize');
  assert.equal(derivations.mn.from, 'm');
});

test('derivations: normalize(weighted(c, m)) chains both', () => {
  // Common pattern: normalize(weighted(...)) to renormalise an
  // arbitrarily-scaled measure back onto the probability scale.
  const { derivations } = derivationsOf(`
m = Normal(mu = 0, sigma = 1)
mw = weighted(2.0, m)
mn = normalize(mw)
`);
  assert.equal(derivations.mw.kind, 'weighted');
  assert.equal(derivations.mn.kind, 'normalize');
  assert.equal(derivations.mn.from, 'mw');
});

// --- superpose derivations ---

test('derivations: superpose(<ref>, <ref>) → kind=superpose with parent names', () => {
  const { derivations } = derivationsOf(`
m1 = Normal(mu = 0, sigma = 1)
m2 = Normal(mu = 5, sigma = 1)
ms = superpose(m1, m2)
`);
  assert.equal(derivations.ms.kind, 'superpose');
  assert.deepEqual(derivations.ms.fromNames, ['m1', 'm2']);
});

test('derivations: superpose with 3+ components keeps the order', () => {
  const { derivations } = derivationsOf(`
m1 = Normal(mu = 0, sigma = 1)
m2 = Normal(mu = 1, sigma = 1)
m3 = Normal(mu = 2, sigma = 1)
ms = superpose(m1, m2, m3)
`);
  assert.deepEqual(derivations.ms.fromNames, ['m1', 'm2', 'm3']);
});

test('derivations: superpose with inline measure lifts the inline arg to a synthetic anon', () => {
  // The lift pass makes every measure-arg position a bare ref before
  // classification, so inline measure components are now first-class.
  // superpose(Normal(...), m) gains a synthetic anon for the inline
  // Normal and `ms.fromNames` references it alongside `m`.
  const { derivations } = derivationsOf(`
m = Normal(mu = 0, sigma = 1)
ms = superpose(Normal(mu = 1, sigma = 1), m)
`);
  assert.equal(derivations.ms.kind, 'superpose');
  assert.equal(derivations.ms.fromNames.length, 2);
  assert.match(derivations.ms.fromNames[0], /^__anon\d+$/);
  assert.equal(derivations.ms.fromNames[1], 'm');
  assert.equal(derivations[derivations.ms.fromNames[0]].kind, 'sample');
});

test('derivations: superpose with a non-derivable component cascades to unsupported', () => {
  // chain (Markov-chain composition / marginalization over the
  // intermediate measure) isn't classified as a derivation — it's
  // out of scope for the visualizer. The dependent superpose
  // drops out via the cascade-prune pass.
  const { derivations } = derivationsOf(`
m = Normal(mu = 0, sigma = 1)
unsupp = chain(m, m)
ms = superpose(m, unsupp)
`);
  assert.ok(!('ms' in derivations));
});

test('derivations: superpose of all-discrete components → discrete', () => {
  const { discrete } = derivationsOf(`
m1 = Poisson(rate = 2)
m2 = Poisson(rate = 5)
ms = superpose(m1, m2)
`);
  assert.equal(discrete.ms, true);
});

test('derivations: superpose with mixed continuous/discrete → continuous', () => {
  const { discrete } = derivationsOf(`
m1 = Poisson(rate = 2)
m2 = Normal(mu = 0, sigma = 1)
ms = superpose(m1, m2)
`);
  assert.equal(discrete.ms, false);
});

test('derivations: weighted with function-of-variate weight is unsupported (for now)', () => {
  // weighted(fn(_*2), m) — the weight depends on the base's variate.
  // Future work; current orchestrator only handles constant weights.
  const { derivations } = derivationsOf(`
m = Normal(mu = 0, sigma = 1)
mw = weighted(fn(_ * 2), m)
`);
  assert.ok(!('mw' in derivations));
});

test('derivations: numeric array literal becomes an array derivation', () => {
  const { derivations } = derivationsOf('observed = [1.2, 3.4, 5.1, 2.8]');
  assert.equal(derivations.observed.kind, 'array');
  assert.deepEqual(derivations.observed.values, [1.2, 3.4, 5.1, 2.8]);
});

test('derivations: array with non-literal entry is unsupported', () => {
  // mixed entries (a ref alongside a literal) — out of scope today.
  const { derivations } = derivationsOf(`
mu = draw(Normal(mu = 0, sigma = 1))
xs = [mu, 1.0]
`);
  assert.ok(!('xs' in derivations));
});

test('derivations: discrete leaf flagged via alias chain', () => {
  const { discrete } = derivationsOf(`
k_dist = Poisson(rate = 3)
k      = draw(k_dist)
`);
  assert.equal(discrete.k, true);
  assert.equal(discrete.k_dist, true);
});

test('derivations: unsupported binding is omitted, dependents drop too', () => {
  // unsupp uses `chain`, which isn't classified. theta1 still has a
  // derivation (it doesn't reference unsupp). Anything that refs
  // unsupp drops via the cascade-prune pass.
  const { derivations } = derivationsOf(`
theta1 = draw(Normal(mu = 0, sigma = 1))
m      = Normal(mu = 0, sigma = 1)
unsupp = chain(m, m)
ghost  = unsupp + 1
`);
  assert.ok(derivations.theta1);
  assert.ok(!('unsupp' in derivations));
  assert.ok(!('ghost' in derivations));
});

test('derivations: bayesian_inference_3 fixture covers the expected nodes', () => {
  // Smoke check on the user's example file. theta1_dist / theta2_dist
  // are sample steps; theta1 / theta2 are aliases; prior, prior2,
  // forward_kernel, etc. are unsupported (lawof / functionof / module).
  const { derivations } = derivationsOf(`
theta1_dist = Normal(mu = 0.0, sigma = 1.0)
theta2_dist = Exponential(rate = 1.0)
theta1 = draw(theta1_dist)
theta2 = draw(theta2_dist)
`);
  assert.equal(derivations.theta1_dist.kind, 'sample');
  assert.equal(derivations.theta2_dist.kind, 'sample');
  assert.deepEqual(derivations.theta1, { kind: 'alias', from: 'theta1_dist' });
  assert.deepEqual(derivations.theta2, { kind: 'alias', from: 'theta2_dist' });
});

// =====================================================================
// relabel — AST-level rewrite to record(...)
// =====================================================================
//
// inlineRelabel handles five shapes (spec §sec:design lines 482-507):
//   1. inline ArrayLiteral
//   2. inline record(...) call (positional rename)
//   3. Identifier → array binding (via xs[i] indexing)
//   4. Identifier → record binding (via xs.<old_field> field access)
//   5. single-name wrap of an arbitrary scalar
// Anything else, or a length mismatch, must bail and leave the binding
// unsupported (so an explicit error surfaces rather than silent garbage).

test('relabel: inline array literal becomes a record derivation', () => {
  const { derivations } = derivationsOf(`
r = relabel([1.2, 3.4], ["x", "y"])
`);
  assert.ok(derivations.r, 'r should be derivable');
  assert.equal(derivations.r.kind, 'record');
  assert.deepEqual(Object.keys(derivations.r.fields), ['x', 'y']);
});

test('relabel: inline record(...) renames positionally', () => {
  const { derivations } = derivationsOf(`
r = relabel(record(a = 1.2, b = 3.4), ["x", "y"])
`);
  assert.ok(derivations.r, 'r should be derivable');
  assert.equal(derivations.r.kind, 'record');
  assert.deepEqual(Object.keys(derivations.r.fields), ['x', 'y']);
});

test('relabel: identifier → array binding reuses element exprs', () => {
  const { derivations } = derivationsOf(`
xs = [1.2, 3.4]
r  = relabel(xs, ["x", "y"])
`);
  assert.ok(derivations.r, 'r should be derivable');
  assert.equal(derivations.r.kind, 'record');
  assert.deepEqual(Object.keys(derivations.r.fields), ['x', 'y']);
});

test('relabel: identifier → record binding renames by field', () => {
  const { derivations } = derivationsOf(`
src = record(a = 1.2, b = 3.4)
r   = relabel(src, ["x", "y"])
`);
  assert.ok(derivations.r, 'r should be derivable');
  assert.equal(derivations.r.kind, 'record');
  assert.deepEqual(Object.keys(derivations.r.fields), ['x', 'y']);
});

test('relabel: single-name wrap of a scalar identifier', () => {
  const { derivations } = derivationsOf(`
mu = 1.5
r  = relabel(mu, ["x"])
`);
  assert.ok(derivations.r, 'r should be derivable');
  assert.equal(derivations.r.kind, 'record');
  assert.deepEqual(Object.keys(derivations.r.fields), ['x']);
});

test('relabel: mismatched names/values bails (binding unsupported)', () => {
  // Three values, two names → no static rewrite possible. The
  // binding falls through to the generic classifier and stays out
  // of the derivation map.
  const { derivations } = derivationsOf(`
r = relabel([1, 2, 3], ["x", "y"])
`);
  assert.ok(!('r' in derivations));
});

// =====================================================================
// logdensityof / densityof — scalar density-evaluation bindings
// =====================================================================
//
// `r = logdensityof(M, x)` classifies as kind='logdensityof' with the
// measure name and the resolved obs value attached. The materialiser
// computes per-prior-atom log-densities via traceeval.walk —
// implemented in the viewer; here we only check the orchestrator
// correctly identifies the binding and primes the cascade.
//
// `densityof(M, x)` is rewritten at AST time to
// `exp(logdensityof(M, x))`; the resulting binding is an evaluate
// node whose IR contains a logdensityof call (no derivation kind of
// its own — it inherits the measure's cascade through that ref).

test('logdensityof: scalar measure with literal obs classifies', () => {
  const { derivations } = derivationsOf(`
y_dist = Normal(mu = 0.0, sigma = 1.0)
lp     = logdensityof(y_dist, 1.5)
`);
  assert.ok(derivations.lp, 'lp should be derivable');
  assert.equal(derivations.lp.kind, 'logdensityof');
  assert.equal(derivations.lp.measureName, 'y_dist');
  assert.equal(derivations.lp.obsValue, 1.5);
});

test('logdensityof: cascade-prunes when measure isn\'t derivable', () => {
  // y_kernel uses lawof, which the orchestrator doesn't derive (it's
  // a reified scope). lp must therefore not appear.
  const { derivations } = derivationsOf(`
m  = Normal(mu = 0.0, sigma = 1.0)
k  = kernelof(m)
lp = logdensityof(k, 1.5)
`);
  assert.ok(!('lp' in derivations));
});

test('densityof: rewritten to exp(logdensityof(...)) at AST time', () => {
  // The binding is evaluate-kind (its IR is exp(logdensityof(...)))
  // and its derivation drops if logdensityof isn't classifiable.
  // When everything's in place, the IR contains an exp wrapping a
  // logdensityof call.
  const { derivations } = derivationsOf(`
y_dist = Normal(mu = 0.0, sigma = 1.0)
d      = densityof(y_dist, 1.5)
`);
  assert.ok(derivations.d, 'd should be derivable');
  // The rewrite produces exp(logdensityof(...)) — ld-classify above
  // turns the inner call into a logdensityof derivation, the outer
  // binding becomes evaluate. Since logdensityof is a value-typed
  // op, classifyDerivation treats the outer binding as value too;
  // we don't pin the exact downstream kind here, only that it
  // didn't fall through unsupported.
});

// =====================================================================
// fchain — applied function composition unrolls to nested calls
// =====================================================================
//
// Per spec §sec:design line 526-532, fchain(f1, …, fN)(x) ≡
// fN(… f2(f1(x))). The orchestrator unrolls applied fchains at AST
// time so the inliner can substitute each component's body
// uniformly. Standalone (unapplied) fchain bindings stay unsupported
// — they're function values, not measures.

test('fchain: pipeline binding applied positionally unrolls to nested calls', () => {
  const { derivations } = derivationsOf(`
y_dist = Normal(mu = 0.0, sigma = 1.0)
y      = draw(y_dist)
f1     = fn(_ + 1.0)
f2     = fn(_ * 2.0)
pipe   = fchain(f1, f2)
z      = pipe(y)
`);
  assert.ok(derivations.z, 'z should be derivable');
  assert.equal(derivations.z.kind, 'evaluate');
  // The unrolled body computes (y + 1) * 2. We don't pin the IR
  // shape exactly (fn-substitution may go through anons), only that
  // the binding got classified.
});

test('fchain: inline fchain(...)(x) form works', () => {
  const { derivations } = derivationsOf(`
y_dist = Normal(mu = 0.0, sigma = 1.0)
y      = draw(y_dist)
f1     = fn(_ + 1.0)
f2     = fn(_ * 2.0)
z      = fchain(f1, f2)(y)
`);
  assert.ok(derivations.z, 'z should be derivable');
  assert.equal(derivations.z.kind, 'evaluate');
});

test('fchain: single-component fchain is identity composition', () => {
  const { derivations } = derivationsOf(`
y_dist = Normal(mu = 0.0, sigma = 1.0)
y      = draw(y_dist)
f1     = fn(_ + 1.0)
z      = fchain(f1)(y)
`);
  assert.ok(derivations.z, 'z should be derivable');
  assert.equal(derivations.z.kind, 'evaluate');
});

test('fchain: standalone (unapplied) fchain binding is not a measure derivation', () => {
  // pipe is a function value, not a variate. It should not appear
  // in the derivations map (no kind for "function value").
  const { derivations } = derivationsOf(`
f1   = fn(_ + 1.0)
f2   = fn(_ * 2.0)
pipe = fchain(f1, f2)
`);
  assert.ok(!('pipe' in derivations));
});

// =====================================================================
// canonicalizeImplicitBoundaries — AST rewrite that turns no-kwargs
// functionof / kernelof into explicit-kwargs form, so a single
// canonical IR shape feeds every downstream consumer (the lowerer,
// signatureOf, inlineOnce).
// =====================================================================

test('canonicalize: no-kwargs functionof gets implicit kwargs from elementof leaves', () => {
  const { bindings } = processSource(`
mu = elementof(reals)
mu2 = mu^2
f = functionof(mu2)
`);
  const rewritten = canonicalizeImplicitBoundaries(bindings);
  const fNode = rewritten.get('f').node.value;
  // After canonicalize: functionof(mu2, mu = mu) — args[0] is the
  // body, args[1+] are the synthesized KeywordArgs.
  assert.equal(fNode.args.length, 2);
  assert.equal(fNode.args[0].type, 'Identifier');
  assert.equal(fNode.args[0].name, 'mu2');
  assert.equal(fNode.args[1].type, 'KeywordArg');
  assert.equal(fNode.args[1].name, 'mu');
  assert.equal(fNode.args[1].value.name, 'mu');
  // A tag flags the rewrite for downstream tooling.
  assert.deepEqual(rewritten.get('f').implicitBoundaries, ['mu']);
});

test('canonicalize: explicit-kwargs functionof is left alone', () => {
  // The user has declared the signature; canonicalize must not add
  // extra inputs even if more elementof leaves exist further upstream.
  const { bindings } = processSource(`
a = elementof(reals)
b = elementof(reals)
combined = a * b
f = functionof(combined, a = a)
`);
  const rewritten = canonicalizeImplicitBoundaries(bindings);
  const fNode = rewritten.get('f').node.value;
  // One explicit kwarg, NOT two (no extra promotion of b).
  const kwargs = fNode.args.slice(1).filter((a) => a.type === 'KeywordArg');
  assert.equal(kwargs.length, 1);
  assert.equal(kwargs[0].name, 'a');
  // No implicitBoundaries tag — the rewrite didn't fire.
  assert.equal(rewritten.get('f').implicitBoundaries, undefined);
});

test('canonicalize: kernelof handled the same as functionof', () => {
  const { bindings } = processSource(`
mu = elementof(reals)
y = draw(Normal(mu = mu, sigma = 1))
K = kernelof(y)
`);
  const rewritten = canonicalizeImplicitBoundaries(bindings);
  const kNode = rewritten.get('K').node.value;
  // K's body refs y → mu, so mu is the implicit boundary.
  const kwargs = kNode.args.slice(1).filter((a) => a.type === 'KeywordArg');
  assert.equal(kwargs.length, 1);
  assert.equal(kwargs[0].name, 'mu');
});

test('canonicalize: body with no elementof leaves stays unchanged', () => {
  // f = functionof(2.0) — fully closed. No leaves → no rewrite.
  const { bindings } = processSource(`
f = functionof(2.0)
`);
  const rewritten = canonicalizeImplicitBoundaries(bindings);
  const fNode = rewritten.get('f').node.value;
  assert.equal(fNode.args.length, 1);
  assert.equal(rewritten.get('f').implicitBoundaries, undefined);
});

test('canonicalize: external() / load_data() (fixed phase) are NOT promoted', () => {
  // Symmetric with the other helpers' phase filter: only parametric
  // leaves become inputs; fixed-phase boundary bindings are closed over.
  const { bindings } = processSource(`
mu = elementof(reals)
ext = external("data.dat")
combo = mu * ext
f = functionof(combo)
`);
  const rewritten = canonicalizeImplicitBoundaries(bindings);
  const fNode = rewritten.get('f').node.value;
  const kwargs = fNode.args.slice(1).filter((a) => a.type === 'KeywordArg');
  // Only `mu` — `ext` is fixed-phase, must not appear.
  assert.equal(kwargs.length, 1);
  assert.equal(kwargs[0].name, 'mu');
});

test('canonicalize: fn(...) is left alone (uses placeholders, not refs)', () => {
  // fn lowers to functionof with placeholder params on its own;
  // canonicalize's auto-promote rule applies only to functionof /
  // kernelof at the AST level, before lowering.
  const { bindings } = processSource(`
f = fn(_ + 1.0)
`);
  const rewritten = canonicalizeImplicitBoundaries(bindings);
  const fNode = rewritten.get('f').node.value;
  // fn(...) AST shape is preserved verbatim — no rewrite.
  assert.equal(fNode.callee.name, 'fn');
  assert.equal(rewritten.get('f').implicitBoundaries, undefined);
});

test('canonicalize: original bindings are not mutated', () => {
  // The pass returns a fresh Map and deep-clones modified nodes;
  // the caller's bindings stay pristine for editor diagnostics.
  const { bindings } = processSource(`
mu = elementof(reals)
f = functionof(mu^2)
`);
  const before = bindings.get('f').node.value;
  const beforeArgsCount = before.args.length;
  canonicalizeImplicitBoundaries(bindings);
  assert.equal(bindings.get('f').node.value.args.length, beforeArgsCount,
    'canonicalize must not mutate the original bindings AST');
});

// =====================================================================
// inlineUserCall: implicit-boundary functionof
//
// Per spec §04 sec:functionof, a functionof declared with no
// boundary kwargs has its parametric (elementof) ancestors as
// implicit inputs. signatureOf already auto-promotes those leaves;
// these tests cover the call-site side — inlineOnce must substitute
// positional args for the same implicit leaves so the resulting IR
// has no unbound elementof refs.
// =====================================================================

test('inline: f(arg) with no-kwargs functionof binds the elementof leaf', () => {
  // The motivating shape from minimal.flatppl:
  //   f_sqrt = functionof(b)
  //   b      = a^0.5
  //   a      = elementof(nonnegreals)
  //   sigma  = f_sqrt(sigma2)
  // After inlining, sigma's chain must compute sigma2^0.5 — NOT
  // contain a lingering ref to the elementof leaf `a`.
  const { bindings } = derivationsOf(`
a = elementof(nonnegreals)
b = a^0.5
f_sqrt = functionof(b)
sigma2 ~ Exponential(rate = 1.0)
sigma = f_sqrt(sigma2)
`);
  // Sigma's lifted IR is an alias chain leading to a pow(sigma2, 0.5)
  // anon. Walk through to verify the substitution happened.
  const sigma = bindings.get('sigma');
  // sigma.ir is a self-ref to the lifted pow call.
  assert.equal(sigma.ir.kind, 'ref');
  const powAnon = bindings.get(sigma.ir.name);
  assert.equal(powAnon.ir.op, 'pow');
  // The first arg should now reference `sigma2`, NOT `a` (the
  // unsubstituted elementof leaf).
  const firstArg = powAnon.ir.args[0];
  assert.equal(firstArg.kind, 'ref');
  assert.equal(firstArg.name, 'sigma2',
    `expected first arg of pow to ref 'sigma2', got '${firstArg.name}' — ` +
    'implicit elementof boundary substitution regressed');
});

test('inline: implicit boundary works with multiple elementof leaves', () => {
  // Two parametric leaves reached through a chained body. Implicit
  // boundary order = BFS visit order of self-refs; both must be
  // substituted from positional call args in matching order.
  const { bindings } = derivationsOf(`
a = elementof(reals)
b = elementof(reals)
s = a + b
f = functionof(s)
x = 3.0
y = 4.0
z = f(x, y)
`);
  // z's chain should compute x + y (3 + 4 = 7 if evaluated).
  // Structurally: the lifted form should reference x and y, not a / b.
  const z = bindings.get('z');
  assert.equal(z.ir.kind, 'ref');
  const addAnon = bindings.get(z.ir.name);
  assert.equal(addAnon.ir.op, 'add');
  const names = addAnon.ir.args.map((arg) => arg.kind === 'ref' ? arg.name : null);
  // Both elementof leaves were substituted — no lingering 'a' or 'b'.
  assert.ok(!names.includes('a') && !names.includes('b'),
    `add args still reference unsubstituted elementof leaves: ${names.join(', ')}`);
  // x and y appear (in some order, depending on BFS).
  assert.ok(names.includes('x') && names.includes('y'),
    `add args should reference x and y; got: ${names.join(', ')}`);
});

test('inline: explicit-kwargs functionof still works unchanged', () => {
  // Regression guard: the implicit-boundary path runs only when
  // surfaceOrder is empty. Functions with explicit boundary kwargs
  // must continue to substitute through the original code path.
  const { bindings } = derivationsOf(`
c = 2.0
f = functionof(c * _par_, par = _par_)
g = f(5)
`);
  // g should inline to c * 5.
  const g = bindings.get('g');
  assert.equal(g.ir.kind, 'call');
  assert.equal(g.ir.op, 'mul');
  // arg[1] is the substituted literal 5.
  const argLit = g.ir.args.find((a) => a.kind === 'lit');
  assert.ok(argLit, 'expected a literal arg after substitution');
  assert.equal(argLit.value, 5);
});

test('inline: implicit-boundary call chained through a deeper body', () => {
  // The elementof leaf is two hops back through evaluable
  // intermediates. The closure walk + substitution must reach
  // through inner = 2.5 + 0.3 * mu, with mu replaced by the call arg.
  const { bindings } = derivationsOf(`
mu = elementof(reals)
inner = 2.5 + 0.3 * mu
outer = inner * inner
f = functionof(outer)
v = 10.0
result = f(v)
`);
  // We don't pin the exact tree (closure-naming can shuffle anons),
  // but we DO assert there are no lingering refs to `mu` in any
  // of result's transitive closure anons — that would mean the
  // substitution missed a path.
  function walkRefs(ir, out) {
    if (!ir || typeof ir !== 'object') return;
    if (ir.kind === 'ref' && ir.ns === 'self') out.add(ir.name);
    if (ir.args) for (const a of ir.args) walkRefs(a, out);
    if (ir.kwargs) for (const k in ir.kwargs) walkRefs(ir.kwargs[k], out);
  }
  const seen = new Set();
  const stack = ['result'];
  while (stack.length > 0) {
    const n = stack.pop();
    if (seen.has(n)) continue;
    seen.add(n);
    const b = bindings.get(n);
    if (!b || !b.ir) continue;
    const refs = new Set();
    walkRefs(b.ir, refs);
    for (const r of refs) {
      // Implicit substitution should have replaced `mu` everywhere
      // in the result's closure. `v` is the substitute.
      assert.notEqual(r, 'mu',
        `binding '${n}' still references unsubstituted elementof 'mu'`);
      stack.push(r);
    }
  }
});

test('inline: implicit boundary excludes external() (fixed phase)', () => {
  // external(...) is fixed-phase and per spec is closed over —
  // not promoted as an implicit input. The call positional arg
  // therefore binds to the elementof leaf only; the external
  // ref stays in the synthesized body.
  const { bindings } = derivationsOf(`
a = elementof(reals)
ext = external("data.dat")
combo = a * ext
f = functionof(combo)
x = 7.0
y = f(x)
`);
  // Walk y's closure: every ref should be either a substituted leaf
  // (x), the external ref (ext), or an internal anon. NO refs to `a`.
  function walkRefs(ir, out) {
    if (!ir || typeof ir !== 'object') return;
    if (ir.kind === 'ref' && ir.ns === 'self') out.add(ir.name);
    if (ir.args) for (const a of ir.args) walkRefs(a, out);
    if (ir.kwargs) for (const k in ir.kwargs) walkRefs(ir.kwargs[k], out);
  }
  const seen = new Set();
  const stack = ['y'];
  let sawExt = false, sawX = false;
  while (stack.length > 0) {
    const n = stack.pop();
    if (seen.has(n)) continue;
    seen.add(n);
    const b = bindings.get(n);
    if (!b || !b.ir) continue;
    const refs = new Set();
    walkRefs(b.ir, refs);
    for (const r of refs) {
      assert.notEqual(r, 'a',
        `binding '${n}' still references the elementof leaf 'a' — ` +
        'implicit substitution missed it');
      if (r === 'ext') sawExt = true;
      if (r === 'x')   sawX = true;
      stack.push(r);
    }
  }
  assert.ok(sawX, 'expected the substituted call arg "x" to appear');
  assert.ok(sawExt, 'expected the closed-over "ext" ref to appear unchanged');
});

// =====================================================================
// collectSelfRefs / leafSampleIR
// =====================================================================

test('collectSelfRefs: nested kwargs yield the full ref set', () => {
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: {
      mu: { kind: 'call', op: 'add', args: [
        { kind: 'ref', ns: 'self', name: 'a' },
        { kind: 'ref', ns: 'self', name: 'b' },
      ]},
      sigma: { kind: 'lit', value: 1 },
    },
  };
  const refs = collectSelfRefs(ir);
  assert.ok(refs.has('a'));
  assert.ok(refs.has('b'));
  assert.equal(refs.size, 2);
});

test('leafSampleIR: walks alias chain to the underlying sample IR', () => {
  const derivations = {
    theta1_dist: { kind: 'sample', distIR: { op: 'Normal' } },
    theta1:      { kind: 'alias',  from: 'theta1_dist' },
    theta1_alt:  { kind: 'alias',  from: 'theta1' },
  };
  assert.equal(leafSampleIR('theta1_alt', derivations).op, 'Normal');
});

test('leafSampleIR: returns null for evaluate-only chains', () => {
  const derivations = {
    s: { kind: 'evaluate', ir: { kind: 'call', op: 'add' } },
  };
  assert.equal(leafSampleIR('s', derivations), null);
});

test('chain: self-reference detected (defensive — analyzer normally rejects)', () => {
  // Manually construct a fake binding map with a cycle.
  const bindings = new Map();
  const stub = (name, deps) => ({
    name, deps: new Set(deps), type: 'call',
    node: { type: 'AssignStatement', value: { type: 'Identifier', name: 'x', loc: { start: { line: 0, col: 0 }, end: { line: 0, col: 1 } } } },
  });
  bindings.set('a', stub('a', ['b']));
  bindings.set('b', stub('b', ['a']));
  const r = buildSampleChain('a', bindings);
  assert.ok(r.unsupported);
});

// =====================================================================
// signatureOf — callable input/output signature for the profile plot
// =====================================================================
//
// Pulls binding.inferredType (typeinfer-side) and binding.ir.paramSources
// (lower.js-side, F1) together into a single shape the viewer can
// consume. Covers fn / functionof / kernelof / likelihoodof.

function sigOf(source, name) {
  // signatureOf reads binding.ir, which is populated by
  // liftInlineSubexpressions (the lift pass that buildDerivations
  // runs internally). Mimic that here so tests stay decoupled from
  // the full derivation builder.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(source);
  return signatureOf(name, liftInlineSubexpressions(bindings));
}

test('signatureOf: fn(_) returns one placeholder input, scalar real output', () => {
  // fn lowers to functionof with a single _arg1_ placeholder param.
  // The body is `_ + 1` — value-typed, so kind='function'.
  const sig = sigOf('f = fn(_ + 1)', 'f');
  assert.equal(sig.kind, 'function');
  assert.equal(sig.inputs.length, 1);
  assert.equal(sig.inputs[0].kwargName, 'arg1');
  assert.equal(sig.inputs[0].paramName, '_arg1_');
  assert.deepEqual(sig.inputs[0].source, { kind: 'placeholder', name: '_arg1_' });
  assert.equal(sig.output.type.kind, 'scalar');
});

test('signatureOf: functionof with placeholder boundary', () => {
  const sig = sigOf('f = functionof(c * _par_, par = _par_)\nc = 2.0', 'f');
  assert.equal(sig.kind, 'function');
  assert.equal(sig.inputs.length, 1);
  assert.equal(sig.inputs[0].kwargName, 'par');
  assert.deepEqual(sig.inputs[0].source, { kind: 'placeholder', name: '_par_' });
});

test('signatureOf: functionof with identifier boundaries → binding sources', () => {
  // Two binding boundaries — paramSources records each by name so the
  // viewer can fetch their empirical samples for auto-range.
  const sig = sigOf(`
theta1_dist = Normal(mu = 0, sigma = 1)
theta2_dist = Exponential(rate = 1)
theta1 = draw(theta1_dist)
theta2 = draw(theta2_dist)
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
`, 'f');
  assert.equal(sig.kind, 'function');
  assert.equal(sig.inputs.length, 2);
  assert.deepEqual(sig.inputs[0].source, { kind: 'binding', name: 'theta1' });
  assert.deepEqual(sig.inputs[1].source, { kind: 'binding', name: 'theta2' });
});

test('signatureOf: kernelof produces kind=kernel with measure-typed output', () => {
  const sig = sigOf(`
theta1 = draw(Normal(mu = 0, sigma = 1))
K = kernelof(Normal(mu = theta1, sigma = 1), theta1 = theta1)
`, 'K');
  assert.equal(sig.kind, 'kernel');
  assert.equal(sig.inputs.length, 1);
  assert.deepEqual(sig.inputs[0].source, { kind: 'binding', name: 'theta1' });
  assert.equal(sig.output.type.kind, 'measure');
});

test('signatureOf: likelihood inherits inputs from K, output is REAL', () => {
  // L = likelihoodof(K, obs); the profile-plot UI evaluates
  // log-density at obs swept over one of K's inputs. The signature
  // therefore exposes K's inputs and a real output.
  const sig = sigOf(`
theta1 = draw(Normal(mu = 0, sigma = 1))
K = kernelof(Normal(mu = theta1, sigma = 1), theta1 = theta1)
obs_value = 1.5
L = likelihoodof(K, obs_value)
`, 'L');
  assert.equal(sig.kind, 'likelihood');
  assert.equal(sig.inputs.length, 1);
  assert.deepEqual(sig.inputs[0].source, { kind: 'binding', name: 'theta1' });
  assert.equal(sig.output.type.kind, 'scalar');
  assert.equal(sig.kernelName, 'K');
  assert.equal(sig.obsValue, 1.5);
});

test('signatureOf: functionof with measure body classifies as kernel', () => {
  // Per spec §sec:functionof-measure: functionof with a measure body
  // produces a kernel. Even when typeinfer leaves the binding as
  // 'deferred' (e.g. for disintegrate-derived bindings that the
  // analyzer rewrites in place), bodyImpliesKernel walks the body to
  // detect the kind structurally — refs to measure-op bindings are
  // measure-typed, so the surrounding functionof is a kernel.
  const sig = sigOf(`
theta1 = draw(Normal(mu = 0, sigma = 1))
obs    = draw(Normal(mu = theta1, sigma = 1))
obs_dist = lawof(record(obs = obs))
fwd_kernel = functionof(obs_dist, theta1 = theta1)
`, 'fwd_kernel');
  assert.equal(sig.kind, 'kernel',
    'functionof whose body is a measure-typed binding should classify as kernel');
});

test('signatureOf: functionof with no kwargs auto-promotes elementof ancestors', () => {
  // Per spec §04 (sec:functionof): a single-argument functionof
  // traces the body's ancestor subgraph back to parametric leaves
  // (elementof bindings) and promotes them as inputs. The lowerer
  // doesn't record those leaves in ir.params (it only stores
  // user-written kwargs), so signatureOf must recover them by
  // walking the body through bindings.
  const sig = sigOf(`
mu = elementof(reals)
mu2 = mu^2
f = functionof(mu2)
`, 'f');
  assert.equal(sig.kind, 'function');
  assert.equal(sig.inputs.length, 1);
  assert.equal(sig.inputs[0].paramName, 'mu');
  assert.equal(sig.inputs[0].kwargName, 'mu');
  assert.deepEqual(sig.inputs[0].source, { kind: 'binding', name: 'mu' });
  assert.equal(sig.inputs[0].type.kind, 'scalar');
});

test('signatureOf: auto-promote collects multiple elementof leaves through chains', () => {
  // Two parametric leaves reached via different chain paths. Both
  // should surface as inputs in the trace order they're encountered.
  const sig = sigOf(`
a = elementof(reals)
b = elementof(reals)
s = a + b
f = functionof(s)
`, 'f');
  assert.equal(sig.inputs.length, 2);
  const names = sig.inputs.map((inp) => inp.paramName).sort();
  assert.deepEqual(names, ['a', 'b']);
});

test('signatureOf: auto-promote walks deep / branching chains, carries types', () => {
  // Stress the trace: three elementof leaves reached through DIFFERENT
  // intermediate chains, each with their own set restriction. Verifies
  // we recurse through every binding (not just the body's direct refs)
  // and surface the correct per-leaf type from the elementof binding.
  //
  //   a  (reals)    → a_sq      = a^2
  //   b  (posreals) → b_log     = log(b)
  //   k  (integers) → k_doubled = k * 2
  //   combined = a_sq + b_log + k_doubled
  //   f        = functionof(combined)
  const sig = sigOf(`
a = elementof(reals)
b = elementof(posreals)
k = elementof(integers)
a_sq      = a^2
b_log     = log(b)
k_doubled = k * 2
combined  = a_sq + b_log + k_doubled
f = functionof(combined)
`, 'f');
  assert.equal(sig.kind, 'function');
  assert.equal(sig.inputs.length, 3);

  const byName = {};
  for (const inp of sig.inputs) byName[inp.paramName] = inp;
  assert.ok(byName.a && byName.b && byName.k,
    'expected a, b, k all surfaced as inputs (got: '
    + Object.keys(byName).join(', ') + ')');

  // Each input's type comes from resolveSourceType walking the
  // elementof binding's inferredType — verify the set restriction
  // was preserved per leaf rather than collapsed to a single shape.
  assert.equal(byName.a.type.kind, 'scalar');
  assert.equal(byName.a.type.prim, 'real');
  assert.equal(byName.b.type.kind, 'scalar');
  assert.equal(byName.b.type.prim, 'real');
  assert.equal(byName.k.type.kind, 'scalar');
  assert.equal(byName.k.type.prim, 'integer');

  // Each source backref points at its own elementof binding —
  // distributeAxes / resolveAxisBaseSet rely on this for auto-range.
  assert.deepEqual(byName.a.source, { kind: 'binding', name: 'a' });
  assert.deepEqual(byName.b.source, { kind: 'binding', name: 'b' });
  assert.deepEqual(byName.k.source, { kind: 'binding', name: 'k' });
});

test('signatureOf: auto-promote dedupes leaves reached by multiple paths', () => {
  // Diamond shape: `a` appears in two branches of the body. The
  // trace must visit each binding name at most once — otherwise
  // `a` would surface twice and the profile-plot UI would render
  // two identical axis entries.
  const sig = sigOf(`
a = elementof(reals)
left  = a + 1
right = a * 2
combined = left + right
f = functionof(combined)
`, 'f');
  assert.equal(sig.inputs.length, 1);
  assert.equal(sig.inputs[0].paramName, 'a');
});

test('inlineForProfile: multi-input auto-promoted functionof rewrites every ref', () => {
  // End-to-end on the no-kwarg multi-input shape that signatureOf
  // synthesizes: a single inlineForProfile call must rewrite every
  // ref-to-input as %local. The profile evaluator can then sweep one
  // axis while binding the others via fixedEnv.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
a = elementof(reals)
b = elementof(reals)
s = a + b
f = functionof(s)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const ds = buildDerivations(lifted);
  const sig = signatureOf('f', lifted);
  const paramNames = sig.inputs.map((inp) => inp.paramName);
  assert.deepEqual(paramNames.sort(), ['a', 'b']);

  const out = inlineForProfile(sig.body, paramNames, ds.bindings, ds.derivations);
  // After inlining, the body is (add %local.a %local.b) — both
  // refs rewritten, no stray (ref self ...) left.
  assert.equal(out.kind, 'call');
  assert.equal(out.op, 'add');
  assert.equal(out.args[0].ns, '%local');
  assert.equal(out.args[1].ns, '%local');
  const localNames = [out.args[0].name, out.args[1].name].sort();
  assert.deepEqual(localNames, ['a', 'b']);
});

test('signatureOf: auto-promote stops at non-input ancestors (constants, draws)', () => {
  // Fixed ancestors (constants / external) are closed over per spec,
  // and stochastic ancestors aren't elementof leaves either —
  // neither should surface as implicit inputs. Result: no inputs.
  const sig = sigOf(`
c = 2.0
y = draw(Normal(mu = 0, sigma = 1))
z = c * y
f = functionof(z)
`, 'f');
  // Auto-promote walks self-refs; it only adds entries for
  // parameterized-phase elementof leaves. Neither c (literal) nor y
  // (draw) qualifies, so inputs stays empty.
  assert.equal(sig.inputs.length, 0);
});

test('signatureOf: auto-promote excludes external() and load_data() (fixed phase)', () => {
  // Per spec §04 sec:functionof: external(...) and load_data(...)
  // share binding.type='input' with elementof (both are surface
  // "inputs" to the module) but their phase is 'fixed' — they're
  // closed over by the reified callable, not promoted as inputs.
  // Only the elementof leaf `mu` should surface here.
  const sig = sigOf(`
mu = elementof(reals)
ext_const = external("hyperparams.json")
combined = mu + ext_const
f = functionof(combined)
`, 'f');
  assert.equal(sig.inputs.length, 1);
  assert.equal(sig.inputs[0].paramName, 'mu');
});

test('signatureOf: explicit boundary kwargs disable auto-promotion', () => {
  // When the user writes `functionof(z, a = a)`, the trace stops at
  // `a` and `a` is the sole input — additional elementof ancestors
  // beyond the boundary are NOT silently promoted. Otherwise a
  // user-declared signature could grow surprise inputs every time
  // an upstream derived binding picked up a new parametric ancestor.
  const sig = sigOf(`
a = elementof(reals)
b = elementof(reals)
s = a + b
f = functionof(s, a = a)
`, 'f');
  assert.equal(sig.inputs.length, 1);
  assert.equal(sig.inputs[0].paramName, 'a');
});

test('signatureOf: returns null for non-callable bindings', () => {
  // Sample / measure / literal bindings have no signature.
  assert.equal(sigOf('y = draw(Normal(mu=0, sigma=1))', 'y'), null);
  assert.equal(sigOf('m = Normal(mu=0, sigma=1)',       'm'), null);
  assert.equal(sigOf('c = 5.0',                         'c'), null);
});

// =====================================================================
// distributeAxes — flatten input cartprod / cartpow into atomic leaves
// =====================================================================

test('distributeAxes: scalar input → one axis', () => {
  // Use a binding-bound input so typeinfer pins the input to a
  // concrete scalar (theta1 is real). A bare `fn(_ + 1)` infers the
  // hole to 'any' since placeholders are unrestricted by default —
  // covered by the separate any/deferred test below.
  const sig = sigOf(`
theta1 = draw(Normal(mu = 0, sigma = 1))
f = functionof(theta1 + 1, theta1 = theta1)
`, 'f');
  const axes = distributeAxes(sig);
  assert.equal(axes.length, 1);
  assert.equal(axes[0].label, 'theta1');
  assert.deepEqual(axes[0].path, []);
  assert.equal(axes[0].leafType.kind, 'scalar');
});

test('distributeAxes: two scalar inputs → two top-level axes', () => {
  const sig = sigOf(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Normal(mu = 0, sigma = 1))
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
`, 'f');
  const axes = distributeAxes(sig);
  assert.equal(axes.length, 2);
  assert.deepEqual(axes.map(a => a.label).sort(), ['theta1', 'theta2']);
});

test('distributeAxes: any / deferred placeholder still emits an axis', () => {
  // Unrestricted placeholders (`fn(_)`, `_par_ = elementof(anything)`)
  // type to 'any'. The UI still needs an axis to profile-sweep over
  // — it just falls back to default range / default value rules
  // since there's no scalar prim to dispatch on.
  const sig = sigOf('f = fn(_)', 'f');
  const axes = distributeAxes(sig);
  assert.equal(axes.length, 1);
  assert.ok(axes[0].leafType.kind === 'any' || axes[0].leafType.kind === 'scalar');
});

test('distributeAxes: empty signature → empty axis list', () => {
  assert.deepEqual(distributeAxes(null), []);
  assert.deepEqual(distributeAxes({}), []);
  assert.deepEqual(distributeAxes({ inputs: [] }), []);
});

// =====================================================================
// inlineForProfile — propagate swept axis through deterministic deps
// =====================================================================

test('inlineForProfile: param self-ref rewrites to %local', () => {
  // (ref self theta1) where theta1 is a swept input becomes
  // (ref %local theta1). The body's evaluator picks the swept value
  // out of env keyed by paramName.
  const ir = { kind: 'ref', ns: 'self', name: 'theta1' };
  const out = inlineForProfile(ir, ['theta1'], new Map(), {});
  assert.equal(out.ns, '%local');
  assert.equal(out.name, 'theta1');
});

test('inlineForProfile: non-param ref left intact when binding non-evaluable', () => {
  // Constants (sample / iid / etc.) stay as self-refs for the
  // viewer's pre-materialise step to bind via fixedEnv.
  const ir = { kind: 'ref', ns: 'self', name: 'c' };
  const bindings = new Map([['c', { ir: { kind: 'lit', value: 5 } }]]);
  const derivations = { c: { kind: 'sample' } }; // non-evaluate
  const out = inlineForProfile(ir, [], bindings, derivations);
  assert.equal(out.ns, 'self');
  assert.equal(out.name, 'c');
});

test('inlineForProfile: evaluate-kind binding inlines its IR', () => {
  // a = c * theta1 (deterministic). When sweeping theta1 we want the
  // evaluator to see (mul <c-inlined> %local.theta1) — `a`'s body
  // gets inlined recursively (c is also evaluate-kind, so its lit IR
  // takes the place of self.c), and theta1 rewrites to %local.
  const aIR = {
    kind: 'call', op: 'mul',
    args: [
      { kind: 'ref', ns: 'self', name: 'c' },
      { kind: 'ref', ns: 'self', name: 'theta1' },
    ],
  };
  const bindings = new Map([
    ['a', { ir: aIR }],
    ['c', { ir: { kind: 'lit', value: 5 } }],
    ['theta1', { ir: { kind: 'call', op: 'Normal', kwargs: {} } }],
  ]);
  const derivations = {
    a: { kind: 'evaluate' },
    c: { kind: 'evaluate' },
    theta1: { kind: 'sample' },
  };
  const body = { kind: 'ref', ns: 'self', name: 'a' };
  const out = inlineForProfile(body, ['theta1'], bindings, derivations);
  assert.equal(out.kind, 'call');
  assert.equal(out.op, 'mul');
  // c inlined as the literal 5 (its full ir).
  assert.equal(out.args[0].kind, 'lit');
  assert.equal(out.args[0].value, 5);
  // theta1 rewritten to %local (it's the swept param).
  assert.equal(out.args[1].ns, '%local');
  assert.equal(out.args[1].name, 'theta1');
});

test('inlineForProfile: stochastic dep stays as self-ref (only evaluable inlined)', () => {
  // theta1 is sampled (kind: 'sample'); even when not the swept
  // param, we leave it as a self-ref so the viewer's pre-materialise
  // step picks one atom's value via fixedEnv. Inlining a stochastic
  // body would substitute the sampling IR, which the per-point
  // evaluator can't run.
  const ir = { kind: 'ref', ns: 'self', name: 'theta1' };
  const bindings = new Map([
    ['theta1', { ir: { kind: 'call', op: 'Normal', kwargs: {} } }],
  ]);
  const derivations = { theta1: { kind: 'sample' } };
  const out = inlineForProfile(ir, [], bindings, derivations);
  assert.equal(out.ns, 'self');
  assert.equal(out.name, 'theta1');
});

test('inlineForProfile: cycle guard — does not loop on self-cycle', () => {
  // Pathological case: a binding whose evaluable derivation refers
  // back to itself. The cycle guard leaves the second-encounter ref
  // intact rather than infinite-recursing.
  const aIR = { kind: 'ref', ns: 'self', name: 'a' };
  const bindings = new Map([['a', { ir: aIR }]]);
  const derivations = { a: { kind: 'evaluate' } };
  const out = inlineForProfile(
    { kind: 'ref', ns: 'self', name: 'a' }, [], bindings, derivations);
  // First lookup expands to aIR, then the cycle guard activates;
  // resulting IR is a self-ref to 'a' (unchanged), no exception.
  assert.equal(out.kind, 'ref');
  assert.equal(out.ns, 'self');
  assert.equal(out.name, 'a');
});

test('inlineForProfile: inlines pruned-but-evaluable call binding', () => {
  // The scenario that motivates the bindings-fallback in
  // inlineForProfile: buildDerivations prunes mu2's derivation
  // because mu2 transitively depends on a parameterized ancestor
  // (mu = elementof). The profile-plot pipeline still needs to
  // inline mu2's body so the swept `ref self mu` reaches the
  // %local rewrite. Without the fallback the body stays at
  // `ref self mu2` and profileN evaluates to NaN every point.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
mu = elementof(reals)
mu2 = mu^2
`);
  const lifted = liftInlineSubexpressions(bindings);
  const ds = buildDerivations(lifted);
  // Pre-condition: mu2's derivation was pruned (sanity-check that
  // we're exercising the fallback rather than the happy path).
  assert.ok(!ds.derivations['mu2'],
    'expected buildDerivations to prune mu2 (depends on parameterized mu)');

  const body = { kind: 'ref', ns: 'self', name: 'mu2' };
  const out = inlineForProfile(body, ['mu'], ds.bindings, ds.derivations);
  // mu2's IR (pow(mu, 2)) inlined, with self.mu rewritten to %local.mu.
  assert.equal(out.kind, 'call');
  assert.equal(out.op, 'pow');
  assert.equal(out.args[0].kind, 'ref');
  assert.equal(out.args[0].ns, '%local');
  assert.equal(out.args[0].name, 'mu');
  assert.equal(out.args[1].kind, 'lit');
  assert.equal(out.args[1].value, 2);
});

test('inlineForProfile: walks call args / kwargs / fields recursively', () => {
  // Coverage: substitution must reach into all structural slots.
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: {
      mu:    { kind: 'ref', ns: 'self', name: 'theta1' },
      sigma: { kind: 'lit', value: 1 },
    },
  };
  const out = inlineForProfile(ir, ['theta1'], new Map(), {});
  assert.equal(out.kwargs.mu.ns, '%local');
});

// =====================================================================
// resolveAxisBaseSet — backref → set descriptor (for auto-range)
// =====================================================================

test('resolveAxisBaseSet: identifier-bound elementof(reals) → reals descriptor', () => {
  // x_set = elementof(reals); used as Identifier boundary. The
  // boundary source is { kind: 'binding', name: 'x_set' } and the
  // resolver surfaces the set restriction (rather than treating
  // x_set as a stochastic binding for empirical-range computation).
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
x_set = elementof(reals)
f = functionof(x_set * 2, x = x_set)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const base = resolveAxisBaseSet(sig.inputs[0].source, lifted);
  assert.deepEqual(base, { kind: 'reals' });
});

test('resolveAxisBaseSet: identifier-bound elementof(interval(0,1))', () => {
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
p_set = elementof(interval(0, 1))
f = functionof(p_set * 2, p = p_set)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const base = resolveAxisBaseSet(sig.inputs[0].source, lifted);
  assert.deepEqual(base, { kind: 'interval', lo: 0, hi: 1 });
});

test('resolveAxisBaseSet: identifier-bound elementof(unitinterval) → [0, 1]', () => {
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
p_set = elementof(unitinterval)
f = functionof(p_set * 2, p = p_set)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const base = resolveAxisBaseSet(sig.inputs[0].source, lifted);
  assert.deepEqual(base, { kind: 'interval', lo: 0, hi: 1 });
});

test('resolveAxisBaseSet: anonymous placeholder boundary → null', () => {
  // `par = _par_` boundary is not bound to any elementof; treated
  // as elementof(anything). No set restriction to surface — the
  // viewer falls back to its leaf-type-based default range.
  const base = resolveAxisBaseSet({ kind: 'placeholder', name: '_par_' }, new Map());
  assert.equal(base, null);
});

test('resolveAxisBaseSet: stochastic binding source → empirical descriptor', () => {
  // theta1 boundary points at a non-elementof binding; the UI is
  // expected to materialise it and compute a quantile range.
  const source = { kind: 'binding', name: 'theta1' };
  const base = resolveAxisBaseSet(source, new Map());
  assert.deepEqual(base, { kind: 'empirical', name: 'theta1' });
});

test('resolveAxisBaseSet: null / unrecognised source → null', () => {
  assert.equal(resolveAxisBaseSet(null, new Map()), null);
  assert.equal(resolveAxisBaseSet({ kind: 'unknown' }, new Map()), null);
});

test('resolveAxisBaseSet: external() source → empirical (not a set restriction)', () => {
  // external(...) and load_data(...) share binding.type='input' with
  // elementof but their RHS op is 'external' / 'load_data', not
  // 'elementof'. The IR-op check (ir.op === 'elementof') correctly
  // skips them so they fall through to the empirical fallback —
  // there's no structural set restriction to surface.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
ext = external("data.dat")
`);
  const lifted = liftInlineSubexpressions(bindings);
  const base = resolveAxisBaseSet({ kind: 'binding', name: 'ext' }, lifted);
  assert.deepEqual(base, { kind: 'empirical', name: 'ext' });
});

// =====================================================================
// fourSigmaQuantileRange — 4-σ central quantile of an empirical sample
// =====================================================================

test('fourSigmaQuantileRange: empty / single → null / [v, v]', () => {
  assert.equal(fourSigmaQuantileRange(null), null);
  assert.equal(fourSigmaQuantileRange([]), null);
  assert.deepEqual(fourSigmaQuantileRange([3.14]), [3.14, 3.14]);
});

test('fourSigmaQuantileRange: monotone array → near-min / near-max', () => {
  // 1000 evenly-spaced values in [0, 1]. 4-σ tail is ~3.17e-5 — we
  // expect the lo/hi to land essentially at the endpoints (interp
  // pulls them very slightly inward from 0 / 1).
  const xs = new Float64Array(1000);
  for (let i = 0; i < 1000; i++) xs[i] = i / 999;
  const [lo, hi] = fourSigmaQuantileRange(xs);
  assert.ok(lo >= 0 && lo < 0.01, 'lo near 0 (got ' + lo + ')');
  assert.ok(hi <= 1 && hi > 0.99, 'hi near 1 (got ' + hi + ')');
});

test('fourSigmaQuantileRange: clipping at high N drops thinnest tails', () => {
  // 1e6 normal-like samples — extreme atoms get clipped. We don't
  // construct true normals (that needs the sampler); just check that
  // a single big-outlier injected into a tight bulk gets dropped.
  const N = 1000000;
  const xs = new Float64Array(N);
  for (let i = 0; i < N; i++) xs[i] = (i / N) * 2 - 1;  // [-1, 1]
  xs[N - 1] = 1e9;  // single outlier
  const [, hi] = fourSigmaQuantileRange(xs);
  assert.ok(hi < 100, 'outlier clipped (hi = ' + hi + ')');
});

// =====================================================================
// findMatchingPresets — record bindings whose kwargs match a callable
// =====================================================================

test('findMatchingPresets: matches record whose kwargs equal the input set', () => {
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
pars1 = record(theta1 = 1.4, theta2 = 1.0)
pars2 = record(theta1 = 0.5, theta2 = 2.0)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const presets = findMatchingPresets(sig, lifted);
  assert.equal(presets.length, 2);
  const byName = {};
  for (const p of presets) byName[p.name] = p;
  assert.deepEqual(byName.pars1.values, { theta1: 1.4, theta2: 1.0 });
  assert.deepEqual(byName.pars2.values, { theta1: 0.5, theta2: 2.0 });
});

test('findMatchingPresets: rejects records with extra or missing kwargs', () => {
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
extra_field   = record(theta1 = 1, theta2 = 2, theta3 = 3)
missing_field = record(theta1 = 1)
wrong_name    = record(theta1 = 1, mu = 2)
correct       = record(theta1 = 1, theta2 = 2)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const presets = findMatchingPresets(sig, lifted);
  assert.deepEqual(presets.map(p => p.name).sort(), ['correct']);
});

test('findMatchingPresets: accepts constant-foldable values, rejects non-constant', () => {
  // resolveConstant folds:
  //   - bare literals
  //   - refs to literal bindings
  //   - simple arithmetic (neg / add / sub / mul, including the
  //     `-3.5` → neg(lit 3.5) lowering for negative numbers).
  // It rejects refs to stochastic bindings, since their value isn't
  // statically determinate.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
c = 5.0
mu = draw(Normal(mu = 0, sigma = 1))
f = functionof(mu, mu = mu)
named_const  = record(mu = c)
lit_value    = record(mu = 1.5)
neg_lit      = record(mu = -2.5)
arith        = record(mu = c + 1)
stochastic   = record(mu = mu)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const presets = findMatchingPresets(sig, lifted);
  const names = presets.map(p => p.name).sort();
  assert.deepEqual(names, ['arith', 'lit_value', 'named_const', 'neg_lit']);
  const byName = {};
  for (const p of presets) byName[p.name] = p;
  assert.equal(byName.named_const.values.mu,  5.0);
  assert.equal(byName.lit_value.values.mu,    1.5);
  assert.equal(byName.neg_lit.values.mu,     -2.5);
  assert.equal(byName.arith.values.mu,        6.0);
});

test('findMatchingPresets: fixed(...) wrapper marks held-constant kwargs', () => {
  // Spec §03: values in a preset record wrapped in `fixed(...)` are
  // a hint to tooling that the value should be held constant during
  // optimization or sweep. The wrapper is identity at runtime, so
  // the resolved value is the same; only the fixedNames set tags
  // which kwargs were marked.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
mixed = record(theta1 = 1.4, theta2 = fixed(2.0))
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const presets = findMatchingPresets(sig, lifted);
  assert.equal(presets.length, 1);
  assert.deepEqual(presets[0].values, { theta1: 1.4, theta2: 2.0 });
  assert.deepEqual(Array.from(presets[0].fixedNames).sort(), ['theta2']);
});

test('findMatchingPresets: empty / null sig → empty list', () => {
  assert.deepEqual(findMatchingPresets(null,            new Map()), []);
  assert.deepEqual(findMatchingPresets({},              new Map()), []);
  assert.deepEqual(findMatchingPresets({ inputs: [] },  new Map()), []);
});

test('findMatchingPresets: record matches fn-derived auto-named arg axes', () => {
  // fn(_ + _) → functionof with paramKwargs ['arg1', 'arg2']. A
  // record(arg1=…, arg2=…) should match.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
g = fn(_ + _)
some_args = record(arg1 = 2.0, arg2 = -3.5)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('g', lifted);
  const presets = findMatchingPresets(sig, lifted);
  assert.equal(presets.length, 1);
  assert.deepEqual(presets[0].values, { arg1: 2.0, arg2: -3.5 });
});

// =====================================================================
// findMatchingDomains — cartprod bindings whose kwargs match a callable
// =====================================================================

test('findMatchingDomains: matches cartprod whose kwargs equal the input set', () => {
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
search = cartprod(theta1 = interval(-3, 3), theta2 = interval(0, 5))
wider  = cartprod(theta1 = interval(-10, 10), theta2 = interval(0, 20))
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const doms = findMatchingDomains(sig, lifted);
  assert.equal(doms.length, 2);
  const byName = {};
  for (const d of doms) byName[d.name] = d;
  assert.deepEqual(byName.search.ranges,
    { theta1: { lo: -3, hi: 3 }, theta2: { lo: 0, hi: 5 } });
  assert.deepEqual(byName.wider.ranges,
    { theta1: { lo: -10, hi: 10 }, theta2: { lo: 0, hi: 20 } });
});

test('findMatchingDomains: rejects cartprod with extra / missing kwargs', () => {
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
extra   = cartprod(theta1 = interval(-1, 1), theta2 = interval(0, 1), theta3 = interval(0, 1))
missing = cartprod(theta1 = interval(-1, 1))
correct = cartprod(theta1 = interval(-1, 1), theta2 = interval(0, 1))
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const doms = findMatchingDomains(sig, lifted);
  assert.deepEqual(doms.map(d => d.name).sort(), ['correct']);
});

test('findMatchingDomains: rejects degenerate / non-literal intervals', () => {
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
theta = draw(Normal(mu = 0, sigma = 1))
f = functionof(theta, theta = theta)
ok       = cartprod(theta = interval(-1, 1))
degenerate = cartprod(theta = interval(1, 1))
ref_bound  = cartprod(theta = interval(theta, 1))
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const doms = findMatchingDomains(sig, lifted);
  assert.deepEqual(doms.map(d => d.name).sort(), ['ok']);
});

test('findMatchingDomains: accepts named-set fields as unbounded factors', () => {
  // A cartprod field can be a named set (`reals` etc.) instead of an
  // interval, marking that kwarg as unbounded — the viewer will fall
  // back to the per-axis auto-fit there but still match the domain.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
mixed = cartprod(theta1 = interval(-4, 4), theta2 = reals)
fully_open = cartprod(theta1 = reals, theta2 = posreals)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f', lifted);
  const doms = findMatchingDomains(sig, lifted);
  const byName = {};
  for (const d of doms) byName[d.name] = d;
  assert.deepEqual(byName.mixed.ranges,   { theta1: { lo: -4, hi: 4 } });
  assert.deepEqual(byName.mixed.setNames, { theta2: 'reals' });
  assert.deepEqual(byName.fully_open.ranges, {});
  assert.deepEqual(byName.fully_open.setNames,
    { theta1: 'reals', theta2: 'posreals' });
});

test('findMatchingDomains: empty / null sig → empty list', () => {
  assert.deepEqual(findMatchingDomains(null,           new Map()), []);
  assert.deepEqual(findMatchingDomains({},             new Map()), []);
  assert.deepEqual(findMatchingDomains({ inputs: [] }, new Map()), []);
});

// =====================================================================
// substituteLocals — replace %local refs with literals from an env
// =====================================================================

test('substituteLocals: replaces %local refs with lit values', () => {
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: {
      mu:    { kind: 'ref', ns: '%local', name: 'theta1' },
      sigma: { kind: 'ref', ns: '%local', name: 'theta2' },
    },
  };
  const out = substituteLocals(ir, { theta1: 1.4, theta2: 1.0 });
  assert.equal(out.kwargs.mu.kind,    'lit');
  assert.equal(out.kwargs.mu.value,    1.4);
  assert.equal(out.kwargs.sigma.kind, 'lit');
  assert.equal(out.kwargs.sigma.value, 1.0);
});

test('substituteLocals: leaves self-refs intact', () => {
  const ir = { kind: 'ref', ns: 'self', name: 'c' };
  const out = substituteLocals(ir, { c: 5 });
  assert.equal(out.ns, 'self');
});

test('substituteLocals: leaves %local refs not in env intact', () => {
  const ir = { kind: 'ref', ns: '%local', name: 'theta1' };
  const out = substituteLocals(ir, { /* theta1 absent */ });
  assert.equal(out.ns, '%local');
});

test('substituteLocals: walks nested args / fields', () => {
  const ir = {
    kind: 'call', op: 'joint',
    fields: [
      { name: 'obs', value: {
        kind: 'call', op: 'iid',
        args: [
          { kind: 'call', op: 'Normal', kwargs: {
            mu:    { kind: 'ref', ns: '%local', name: 't' },
            sigma: { kind: 'lit', value: 1 },
          }},
          { kind: 'lit', value: 10 },
        ],
      }},
    ],
  };
  const out = substituteLocals(ir, { t: 2.5 });
  assert.equal(out.fields[0].value.args[0].kwargs.mu.value, 2.5);
});

// =====================================================================
// implicitKernelSignature — synthesise a kernel signature for a
// stochastic binding whose distIR transitively depends on elementof
// ancestors. Per spec §04 (sec:kernelof, sec:functionof), this is
// the natural "kernelof(x) with no boundary kwargs" semantics: trace
// back to parametric leaves and promote them as inputs.
// =====================================================================

// Helper: process source through processSource + buildDerivations,
// return both the lifted bindings and the derivations table —
// implicitKernelSignature needs the lifted form so its expandMeasureIR
// structural fallback can walk binding.ir.
function bindingsAndDerivationsFor(source) {
  const { bindings } = processSource(source);
  const ds = buildDerivations(bindings);
  return { bindings: ds.bindings, derivations: ds.derivations };
}

test('implicitKernelSignature: single elementof ancestor → one input', () => {
  // The canonical case: a draw whose distIR references a
  // parameterized leaf. buildDerivations prunes x's derivation
  // (parameterized ancestor), so implicitKernelSignature is the
  // only way to surface a plottable signature.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
`);
  // Sanity: x's derivation got pruned, otherwise we'd be testing
  // the wrong path.
  assert.ok(!derivations['x'],
    'expected buildDerivations to prune x (parameterized ancestor)');
  const sig = implicitKernelSignature('x', bindings, derivations);
  assert.ok(sig, 'expected a synthesised kernel signature');
  assert.equal(sig.kind, 'kernel');
  assert.equal(sig.implicit, true);
  assert.equal(sig.inputs.length, 1);
  assert.equal(sig.inputs[0].paramName, 'mu');
  assert.equal(sig.inputs[0].kwargName, 'mu');
  assert.deepEqual(sig.inputs[0].source, { kind: 'binding', name: 'mu' });
  // The body should be the structurally-expanded measure IR — for
  // a leaf distribution Normal(...) that's the call itself.
  assert.equal(sig.body.kind, 'call');
  assert.equal(sig.body.op, 'Normal');
});

test('implicitKernelSignature: multiple elementof ancestors → all promoted', () => {
  // Two parametric leaves both reached from the same draw — both
  // surface as inputs (deduped if they appear multiple times).
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
sigma = elementof(posreals)
x = draw(Normal(mu = mu, sigma = sigma))
`);
  const sig = implicitKernelSignature('x', bindings, derivations);
  assert.ok(sig);
  assert.equal(sig.inputs.length, 2);
  const names = sig.inputs.map((i) => i.paramName).sort();
  assert.deepEqual(names, ['mu', 'sigma']);
});

test('implicitKernelSignature: excludes external() (fixed phase)', () => {
  // Per spec, fixed-phase leaves are closed over — not promoted
  // as kernel inputs. external() / load_data() bindings have
  // type='input' but phase='fixed', so they must be filtered out.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
ext = external("data.dat")
x = draw(Normal(mu = mu, sigma = ext))
`);
  const sig = implicitKernelSignature('x', bindings, derivations);
  assert.ok(sig);
  // Only mu — ext is fixed-phase and must NOT be an input.
  const names = sig.inputs.map((i) => i.paramName);
  assert.deepEqual(names, ['mu']);
});

test('implicitKernelSignature: no elementof ancestors → null', () => {
  // A stochastic binding with no parametric ancestors has a
  // perfectly good derivation already; there's nothing for an
  // implicit kernel to reify. Return null so callers fall back
  // to the regular plot path.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
x = draw(Normal(mu = 0, sigma = 1))
`);
  const sig = implicitKernelSignature('x', bindings, derivations);
  assert.equal(sig, null);
});

test('implicitKernelSignature: walks through transitive deterministic deps', () => {
  // The parametric leaf is two hops back through evaluable
  // intermediates. expandMeasureIR's structural fallback walks the
  // chain so collectSelfRefs on the body still surfaces `mu` —
  // promoted as the input.
  //
  //   mu = elementof(reals)
  //   resolution = 2.5 + 0.3 * mu      (call, parameterized)
  //   x = draw(Normal(mu = 0, sigma = resolution))
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
resolution = 2.5 + 0.3 * mu
x = draw(Normal(mu = 0, sigma = resolution))
`);
  const sig = implicitKernelSignature('x', bindings, derivations);
  assert.ok(sig, 'expected a kernel signature even with derived intermediates');
  const names = sig.inputs.map((i) => i.paramName);
  assert.ok(names.includes('mu'),
    `expected 'mu' among inputs; got ${names.join(', ')}`);
});

test('implicitKernelSignature: bindings=null → null', () => {
  // No bindings → no way to walk ancestors; bail cleanly.
  const sig = implicitKernelSignature('x', null, {});
  assert.equal(sig, null);
});

test('implicitKernelSignature: unknown binding name → null', () => {
  // expandMeasureIR with bindings fallback returns null for an
  // unknown name; the signature is null in turn.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
`);
  const sig = implicitKernelSignature('nope', bindings, derivations);
  assert.equal(sig, null);
});

test('implicitKernelSignature: non-stochastic subject → null', () => {
  // Per the phase-based dispatch contract: only stochastic
  // bindings get a kernel interpretation. A deterministic
  // parametric-phase binding (mu2 = mu^2) must return null so
  // callers route to implicitFunctionSignature instead. Without
  // this gate, the helper would produce a "kernel" with body
  // pow(mu, 2) and the kernel-sample sampler would error
  // because pow isn't a distribution.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
mu2 = mu^2
`);
  const sig = implicitKernelSignature('mu2', bindings, derivations);
  assert.equal(sig, null);
});

test('implicitKernelSignature: parameterized measure binding → kernel sig', () => {
  // `m = iid(Normal(mu, 1), 3)` has phase='parameterized' (no draw
  // anywhere) but inferredType.kind='measure'. The dispatch gate
  // must accept measure-typed bindings even when phase isn't
  // strictly stochastic — otherwise plotting a measure with open
  // parameters would dead-end at "Not plottable".
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
x = iid(Normal(mu = mu, sigma = 1), 3)
`);
  const sig = implicitKernelSignature('x', bindings, derivations);
  assert.ok(sig);
  assert.equal(sig.body.kind, 'call');
  assert.equal(sig.body.op, 'iid');
  assert.equal(sig.body.args[0].op, 'Normal');
  // The iid count is preserved as the second positional.
  assert.equal(sig.body.args[1].kind, 'lit');
  assert.equal(sig.body.args[1].value, 3);
});

// =====================================================================
// implicitFunctionSignature — symmetric counterpart to
// implicitKernelSignature for parametric-phase deterministic
// bindings. Clicking on `mu2 = mu^2` (with mu = elementof(reals))
// is equivalent to plotting `functionof(mu2)`: a function of mu,
// evaluated through the profile-plot path.
// =====================================================================

test('implicitFunctionSignature: deterministic parametric binding → function sig', () => {
  // The motivating case: mu2 = mu^2 with mu = elementof. The
  // resulting sig has mu as input, kind='function', body is the
  // value-position pow IR. The viewer routes this to profile mode.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
mu2 = mu^2
`);
  const sig = implicitFunctionSignature('mu2', bindings, derivations);
  assert.ok(sig);
  assert.equal(sig.kind, 'function');
  assert.equal(sig.implicit, true);
  assert.equal(sig.inputs.length, 1);
  assert.equal(sig.inputs[0].paramName, 'mu');
  // Body is the binding's lowered IR — pow(mu, 2). We do NOT
  // expand through expandMeasureIR here (the body is value-
  // typed, not a measure).
  assert.equal(sig.body.kind, 'call');
  assert.equal(sig.body.op, 'pow');
  // Output type carried from the binding's inferredType so
  // enumerateOutputLeaves / domain-fitting work.
  assert.equal(sig.output.type.kind, 'scalar');
});

test('implicitFunctionSignature: multiple parametric leaves → all promoted', () => {
  const { bindings, derivations } = bindingsAndDerivationsFor(`
a = elementof(reals)
b = elementof(reals)
combo = a^2 + b^2
`);
  const sig = implicitFunctionSignature('combo', bindings, derivations);
  assert.ok(sig);
  assert.equal(sig.inputs.length, 2);
  const names = sig.inputs.map((i) => i.paramName).sort();
  assert.deepEqual(names, ['a', 'b']);
});

test('implicitFunctionSignature: stochastic subject → null', () => {
  // A draw / iid binding is the kernel-side's job; this helper
  // returns null so the viewer's dispatch doesn't double-handle it.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
`);
  const sig = implicitFunctionSignature('x', bindings, derivations);
  assert.equal(sig, null);
});

test('implicitFunctionSignature: fixed-phase binding → null', () => {
  // Closed-form computable bindings have no profile to plot —
  // they're a single value. The viewer renders them via the
  // fixed-scalar / fixed-record path; this helper returns null.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
ext = external("data.dat")
shifted = ext + 1
`);
  const sig = implicitFunctionSignature('shifted', bindings, derivations);
  assert.equal(sig, null);
});

test('implicitFunctionSignature: callable bindings → null', () => {
  // functionof / kernelof / fn / likelihood already have a real
  // signature via signatureOf; implicit should NOT shadow that.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
f = functionof(mu^2, x = mu)
`);
  const sig = implicitFunctionSignature('f', bindings, derivations);
  assert.equal(sig, null);
});

test('implicitFunctionSignature: elementof binding itself → null', () => {
  // The leaf elementof binding plots via its axis / set
  // descriptor, not as a parametric function of itself.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
`);
  const sig = implicitFunctionSignature('mu', bindings, derivations);
  assert.equal(sig, null);
});

test('implicitFunctionSignature: walks through transitive evaluables', () => {
  // Elementof leaf two hops back. Same BFS shape as the kernel
  // case — without it, only `inner` would be in the body's direct
  // refs and no inputs would surface.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
inner = 2.5 + 0.3 * mu
outer = inner * inner
`);
  const sig = implicitFunctionSignature('outer', bindings, derivations);
  assert.ok(sig);
  const names = sig.inputs.map((i) => i.paramName);
  assert.deepEqual(names, ['mu']);
});

test('implicitFunctionSignature: excludes external() (fixed phase)', () => {
  // Same spec rule as everywhere else: external / load_data are
  // closed over, not promoted.
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
ext = external("data.dat")
combo = mu * ext
`);
  const sig = implicitFunctionSignature('combo', bindings, derivations);
  assert.ok(sig);
  assert.deepEqual(sig.inputs.map((i) => i.paramName), ['mu']);
});

test('implicitFunctionSignature: bindings=null → null', () => {
  assert.equal(implicitFunctionSignature('x', null, {}), null);
});

test('implicitFunctionSignature: unknown binding name → null', () => {
  const { bindings, derivations } = bindingsAndDerivationsFor(`
mu = elementof(reals)
mu2 = mu^2
`);
  assert.equal(implicitFunctionSignature('nope', bindings, derivations), null);
});

// =====================================================================
// expandMeasureIR — structural fallback via the `bindings` parameter
// =====================================================================
//
// When buildDerivations prunes a derivation (parameterized ancestor),
// the derivation-based path of expandMeasureIR returns nothing and the
// caller wants the lifted binding.ir walked directly. The structural
// helper handles each measure-shape op per spec §06 / §sec:kernelof.
//
// We drive these via expandMeasureIR(name, derivations, undefined,
// bindings) with a bindings Map carrying hand-built `.ir` shapes, and
// a derivations table empty for `name` so the primary path falls
// through to the structural fallback.

// Helper: build a bindings Map from { name → ir } and an empty
// derivations table. Returns the result of expandMeasureIR walking
// the named binding's .ir structurally.
function expandStructural(name, bindingMap) {
  const bindings = new Map();
  for (const k in bindingMap) bindings.set(k, { ir: bindingMap[k] });
  return expandMeasureIR(name, {}, undefined, bindings);
}

test('expandMeasureIR structural: draw(M) unwraps to M', () => {
  // Per spec: `lawof(draw(M)) ≡ M`. The structural walker pre-
  // strips the draw / lawof wrapper so the caller gets a measure
  // shape regardless of whether the binding was written as
  // `m = M` vs `y = draw(M)` vs `m = lawof(y)`.
  const innerNormal = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 },
  }};
  const out = expandStructural('x', {
    x: { kind: 'call', op: 'draw', args: [innerNormal] },
  });
  assert.equal(out.op, 'Normal');
});

test('expandMeasureIR structural: lawof(M) unwraps to M', () => {
  const innerNormal = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 },
  }};
  const out = expandStructural('x', {
    x: { kind: 'call', op: 'lawof', args: [innerNormal] },
  });
  assert.equal(out.op, 'Normal');
});

test('expandMeasureIR structural: leaf distribution returned as-is', () => {
  // SAMPLEABLE_DISTRIBUTIONS short-circuit: the leaf call is its
  // own measure IR. Refs inside its kwargs stay intact — the
  // sampler resolves them per-atom via refArrays / env.
  const irNormal = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'ref', ns: 'self', name: 'mu' },
    sigma: { kind: 'lit', value: 1 },
  }};
  const out = expandStructural('x', { x: irNormal });
  // Exact same node back (structural is a pass-through here).
  assert.equal(out, irNormal);
});

test('expandMeasureIR structural: iid(M, n) recurses on the measure arg', () => {
  // The iid count and other dims are preserved positionally; only
  // arg[0] (the measure) is expanded structurally. This is the
  // shape behind `x = iid(Normal(...), 3)`.
  const innerNormal = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 },
  }};
  const out = expandStructural('x', {
    x: { kind: 'call', op: 'iid', args: [
      { kind: 'call', op: 'draw', args: [innerNormal] },  // draw wrapper
      { kind: 'lit', value: 3 },
    ]},
  });
  assert.equal(out.op, 'iid');
  // arg[0] structurally expanded (draw unwrapped → Normal).
  assert.equal(out.args[0].op, 'Normal');
  // arg[1] preserved verbatim.
  assert.equal(out.args[1].kind, 'lit');
  assert.equal(out.args[1].value, 3);
});

test('expandMeasureIR structural: nested iid(iid(M, n), m) recurses through', () => {
  // The recursion preserves the outer iid; the inner iid is
  // expanded recursively (its own arg[0] gets walked too).
  const innerNormal = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 },
  }};
  const out = expandStructural('x', {
    x: { kind: 'call', op: 'iid', args: [
      { kind: 'call', op: 'iid', args: [innerNormal, { kind: 'lit', value: 3 }] },
      { kind: 'lit', value: 2 },
    ]},
  });
  assert.equal(out.op, 'iid');
  assert.equal(out.args[0].op, 'iid');
  assert.equal(out.args[0].args[0].op, 'Normal');
});

test('expandMeasureIR structural: joint(...) recurses on each field', () => {
  // record-style measure with named fields: each field.value is a
  // measure that needs to be structurally expanded so embedded
  // draws/lawofs get unwrapped uniformly.
  const m1 = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 },
  }};
  const m2 = { kind: 'call', op: 'Exponential', kwargs: {
    rate: { kind: 'lit', value: 1 },
  }};
  const out = expandStructural('x', {
    x: { kind: 'call', op: 'joint', fields: [
      { name: 'a', value: { kind: 'call', op: 'draw', args: [m1] } },
      { name: 'b', value: m2 },
    ]},
  });
  assert.equal(out.op, 'joint');
  assert.equal(out.fields.length, 2);
  assert.equal(out.fields[0].name, 'a');
  assert.equal(out.fields[0].value.op, 'Normal');  // draw unwrapped
  assert.equal(out.fields[1].name, 'b');
  assert.equal(out.fields[1].value.op, 'Exponential');
});

test('expandMeasureIR structural: weighted(w, M) recurses on the measure arg', () => {
  // weighted is positional: arg[0] is the weight (scalar or
  // function-of-variate), arg[1] is the measure. Only the
  // measure gets expanded; the weight stays as-is for the
  // walker to evaluate per-atom.
  const innerNormal = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 },
  }};
  const weight = { kind: 'ref', ns: 'self', name: 'w' };
  const out = expandStructural('x', {
    x: { kind: 'call', op: 'weighted', args: [
      weight,
      { kind: 'call', op: 'draw', args: [innerNormal] },
    ]},
  });
  assert.equal(out.op, 'weighted');
  // arg[0] left intact for per-atom evaluation.
  assert.equal(out.args[0], weight);
  // arg[1] (the measure) structurally expanded.
  assert.equal(out.args[1].op, 'Normal');
});

test('expandMeasureIR structural: logweighted(lw, M) same shape as weighted', () => {
  const innerNormal = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 },
  }};
  const out = expandStructural('x', {
    x: { kind: 'call', op: 'logweighted', args: [
      { kind: 'lit', value: -1.5 },
      { kind: 'call', op: 'lawof', args: [innerNormal] },
    ]},
  });
  assert.equal(out.op, 'logweighted');
  assert.equal(out.args[1].op, 'Normal');
});

test('expandMeasureIR structural: unknown op is passed through unchanged', () => {
  // normalize / superpose / truncate / pushfwd aren't density-
  // scoreable today, but the structural walker should still
  // return the IR so downstream code can inspect / sample-plan it
  // (the materialiser will surface a precise error if needed).
  const inner = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 },
  }};
  const irNormalize = { kind: 'call', op: 'normalize', args: [inner] };
  const out = expandStructural('x', { x: irNormalize });
  // Returned as-is, not null, not unwrapped — the inner Normal
  // is NOT structurally walked here (only the documented measure-
  // combinator shapes recurse). The walker stops at the unknown op
  // so the caller decides how to handle it.
  assert.equal(out, irNormalize);
});

test('expandMeasureIR structural: ref self <name> follows through bindings', () => {
  // A self-ref inside a measure binding's IR delegates back to
  // expandMeasureIR(refName), which falls through to the
  // structural walker for the referenced binding's .ir. This is
  // how the orchestrator chains through lifted-subexpression
  // anonymous bindings.
  const innerNormal = { kind: 'call', op: 'Normal', kwargs: {
    mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 },
  }};
  const bindings = new Map([
    ['x', { ir: { kind: 'call', op: 'iid', args: [
      { kind: 'ref', ns: 'self', name: '__anon0' },
      { kind: 'lit', value: 5 },
    ]}}],
    ['__anon0', { ir: innerNormal }],
  ]);
  const out = expandMeasureIR('x', {}, undefined, bindings);
  assert.equal(out.op, 'iid');
  assert.equal(out.args[0].op, 'Normal');
  assert.equal(out.args[1].value, 5);
});

test('expandMeasureIR structural: missing binding name → null', () => {
  // Symmetric with the derivation-path: caller asks for a binding
  // that doesn't exist; structural walker returns null cleanly.
  const out = expandMeasureIR('nope', {}, undefined, new Map());
  assert.equal(out, null);
});

test('expandMeasureIR structural: binding without .ir → null', () => {
  // type='input' bindings (elementof / external) have no .ir to
  // walk — they're leaves of the value graph. Asking for one as a
  // measure is a caller bug, but we return null rather than
  // throw, matching the derivation-path's behavior.
  const bindings = new Map([['x', { type: 'input' /* no ir */ }]]);
  const out = expandMeasureIR('x', {}, undefined, bindings);
  assert.equal(out, null);
});

// =====================================================================
// Dirac — point-mass measure
// =====================================================================
//
// Spec §sec:measure-algebra: `Dirac(value = v)` concentrates mass 1 at v.
// The engine handles it on three fronts:
//   1. As a measure binding (`m = Dirac(value = 5)`) — sample step on
//      the Dirac IR, the worker's REGISTRY entry emits N copies.
//   2. As a draw (`y = draw(Dirac(value = e))`) — identity rewrite in
//      classifyForChain folds the draw away, the chain step is an
//      `evaluate` on the value IR.
//   3. As a downstream dep — the alias / measure-algebra path keeps
//      working because Dirac is in SAMPLEABLE_DISTRIBUTIONS.

test('Dirac: measure binding produces a sample step on the Dirac IR', () => {
  const r = chainOf('m = Dirac(value = 5.0)', 'm');
  assert.equal(r.unsupported, undefined);
  assert.equal(r.chain.length, 1);
  assert.equal(r.chain[0].name, 'm');
  assert.equal(r.chain[0].kind, 'sample');
  assert.equal(r.chain[0].ir.op, 'Dirac');
  assert.equal(r.chain[0].ir.kwargs.value.value, 5.0);
});

test('Dirac: draw(Dirac(value = lit)) becomes an evaluate step on the value', () => {
  // Identity rewrite: y = draw(Dirac(value = 5)) ≡ y = 5.
  // Worker side, this evaluates the literal IR rather than spinning
  // up a degenerate sampler.
  const r = chainOf('y = draw(Dirac(value = 5.0))', 'y');
  assert.equal(r.unsupported, undefined);
  assert.equal(r.chain.length, 1);
  assert.equal(r.chain[0].name, 'y');
  assert.equal(r.chain[0].kind, 'evaluate');
  // The override IR is the value arg, not the wrapping draw call.
  assert.equal(r.chain[0].ir.kind, 'lit');
  assert.equal(r.chain[0].ir.value, 5.0);
});

test('Dirac: draw(Dirac(value = ref)) evaluates the ref through the chain', () => {
  // The value arg can reference an upstream binding. The rewrite
  // preserves the ref; the worker's evaluator resolves it from the
  // refArrays env at draw time.
  const r = chainOf(`
mu = 3.14
y  = draw(Dirac(value = mu))
`, 'y');
  assert.equal(r.unsupported, undefined);
  // Two steps: mu (evaluate) and y (evaluate via the rewrite).
  assert.deepEqual(r.chain.map(s => s.name), ['mu', 'y']);
  assert.equal(r.chain[1].kind, 'evaluate');
  // The override IR is the ref to mu.
  assert.equal(r.chain[1].ir.kind, 'ref');
  assert.equal(r.chain[1].ir.name, 'mu');
});

test('Dirac: stochastic value flows through the rewrite normally', () => {
  // y = draw(Dirac(value = stochastic_x)) ≡ y = stochastic_x.
  // The chain has the upstream Normal sample, then y as evaluate via
  // the rewrite — same pattern as `y = stochastic_x` directly.
  const r = chainOf(`
x = draw(Normal(mu = 0, sigma = 1))
y = draw(Dirac(value = x))
`, 'y');
  assert.equal(r.unsupported, undefined);
  assert.deepEqual(r.chain.map(s => s.name), ['x', 'y']);
  assert.equal(r.chain[0].kind, 'sample');
  assert.equal(r.chain[1].kind, 'evaluate');
  assert.equal(r.chain[1].ir.kind, 'ref');
  assert.equal(r.chain[1].ir.name, 'x');
});

test('Dirac: leafSampleIR returns the Dirac IR for measure-alias bindings', () => {
  // Viewer's fixed-Dirac path uses leafSampleIR to pluck the Dirac
  // IR (incl. the value sub-IR) for surface-form text rendering.
  const { bindings } = processSource('m = Dirac(value = 5.0)');
  const { derivations } = buildDerivations(bindings);
  const ir = leafSampleIR('m', derivations);
  assert.ok(ir, 'leafSampleIR returns non-null');
  assert.equal(ir.op, 'Dirac');
  assert.equal(ir.kwargs.value.kind, 'lit');
  assert.equal(ir.kwargs.value.value, 5.0);
});

// =====================================================================
// Measure-form canonicalisation: lawof and positional Dirac normalize
// to Dirac(value = …)
// =====================================================================
//
// Per spec §sec:variate-measure + §sec:lawof, lawof of a value-typed e
// is the probability measure that a draw of e is governed by — for a
// deterministic e that's a Dirac at e. And per §sec:calling-convention,
// positional and keyword forms of built-in distribution calls are
// semantically identical. The orchestrator canonicalises both into
// Dirac(value = e) so downstream consumers see one shape per
// equivalence class.

test('Dirac canonicalisation: positional Dirac with binding ref classifies as alias', () => {
  // m = Dirac(some_lit) — positional. classifyDerivation
  // canonicalises to Dirac(value = some_lit), then promotes the
  // ref-to-binding case to 'alias' (same equivalence class as
  // lawof(ref) — the binding's per-atom value is the ref's
  // per-atom value, so getMeasure can short-circuit through the
  // alias chain without invoking the sampler).
  const { bindings } = processSource(`
some_lit = 5.0
m = Dirac(some_lit)
`);
  const { derivations } = buildDerivations(bindings);
  assert.ok(derivations.m, 'm derivable');
  assert.equal(derivations.m.kind, 'alias');
  assert.equal(derivations.m.from, 'some_lit');
});

test('Dirac canonicalisation: positional Dirac with literal value stays a sample step', () => {
  // m = Dirac(5.0) — value is a literal, not a ref. There's no
  // binding to alias to, so the sample path is taken (Dirac with
  // kwarg-form distIR) and the sampler emits N copies of the value.
  const { bindings } = processSource('m = Dirac(5.0)');
  const { derivations } = buildDerivations(bindings);
  assert.equal(derivations.m && derivations.m.kind, 'sample');
  assert.equal(derivations.m.distIR.op, 'Dirac');
  // After canonicalisation the value kwarg is present (was positional).
  assert.ok(derivations.m.distIR.kwargs && derivations.m.distIR.kwargs.value);
  assert.equal(derivations.m.distIR.kwargs.value.kind, 'lit');
});

test('lawof canonicalisation: lawof(value-binding-ref) classifies as alias', () => {
  // m = lawof(some_lit). The engine routes refs through 'alias'
  // (lighter than synthesising a Dirac sample step + sampler).
  // Viewer follows the alias chain to render m exactly like some_lit.
  const { bindings } = processSource(`
some_lit = 5.0
m = lawof(some_lit)
`);
  const { derivations } = buildDerivations(bindings);
  assert.ok(derivations.m, 'm derivable');
  assert.equal(derivations.m.kind, 'alias');
  assert.equal(derivations.m.from, 'some_lit');
});

test('draw(Dirac(positional)) ≡ e: identity rewrite via canonicalisation', () => {
  // Per-atom: y = draw(Dirac(some_lit)) ≡ y = some_lit.
  // After resolveMeasure normalises Dirac(positional) → Dirac(value=…),
  // the existing draw(Dirac) rewrite folds y to an evaluate step.
  const r = chainOf(`
some_lit = 5.0
y = draw(Dirac(some_lit))
`, 'y');
  assert.equal(r.unsupported, undefined);
  const yStep = r.chain.find(s => s.name === 'y');
  assert.ok(yStep);
  assert.equal(yStep.kind, 'evaluate');
});

test('draw(lawof(ref)) ≡ ref: identity rewrite via canonicalisation', () => {
  // resolveMeasure recursively resolves lawof(ref-to-binding) by
  // chasing the ref → binding's lowered RHS, applying the lawof
  // canonicalisation along the way; the outer draw then sees
  // Dirac(value=…) and rewrites to an evaluate step.
  const r = chainOf(`
some_lit = 5.0
m = lawof(some_lit)
y = draw(m)
`, 'y');
  assert.equal(r.unsupported, undefined);
  const yStep = r.chain.find(s => s.name === 'y');
  assert.ok(yStep);
  assert.equal(yStep.kind, 'evaluate');
});

test('phase sharpening: degenerate draws inherit value phase', () => {
  // Spec identity: draw(Dirac(value=e)) ≡ e and draw(lawof(e)) ≡ e
  // for value-typed e. Phase sharpening makes degenerate draws
  // inherit the value's phase rather than the structural-stochastic
  // default. Non-degenerate draws (Normal, …) keep structural rule.
  const { computePhases } = require('../analyzer');
  const { bindings } = processSource(`
some_lit = 5.0
some_arr = [1.2, 3.4]
m1 = lawof(some_lit)
m2 = Dirac(some_lit)
y1 = draw(m1)
y2 = draw(m2)
y3 = draw(lawof(some_arr))
y4 = draw(Dirac(some_arr))
real = draw(Normal(mu = 0, sigma = 1))
`);
  const phases = computePhases(bindings);
  assert.equal(phases.get('m1'), 'fixed');
  assert.equal(phases.get('m2'), 'fixed');
  assert.equal(phases.get('y1'), 'fixed', 'draw(lawof(value-binding)) → fixed');
  assert.equal(phases.get('y2'), 'fixed', 'draw(Dirac(value-binding)) → fixed');
  assert.equal(phases.get('y3'), 'fixed', 'draw(inline lawof) → fixed');
  assert.equal(phases.get('y4'), 'fixed', 'draw(inline Dirac) → fixed');
  assert.equal(phases.get('real'), 'stochastic', 'real Normal draw stays stochastic');
});

// =====================================================================
// enumerateOutputLeaves + extractOutputIR
// =====================================================================
//
// Output-side counterpart to distributeAxes / collectSelfRefs:
// walks a callable's specialized output type to enumerate its
// scalar leaves, then extracts the corresponding sub-IR from the
// body for any chosen leaf. Used by the viewer to expose an
// "Output:" dropdown for multi-output functions and route the
// profile-plot evaluation to the chosen scalar leaf.

const T = require('../types');
const {
  enumerateOutputLeaves, extractOutputIR,
} = require('../orchestrator');

test('enumerateOutputLeaves: scalar output → single empty-path entry', () => {
  const leaves = enumerateOutputLeaves(T.REAL);
  assert.equal(leaves.length, 1);
  assert.deepEqual(leaves[0].path, []);
  assert.equal(leaves[0].label, '');
  assert.ok(T.equal(leaves[0].leafType, T.REAL));
});

test('enumerateOutputLeaves: record output → one entry per field', () => {
  const t = { kind: 'record', fields: { a: T.REAL, b: T.INTEGER } };
  const leaves = enumerateOutputLeaves(t);
  assert.equal(leaves.length, 2);
  const byPath = {};
  for (const L of leaves) byPath[L.path.join('.')] = L;
  assert.ok(byPath['a'] && T.equal(byPath['a'].leafType, T.REAL));
  assert.ok(byPath['b'] && T.equal(byPath['b'].leafType, T.INTEGER));
  assert.equal(byPath['a'].label, '.a');
  assert.equal(byPath['b'].label, '.b');
});

test('enumerateOutputLeaves: tuple output → 1-indexed positional leaves', () => {
  const t = { kind: 'tuple', elems: [T.REAL, T.INTEGER] };
  const leaves = enumerateOutputLeaves(t);
  assert.equal(leaves.length, 2);
  assert.deepEqual(leaves[0].path, [1]);
  assert.equal(leaves[0].label, '[1]');
  assert.deepEqual(leaves[1].path, [2]);
  assert.equal(leaves[1].label, '[2]');
});

test('enumerateOutputLeaves: nested record-in-record', () => {
  const t = { kind: 'record', fields: {
    inner: { kind: 'record', fields: { x: T.REAL, y: T.REAL } },
    flag:  T.BOOLEAN,
  }};
  const leaves = enumerateOutputLeaves(t);
  // inner.x, inner.y, flag — three scalar leaves total.
  assert.equal(leaves.length, 3);
  const labels = leaves.map(L => L.label).sort();
  assert.deepEqual(labels, ['.flag', '.inner.x', '.inner.y']);
});

test('extractOutputIR: scalar output → returns body unchanged', () => {
  const body = { kind: 'call', op: 'mul', args: [{ kind: 'lit', value: 2 }] };
  assert.deepEqual(extractOutputIR(body, []), body);
});

test('extractOutputIR: record body, field path → field value IR', () => {
  const body = {
    kind: 'call', op: 'record',
    fields: [
      { name: 'a', value: { kind: 'lit', value: 1 } },
      { name: 'b', value: { kind: 'lit', value: 2 } },
    ],
  };
  const a = extractOutputIR(body, ['a']);
  assert.equal(a && a.value, 1);
  const b = extractOutputIR(body, ['b']);
  assert.equal(b && b.value, 2);
});

test('extractOutputIR: tuple body, 1-indexed slot → arg IR', () => {
  const body = {
    kind: 'call', op: 'tuple',
    args: [
      { kind: 'lit', value: 'first' },
      { kind: 'lit', value: 'second' },
    ],
  };
  assert.equal(extractOutputIR(body, [1]).value, 'first');
  assert.equal(extractOutputIR(body, [2]).value, 'second');
});

test('extractOutputIR: missing field / out-of-range slot → null', () => {
  const body = { kind: 'call', op: 'record', fields: [
    { name: 'a', value: { kind: 'lit', value: 1 } },
  ]};
  assert.equal(extractOutputIR(body, ['nonexistent']), null);
  // Not a record/tuple/array op — path with segment unresolvable.
  assert.equal(extractOutputIR({ kind: 'lit', value: 42 }, ['x']), null);
});

test('lift: fn(record(a=_, b=2*_)) keeps both holes as function params', () => {
  // Regression for an aliasing-machinery bug: liftInlineSubexpressions
  // visited record kwargs and lifted each kwarg value (including the
  // raw `_` Hole node) into a separate anon binding, clearing the
  // function's params. After the fix, lift bails on any expression
  // containing a Hole or Placeholder marker.
  const { bindings } = processSource(`
f_demo = fn(record(a = _, b = 2 * _))
`);
  const r = buildDerivations(bindings);
  const fb = r.bindings.get('f_demo');
  assert.deepEqual(fb.ir.params, ['_arg1_', '_arg2_']);
  // signatureOf should expose both params as inputs.
  const sig = require('../orchestrator').signatureOf('f_demo', r.bindings);
  assert.equal(sig.inputs.length, 2);
});

test('jointchain: positional form lifts inline M and K, classifies as tuple', () => {
  // Pre-existing gap in inlineChainOps: only handled positional
  // jointchain when both args were already Identifiers. Inline
  // expressions like jointchain(Exponential(1), fn(Normal(1, _)))
  // bailed at the type check and the binding stayed unsupported.
  // Now lifts inline expressions to anon bindings before lookup.
  const { bindings } = processSource(`
funnel = jointchain(Exponential(rate = 1), fn(Normal(mu = 1, sigma = _)))
`);
  const r = buildDerivations(bindings);
  assert.ok(r.derivations.funnel, 'funnel derivable');
  assert.equal(r.derivations.funnel.kind, 'tuple');
  assert.equal(r.derivations.funnel.elems.length, 2);
  // The b component should resolve to a Normal draw whose sigma
  // is the a-variate ref (conditional dependence preserved). The
  // draw of an inline measure ref takes the engine's lighter
  // 'alias' classification — same downstream behaviour as 'sample'
  // since getMeasure chases the alias to the Normal sample step.
  const bAnon = r.derivations.funnel.elems[1];
  const bDeriv = r.derivations[bAnon];
  assert.ok(bDeriv && (bDeriv.kind === 'sample' || bDeriv.kind === 'alias'),
    'b component should be sample- or alias-classified');
});

// =====================================================================
// Fixed-phase pre-evaluation (spec §sec:random end-to-end)
// =====================================================================
//
// buildDerivations runs a fixed-phase pre-eval pass after the main
// classifier. For each fixed-phase binding it tries to compute the
// value end-to-end via sampler.evaluateExpr, exposes them as a
// `fixedValues` Map, and reclassifies based on the result shape.

test('pre-eval: spec example computes random_data + threads state', () => {
  const { bindings } = processSource(`
    rngseed = [0xb2, 0x51, 0xa4, 0x93, 0x49, 0xd8, 0x68, 0x88]
    rstate = rnginit(rngseed)
    random_data, rstate2 = rand(rstate, iid(Normal(0, 1), 10))
    more_random_data, rstate3 = rand(rstate2, iid(Exponential(1), 5))
  `);
  const { derivations, fixedValues } = buildDerivations(bindings);

  // Every named binding has a value in fixedValues.
  for (const name of ['rngseed', 'rstate', 'random_data', 'rstate2',
                      'more_random_data', 'rstate3']) {
    assert.ok(fixedValues.has(name), `missing ${name} in fixedValues`);
  }

  // random_data is a length-10 array of finite reals.
  const rd = fixedValues.get('random_data');
  assert.equal(rd.length, 10);
  for (const v of rd) assert.ok(Number.isFinite(v));

  // more_random_data is length 5; Exponential samples are positive.
  const md = fixedValues.get('more_random_data');
  assert.equal(md.length, 5);
  for (const v of md) assert.ok(v >= 0, 'exponential is non-negative');

  // States are opaque objects, not numbers.
  assert.equal(typeof fixedValues.get('rstate'),  'object');
  assert.equal(typeof fixedValues.get('rstate2'), 'object');
  assert.equal(typeof fixedValues.get('rstate3'), 'object');

  // Plottable arrays live in fixedValues — the viewer's getMeasure
  // short-circuits any binding present there to a Float64Array,
  // bypassing the per-atom evaluate path that would mis-broadcast
  // a length-10 array. The derivation kind stays whatever the
  // classifier produced (typically 'evaluate' for tuple_get) — it's
  // unused once fixedValues claims the binding.
  assert.deepEqual(Array.from(fixedValues.get('random_data')), Array.from(rd));
  assert.equal(fixedValues.get('more_random_data').length, 5);
});

test('pre-eval: deterministic across calls (same seed → same values)', () => {
  const src = `
    seed = [1, 2, 3, 4, 5, 6, 7, 8]
    s = rnginit(seed)
    x, s2 = rand(s, iid(Normal(0, 1), 10))
  `;
  const r1 = buildDerivations(processSource(src).bindings);
  const r2 = buildDerivations(processSource(src).bindings);
  assert.deepEqual(
    Array.from(r1.fixedValues.get('x')),
    Array.from(r2.fixedValues.get('x')),
    'rand(rnginit(...), ...) is deterministic w.r.t. seed');
});

test('pre-eval: downstream scalars consume fixed arrays', () => {
  const { bindings } = processSource(`
    seed = [9, 9, 9, 9]
    s = rnginit(seed)
    samples, s2 = rand(s, iid(Normal(0, 1), 100))
    s_max = maximum(samples)
    s_min = minimum(samples)
  `);
  const { fixedValues } = buildDerivations(bindings);
  const samples = fixedValues.get('samples');
  assert.equal(samples.length, 100);
  // The reduction values should exist and match Math.{max,min} of the
  // 100-sample array, since maximum / minimum are evaluable and their
  // operand is a fixed-phase array.
  assert.ok(fixedValues.has('s_max'));
  assert.ok(fixedValues.has('s_min'));
  const expectedMax = Math.max(...samples);
  const expectedMin = Math.min(...samples);
  assert.equal(fixedValues.get('s_max'), expectedMax);
  assert.equal(fixedValues.get('s_min'), expectedMin);
});

test('pre-eval: stops at parameterized boundary (no infinite loop)', () => {
  const { bindings } = processSource(`
    a = elementof(reals)
    b = a + 1
    c = b * 2
  `);
  const { fixedValues } = buildDerivations(bindings);
  // a / b / c are parameterized, NOT fixed-phase. Pre-eval must skip
  // them and the iteration must terminate (this test would hang if
  // the loop didn't quit cleanly on a non-fixed phase).
  assert.equal(fixedValues.has('a'), false);
  assert.equal(fixedValues.has('b'), false);
  assert.equal(fixedValues.has('c'), false);
});

// =====================================================================
// Auto-splat (spec §sec:calling-convention) for user-defined callables
// =====================================================================
//
// `f(record(a=x, b=y))` and `f(some_record_value)` are equivalent to
// `f(a=x, b=y)`. Tested here at the orchestrator level: the user-call
// inliner splats the record, the closure walk sees the substituted
// boundaries, and pre-eval evaluates the result.

test('auto-splat: forward_kernel(rand_pars) drives end-to-end rand chain', () => {
  // The spec example pattern: prior-as-lawof(record(...)), draw a
  // record-of-parameters, splat into the forward kernel, then sample
  // the resulting measure. Pre-eval should compute every step.
  const { bindings, diagnostics } = processSource(`
    theta1 = draw(Normal(0, 1))
    theta2 = draw(Exponential(1))
    forward_kernel = functionof(joint(obs = iid(Normal(mu = theta1, sigma = theta2), 10)), theta1 = theta1, theta2 = theta2)
    prior = lawof(record(theta1 = theta1, theta2 = theta2))
    rs = rnginit([1,2,3,4])
    rp, rs2 = rand(rs, prior)
    ro, _ = rand(rs2, forward_kernel(rp))
  `);
  const errs = diagnostics.filter(d => d.severity === 'error');
  assert.deepEqual(errs, [], `unexpected errors: ${JSON.stringify(errs)}`);

  const { fixedValues } = buildDerivations(bindings);
  // rp is a record with both fields populated.
  const rp = fixedValues.get('rp');
  assert.equal(typeof rp, 'object');
  assert.equal(typeof rp.theta1, 'number');
  assert.equal(typeof rp.theta2, 'number');
  // ro is the sampled record-shaped measure: { obs: array(10) }.
  const ro = fixedValues.get('ro');
  assert.ok(ro && typeof ro === 'object', 'ro should be a record');
  assert.ok(Array.isArray(ro.obs), 'ro.obs should be an array');
  assert.equal(ro.obs.length, 10);
  for (const v of ro.obs) assert.ok(Number.isFinite(v), 'finite obs sample');
});

test('auto-splat: inline record(...) call splats fields', () => {
  const { bindings, diagnostics } = processSource(`
    f = functionof(a + b, a = c, b = d)
    c = elementof(reals)
    d = elementof(reals)
    y = f(record(a = 3, b = 4))
  `);
  const errs = diagnostics.filter(d => d.severity === 'error');
  assert.deepEqual(errs, [], `unexpected errors: ${JSON.stringify(errs)}`);
});
