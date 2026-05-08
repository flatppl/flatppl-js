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
  signatureOf, distributeAxes,
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
