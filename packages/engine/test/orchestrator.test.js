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
  findMatchingPresets,
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
// findMatchingPresets — preset bindings whose kwargs match a callable
// =====================================================================

test('findMatchingPresets: matches preset whose kwargs equal the input set', () => {
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
pars1 = preset(theta1 = 1.4, theta2 = 1.0)
pars2 = preset(theta1 = 0.5, theta2 = 2.0)
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

test('findMatchingPresets: rejects presets with extra or missing kwargs', () => {
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
f = functionof(theta1 + theta2, theta1 = theta1, theta2 = theta2)
extra_field   = preset(theta1 = 1, theta2 = 2, theta3 = 3)
missing_field = preset(theta1 = 1)
wrong_name    = preset(theta1 = 1, mu = 2)
correct       = preset(theta1 = 1, theta2 = 2)
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
named_const  = preset(mu = c)
lit_value    = preset(mu = 1.5)
neg_lit      = preset(mu = -2.5)
arith        = preset(mu = c + 1)
stochastic   = preset(mu = mu)
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

test('findMatchingPresets: empty / null sig → empty list', () => {
  assert.deepEqual(findMatchingPresets(null,            new Map()), []);
  assert.deepEqual(findMatchingPresets({},              new Map()), []);
  assert.deepEqual(findMatchingPresets({ inputs: [] },  new Map()), []);
});

test('findMatchingPresets: preset matches fn-derived auto-named arg axes', () => {
  // fn(_ + _) → functionof with paramKwargs ['arg1', 'arg2']. A
  // preset(arg1=…, arg2=…) should match.
  const { liftInlineSubexpressions } = require('../orchestrator');
  const { bindings } = processSource(`
g = fn(_ + _)
some_args = preset(arg1 = 2.0, arg2 = -3.5)
`);
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('g', lifted);
  const presets = findMatchingPresets(sig, lifted);
  assert.equal(presets.length, 1);
  assert.deepEqual(presets[0].values, { arg1: 2.0, arg2: -3.5 });
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
