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
    kind: 'call', op: 'log',  // not in EVALUABLE_OPS yet
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

test('derivations: inline draw(Dist(...)) becomes a sample step on the inner', () => {
  const { derivations } = derivationsOf(`
y = draw(Normal(mu = 0, sigma = 1))
`);
  assert.equal(derivations.y.kind, 'sample');
  assert.equal(derivations.y.distIR.op, 'Normal');
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
  // prior is a multivariate lawof — unsupported. theta1 still has a
  // derivation (it doesn't reference prior). Anything that refs prior
  // would cascade-drop.
  const { derivations } = derivationsOf(`
theta1 = draw(Normal(mu = 0, sigma = 1))
prior  = lawof(record(theta1 = theta1))
ghost  = prior + 1
`);
  assert.ok(derivations.theta1);
  assert.ok(!('prior' in derivations));
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
