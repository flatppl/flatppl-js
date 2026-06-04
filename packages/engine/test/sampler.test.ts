'use strict';

// Tests for engine/sampler.js — built-in distribution sampling and
// analytical density via stdlib.
//
// Coverage:
//   - rand(state, measureIR, env) for each registered distribution
//   - Reproducibility via Philox-state threading (same state → same value)
//   - Param resolution from env for ref-typed kwargs
//   - Param translation (FlatPPL spec names → stdlib positional)
//   - density() return shape: continuous vs discrete reference, support,
//     plot range
//   - Statistical sanity: mean/variance of N samples ≈ analytical mean/var
//   - canSample / isKnownDistribution gating
//   - Error handling: unknown distributions, unbound refs

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const rng = require('../rng.ts');

// Helper: build a distribution call IR from a name + kwargs (numeric values
// only, lifted to lit nodes).
function distIR(op: any, kwargs: any) {
  const out: any = {};
  for (const [k, v] of Object.entries(kwargs)) {
    out[k] = { kind: 'lit', value: v, loc: synthLoc() };
  }
  return { kind: 'call', op, kwargs: out, loc: synthLoc() };
}

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}

// Helper: ref node for env-resolved params.
function refIR(name: any) {
  return { kind: 'ref', ns: 'self', name, loc: synthLoc() };
}

// Helper: take N samples, return a Float64Array. Reseeds from same state
// every call so it's deterministic and repeatable.
function takeN(measureIR: any, env: any, n: any, seed = [1, 2, 3]) {
  let state = rng.seedFromBytes(seed);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [v, next] = sampler.rand(state, measureIR, env);
    out[i] = v;
    state = next;
  }
  return out;
}

// =====================================================================
// Distribution registry
// =====================================================================

test('isKnownDistribution: returns true for registered, false otherwise', () => {
  assert.equal(sampler.isKnownDistribution('Normal'), true);
  assert.equal(sampler.isKnownDistribution('Exponential'), true);
  assert.equal(sampler.isKnownDistribution('NotARealDistribution'), false);
});

test('listDistributions: includes the v1 set', () => {
  const list = sampler.listDistributions();
  for (const name of [
    'Normal', 'Exponential', 'LogNormal', 'Beta', 'Gamma',
    'Cauchy', 'StudentT', 'Bernoulli', 'Binomial', 'Poisson',
  ]) {
    assert.ok(list.includes(name), `expected ${name} in registry`);
  }
});

// =====================================================================
// Basic rand — one sample per distribution, sane outputs
// =====================================================================

test('rand: Normal(0, 1) produces finite real-valued samples', () => {
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const samples = takeN(ir, {}, 1000);
  for (const v of samples) {
    assert.ok(Number.isFinite(v), `non-finite sample: ${v}`);
  }
});

test('rand: Exponential(rate=1) produces non-negative samples', () => {
  const ir = distIR('Exponential', { rate: 1 });
  const samples = takeN(ir, {}, 1000);
  for (const v of samples) {
    assert.ok(v >= 0, `negative exponential sample: ${v}`);
  }
});

test('rand: Bernoulli(p=0.3) produces 0 or 1', () => {
  const ir = distIR('Bernoulli', { p: 0.3 });
  const samples = takeN(ir, {}, 1000);
  for (const v of samples) {
    assert.ok(v === 0 || v === 1, `Bernoulli sample not in {0,1}: ${v}`);
  }
});

test('rand: Beta(2, 5) produces samples in [0, 1]', () => {
  const ir = distIR('Beta', { alpha: 2, beta: 5 });
  const samples = takeN(ir, {}, 1000);
  for (const v of samples) {
    assert.ok(v >= 0 && v <= 1, `Beta sample out of [0,1]: ${v}`);
  }
});

test('rand: Poisson(rate=3) produces non-negative integers', () => {
  const ir = distIR('Poisson', { rate: 3 });
  const samples = takeN(ir, {}, 500);
  for (const v of samples) {
    assert.ok(Number.isInteger(v), `Poisson sample not integer: ${v}`);
    assert.ok(v >= 0, `Poisson sample negative: ${v}`);
  }
});

test('rand: Binomial(n=10, p=0.5) produces integers in [0, 10]', () => {
  const ir = distIR('Binomial', { n: 10, p: 0.5 });
  const samples = takeN(ir, {}, 500);
  for (const v of samples) {
    assert.ok(Number.isInteger(v), `Binomial sample not integer: ${v}`);
    assert.ok(v >= 0 && v <= 10, `Binomial sample out of [0,10]: ${v}`);
  }
});

test('rand: Gamma(shape=2, rate=1) produces positive samples', () => {
  const ir = distIR('Gamma', { shape: 2, rate: 1 });
  const samples = takeN(ir, {}, 1000);
  for (const v of samples) {
    assert.ok(v > 0, `Gamma sample not positive: ${v}`);
  }
});

test('rand: LogNormal(0, 1) produces positive samples', () => {
  const ir = distIR('LogNormal', { mu: 0, sigma: 1 });
  const samples = takeN(ir, {}, 1000);
  for (const v of samples) {
    assert.ok(v > 0, `LogNormal sample not positive: ${v}`);
  }
});

test('rand: StudentT(nu=3) produces finite samples', () => {
  const ir = distIR('StudentT', { nu: 3 });
  const samples = takeN(ir, {}, 1000);
  for (const v of samples) {
    assert.ok(Number.isFinite(v), `StudentT sample not finite: ${v}`);
  }
});

test('rand: Cauchy(0, 1) produces samples (heavy-tailed but finite)', () => {
  const ir = distIR('Cauchy', { location: 0, scale: 1 });
  const samples = takeN(ir, {}, 1000);
  for (const v of samples) {
    assert.ok(Number.isFinite(v), `Cauchy sample not finite: ${v}`);
  }
});

// =====================================================================
// Reproducibility
// =====================================================================

test('rand: same state + same measure → same sample', () => {
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const state = rng.seedFromBytes([42]);
  const [v1] = sampler.rand(state, ir, {});
  const [v2] = sampler.rand(state, ir, {});
  assert.equal(v1, v2);
});

test('rand: state advances; consecutive samples differ', () => {
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  let state = rng.seedFromBytes([42]);
  const [v1, s1] = sampler.rand(state, ir, {});
  const [v2] = sampler.rand(s1, ir, {});
  // Two independent draws shouldn't equal each other (probability ~0).
  assert.notEqual(v1, v2);
});

test('rand: full reproducibility — same seed yields identical streams', () => {
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const s1 = takeN(ir, {}, 100, [7, 7, 7]);
  const s2 = takeN(ir, {}, 100, [7, 7, 7]);
  assert.deepEqual(Array.from(s1), Array.from(s2));
});

// =====================================================================
// Param resolution from env
// =====================================================================

test('rand: resolves ref-typed parameters from env', () => {
  // Normal(mu = mu_p, sigma = 1) with env { mu_p: 5 }
  const ir = {
    kind: 'call',
    op: 'Normal',
    kwargs: {
      mu:    refIR('mu_p'),
      sigma: { kind: 'lit', value: 1, loc: synthLoc() },
    },
    loc: synthLoc(),
  };
  const samples = takeN(ir, { mu_p: 5 }, 1000);
  // Sample mean should be near 5, not near 0.
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  assert.ok(Math.abs(mean - 5) < 0.2, `mean ${mean} should be ~5`);
});

test('rand: arithmetic in parameters is resolved', () => {
  // Normal(mu = mu_p + 10, sigma = 1)
  const ir = {
    kind: 'call',
    op: 'Normal',
    kwargs: {
      mu: {
        kind: 'call', op: 'add',
        args: [refIR('mu_p'), { kind: 'lit', value: 10, loc: synthLoc() }],
        loc: synthLoc(),
      },
      sigma: { kind: 'lit', value: 1, loc: synthLoc() },
    },
    loc: synthLoc(),
  };
  const samples = takeN(ir, { mu_p: 0 }, 1000);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  assert.ok(Math.abs(mean - 10) < 0.2, `mean ${mean} should be ~10`);
});

test('rand: throws on unbound ref', () => {
  const ir: any = distIR('Normal', { mu: 0, sigma: 1 });
  ir.kwargs.mu = refIR('not_in_env');
  const state = rng.seedFromBytes([1]);
  assert.throws(
    () => sampler.rand(state, ir, {}),
    /unbound .* reference 'not_in_env'/i
  );
});

test('rand: throws on unknown distribution', () => {
  const ir = distIR('NotARealDist', { x: 0 });
  const state = rng.seedFromBytes([1]);
  assert.throws(
    () => sampler.rand(state, ir, {}),
    /not a known distribution/
  );
});

// =====================================================================
// Statistical sanity (mean / variance ≈ analytical)
// =====================================================================

test('Normal(2, 0.5): empirical mean and stdev close to analytical', () => {
  const ir = distIR('Normal', { mu: 2, sigma: 0.5 });
  const samples = takeN(ir, {}, 10000);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;
  // 3σ bounds for sample mean: stdev_of_mean = sigma/sqrt(n) = 0.5/100 = 0.005
  // For variance: somewhat looser bound. Check within 5%.
  assert.ok(Math.abs(mean - 2) < 0.05, `mean ${mean} not close to 2`);
  assert.ok(Math.abs(variance - 0.25) < 0.025, `variance ${variance} not close to 0.25`);
});

test('Exponential(rate=2): empirical mean ≈ 0.5', () => {
  const ir = distIR('Exponential', { rate: 2 });
  const samples = takeN(ir, {}, 10000);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  // True mean = 1/rate = 0.5; variance = 1/rate^2 = 0.25
  // stdev_of_mean = sqrt(0.25/10000) = 0.005
  assert.ok(Math.abs(mean - 0.5) < 0.05, `mean ${mean} not close to 0.5`);
});

test('Bernoulli(p=0.7): empirical proportion ≈ 0.7', () => {
  const ir = distIR('Bernoulli', { p: 0.7 });
  const samples = takeN(ir, {}, 10000);
  const ones = samples.reduce((a, b) => a + b, 0);
  const p = ones / samples.length;
  // stdev = sqrt(0.7 * 0.3 / 10000) ≈ 0.0046; 3σ ≈ 0.014
  assert.ok(Math.abs(p - 0.7) < 0.02, `proportion ${p} not close to 0.7`);
});

// =====================================================================
// Analytical: makeAnalytical, density
// =====================================================================

test('makeAnalytical: returns stdlib instance with .pdf, .cdf, etc.', () => {
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const dist = sampler.makeAnalytical(ir, {});
  // Standard Normal PDF at 0 is 1/sqrt(2π) ≈ 0.3989
  assert.ok(Math.abs(dist.pdf(0) - 0.3989) < 0.001);
  // CDF at mean = 0.5
  assert.ok(Math.abs(dist.cdf(0) - 0.5) < 1e-9);
  // Mean
  assert.equal(dist.mean, 0);
});

test('density: continuous distribution returns lebesgue-reference grid', () => {
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const d = sampler.density(ir, {});
  assert.equal(d.reference, 'lebesgue');
  assert.equal(d.xs.length, d.ys.length);
  assert.ok(d.xs.length >= 100);  // default grid is reasonably dense
  // Support range from default 0.001/0.999 quantiles ≈ ±3.09 for Normal(0,1)
  assert.ok(d.support[0] < -2 && d.support[0] > -4);
  assert.ok(d.support[1] >  2 && d.support[1] <  4);
  // PDF values should be non-negative
  for (const y of d.ys) assert.ok(y >= 0);
  // PDF should peak near zero and integrate close to 1.
  const dx = (d.support[1] - d.support[0]) / (d.xs.length - 1);
  let area = 0;
  for (const y of d.ys) area += y * dx;
  assert.ok(Math.abs(area - 1) < 0.01, `PDF integrates to ${area}, not ~1`);
});

test('density: opts.range overrides the quantile-derived plot range', () => {
  const ir = distIR('Exponential', { rate: 1 });
  const d = sampler.density(ir, {}, { range: [0, 5], gridPoints: 50 });
  assert.equal(d.support[0], 0);
  assert.equal(d.support[1], 5);
  assert.equal(d.xs.length, 50);
  // First grid point at lo, last at hi.
  assert.equal(d.xs[0], 0);
  assert.equal(d.xs[d.xs.length - 1], 5);
  // PDF non-negative; Exponential mode at 0.
  for (const y of d.ys) assert.ok(y >= 0);
});

test('density: discrete distribution returns counting-reference atoms', () => {
  const ir = distIR('Poisson', { rate: 3 });
  const d = sampler.density(ir, {});
  assert.equal(d.reference, 'counting');
  // xs should be integers
  for (const x of d.xs) assert.ok(Number.isInteger(x));
  // PMF should sum (approximately) to 1 within the quantile range
  let total = 0;
  for (const y of d.ys) total += y;
  assert.ok(total > 0.95 && total < 1.005,
    `PMF sum ${total} should be close to 1 (quantile-bounded)`);
});

test('density: Bernoulli — exactly two atoms (0 and 1)', () => {
  const ir = distIR('Bernoulli', { p: 0.3 });
  const d = sampler.density(ir, {});
  assert.equal(d.reference, 'counting');
  assert.deepEqual(Array.from(d.xs), [0, 1]);
  // p(0) = 0.7, p(1) = 0.3
  assert.ok(Math.abs(d.ys[0] - 0.7) < 1e-9);
  assert.ok(Math.abs(d.ys[1] - 0.3) < 1e-9);
});

test('density: custom quantile bounds tighten the plot range', () => {
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const wide   = sampler.density(ir, {}, { qLo: 0.001, qHi: 0.999 });
  const narrow = sampler.density(ir, {}, { qLo: 0.1, qHi: 0.9 });
  assert.ok(narrow.support[0] > wide.support[0]);
  assert.ok(narrow.support[1] < wide.support[1]);
});

test('density: custom grid resolution', () => {
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const fine   = sampler.density(ir, {}, { gridPoints: 500 });
  const coarse = sampler.density(ir, {}, { gridPoints: 50 });
  assert.equal(fine.xs.length, 500);
  assert.equal(coarse.xs.length, 50);
});

// =====================================================================
// Param translation (regression — make sure each entry's params work)
// =====================================================================

test('Gamma uses spec names (shape, rate) — passes through correctly', () => {
  const ir = distIR('Gamma', { shape: 5, rate: 1 });
  const dist = sampler.makeAnalytical(ir, {});
  assert.equal(dist.mean, 5);  // mean = shape/rate = 5
});

test('Cauchy uses spec names (location, scale)', () => {
  const ir = distIR('Cauchy', { location: 3, scale: 1 });
  const dist = sampler.makeAnalytical(ir, {});
  assert.equal(dist.median, 3);
});

test('StudentT uses spec name (nu)', () => {
  const ir = distIR('StudentT', { nu: 5 });
  const dist = sampler.makeAnalytical(ir, {});
  // mean = 0 for nu > 1
  assert.equal(dist.mean, 0);
});

test('Poisson uses spec name (rate)', () => {
  const ir = distIR('Poisson', { rate: 4 });
  const dist = sampler.makeAnalytical(ir, {});
  assert.equal(dist.mean, 4);
});

// =====================================================================
// makeParametricSampler — params resolved per draw rather than baked
// into the factory closure. This is the per-i-params fast path used by
// the worker for sampleN with refArrays.
// =====================================================================

test('makeParametricSampler: drawWith resolves params from env', () => {
  // ref-typed params; params change per drawWith call.
  const ir = {
    kind: 'call', op: 'Normal', kwargs: {
      mu:    { kind: 'ref', ns: 'self', name: 'mu', loc: synthLoc() },
      sigma: { kind: 'ref', ns: 'self', name: 'sigma', loc: synthLoc() },
    }, loc: synthLoc(),
  };
  const state = rng.stateFromKey(7);
  const s = sampler.makeParametricSampler(state, ir);
  // First draw: mu=0, sigma=1.
  const v0 = s.drawWith({ mu: 0, sigma: 1 });
  // Second draw: mu=10, sigma=0.001 — should be tightly around 10.
  const v1 = s.drawWith({ mu: 10, sigma: 0.001 });
  assert.equal(Number.isFinite(v0), true);
  assert.ok(Math.abs(v1 - 10) < 0.01,
    `expected v1 ≈ 10 with tiny sigma, got ${v1}`);
});

test('makeParametricSampler: matches makeSampler for static params', () => {
  // For literal-kwarg IR with the same env, drawing K values via the
  // parametric path should produce the same values as the baked-in
  // factory path, given the same starting state.
  const ir = distIR('Exponential', { rate: 2 });
  const seed = 42, K = 50;

  const a = sampler.makeSampler(rng.stateFromKey(seed), ir, {});
  const b = sampler.makeParametricSampler(rng.stateFromKey(seed), ir);
  for (let i = 0; i < K; i++) {
    const va = a.draw();
    const vb = b.drawWith({});
    assert.equal(va, vb, `draw ${i}: ${va} vs ${vb}`);
  }
});

test('makeParametricSampler: factory built once across many draws', () => {
  // Sanity perf check: 50K parametric draws of Normal with per-call
  // params should be dramatically faster than 50K rand() calls (which
  // rebuild the factory each time). We don't assert a hard ratio (CI
  // jitter), just that the parametric path completes well under a
  // second — the rand() path is the slow baseline in production.
  const ir = {
    kind: 'call', op: 'Normal', kwargs: {
      mu:    { kind: 'ref', ns: 'self', name: 'mu',    loc: synthLoc() },
      sigma: { kind: 'ref', ns: 'self', name: 'sigma', loc: synthLoc() },
    }, loc: synthLoc(),
  };
  const N = 50_000;
  const state = rng.stateFromKey(123);
  const s = sampler.makeParametricSampler(state, ir);
  const env: any = {};
  const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    env.mu = i * 0.001;
    env.sigma = 1 + (i % 10) * 0.01;
    s.drawWith(env);
  }
  const elapsed = Date.now() - t0;
  assert.ok(elapsed < 2000, `parametric path too slow: ${elapsed}ms for ${N} draws`);
});

// =====================================================================
// FlatPPL primitives: rnginit, rand, rngstate, tuple_get
// =====================================================================
//
// Covers spec §sec:random end-to-end via sampler.evaluateExpr — the
// entry point the orchestrator and worker dispatch to. tuple_get is the
// internal IR op emitted by the analyzer's multi-LHS rewriter; rnginit
// builds an opaque rngstate from a byte seed; rand threads state
// through measure draws via the measure walker (sampler.walk).

function lit(v: any)   { return { kind: 'lit', value: v, loc: synthLoc() }; }
function call(op: any, args: any, kwargs?: any) {
  const out: any = { kind: 'call', op, loc: synthLoc() };
  if (args && args.length)             out.args   = args;
  if (kwargs && Object.keys(kwargs).length) out.kwargs = kwargs;
  return out;
}

test('rnginit: produces a Philox state from a byte seed', () => {
  const ir = call('rnginit', [call('vector', [lit(0xb2), lit(0x51), lit(0xa4), lit(0x93)])]);
  const state = sampler.evaluateExpr(ir, {});
  assert.ok(state && Array.isArray(state.key) && state.key.length === 2,
    'rnginit returns a Philox state object');
  assert.ok(Array.isArray(state.counter) && state.counter.length === 4);
});

test('rnginit: rejects non-byte vectors', () => {
  // 256 is out of byte range.
  const ir = call('rnginit', [call('vector', [lit(0), lit(256)])]);
  assert.throws(() => sampler.evaluateExpr(ir, {}), /byte vector/);
});

test('rngstate <-> bytes round-trip (rng.bytesFromState ∘ rng.stateFromBytes)', () => {
  const original = rng.seedFromBytes([1, 2, 3, 4]);
  const bytes = rng.bytesFromState(original);
  const restored = rng.stateFromBytes(bytes);
  assert.deepEqual(restored.key, original.key);
  assert.deepEqual(restored.counter, original.counter);
});

test('rngstate: rejects bytes lacking the engine magic', () => {
  // 24 zero bytes: magic mismatch.
  const bad = new Array(24).fill(0);
  assert.throws(() => rng.stateFromBytes(bad), /magic mismatch/);
});

test('rand on Normal: returns (value, new_state) tuple, threads state', () => {
  const init = sampler.evaluateExpr(
    call('rnginit', [call('vector', [lit(7), lit(0), lit(0), lit(0)])]),
    {});
  const measureIR = call('Normal', [], { mu: lit(0), sigma: lit(1) });
  const tuple = sampler.evaluateExpr(
    call('rand', [{ kind: 'ref', ns: 'self', name: 'rs', loc: synthLoc() }, measureIR]),
    { rs: init });
  assert.ok(Array.isArray(tuple) && tuple.length === 2);
  const [value, newState] = tuple;
  assert.equal(typeof value, 'number');
  assert.ok(newState && newState.key, 'second slot is a state');
  assert.notDeepEqual(newState.counter, init.counter, 'state advanced');
});

test('rand on iid(Normal, n): returns array of n + new state', () => {
  const init = rng.seedFromBytes([42]);
  const iidIR = call('iid', [
    call('Normal', [], { mu: lit(0), sigma: lit(1) }),
    lit(10),
  ]);
  const tuple = sampler.evaluateExpr(
    call('rand', [{ kind: 'ref', ns: 'self', name: 'rs', loc: synthLoc() }, iidIR]),
    { rs: init });
  const [arr, newState] = tuple;
  assert.ok(Array.isArray(arr) && arr.length === 10);
  for (const v of arr) assert.equal(typeof v, 'number');
  assert.ok(newState && newState.key);
});

test('rand: chained calls thread state through (deterministic)', () => {
  // (random_data, rstate2) = rand(rstate, iid(Normal,5))
  // (more,        rstate3) = rand(rstate2, iid(Exponential,3))
  // Compute twice with same seed; results identical.
  function run() {
    const rs1 = sampler.evaluateExpr(
      call('rnginit', [call('vector', [lit(0xb2), lit(0x51), lit(0xa4), lit(0x93)])]),
      {});
    const t1 = sampler.evaluateExpr(
      call('rand', [{ kind: 'ref', ns: 'self', name: 'rs', loc: synthLoc() },
                    call('iid', [call('Normal', [], { mu: lit(0), sigma: lit(1) }), lit(5)])]),
      { rs: rs1 });
    const t2 = sampler.evaluateExpr(
      call('rand', [{ kind: 'ref', ns: 'self', name: 'rs', loc: synthLoc() },
                    call('iid', [call('Exponential', [], { rate: lit(1) }), lit(3)])]),
      { rs: t1[1] });
    return [t1[0], t2[0]];
  }
  const a = run();
  const b = run();
  assert.deepEqual(a, b, 'same seed → same outputs across runs');
});

test('rand on joint(record): returns record + new state', () => {
  const init = rng.seedFromBytes([5]);
  const jointIR = {
    kind: 'call', op: 'joint',
    fields: [
      { name: 'x', value: call('Normal',      [], { mu: lit(0), sigma: lit(1) }) },
      { name: 'y', value: call('Exponential', [], { rate: lit(1) }) },
    ],
    loc: synthLoc(),
  };
  const tuple = sampler.evaluateExpr(
    call('rand', [{ kind: 'ref', ns: 'self', name: 'rs', loc: synthLoc() }, jointIR]),
    { rs: init });
  const [rec, newState] = tuple;
  assert.equal(typeof rec, 'object');
  assert.equal(typeof rec.x, 'number');
  assert.equal(typeof rec.y, 'number');
  assert.ok(newState && newState.key);
});

test('rand: refuses a non-state first arg with a clear error', () => {
  const measureIR = call('Normal', [], { mu: lit(0), sigma: lit(1) });
  assert.throws(
    () => sampler.evaluateExpr(call('rand', [lit(42), measureIR]), {}),
    /must be an rngstate/);
});

test('tuple_get: projects an evaluated tuple by literal slot', () => {
  // tuple_get over an inline tuple call.
  const ir = call('tuple_get', [
    call('tuple', [lit(10), lit(20), lit(30)]),
    lit(1),
  ]);
  assert.equal(sampler.evaluateExpr(ir, {}), 20);
});

test('indicesof: vector returns 1-based index Value', () => {
  // 1-D vector input → rank-1 Value of length N with entries 1..N.
  const ir = call('indicesof', [call('vector', [lit(10), lit(20), lit(30), lit(40)])]);
  const out = sampler.evaluateExpr(ir, {});
  assert.ok(out && out.shape && out.data,
    'indicesof returns a shape-tagged Value, got ' + JSON.stringify(out));
  assert.deepEqual(Array.from(out.shape), [4]);
  assert.deepEqual(Array.from(out.data), [1, 2, 3, 4]);
});

test('indicesof0: vector returns 0-based index Value', () => {
  const ir = call('indicesof0', [call('vector', [lit(10), lit(20), lit(30), lit(40)])]);
  const out = sampler.evaluateExpr(ir, {});
  assert.deepEqual(Array.from(out.shape), [4]);
  assert.deepEqual(Array.from(out.data), [0, 1, 2, 3]);
});

test('indicesof: empty vector returns empty index Value', () => {
  const ir = call('indicesof', [call('vector', [])]);
  const out = sampler.evaluateExpr(ir, {});
  assert.deepEqual(Array.from(out.shape), [0]);
  assert.equal(out.data.length, 0);
});

test('indicesof: rank-2 array yields tuple of per-axis index Values', () => {
  // 2×3 matrix → tuple([1,2], [1,2,3]). Synthesise a rank-2 Value
  // directly (rowstack takes a single nested-array arg; easier to
  // hand-construct the shape-tagged Value for a focused test).
  const v: any = {
    shape: [2, 3],
    data: new Float64Array([1, 2, 3, 4, 5, 6]),
  };
  const env = { M: v };
  const ir = call('indicesof', [{ kind: 'ref', ns: 'self', name: 'M', loc: synthLoc() }]);
  const out = sampler.evaluateExpr(ir, env);
  assert.ok(Array.isArray(out),
    'rank-2 indicesof returns a per-axis tuple, got ' + JSON.stringify(out));
  assert.equal(out.length, 2);
  assert.deepEqual(Array.from(out[0].data), [1, 2]);
  assert.deepEqual(Array.from(out[1].data), [1, 2, 3]);
});

test('indicesof0: motivating use case — chebyshev_series-style index broadcast', () => {
  // Verifies that `indicesof0(coeffs)` produces an Int-valued vector
  // suitable for index-driven broadcasts. We check the values
  // directly here (the broadcast itself is exercised in
  // broadcast tests).
  const coeffs = call('vector', [lit(1.0), lit(0.5), lit(-0.25), lit(0.1)]);
  const ir = call('indicesof0', [coeffs]);
  const out = sampler.evaluateExpr(ir, {});
  assert.deepEqual(Array.from(out.data), [0, 1, 2, 3]);
});

test('tuple_get: applied to a rand result extracts the value slot', () => {
  const init = rng.seedFromBytes([99]);
  const measureIR = call('Normal', [], { mu: lit(0), sigma: lit(1) });
  const randCall = call('rand', [
    { kind: 'ref', ns: 'self', name: 'rs', loc: synthLoc() }, measureIR]);
  const valueSlot = call('tuple_get', [randCall, lit(0)]);
  const stateSlot = call('tuple_get', [randCall, lit(1)]);
  const env = { rs: init };
  // Independent calls to rand both start from `rs` so the sample
  // values match — same seed, same draw — and the trailing state
  // matches too.
  const v = sampler.evaluateExpr(valueSlot, env);
  const s = sampler.evaluateExpr(stateSlot, env);
  assert.equal(typeof v, 'number');
  assert.ok(s && s.key);
});
