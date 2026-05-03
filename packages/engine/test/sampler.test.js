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

const sampler = require('../sampler');
const rng = require('../rng');

// Helper: build a distribution call IR from a name + kwargs (numeric values
// only, lifted to lit nodes).
function distIR(op, kwargs) {
  const out = {};
  for (const [k, v] of Object.entries(kwargs)) {
    out[k] = { kind: 'lit', value: v, loc: synthLoc() };
  }
  return { kind: 'call', op, kwargs: out, loc: synthLoc() };
}

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}

// Helper: ref node for env-resolved params.
function refIR(name) {
  return { kind: 'ref', ns: 'self', name, loc: synthLoc() };
}

// Helper: take N samples, return a Float64Array. Reseeds from same state
// every call so it's deterministic and repeatable.
function takeN(measureIR, env, n, seed = [1, 2, 3]) {
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
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
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
