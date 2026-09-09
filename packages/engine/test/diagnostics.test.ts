const { test } = require('node:test');
const assert = require('node:assert/strict');
const D = require('../diagnostics.ts');

test('splitRHat ~ 1 for well-mixed iid normal chains', () => {
  // Deterministic pseudo-normal via Box-Muller on a seeded LCG (test-local).
  let s = 12345;
  const u = () => (s = (1103515245 * s + 12345) % 2147483648) / 2147483648;
  const norm = () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
  const chains = [0, 1, 2, 3].map(() => Float64Array.from({ length: 2000 }, norm));
  const rhat = D.splitRHat(chains);
  assert.ok(rhat > 0.98 && rhat < 1.05, `rhat ${rhat} not ~1`);
});

test('splitRHat large for chains stuck at different locations', () => {
  const chains = [
    Float64Array.from({ length: 1000 }, () => 0),
    Float64Array.from({ length: 1000 }, () => 10),
  ];
  assert.ok(D.splitRHat(chains) > 2, 'separated chains -> large rhat');
});

test('essBulk <= total draws and positive for iid chains', () => {
  let s = 999;
  const u = () => (s = (1103515245 * s + 12345) % 2147483648) / 2147483648;
  const chains = [0, 1].map(() => Float64Array.from({ length: 1000 }, u));
  const ess = D.essBulk(chains);
  assert.ok(ess > 0 && ess <= 2000, `ess ${ess} out of range`);
  assert.ok(ess > 500, `iid ess ${ess} unexpectedly low`);
});

test('essBulk matches polynomial lag sums for long ramps at different scales', () => {
  const n = 1025, h = (n - 1) / 2;
  // Two identical unit-slope ramps. Finite sums of i and
  // i² give the autocovariance exactly, independently of direct/FFT code.
  const W = (n + 1) / (12 * n);
  const varPlus = (n - 1) / n * W;
  const rho = (t: number) => {
    const k = n - t;
    const s1 = k * (k - 1) / 2, s2 = k * (k - 1) * (2 * k - 1) / 6;
    const acov = (s2 + (t - 2 * h) * s1 + k * h * (h - t)) / n ** 3;
    return 1 - (W - acov) / varPlus;
  };
  let sum = 0;
  for (let t = 1; t < n - 1; t += 2) {
    const pair = rho(t) + rho(t + 1);
    if (pair < 0) break;
    sum += pair;
  }
  const expected = 2 * n / Math.max(1 + 2 * sum, 1 / Math.log10(2 * n));
  for (const scale of [1, 1e-100, 1e100]) {
    const chains = [0, 1].map(() =>
      Float64Array.from({ length: n }, (_, i) => scale * (i - h) / n));
    assert.ok(Math.abs(D.essBulk(chains) - expected) < 1e-10 * expected);
  }
});

test('essBulk retains direct finite sums when variance is subnormal', () => {
  const n = 1025;
  const chains = [-0.25, 0.25].map(o => Float64Array.from({ length: n },
    (_, i) => 1e-161 * ((i - (n - 1) / 2) / n + o)));
  const means = chains.map(c => c.reduce((a, x) => a + x, 0) / n);
  const grand = (means[0] + means[1]) / 2;
  const W = chains.reduce((a, c, k) => a + c.reduce((s, x) => s + (x - means[k]) ** 2, 0) / (n - 1), 0) / 2;
  const B = n * means.reduce((s, x) => s + (x - grand) ** 2, 0);
  const vp = (n - 1) / n * W + B / n;
  const rho = (t: number) => {
    const cov = chains.reduce((a, c, k) => a + c.subarray(0, n - t).reduce(
      (s, x, i) => s + (x - means[k]) * (c[i + t] - means[k]), 0) / n, 0) / 2;
    return 1 - (W - cov) / vp;
  };
  let sum = 0;
  for (let t = 1; t < n - 1; t += 2) {
    const pair = rho(t) + rho(t + 1);
    if (pair < 0) break;
    sum += pair;
  }
  // Compatibility with rounded finite sums, not a real-arithmetic ESS oracle:
  // rescaling around underflow must not silently change the estimator.
  assert.equal(D.essBulk(chains), 2 * n / Math.max(1 + 2 * sum, 1 / Math.log10(2 * n)));
});
