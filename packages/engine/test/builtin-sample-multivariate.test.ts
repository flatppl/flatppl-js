'use strict';

// builtin_sample for the §08 multivariate kernels without a scalar leaf
// sampler (spec §07 "Engine requirements": builtin_sample for every
// built-in kernel an engine supports).
//
// Oracles are closed forms, never the sampler's own construction:
//   - Dirichlet(a): mean a/a0, cov (a0·diag(a) - a·aᵀ)/(a0²(a0+1)), and the
//     first coordinate ~ Beta(a1, a0 - a1) (KS against its closed-form CDF).
//   - Multinomial(n, p): mean n·p, cov n(diag(p) - p·pᵀ).
//   - Wishart(nu, V): mean nu·V, Var(W_ij) = nu(V_ij² + V_ii·V_jj).
//   - InverseWishart(nu, Ψ): mean Ψ/(nu - n - 1),
//     Var(X_ii) = 2Ψ_ii²/((nu - n - 1)²(nu - n - 3)).
//   - LKJ(n, eta): (R_ij + 1)/2 ~ Beta(b, b), b = eta - 1 + n/2, so
//     E[R_ij²] = 1/(2b + 1) and E[R_ij⁴] = 3/((2b + 1)(2b + 3)).
//     LKJ(2, 1) has R_12 ~ Uniform(-1, 1).
//   - BinnedPoissonProcess: bin count ~ Poisson(intensity mass of the bin).
// Each moment check pairs with a support check.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const engine = require('../index.ts');
const orchestrator = require('../orchestrator.ts');

function fixed(src: string): Map<string, any> {
  const r = engine.processSource(src);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  const built = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const errs = (built.diagnostics || []).filter((d: any) => d.severity === 'error').map((d: any) => d.message);
  assert.deepEqual(errs, []);
  return built.fixedValues;
}

const flat = (v: any): number[] => Array.from(Array.isArray(v) ? v : v.data);

function atoms(X: any, len: number): number[][] {
  assert.ok(Array.isArray(X), 'builtin_sample with n returns an array of variates');
  return X.map((x: any) => {
    const a = flat(x);
    assert.equal(a.length, len);
    return a;
  });
}

function mean(rows: number[][], j: number): number {
  return rows.reduce((s, r) => s + r[j], 0) / rows.length;
}

function cov(rows: number[][], i: number, j: number): number {
  const mi = mean(rows, i), mj = mean(rows, j);
  return rows.reduce((s, r) => s + (r[i] - mi) * (r[j] - mj), 0) / (rows.length - 1);
}

// Standard error of the sample covariance, from the sample itself.
function covSE(rows: number[][], i: number, j: number): number {
  const mi = mean(rows, i), mj = mean(rows, j), c = cov(rows, i, j);
  const m = rows.reduce((s, r) => s + ((r[i] - mi) * (r[j] - mj)) ** 2, 0) / rows.length;
  return Math.sqrt((m - c * c) / rows.length);
}

// |estimate - truth| within k standard errors.
function near(est: number, truth: number, se: number, what: string, k = 5) {
  assert.ok(Math.abs(est - truth) < k * se, `${what}: ${est}, want ${truth} ± ${k}·${se}`);
}

function ksAgainst(xs: number[], cdf: (x: number) => number): number {
  const s = xs.slice().sort((a, b) => a - b);
  let d = 0;
  for (let i = 0; i < s.length; i++) {
    const f = cdf(s[i]);
    d = Math.max(d, Math.abs(f - i / s.length), Math.abs(f - (i + 1) / s.length));
  }
  return d;
}

// Cholesky succeeds iff the symmetric matrix is positive definite.
function isSymmetricPD(a: number[], n: number): boolean {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      if (Math.abs(a[i * n + j] - a[j * n + i]) > 1e-9 * (1 + Math.abs(a[i * n + j]))) return false;
    }
  }
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (!(s > 0)) return false;
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  return true;
}

function stateAdvancesAndChains(kernel: string, input: string) {
  const fv = fixed(`
s0 = rnginit([3, 1, 4, 1])
x1, s1 = builtin_sample(s0, ${kernel}, ${input})
x2, s2 = builtin_sample(s1, ${kernel}, ${input})
xs, s3 = builtin_sample(s0, ${kernel}, ${input}, 2)
`);
  assert.notDeepEqual(fv.get('s1'), fv.get('s0'), `${kernel}: state advances`);
  assert.notDeepEqual(fv.get('s2'), fv.get('s1'), `${kernel}: chained state advances`);
  assert.notDeepEqual(flat(fv.get('x2')), flat(fv.get('x1')), `${kernel}: chained draws differ`);
  // IID draws thread one state: atom 1 of n draws from s0 is the single draw from s0.
  const xs = fv.get('xs');
  assert.deepEqual(flat(xs[0]), flat(fv.get('x1')));
  assert.deepEqual(flat(xs[1]), flat(fv.get('x2')));
  assert.deepEqual(fv.get('s3'), fv.get('s2'));
}

test('builtin_sample(Dirichlet): simplex support, closed-form moments and Beta marginal', () => {
  const N = 20000;
  const a = [1.0, 2.0, 3.0];
  const a0 = 6;
  const fv = fixed(`
s0 = rnginit([11, 12, 13, 14])
X, s1 = builtin_sample(s0, Dirichlet, record(alpha = [1.0, 2.0, 3.0]), ${N})
`);
  const rows = atoms(fv.get('X'), 3);
  for (const r of rows) {
    assert.ok(r.every((v) => v > 0 && v < 1), 'coordinates in (0, 1)');
    assert.ok(Math.abs(r[0] + r[1] + r[2] - 1) < 1e-12, 'sums to 1');
  }
  const c = a0 * a0 * (a0 + 1);
  for (let i = 0; i < 3; i++) {
    const v = a[i] * (a0 - a[i]) / c;
    near(mean(rows, i), a[i] / a0, Math.sqrt(v / N), `mean[${i}]`);
    for (let j = 0; j < i; j++) {
      near(cov(rows, i, j), -a[i] * a[j] / c, covSE(rows, i, j), `cov[${i}][${j}]`);
    }
  }
  // x_1 ~ Beta(1, 5): F(x) = 1 - (1 - x)^5. 1.95/sqrt(N) is the 0.1 % KS value.
  const d = ksAgainst(rows.map((r) => r[0]), (x) => 1 - Math.pow(1 - x, 5));
  assert.ok(d < 1.95 / Math.sqrt(N), `KS distance ${d}`);
  assert.notDeepEqual(fv.get('s1'), fv.get('s0'));
});

test('builtin_sample(Multinomial): count support and closed-form moments, small and large n', () => {
  const N = 20000;
  const p = [0.2, 0.3, 0.5];
  for (const n of [7, 1000]) {
    const fv = fixed(`
s0 = rnginit([21, 22, 23, 24])
X, _ = builtin_sample(s0, Multinomial, record(n = ${n}, p = [0.2, 0.3, 0.5]), ${N})
`);
    const rows = atoms(fv.get('X'), 3);
    for (const r of rows) {
      assert.ok(r.every((v) => Number.isInteger(v) && v >= 0), 'non-negative integer counts');
      assert.equal(r[0] + r[1] + r[2], n, 'counts sum to n');
    }
    for (let i = 0; i < 3; i++) {
      const v = n * p[i] * (1 - p[i]);
      near(mean(rows, i), n * p[i], Math.sqrt(v / N), `n=${n} mean[${i}]`);
      for (let j = 0; j < i; j++) {
        near(cov(rows, i, j), -n * p[i] * p[j], covSE(rows, i, j), `n=${n} cov[${i}][${j}]`);
      }
    }
  }
});

test('builtin_sample(Multinomial): a zero-probability category stays empty', () => {
  const fv = fixed(`
s0 = rnginit([5, 6, 7, 8])
X, _ = builtin_sample(s0, Multinomial, record(n = 9, p = [0.5, 0.0, 0.5]), 200)
`);
  for (const r of atoms(fv.get('X'), 3)) {
    assert.equal(r[1], 0);
    assert.equal(r[0] + r[2], 9);
  }
});

test('builtin_sample(Wishart): positive-definite support and closed-form moments', () => {
  const N = 20000;
  const nu = 5;
  const V = [[2.0, 0.5], [0.5, 1.0]];
  const fv = fixed(`
s0 = rnginit([31, 32, 33, 34])
X, _ = builtin_sample(s0, Wishart, record(nu = 5.0, scale = rowstack([[2.0, 0.5], [0.5, 1.0]])), ${N})
`);
  const rows = atoms(fv.get('X'), 4);
  for (const r of rows) assert.ok(isSymmetricPD(r, 2), 'symmetric positive definite');
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const v = nu * (V[i][j] ** 2 + V[i][i] * V[j][j]);
      near(mean(rows, 2 * i + j), nu * V[i][j], Math.sqrt(v / N), `mean[${i}][${j}]`);
      const k = 2 * i + j;
      near(cov(rows, k, k), v, covSE(rows, k, k), `var[${i}][${j}]`);
    }
  }
});

test('builtin_sample(InverseWishart): positive-definite support and closed-form moments', () => {
  const N = 20000;
  const nu = 10, n = 2;
  const P = [[2.0, 0.5], [0.5, 1.0]];
  const fv = fixed(`
s0 = rnginit([41, 42, 43, 44])
X, _ = builtin_sample(s0, InverseWishart, record(nu = 10.0, scale = rowstack([[2.0, 0.5], [0.5, 1.0]])), ${N})
`);
  const rows = atoms(fv.get('X'), 4);
  for (const r of rows) assert.ok(isSymmetricPD(r, 2), 'symmetric positive definite');
  const k = nu - n - 1;
  for (let i = 0; i < 2; i++) {
    const v = 2 * P[i][i] ** 2 / (k * k * (nu - n - 3));
    near(mean(rows, 3 * i), P[i][i] / k, Math.sqrt(v / N), `mean[${i}][${i}]`);
    near(cov(rows, 3 * i, 3 * i), v, covSE(rows, 3 * i, 3 * i), `var[${i}][${i}]`);
  }
  // Off-diagonal mean only: its variance is finite but has no simpler form here.
  const m01 = mean(rows, 1);
  near(m01, P[0][1] / k, Math.sqrt(cov(rows, 1, 1) / N), 'mean[0][1]');
});

test('builtin_sample(LKJ): correlation-matrix support and the Beta marginal law', () => {
  const N = 20000;
  const n = 4, eta = 0.7;
  const b = eta - 1 + n / 2;
  const fv = fixed(`
s0 = rnginit([51, 52, 53, 54])
X, _ = builtin_sample(s0, LKJ, record(n = 4, eta = 0.7), ${N})
`);
  const rows = atoms(fv.get('X'), 16);
  for (const r of rows) {
    for (let i = 0; i < n; i++) assert.ok(Math.abs(r[i * n + i] - 1) < 1e-12, 'unit diagonal');
    assert.ok(isSymmetricPD(r, n), 'symmetric positive definite');
  }
  const m2 = 1 / (2 * b + 1);
  const m4 = 3 / ((2 * b + 1) * (2 * b + 3));
  for (const [i, j] of [[1, 0], [2, 0], [3, 2]]) {
    const xs = rows.map((r) => r[i * n + j]);
    near(xs.reduce((s, x) => s + x, 0) / N, 0, Math.sqrt(m2 / N), `mean R[${i}][${j}]`);
    const e2 = xs.reduce((s, x) => s + x * x, 0) / N;
    near(e2, m2, Math.sqrt((m4 - m2 * m2) / N), `E[R[${i}][${j}]^2]`);
  }
});

test('builtin_sample(LKJ(2, 1)): the correlation is Uniform(-1, 1)', () => {
  const N = 20000;
  const fv = fixed(`
s0 = rnginit([55, 56, 57, 58])
X, _ = builtin_sample(s0, LKJ, record(n = 2, eta = 1.0), ${N})
`);
  const rs = atoms(fv.get('X'), 4).map((r) => r[1]);
  const d = ksAgainst(rs, (x) => (x + 1) / 2);
  assert.ok(d < 1.95 / Math.sqrt(N), `KS distance ${d}`);
});

test('builtin_sample(LKJCholesky): triangular unit-row support and the Beta marginal law', () => {
  const N = 20000;
  const n = 3, eta = 2.0;
  const b = eta - 1 + n / 2;
  const fv = fixed(`
s0 = rnginit([61, 62, 63, 64])
X, _ = builtin_sample(s0, LKJCholesky, record(n = 3, eta = 2.0), ${N})
`);
  const rows = atoms(fv.get('X'), 9);
  const r21: number[] = [];
  for (const L of rows) {
    for (let i = 0; i < n; i++) {
      assert.ok(L[i * n + i] > 0, 'positive diagonal');
      let norm2 = 0;
      for (let j = 0; j < n; j++) {
        if (j > i) assert.equal(L[i * n + j], 0, 'lower triangular');
        norm2 += L[i * n + j] ** 2;
      }
      assert.ok(Math.abs(norm2 - 1) < 1e-12, 'unit-norm row');
    }
    // R = L·Lᵀ, entry (3, 2).
    r21.push(L[2 * n] * L[n] + L[2 * n + 1] * L[n + 1]);
  }
  const m2 = 1 / (2 * b + 1);
  const m4 = 3 / ((2 * b + 1) * (2 * b + 3));
  const e2 = r21.reduce((s, x) => s + x * x, 0) / N;
  near(e2, m2, Math.sqrt((m4 - m2 * m2) / N), 'E[R32^2]');
});

test('builtin_sample(BinnedPoissonProcess): bin counts follow the bin rates', () => {
  const N = 20000;
  const rates = [3.0, 0.0, 12.5];
  const fv = fixed(`
s0 = rnginit([71, 72, 73, 74])
X, _ = builtin_sample(s0, BinnedPoissonProcess, record(rates = [3.0, 0.0, 12.5]), ${N})
`);
  const rows = atoms(fv.get('X'), 3);
  for (const r of rows) assert.ok(r.every((v) => Number.isInteger(v) && v >= 0), 'counts');
  for (let k = 0; k < 3; k++) {
    if (rates[k] === 0) {
      assert.ok(rows.every((r) => r[k] === 0), 'zero-rate bin stays empty');
      continue;
    }
    near(mean(rows, k), rates[k], Math.sqrt(rates[k] / N), `mean[${k}]`);
    near(cov(rows, k, k), rates[k], covSE(rows, k, k), `var[${k}]`);
  }
});

test('builtin_sample: every multivariate kernel threads and chains the RNG state', () => {
  const cases: [string, string][] = [
    ['MvNormal', 'record(mu = [0.0, 1.0], cov = rowstack([[1.0, 0.2], [0.2, 2.0]]))'],
    ['Dirichlet', 'record(alpha = [0.5, 1.5, 2.0])'],
    ['Multinomial', 'record(n = 20, p = [0.25, 0.25, 0.5])'],
    ['Wishart', 'record(nu = 4.0, scale = rowstack([[1.0, 0.0], [0.0, 1.0]]))'],
    ['InverseWishart', 'record(nu = 5.0, scale = rowstack([[1.0, 0.0], [0.0, 1.0]]))'],
    ['LKJ', 'record(n = 3, eta = 1.5)'],
    ['LKJCholesky', 'record(n = 3, eta = 1.5)'],
    ['BinnedPoissonProcess', 'record(rates = [4.0, 9.0])'],
  ];
  for (const [kernel, input] of cases) stateAdvancesAndChains(kernel, input);
});

test('builtin_sample(K) and rand(K(...), s) draw the same variate', () => {
  const fv = fixed(`
s0 = rnginit([1, 2, 3, 4])
a, sa = builtin_sample(s0, Wishart, record(nu = 4.0, scale = rowstack([[1.0, 0.3], [0.3, 1.0]])))
b, sb = rand(Wishart(nu = 4.0, scale = rowstack([[1.0, 0.3], [0.3, 1.0]])), s0)
c, sc = builtin_sample(s0, Dirichlet, record(alpha = [1.0, 2.0]))
d, sd = rand(Dirichlet([1.0, 2.0]), s0)
`);
  assert.deepEqual(flat(fv.get('a')), flat(fv.get('b')));
  assert.deepEqual(fv.get('sa'), fv.get('sb'));
  assert.deepEqual(flat(fv.get('c')), flat(fv.get('d')));
  assert.deepEqual(fv.get('sc'), fv.get('sd'));
});

test('value-position rand(BinnedPoissonProcess(bins, intensity)) uses the bin masses', () => {
  // weighted(10, Uniform(0, 2)) puts mass 5 in each unit bin;
  // weighted(4, Exponential(1)) puts 4(1 - e^-0.5) in [0, 0.5].
  const N = 5000;
  const fv = fixed(`
s0 = rnginit([81, 82, 83, 84])
lam = weighted(10.0, Uniform(interval(0.0, 2.0)))
X, s1 = rand(iid(BinnedPoissonProcess(bins = [0.0, 1.0, 2.0], intensity = lam), ${N}), s0)
Y, _ = rand(iid(BinnedPoissonProcess(bins = [0.0, 0.5, 2.0], intensity = weighted(4.0, Exponential(rate = 1.0))), ${N}), s1)
`);
  const xs = atoms(fv.get('X'), 2);
  for (let k = 0; k < 2; k++) near(mean(xs, k), 5, Math.sqrt(5 / N), `X mean[${k}]`);
  const ys = atoms(fv.get('Y'), 2);
  const r = [4 * (1 - Math.exp(-0.5)), 4 * (Math.exp(-0.5) - Math.exp(-2))];
  for (let k = 0; k < 2; k++) near(mean(ys, k), r[k], Math.sqrt(r[k] / N), `Y mean[${k}]`);
});
