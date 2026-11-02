'use strict';

// Value-position rand over pushfwd (spec §06: pushfwd(f, M) samples f(x),
// x ~ M). Lift rewrites MvNormal and matrix locscale to
// pushfwd(<registry bijection>, iid(Normal(0, 1), n)), so a native
// `x, s = rand(MvNormal(...), s0)` reaches the pushfwd walker.
//
// Oracles are closed forms: MvNormal(mu, C) and locscale(iid(Normal), b, L)
// have mean mu (b) and covariance C (L·Lᵀ); 2·Exponential(1) has mean 2 and
// variance 4; exp of Normal(0, 1) has mean e^½ and variance (e - 1)e.

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
const N = 20000;

function moments(rows: number[][]) {
  const d = rows[0].length;
  const mean = Array.from({ length: d }, (_, j) => rows.reduce((s, r) => s + r[j], 0) / rows.length);
  const cov = (i: number, j: number) =>
    rows.reduce((s, r) => s + (r[i] - mean[i]) * (r[j] - mean[j]), 0) / (rows.length - 1);
  const covSE = (i: number, j: number) => {
    const c = cov(i, j);
    const m = rows.reduce((s, r) => s + ((r[i] - mean[i]) * (r[j] - mean[j])) ** 2, 0) / rows.length;
    return Math.sqrt((m - c * c) / rows.length);
  };
  return { mean, cov, covSE };
}

function checkGaussian(rows: number[][], mu: number[], C: number[][]) {
  const { mean, cov, covSE } = moments(rows);
  for (let i = 0; i < mu.length; i++) {
    const se = Math.sqrt(C[i][i] / rows.length);
    assert.ok(Math.abs(mean[i] - mu[i]) < 5 * se, `mean[${i}] = ${mean[i]}, want ${mu[i]}`);
    for (let j = 0; j <= i; j++) {
      assert.ok(Math.abs(cov(i, j) - C[i][j]) < 5 * covSE(i, j),
        `cov[${i}][${j}] = ${cov(i, j)}, want ${C[i][j]}`);
    }
  }
}

test('native rand(iid(MvNormal(mu, cov), n)) has the spec mean and covariance', () => {
  const fv = fixed(`
s0 = rnginit([9, 8, 7, 6])
mu = [1.0, -2.0]
C = rowstack([[2.0, -0.6], [-0.6, 0.5]])
X, s1 = rand(iid(MvNormal(mu = mu, cov = C), ${N}), s0)
`);
  const X = fv.get('X');
  assert.equal(X.length, N);
  checkGaussian(X.map(flat), [1, -2], [[2, -0.6], [-0.6, 0.5]]);
  assert.notDeepEqual(fv.get('s1'), fv.get('s0'));
});

test('native rand(MvNormal) matches builtin_sample(MvNormal) and threads the state', () => {
  const fv = fixed(`
s0 = rnginit([9, 8, 7, 6])
C = rowstack([[2.0, -0.6], [-0.6, 0.5]])
x, s1 = rand(MvNormal([1.0, -2.0], C), s0)
y, s2 = builtin_sample(s0, MvNormal, record(mu = [1.0, -2.0], cov = C))
z, s3 = rand(MvNormal([1.0, -2.0], C), s1)
`);
  const x = flat(fv.get('x')), y = flat(fv.get('y'));
  assert.equal(x.length, 2);
  for (let i = 0; i < 2; i++) assert.ok(Math.abs(x[i] - y[i]) < 1e-14, `coord ${i}: ${x[i]} vs ${y[i]}`);
  assert.deepEqual(fv.get('s1'), fv.get('s2'));
  assert.notDeepEqual(fv.get('s1'), fv.get('s0'));
  assert.notDeepEqual(flat(fv.get('z')), x);
  assert.notDeepEqual(fv.get('s3'), fv.get('s1'));
});

test('rand(locscale(iid(Normal), b, L)) with a matrix scale has mean b and covariance L·Lᵀ', () => {
  const fv = fixed(`
s0 = rnginit([5, 4, 3, 2])
L = rowstack([[1.0, 0.0], [0.5, 2.0]])
Y, _ = rand(iid(locscale(iid(Normal(0.0, 1.0), 2), [3.0, 4.0], L), ${N}), s0)
`);
  checkGaussian(fv.get('Y').map(flat), [3, 4], [[1, 0.5], [0.5, 4.25]]);
});

test('rand(pushfwd(f, M)) applies a user bijection and a builtin function', () => {
  const fv = fixed(`
s0 = rnginit([1, 3, 5, 7])
b = bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0))
U, s1 = rand(iid(pushfwd(b, Exponential(rate = 1.0)), ${N}), s0)
Q, _ = rand(iid(pushfwd(exp, Normal(0.0, 1.0)), ${N}), s1)
`);
  const u = flat(fv.get('U'));
  assert.ok(u.every((v) => v > 0));
  const { mean: mu, cov: cu, covSE: su } = moments(u.map((v) => [v]));
  assert.ok(Math.abs(mu[0] - 2) < 5 * Math.sqrt(4 / N), `2·Exp mean ${mu[0]}`);
  assert.ok(Math.abs(cu(0, 0) - 4) < 5 * su(0, 0), `2·Exp variance ${cu(0, 0)}`);
  const q = flat(fv.get('Q'));
  assert.ok(q.every((v) => v > 0));
  const vq = (Math.E - 1) * Math.E;
  const { mean: mq } = moments(q.map((v) => [v]));
  assert.ok(Math.abs(mq[0] - Math.sqrt(Math.E)) < 5 * Math.sqrt(vq / N), `LogNormal mean ${mq[0]}`);
});
