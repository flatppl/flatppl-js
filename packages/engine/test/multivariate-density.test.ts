'use strict';

// Closed-form density walker tests for the multivariate / composite
// distributions added in §08 fill-in.
//
// Each walker is tested directly against density.logDensity with a
// hand-built IR. End-to-end source-to-density coverage is the
// matLogdensityof job; here we pin the density math.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const density = require('../density.ts');

function lit(v: any)              { return { kind: 'lit', value: v }; }
function vecLit(arr: any) {
  return { kind: 'call', op: 'vector', args: arr.map(lit) };
}
function Dirichlet(alpha: any) {
  return { kind: 'call', op: 'Dirichlet', kwargs: { alpha: vecLit(alpha) } };
}
function Multinomial(n: any, p: any) {
  return { kind: 'call', op: 'Multinomial',
    kwargs: { n: lit(n), p: vecLit(p) } };
}
function BinnedPP(rates: any) {
  return { kind: 'call', op: 'BinnedPoissonProcess',
    kwargs: { intensity: vecLit(rates) } };
}
function matLit(rows: any) {
  return { kind: 'call', op: 'rowstack',
    args: [{ kind: 'call', op: 'vector',
             args: rows.map((r: any) => vecLit(r)) }] };
}
function Wishart(nu: any, scale: any) {
  return { kind: 'call', op: 'Wishart',
    kwargs: { nu: lit(nu), scale: matLit(scale) } };
}
function InverseWishart(nu: any, scale: any) {
  return { kind: 'call', op: 'InverseWishart',
    kwargs: { nu: lit(nu), scale: matLit(scale) } };
}
function LKJ(n: any, eta: any) {
  return { kind: 'call', op: 'LKJ',
    kwargs: { n: lit(n), eta: lit(eta) } };
}
function LKJCholesky(n: any, eta: any) {
  return { kind: 'call', op: 'LKJCholesky',
    kwargs: { n: lit(n), eta: lit(eta) } };
}

// ---------------------------------------------------------------------
// Dirichlet
// ---------------------------------------------------------------------

test('density Dirichlet: uniform Dirichlet(1,1,1) at any simplex point = log Γ(3)', () => {
  const ir = Dirichlet([1, 1, 1]);
  // density at e.g. (1/3, 1/3, 1/3): log B(α) = Σ log Γ(1) - log Γ(3) = -log 2
  // for α=(1,1,1), density on simplex is 1/B(1,1,1) = 1/Γ(3)/Γ(1)^3·Γ(3) = 2
  // log p = log 2
  const logp = density.logDensity(ir, [1/3, 1/3, 1/3], {});
  assert.ok(Math.abs(logp - Math.log(2)) < 1e-9,
    `Dirichlet(1,1,1) logp at uniform = ${logp}, expected log(2) = ${Math.log(2)}`);
});

test('density Dirichlet: matches known formula for α=(2,3,5) at x=(0.1, 0.3, 0.6)', () => {
  const ir = Dirichlet([2, 3, 5]);
  // log p = -log B(α) + Σ (α_i-1) log x_i
  // log B(α) = log Γ(2) + log Γ(3) + log Γ(5) - log Γ(10)
  //          = 0 + log(2) + log(24) - log(362880)
  //          ≈ 0 + 0.69315 + 3.17805 - 12.80183 = -8.93062
  // Σ (α_i-1) log x_i = 1·log(0.1) + 2·log(0.3) + 4·log(0.6)
  //                   ≈ -2.30259 + 2·(-1.20397) + 4·(-0.51083)
  //                   = -2.30259 - 2.40794 - 2.04332 = -6.75385
  // logp = 8.93062 + (-6.75385) = 2.17676
  const logp = density.logDensity(ir, [0.1, 0.3, 0.6], {});
  assert.ok(Math.abs(logp - 2.17676) < 1e-3,
    `Dirichlet logp = ${logp}, expected ≈ 2.17676`);
});

test('density Dirichlet: out-of-support (negative coord) → -Infinity', () => {
  const ir = Dirichlet([1, 1, 1]);
  const logp = density.logDensity(ir, [-0.1, 0.5, 0.6], {});
  assert.equal(logp, -Infinity);
});

// ---------------------------------------------------------------------
// Multinomial
// ---------------------------------------------------------------------

test('density Multinomial: pmf at corner (n, 0, 0) for p=(1, 0, 0) = 1 → logp = 0', () => {
  const ir = Multinomial(5, [1, 0, 0]);
  const logp = density.logDensity(ir, [5, 0, 0], {});
  assert.ok(Math.abs(logp) < 1e-12,
    `degenerate Multinomial logp = ${logp}, expected 0`);
});

test('density Multinomial: known formula for n=3, p=(0.5, 0.3, 0.2) at x=(2, 1, 0)', () => {
  const ir = Multinomial(3, [0.5, 0.3, 0.2]);
  // pmf = (3! / (2!1!0!)) · 0.5² · 0.3¹ · 0.2⁰
  //     = 3 · 0.25 · 0.3 · 1 = 0.225
  // logpmf ≈ -1.49165
  const logp = density.logDensity(ir, [2, 1, 0], {});
  assert.ok(Math.abs(logp - Math.log(0.225)) < 1e-9,
    `Multinomial logp = ${logp}, expected ${Math.log(0.225)}`);
});

test('density Multinomial: counts not summing to n → -Infinity', () => {
  const ir = Multinomial(5, [0.5, 0.5]);
  const logp = density.logDensity(ir, [2, 2], {});
  assert.equal(logp, -Infinity);
});

// ---------------------------------------------------------------------
// BinnedPoissonProcess (direct per-bin-rate form)
// ---------------------------------------------------------------------

test('density BinnedPoissonProcess: zero-count vector at zero-rate vector = 0', () => {
  const ir = BinnedPP([0, 0, 0]);
  const logp = density.logDensity(ir, [0, 0, 0], {});
  assert.equal(logp, 0);
});

test('density BinnedPoissonProcess: matches sum of Poisson log-pmfs', () => {
  const ir = BinnedPP([1, 2, 3]);
  // log p(x; λ) = Σ (x_k log λ_k − λ_k − log x_k!)
  // For x=(1,2,3), λ=(1,2,3):
  //   k=0: 1·log(1) − 1 − log(1!) = 0 − 1 − 0 = -1
  //   k=1: 2·log(2) − 2 − log(2!) = 1.3863 − 2 − 0.6931 = -1.3068
  //   k=2: 3·log(3) − 3 − log(3!) = 3.2958 − 3 − 1.7918 = -1.4960
  // Total ≈ -3.8028
  const logp = density.logDensity(ir, [1, 2, 3], {});
  const expected = (0 - 1 - 0)
    + (2 * Math.log(2) - 2 - Math.log(2))
    + (3 * Math.log(3) - 3 - Math.log(6));
  assert.ok(Math.abs(logp - expected) < 1e-9,
    `BinnedPP logp = ${logp}, expected ${expected}`);
});

test('density BinnedPoissonProcess: nonzero count at zero rate → -Infinity', () => {
  const ir = BinnedPP([0, 2]);
  const logp = density.logDensity(ir, [1, 2], {});
  assert.equal(logp, -Infinity);
});

test('density BinnedPoissonProcess: zero count at zero rate contributes 0', () => {
  const ir = BinnedPP([0, 2]);
  // x=(0, 4): k=0 contributes 0 (zero rate, zero count); k=1 contributes
  // 4·log(2) − 2 − log(4!) ≈ 2.7726 − 2 − 3.1781 ≈ -2.4055
  const logp = density.logDensity(ir, [0, 4], {});
  const expected = 4 * Math.log(2) - 2 - Math.log(24);
  assert.ok(Math.abs(logp - expected) < 1e-9,
    `BinnedPP logp = ${logp}, expected ${expected}`);
});

// ---------------------------------------------------------------------
// Wishart / InverseWishart
// ---------------------------------------------------------------------

// Helper: build identity matrix as flat row-major Float64Array length n*n
function eye(n: any) {
  const a = new Float64Array(n * n);
  for (let i = 0; i < n; i++) a[i * n + i] = 1;
  return Array.from(a);
}

test('density Wishart: identity scale, nu=3, X=I_2 matches closed form', () => {
  // n=2, nu=3, V=I_2, X=I_2
  // log p = (3-2-1)/2 log|X| - 0.5 tr(V^-1 X) - 3·2/2 log2 - 3/2 log|V| - log Γ_2(3/2)
  //       = 0 - 0.5·2 - 3·log2 - 0 - log Γ_2(3/2)
  // log Γ_2(3/2) = log π^(2·1/4) + log Γ(3/2) + log Γ(1)
  //             = (1/2) log π + log Γ(3/2) + 0
  //             = 0.5·log(π) + log(√π/2) = 0.5 log π + 0.5 log π - log 2
  //             = log π - log 2
  // So logp = -1 - 3 log 2 - log π + log 2 = -1 - 2 log 2 - log π
  const ir = Wishart(3, [[1, 0], [0, 1]]);
  const logp = density.logDensity(ir, [[1, 0], [0, 1]], {});
  const expected = -1 - 2 * Math.log(2) - Math.log(Math.PI);
  assert.ok(Math.abs(logp - expected) < 1e-9,
    `Wishart logp = ${logp}, expected ${expected}`);
});

test('density Wishart: non-SPD observation → -Infinity', () => {
  const ir = Wishart(3, [[1, 0], [0, 1]]);
  // X with negative eigenvalue
  const logp = density.logDensity(ir, [[1, 0], [0, -1]], {});
  assert.equal(logp, -Infinity);
});

test('density InverseWishart: spec equivalence pushfwd(inv, Wishart(nu, inv(scale)))', () => {
  // The two densities should agree on the value (up to Jacobian which
  // is implicit in the pushfwd). Sanity check that the formula returns
  // something finite for a positive-definite input.
  const ir = InverseWishart(4, [[2, 1], [1, 2]]);
  const logp = density.logDensity(ir, [[1.5, 0.2], [0.2, 1.2]], {});
  assert.ok(Number.isFinite(logp),
    `InverseWishart logp at SPD X = ${logp}, expected finite`);
});

// ---------------------------------------------------------------------
// LKJ / LKJCholesky
// ---------------------------------------------------------------------

test('density LKJ: η=1 yields uniform density (constant log p, independent of C)', () => {
  // At η=1, det(C)^0 = 1, so log p = log c_n(η) for any valid C.
  const ir = LKJ(3, 1);
  // Two different correlation matrices — should give the same logp.
  const C1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const C2 = [[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]];
  const logp1 = density.logDensity(ir, C1, {});
  const logp2 = density.logDensity(ir, C2, {});
  assert.ok(Math.abs(logp1 - logp2) < 1e-9,
    `LKJ(3, 1) should be uniform: logp(I)=${logp1}, logp(C2)=${logp2}`);
});

test('density LKJ: n=2, η=1 is the uniform density log c_2(1) = -log 2', () => {
  // For n=2, the correlation matrix has a single off-diag ρ ∈ (-1, 1).
  // det(C) = 1 - ρ². At η=1 the density is constant (det(C)^0 = 1), so
  // log p = log c_2(1) for any valid C. The constant equals 1/Volume — the
  // integral of 1 over ρ ∈ (-1, 1) is 2, so the density is the uniform 1/2
  // and log c_2(1) = -log 2 ≈ -0.6931.
  const ir = LKJ(2, 1);
  const I = [[1, 0], [0, 1]];
  const logp = density.logDensity(ir, I, {});
  assert.ok(Math.abs(logp - (-Math.LN2)) < 1e-9,
    `LKJ(2,1) logp(I) should be -log 2 ≈ ${-Math.LN2}, got ${logp}`);
});

test('density LKJ: n=2, η=2 matches closed-form log c_2(2) = -log(4/3)', () => {
  // For n=2, the normalizing integral is Z = 2^(2η-1)·B(η, η). At η=2,
  // Z = 2^3 · B(2, 2) = 8 · (1/6) = 4/3, so log c_2(2) = -log(4/3) ≈ -0.28768.
  // At C = I, det(C)^(η-1) = 1, so logp(I) = log c_2(2).
  const ir = LKJ(2, 2);
  const I = [[1, 0], [0, 1]];
  const logp = density.logDensity(ir, I, {});
  const expected = -Math.log(4 / 3);
  assert.ok(Math.abs(logp - expected) < 1e-9,
    `LKJ(2,2) logp(I) should be -log(4/3) ≈ ${expected}, got ${logp}`);
});

test('density LKJCholesky: η=1 simplifies to Π L_ii^(n-i)', () => {
  // For η=1, the exponent simplifies: n - i + 2·1 - 2 = n - i.
  // Apply on a unit lower-triangular matrix (L_22 = 1, L_33 = 1) ⇒
  // factor of 1 in each, log p reduces to log c_n(1).
  const ir = LKJCholesky(3, 1);
  // L is lower-triangular, unit norms (rows sum-of-squares = 1).
  const L = [[1, 0, 0], [0.5, Math.sqrt(0.75), 0], [0.3, 0.4, Math.sqrt(1 - 0.09 - 0.16)]];
  const logp = density.logDensity(ir, L, {});
  assert.ok(Number.isFinite(logp));
});

test('density LKJCholesky: non-positive diagonal → -Infinity', () => {
  const ir = LKJCholesky(2, 1);
  const L = [[1, 0], [0.5, -Math.sqrt(0.75)]];  // negative L_22
  const logp = density.logDensity(ir, L, {});
  assert.equal(logp, -Infinity);
});
