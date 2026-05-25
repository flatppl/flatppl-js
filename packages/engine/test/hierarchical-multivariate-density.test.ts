'use strict';

// =====================================================================
// Per-atom (hierarchical-prior) multivariate density evaluation.
// =====================================================================
//
// The pre-FlatPDL per-kernel walkers (walkMvNormal, walkDirichlet, …)
// all assumed atom-independent kwargs and broadcast a single log-density
// across the N-atom accumulator. That made hierarchical priors over
// multivariate kernels — e.g. `Sigma ~ Wishart(...); X ~ MvNormal(mu,
// Sigma)` — silently scored against the wrong Σ. The FlatPDL refactor
// (engine-concepts §13.6) routes density.ts's multivariate dispatch
// through a single `walkMultivariate` that resolves kwargs per atom when
// any kwarg expression references a per-atom name. These tests pin that
// behaviour for representative kernels: a hierarchical mu-Normal, a
// hierarchical alpha-Dirichlet, and a hierarchical scale-Wishart.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const density = require('../density.ts');
const densityPrims = require('../density-prims.ts');

// Convenience IR constructors.
function lit(v: any)        { return { kind: 'lit', value: v }; }
function refSelf(name: any) { return { kind: 'ref', ns: 'self', name }; }
function arrayIR(elems: any[]) {
  return { kind: 'call', op: 'vector', args: elems };
}
function rowstack(rows: any[]) {
  // rowstack takes ONE vector-of-vectors argument; wrap the rows in
  // an outer vector(...) so the ARITH_OPS spread doesn't drop the tail.
  return { kind: 'call', op: 'rowstack',
    args: [{ kind: 'call', op: 'vector', args: rows }] };
}

// =====================================================================
// MvNormal: per-atom mu
// =====================================================================
//
// Per-atom mu = [mu0_i, 0] with mu0 supplied via refArrays. The variate
// is a length-2 vector x shared across atoms. For each atom i:
//   logp_i = MvNormal({mu=[mu0_i, 0], cov=I}, x)
// We check against `builtinLogdensityof` called atom-by-atom.

test('walkMultivariate: per-atom mu in MvNormal evaluates per atom', () => {
  const N = 4;
  const mu0 = new Float64Array([0.0, 1.0, -1.0, 2.5]);
  // mu = [mu0, 0]: rank-1 vector of length 2.
  const muIR = arrayIR([refSelf('mu0'), lit(0.0)]);
  const covIR = rowstack([arrayIR([lit(1.0), lit(0.0)]),
                          arrayIR([lit(0.0), lit(1.0)])]);
  const ir = { kind: 'call', op: 'MvNormal',
               kwargs: { mu: muIR, cov: covIR } };
  const x = { shape: [2], data: new Float64Array([0.5, -0.5]) };
  const logps = density.logDensityN(ir, x, { mu0 }, N, {});
  for (let i = 0; i < N; i++) {
    const expected = densityPrims.builtinLogdensityof('MvNormal',
      { mu: [mu0[i], 0.0],
        cov: { shape: [2, 2], data: new Float64Array([1, 0, 0, 1]) } },
      x);
    assert.ok(Math.abs(logps[i] - expected) < 1e-12,
      `atom ${i}: got ${logps[i]}, expected ${expected}`);
  }
});

// =====================================================================
// MvNormal: per-atom cov via Sigma_i = diag(s_i, s_i) — a Σ ~ Wishart
// outer prior shape, evaluated against a shared variate. We bypass the
// Wishart sampler and supply Sigma diagonals directly as a per-atom ref.
// =====================================================================

test('walkMultivariate: per-atom cov in MvNormal evaluates per atom', () => {
  const N = 3;
  const sigDiag = new Float64Array([1.0, 2.0, 0.5]);
  // cov_i = diag(sigDiag[i], sigDiag[i])
  const sIR = refSelf('s');
  const covIR = rowstack([arrayIR([sIR, lit(0.0)]),
                          arrayIR([lit(0.0), sIR])]);
  const ir = { kind: 'call', op: 'MvNormal',
               kwargs: { mu: arrayIR([lit(0.0), lit(0.0)]), cov: covIR } };
  const x = { shape: [2], data: new Float64Array([0.3, -0.2]) };
  const logps = density.logDensityN(ir, x, { s: sigDiag }, N, {});
  for (let i = 0; i < N; i++) {
    const s = sigDiag[i];
    const cov = { shape: [2, 2], data: new Float64Array([s, 0, 0, s]) };
    const expected = densityPrims.builtinLogdensityof('MvNormal',
      { mu: [0, 0], cov }, x);
    assert.ok(Math.abs(logps[i] - expected) < 1e-12,
      `atom ${i}: got ${logps[i]}, expected ${expected}`);
  }
});

// =====================================================================
// Dirichlet: per-atom alpha (the canonical hierarchical-concentration shape)
// =====================================================================

test('walkMultivariate: per-atom alpha in Dirichlet evaluates per atom', () => {
  const N = 4;
  const a = new Float64Array([0.5, 1.0, 2.0, 5.0]);
  // alpha_i = [a_i, a_i, a_i] — symmetric concentration scaled per atom.
  const aIR = refSelf('a');
  const alphaIR = arrayIR([aIR, aIR, aIR]);
  const ir = { kind: 'call', op: 'Dirichlet', kwargs: { alpha: alphaIR } };
  const x = { shape: [3], data: new Float64Array([0.2, 0.3, 0.5]) };
  const logps = density.logDensityN(ir, x, { a }, N, {});
  for (let i = 0; i < N; i++) {
    const ai = a[i];
    const expected = densityPrims.builtinLogdensityof('Dirichlet',
      { alpha: [ai, ai, ai] }, x);
    assert.ok(Math.abs(logps[i] - expected) < 1e-12,
      `atom ${i}: got ${logps[i]}, expected ${expected}`);
  }
});

// =====================================================================
// Wishart: per-atom nu (degrees of freedom) — value-position parameter.
// scale is shared (a 2×2 identity), nu varies per atom.
// =====================================================================

test('walkMultivariate: per-atom nu in Wishart evaluates per atom', () => {
  const N = 3;
  const nuArr = new Float64Array([3.0, 4.0, 5.5]);
  const scaleIR = rowstack([arrayIR([lit(1.0), lit(0.0)]),
                            arrayIR([lit(0.0), lit(1.0)])]);
  const ir = { kind: 'call', op: 'Wishart',
               kwargs: { nu: refSelf('nu'), scale: scaleIR } };
  // The variate is a 2×2 SPD matrix supplied as the value.
  const X = { shape: [2, 2], data: new Float64Array([1.5, 0.2, 0.2, 1.0]) };
  const logps = density.logDensityN(ir, X, { nu: nuArr }, N, {});
  for (let i = 0; i < N; i++) {
    const expected = densityPrims.builtinLogdensityof('Wishart',
      { nu: nuArr[i],
        scale: { shape: [2, 2], data: new Float64Array([1, 0, 0, 1]) } },
      X);
    assert.ok(Math.abs(logps[i] - expected) < 1e-12,
      `atom ${i}: got ${logps[i]}, expected ${expected}`);
  }
});

// =====================================================================
// Atom-indep path still hits the resolve-once-broadcast hot path.
// We can't observe the path directly, but the numbers must match the
// primitive form.
// =====================================================================

test('walkMultivariate: atom-indep mu/cov MvNormal still bit-matches the primitive', () => {
  const N = 5;
  const muIR = arrayIR([lit(1.0), lit(2.0)]);
  const covIR = rowstack([arrayIR([lit(2.0), lit(0.5)]),
                          arrayIR([lit(0.5), lit(2.0)])]);
  const ir = { kind: 'call', op: 'MvNormal', kwargs: { mu: muIR, cov: covIR } };
  const x = { shape: [2], data: new Float64Array([1.5, 2.5]) };
  const logps = density.logDensityN(ir, x, null, N, {});
  const expected = densityPrims.builtinLogdensityof('MvNormal',
    { mu: [1, 2],
      cov: { shape: [2, 2], data: new Float64Array([2, 0.5, 0.5, 2]) } },
    x);
  for (let i = 0; i < N; i++) {
    assert.ok(Math.abs(logps[i] - expected) < 1e-12);
  }
});
