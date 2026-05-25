'use strict';

// =====================================================================
// density-prims.ts — `builtin_logdensityof` and the multivariate
// kernel log-density dispatch table.
// =====================================================================
//
// This module realises the FlatPDL `builtin_logdensityof(kernel,
// kernel_input, x)` primitive of spec §07 §sec:measure-eval-prims. It
// is the post-routing per-kernel formula ABI (engine-concepts §13.6):
// the single, kernel-keyed entry point that returns the log-density
// of `kernel(kernel_input)` at `x` w.r.t. the kernel's reference
// measure. The engine's density walker, the surface `builtin_*` call,
// and any future codegen path all share this dispatch.
//
// Two dispatch shapes are exposed:
//
//   builtinLogdensityof(name, kernelInput, x)
//     `kernelInput` is a record `{ paramName: value, ... }` of
//     already-evaluated values; matches the surface form of spec §07
//     where `kernel_input` is a record matching the kernel's kwarg
//     interface. The natural shape for the surface `builtin_*` call
//     dispatcher and for multivariate kernels (whose params live in
//     named slots).
//
//   builtinLogdensityofPositional(name, params, x)
//     `params` is the same positional list `REGISTRY[name].logpdfFn`
//     itself consumes (resolveParams result). Used by the density
//     walker's per-atom leaf path — it has already done the per-atom
//     resolution work and would otherwise pay an extra record build.
//     Univariate only.
//
// The multivariate dispatch table `MV_DENSITY_FNS` maps a kernel name
// (MvNormal, Dirichlet, Multinomial, BinnedPoissonProcess, Wishart,
// InverseWishart, LKJ, LKJCholesky) to a `(x, kernelInput) → number`
// function. The math is exactly what density.ts's per-kernel `walk*`
// functions used to inline — pulled into one place so it is reused
// across the leaf walker, the surface op, and (in time) the FlatPDL
// codegen path.

import type { Value as _Value } from './engine-types';

const valueLib  = require('./value.ts');
const valueOps  = require('./value-ops.ts');
const samplerLib = require('./sampler.ts');
const stdlibGammaln = require('@stdlib/math-base-special-gammaln');

// =====================================================================
// Helpers shared by the multivariate density formulae.
// =====================================================================

// log multivariate gamma function Γ_p(a) for natural-number p:
//   log Γ_p(a) = (p(p-1)/4) log π + Σ_{i=1..p} log Γ(a + (1-i)/2)
function logMvGamma(p: number, a: number) {
  let s = (p * (p - 1) / 4) * Math.log(Math.PI);
  for (let i = 1; i <= p; i++) {
    s += stdlibGammaln(a + (1 - i) / 2);
  }
  return s;
}

// log determinant of a square Value via lower-Cholesky; null when not
// positive-definite. log|A| = 2 · Σ log L_ii.
function logDetSPD(A: any): number | null {
  try {
    const L = samplerLib._internal.ARITH_OPS.lower_cholesky(A);
    const n = L.shape[0];
    let ld = 0;
    for (let i = 0; i < n; i++) {
      const lii = L.data[i * n + i];
      if (!(lii > 0)) return null;
      ld += Math.log(lii);
    }
    return 2 * ld;
  } catch {
    return null;
  }
}

// trace(A · B) for n×n Values A, B (sum over i of (AB)_ii).
function traceProduct(A: any, B: any) {
  const n = A.shape[0];
  let tr = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      tr += A.data[i * n + j] * B.data[j * n + i];
    }
  }
  return tr;
}

// Normalise an evaluated nested-array / typed-array / Value to a Value.
function _toValue(x: any): any {
  if (valueLib.isValue(x)) return x;
  if (Array.isArray(x) && x.length > 0 && Array.isArray(x[0])) {
    return valueOps._nestedToValue(x);
  }
  return valueLib.asValue(x);
}

// Normalise the input value `x` for a vector-variate kernel to a
// length-n Float64Array view. Accepts Value([n]) and Float64Array(n).
function _asVectorOfLength(x: any, n: number, kernelName: string): Float64Array {
  if (valueLib.isValue(x)) {
    if (x.shape.length === 1 && x.shape[0] === n) return x.data as Float64Array;
    throw new Error(`builtin_logdensityof(${kernelName}): variate must be a length-${n} vector, `
      + `got Value shape=${JSON.stringify(x.shape)}`);
  }
  if (x && x.BYTES_PER_ELEMENT && typeof x.length === 'number') {
    if (x.length !== n) {
      throw new Error(`builtin_logdensityof(${kernelName}): variate length ${x.length} ≠ expected ${n}`);
    }
    return x as Float64Array;
  }
  if (Array.isArray(x)) {
    if (x.length !== n) {
      throw new Error(`builtin_logdensityof(${kernelName}): variate length ${x.length} ≠ expected ${n}`);
    }
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = +x[i];
    return out;
  }
  throw new Error(`builtin_logdensityof(${kernelName}): unsupported variate type ${typeof x}`);
}

// Normalise the input value `x` for a matrix-variate kernel to a
// Value{shape:[n,n], data:...}.
function _asMatrixOfSize(x: any, n: number, kernelName: string): any {
  const v = _toValue(x);
  if (v.shape.length === 2 && v.shape[0] === n && v.shape[1] === n) return v;
  if (v.shape.length === 1 && v.shape[0] === n * n) {
    return { shape: [n, n], data: v.data };
  }
  throw new Error(`builtin_logdensityof(${kernelName}): variate must be ${n}x${n} matrix, `
    + `got shape=${JSON.stringify(v.shape)}`);
}

// Normalise a vector-typed kernel-input parameter (mu, alpha, p, …)
// to a plain JS-number array.
function _paramAsNumberArray(v: any, kernelName: string, paramName: string): number[] {
  if (valueLib.isValue(v)) return Array.from(v.data) as number[];
  if (Array.isArray(v)) return v.map((x: any) => +x);
  if (v && v.BYTES_PER_ELEMENT) return Array.from(v as ArrayLike<number>) as number[];
  throw new Error(`builtin_logdensityof(${kernelName}): param '${paramName}' must be a vector`);
}

// Normalise a matrix-typed kernel-input parameter (cov, scale, …) to a
// rank-2 Value.
function _paramAsMatrix(v: any, kernelName: string, paramName: string): any {
  const m = _toValue(v);
  if (m.shape.length !== 2 || m.shape[0] !== m.shape[1]) {
    throw new Error(`builtin_logdensityof(${kernelName}): param '${paramName}' must be a square matrix, `
      + `got shape=${JSON.stringify(m.shape)}`);
  }
  return m;
}

// =====================================================================
// MV_DENSITY_FNS — one entry per multivariate kernel.
// =====================================================================
//
// Each function takes (x, kernelInput) with x a Value of the kernel's
// natural shape (length-n vector or n×n matrix as required) and
// kernelInput a record `{ paramName: value, ... }`. Returns a scalar
// log-density.

const MV_DENSITY_FNS: Record<string, (x: any, kw: any) => number> = {

  // MvNormal(mu, cov) — n-variate normal.
  //   log p(x) = -½n log(2π) - ½ log|cov| - ½ (x-μ)ᵀ cov⁻¹ (x-μ)
  // Computed via L = lower_cholesky(cov):
  //   log|cov| = 2 Σ log L_ii
  //   (x-μ)ᵀ cov⁻¹ (x-μ) = ‖L⁻¹(x-μ)‖²
  MvNormal: function (x: any, kw: any): number {
    if (kw == null || !('mu' in kw) || !('cov' in kw)) {
      throw new Error('builtin_logdensityof(MvNormal): requires mu and cov');
    }
    const muArr = _paramAsNumberArray(kw.mu, 'MvNormal', 'mu');
    const n = muArr.length;
    const cov = _paramAsMatrix(kw.cov, 'MvNormal', 'cov');
    if (cov.shape[0] !== n) {
      throw new Error(`builtin_logdensityof(MvNormal): cov is ${cov.shape[0]}x${cov.shape[1]}, `
        + `but mu has length ${n}`);
    }
    const xv = _asVectorOfLength(x, n, 'MvNormal');
    const L = samplerLib._internal.ARITH_OPS.lower_cholesky(cov);
    let logDet = 0;
    for (let i = 0; i < n; i++) logDet += Math.log(L.data[i * n + i]);
    logDet *= 2;
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = xv[i] - muArr[i];
    // Forward solve L y = d.
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = d[i];
      for (let k = 0; k < i; k++) s -= L.data[i * n + k] * y[k];
      y[i] = s / L.data[i * n + i];
    }
    let mahal2 = 0;
    for (let i = 0; i < n; i++) mahal2 += y[i] * y[i];
    return -0.5 * (n * Math.log(2 * Math.PI) + logDet + mahal2);
  },

  // Dirichlet(alpha) on the standard simplex of dim n=length(alpha):
  //   log p(x) = log Γ(Σα) − Σ log Γ(α_i) + Σ (α_i − 1) log x_i
  Dirichlet: function (x: any, kw: any): number {
    if (kw == null || !('alpha' in kw)) {
      throw new Error('builtin_logdensityof(Dirichlet): requires alpha');
    }
    const alpha = _paramAsNumberArray(kw.alpha, 'Dirichlet', 'alpha');
    const n = alpha.length;
    if (n < 1) {
      throw new Error('builtin_logdensityof(Dirichlet): alpha must be non-empty');
    }
    const xv = _asVectorOfLength(x, n, 'Dirichlet');
    let alphaSum = 0, logBeta = 0;
    for (let i = 0; i < n; i++) {
      alphaSum += alpha[i];
      logBeta  += stdlibGammaln(alpha[i]);
    }
    logBeta -= stdlibGammaln(alphaSum);
    let lp = -logBeta;
    for (let i = 0; i < n; i++) {
      const xi = +xv[i];
      if (!(xi > 0)) return -Infinity;
      lp += (alpha[i] - 1) * Math.log(xi);
    }
    return lp;
  },

  // Multinomial(n, p) over a K-vector of nonneg integers summing to n:
  //   log p(x) = log Γ(n+1) − Σ log Γ(x_i+1) + Σ x_i log p_i
  Multinomial: function (x: any, kw: any): number {
    if (kw == null || !('n' in kw) || !('p' in kw)) {
      throw new Error('builtin_logdensityof(Multinomial): requires n and p');
    }
    const nTotal = +kw.n;
    const p = _paramAsNumberArray(kw.p, 'Multinomial', 'p');
    const K = p.length;
    if (K < 1) throw new Error('builtin_logdensityof(Multinomial): p must be non-empty');
    const xv = _asVectorOfLength(x, K, 'Multinomial');
    let sumX = 0;
    let lp = stdlibGammaln(nTotal + 1);
    for (let i = 0; i < K; i++) {
      const xi = +xv[i];
      if (xi < 0 || xi !== Math.round(xi)) return -Infinity;
      sumX += xi;
      lp -= stdlibGammaln(xi + 1);
      if (xi > 0) {
        if (!(p[i] > 0)) return -Infinity;
        lp += xi * Math.log(p[i]);
      }
    }
    if (sumX !== nTotal) return -Infinity;
    return lp;
  },

  // BinnedPoissonProcess(bins, intensity) — `intensity` is the per-bin
  // rate vector; the variate is the per-bin count vector. Density:
  //   log p(x) = Σ_k (x_k log λ_k − λ_k − log x_k!)
  BinnedPoissonProcess: function (x: any, kw: any): number {
    if (kw == null || !('intensity' in kw)) {
      throw new Error('builtin_logdensityof(BinnedPoissonProcess): requires intensity');
    }
    const rates = _paramAsNumberArray(kw.intensity, 'BinnedPoissonProcess', 'intensity');
    const K = rates.length;
    const xv = _asVectorOfLength(x, K, 'BinnedPoissonProcess');
    let lp = 0;
    for (let k = 0; k < K; k++) {
      const xk = +xv[k];
      const lam = +rates[k];
      if (xk < 0 || xk !== Math.round(xk) || lam < 0) return -Infinity;
      if (lam === 0) {
        if (xk !== 0) return -Infinity;
      } else {
        lp += xk * Math.log(lam) - lam - stdlibGammaln(xk + 1);
      }
    }
    return lp;
  },

  // Wishart(nu, scale) on n×n SPD matrices:
  //   log p(X) = (ν-n-1)/2 log|X| − ½ tr(V⁻¹ X) − νn/2 log2 − ν/2 log|V|
  //            − log Γ_n(ν/2)
  Wishart: function (x: any, kw: any): number {
    return _wishart(x, kw, /*invert=*/false);
  },
  // InverseWishart(nu, scale=Ψ):
  //   log p(X) = ν/2 log|Ψ| − (ν+n+1)/2 log|X| − ½ tr(Ψ X⁻¹) − νn/2 log2
  //            − log Γ_n(ν/2)
  InverseWishart: function (x: any, kw: any): number {
    return _wishart(x, kw, /*invert=*/true);
  },

  // LKJ(n, eta) — correlation-matrix density:
  //   p(C) = c_n(η) · det(C)^(η-1)
  LKJ: function (x: any, kw: any): number {
    if (kw == null || !('n' in kw) || !('eta' in kw)) {
      throw new Error('builtin_logdensityof(LKJ): requires n and eta');
    }
    const n = +kw.n | 0;
    const eta = +kw.eta;
    const C = _asMatrixOfSize(x, n, 'LKJ');
    const ld = logDetSPD(C);
    if (ld == null) return -Infinity;
    return _logCnLKJ(n, eta) + (eta - 1) * ld;
  },

  // LKJCholesky(n, eta) — Cholesky factor of an LKJ-distributed C:
  //   p(L) = c_n(η) · Π_{i=2..n} L_ii^(n − i + 2η − 2)
  LKJCholesky: function (x: any, kw: any): number {
    if (kw == null || !('n' in kw) || !('eta' in kw)) {
      throw new Error('builtin_logdensityof(LKJCholesky): requires n and eta');
    }
    const n = +kw.n | 0;
    const eta = +kw.eta;
    const L = _asMatrixOfSize(x, n, 'LKJCholesky');
    let lp = _logCnLKJ(n, eta);
    for (let i = 1; i < n; i++) {
      const lii = L.data[i * n + i];
      if (!(lii > 0)) return -Infinity;
      lp += (n - (i + 1) + 2 * eta - 2) * Math.log(lii);
    }
    return lp;
  },
};

function _wishart(x: any, kw: any, invert: boolean): number {
  const label = invert ? 'InverseWishart' : 'Wishart';
  if (kw == null || !('nu' in kw) || !('scale' in kw)) {
    throw new Error(`builtin_logdensityof(${label}): requires nu and scale`);
  }
  const nu = +kw.nu;
  const V = _paramAsMatrix(kw.scale, label, 'scale');
  const n = V.shape[0];
  if (!(nu > n - 1)) return -Infinity;
  const X = _asMatrixOfSize(x, n, label);
  const logDetV = logDetSPD(V);
  const logDetX = logDetSPD(X);
  if (logDetV == null || logDetX == null) return -Infinity;
  const logG = logMvGamma(n, nu / 2);
  if (!invert) {
    const Vinv = samplerLib._internal.ARITH_OPS.inv(V);
    const tr = traceProduct(Vinv, X);
    return (nu - n - 1) / 2 * logDetX
         - 0.5 * tr
         - (nu * n / 2) * Math.LN2
         - (nu / 2) * logDetV
         - logG;
  }
  const Xinv = samplerLib._internal.ARITH_OPS.inv(X);
  const tr = traceProduct(V, Xinv);
  return (nu / 2) * logDetV
       - (nu + n + 1) / 2 * logDetX
       - 0.5 * tr
       - (nu * n / 2) * Math.LN2
       - logG;
}

// LKJ normalization constant log c_n(η).
function _logCnLKJ(n: number, eta: number) {
  let lc = 0;
  for (let k = 1; k <= n - 1; k++) {
    const m = n - k;
    lc += (2 * eta - 2 + m) * m * Math.LN2;
    const a = eta + (m - 1) / 2;
    lc += m * (2 * stdlibGammaln(a) - stdlibGammaln(2 * a));
  }
  return lc;
}

// =====================================================================
// Dispatch
// =====================================================================

/**
 * Whether `kernelName` is a kernel `builtinLogdensityof` knows about —
 * a univariate REGISTRY entry, a multivariate entry in MV_DENSITY_FNS,
 * or `Dirac` (handled as a special case of point-mass below).
 */
function isBuiltinKernel(kernelName: string): boolean {
  if (kernelName === 'Dirac') return true;
  return samplerLib.isKnownDistribution(kernelName)
      || Object.prototype.hasOwnProperty.call(MV_DENSITY_FNS, kernelName);
}

/**
 * The FlatPDL primitive in its record-input form:
 *
 *   lp = builtin_logdensityof(kernel, kernel_input, x)
 *
 * Per spec §07 §sec:measure-eval-prims:
 *   - `kernel` (here `kernelName`) is the kernel constructor name.
 *   - `kernel_input` is a record matching the kernel's kwarg interface;
 *     values must be already evaluated (numbers / Values).
 *   - `x` is the variate value; scalar (number) for univariate kernels,
 *     a length-n vector or n×n matrix Value for multivariate.
 * Returns the scalar log-density, `-Infinity` outside support.
 */
function builtinLogdensityof(kernelName: string, kernelInput: any, x: any): number {
  // Univariate via REGISTRY.
  if (samplerLib.isKnownDistribution(kernelName)) {
    const entry = samplerLib._internal.REGISTRY[kernelName];
    // Uniform's `support` is a set; we accept either an interval-shaped
    // object `{lo, hi}` or a 2-array `[lo, hi]` here. The IR-level
    // customResolveParams path stays the engine's primary form; this
    // record-input form is the FlatPDL spec shape.
    if (kernelName === 'Uniform') {
      const s = kernelInput && (kernelInput.support);
      let lo: number, hi: number;
      if (Array.isArray(s) && s.length === 2) { lo = +s[0]; hi = +s[1]; }
      else if (s && typeof s === 'object' && ('lo' in s) && ('hi' in s)) { lo = +s.lo; hi = +s.hi; }
      else if (s && typeof s === 'object' && ('a' in s) && ('b' in s))  { lo = +s.a;  hi = +s.b;  }
      else throw new Error('builtin_logdensityof(Uniform): support must be [lo,hi], {lo,hi}, or {a,b}');
      return entry.logpdfFn(_scalarOf(x, 'Uniform'), lo, hi);
    }
    const params: any[] = [];
    for (let i = 0; i < entry.params.length; i++) {
      const pName = entry.params[i];
      if (kernelInput != null
          && Object.prototype.hasOwnProperty.call(kernelInput, pName)) {
        params.push(kernelInput[pName]);
      } else if (entry.aliases[pName]
                 && kernelInput != null
                 && Object.prototype.hasOwnProperty.call(kernelInput, entry.aliases[pName])) {
        params.push(kernelInput[entry.aliases[pName]]);
      } else {
        throw new Error(`builtin_logdensityof(${kernelName}): missing param '${pName}'`);
      }
    }
    return entry.logpdfFn(_scalarOf(x, kernelName), ...params);
  }
  // Multivariate.
  const fn = MV_DENSITY_FNS[kernelName];
  if (fn) return fn(x, kernelInput);
  throw new Error(`builtin_logdensityof: unknown kernel '${kernelName}'`);
}

/**
 * Positional-params shortcut for univariate kernels — what the density
 * walker's leaf path calls. `params` is what `resolveParams` returns.
 * Multivariate kernels need the record form `builtinLogdensityof`.
 */
function builtinLogdensityofPositional(kernelName: string, params: any[], x: any): number {
  if (!samplerLib.isKnownDistribution(kernelName)) {
    throw new Error(`builtin_logdensityof (positional): '${kernelName}' is not a univariate `
      + `REGISTRY kernel — use the record form for multivariate kernels`);
  }
  const entry = samplerLib._internal.REGISTRY[kernelName];
  return entry.logpdfFn(_scalarOf(x, kernelName), ...params);
}

// Coerce x to a JS number for univariate dispatch.
function _scalarOf(x: any, kernelName: string): number {
  if (typeof x === 'number') return x;
  if (typeof x === 'boolean') return x ? 1 : 0;
  if (valueLib.isValue(x)) {
    if (x.shape.length === 0) return x.data[0];
    if (x.shape.length === 1 && x.shape[0] === 1) return x.data[0];
    throw new Error(`builtin_logdensityof(${kernelName}): univariate kernel needs a scalar `
      + `variate; got Value shape=${JSON.stringify(x.shape)}`);
  }
  return +x;
}

// =====================================================================
// Variate-shape lookup for multivariate kernels.
// =====================================================================
//
// The generic multivariate walker in density.ts uses this to decide
// how much of `value` to consume off the head and how to shape the
// per-atom variate before calling the primitive. `sizeFrom` reads
// the resolved kernel-input record (kwargs already evaluated to
// numbers / Values) and returns the kernel's n.
//
// `kind: 'vector'` ⇒ consume a length-n slice; `kind: 'matrix'`
// ⇒ consume an n×n block. Variate-shape parameters that are kwargs
// (`LKJ.n` / `LKJCholesky.n`) are read literally; data-shape
// parameters (mu / alpha / p / intensity / scale) determine the size
// from their own shape.

const MV_VARIATE_SHAPE: Record<string, {
  kind: 'vector' | 'matrix';
  sizeFrom: (kw: any) => number;
}> = {
  MvNormal:             { kind: 'vector', sizeFrom: (kw) => _lenOf(kw.mu) },
  Dirichlet:            { kind: 'vector', sizeFrom: (kw) => _lenOf(kw.alpha) },
  Multinomial:          { kind: 'vector', sizeFrom: (kw) => _lenOf(kw.p) },
  BinnedPoissonProcess: { kind: 'vector', sizeFrom: (kw) => _lenOf(kw.intensity) },
  Wishart:              { kind: 'matrix', sizeFrom: (kw) => _matSizeOf(kw.scale) },
  InverseWishart:       { kind: 'matrix', sizeFrom: (kw) => _matSizeOf(kw.scale) },
  LKJ:                  { kind: 'matrix', sizeFrom: (kw) => (+kw.n) | 0 },
  LKJCholesky:          { kind: 'matrix', sizeFrom: (kw) => (+kw.n) | 0 },
};

function _lenOf(v: any): number {
  if (valueLib.isValue(v)) return v.shape[0];
  if (Array.isArray(v) || (v && v.BYTES_PER_ELEMENT)) return v.length;
  throw new Error('density-prims: param expected to be a vector, got '
    + (v == null ? 'null' : typeof v));
}

function _matSizeOf(v: any): number {
  const m = _toValue(v);
  if (m.shape.length !== 2 || m.shape[0] !== m.shape[1]) {
    throw new Error('density-prims: scale must be a square matrix, got shape='
      + JSON.stringify(m.shape));
  }
  return m.shape[0];
}

/** True iff `kernelName` is multivariate (variate is vector or matrix). */
function isMultivariateKernel(kernelName: string): boolean {
  return Object.prototype.hasOwnProperty.call(MV_VARIATE_SHAPE, kernelName);
}

/** Variate-shape descriptor for a multivariate kernel; null otherwise. */
function multivariateShape(kernelName: string) {
  return MV_VARIATE_SHAPE[kernelName] || null;
}

module.exports = {
  builtinLogdensityof,
  builtinLogdensityofPositional,
  isBuiltinKernel,
  isMultivariateKernel,
  multivariateShape,
  MV_DENSITY_FNS,
  // Test surface
  _internal: { logMvGamma, logDetSPD, traceProduct, _logCnLKJ },
};
