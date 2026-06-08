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
  // Spec §03: a vec-of-vec is NOT a matrix. Density kernels with
  // matrix variates require a true rank-2 array of scalars; reject
  // the nested-vector form so the user is told to `rowstack(...)`.
  valueLib.requireMatrix(v, `builtin_logdensityof(${kernelName}): variate`);
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
  // Spec §03: a vec-of-vec is NOT a matrix. Multivariate kernel params
  // (cov, scale, …) must be a true rank-2 array of scalars.
  valueLib.requireMatrix(m, `builtin_logdensityof(${kernelName}): param '${paramName}'`);
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
  //
  // Implementation IS the canonical decomposition per engine-concepts
  // §22:  MvNormal(mu, Sigma) = pushfwd(affine(L, mu), iid(Normal, n))
  // with L = lower_cholesky(cov). The log-density factorises as
  //   log p_MvNormal(x | mu, Sigma)
  //     = log p_iidNormal(L⁻¹(x − mu), n) + log|det J_{f⁻¹}(x)|
  //     = (-½ n log(2π) - ½ ‖z‖²) + (-Σ log|diag(L)|)
  // where `z = L⁻¹(x − mu)` and `logDetJ = -Σ log|diag(L)|` come from
  // the `affine` bijection registry entry. This is the SAME numerical
  // path the legacy inline code took (forward-sub against L, log-det
  // via diag(L)), now expressed as a registry composition so the
  // closed-form math has ONE owner — `bijection-registry.ts` — instead
  // of two parallel implementations drifting.
  //
  // Equivalence is pinned by `test/bijection-registry.test.ts`:
  // "affine density: registry-composed MvNormal score equals
  // density-prim closed form".
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
    // Registry path: z = affine.atomBatchedInverse(x, {L, b=mu}) with
    // a single-atom batch (N=1). The registry's forward-substitute is
    // the same closed-form as the legacy inline d-/y-loops.
    const bijRegistry = require('./bijection-registry.ts');
    const xMat = { shape: [1, n], data: xv };
    const muVec = { shape: [n], data: new Float64Array(muArr) };
    const z = bijRegistry.affineAtomBatchedInverse(
      xMat, { L: L, b: muVec }, /*N=*/1);
    const logDetJ = bijRegistry.affineLogDetJ(null, { L: L }, /*N=*/1);
    let mahal2 = 0;
    for (let i = 0; i < n; i++) mahal2 += z.data[i] * z.data[i];
    return -0.5 * (n * Math.log(2 * Math.PI) + mahal2) + logDetJ;
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

// LKJ log-density multiplicative constant `log c_n(η)`.
//
// The LKJ density is `p(C) = c_n(η)·det(C)^(η-1)`, where `c_n(η) = 1/Z` and
// `Z = ∫ det(C)^(η-1) dC` is the normalizing INTEGRAL over the correlation
// manifold. The closed form below is the standard product expansion of `log Z`
// (NOTE: the formula PRINTED as `c_n(η)` in docs/08-distributions.md is in fact
// `Z`, its reciprocal — that is why we accumulate it as `log Z` and return the negation)
// (see the LKJ normalizer in e.g. Lewandowski–Kurowicka–Joe 2009 / Stan);
// since `c_n = 1/Z`, `log c_n(η) = −log Z`. We therefore NEGATE the accumulated
// `log Z` on return. Both call sites (LKJ and LKJCholesky) ADD this quantity,
// which is correct now that it carries the density's multiplicative constant
// rather than the integral.
//
// Anchor: n=2, η=1 gives the uniform density over ρ∈(−1,1) = 1/2, so
// `log c_2(1) = −log 2 ≈ −0.6931`.
function _logCnLKJ(n: number, eta: number) {
  let lc = 0;
  for (let k = 1; k <= n - 1; k++) {
    const m = n - k;
    lc += (2 * eta - 2 + m) * m * Math.LN2;
    const a = eta + (m - 1) / 2;
    lc += m * (2 * stdlibGammaln(a) - stdlibGammaln(2 * a));
  }
  return -lc;
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
// Canonical measurable transports (spec §07 §sec:measure-eval-prims)
// =====================================================================
//
// `builtin_touniform`/`fromuniform` and `builtin_tonormal`/`fromnormal`
// transport between a kernel's measure and the standard uniform /
// standard normal reference of matching dimension. Per spec:
//
//   • Continuous univariate kernels: touniform = CDF F; fromuniform = F⁻¹.
//   • Continuous multivariate kernels: measure-specific transport
//     defined alongside the measure (spec §08). v0.1 implements
//     MvNormal here (Cholesky-based); others are open follow-ups.
//   • Discrete kernels: no canonical transport — static error.
//
// The four functions are mutually consistent via probit/invprobit:
//   touniform(x) ≡ invprobit.(tonormal(x))
//   tonormal(x)  ≡ probit.(touniform(x))     (+ the two inverses)
// We implement one direction natively per kernel and derive the others
// from the consistency relations.

// Element-wise Φ (normal CDF) — invprobit(z) — and its inverse Φ⁻¹ —
// probit(p). Per spec §07 and stdlib convention.
const _stdlibErfc    = require('@stdlib/math-base-special-erfc');
const _stdlibErfcinv = require('@stdlib/math-base-special-erfcinv');
function _normalCdf(z: number): number {
  // Φ(z) = ½ erfc(-z/√2)
  return 0.5 * _stdlibErfc(-z / Math.SQRT2);
}
function _normalQuantile(p: number): number {
  // Φ⁻¹(p) = -√2 · erfc⁻¹(2p)
  return -Math.SQRT2 * _stdlibErfcinv(2 * p);
}

// Extract the positional param list for a univariate kernel from a
// kernel_input record. Mirrors `builtinLogdensityof`'s logic; lives
// here so the transport primitives reuse it. Throws when a required
// param is missing.
function _univariateParams(kernelName: string, kernelInput: any, entry: any): any[] {
  if (kernelName === 'Uniform') {
    // Uniform's stdlib Ctor takes (a, b); kernel_input.support is the
    // set object. Accept the same shapes builtinLogdensityof accepts.
    const s = kernelInput && kernelInput.support;
    if (Array.isArray(s) && s.length === 2) return [+s[0], +s[1]];
    if (s && typeof s === 'object' && 'lo' in s && 'hi' in s) return [+s.lo, +s.hi];
    if (s && typeof s === 'object' && 'a' in s && 'b' in s)   return [+s.a,  +s.b];
    throw new Error('builtin: Uniform support must be [lo,hi], {lo,hi}, or {a,b}');
  }
  const out: any[] = [];
  for (let i = 0; i < entry.params.length; i++) {
    const p = entry.params[i];
    if (kernelInput != null
        && Object.prototype.hasOwnProperty.call(kernelInput, p)) {
      out.push(kernelInput[p]);
    } else if (entry.aliases[p]
               && kernelInput != null
               && Object.prototype.hasOwnProperty.call(kernelInput, entry.aliases[p])) {
      out.push(kernelInput[entry.aliases[p]]);
    } else {
      throw new Error(`builtin: '${kernelName}' missing param '${p}'`);
    }
  }
  return out;
}

// Build a stdlib Ctor-style distribution object for a univariate kernel.
// Returns null when the entry has no Ctor (no current kernel hits that,
// but the guard keeps a future engine extension honest).
function _makeUnivariateCtor(kernelName: string, kernelInput: any): any {
  const entry = samplerLib._internal.REGISTRY[kernelName];
  if (!entry || !entry.Ctor) return null;
  const params = _univariateParams(kernelName, kernelInput, entry);
  // stdlib Ctors use `new Ctor(...params)`. Our synthetic Ctors are
  // also called with `new`, so the two paths converge.
  return new entry.Ctor(...params);
}

// MvNormal-only multivariate transports. Both consume an n-vector and
// produce an n-vector; the transport is invertible. mu must be a JS
// number[] / typed array / Value; cov a square Value.
function _mvNormalTonormal(x: any, kernelInput: any): any {
  if (!kernelInput || !('mu' in kernelInput) || !('cov' in kernelInput)) {
    throw new Error('builtin_tonormal(MvNormal): requires mu and cov');
  }
  const muArr = _paramAsNumberArray(kernelInput.mu, 'MvNormal', 'mu');
  const n = muArr.length;
  const cov = _paramAsMatrix(kernelInput.cov, 'MvNormal', 'cov');
  if (cov.shape[0] !== n) {
    throw new Error(`builtin_tonormal(MvNormal): cov is ${cov.shape[0]}x${cov.shape[1]} `
      + `but mu has length ${n}`);
  }
  const xv = _asVectorOfLength(x, n, 'MvNormal');
  const L = samplerLib._internal.ARITH_OPS.lower_cholesky(cov);
  // z = L⁻¹ (x − μ) via forward substitution.
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = xv[i] - muArr[i];
    for (let k = 0; k < i; k++) s -= L.data[i * n + k] * out[k];
    out[i] = s / L.data[i * n + i];
  }
  return { shape: [n], data: out };
}
function _mvNormalFromnormal(z: any, kernelInput: any): any {
  if (!kernelInput || !('mu' in kernelInput) || !('cov' in kernelInput)) {
    throw new Error('builtin_fromnormal(MvNormal): requires mu and cov');
  }
  const muArr = _paramAsNumberArray(kernelInput.mu, 'MvNormal', 'mu');
  const n = muArr.length;
  const cov = _paramAsMatrix(kernelInput.cov, 'MvNormal', 'cov');
  if (cov.shape[0] !== n) {
    throw new Error(`builtin_fromnormal(MvNormal): cov is ${cov.shape[0]}x${cov.shape[1]} `
      + `but mu has length ${n}`);
  }
  const zv = _asVectorOfLength(z, n, 'MvNormal');
  const L = samplerLib._internal.ARITH_OPS.lower_cholesky(cov);
  // x = μ + L z; L is lower-triangular row-major.
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = muArr[i];
    for (let k = 0; k <= i; k++) s += L.data[i * n + k] * zv[k];
    out[i] = s;
  }
  return { shape: [n], data: out };
}

// Multivariate kernels other than MvNormal — transport is spec-defined
// per measure but not yet implemented in flatppl-js. Surface as a
// clear "not yet implemented" error rather than the spec's "static
// error" (a static phase check would belong in the analyzer; for now
// the runtime is the gate).
function _multivariateTransportNotImpl(kernelName: string, fn: string): never {
  throw new Error(`builtin_${fn}: '${kernelName}' transport is spec-defined `
    + `but not yet implemented in flatppl-js. (MvNormal works; Dirichlet, `
    + `Wishart, etc. are open follow-ups — see TODO-flatppl-js.md.)`);
}

// Reject transports on discrete kernels per spec §07 ("use of an
// undefined transport function is a static error" — we surface at
// runtime until a static phase check lands).
function _rejectDiscreteTransport(kernelName: string, fn: string): never {
  throw new Error(`builtin_${fn}: '${kernelName}' is discrete; the four `
    + `transport primitives are defined only for continuous kernels (spec §07)`);
}

/**
 * builtin_touniform(kernel, kernel_input, x) — transport x from the
 * kernel's measure to the standard uniform reference.
 *
 *   - Continuous univariate kernel: F(x), via the stdlib Ctor's `cdf`.
 *   - Continuous multivariate kernel: derived from `tonormal` via
 *     `invprobit` per the spec's consistency relation. MvNormal works
 *     in v0.1; other multivariate kernels throw "not yet implemented".
 *   - Discrete kernel: static error.
 */
function builtinTouniform(kernelName: string, kernelInput: any, x: any): any {
  if (kernelName === 'MvNormal') {
    const z = _mvNormalTonormal(x, kernelInput);
    // touniform ≡ invprobit ∘ tonormal — elementwise normal CDF.
    const out = new Float64Array(z.data.length);
    for (let i = 0; i < z.data.length; i++) out[i] = _normalCdf(z.data[i]);
    return { shape: z.shape, data: out };
  }
  if (isMultivariateKernel(kernelName)) {
    _multivariateTransportNotImpl(kernelName, 'touniform');
  }
  const entry = samplerLib._internal.REGISTRY[kernelName];
  if (!entry) throw new Error(`builtin_touniform: unknown kernel '${kernelName}'`);
  if (entry.discrete) _rejectDiscreteTransport(kernelName, 'touniform');
  const dist = _makeUnivariateCtor(kernelName, kernelInput);
  if (!dist || typeof dist.cdf !== 'function') {
    throw new Error(`builtin_touniform: '${kernelName}' has no cdf available`);
  }
  return dist.cdf(_scalarOf(x, kernelName));
}

/**
 * builtin_fromuniform(kernel, kernel_input, u) — inverse of touniform.
 *   - Continuous univariate: F⁻¹(u) via stdlib Ctor's `quantile`.
 *   - MvNormal: fromnormal(probit(u)) elementwise.
 *   - Other multivariate: not yet implemented.
 *   - Discrete: static error.
 */
function builtinFromuniform(kernelName: string, kernelInput: any, u: any): any {
  if (kernelName === 'MvNormal') {
    const uv = _asVectorOfLength(u, _paramAsNumberArray(kernelInput.mu, 'MvNormal', 'mu').length, 'MvNormal');
    const z = new Float64Array(uv.length);
    for (let i = 0; i < uv.length; i++) z[i] = _normalQuantile(uv[i]);
    return _mvNormalFromnormal({ shape: [uv.length], data: z }, kernelInput);
  }
  if (isMultivariateKernel(kernelName)) {
    _multivariateTransportNotImpl(kernelName, 'fromuniform');
  }
  const entry = samplerLib._internal.REGISTRY[kernelName];
  if (!entry) throw new Error(`builtin_fromuniform: unknown kernel '${kernelName}'`);
  if (entry.discrete) _rejectDiscreteTransport(kernelName, 'fromuniform');
  const dist = _makeUnivariateCtor(kernelName, kernelInput);
  if (!dist || typeof dist.quantile !== 'function') {
    throw new Error(`builtin_fromuniform: '${kernelName}' has no quantile available`);
  }
  return dist.quantile(_scalarOf(u, kernelName));
}

/**
 * builtin_tonormal(kernel, kernel_input, x) — transport x to standard
 * normal reference.
 *   - Univariate continuous: probit(touniform(x)) (consistency relation).
 *   - MvNormal: L⁻¹(x − μ) directly (the natural form per spec §08).
 *   - Other multivariate: not yet implemented.
 *   - Discrete: static error.
 */
function builtinTonormal(kernelName: string, kernelInput: any, x: any): any {
  if (kernelName === 'MvNormal') return _mvNormalTonormal(x, kernelInput);
  if (isMultivariateKernel(kernelName)) {
    _multivariateTransportNotImpl(kernelName, 'tonormal');
  }
  const entry = samplerLib._internal.REGISTRY[kernelName];
  if (!entry) throw new Error(`builtin_tonormal: unknown kernel '${kernelName}'`);
  if (entry.discrete) _rejectDiscreteTransport(kernelName, 'tonormal');
  const u = builtinTouniform(kernelName, kernelInput, x);
  return _normalQuantile(+u);
}

/**
 * builtin_fromnormal(kernel, kernel_input, z) — inverse of tonormal.
 *   - Univariate continuous: fromuniform(invprobit(z)).
 *   - MvNormal: μ + L·z directly.
 *   - Other multivariate: not yet implemented.
 *   - Discrete: static error.
 */
function builtinFromnormal(kernelName: string, kernelInput: any, z: any): any {
  if (kernelName === 'MvNormal') return _mvNormalFromnormal(z, kernelInput);
  if (isMultivariateKernel(kernelName)) {
    _multivariateTransportNotImpl(kernelName, 'fromnormal');
  }
  const entry = samplerLib._internal.REGISTRY[kernelName];
  if (!entry) throw new Error(`builtin_fromnormal: unknown kernel '${kernelName}'`);
  if (entry.discrete) _rejectDiscreteTransport(kernelName, 'fromnormal');
  const u = _normalCdf(_scalarOf(z, kernelName));
  return builtinFromuniform(kernelName, kernelInput, u);
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

// =====================================================================
// Type-mode consume/rest walker (engine-concepts §17.3)
// =====================================================================
//
// The value-mode density walker in `density.ts` propagates a `rest`
// alongside the running log-density: each measure-kind handler
// consumes its footprint off the variate value and returns whatever's
// left. The empty-rest assertion at the end is what catches
// shape-mismatched joint/iid observations.
//
// This is the **type-mode counterpart**: same dispatch shape, but
// the "value" is a structural TYPE (engine-types `Type`), and the
// "rest" is the type of whatever variate-shape remains unconsumed.
// An empty rest after walking the whole measure IR is the static
// shape-mismatch check; a non-empty rest means the user's observation
// type doesn't fit the measure's footprint.
//
// Return shape:
//   { logpType: 'real',
//     rest:     <Type | null>,
//     error?:   <string when a mismatch is statically provable>,
//     deferred?: boolean   // true when the walker couldn't decide
//                          // (e.g. %dynamic shapes); caller treats
//                          // as "passes static check, rely on
//                          // runtime" rather than as an error.
//   }
//
// v0.1 scope: scalar leaves, weighted/logweighted/truncate/normalize
// passthroughs, joint (positional + record), iid (literal n; dynamic
// n stays deferred), and the multivariate kernels (MvNormal, etc.)
// whose footprint is a known vector/matrix dim. Select/pushfwd are
// open follow-ups.

const T = require('./types.ts');

type StaticRest = any | null;
interface StaticConsumeResult {
  logpType: any;            // T.REAL when the measure is well-typed
  rest: StaticRest;         // null = consumed-cleanly
  error?: string;           // present iff a static shape mismatch
  deferred?: boolean;       // true when the walker can't statically decide
}

/**
 * Type-mode consume/rest walker. Given a measure IR and the variate's
 * structural type, return whether the measure's footprint exactly
 * matches the variate shape (`rest: null`), what's left if not, and
 * a static error string when a mismatch is statically provable.
 *
 * Use `staticDensityShapeCheck` (below) for the strict empty-rest
 * wrapper — typically what callers want.
 */
function staticConsume(ir: any, variateType: any): StaticConsumeResult {
  if (!ir) return { logpType: T.REAL, rest: variateType, deferred: true };
  // Self-ref measures (e.g. `m = Normal(0, 1); lawof(m)`) are
  // beyond v0.1 scope — caller can pre-expand if needed.
  if (ir.kind !== 'call') {
    return { logpType: T.REAL, rest: variateType, deferred: true };
  }
  const op = ir.op;
  // --- Univariate scalar leaves -------------------------------------
  if (samplerLib.isKnownDistribution(op)) {
    return _consumeScalarLeaf(variateType);
  }
  // --- Multivariate kernels -----------------------------------------
  if (isMultivariateKernel(op)) {
    return _consumeMultivariateLeaf(op, ir, variateType);
  }
  // --- Pass-throughs (don't change variate footprint) ---------------
  // weighted/logweighted: (weight, base) — base in args[1].
  // normalize / truncate: (base, ...) — base in args[0].
  if (op === 'weighted' || op === 'logweighted') {
    return staticConsume(ir.args && ir.args[1], variateType);
  }
  if (op === 'normalize' || op === 'truncate') {
    return staticConsume(ir.args && ir.args[0], variateType);
  }
  if (op === 'pushfwd') {
    return _consumePushfwd(ir, variateType);
  }
  // --- joint / record (positional and kwarg field forms) ------------
  if (op === 'joint' || op === 'record') {
    return _consumeJoint(ir, variateType);
  }
  // --- iid(M, size) -------------------------------------------------
  if (op === 'iid') {
    return _consumeIid(ir, variateType);
  }
  // --- select(branches, logweights) ---------------------------------
  // Engine-concepts §12: superpose / measure-ifelse / xs[i] all share
  // ONE variate space (spec §06). Each branch must consume the
  // variate identically; a branch whose footprint disagrees is a
  // static error. Branches may carry `ref` or `ir` form (the
  // post-2026-05 unified shape).
  if (op === 'select') {
    return _consumeSelect(ir, variateType);
  }
  // --- jointchain (dependent composition) ---------------------------
  // jointchain canonicalises into joint/record before the density
  // walker sees it (per the canonicaliser comment in
  // density.walkJointchainStub). At the static-shape layer we don't
  // need a separate handler — joint/record cover the canonical form.
  // --- others -------------------------------------------------------
  // Defer rather than error: the value-mode walker handles these;
  // the type-mode check just doesn't statically verify them yet.
  return { logpType: T.REAL, rest: null, deferred: true };
}

function _consumeScalarLeaf(t: any): StaticConsumeResult {
  if (!t) return { logpType: T.REAL, rest: null, deferred: true };
  // Scalar variate types pass without rest.
  if (t.kind === 'scalar') return { logpType: T.REAL, rest: null };
  // Rank-1 array with leading dim 1 also satisfies "consumes one entry".
  if (t.kind === 'array' && Array.isArray(t.shape) && t.shape.length === 1) {
    const len = t.shape[0];
    if (len === 1) return { logpType: T.REAL, rest: null };
    if (typeof len === 'number') {
      return { logpType: T.REAL,
        rest: T.array(1, [len - 1], t.elem) };
    }
    // %dynamic: defer the empty-rest check to runtime.
    return { logpType: T.REAL, rest: t, deferred: true };
  }
  if (t.kind === 'deferred' || t.kind === 'any') {
    return { logpType: T.REAL, rest: t, deferred: true };
  }
  return { logpType: T.REAL, rest: t,
    error: 'scalar leaf expected scalar variate, got ' + T.show(t) };
}

function _consumeMultivariateLeaf(op: string, ir: any, t: any): StaticConsumeResult {
  // The footprint is a length-n vector for vector-variate kernels
  // (MvNormal, Dirichlet, Multinomial, BinnedPoissonProcess) and an
  // n×n matrix for matrix-variate kernels (Wishart, InverseWishart,
  // LKJ, LKJCholesky). v0.1: defer when the kwargs aren't statically
  // resolved enough to read n (full inference of kw types is a
  // follow-up); otherwise check footprint matches the leading
  // variate axes.
  const shape = multivariateShape(op);
  if (!shape) return { logpType: T.REAL, rest: t, deferred: true };
  if (!t || (t.kind !== 'array')) {
    return { logpType: T.REAL, rest: t,
      error: op + ' expects array variate, got ' + T.show(t) };
  }
  // Need (kind, n) — we only know the kind statically here (vector vs
  // matrix); reading n requires running the kwarg expressions. For now
  // accept if the variate's leading dim matches the kind's
  // dimensionality (1 dim for vector, 2 dims for matrix); defer the
  // length check.
  const wantRank = shape.kind === 'vector' ? 1 : 2;
  if (t.rank >= wantRank) {
    return { logpType: T.REAL, rest: null, deferred: true };
  }
  return { logpType: T.REAL, rest: t,
    error: op + ' expects rank-≥' + wantRank + ' variate, got rank ' + t.rank };
}

function _consumeJoint(ir: any, t: any): StaticConsumeResult {
  // Record / kwarg-joint with `ir.fields = [{name, value: subIR}, …]`:
  // variate must be a record with each field's name; walk recursively.
  if (Array.isArray(ir.fields)) {
    if (!t || t.kind !== 'record' || !t.fields) {
      return { logpType: T.REAL, rest: t,
        error: 'joint(fields) expects a record variate, got ' + T.show(t) };
    }
    // For each declared field name, walk staticConsume against the
    // field's type. The field's full type is consumed by the
    // sub-measure (no inner rest expected).
    for (const f of ir.fields) {
      const ft = t.fields[f.name];
      if (ft == null) {
        return { logpType: T.REAL, rest: t,
          error: 'joint missing field "' + f.name + '" in variate type' };
      }
      const sub = staticConsume(f.value, ft);
      if (sub.error) return sub;
      // We require each field-measure to fully consume its field
      // type; otherwise the joint is malformed.
      if (sub.rest != null && !sub.deferred) {
        return { logpType: T.REAL, rest: t,
          error: 'joint field "' + f.name + '" left a non-empty rest' };
      }
    }
    return { logpType: T.REAL, rest: null };
  }
  // Positional joint (`joint(M1, M2, ...)`): variate must be a flat
  // vector (or rank-1 array) long enough to feed each Mi's
  // footprint in order. Walk each Mi consuming from a shrinking
  // rest. If any Mi can't statically determine its footprint, the
  // whole joint becomes deferred (consistent with value-mode
  // walker's runtime semantics — defer == "passes static, rely on
  // runtime").
  if (Array.isArray(ir.args) && ir.args.length > 0) {
    let cur: any = t;
    let anyDeferred = false;
    for (let i = 0; i < ir.args.length; i++) {
      const sub = staticConsume(ir.args[i], cur);
      if (sub.error) return sub;
      if (sub.deferred) anyDeferred = true;
      cur = sub.rest;
      if (cur == null) {
        // Component fully consumed the variate. If more components
        // follow, that's a static error (under-long variate).
        if (i + 1 < ir.args.length) {
          return { logpType: T.REAL, rest: null,
            error: 'joint: variate exhausted after component ' + (i + 1)
              + ' but ' + (ir.args.length - i - 1) + ' more remain' };
        }
      }
    }
    return { logpType: T.REAL, rest: cur, deferred: anyDeferred };
  }
  return { logpType: T.REAL, rest: null, deferred: true };
}

function _consumeIid(ir: any, t: any): StaticConsumeResult {
  const args = ir.args || [];
  if (args.length !== 2) {
    return { logpType: T.REAL, rest: t,
      error: 'iid expects 2 args, got ' + args.length };
  }
  const M = args[0];
  // We need n: the second arg's literal-int or const-eval value.
  // Without a const-eval resolver here, only literal n works. The
  // user-facing pass at the analyser layer would wire one in.
  const sizeIR = args[1];
  let n: number | null = null;
  if (sizeIR && sizeIR.kind === 'lit' && Number.isInteger(sizeIR.value)) {
    n = sizeIR.value;
  }
  // The variate type must be array(rank>=1, [n, ...], elem) where
  // elem matches M's variate type. v0.1: defer the full elem check;
  // just verify the leading axis matches.
  if (!t || t.kind !== 'array') {
    if (t && (t.kind === 'deferred' || t.kind === 'any')) {
      return { logpType: T.REAL, rest: null, deferred: true };
    }
    return { logpType: T.REAL, rest: t,
      error: 'iid expects array variate, got ' + T.show(t) };
  }
  if (n == null) {
    // Dynamic n: leading axis can be anything; defer the empty-rest
    // check to runtime.
    return { logpType: T.REAL, rest: null, deferred: true };
  }
  const lead = Array.isArray(t.shape) ? t.shape[0] : null;
  if (typeof lead === 'number' && lead !== n) {
    return { logpType: T.REAL, rest: t,
      error: 'iid expects ' + n + ' elements, got variate with leading axis '
        + lead };
  }
  return { logpType: T.REAL, rest: null, deferred: typeof lead !== 'number' };
}

// pushfwd(f, M) — variate type at the surface is f's codomain (the
// type of `f(x)` for x : domain). Recursively check M with f's
// DOMAIN type — what M actually produces. Pure bijections preserve
// rank/structure, so the common scalar↔scalar and vector↔vector
// cases can be checked statically by reading f's body's inferred
// type. Higher-rank or non-typed bijections fall through to
// deferred.
function _consumePushfwd(ir: any, variateType: any): StaticConsumeResult {
  const args = ir.args || [];
  if (args.length < 2) {
    return { logpType: T.REAL, rest: variateType, deferred: true };
  }
  const baseIR = args[1];
  // Read the bijection's codomain from its forward fn's body type if
  // available. The forward fn lives on `ir.bijection.f` (mirror of
  // `fInv` shape) when the lowerer annotated it. Without an
  // annotation we can't tell what shape f produces; defer.
  const bij = ir.bijection;
  if (!bij || !bij.f || !bij.f.body) {
    return staticConsume(baseIR, variateType);
  }
  const fBodyT = bij.f.body && bij.f.body.meta && bij.f.body.meta.type;
  // Most common case: scalar in / scalar out. Both M and the
  // surface variate should be scalar-shaped; the consume check
  // reduces to "recurse into M with the same variate type" (the
  // pushfwd is shape-preserving — only the value-mode logvolume
  // term shifts the density).
  if (fBodyT && fBodyT.kind === 'scalar'
      && variateType && variateType.kind === 'scalar') {
    return staticConsume(baseIR, variateType);
  }
  // Higher-rank / structured bijections — defer until the
  // bijection annotation carries explicit domain/codomain types.
  return { logpType: T.REAL, rest: null, deferred: true };
}

// Walk each branch of a `select` against the same variate type.
// Per spec §06, every branch shares one variate space — they must
// all produce the SAME rest. Mismatch = static error.
function _consumeSelect(ir: any, t: any): StaticConsumeResult {
  const branches = ir.branches;
  if (!Array.isArray(branches) || branches.length === 0) {
    return { logpType: T.REAL, rest: t, deferred: true };
  }
  let firstRest: any = undefined;
  let firstSize = -1;
  let anyDeferred = false;
  for (let k = 0; k < branches.length; k++) {
    const b = branches[k];
    // Branch shape: `{ ir }` (inline measure) or `{ ref }` (binding).
    // We can statically check only the `ir` form; refs need a binding
    // lookup we don't have here. Defer when refs appear.
    const branchIR = b && (b.ir || (b.ref ? null : b));
    if (!branchIR || typeof branchIR !== 'object' || branchIR.kind !== 'call') {
      anyDeferred = true;
      continue;
    }
    const sub = staticConsume(branchIR, t);
    if (sub.error) return sub;
    if (sub.deferred) { anyDeferred = true; continue; }
    const sz = _restSize(sub.rest);
    if (firstRest === undefined) {
      firstRest = sub.rest;
      firstSize = sz;
    } else if (sz !== firstSize) {
      return { logpType: T.REAL, rest: t,
        error: 'select: branch ' + k + ' consumes a different footprint '
          + 'than earlier branches (one variate space per spec §06)' };
    }
  }
  return { logpType: T.REAL,
    rest: firstRest === undefined ? null : firstRest,
    deferred: anyDeferred };
}

// Rough "footprint size" of a static rest, used only for select's
// per-branch agreement check. null/scalar → 0; array → leading-dim
// length; record → number of remaining fields.
function _restSize(r: any): number {
  if (r == null) return 0;
  if (r.kind === 'scalar') return 1;
  if (r.kind === 'array' && Array.isArray(r.shape)) {
    return typeof r.shape[0] === 'number' ? r.shape[0] : -1;
  }
  if (r.kind === 'record' && r.fields) return Object.keys(r.fields).length;
  return -1;
}

/**
 * Strict wrapper: `staticConsume` + assert empty-rest. Returns an
 * error string if the measure's IR and the variate type are
 * statically incompatible, null when the check passes (or is
 * deferred to runtime).
 */
function staticDensityShapeCheck(ir: any, variateType: any): string | null {
  const r = staticConsume(ir, variateType);
  if (r.error) return r.error;
  if (r.deferred) return null;   // can't statically prove; trust runtime
  if (r.rest != null) {
    return 'measure did not fully consume variate type (rest: '
      + T.show(r.rest) + ')';
  }
  return null;
}

// =====================================================================
// Chain composition: consume/rest at the chain's input-set level
// (engine-concepts §19.4)
// =====================================================================
//
// fchain, kchain, jointchain all share the same structural pattern: a
// left-to-right fold over per-step typed composition. The shape is
// consume/rest at the chain's open-input set — each step "consumes"
// the prior step's output; empty rest at end ⇒ closed (measure or
// applied function); non-empty rest ⇒ residual inputs the chain still
// needs.
//
// This helper takes PRE-INFERRED step types (caller has already done
// inferExpr per step) and returns the chain's composed result type +
// step-boundary diagnostics. It does not walk IR — typeinfer / the
// classifier feeds it types.

/** A single chain step, with optional source info for diagnostics. */
interface ChainStep {
  /** The inferred type of the step expression. Must be `funcType` for
   *  mode='func'; `measureType` / `kernelType` for the kernel modes
   *  (Phase 2; currently rejected with `not yet implemented`). */
  type:  any;
  /** Source location for step-boundary diagnostics. Optional but
   *  strongly recommended; the diagnostic points here. */
  loc?:  any;
  /** Binding name if the step is a ref to a named binding (else
   *  inline). Used in diagnostic text to disambiguate steps. */
  name?: string;
}

interface ChainCompositionResult {
  /** The chain's inferred result type. funcType for mode='func';
   *  kernelType (or measureType when residual collapses to ∅) for
   *  the kernel modes. `deferred()` when any step was deferred /
   *  `any` and the rest couldn't statically resolve. `failed(...)`
   *  on a static error (also reflected in `diagnostics`). */
  resultType:  any;
  /** Per-step-boundary errors anchored at the failing step's loc.
   *  Empty when the chain types cleanly. Caller pushes these into
   *  its own diagnostics stream. */
  diagnostics: any[];
}

/**
 * Match step `i`'s result type against step `i+1`'s input list.
 * Returns `{ ok: true, subst }` on success, `{ ok: false, reason }`
 * on a static mismatch. Auto-splatting per spec §04
 * sec:calling-convention: a record-typed result matches multi-input
 * step boundaries by field name. A single-input step boundary
 * matches positionally against any compatible result type.
 */
function _matchChainBoundary(prevResult: any, nextInputs: any[]) {
  // Single-input boundary: positional match, any compatible type.
  if (nextInputs.length === 1) {
    const s = T.unify(nextInputs[0].type, prevResult, new Map());
    if (s == null) {
      return { ok: false,
        reason: 'cannot unify previous step\'s result type ' + T.show(prevResult)
              + ' with next step\'s input "' + nextInputs[0].name
              + '" of type ' + T.show(nextInputs[0].type) };
    }
    return { ok: true, subst: s };
  }
  // Multi-input boundary: previous result must be record-typed, with
  // exactly the field names the next step's inputs declare. Auto-splat
  // per §04 sec:calling-convention.
  if (prevResult.kind === 'record') {
    const fieldNames = Object.keys(prevResult.fields);
    const inputNames = nextInputs.map(i => i.name);
    // Field-name set check: every input name must have a matching field;
    // surplus fields are an error (the next step has nothing to bind them to).
    for (const n of inputNames) {
      if (!Object.prototype.hasOwnProperty.call(prevResult.fields, n)) {
        return { ok: false,
          reason: 'auto-splat at step boundary: previous result record '
                + T.show(prevResult)
                + ' is missing field "' + n + '" expected by next step' };
      }
    }
    for (const n of fieldNames) {
      if (inputNames.indexOf(n) === -1) {
        return { ok: false,
          reason: 'auto-splat at step boundary: previous result record '
                + T.show(prevResult)
                + ' has extra field "' + n + '" with no matching next-step input' };
      }
    }
    let s: any = new Map();
    for (const inp of nextInputs) {
      s = T.unify(inp.type, prevResult.fields[inp.name], s);
      if (s == null) {
        return { ok: false,
          reason: 'auto-splat at step boundary: field "' + inp.name
                + '" type ' + T.show(prevResult.fields[inp.name])
                + ' does not unify with next-step input type '
                + T.show(inp.type) };
      }
    }
    return { ok: true, subst: s };
  }
  return { ok: false,
    reason: 'multi-input step boundary requires a record-typed previous result '
          + 'for auto-splatting (per spec §04 calling convention); got '
          + T.show(prevResult) };
}

function _stepLabel(step: ChainStep, index: number): string {
  if (step.name) return 'step ' + index + ' (`' + step.name + '`)';
  return 'step ' + index;
}

/**
 * Pull the variate space from a step's output. For a kernel step the
 * variate is `step.result.domain` (kernel returns a measure); for a
 * measure-typed step (chain base only) it's `step.domain`. Returns
 * null when the type isn't kernel- or measure-shaped (which surfaces
 * as a step-type error upstream).
 */
function _chainStepVariate(stepType: any): any {
  if (!stepType) return null;
  if (stepType.kind === 'kernel') {
    return stepType.result && stepType.result.kind === 'measure'
      ? stepType.result.domain : null;
  }
  if (stepType.kind === 'measure') return stepType.domain;
  return null;
}

/**
 * Internal — kernel-side chain composition (modes 'kchain-marginal'
 * and 'jointchain-retain'). Spec §06 dependent composition:
 * - step 0: measureType (closed-first) or kernelType (kernel-first).
 * - steps i≥1: kernelType with inputs matching step_{i-1}'s variate.
 * - result variate: last step's variate (kchain) OR joint of all
 *   step variates (jointchain).
 * - residual inputs: step_0's inputs if kernel-first, else ∅.
 *   Empty residual ⇒ measureType result (kernel↔measure collapse);
 *   non-empty ⇒ kernelType result.
 */
function _inferKernelChain(
  steps: ChainStep[],
  mode: 'kchain-marginal' | 'jointchain-retain',
  labels: string[] | null
): ChainCompositionResult {
  const diagnostics: any[] = [];
  // Step 0 may be measure (closed-first) or kernel (kernel-first).
  const s0 = steps[0];
  const s0kind = s0.type.kind;
  if (s0kind !== 'measure' && s0kind !== 'kernel') {
    diagnostics.push({
      severity: 'error',
      message: mode + ' step 0 (`' + (s0.name || '<inline>')
             + '`) must be a measure or kernel (got ' + T.show(s0.type) + ')',
      loc: s0.loc,
    });
    return { resultType: T.failed(mode + ' bad step 0'), diagnostics };
  }
  // Steps i ≥ 1: kernel by spec §06 line 223-225, BUT per the kernel↔
  // measure unification (§06 line 86-91), a measure-typed step IS a
  // nullary kernel and is accepted here as a closed-kernel step. (The
  // "non-nullary" spec rule is intentionally not enforced at type-
  // check time — existing engine fixtures rely on the looseness, and
  // the structural inputs-vs-variate match below already catches the
  // shape errors that matter for downstream materialisation.)
  for (let i = 1; i < steps.length; i++) {
    const k = steps[i].type.kind;
    if (k !== 'kernel' && k !== 'measure') {
      diagnostics.push({
        severity: 'error',
        message: mode + ' ' + _stepLabel(steps[i], i)
               + ' must be a kernel or measure (got '
               + T.show(steps[i].type) + ')',
        loc: steps[i].loc,
      });
      return { resultType: T.failed(mode + ' non-kernel step'), diagnostics };
    }
  }
  // Boundary walk: step_i's variate → step_{i+1}'s inputs. Same matcher
  // as fchain's, applied at the value level (the measure's domain).
  // A nullary next-step (measure-typed or kernel with empty inputs) is
  // a closed kernel — no boundary match required; the next step
  // produces a measure regardless of the prior step's variate.
  for (let i = 0; i < steps.length - 1; i++) {
    const next = steps[i + 1];
    const nextInputs = (next.type.kind === 'kernel' && next.type.inputs) || [];
    if (nextInputs.length === 0) continue;     // closed kernel — no binding
    const prevVariate = _chainStepVariate(steps[i].type);
    if (prevVariate == null) {
      diagnostics.push({
        severity: 'error',
        message: mode + ' ' + _stepLabel(steps[i], i)
               + ' has no resolvable variate space',
        loc: steps[i].loc,
      });
      return { resultType: T.failed(mode + ' unresolved variate'), diagnostics };
    }
    const m = _matchChainBoundary(prevVariate, nextInputs);
    if (!m.ok) {
      diagnostics.push({
        severity: 'error',
        message: mode + ' step boundary ' + i + ' → ' + (i + 1) + ': '
               + m.reason
               + ' (from ' + _stepLabel(steps[i], i) + ' to '
               + _stepLabel(next, i + 1) + ')',
        loc: next.loc || steps[i].loc,
      });
      return { resultType: T.failed(mode + ' step-boundary mismatch'),
               diagnostics };
    }
  }
  // Compute the chain's variate.
  let resultVariate: any;
  if (mode === 'kchain-marginal') {
    // kchain: only the last step's variate survives.
    resultVariate = _chainStepVariate(steps[steps.length - 1].type);
    if (resultVariate == null) {
      // Should never trip — boundary walk above would have caught this
      // — but a defensive fallback keeps the helper total.
      resultVariate = T.deferred();
    }
  } else {
    // jointchain: retain every step's variate. Keyword form (labels)
    // → record; positional form → tuple (spec §06 line 252's
    // `cat(...variates...)` is array-of-array semantically; the type
    // system models heterogeneous positional retention as tuple,
    // since arrays require homogeneous element types per §03).
    const stepVariates = steps.map(s => _chainStepVariate(s.type));
    if (labels && labels.length === steps.length) {
      const fields: Record<string, any> = {};
      for (let i = 0; i < labels.length; i++) {
        fields[labels[i]] = stepVariates[i];
      }
      resultVariate = T.record(fields);
    } else {
      resultVariate = T.tuple(stepVariates);
    }
  }
  // Residual inputs: step 0's inputs survive only if kernel-first.
  // measure-first chain ⇒ empty residual ⇒ measureType result. Spec
  // §06 line 87-90 collapse falls out without a special case.
  const residual = s0kind === 'kernel' ? s0.type.inputs.slice() : [];
  const result = residual.length === 0
    ? T.measure(resultVariate)
    : T.kernelType(residual, T.measure(resultVariate));
  return { resultType: result, diagnostics };
}

/**
 * Compose a typed chain of steps left-to-right. See engine-concepts
 * §19.4 for the design.
 *
 * `mode` selects the per-step rule:
 *   - 'func'              — fchain. Steps are funcType; each step's
 *                            result feeds the next step's inputs
 *                            (with auto-splat for multi-input next-
 *                            step boundaries). Result type is
 *                            funcType(step_0.inputs, step_N.result).
 *   - 'kchain-marginal'   — kchain. Step 0 is measure or kernel;
 *                            steps i≥1 are kernels whose inputs match
 *                            step_{i-1}'s variate space. Intermediate
 *                            variates marginalised. Result variate =
 *                            last step's variate. Kernel-first →
 *                            kernelType(step_0.inputs, …); measure-
 *                            first → measureType(…).
 *   - 'jointchain-retain' — jointchain. Same per-step rule as
 *                            kchain; result variate retains every
 *                            step's variate. `opts.labels` (kwarg
 *                            form) → record-typed result; null →
 *                            tuple-typed (positional `cat` of
 *                            variates, spec §06 line 252).
 *
 * `opts.labels`: optional array of step labels (kwarg-form jointchain
 * surfaces these as field names). Used only by 'jointchain-retain'.
 */
function inferChainComposition(
  steps: ChainStep[],
  mode: 'func' | 'kchain-marginal' | 'jointchain-retain',
  opts?: { labels?: string[] | null }
): ChainCompositionResult {
  if (!Array.isArray(steps) || steps.length === 0) {
    return { resultType: T.failed('chain composition requires ≥ 1 step'),
             diagnostics: [] };
  }
  // Defer if any step's type is unresolved — typeinfer trusts runtime
  // for these. A deferred chain can still be referenced; we only
  // refuse to commit to a result type.
  for (const s of steps) {
    if (!s.type) {
      return { resultType: T.deferred(), diagnostics: [] };
    }
    if (s.type.kind === 'deferred' || s.type.kind === 'any') {
      return { resultType: T.deferred(), diagnostics: [] };
    }
    if (s.type.kind === 'failed') {
      return { resultType: T.failed('chain cascade'), diagnostics: [] };
    }
  }
  if (mode === 'kchain-marginal' || mode === 'jointchain-retain') {
    return _inferKernelChain(steps, mode, (opts && opts.labels) || null);
  }
  // mode === 'func' — every step must be a funcType.
  const diagnostics: any[] = [];
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i];
    if (s.type.kind !== 'function') {
      diagnostics.push({
        severity: 'error',
        message: 'fchain ' + _stepLabel(s, i)
               + ' is not a function (got ' + T.show(s.type) + ')',
        loc: s.loc,
      });
      return { resultType: T.failed('fchain non-function step'), diagnostics };
    }
  }
  // Single-step fchain ≡ the step itself (per spec, fchain is left-
  // associative and the identity case is `fchain(f) ≡ f`). Engine
  // tests already cover this for the applied form via inlineFchain.
  if (steps.length === 1) {
    return { resultType: steps[0].type, diagnostics: [] };
  }
  // Walk step boundaries left-to-right.
  for (let i = 0; i < steps.length - 1; i++) {
    const prev = steps[i];
    const next = steps[i + 1];
    const prevResult = prev.type.result;
    const nextInputs = next.type.inputs || [];
    if (nextInputs.length === 0) {
      diagnostics.push({
        severity: 'error',
        message: 'fchain ' + _stepLabel(next, i + 1)
               + ' has no inputs; the spec forbids nullary callables (§04 sec:calling-convention)',
        loc: next.loc || prev.loc,
      });
      return { resultType: T.failed('fchain nullary step'), diagnostics };
    }
    const m = _matchChainBoundary(prevResult, nextInputs);
    if (!m.ok) {
      diagnostics.push({
        severity: 'error',
        message: 'fchain step boundary ' + i + ' → ' + (i + 1) + ': ' + m.reason
               + ' (from ' + _stepLabel(prev, i) + ' to ' + _stepLabel(next, i + 1) + ')',
        loc: next.loc || prev.loc,
      });
      return { resultType: T.failed('fchain step-boundary mismatch'), diagnostics };
    }
  }
  // Chain types cleanly. Result: take step 0's inputs, step N-1's result.
  const result = T.funcType(steps[0].type.inputs.slice(),
                            steps[steps.length - 1].type.result);
  return { resultType: result, diagnostics: [] };
}

module.exports = {
  builtinLogdensityof,
  builtinLogdensityofPositional,
  // FlatPDL canonical transports (spec §07).
  builtinTouniform,
  builtinFromuniform,
  builtinTonormal,
  builtinFromnormal,
  isBuiltinKernel,
  isMultivariateKernel,
  multivariateShape,
  MV_DENSITY_FNS,
  // Type-mode consume/rest (engine-concepts §17.3)
  staticConsume,
  staticDensityShapeCheck,
  // Chain composition (engine-concepts §19.4) — consume/rest at the
  // chain's input-set level. Used by fchain (mode='func') and, in
  // Phase 2, by kchain / jointchain.
  inferChainComposition,
  // Test surface
  _internal: { logMvGamma, logDetSPD, traceProduct, _logCnLKJ,
               _normalCdf, _normalQuantile,
               _matchChainBoundary },
};
