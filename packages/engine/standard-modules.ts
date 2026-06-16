'use strict';

// =====================================================================
// Standard-module registry (spec §09)
// =====================================================================
//
// `standard_module(name, compat)` in surface FlatPPL resolves to a
// registry of engine-provided synthetic modules. Unlike `load_module`,
// which compiles user-authored `.flatppl` text, a standard module is
// implemented in JS — its bindings are engine functions exposed through
// a synthetic LoweredModule-shaped descriptor.
//
// Architectural decisions (TODO-flatppl-js.md "Multi-file models",
// locked 2026-05-10):
//   - Standard modules are engine-provided. The standard-module
//     category exists for capabilities that need engine support
//     (performance, primitives not expressible in base FlatPPL).
//   - Each entry is indexed by (name, compat). compat follows the same
//     two-part grammar as `flatppl_compat` (e.g. "0.1"). When multiple
//     versions are registered, lookup matches by major.minor compat.
//
// This file lands the REGISTRY SKELETON. Initial registry is empty;
// each standard module is added in its own follow-up commit (see TODO
// §09 — particle-physics, generalized-linear-models, ext-linear-algebra,
// special-functions, polynomials, distances).

// One entry per registered (name, compat) pair.
//
//   name        — module identifier (e.g. 'particle-physics')
//   compat      — version compat string (e.g. '0.1')
//   bindings    — Map<binding-name, BindingDescriptor>
//                 where BindingDescriptor is:
//                   { kind: 'function', impl: (...args) => any, sig?: Signature }
//                   { kind: 'distribution', /* TBD: spec for distribution refs */ }
//                   { kind: 'value', value: any }
//                 The classifier exposes the binding name through the
//                 standard-module's `get_field` access; runtime
//                 dispatch reads the descriptor's `impl`.
interface StandardModuleEntry {
  name: string;
  compat: string;
  bindings: Map<string, BindingDescriptor>;
}

interface BindingDescriptor {
  kind: 'function' | 'distribution' | 'value';
  /** For 'function': the JS callable. Surface form `mod.foo(args)`
   *  dispatches through this. */
  impl?: (...args: any[]) => any;
  /** For 'value': a precomputed JS value (number / Value / record). */
  value?: any;
  /** Optional type signature in the engine's types.ts SIGNATURE_FACTORIES
   *  shape. When omitted, typeinfer falls back to `deferred`. */
  sig?: any;
}

// The registry. Indexed by `${name}@${compat}` — exact-match for now;
// loose semver matching is deferred.
const _REGISTRY = new Map<string, StandardModuleEntry>();

/**
 * Look up a standard module by name + compat. Returns null when no
 * registered module matches.
 */
function lookupStandardModule(name: string, compat: string): StandardModuleEntry | null {
  const key = `${name}@${compat}`;
  return _REGISTRY.get(key) || null;
}

/**
 * Register a standard module. Called once at engine boot per module;
 * subsequent calls for the same (name, compat) replace the previous
 * entry (useful for tests).
 */
function registerStandardModule(entry: StandardModuleEntry): void {
  const key = `${entry.name}@${entry.compat}`;
  _REGISTRY.set(key, entry);
}

/**
 * List all registered standard modules. Used by the viewer's module
 * navigator and by introspection tests.
 */
function listStandardModules(): Array<{ name: string; compat: string }> {
  const out: Array<{ name: string; compat: string }> = [];
  for (const entry of _REGISTRY.values()) {
    out.push({ name: entry.name, compat: entry.compat });
  }
  return out;
}

/**
 * Test-only: clear all registered modules. Used by test setup to
 * avoid cross-test pollution.
 */
function _clearStandardModules(): void {
  _REGISTRY.clear();
}

// =====================================================================
// Built-in standard-module registrations
// =====================================================================
//
// Each standard module is registered as one entry below. Adding a new
// binding to a module: extend that module's `bindings` map. Adding a
// new module: copy a `_register*` function and call it from
// `_registerBuiltinStandardModules`.
//
// Conventions:
//   - Signatures use the engine's types.ts factories (funcType,
//     kernelType, scalar(...)).
//   - Functions take their arguments in the SAME ORDER as
//     sig.inputs — the standard-module dispatcher matches actual
//     kwargs / positional args to that order before invoking.
//   - Complex-valued functions return `{re, im}` JS objects matching
//     sampler-complex.ts's scalar convention.

const T = require('./types.ts');

// --- polynomials ------------------------------------------------------
//
// `chebyshev(n, x)` — Chebyshev polynomial of the first kind T_n(x),
// computed via the three-term recurrence
//
//   T_0(x) = 1,   T_1(x) = x,   T_n(x) = 2x · T_{n-1}(x) − T_{n-2}(x).
//
// Numerically stable for moderate n (no fast cancellation for typical
// inputs); for very large n / |x| > 1 the recurrence is still well-
// conditioned for T_n itself (it's bounded by 1 inside [-1, 1] and
// grows polynomially outside). spec §09 polynomials module.

function _chebyshev(n: number, x: number): number {
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
    throw new Error('chebyshev: n must be a non-negative integer (got ' + n + ')');
  }
  if (n === 0) return 1;
  if (n === 1) return x;
  let prev = 1, cur = x;
  for (let k = 2; k <= n; k++) {
    const next = 2 * x * cur - prev;
    prev = cur;
    cur = next;
  }
  return cur;
}

// `legendre(n, x)` — Legendre polynomial P_n(x), three-term recurrence
//
//   P_0 = 1,  P_1 = x,  n·P_n = (2n−1)·x·P_{n−1} − (n−1)·P_{n−2}.
//
// (spec §09; oracle: matches the explicit closed forms P_2=(3x²−1)/2,
// P_3=(5x³−3x)/2, P_4=(35x⁴−30x²+3)/8 to machine precision.)
function _legendre(n: number, x: number): number {
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
    throw new Error('legendre: n must be a non-negative integer (got ' + n + ')');
  }
  if (n === 0) return 1;
  if (n === 1) return x;
  let prev = 1, cur = x;
  for (let k = 2; k <= n; k++) {
    const next = ((2 * k - 1) * x * cur - (k - 1) * prev) / k;
    prev = cur;
    cur = next;
  }
  return cur;
}

// `hermite(n, x)` — physicist's Hermite polynomial H_n(x) (spec §09
// names the physicist's variant explicitly), three-term recurrence
//
//   H_0 = 1,  H_1 = 2x,  H_n = 2x·H_{n−1} − 2(n−1)·H_{n−2}.
//
// (oracle: matches H_2=4x²−2, H_3=8x³−12x, H_4=16x⁴−48x²+12.)
function _hermite(n: number, x: number): number {
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
    throw new Error('hermite: n must be a non-negative integer (got ' + n + ')');
  }
  if (n === 0) return 1;
  if (n === 1) return 2 * x;
  let prev = 1, cur = 2 * x;
  for (let k = 2; k <= n; k++) {
    const next = 2 * x * cur - 2 * (k - 1) * prev;
    prev = cur;
    cur = next;
  }
  return cur;
}

// `laguerre(n, x)` — Laguerre polynomial L_n(x), three-term recurrence
//
//   L_0 = 1,  L_1 = 1−x,  n·L_n = (2n−1−x)·L_{n−1} − (n−1)·L_{n−2}.
//
// (oracle: matches L_2=(x²−4x+2)/2, L_3=(−x³+9x²−18x+6)/6.)
function _laguerre(n: number, x: number): number {
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
    throw new Error('laguerre: n must be a non-negative integer (got ' + n + ')');
  }
  if (n === 0) return 1;
  if (n === 1) return 1 - x;
  let prev = 1, cur = 1 - x;
  for (let k = 2; k <= n; k++) {
    const next = ((2 * k - 1 - x) * cur - (k - 1) * prev) / k;
    prev = cur;
    cur = next;
  }
  return cur;
}

function _registerPolynomials() {
  const bindings = new Map<string, BindingDescriptor>();
  const polySig = () => T.funcType(
    [{ name: 'n', type: T.INTEGER }, { name: 'x', type: T.REAL }],
    T.REAL,
  );
  bindings.set('chebyshev', { kind: 'function', sig: polySig(), impl: _chebyshev });
  bindings.set('legendre',  { kind: 'function', sig: polySig(), impl: _legendre });
  bindings.set('hermite',   { kind: 'function', sig: polySig(), impl: _hermite });
  bindings.set('laguerre',  { kind: 'function', sig: polySig(), impl: _laguerre });
  registerStandardModule({
    name: 'polynomials',
    compat: '0.1',
    bindings,
  });
}

// --- particle-physics -------------------------------------------------
//
// `resonance_breitwigner(sigma, m, width, ma, mb, l, d)` — complex-
// valued relativistic Breit-Wigner amplitude with a mass-dependent
// width for a two-body decay R → a b with orbital angular momentum
// l, per spec §09 particle-physics module + Section 50 of Navas et al.
// (2024).
//
// Definition:
//   BW(σ) = 1 / (m² − σ − i · m · Γ(σ))
//
// with mass-dependent width
//   Γ(σ) = Γ · (m/√σ) · (p(σ)/p_0) · (F_l(p(σ)) / F_l(p_0))²
//
// where p(σ) is the breakup momentum, F_l is the Blatt-Weisskopf
// barrier factor, and p_0 = p(m²). Auxiliary helpers below.

// Källén function λ(x, y, z) = x² + y² + z² − 2xy − 2yz − 2xz.
function _kaellen(x: number, y: number, z: number): number {
  return x * x + y * y + z * z - 2 * (x * y + y * z + x * z);
}

// Breakup momentum p(σ) for R → a b at invariant mass-squared σ. For
// σ above threshold (σ ≥ (ma + mb)²) the momentum is real-positive;
// below threshold it's analytically continued (the engine returns the
// real magnitude; sub-threshold behaviour is not load-bearing for the
// canonical-mass-region resonance peak the function represents).
function _breakupMomentum(sigma: number, ma: number, mb: number): number {
  const lam = _kaellen(sigma, ma * ma, mb * mb);
  // Sub-threshold: clamp to zero (the BW form's denominator stays
  // finite via the iΓ imaginary part).
  if (lam <= 0) return 0;
  return Math.sqrt(lam) / (2 * Math.sqrt(sigma));
}

// Blatt-Weisskopf squared form factor F_l(p)² in the dimensionless
// variable z = (p · d)². PDG convention; tabulated for l = 0..4.
// Higher l values throw — extending the table is mechanical when a
// consumer needs them.
function _blattWeisskopfSquared(l: number, pd2: number): number {
  switch (l) {
    case 0: return 1;
    case 1: return 1 / (1 + pd2);
    case 2: return 1 / (9 + 3 * pd2 + pd2 * pd2);
    case 3: {
      const z2 = pd2 * pd2, z3 = z2 * pd2;
      return 1 / (225 + 45 * pd2 + 6 * z2 + z3);
    }
    case 4: {
      const z2 = pd2 * pd2, z3 = z2 * pd2, z4 = z3 * pd2;
      return 1 / (11025 + 1575 * pd2 + 135 * z2 + 10 * z3 + z4);
    }
    default:
      throw new Error('resonance_breitwigner: Blatt-Weisskopf factor for l='
        + l + ' not implemented (supported: 0..4)');
  }
}

function _resonanceBreitwigner(
  sigma: number, m: number, width: number,
  ma: number, mb: number, l: number, d: number,
): { re: number; im: number } {
  if (!Number.isFinite(sigma) || sigma <= 0) {
    return { re: NaN, im: NaN };  // outside the support
  }
  const m2 = m * m;
  const p = _breakupMomentum(sigma, ma, mb);
  const p0 = _breakupMomentum(m2, ma, mb);
  // F_l(p)² / F_l(p0)² ratio in the mass-dependent width. The factor
  // d appears squared in pd2; p · d has units of (energy · length).
  const pd2 = (p * d) * (p * d);
  const p0d2 = (p0 * d) * (p0 * d);
  const F2  = _blattWeisskopfSquared(l, pd2);
  const F02 = _blattWeisskopfSquared(l, p0d2);
  // Γ(σ) — guard against p0 = 0 (massless daughters at threshold);
  // in the degenerate ℓ=0, ma=mb=0 case Γ(σ) collapses to Γ as the
  // ratios cancel exactly (spec §09 closing note). The reformulation
  // below preserves that limit via `(m/√σ) · (p · F²) / (p0 · F0²)`
  // — the factors cancel when ma=mb=0, l=0.
  const sqrtSigma = Math.sqrt(sigma);
  let gammaSigma: number;
  if (p0 === 0) {
    // Threshold-only resonance — fall back to mass-independent width.
    gammaSigma = width;
  } else {
    gammaSigma = width * (m / sqrtSigma) * (p / p0) * (F2 / F02);
  }
  // BW(σ) = 1 / (m² − σ − i · m · Γ(σ)). Real denominator (m²-σ),
  // imag denominator (−m·Γ). Reciprocal of a complex (a − ib) is
  // (a + ib) / (a² + b²).
  const reDen = m2 - sigma;
  const imDen = -m * gammaSigma;
  const norm = reDen * reDen + imDen * imDen;
  return { re: reDen / norm, im: -imDen / norm };
}

// --- spec-surfaced particle-physics helpers (§09) --------------------
// `kallen` is `_kaellen` directly. The spec's `breakup_momentum(m, ma, mb)`
// takes the invariant mass m (not σ=m²) that `_breakupMomentum` expects, and
// `blatt_weisskopf(l, p, d)` is F_ℓ = √(zˡ/χ_ℓ(z)) with z=(d·p)², whereas
// `_blattWeisskopfSquared` is 1/χ_ℓ(z) (the zˡ numerator is supplied here).
function _breakupMomentumMass(m: number, ma: number, mb: number): number {
  return _breakupMomentum(m * m, ma, mb);
}
function _blattWeisskopf(l: number, p: number, d: number): number {
  const z = (d * p) * (d * p);
  return Math.sqrt(Math.pow(z, l) * _blattWeisskopfSquared(l, z));
}

// --- systematic-interpolation functions (spec §09 hepphys.interp_*) ----
// All share the signature (left, center, right, alpha): anchor outputs at
// alpha = −1, 0, +1, evaluated at alpha. These match the pyhf interpcodes
// (verified against pyhf's interpolators to machine precision):
//   pwlin → code0, pwexp → code1, poly2_lin → code2,
//   poly6_lin → code4p (histosys default), poly6_exp → code4 (normsys default).

function _interpPwlin(left: number, center: number, right: number, alpha: number): number {
  return alpha >= 0 ? center + alpha * (right - center)
                    : center + alpha * (center - left);
}
function _interpPwexp(left: number, center: number, right: number, alpha: number): number {
  // interp_pwlin applied in log-space (requires left, center, right > 0).
  return Math.exp(_interpPwlin(Math.log(left), Math.log(center), Math.log(right), alpha));
}
function _interpPoly2Lin(left: number, center: number, right: number, alpha: number): number {
  const S = (right - left) / 2;
  const A = (right + left) / 2 - center;
  if (alpha > 1) return (center + S + A) + (alpha - 1) * (S + 2 * A);
  if (alpha < -1) return (center - S + A) + (alpha + 1) * (S - 2 * A);
  return center + S * alpha + A * alpha * alpha;
}
// 6th-order interpolation: A⁻¹ (pyhf code4/code4p, α0 = 1) maps the boundary
// vector b to the polynomial coefficients a₁…a₆ for |α| < 1.
const _POLY6_AINV = [
  [15 / 16, -15 / 16, -7 / 16, -7 / 16, 1 / 16, -1 / 16],
  [3 / 2, 3 / 2, -9 / 16, 9 / 16, 1 / 16, 1 / 16],
  [-5 / 8, 5 / 8, 5 / 8, 5 / 8, -1 / 8, 1 / 8],
  [-3 / 2, -3 / 2, 7 / 8, -7 / 8, -1 / 8, -1 / 8],
  [3 / 16, -3 / 16, -3 / 16, -3 / 16, 1 / 16, -1 / 16],
  [1 / 2, 1 / 2, -5 / 16, 5 / 16, 1 / 16, 1 / 16],
];
// pyhf code4: multiplicative factor, polynomial inside [−1,1], EXPONENTIAL
// continuation outside. f(α) = center · I(α) with I built from the ratios.
function _interpPoly6Exp(left: number, center: number, right: number, alpha: number): number {
  const du = right / center, dd = left / center;
  if (alpha >= 1) return center * Math.pow(du, alpha);
  if (alpha <= -1) return center * Math.pow(dd, -alpha);
  const ldu = Math.log(du), ldd = Math.log(dd);
  const b = [du - 1, dd - 1, ldu * du, -ldd * dd, ldu * ldu * du, ldd * ldd * dd];
  let fac = 1;
  for (let i = 0; i < 6; i++) {
    let ci = 0;
    for (let j = 0; j < 6; j++) ci += _POLY6_AINV[i][j] * b[j];
    fac += ci * Math.pow(alpha, i + 1);
  }
  return center * fac;
}
// pyhf code4p: ADDITIVE delta, polynomial inside [−1,1], LINEAR continuation
// outside. f(α) = center + Δ(α).
function _interpPoly6Lin(left: number, center: number, right: number, alpha: number): number {
  const du = right - center, dd = center - left;
  if (alpha > 1) return center + alpha * du;
  if (alpha < -1) return center + alpha * dd;
  const S = 0.5 * (du + dd), A = 0.0625 * (du - dd);
  const a2 = alpha * alpha;
  return center + alpha * S + A * (3 * a2 * a2 * a2 - 10 * a2 * a2 + 15 * a2);
}

// The interp functions are scalar in the spec §09 signature, but HistFactory
// applies them to PER-BIN histogram vectors (spec §12 histosys: vector
// anchors, scalar α). Wrap the scalar core so a Value/array anchor broadcasts
// elementwise (scalar args broadcast across bins), returning a Value vector;
// all-scalar calls (normsys: scalar lo/hi factors) stay scalar.
const _valueLib = require('./value.ts');
function _interpElementwise(
  core: (l: number, c: number, r: number, a: number) => number,
): (l: any, c: any, r: any, a: any) => any {
  const asArr = (x: any): number[] | null =>
    _valueLib.isValue(x) ? Array.from(x.data as Float64Array)
      : (Array.isArray(x) ? x : null);
  return (left, center, right, alpha) => {
    const aL = asArr(left), aC = asArr(center), aR = asArr(right), aA = asArr(alpha);
    if (!aL && !aC && !aR && !aA) return core(left, center, right, alpha);
    const n = (aL || aC || aR || aA)!.length;
    const el = (x: any, xa: number[] | null, i: number) => (xa ? xa[i] : x);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = core(el(left, aL, i), el(center, aC, i), el(right, aR, i), el(alpha, aA, i));
    }
    return _valueLib.vector(out);
  };
}

function _registerParticlePhysics() {
  const bindings = new Map<string, BindingDescriptor>();
  const interpSig = T.funcType(
    [{ name: 'left', type: T.REAL }, { name: 'center', type: T.REAL },
     { name: 'right', type: T.REAL }, { name: 'alpha', type: T.REAL }],
    T.REAL,
  );
  for (const [name, impl] of [
    ['interp_pwlin', _interpPwlin],
    ['interp_pwexp', _interpPwexp],
    ['interp_poly2_lin', _interpPoly2Lin],
    ['interp_poly6_lin', _interpPoly6Lin],
    ['interp_poly6_exp', _interpPoly6Exp],
  ] as Array<[string, (l: number, c: number, r: number, a: number) => number]>) {
    bindings.set(name, { kind: 'function', sig: interpSig, impl: _interpElementwise(impl) });
  }
  bindings.set('resonance_breitwigner', {
    kind: 'function',
    sig: T.funcType(
      [
        { name: 'sigma', type: T.REAL },
        { name: 'm',     type: T.REAL },
        { name: 'width', type: T.REAL },
        { name: 'ma',    type: T.REAL },
        { name: 'mb',    type: T.REAL },
        { name: 'l',     type: T.INTEGER },
        { name: 'd',     type: T.REAL },
      ],
      T.COMPLEX,
    ),
    impl: _resonanceBreitwigner,
  });
  bindings.set('kallen', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'x', type: T.REAL }, { name: 'y', type: T.REAL }, { name: 'z', type: T.REAL }],
      T.REAL,
    ),
    impl: _kaellen,
  });
  bindings.set('breakup_momentum', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'm', type: T.REAL }, { name: 'ma', type: T.REAL }, { name: 'mb', type: T.REAL }],
      T.REAL,
    ),
    impl: _breakupMomentumMass,
  });
  bindings.set('blatt_weisskopf', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'l', type: T.INTEGER }, { name: 'p', type: T.REAL }, { name: 'd', type: T.REAL }],
      T.REAL,
    ),
    impl: _blattWeisskopf,
  });
  // interp_* land here; the §09 distributions (CrystalBall, Argus, Voigtian,
  // …) live in the sampler REGISTRY (density layer) and dispatch by name.
  // Remaining deferred bindings: wignerd, and blatt_weisskopf ℓ=5..7 (the
  // implemented barrier polynomials cover ℓ=0..4).
  registerStandardModule({
    name: 'particle-physics',
    compat: '0.1',
    bindings,
  });
}

// --- ext-linear-algebra ----------------------------------------------
//
// Matrix factorisations, decompositions, and operations not in `base`.
// Spec §09. Multi-output decompositions return records.
//
// Initial coverage (M-sized commit landing the foundations):
//   - lu(A)        → record(P, L, U) with P·A = L·U
//   - kron(A, B)   → Kronecker tensor product
//   - matexp(A)    → matrix exponential via scaling + squaring Padé(13)
//
// Follow-up commits add qr / lstsq, symmetric eigen / eigmax / eigmin,
// svd / rank — each algorithm class as its own commit per the
// per-significant-step rule.

const extLinalg = require('./ext-linalg.ts');

function _registerExtLinearAlgebra() {
  const bindings = new Map<string, BindingDescriptor>();

  // For dynamic-shape matrix args + record returns, the signature uses
  // %dynamic dims and a record-of-matrices result type. The dispatcher
  // doesn't enforce concrete dims today — it threads the args through
  // the impl unchanged. Concrete shape validation happens inside the
  // impl (square-matrix checks, etc.).
  const dynMat = T.array(2, ['%dynamic', '%dynamic'], T.REAL);

  bindings.set('lu', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'A', type: dynMat }],
      T.record({ P: dynMat, L: dynMat, U: dynMat }),
    ),
    impl: extLinalg._lu,
  });

  bindings.set('kron', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'A', type: dynMat }, { name: 'B', type: dynMat }],
      dynMat,
    ),
    impl: extLinalg._kron,
  });

  bindings.set('matexp', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'A', type: dynMat }],
      dynMat,
    ),
    impl: extLinalg._matexp,
  });

  // qr(A) — Householder reflections, returns record(Q, R) per spec §07.
  // For an m×n matrix with m ≥ n, Q is m×n with orthonormal columns
  // and R is n×n upper-triangular such that A = Q · R.
  bindings.set('qr', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'A', type: dynMat }],
      T.record({ Q: dynMat, R: dynMat }),
    ),
    impl: extLinalg._qr,
  });

  // lstsq(A, b) — minimum-norm least-squares via QR. b is a length-m
  // vector; result x is a length-k vector minimising ||A·x - b||₂.
  const dynVec = T.array(1, ['%dynamic'], T.REAL);
  bindings.set('lstsq', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'A', type: dynMat }, { name: 'b', type: dynVec }],
      dynVec,
    ),
    impl: extLinalg._lstsq,
  });

  // eigen(A) — symmetric eigendecomposition via Jacobi rotations.
  // Returns record(values, vectors) where vectors' columns are the
  // orthonormal eigenvectors. Non-symmetric input throws (spec
  // allows complex eigenvalues but the matrix-return path doesn't
  // have a complex-matrix surface yet; deferred).
  bindings.set('eigen', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'A', type: dynMat }],
      T.record({ values: dynVec, vectors: dynMat }),
    ),
    impl: extLinalg._eigen,
  });

  // eigmax(A) / eigmin(A) — max/min eigenvalue of a symmetric A.
  bindings.set('eigmax', {
    kind: 'function',
    sig: T.funcType([{ name: 'A', type: dynMat }], T.REAL),
    impl: extLinalg._eigmax,
  });
  bindings.set('eigmin', {
    kind: 'function',
    sig: T.funcType([{ name: 'A', type: dynMat }], T.REAL),
    impl: extLinalg._eigmin,
  });

  // svd(A) — thin SVD via one-sided Jacobi rotations. For m × n A
  // with m ≥ n, returns record(U, S, V) with U (m×n orthonormal),
  // S (length-n descending singular values), V (n×n orthonormal),
  // satisfying A = U · diag(S) · V^T.
  bindings.set('svd', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'A', type: dynMat }],
      T.record({ U: dynMat, S: dynVec, V: dynMat }),
    ),
    impl: extLinalg._svd,
  });

  // rank(A) — numerical rank via SVD with the canonical
  //   tol = max(m, n) · σ_max · ε_machine
  // tolerance (matches NumPy / Octave defaults).
  bindings.set('rank', {
    kind: 'function',
    sig: T.funcType([{ name: 'A', type: dynMat }], T.INTEGER),
    impl: extLinalg._rank,
  });

  registerStandardModule({
    name: 'ext-linear-algebra',
    compat: '0.1',
    bindings,
  });
}

// Engine-startup registration: runs once on module require so the
// registry is populated before typeinfer / orchestrator / sampler
// consult it. Tests that need a clean slate (cross-test isolation)
// call `_clearStandardModules()` and re-register with bespoke
// descriptors via `registerStandardModule` directly.
function _registerBuiltinStandardModules() {
  _registerPolynomials();
  _registerParticlePhysics();
  _registerExtLinearAlgebra();
}

_registerBuiltinStandardModules();

module.exports = {
  lookupStandardModule,
  registerStandardModule,
  listStandardModules,
  _clearStandardModules,
  _registerBuiltinStandardModules,
  // Exported for tests
  _internal: {
    _chebyshev,
    _legendre,
    _hermite,
    _laguerre,
    _resonanceBreitwigner,
    _breakupMomentum,
    _breakupMomentumMass,
    _kaellen,
    _blattWeisskopfSquared,
    _blattWeisskopf,
  },
};
