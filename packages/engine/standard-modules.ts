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

function _registerPolynomials() {
  const bindings = new Map<string, BindingDescriptor>();
  bindings.set('chebyshev', {
    kind: 'function',
    sig: T.funcType(
      [{ name: 'n', type: T.INTEGER }, { name: 'x', type: T.REAL }],
      T.REAL,
    ),
    impl: _chebyshev,
  });
  // legendre / hermite / laguerre are also in the spec but deferred
  // until a consumer needs them.
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

function _registerParticlePhysics() {
  const bindings = new Map<string, BindingDescriptor>();
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
  // Other particle-physics bindings (interp_*, CrystalBall, Argus, …)
  // are deferred until a consumer needs them.
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
    _resonanceBreitwigner,
    _breakupMomentum,
    _kaellen,
    _blattWeisskopfSquared,
  },
};
