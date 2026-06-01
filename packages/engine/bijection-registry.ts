'use strict';

// =====================================================================
// bijection-registry.ts — open registry of known bijections
// =====================================================================
//
// Architectural framing: flatppl-dev/flatppl-engine-concepts.md §22
// ("Multivariate distributions as derived measures"). Every practical
// multivariate distribution decomposes into pushfwd-of-iid through a
// known bijection: `MvNormal = pushfwd(affine, iid(Normal, D))`,
// `MvLogNormal = pushfwd(exp, MvNormal)`, `Dirichlet = pushfwd(stick-
// breaking, iid(Beta, D-1))`, etc. The registry is the single point of
// support for the bijection half of that decomposition — sample-side
// fast paths, density-side inverse + log|det J|, declared shape
// contract.
//
// This module is INTENDED to be load-bearing across the multivariate
// distribution surface. Adding a new bijection (exp/log, stick-breaking,
// Cholesky-pack, projection, …) is one registry entry; the matPushfwd
// fast path + density walker consult the registry by name and dispatch
// to the entry's hot code. No per-multivariate codepath in the engine.
//
// Phase 5.1 Session 1 lands:
//   - The registry surface (registerBijection / getBijection / has).
//   - The `affine` entry's sample-side atom-batched forward — the hot
//     path matMvNormal already implements inline today.
//   - matMvNormal refactored to consume this registry, demonstrating
//     §22's "matMvNormal is a thin shortcut over the canonical
//     decomposition" claim concretely.
//
// Phase 5.1 Session 2 also lands:
//   - The `affine` entry's density side — atom-batched inverse
//     (forward-substitute against lower-triangular L; elementwise for
//     diag-stored L) and logDetJ (-sum(log|diag(L)|), constant in y
//     for affine maps).
//
// Phase 5.1 follow-ups (later sessions):
//   - `matPushfwd` atom-aware fast path that pattern-matches
//     `pushfwd(<bijection-typed-binding>, iid(scalar, D))` and consults
//     the registry directly (no separate matMvNormal needed).
//   - `walkPushfwd` vector-base extension that consumes a D-vector
//     atom-batched and dispatches to the registry's
//     `atomBatchedInverse` + `logDetJ`.
//   - Surface lowering at lift time: `MvNormal(mu, Sigma)` →
//     `pushfwd(<affine-binding>, iid(Normal(0,1), D))`. Once landed, the
//     joint / nested-broadcast composite-body detectors can accept
//     multivariate components for free (the lowered form is already a
//     measure expression the materialiser handles).
//   - Additional bijection entries: exp / log, stick-breaking, Cholesky-
//     pack, projection (engine-concepts §22.4 Phase 5.x).

// ---------------------------------------------------------------------
// Registry shape — engine-concepts §22.3
// ---------------------------------------------------------------------

/**
 * One bijection registry entry. Each describes a closed-form invertible
 * map `f : Z → Y` plus the sample-side / density-side fast paths the
 * engine consults during pushfwd materialisation and density evaluation.
 *
 * Both `atomBatchedForward` and `atomBatchedInverse` operate on atom-
 * batched Values (shape `[N, ...inShape]` and `[N, ...outShape]`
 * respectively). Per-cell / per-atom params are passed through `params`;
 * the entry decides which params vary per atom and routes accordingly.
 *
 * `logDetJ` returns `log|det J_{f^{-1}}(y)|` per atom — i.e. the volume-
 * change factor consumed by `density.walkPushfwd` to score post-image
 * atoms against pre-image densities. For affine maps with fixed `L`,
 * the value is a constant (no per-atom variation); the registry signals
 * that by returning a plain number; for non-constant Jacobians it
 * returns a Float64Array of length N.
 */
interface BijectionEntry {
  /** Canonical name; used for registry lookup. */
  name: string;
  /**
   * Sample-side atom-batched forward map. Given a pre-image Value `z`
   * with leading atom axis of size `N`, return the post-image Value
   * with the same leading atom axis. `params` is bijection-specific
   * (e.g. `{L, b}` for affine). Used by matPushfwd's atom-aware fast
   * path and by surface-name shortcut materialisers (e.g. matMvNormal).
   */
  atomBatchedForward(z: any, params: any, N: number): any;
  /**
   * Density-side atom-batched inverse map. Given post-image atom-batched
   * `y` and `params`, return the pre-image atom-batched `z`. Used by
   * `density.walkPushfwd` when scoring observed data through the
   * bijection's inverse. Optional in Phase 5.1 Session 1 — added per
   * bijection as density-side support lands.
   */
  atomBatchedInverse?(y: any, params: any, N: number): any;
  /**
   * Atom-batched log|det J_{f^{-1}}(y)|. Returns either a scalar (when
   * the determinant doesn't depend on `y`, e.g. affine with fixed `L`)
   * or a Float64Array of length N. Optional in Phase 5.1 Session 1.
   */
  logDetJ?(y: any, params: any, N: number): number | Float64Array;
  /**
   * Declared shape contract. Engines / classifiers can read this to do
   * shape inference / cross-engine codegen without invoking the entry.
   * `inShape` / `outShape` strings name the bijection's logical shapes
   * (e.g. `'[D]'` for vector → vector affine, `'[]'` for scalar → scalar
   * exp/log, `'[D-1]→[D]'` for stick-breaking).
   */
  shapeContract: { inShape: string; outShape: string };
}

const BIJECTION_REGISTRY: Map<string, BijectionEntry> = new Map();

/** Register a bijection entry. Throws on duplicate `name`. */
function registerBijection(entry: BijectionEntry) {
  if (!entry || !entry.name) {
    throw new Error('bijection-registry: entry must have a name');
  }
  if (BIJECTION_REGISTRY.has(entry.name)) {
    throw new Error('bijection-registry: duplicate entry for ' + entry.name);
  }
  BIJECTION_REGISTRY.set(entry.name, entry);
}

/** Look up a bijection by name. Returns undefined when unregistered. */
function getBijection(name: string): BijectionEntry | undefined {
  return BIJECTION_REGISTRY.get(name);
}

/** Yes/no membership probe. */
function hasBijection(name: string): boolean {
  return BIJECTION_REGISTRY.has(name);
}

/** All registered names, in insertion order — for diagnostics / docs. */
function registeredNames(): string[] {
  return Array.from(BIJECTION_REGISTRY.keys());
}

// ---------------------------------------------------------------------
// Affine bijection — y = L·z + b
// ---------------------------------------------------------------------
//
// Per spec §06 / engine-concepts §22.1: the universal multivariate
// transform. `MvNormal(mu, Sigma) = pushfwd(affine(L, mu),
// iid(Normal(0, 1), D))` where `L = lower_cholesky(Sigma)` and the
// sample-side `forward` is `y = L·z + mu`. Same map handles MvStudentT
// (with iid StudentT base) and any user-defined affine pushforward.
//
// Params:
//   - `L`: rank-2 matrix (shape `[D, D]`). May be `valueLib.isDiagStored`
//     for the diag-only case — the dense path goes through ops.dispatch
//     (mul, atom-batched); the diag-stored path takes the existing
//     `valueOps.mulN` fast lane (matMvNormal's null-fallthrough
//     pre-check).
//   - `b`: rank-1 vector (shape `[D]`). Atom-independent through
//     Sessions 1+2 — per-atom `b` arrives with the matPushfwd
//     vector-base extension (Phase 5.1 Session 3+) so per-cell affine
//     pushforwards over outer-batch axes can land without changing the
//     registry contract.
//
// Sample-side atom-batched contract:
//   Input  z: shape `[N, D]` (post-iid stack of standard scalar draws).
//   Output y: shape `[N, D]` (atom-major, contiguous).

/**
 * Sample-side affine forward: y[n, :] = L · z[n, :] + b.
 *
 * Routes the matmul through `ops.dispatch('mul', [L, z], { atomN: N })`
 * for dense L and through `valueOps.mulN(L, z, N)` for the diag-stored
 * fast lane (matching the legacy matMvNormal pre-check; diag's null-
 * fallthrough semantics don't fit variant dispatch). The shift is
 * `ops.dispatch('add', [b, Lz], { atomN: N })` — the registry's
 * atom-aware variant routes to `_atomBroadcastBinop`.
 *
 * Exported in module form so density-side code (Phase 5.1 Session 2+)
 * can reuse the same diag/dense routing in its inverse path.
 */
function affineAtomBatchedForward(z: any, params: any, N: number): any {
  const ops      = require('./ops.ts');
  const valueOps = require('./value-ops.ts');
  const valueLib = require('./value.ts');
  const L = params.L;
  const b = params.b;
  if (!L || !b) {
    throw new Error('bijection.affine: params must include L and b');
  }
  // L·z — dense vs diag-stored split (matMvNormal historical fast lane).
  const Lz = (valueLib.isDiagStored && valueLib.isDiagStored(L))
    ? valueOps.mulN(L, z, N)
    : ops.dispatch('mul', [L, z], { atomN: N });
  // b + Lz, atom-batched broadcast of b ([D]) over Lz ([N, D]).
  return ops.dispatch('add', [b, Lz], { atomN: N });
}

// ---------------------------------------------------------------------
// Affine density side — y → z = L⁻¹·(y - b)
// ---------------------------------------------------------------------
//
// Density-side contract symmetric to the forward map:
//   - `atomBatchedInverse(y, {L, b}, N)` returns atom-batched
//     z = L⁻¹·(y - b). For Cholesky-positive lower-triangular L (the
//     MvNormal case) we forward-substitute against L per atom — O(N·D²)
//     and exact. Diag-stored L collapses to elementwise (y - b) / L.
//   - `logDetJ({L}, N)` returns log|det J_{f⁻¹}(y)| = -log|det(L)| =
//     -sum(log|diag(L)|). For affine maps the determinant doesn't
//     depend on `y`, so the contract returns a scalar (the registry
//     entry's return type is `number | Float64Array` — a scalar means
//     the same value for every atom).
//
// Both feed into walkPushfwd's vector-base extension (Phase 5.1
// Session 3+) and into matPushfwd's vector-base extension; both are
// already callable standalone so user code that constructs affine
// pushforwards through the registry can score against them today.

/**
 * Density-side affine inverse: z[n, :] = L⁻¹ · (y[n, :] - b).
 *
 * For dense lower-triangular L, solves L·z = (y - b) per atom by
 * forward-substitution. For `valueLib.isDiagStored` L, the inverse is
 * the elementwise reciprocal — `z = (y - b) / diag(L)`.
 *
 * Returns a fresh `[N, D]` Float64Array Value.
 */
function affineAtomBatchedInverse(y: any, params: any, N: number): any {
  const valueLib = require('./value.ts');
  const L = params.L;
  const b = params.b;
  if (!L || !b) {
    throw new Error('bijection.affine: params must include L and b');
  }
  if (!y || !y.shape || y.shape.length !== 2 || y.shape[0] !== N) {
    throw new Error('bijection.affine: inverse expects y of shape [N, D]; '
      + 'got ' + JSON.stringify(y && y.shape));
  }
  const D = y.shape[1];
  const yData = y.data;
  const bData = b.data;
  if (b.shape.length !== 1 || b.shape[0] !== D) {
    throw new Error('bijection.affine: b shape [' + (b.shape && b.shape.join(','))
      + '] mismatched against y\'s D=' + D);
  }
  const out = new Float64Array(N * D);
  const diag = valueLib.isDiagStored && valueLib.isDiagStored(L);
  if (diag) {
    // L diag-stored: z[n, i] = (y[n, i] - b[i]) / L[i].
    if (L.data.length !== D) {
      throw new Error('bijection.affine: diag-stored L of length '
        + L.data.length + ' mismatched against D=' + D);
    }
    for (let n = 0; n < N; n++) {
      const baseN = n * D;
      for (let i = 0; i < D; i++) {
        const Lii = L.data[i];
        if (!(Lii !== 0)) {
          throw new Error('bijection.affine: diag L has zero entry at index '
            + i + ' — inverse undefined');
        }
        out[baseN + i] = (yData[baseN + i] - bData[i]) / Lii;
      }
    }
    return { shape: [N, D], data: out };
  }
  // Dense lower-triangular L: forward-substitute per atom.
  // L·z = (y - b) ⇒ z[i] = ((y[i] - b[i]) - sum_{j<i} L[i,j]·z[j]) / L[i,i].
  if (L.shape.length !== 2 || L.shape[0] !== D || L.shape[1] !== D) {
    throw new Error('bijection.affine: dense L shape ['
      + (L.shape && L.shape.join(',')) + '] mismatched against D=' + D);
  }
  const Ld = L.data;
  for (let n = 0; n < N; n++) {
    const baseN = n * D;
    for (let i = 0; i < D; i++) {
      let acc = yData[baseN + i] - bData[i];
      for (let j = 0; j < i; j++) acc -= Ld[i * D + j] * out[baseN + j];
      const Lii = Ld[i * D + i];
      if (!(Lii !== 0)) {
        throw new Error('bijection.affine: dense L has zero diagonal at index '
          + i + ' — inverse undefined');
      }
      out[baseN + i] = acc / Lii;
    }
  }
  return { shape: [N, D], data: out };
}

/**
 * Density-side affine log|det J_{f⁻¹}(y)| = -log|det(L)| =
 * -sum(log|diag(L)|). Independent of `y` — returned as a scalar.
 *
 * `y` is accepted for signature symmetry with non-affine bijections;
 * it's unused (the Jacobian of an affine map is the constant L).
 */
function affineLogDetJ(_y: any, params: any, _N: number): number {
  const valueLib = require('./value.ts');
  const L = params.L;
  if (!L) {
    throw new Error('bijection.affine: logDetJ requires params.L');
  }
  let s = 0;
  if (valueLib.isDiagStored && valueLib.isDiagStored(L)) {
    for (let i = 0; i < L.data.length; i++) {
      const v = L.data[i];
      if (!(v !== 0)) {
        throw new Error('bijection.affine: diag L has zero entry at index '
          + i + ' — logDetJ undefined');
      }
      s += Math.log(Math.abs(v));
    }
  } else {
    if (L.shape.length !== 2 || L.shape[0] !== L.shape[1]) {
      throw new Error('bijection.affine: dense L must be square; got '
        + JSON.stringify(L.shape));
    }
    const D = L.shape[0];
    for (let i = 0; i < D; i++) {
      const v = L.data[i * D + i];
      if (!(v !== 0)) {
        throw new Error('bijection.affine: dense L has zero diagonal at index '
          + i + ' — logDetJ undefined');
      }
      s += Math.log(Math.abs(v));
    }
  }
  return -s;
}

registerBijection({
  name: 'affine',
  atomBatchedForward: affineAtomBatchedForward,
  atomBatchedInverse: affineAtomBatchedInverse,
  logDetJ: affineLogDetJ,
  shapeContract: { inShape: '[D]', outShape: '[D]' },
});

module.exports = {
  registerBijection,
  getBijection,
  hasBijection,
  registeredNames,
  // Direct handles so callers (matMvNormal today; matPushfwd tomorrow)
  // that know they want the affine entry can skip the lookup.
  affineAtomBatchedForward,
  affineAtomBatchedInverse,
  affineLogDetJ,
};
