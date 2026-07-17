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
  // L·z. Three cases:
  //   - atom-BATCHED dense L [N, D, D] (5f-2 hierarchical scale): the
  //     generic `mul`-with-atomN dispatch does NOT do per-atom matvec
  //     for rank-3 L (it mis-shapes to [N, D, D-broadcast]); do the
  //     per-atom y[n,:] = L[n] · z[n,:] explicitly here.
  //   - diag-stored L (atom-indep): `valueOps.mulN` fast lane.
  //   - atom-INDEPENDENT dense L [D, D]: `ops.dispatch('mul', …,
  //     {atomN})` matvecs the shared L against every atom (the
  //     matMvNormal historical path).
  let Lz;
  if (!(valueLib.isDiagStored && valueLib.isDiagStored(L))
      && L.shape && L.shape.length === 3 && L.shape[0] === N) {
    const D = L.shape[1];
    if (L.shape[2] !== D) {
      throw new Error('bijection.affine: atom-batched L must be [N, D, D]; got '
        + JSON.stringify(L.shape));
    }
    if (!z || !z.shape || z.shape.length !== 2 || z.shape[0] !== N || z.shape[1] !== D) {
      throw new Error('bijection.affine: z must be [N, D]=[' + N + ',' + D
        + '] for atom-batched L; got ' + JSON.stringify(z && z.shape));
    }
    const out = new Float64Array(N * D);
    const Ld = L.data, zd = z.data;
    for (let n = 0; n < N; n++) {
      const baseN = n * D, Ln = n * D * D;
      for (let i = 0; i < D; i++) {
        let acc = 0;
        for (let j = 0; j < D; j++) acc += Ld[Ln + i * D + j] * zd[baseN + j];
        out[baseN + i] = acc;
      }
    }
    Lz = { shape: [N, D], data: out };
  } else {
    Lz = (valueLib.isDiagStored && valueLib.isDiagStored(L))
      ? valueOps.mulN(L, z, N)
      : ops.dispatch('mul', [L, z], { atomN: N });
  }
  // b + Lz, atom-aware: b may be atom-indep [D] (broadcast over [N, D])
  // OR atom-batched [N, D] (elementwise) — `add` with atomN handles both.
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
  // 5f-2: b may be atom-independent [D] (shared mean) OR atom-batched
  // [N, D] (per-atom mean, hierarchical priors). `bStride === 0` reuses
  // b[i] for every atom — making the atom-indep path numerically
  // IDENTICAL to the pre-5f-2 code (stride-0 reuse, no separate branch).
  const bAtomBatched = b.shape.length === 2 && b.shape[0] === N && b.shape[1] === D;
  const bAtomIndep   = b.shape.length === 1 && b.shape[0] === D;
  if (!bAtomBatched && !bAtomIndep) {
    throw new Error('bijection.affine: b shape [' + (b.shape && b.shape.join(','))
      + '] must be [D] or [N, D] for D=' + D + ', N=' + N);
  }
  const bStride = bAtomBatched ? D : 0;
  const out = new Float64Array(N * D);
  const diag = valueLib.isDiagStored && valueLib.isDiagStored(L);
  if (diag) {
    // L diag-stored (atom-indep only — diag format has no atom axis):
    // z[n, i] = (y[n, i] - b[n*bStride + i]) / L[i].
    if (L.data.length !== D) {
      throw new Error('bijection.affine: diag-stored L of length '
        + L.data.length + ' mismatched against D=' + D);
    }
    for (let n = 0; n < N; n++) {
      const baseN = n * D;
      const bn = n * bStride;
      for (let i = 0; i < D; i++) {
        const Lii = L.data[i];
        if (!(Lii !== 0)) {
          throw new Error('bijection.affine: diag L has zero entry at index '
            + i + ' — inverse undefined');
        }
        out[baseN + i] = (yData[baseN + i] - bData[bn + i]) / Lii;
      }
    }
    return { shape: [N, D], data: out };
  }
  // Dense lower-triangular L: forward-substitute per atom.
  // L·z = (y - b) ⇒ z[i] = ((y[i] - b[i]) - sum_{j<i} L[i,j]·z[j]) / L[i,i].
  //
  // 5f-2: L may be atom-independent [D, D] (shared scale) OR atom-batched
  // [N, D, D] (per-atom scale). `LStride === 0` reuses the single L for
  // every atom — the atom-indep path is byte-identical to pre-5f-2.
  const LAtomBatched = L.shape.length === 3 && L.shape[0] === N
    && L.shape[1] === D && L.shape[2] === D;
  const LAtomIndep   = L.shape.length === 2 && L.shape[0] === D && L.shape[1] === D;
  if (!LAtomBatched && !LAtomIndep) {
    throw new Error('bijection.affine: dense L shape ['
      + (L.shape && L.shape.join(',')) + '] must be [D, D] or [N, D, D] for D='
      + D + ', N=' + N);
  }
  const LStride = LAtomBatched ? D * D : 0;
  const Ld = L.data;
  for (let n = 0; n < N; n++) {
    const baseN = n * D;   // row offset into yData / out (the solution)
    const Ln = n * LStride;   // 0 for atom-indep L
    const bn = n * bStride;   // 0 for atom-indep b
    for (let i = 0; i < D; i++) {
      let acc = yData[baseN + i] - bData[bn + i];
      // sum_{j<i} L[n,i,j] · z[n,j]: L is atom-strided (Ln), the
      // already-solved z entries are atom-strided too (baseN, same atom).
      for (let j = 0; j < i; j++) acc -= Ld[Ln + i * D + j] * out[baseN + j];
      const Lii = Ld[Ln + i * D + i];
      if (!(Lii !== 0)) {
        throw new Error('bijection.affine: dense L has zero diagonal at index '
          + i + (LAtomBatched ? ' atom ' + n : '') + ' — inverse undefined');
      }
      out[baseN + i] = acc / Lii;
    }
  }
  return { shape: [N, D], data: out };
}

/**
 * Density-side affine log|det J_{f⁻¹}(y)| = -log|det(L)| =
 * -sum(log|diag(L)|). Independent of `y` — the Jacobian of an affine
 * map is the constant L.
 *
 * Return type (per the BijectionEntry contract, `number | Float64Array`):
 *   - atom-INDEPENDENT L ([D,D] dense or diag-stored) → scalar `number`,
 *     the same value for every atom.
 *   - atom-BATCHED L ([N,D,D] dense, 5f-2 hierarchical case) →
 *     `Float64Array(N)`, one log-det per atom.
 *
 * `y` is accepted for signature symmetry with non-affine bijections;
 * it's unused here.
 */
function affineLogDetJ(_y: any, params: any, N: number): number | Float64Array {
  const valueLib = require('./value.ts');
  const L = params.L;
  if (!L) {
    throw new Error('bijection.affine: logDetJ requires params.L');
  }
  // Atom-batched dense L [N, D, D] → per-atom Float64Array. (Diag-stored
  // L has no atom axis, so it can only be the scalar atom-indep case.)
  const diagStored = valueLib.isDiagStored && valueLib.isDiagStored(L);
  if (!diagStored && L.shape.length === 3
      && L.shape[0] === N && L.shape[1] === L.shape[2]) {
    const D = L.shape[1];
    const outJ = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      let s = 0;
      const Ln = n * D * D;
      for (let i = 0; i < D; i++) {
        const v = L.data[Ln + i * D + i];
        if (!(v !== 0)) {
          throw new Error('bijection.affine: dense L has zero diagonal at index '
            + i + ' atom ' + n + ' — logDetJ undefined');
        }
        s += Math.log(Math.abs(v));
      }
      outJ[n] = -s;
    }
    return outJ;
  }
  // Atom-independent L → scalar.
  let s = 0;
  if (diagStored) {
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
      throw new Error('bijection.affine: dense L must be [D,D] or [N,D,D]; got '
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

// =====================================================================
// Elementary scalar bijections + symbolic path inversion (spec §06)
// =====================================================================
//
// The general analogue of Julia's InverseFunctions.jl + Changes-
// OfVariables.jl, but operating on FlatPIR value expressions rather
// than host functions — the engine-concepts term-rewriting philosophy.
// Two parts:
//
//   1. ELEMENTARY_BIJECTIONS — a per-op rule table for the spec §06
//      case-1 "known-bijection registry" set. Each entry gives the
//      INVERSE rewrite and the FORWARD log-abs-det-Jacobian (LADJ)
//      rewrite of one op, as a function of WHICH ARGUMENT is the free
//      (inversion) variable. A binary op (`add`/`sub`/`mul`/`divide`/
//      `pow`) is a bijection only once the OTHER argument is frozen
//      (closed over) — exactly Julia's `Base.Fix1`/`Fix2`. "Which arg
//      is free" is not declared; it is read structurally from the IR
//      by `invertExpr` (the free variable flows through exactly one
//      argument; the rest are frozen).
//
//   2. invertExpr — the path-inversion pass over a STRAIGHT-LINE value
//      expression of ONE free input. It peels the expression
//      output→input, reverse-composing per-op inverses, and sums the
//      per-op forward LADJs. Output is ORDINARY value IR, so both the
//      inverse and the LADJ evaluate through the normal batched
//      evaluator (`evaluateExprN`) — they ride the batch-flatten
//      machinery for free (the §20.10 vertical collapse applies to a
//      whole grid of inversion points in one pass).
//
// v1 is symbolic-registry-only: no numerical root-find fallback (a
// non-registry monotone op refuses), and no general multivariate
// nonlinear inversion (the `affine` entry above already covers the
// linear multivariate case). `bijection(f, f_inv, logvolume)` (spec
// §04) is the user-supplied complement — it hands the inverse + volume
// in directly, bypassing this synthesis.
//
// CONVENTION. `ladjIR` is the FORWARD LADJ, log|d(out)/d(free)|. The
// pushforward density relation is
//     log p_out(y) = log p_free(f⁻¹(y)) − ladjIR(f⁻¹(y))
// i.e. the registry's inverse-volume `logDetJ(y) = −ladjIR(f⁻¹(y))`.
// A caller ingesting `bijection`'s FORWARD `logvolume` negates to match
// `logDetJ`; a caller using `ladjIR` directly subtracts it.

function _isLit(n: any): boolean {
  return !!(n && n.kind === 'lit' && typeof n.value === 'number');
}
function _lit(v: number): any {
  return { kind: 'lit', value: v, numType: Number.isInteger(v) ? 'integer' : 'real' };
}
function _call(op: string, args: any[]): any {
  return { kind: 'call', op, args };
}
/** log|e| — the per-op LADJ building block for non-constant Jacobians. */
function _logAbs(e: any): any {
  return _call('log', [_call('abs', [e])]);
}

/**
 * Does `ir` contain a node matching `matches` anywhere in its sub-IR?
 * Walks every IR sub-position (args / kwargs / fields / body / branches
 * / selector) so the free-variable search can't miss an occurrence.
 */
function _irContainsRef(ir: any, matches: (n: any) => boolean): boolean {
  if (!ir || typeof ir !== 'object') return false;
  if (matches(ir)) return true;
  if (Array.isArray(ir.args)) {
    for (const a of ir.args) if (_irContainsRef(a, matches)) return true;
  }
  if (ir.kwargs) {
    for (const k in ir.kwargs) {
      if (Object.prototype.hasOwnProperty.call(ir.kwargs, k)
          && _irContainsRef(ir.kwargs[k], matches)) return true;
    }
  }
  if (Array.isArray(ir.fields)) {
    for (const f of ir.fields) if (f && _irContainsRef(f.value, matches)) return true;
  }
  if (ir.body && _irContainsRef(ir.body, matches)) return true;
  if (Array.isArray(ir.branches)) {
    for (const b of ir.branches) if (_irContainsRef(b, matches)) return true;
  }
  if (ir.selector && _irContainsRef(ir.selector, matches)) return true;
  return false;
}

// Per-op inverse + forward-LADJ rules. `idx` is the on-path (free)
// argument index; `args` the node's args; `Y` the running inverse
// expression (what the op's OUTPUT equals); `X` the on-path argument's
// forward subexpression (what flows INTO the op, in terms of the free
// variable). A rule returns `null` when the shape is not invertible
// (e.g. `pow` with a non-literal exponent, or the base-frozen `a^x`
// case deferred from v1).
const ELEMENTARY_BIJECTIONS: Record<string, {
  invert: (idx: number, args: any[], Y: any) => any,
  ladj: (idx: number, args: any[], X: any) => any,
}> = {
  // y = e^x → x = log y ; log|d e^x/dx| = x.
  exp: { invert: (_i, _a, Y) => _call('log', [Y]), ladj: (_i, _a, X) => X },
  // y = log x → x = e^y ; log|d log x/dx| = −log|x|.
  log: { invert: (_i, _a, Y) => _call('exp', [Y]), ladj: (_i, _a, X) => _call('neg', [_logAbs(X)]) },
  // y = −x → x = −y ; LADJ 0.
  neg: { invert: (_i, _a, Y) => _call('neg', [Y]), ladj: () => _lit(0) },
  // y = a + x (commutative) → x = y − a ; LADJ 0.
  add: { invert: (i, a, Y) => _call('sub', [Y, a[1 - i]]), ladj: () => _lit(0) },
  // y = a − b. free=arg0: x = y + b. free=arg1: x = a − y. LADJ 0.
  sub: {
    invert: (i, a, Y) => i === 0 ? _call('add', [Y, a[1]]) : _call('sub', [a[0], Y]),
    ladj: () => _lit(0),
  },
  // y = c · x (commutative) → x = y / c ; log|c|.
  mul: { invert: (i, a, Y) => _call('divide', [Y, a[1 - i]]), ladj: (i, a) => _logAbs(a[1 - i]) },
  // y = a / b. free=arg0 (x/c): x = y·c, LADJ −log|c|. free=arg1 (a/x):
  // x = a/y, LADJ log|a| − 2 log|x|.
  divide: {
    invert: (i, a, Y) => i === 0 ? _call('mul', [Y, a[1]]) : _call('divide', [a[0], Y]),
    ladj: (i, a, X) => i === 0
      ? _call('neg', [_logAbs(a[1])])
      : _call('sub', [_logAbs(a[0]), _call('mul', [_lit(2), _logAbs(X)])]),
  },
  // y = x^k, k a literal exponent (the spec §06 case-1 form; base free).
  //   inverse: x = y^(1/k).   LADJ: log|k·x^(k−1)| = log|k| + (k−1)log|x|.
  // NOTE (v1 limitation): `pow(Y, 1/k)` is NaN for an odd-integer root of
  // a negative output — fine for positive-support maps (transport's
  // base x+δ > 0); a sign-preserving root / numerical fallback is the
  // refinement. Base-frozen `a^x` (idx 1) is deferred → null.
  pow: {
    // Only a NON-ZERO finite literal exponent gives an invertible map:
    // pow(_, 0) is the constant 1 (not invertible — declining here makes
    // walkPushfwd refuse loudly instead of synthesising a `y^(1/0)` = NaN
    // inverse), #310.
    invert: (i, a, Y) => (i === 0 && _isLit(a[1]) && a[1].value !== 0 && Number.isFinite(a[1].value))
      ? _call('pow', [Y, _lit(1 / a[1].value)]) : null,
    ladj: (i, a, X) => (i === 0 && _isLit(a[1]) && a[1].value !== 0 && Number.isFinite(a[1].value))
      ? _call('add', [_lit(Math.log(Math.abs(a[1].value))), _call('mul', [_lit(a[1].value - 1), _logAbs(X)])])
      : null,
  },

  // ---------------------------------------------------------------------
  // §06 known-bijection registry extension (#260 (b)): the remaining
  // monotone §07 elementary unary functions. Each entry mirrors the
  // oracle-verified Rust determiniser (flatppl-rust crates/determinizer
  // src/invert.rs `bare_bijection`) — same inverse builtin, same forward
  // log-abs-det-Jacobian (LADJ) formula — independently re-verified here
  // against scipy to < 1e-9 (packages/engine/test/pushfwd-registry-density.test.ts).
  // Domain restriction (log/log10/sqrt need positive-ish support; log1p
  // needs (-1,inf); logit/probit need (0,1)) is NOT encoded here — these
  // per-op rules are domain-AGNOSTIC path-inversion, exactly like the
  // already-shipped `log`/`pow` entries above. The domain guard lives
  // separately in typeinfer.ts (`checkPushfwdDomainContracts`, #260 (c)),
  // which refuses the whole `pushfwd` when the base measure's support is
  // PROVABLY outside the op's required domain — refuse-don't-mislower,
  // never a silently sub-probability density.

  // y = sqrt(x) = x^(1/2) ⇒ x = y²; log|f'| = log(1/2) − (1/2)·log(x).
  // `sqrt` is its own IR op (not folded to `pow`), so it needs its own
  // entry — mirrors invert.rs `bare_bijection("sqrt")`, which delegates to
  // `derive_pow` with k=0.5 (same inverse + LADJ shape, k=0.5 literal).
  // Domain: nonnegreals (spec §07; the shared "positive-ish" guard also
  // used by log/log10/pow — see typeinfer.ts `_isWithinPositiveDomain`).
  sqrt: {
    invert: (_i, _a, Y) => _call('pow', [Y, _lit(2)]),
    ladj: (_i, _a, X) => _call('add', [
      _lit(Math.log(0.5)),
      _call('mul', [_lit(-0.5), _call('log', [X])]),
    ]),
  },
  // y = log10(x) = ln(x)/ln(10) ⇒ x = 10^y; log|f'| = −log(x) − log(ln 10).
  // Domain: posreals (spec §07; same guard as `log`).
  log10: {
    invert: (_i, _a, Y) => _call('pow', [_lit(10), Y]),
    ladj: (_i, _a, X) => _call('neg', [
      _call('add', [_call('log', [X]), _lit(Math.log(Math.log(10)))]),
    ]),
  },
  // y = log1p(x) = ln(1+x) ⇒ x = expm1(y); log|f'| = −log1p(x).
  // Domain: interval(−1, ∞) (spec §07).
  log1p: {
    invert: (_i, _a, Y) => _call('expm1', [Y]),
    ladj: (_i, _a, X) => _call('neg', [_call('log1p', [X])]),
  },
  // y = expm1(x) = eˣ − 1 ⇒ x = log1p(y); log|f'| = x (identity, d/dx eˣ = eˣ).
  // Domain: ℝ (unrestricted).
  expm1: {
    invert: (_i, _a, Y) => _call('log1p', [Y]),
    ladj: (_i, _a, X) => X,
  },
  // y = logit(p) = ln(p/(1−p)) ⇒ p = invlogit(y); log|f'| = −log(p) − log(1−p).
  // Domain: interval(0, 1) (spec §07).
  logit: {
    invert: (_i, _a, Y) => _call('invlogit', [Y]),
    ladj: (_i, _a, X) => _call('neg', [
      _call('add', [_call('log', [X]), _call('log', [_call('sub', [_lit(1), X])])]),
    ]),
  },
  // y = invlogit(x) = 1/(1+e⁻ˣ) ⇒ x = logit(y); log|f'| = log σ(x) + log(1−σ(x)).
  // Domain: ℝ (unrestricted).
  invlogit: {
    invert: (_i, _a, Y) => _call('logit', [Y]),
    ladj: (_i, _a, X) => {
      const s = _call('invlogit', [X]);
      return _call('add', [_call('log', [s]), _call('log', [_call('sub', [_lit(1), s])])]);
    },
  },
  // y = probit(p) = Φ⁻¹(p) ⇒ p = invprobit(y); log|f'| = ½ln(2π) + ½·probit(p)².
  // Domain: interval(0, 1) (spec §07).
  probit: {
    invert: (_i, _a, Y) => _call('invprobit', [Y]),
    ladj: (_i, _a, X) => {
      const pr = _call('probit', [X]);
      const sq = _call('pow', [pr, _lit(2)]);
      return _call('add', [_lit(0.5 * Math.log(2 * Math.PI)), _call('mul', [_lit(0.5), sq])]);
    },
  },
  // y = invprobit(x) = Φ(x) ⇒ x = probit(y); log|f'| = ln φ(x) = −½ln(2π) − ½x².
  // Domain: ℝ (unrestricted).
  invprobit: {
    invert: (_i, _a, Y) => _call('probit', [Y]),
    ladj: (_i, _a, X) => {
      const sq = _call('pow', [X, _lit(2)]);
      const inner = _call('add', [_lit(0.5 * Math.log(2 * Math.PI)), _call('mul', [_lit(0.5), sq])]);
      return _call('neg', [inner]);
    },
  },
  // y = atan(x) ⇒ x = tan(y) (valid on atan's range (−π/2, π/2), where tan
  // is the single-valued inverse); log|f'| = −log(1 + x²). Domain: ℝ.
  atan: {
    invert: (_i, _a, Y) => _call('tan', [Y]),
    ladj: (_i, _a, X) => {
      const sq = _call('pow', [X, _lit(2)]);
      return _call('neg', [_call('log', [_call('add', [_lit(1), sq])])]);
    },
  },
  // y = sinh(x) ⇒ x = asinh(y); log|f'| = log(cosh(x)). Domain: ℝ.
  sinh: {
    invert: (_i, _a, Y) => _call('asinh', [Y]),
    ladj: (_i, _a, X) => _call('log', [_call('cosh', [X])]),
  },
  // y = asinh(x) ⇒ x = sinh(y); log|f'| = −½·log(1 + x²). Domain: ℝ.
  asinh: {
    invert: (_i, _a, Y) => _call('sinh', [Y]),
    ladj: (_i, _a, X) => {
      const sq = _call('pow', [X, _lit(2)]);
      return _call('mul', [_lit(-0.5), _call('log', [_call('add', [_lit(1), sq])])]);
    },
  },
  // y = tanh(x) ⇒ x = atanh(y); log|f'| = log(1 − tanh(x)²). Domain: ℝ.
  // `atanh` exists as a builtin (ops-declarations.ts) — verified before
  // adding this entry.
  tanh: {
    invert: (_i, _a, Y) => _call('atanh', [Y]),
    ladj: (_i, _a, X) => {
      const th = _call('tanh', [X]);
      const sq = _call('pow', [th, _lit(2)]);
      return _call('log', [_call('sub', [_lit(1), sq])]);
    },
  },
};

/**
 * Symbolically invert a straight-line FlatPIR value expression of one
 * free input (spec §06 case-1 known-bijection composition).
 *
 * `outputExpr` is the IR for `y = f(free, …frozen)`. `freeRef` selects
 * the free input — either a predicate `(node) => boolean` or a
 * `{ ns?, name }` ref descriptor. `outputValue` is the IR node to stand
 * in for `y` in the returned inverse (the caller binds it to the
 * observed value at eval time).
 *
 * Returns `{ inverseIR, ladjIR }` or `null` when `f` is not a
 * single-occurrence straight-line composition of registry bijections
 * (free input absent, appears in more than one argument, or an op on
 * the path has no rule). `inverseIR` is in terms of `outputValue` +
 * the frozen leaves; `ladjIR` (FORWARD LADJ, see CONVENTION above) is in
 * terms of `freeRef` + the frozen leaves — the caller evaluates it at
 * the inverted free value.
 */
function invertExpr(spec: { outputExpr: any, freeRef: any, outputValue: any }): { inverseIR: any, ladjIR: any } | null {
  const { outputExpr, freeRef, outputValue } = spec;
  const matches: (n: any) => boolean = typeof freeRef === 'function'
    ? freeRef
    : (n: any) => !!(n && n.kind === 'ref' && n.name === freeRef.name
        && (freeRef.ns == null || n.ns === freeRef.ns));
  if (!_irContainsRef(outputExpr, matches)) return null;

  // 1. Trace the straight-line path output→leaf. At each node exactly
  //    one argument may contain the free variable (the on-path arg);
  //    the rest are frozen. More than one → not single-input straight-
  //    line (e.g. `u * u`) → refuse.
  const path: Array<{ node: any, onPathIdx: number }> = [];
  let cur = outputExpr;
  while (!(cur.kind === 'ref' && matches(cur))) {
    if (cur.kind !== 'call' || !Array.isArray(cur.args)) return null;
    let onPathIdx = -1;
    for (let i = 0; i < cur.args.length; i++) {
      if (_irContainsRef(cur.args[i], matches)) {
        if (onPathIdx !== -1) return null;   // free var in >1 arg
        onPathIdx = i;
      }
    }
    if (onPathIdx === -1) return null;
    if (!ELEMENTARY_BIJECTIONS[cur.op]) return null;   // op not invertible
    path.push({ node: cur, onPathIdx });
    cur = cur.args[onPathIdx];
  }
  // Identity: outputExpr IS the free ref.
  if (path.length === 0) return { inverseIR: outputValue, ladjIR: _lit(0) };

  // 2. inverseIR: fold output→input, applying each op's inverse rule.
  let Y = outputValue;
  for (const { node, onPathIdx } of path) {
    const inv = ELEMENTARY_BIJECTIONS[node.op].invert(onPathIdx, node.args, Y);
    if (inv == null) return null;
    Y = inv;
  }

  // 3. ladjIR: sum the per-op forward LADJ, each evaluated at the op's
  //    on-path INPUT subexpression (already in terms of the free var).
  const terms: any[] = [];
  for (const { node, onPathIdx } of path) {
    const lj = ELEMENTARY_BIJECTIONS[node.op].ladj(onPathIdx, node.args, node.args[onPathIdx]);
    if (lj == null) return null;
    terms.push(lj);
  }
  const ladjIR = terms.reduce((acc: any, t: any) => acc == null ? t : _call('add', [acc, t]), null);

  return { inverseIR: Y, ladjIR };
}

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
  // Symbolic scalar inverse + LADJ (spec §06 case-1; the InverseFunctions
  // + ChangesOfVariables analogue over FlatPIR).
  invertExpr,
  ELEMENTARY_BIJECTIONS,
};
