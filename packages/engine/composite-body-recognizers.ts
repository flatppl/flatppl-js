'use strict';

// =====================================================================
// COMPOSITE_BODY_RECOGNIZERS — kernel-broadcast body shape recognition
// =====================================================================
//
// `broadcast(K, args…)` where K is a user-defined kernel binding (not
// a built-in distribution) needs structural recognition of K's body
// shape so the materialiser knows HOW to execute the broadcast. The
// composite shapes the engine targets (one per Phase-4 entry of the
// broadcast staged plan) are:
//
//   - iid:               `lawof(iid(<BuiltinDist>(<kw>), <n>))`
//   - vector_per_cell:   inner DistCall's params are rank-1 per cell
//                        (random-effects pattern); Phase 4.1.
//   - joint:             `lawof(joint(<kernels>))`; Phase 4.2.
//   - jointchain:        `lawof(jointchain(<base>, <kernels…>))`;
//                        Phase 4.3.
//   - nested_broadcast:  body contains a `broadcast(…)` itself;
//                        Phase 4.4.
//
// Each recognizer is a (d, ctx) → CompositeBody | null function. They
// are tried in registration order; the first non-null result wins.
//
// The matching catch-all walker (Phase 4.5) reads the registry's
// supported set when emitting its "this shape isn't supported by any
// fast path, falling to per-atom interpretation" diagnostic.
//
// === Cross-engine architecture note ===========================
// The recognizer SURFACE is cross-engine architecture. JAX/MLIR or
// Reactant.jl backends need an analogous body-shape recogniser table
// to know which composite kernel bodies they can lower to vectorised
// codegen rules. The implementations themselves are engine-specific
// (closed-form expand in JS; per-shape codegen rule emission in
// MLIR); the SURFACE shape is shared.
//
// Phase 1.2 lands the registry surface + the iid entry. Subsequent
// composite shapes get added as Phase 4 entries.

// ---------------------------------------------------------------------
// Tagged-union body descriptor + recognizer interface
// ---------------------------------------------------------------------
//
// Type-only declarations (no `export` keyword — esbuild CJS/ESM
// gotcha; see engine/ARCHITECTURE.md "Bundle build gotchas").

/** Common metadata shared by every CompositeBody variant. */
interface CompositeBodyBase {
  /** The user-binding object whose IR matched the recognizer. */
  binding: any;
  /** Kernel parameter names (the placeholder names — declared via
   *  `params` in the functionof IR). */
  params: string[];
  /** Surface kwarg names for the kernel parameters (`paramKwargs`
   *  from the IR — the names broadcast args bind to). */
  paramKwargs: string[];
}

/**
 * Tagged-union descriptor returned by composite-body recognizers.
 *
 * Phase 1.2 lands `kind: 'iid'` only. Phase 4 entries will add
 * 'vector_per_cell', 'joint', 'jointchain', 'nested_broadcast' — each
 * with their own structured fields. matKernelBroadcast (and any
 * future composite executor) discriminates on `kind` before reading
 * variant-specific fields.
 */
type CompositeBody =
  | (CompositeBodyBase & {
      kind: 'iid';
      /** Inner builtin distribution op name (e.g. 'Normal'). */
      distOp: string;
      /** Inner builtin's parameter names per sampler.REGISTRY. May be
       *  empty when sampler isn't loadable (classify-time path). */
      distParams: string[];
      /** Inner builtin's kwargs IR. Placeholders inside reference the
       *  kernel params; runtime substitutes via _substituteKernelParams. */
      distKwargs: Record<string, any>;
      /** iid repetition count. Integer at runtime; NaN at classify
       *  time when sourced from a non-fixed ref. */
      n: number;
    });

/**
 * A recognizer takes the derivation `d` and materialiser `ctx`,
 * inspects the kernel binding's IR, and returns a CompositeBody on
 * match — or null if this recognizer doesn't handle the shape.
 *
 * Recognizers must be cheap (structural IR walk only; no worker
 * dispatch, no Value materialisation). They run once per kernel-
 * broadcast materialisation, before any sampling work begins.
 */
type CompositeBodyRecognizer = (d: any, ctx: any) => CompositeBody | null;

// ---------------------------------------------------------------------
// Registry + dispatch
// ---------------------------------------------------------------------

/** Ordered list of recognizers — first-match-wins. */
const REGISTRY: CompositeBodyRecognizer[] = [];

/**
 * Register a recognizer. Earlier registrations are tried first, so
 * register more-specific recognizers BEFORE more-permissive ones
 * (e.g. nested_broadcast before joint, since a `joint(broadcast(...),
 * …)` would match both).
 */
function registerCompositeBodyRecognizer(r: CompositeBodyRecognizer): void {
  REGISTRY.push(r);
}

/**
 * Dispatch entry. Walks registered recognizers in registration order;
 * returns the first non-null result, or null if none matched.
 *
 * matKernelBroadcast falls through to its "unknown distribution
 * kernel" rejection (and, in Phase 4.5, to the per-atom catch-all
 * walker) when this returns null.
 */
function tryRecognizeCompositeBody(d: any, ctx: any): CompositeBody | null {
  for (const r of REGISTRY) {
    const result = r(d, ctx);
    if (result) return result;
  }
  return null;
}

/** Test-only: list the registered recognizer labels (currently just
 *  count — recognizers are bare functions today; future entries may
 *  carry metadata). */
function _testRegisteredCompositeBodyRecognizerCount(): number {
  return REGISTRY.length;
}

// ---------------------------------------------------------------------
// Built-in recognizers
// ---------------------------------------------------------------------
//
// The iid recognizer wraps the structural shape detector that lives
// in kernel-broadcast-shape.ts (also consumed by classify-time via
// `isIidCompositeKernelBinding`). Wrapping rather than duplicating
// preserves the single source of structural truth.

const kernelBroadcastShape = require('./kernel-broadcast-shape.ts');

// ---------- iid: `lawof(iid(<BuiltinDist>(<kw>), <n>))` -------------
//
// Today's only composite-body shape. Today's matKernelBroadcast
// per-cell loop unrolls these with `repeat = n` so each broadcast
// cell produces an iid block of size n. Result shape per atom:
// [G, n] (one inner block per outer broadcast cell).

registerCompositeBodyRecognizer((d, ctx) => {
  if (!ctx || !ctx.bindings) return null;
  const desc = kernelBroadcastShape.detectIidKernelBinding(
    d.distOp, ctx.bindings, ctx.fixedValues);
  if (!desc) return null;
  return {
    kind: 'iid',
    binding: desc.binding,
    params: desc.params,
    paramKwargs: desc.paramKwargs,
    distOp: desc.distOp,
    distParams: desc.distParams,
    distKwargs: desc.distKwargs,
    n: desc.n,
  };
});

// =====================================================================
// CJS facade
// =====================================================================

module.exports = {
  registerCompositeBodyRecognizer,
  tryRecognizeCompositeBody,
  _testRegisteredCompositeBodyRecognizerCount,
};
