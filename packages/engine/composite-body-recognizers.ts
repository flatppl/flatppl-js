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
    })
  | (CompositeBodyBase & {
      kind: 'nested_broadcast';
      /** Inner broadcast's distOp — either a bare scalar sampler-
       *  REGISTRY dist (Phase 4.4 original scope) OR a
       *  VECTOR_OUTPUT_DISTRIBUTIONS entry like MvNormal (Phase 5.1
       *  Session 5b extension). The executor branches on
       *  `innerIsVectorOutput` to dispatch vector-output inner per
       *  (outer_j, inner_k) cell through the registry-backed
       *  materialiser. */
      innerDistOp: string;
      /** Inner distribution's REGISTRY param names. Empty for vector-
       *  output inner dists. */
      innerDistParams: string[];
      /** Inner broadcast's kwargs IR. Each kwarg's IR may reference outer
       *  kernel placeholders (`%local`) AND/OR closed-over self-refs
       *  (`self`). The executor's per-(outer_j, inner_k) loop substitutes
       *  outer placeholders cell-by-cell and slices inner-collection args
       *  per inner cell. */
      innerKwargs: Record<string, any>;
      /** True when `innerDistOp` is in `VECTOR_OUTPUT_DISTRIBUTIONS`
       *  (Phase 5.1 Session 5b). */
      innerIsVectorOutput: boolean;
      /** Per-inner-cell event dim along the nested stitching axis: 1
       *  for scalar dists, n for MvNormal etc. NaN at classify-time
       *  without literal mu; materialiser resolves at runtime. */
      innerEventDim: number;
    })
  | (CompositeBodyBase & {
      kind: 'jointchain';
      /** Ordered chain steps (length ≥ 2). Step 0 is the base measure
       *  (a closed-first sampleable DistCall, kernel placeholders
       *  embedded). Each subsequent step is a single-input kernel
       *  whose body is `lawof(<sampleable DistCall>)`; the step's
       *  `inputParam` receives the previous step's per-atom column at
       *  execute time. */
      steps: Array<{
        base?: {
          distOp: string;
          distParams: string[];
          distKwargs: Record<string, any>;
        };
        kernel?: {
          inputParam: string;
          distOp: string;
          distParams: string[];
          distKwargs: Record<string, any>;
        };
      }>;
    })
  | (CompositeBodyBase & {
      kind: 'joint';
      /** Component layout: 'positional' produces a concatenated per-cell
       *  vector; 'keyword' carries field names for record-typed variates.
       *  Phase 4.2 produces flat per-cell vectors in both cases, with
       *  keyword field names retained as metadata for downstream
       *  consumers. */
      layout: 'positional' | 'keyword';
      /** Ordered components — each either a sampleable scalar
       *  distribution call (Phase 4.2 original scope) OR a
       *  VECTOR_OUTPUT distribution call like MvNormal (Phase 5.1
       *  Session 5a extension). `eventDim` records the per-cell output
       *  width: 1 for scalar dists, n for MvNormal-style vector
       *  outputs. The joint executor stitches into
       *  `[N, K, sum_c(eventDim_c)]` atom-major. */
      components: Array<{
        /** Surface field name; undefined for positional layout. */
        surfaceName?: string;
        /** Component distribution opcode. */
        distOp: string;
        /** Component params per sampler.REGISTRY (scalar) — empty for
         *  vector-output dists, which the executor materialises through
         *  a kind-specific path (matMvNormal etc.) rather than the
         *  worker's sampleN. */
        distParams: string[];
        /** Component kwargs IR with kernel placeholders still embedded —
         *  the executor substitutes them per cell via
         *  `_substituteKernelParams`. */
        distKwargs: Record<string, any>;
        /** True when `distOp` belongs to `ir-shared.VECTOR_OUTPUT_
         *  DISTRIBUTIONS`. The executor dispatches per-cell through
         *  the materialiser (registry-backed) rather than sampleN. */
        isVectorOutput: boolean;
        /** Per-cell output dim along the joint's stitching axis: 1 for
         *  scalar dists, n for MvNormal etc. NaN at classify-time when
         *  recogniser ran without binding env (recogniser caller
         *  resolves at materialise-time). */
        eventDim: number;
      }>;
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

// ---------- joint: `lawof(joint(<components>))` --------------------
//
// Phase 4.2 entry. Components are either positional (concat-vector
// variate) or keyword (named record). The executor (`_executeJoint-
// Composite` in mat-broadcast.ts) handles both layouts; the recogniser
// preserves the layout flag and per-component surface names so the
// stitching pass can label outputs.
//
// Component scope: built-in sampleable scalar distributions (anything
// in sampler.REGISTRY). Vector-valued components defer to Phase 5.1.

registerCompositeBodyRecognizer((d, ctx) => {
  if (!ctx || !ctx.bindings) return null;
  const desc = kernelBroadcastShape.detectJointKernelBinding(
    d.distOp, ctx.bindings);
  if (!desc) return null;
  return {
    kind: 'joint',
    binding: desc.binding,
    params: desc.params,
    paramKwargs: desc.paramKwargs,
    layout: desc.layout,
    // Components carry `isVectorOutput` + `eventDim` since Phase 5.1
    // Session 5a — the joint executor branches on isVectorOutput to
    // dispatch vector-output components through the registry-backed
    // per-cell materialiser path.
    components: desc.components,
  };
});

// ---------- jointchain: `lawof(jointchain(<base>, <K_1>, …))` ------
//
// Phase 4.3 entry. Closed-first chains where step 0 is a base measure
// (sampleable DistCall via an anon binding) and each subsequent step
// is a single-input kernel whose body is `lawof(<sampleable DistCall>)`.
// The executor (`_executeJointChainComposite` in mat-broadcast.ts)
// threads state step-by-step within each cell: step k's sampleN sees
// the prev-variate parameter bound to a per-atom column from step k-1.

registerCompositeBodyRecognizer((d, ctx) => {
  if (!ctx || !ctx.bindings) return null;
  const desc = kernelBroadcastShape.detectJointChainKernelBinding(
    d.distOp, ctx.bindings);
  if (!desc) return null;
  return {
    kind: 'jointchain',
    binding: desc.binding,
    params: desc.params,
    paramKwargs: desc.paramKwargs,
    steps: desc.steps,
  };
});

// ---------- nested_broadcast: `lawof(broadcast(<bare_dist>, kw))` ---
//
// Phase 4.4 entry. Outer kernel body is itself a broadcast — typical
// pattern for nested observation models (per-group, per-observation).
// MVP: inner broadcast head is a sampler-REGISTRY-known scalar
// distribution; inner kwargs may mix outer placeholders + closed-over
// self-refs. Composite-bodied inner kernels defer.

registerCompositeBodyRecognizer((d, ctx) => {
  if (!ctx || !ctx.bindings) return null;
  const desc = kernelBroadcastShape.detectNestedBroadcastKernelBinding(
    d.distOp, ctx.bindings);
  if (!desc) return null;
  return {
    kind: 'nested_broadcast',
    binding: desc.binding,
    params: desc.params,
    paramKwargs: desc.paramKwargs,
    innerDistOp: desc.innerDistOp,
    innerDistParams: desc.innerDistParams,
    innerKwargs: desc.innerKwargs,
    // Phase 5.1 Session 5b — MvNormal inner support.
    innerIsVectorOutput: desc.innerIsVectorOutput,
    innerEventDim: desc.innerEventDim,
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
