'use strict';

// =====================================================================
// ops.ts — unified op declaration registry + dispatcher (Phase 1)
// =====================================================================
//
// One declaration per non-scalar op drives multiple engine surfaces:
//   - the static type signature (consulted by `types.ts`)
//   - the runtime impl (consulted by `sampler.evaluateCall`)
//   - the atom-batched dispatch (consulted by `evaluateExprN` /
//     materialiser per-atom paths)
//   - the builtins / EVALUABLE_OPS catalogue checks
//
// The §17.5 sampler split landed the *structural* decomposition; this
// declaration model is the *semantic* unification — one shape model
// shared between static analysis and runtime dispatch.
//
// Atom-batching is the engine's job, not the op's. Each declaration's
// `logical` takes atom-indep inputs matching its signature; the
// dispatcher recognises atom-batched inputs (shape=[N, …logical]) and
// either calls a `batched` fast-path (when provided) or runs `logical`
// per atom and stitches the per-atom results back into `[N, …]`.
//
// Scope (Phase 1):
//   - Non-scalar value-domain ops with fixed input arities + ranks
//     (cross, self_outer, inv, linsolve, lower_cholesky, ...).
//   - Scalar arith ops stay on ARITH_OPS_N / _SCALAR_PRIM_ARITY.
//   - Higher-order ops (broadcast / aggregate / reduce / scan / filter)
//     and measure-algebra ops are NOT covered here — phases 4-5 lift
//     the model to cover them.
//
// This module is ADDITIVE in Phase 1: it ships the registry +
// dispatcher + conformance harness, but does NOT replace any
// existing ARITH_OPS entries. The transition plan is to declare ops
// here in parallel, conformance-test them against ARITH_OPS, then
// route `evaluateCall` through `dispatch()` once every consumer
// agrees.

const valueLib = require('./value.ts');

// ---------------------------------------------------------------------
// Op declaration shape
// ---------------------------------------------------------------------
//
// A declaration is a plain object:
//
//   {
//     name:      string,                 // op name (matches IR `op` field)
//     signature: TypeSignature,          // {args, kwargs?, result}; logical
//                                        // shape, NO atom-batch axis
//     argRanks:  number[],               // logical rank of each positional
//                                        // arg (drives batch-axis detection)
//     logical:   (...args) => result,    // atom-indep impl; takes inputs
//                                        // matching the signature ranks
//     batched?:  (args, N) => result,    // optional fast-path for atom-
//                                        // batched inputs (shape=[N, …])
//   }
//
// The signature is the same shape `types.SIGNATURE_FACTORIES` returns
// today; the registry just exposes it by name.
//
// `argRanks` is a Phase-1 convenience: derived from the signature in
// principle, but the type AST's shape-variable handling is more
// involved than we need now, so we list ranks explicitly per op until
// the signature-driven derivation matures.

// Op kind discriminator (engine-concepts §18.7):
//
//   - 'fixed-rank'        — each arg has a fixed logical rank; the
//                           dispatcher detects atom-batching as
//                           `rank == argRanks[k] + 1`. Default.
//   - 'rank-polymorphic'  — arg ranks vary per-call (e.g. transpose
//                           accepts vector or matrix; linsolve(A, b)
//                           takes b as either). The dispatcher does
//                           NOT auto-atom-batch — explicit
//                           `broadcast` wrapping is the contract for
//                           batching such ops. `argRanks` is unused.
//   - 'variadic'          — arg count varies per-call (e.g. cat,
//                           vector). The dispatcher calls `logical`
//                           with all args spread; no atom-batch
//                           detection. `argRanks` is unused.
//   - 'higher-order'      — at least one arg is a callable; the op
//                           threads body-inference / function-call
//                           machinery through dispatch. Deferred —
//                           see §18.7. Engines must surface a clear
//                           error if the dispatcher encounters this
//                           kind before the dedicated dispatch
//                           extension lands.
type OpKind = 'fixed-rank' | 'rank-polymorphic' | 'variadic' | 'higher-order';

interface OpDecl {
  name: string;
  signature?: any;
  // Required for kind='fixed-rank'; ignored for 'rank-polymorphic',
  // 'variadic', 'higher-order'.
  argRanks?: number[];
  kind?: OpKind;                      // default 'fixed-rank'
  logical?: (...args: any[]) => any;  // optional: variant-only ops omit it
  batched?: (args: any[], N: number) => any;
  // Type-directed shape-pattern variants — the P1 keystone per
  // engine-concepts §18.11. Each variant declares the input shape
  // pattern it applies to (per-arg rank / shape / Klein-4 tag /
  // struct flag / dtype) plus the surrounding wrapping op
  // ('broadcast' / 'aggregate' / 'direct'). The dispatcher picks the
  // most-specific applicable variant; ties broken by registration
  // order. Variants are tried BEFORE the kind-based legacy paths,
  // so they take priority. Variant-only ops (no `logical`) auto-
  // create a minimal OpDecl via `registerVariant`.
  variants?: OpVariant[];
}

// ---------------------------------------------------------------------
// Type-directed shape-pattern variants (engine-concepts §18.11)
// ---------------------------------------------------------------------
//
// A variant declares: "for input args matching THESE patterns, under
// THIS wrapping context, dispatch to THIS impl." Variants generalise
// the single-`logical` model to cover the dispatch axes the engine
// currently fragments across multiple modules:
//
//   - Per-arg rank, shape, Klein-4 tag, structured-matrix flag, dtype.
//   - Wrapping op ('broadcast' / 'aggregate' / 'direct') — same spec
//     op has different semantics under different wrappers (mul direct
//     = matrix product; mul broadcasted = Hadamard elementwise).
//
// Adding a new shape variant becomes one registry entry, not edits
// across `value-ops.mul`'s shape-switch, `_BROADCASTED_PRIMS_CACHE`,
// `ARITH_OPS_N`, `_maybeFastBroadcasted`, and `_broadcastApply`.
//
// **Specificity ordering.** When multiple variants match, the most
// specific wins (most constrained pattern fields). The score is:
//
//   - wrappingOp constraint: 50 pts (most categorical)
//   - struct flag:           10 pts/arg
//   - tag constraint:         2 pts/arg
//   - rank constraint:        1 pt/arg
//   - explicit shape dim:     1 pt/dim
//   - dtype constraint:       1 pt/arg
//
// Ties broken by registration order: first registered wins. This is
// deterministic but inverts the usual Julia-multimethod convention; we
// pick it because most variants are registered ONCE per category and
// later-registered variants conceptually OVERRIDE earlier ones
// (consistent with module-load order).
//
// **Extensibility.** Adding new pattern dimensions (e.g. P2's
// sample/batch/event shape; P3a's axisStack) means adding optional
// fields to ArgPattern + scoring weights to _specificityScore. No
// caller migration required.

type Klein4Tag = 'N' | 'T' | 'A' | 'C';
type ArgKind = 'value' | 'measure' | 'function' | 'kernel' | 'tuple';
type ShapeDim = number | '%dynamic' | '*';
type ShapePattern = Array<ShapeDim>;

// ArgInfo — typed dispatch input. The matcher inspects this struct
// rather than raw JS values, so a single registry can dispatch on
// value-domain args (Value with shape/tag/struct/dtype) AND
// measure-domain args (measure type with sample/batch/event shape
// from P2) uniformly.
//
// Construction:
//   - `argInfoFromValue(v)` wraps a Value (or raw scalar) for
//     value-domain dispatch.
//   - `argInfoFromMeasure(type, opts?)` wraps a measure type for
//     measure-domain dispatch; the engine's typeinfer layer produces
//     these from `inferredType` on measure-typed bindings.
//   - `argInfoFromKernel(type)` / `argInfoFromFunction(type)` wrap
//     callable types.
//
// `dispatch(name, rawArgs, opts)` (the existing entry point) auto-
// wraps each raw arg via argInfoFromValue and routes to the new
// typed entry `dispatchTyped(name, argInfos, opts)`. Existing
// value-only variants and callers stay unchanged.
//
// The `axisStack` slot carries P3a's static outer-axis context from
// the call site so variants can pattern-match on enclosing iteration
// contexts (kernel-broadcast outer / iid sample / aggregate output
// axes). Today populated only by callers that know the context;
// future fusion (b) wires it through the materialiser.
interface ArgInfo {
  kind: ArgKind;
  // Value-side (kind === 'value'):
  value?: any;                            // the Value or raw scalar
  // Measure-side (kind === 'measure'):
  measureType?: any;                      // typeinfer's measure type
  // Callable-side (kind ∈ {'function', 'kernel'}):
  callableType?: any;                     // typeinfer's function/kernel type
  // IR-context (any kind):
  axisStack?: any[];                      // P3a outer-axis context
}

interface ArgPattern {
  // ---------------------------------------------------------------
  // Kind discriminator (default 'value').
  // ---------------------------------------------------------------
  kind?: ArgKind;

  // ---------------------------------------------------------------
  // Value-side constraints (apply when arg is a Value).
  // ---------------------------------------------------------------
  // Logical rank (no atom-batch axis). Atom-batching is detected by
  // the dispatcher BELOW this layer; patterns describe logical shape.
  rank?: number;
  // Concrete dim sizes ('*' = any size, useful for partial constraints
  // like "rank-2 with second dim = 3"). Implies `rank` (length of the
  // shape array).
  shape?: ShapePattern;
  // Klein-4 transpose tag (value.ts §2.1). Single tag matches that
  // tag literally; an array matches if the arg's tag is in the array
  // (e.g. `tag: ['T', 'A']` matches "swapped" Klein-4 elements —
  // those for which `isTransposeView(v) === true`; `tag: ['N', 'C']`
  // matches "unswapped" elements).
  tag?: Klein4Tag | Klein4Tag[];
  // Structured-matrix occupancy/refinement (value.ts §2.2).
  struct?: 'diag' | 'tri' | 'sym';
  // Element type at the storage level.
  dtype?: 'real' | 'complex';

  // ---------------------------------------------------------------
  // Measure-side constraints (apply when arg.kind === 'measure', P2).
  // ---------------------------------------------------------------
  // Three-shape decomposition (Pyro/TFP convention; engine-concepts
  // §20.10.5 item 2). Each shape uses ShapePattern semantics: '*'
  // matches any dim; '%dynamic' matches '%dynamic' literally; a
  // number matches the exact value. The pattern's length must equal
  // the arg's shape length (no rank polymorphism here; if you want
  // "any sampleShape" use absence of the field).
  sampleShape?: ShapePattern;
  batchShape?:  ShapePattern;
  eventShape?:  ShapePattern;

  // ---------------------------------------------------------------
  // IR-context constraints (apply to any kind, from P3a axisStack).
  // ---------------------------------------------------------------
  // Match outer-axis context. Each entry constrains the axis-stack
  // entry at that position (in OUTER-TO-INNER order). `source: '*'`
  // accepts any source. Pattern length must equal the arg's
  // axisStack length (no partial-prefix match — explicit '*' entries
  // express that).
  axisStack?: Array<{ source?: 'iid' | 'broadcast' | 'kernel_broadcast' | 'aggregate' | '*' }>;
}

type WrappingOp = 'broadcast' | 'aggregate' | 'direct';

interface OpVariant {
  // One pattern per positional arg; length must equal the call's arg
  // count for the variant to match.
  argPatterns: ArgPattern[];
  // If set, the variant matches ONLY when the dispatcher is called
  // with opts.wrappingOp equal to this value. If omitted, the variant
  // matches any wrapping context (general fallback).
  wrappingOp?: WrappingOp;
  // The atom-indep impl. Receives the Values as passed to dispatch
  // (NOT pre-stripped of atom axis). Used when every arg's rank
  // matches its argPattern.rank exactly.
  impl: (args: any[], ctx?: DispatchCtx) => any;
  // Atom-batched fast-path. When one or more args have shape=[N,
  // ...rank-of-pattern], the dispatcher routes here with the shared
  // atom count N (engine-concepts §2.1 leading-axis-batch convention).
  // If absent, the dispatcher falls back to per-atom slicing via
  // _argAtAtom + _stackPerAtom (correctness-equivalent but slower).
  // Receives the Values UNCHANGED (atom axis still present); the
  // impl decides how to use N.
  batched?: (args: any[], N: number, ctx?: DispatchCtx) => any;
  // Optional short label for debug / conformance reports.
  label?: string;
}

interface DispatchOpts {
  // Surrounding wrapping op. Default is 'direct' (op called directly,
  // not inside broadcast / aggregate).
  wrappingOp?: WrappingOp;
  // Atom-batched dispatch (engine-concepts §2.1 leading-axis-batch).
  // When set, the dispatcher tries atom-aware variant matching FIRST
  // (variants with a `batched` impl whose argPatterns match each arg
  // with the leading atom dim of size `atomN` stripped); callers
  // signal atom-batching explicitly since rank alone can't always
  // distinguish "rank-r matrix" from "atom-batched rank-(r-1)
  // vector" (e.g. shape=[N, n] is both).
  atomN?: number;
}

interface DispatchCtx extends DispatchOpts {
  // Future: env, evaluateExpr, axisStack annotations, etc. Variants
  // that need richer context (e.g. higher-order body inlining) read
  // additional fields populated by the caller.
  [key: string]: any;
}

function _argTag(arg: any): string {
  if (valueLib.isValue(arg)) return arg.t || 'N';
  return 'N';
}

function _argStruct(arg: any): string | null {
  if (!valueLib.isValue(arg)) return null;
  // value.ts exposes `isDiagStored` / occupancy flags. For now only
  // 'diag' is exposed; tri / sym refinement detection lives in
  // value.ts and can be lifted here when needed.
  if (valueLib.isDiagStored && valueLib.isDiagStored(arg)) return 'diag';
  return null;
}

function _argDtype(arg: any): 'real' | 'complex' {
  if (valueLib.isValue(arg) && arg.dtype === 'complex') return 'complex';
  return 'real';
}

// ---------------------------------------------------------------------
// ArgInfo constructors
// ---------------------------------------------------------------------

// Wrap a raw Value (or scalar) for typed dispatch. The matcher
// inspects argInfo.value for value-side constraints (rank/shape/etc.)
// and argInfo.axisStack for IR-context constraints.
function argInfoFromValue(value: any, opts?: { axisStack?: any[] }): ArgInfo {
  return { kind: 'value', value, axisStack: opts && opts.axisStack };
}

// Wrap a measure type (from typeinfer's inferredType on a measure-
// typed binding) for typed dispatch. The matcher inspects
// argInfo.measureType for measure-side constraints (sampleShape /
// batchShape / eventShape from P2) and argInfo.axisStack for the
// P3a outer-axis context.
function argInfoFromMeasure(measureType: any, opts?: { value?: any; axisStack?: any[] }): ArgInfo {
  return {
    kind: 'measure',
    measureType,
    value: opts && opts.value,
    axisStack: opts && opts.axisStack,
  };
}

// Wrap a function/kernel type for typed dispatch — used by
// higher-order op variants (broadcast, aggregate) that need to
// pattern-match on the callable's input/output shape.
function argInfoFromFunction(callableType: any): ArgInfo {
  return { kind: 'function', callableType };
}
function argInfoFromKernel(callableType: any): ArgInfo {
  return { kind: 'kernel', callableType };
}

// Promote a raw JS arg to ArgInfo (auto-detect kind). Used by the
// legacy `dispatch(name, args, opts)` entry to convert untyped args
// into the typed representation.
function _autoArgInfo(arg: any): ArgInfo {
  // ArgInfo objects pass through (already typed).
  if (arg && typeof arg === 'object' && typeof arg.kind === 'string'
      && (arg.kind === 'value' || arg.kind === 'measure'
          || arg.kind === 'function' || arg.kind === 'kernel'
          || arg.kind === 'tuple')) {
    return arg as ArgInfo;
  }
  return argInfoFromValue(arg);
}

// ---------------------------------------------------------------------
// Pattern matching
// ---------------------------------------------------------------------

// Match a ShapePattern against an actual shape array. '*' matches any
// dim; '%dynamic' matches '%dynamic' literally; a number matches the
// exact value. Returns false if lengths differ.
function _matchShape(p: ShapePattern, actual: any[] | undefined): boolean {
  if (!actual) return false;
  if (p.length !== actual.length) return false;
  for (let i = 0; i < p.length; i++) {
    const want = p[i];
    if (want === '*') continue;
    if (want !== actual[i]) return false;
  }
  return true;
}

// Match an axisStack pattern against an actual axis-stack array.
// Each pattern entry constrains the corresponding actual entry's
// `source`. Pattern length must equal actual length.
function _matchAxisStack(
  p: NonNullable<ArgPattern['axisStack']>,
  actual: any[] | undefined,
): boolean {
  if (!actual) return false;
  if (p.length !== actual.length) return false;
  for (let i = 0; i < p.length; i++) {
    const want = p[i].source;
    if (want === undefined || want === '*') continue;
    if (actual[i].source !== want) return false;
  }
  return true;
}

function _matchArgPattern(p: ArgPattern, arg: any): boolean {
  // Promote raw arg to ArgInfo if needed (so existing callers passing
  // raw Values still work).
  const info: ArgInfo = (arg && typeof arg === 'object' && typeof arg.kind === 'string'
                         && (arg.kind === 'value' || arg.kind === 'measure'
                             || arg.kind === 'function' || arg.kind === 'kernel'
                             || arg.kind === 'tuple'))
                       ? arg as ArgInfo
                       : argInfoFromValue(arg);

  // Kind discriminator. Pattern default is 'value'.
  const wantKind: ArgKind = p.kind || 'value';
  if (info.kind !== wantKind) return false;

  // axisStack constraint applies regardless of kind.
  if (p.axisStack !== undefined) {
    if (!_matchAxisStack(p.axisStack, info.axisStack)) return false;
  }

  if (info.kind === 'value') {
    return _matchValueArg(p, info.value);
  }
  if (info.kind === 'measure') {
    return _matchMeasureArg(p, info.measureType);
  }
  if (info.kind === 'function' || info.kind === 'kernel') {
    // No specific callable-side constraints yet (rank/shape/etc.
    // don't apply directly to a callable). Future extensions can
    // add input-arity / result-rank constraints.
    return true;
  }
  return false;
}

function _matchValueArg(p: ArgPattern, value: any): boolean {
  // Bare scalars (JS number / boolean) match rank-0 patterns only.
  if (!valueLib.isValue(value)) {
    if (typeof value === 'number' || typeof value === 'boolean') {
      if (p.rank !== undefined && p.rank !== 0) return false;
      if (p.shape !== undefined && p.shape.length > 0) return false;
      if (p.tag !== undefined && p.tag !== 'N') return false;
      if (p.struct !== undefined) return false;
      if (p.dtype !== undefined && p.dtype !== 'real') return false;
      return true;
    }
    // Non-Value, non-scalar (nested JS array, Float64Array). Allow
    // only when the pattern places NO value-side constraints.
    return p.rank === undefined && p.shape === undefined
        && p.tag === undefined && p.struct === undefined
        && p.dtype === undefined;
  }
  const rank = value.shape.length;
  if (p.rank !== undefined && rank !== p.rank) return false;
  if (p.shape !== undefined) {
    if (!_matchShape(p.shape, value.shape)) return false;
  }
  if (p.tag !== undefined) {
    const argTag = _argTag(value);
    if (Array.isArray(p.tag)) {
      if (p.tag.indexOf(argTag as Klein4Tag) < 0) return false;
    } else if (argTag !== p.tag) {
      return false;
    }
  }
  if (p.struct !== undefined && _argStruct(value) !== p.struct) return false;
  if (p.dtype !== undefined && _argDtype(value) !== p.dtype) return false;
  return true;
}

function _matchMeasureArg(p: ArgPattern, measureType: any): boolean {
  // No type info → cannot match measure-specific constraints. But if
  // the pattern doesn't constrain anything beyond `kind: 'measure'`,
  // pass through (the kind check at the caller already succeeded).
  if (!measureType) {
    return p.sampleShape === undefined
        && p.batchShape  === undefined
        && p.eventShape  === undefined
        && p.rank        === undefined
        && p.shape       === undefined
        && p.dtype       === undefined;
  }
  if (p.sampleShape !== undefined && !_matchShape(p.sampleShape, measureType.sampleShape)) {
    return false;
  }
  if (p.batchShape !== undefined && !_matchShape(p.batchShape, measureType.batchShape)) {
    return false;
  }
  if (p.eventShape !== undefined && !_matchShape(p.eventShape, measureType.eventShape)) {
    return false;
  }
  // Allow rank/shape constraints to ALSO apply to the measure's
  // domain — useful for "a measure over a rank-2 variate".
  if (p.rank !== undefined || p.shape !== undefined) {
    const dom = measureType.domain;
    if (!dom || dom.kind !== 'array') return false;
    if (p.rank !== undefined && dom.rank !== p.rank) return false;
    if (p.shape !== undefined && !_matchShape(p.shape, dom.shape)) return false;
  }
  // dtype applies to the variate's element type (real vs complex).
  if (p.dtype !== undefined) {
    let elem: any = measureType.domain;
    while (elem && elem.kind === 'array') elem = elem.elem;
    if (!elem || elem.kind !== 'scalar') return false;
    const isCx = elem.prim === 'complex';
    if (p.dtype === 'complex' && !isCx) return false;
    if (p.dtype === 'real'    &&  isCx) return false;
  }
  return true;
}

function _matchVariant(v: OpVariant, args: any[], opts: DispatchOpts): boolean {
  // wrappingOp constraint: if the variant declares one, it must equal
  // the caller's; if it doesn't declare one, the variant matches any.
  if (v.wrappingOp !== undefined) {
    const callerWrap: WrappingOp = opts.wrappingOp || 'direct';
    if (v.wrappingOp !== callerWrap) return false;
  }
  if (v.argPatterns.length !== args.length) return false;
  for (let i = 0; i < args.length; i++) {
    if (!_matchArgPattern(v.argPatterns[i], args[i])) return false;
  }
  return true;
}

function _specificityScore(v: OpVariant): number {
  let s = 0;
  if (v.wrappingOp !== undefined) s += 50;
  for (const p of v.argPatterns) {
    // Kind discriminator: explicitly setting `kind` (any non-default
    // value) is a categorical filter, weighted between wrappingOp
    // and the per-arg refinements.
    if (p.kind !== undefined && p.kind !== 'value') s += 5;
    // Value-side refinements:
    if (p.rank !== undefined) s += 1;
    if (p.shape !== undefined) {
      for (const d of p.shape) if (d !== '*') s += 1;
    }
    if (p.tag !== undefined) s += 2;
    if (p.struct !== undefined) s += 10;
    if (p.dtype !== undefined) s += 1;
    // Measure-side refinements (P2): each non-'*' dim adds 1.
    if (p.sampleShape !== undefined) {
      for (const d of p.sampleShape) if (d !== '*') s += 1;
    }
    if (p.batchShape !== undefined) {
      for (const d of p.batchShape) if (d !== '*') s += 1;
    }
    if (p.eventShape !== undefined) {
      for (const d of p.eventShape) if (d !== '*') s += 1;
    }
    // IR-context refinements (P3a): each non-'*' source adds 2 (a
    // categorical filter like the tag, scaled to its outer-axis
    // significance).
    if (p.axisStack !== undefined) {
      for (const e of p.axisStack) if (e.source !== undefined && e.source !== '*') s += 2;
    }
  }
  return s;
}

// Pick the most-specific matching variant. Returns null when no
// variant matches. Ties broken by registration order (first wins) —
// `variants` is iterated in declaration order, and the SECOND match
// only replaces the first if it scores strictly higher.
function _pickVariant(variants: OpVariant[], args: any[], opts: DispatchOpts): OpVariant | null {
  let best: OpVariant | null = null;
  let bestScore = -1;
  for (const v of variants) {
    if (!_matchVariant(v, args, opts)) continue;
    const s = _specificityScore(v);
    if (s > bestScore) { best = v; bestScore = s; }
  }
  return best;
}

// Atom-aware variant matching (engine-concepts §2.1 leading-axis-batch
// convention). For each variant V whose argPatterns specify static
// ranks and which provides a `batched` fast-path, retry matching with
// each arg's shape "atom-stripped" — i.e. if arg.shape.length ===
// pattern.rank + 1 AND the leading dim equals the caller-declared
// atom count N, treat arg as atom-batched and match the pattern
// against arg.shape[1:].
//
// **Caller signals atom-batching via opts.atomN.** Rank alone can't
// distinguish "rank-2 matrix" from "atom-batched rank-1 vector
// (shape=[N, n])" since both have rank 2 — the caller knows which it
// has and passes opts.atomN so the dispatcher routes correctly.
// (Engines that want auto-detection can supply argRanks at the decl
// level — that path stays available for fixed-rank kind-based
// dispatch.)
//
// Match rules:
//   - opts.atomN must be set; the value is the atom count.
//   - For each arg:
//       - if arg's rank matches pattern.rank exactly → atom-indep.
//       - if arg's rank == pattern.rank + 1 AND arg.shape[0] === atomN
//         → atom-batched (sliced view matched against the pattern).
//       - else → no match.
//   - At least one arg must be atom-batched (otherwise the exact-
//     rank path would have matched first).
//   - The variant MUST have a `batched` impl.
//
// Returns { variant, N } on match (N === opts.atomN) or null when no
// atom-aware variant matches. Same specificity / tie-break as
// _pickVariant.
function _pickVariantAtomAware(
  variants: OpVariant[], args: any[], opts: DispatchOpts,
): { variant: OpVariant; N: number } | null {
  const atomN = opts.atomN;
  if (typeof atomN !== 'number') return null;
  let best: { variant: OpVariant; N: number } | null = null;
  let bestScore = -1;
  for (const v of variants) {
    if (typeof v.batched !== 'function') continue;
    // wrappingOp constraint (same as _matchVariant).
    if (v.wrappingOp !== undefined) {
      const callerWrap: WrappingOp = opts.wrappingOp || 'direct';
      if (v.wrappingOp !== callerWrap) continue;
    }
    if (v.argPatterns.length !== args.length) continue;
    let atomBatchedCount = 0;
    let ok = true;
    for (let i = 0; i < args.length; i++) {
      const p = v.argPatterns[i];
      const arg = args[i];
      // Try exact-rank match first (atom-indep arg).
      if (_matchArgPattern(p, arg)) continue;
      // Otherwise: must have a static pattern rank to slice against.
      if (p.rank === undefined) { ok = false; break; }
      // The arg must be a Value (only Values carry shape metadata).
      if (!valueLib.isValue(arg)) { ok = false; break; }
      if (arg.shape.length !== p.rank + 1) { ok = false; break; }
      if (arg.shape[0] !== atomN) { ok = false; break; }
      // Sliced-view pattern match (shape-only; no storage allocation).
      const slicedView: any = Object.assign({}, arg, { shape: arg.shape.slice(1) });
      if (!_matchArgPattern(p, slicedView)) { ok = false; break; }
      atomBatchedCount++;
    }
    if (!ok || atomBatchedCount === 0) continue;
    const s = _specificityScore(v);
    if (s > bestScore) { best = { variant: v, N: atomN }; bestScore = s; }
  }
  return best;
}

// ---------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------

const REGISTRY: Map<string, OpDecl> = new Map();

function register(decl: OpDecl) {
  if (REGISTRY.has(decl.name)) {
    throw new Error('ops.register: duplicate op declaration for ' + decl.name);
  }
  const kind: OpKind = decl.kind || 'fixed-rank';
  // Fixed-rank ops require argRanks; rank-polymorphic / variadic /
  // higher-order ops don't (argRanks is unused for those kinds).
  // Variant-only ops (no `logical`, only `variants`) also skip argRanks
  // — atom-batch detection happens per-variant.
  if (kind === 'fixed-rank' && decl.logical) {
    if (!Array.isArray(decl.argRanks)) {
      throw new Error('ops.register: op \'' + decl.name +
        '\' is kind=fixed-rank but argRanks is missing');
    }
    if (decl.signature && decl.argRanks.length !== decl.signature.args.length) {
      throw new Error('ops.register: op \'' + decl.name +
        '\' argRanks.length (' + decl.argRanks.length +
        ') must match signature.args.length (' +
        decl.signature.args.length + ')');
    }
  }
  if (!decl.logical && (!decl.variants || decl.variants.length === 0)) {
    throw new Error('ops.register: op \'' + decl.name +
      '\' has no logical impl and no variants');
  }
  REGISTRY.set(decl.name, decl);
}

// Register a shape-pattern variant on an op. If no OpDecl exists yet
// for `opName`, auto-creates a minimal variant-only decl. This is the
// preferred entry point for ops that have NO atom-indep "logical"
// reference impl (e.g. broadcasted scalar primitives whose canonical
// form IS the variant impl). Multiple `registerVariant` calls on the
// same op accumulate variants; the dispatcher matches the most
// specific applicable pattern at call time.
function registerVariant(opName: string, variant: OpVariant): void {
  if (!variant || !Array.isArray(variant.argPatterns) || typeof variant.impl !== 'function') {
    throw new Error('ops.registerVariant: variant must have argPatterns + impl');
  }
  let decl = REGISTRY.get(opName);
  if (!decl) {
    decl = { name: opName, kind: 'fixed-rank', variants: [] };
    REGISTRY.set(opName, decl);
  }
  if (!decl.variants) decl.variants = [];
  decl.variants.push(variant);
}

function lookup(name: string): OpDecl | null {
  return REGISTRY.get(name) || null;
}

// `isDeclared` is the routing gate used by `evaluateCall`: it answers
// "can `dispatch(name, args)` be called for direct (no wrappingOp)
// invocation?" Variant-only ops (no `logical`, only wrappingOp-
// constrained variants like the broadcasted-primitives table) are
// NOT declared in this sense — they're invoked through
// `dispatchVariant` with the right `wrappingOp` opt. Callers that
// need "is the op in the registry at all?" use `lookup(name) !=
// null` instead.
function isDeclared(name: string): boolean {
  const decl = REGISTRY.get(name);
  if (!decl) return false;
  return typeof decl.logical === 'function';
}

// "Does this op have ANY registered variant for the given wrappingOp?"
// Used by `_maybeFastBroadcasted` to gate the broadcast fast-path
// before evaluating arg IRs. Returns false when the op has no
// variants OR has no variant whose `wrappingOp` matches.
function hasVariantFor(name: string, wrappingOp: WrappingOp): boolean {
  const decl = REGISTRY.get(name);
  if (!decl || !decl.variants) return false;
  for (const v of decl.variants) {
    if (v.wrappingOp === wrappingOp) return true;
    // Variants with no wrappingOp constraint match any wrapping; also count.
    if (v.wrappingOp === undefined) return true;
  }
  return false;
}

function listDeclared(): string[] {
  return Array.from(REGISTRY.keys()).filter(n => isDeclared(n));
}

function signatureOf(name: string): any {
  const decl = REGISTRY.get(name);
  return decl ? decl.signature : null;
}

// ---------------------------------------------------------------------
// Runtime dispatch — atom-batched broadcasting
// ---------------------------------------------------------------------
//
// `dispatch(name, args)` is the unified entry point. It:
//   1. Looks up the declaration.
//   2. For each positional arg, determines whether it's atom-indep
//      (shape rank == logical rank) or atom-batched (rank == logical
//      rank + 1, with the leading dim N).
//   3. If every arg is atom-indep, calls `logical(...args)` directly.
//   4. If any arg is atom-batched, all atom-batched args must share
//      the same N. Then:
//        - If `batched` is provided, calls `batched(args, N)`.
//        - Else runs the per-atom fallback: for each i in 0..N,
//          extract atom-i sub-Values from the batched args (keeping
//          atom-indep args unchanged), call `logical`, collect, and
//          pack back into a Value of shape=[N, ...resultShape].
//
// Bare JS arrays / Float64Arrays without a `.shape` field are treated
// as atom-indep at this layer — they don't carry batch info. Callers
// that need atom-batched scalars/arrays must use Values.

// Detect the atom-batch size N for an arg given its declared logical
// rank. Returns:
//   { batched: false }              — atom-indep (matches logical rank)
//   { batched: true,  N: number }   — atom-batched; leading dim is N
//   throws                          — rank doesn't fit either case
//
// engine-concepts §20 / TODO Phase 1: a rank-0 Value passed to a
// scalar op (`logicalRank === 0`) is atom-indep (constant held across
// the N axis); the dispatcher does NOT slice per atom. valueOps
// handles rank-0 × rank-N broadcasting downstream. Higher-rank ops
// reject rank-0 inputs as a type error (e.g. `inv(scalar)` is
// undefined) — that strictness is correct.
function _classifyArg(v: any, logicalRank: number): { batched: boolean; N?: number } {
  if (v == null) return { batched: false };
  // Value: inspect shape vs logical rank.
  if (valueLib.isValue(v)) {
    const rank = v.shape.length;
    if (rank === logicalRank) return { batched: false };
    if (rank === logicalRank + 1) return { batched: true, N: v.shape[0] };
    throw new Error(
      'ops.dispatch: argument rank ' + rank +
      ' incompatible with logical rank ' + logicalRank +
      ' (expected ' + logicalRank + ' or ' + (logicalRank + 1) +
      ' for atom-batched)');
  }
  // Bare numeric — only valid where logicalRank === 0.
  if (typeof v === 'number' || typeof v === 'boolean') {
    if (logicalRank !== 0) {
      throw new Error('ops.dispatch: scalar passed where rank-' +
        logicalRank + ' input expected');
    }
    return { batched: false };
  }
  // Bare typed array / nested JS array — assume atom-indep, ranks not
  // tracked. Phase 1 leaves stricter validation to the op's `logical`.
  return { batched: false };
}

// Extract atom i from a possibly-batched arg. For atom-indep args
// returns the arg unchanged; for atom-batched Values returns a
// sub-Value of shape=`v.shape.slice(1)` backed by a subarray view of
// the underlying Float64Array.
function _argAtAtom(v: any, batched: boolean, i: number): any {
  if (!batched) return v;
  // v must be a Value with shape=[N, …rest]
  const tailDims = v.shape.slice(1);
  const tailLen = tailDims.reduce((a: number, b: number) => a * b, 1);
  // Real part subarray view.
  const subData = v.data.subarray(i * tailLen, (i + 1) * tailLen);
  const sub: any = { shape: tailDims, data: subData };
  // Carry complex imaginary part if present.
  if (v.dtype === 'complex' && v.im) {
    sub.dtype = 'complex';
    sub.im = v.im.subarray(i * tailLen, (i + 1) * tailLen);
  }
  return sub;
}

// Stitch per-atom logical results into a single Value shape=[N, ...].
// Every per-atom result must have the same shape; mismatch is an
// engine bug (the logical impl violated its signature).
function _stackPerAtom(perAtomResults: any[], N: number): any {
  if (N === 0) {
    // Edge case: zero-atom output. Return a Value with shape=[0, …]
    // matching the first dimension of a hypothetical result. Caller
    // shouldn't usually hit this; surface a clear failure if it does.
    throw new Error('ops.dispatch: cannot stack 0 atoms (caller must ensure N > 0)');
  }
  const first = perAtomResults[0];
  let tailShape: number[];
  let isComplex = false;
  if (valueLib.isValue(first)) {
    tailShape = first.shape;
    isComplex = first.dtype === 'complex';
  } else if (first instanceof Float64Array) {
    tailShape = [first.length];
  } else if (Array.isArray(first)) {
    // Nested JS array — flatten to detect shape. We accept this for
    // back-compat with ops that haven't migrated to Values, but the
    // recommended `logical` returns Values for non-scalar outputs.
    let probe: any = first;
    tailShape = [];
    while (Array.isArray(probe) || (probe && probe.BYTES_PER_ELEMENT !== undefined)) {
      tailShape.push(probe.length);
      probe = probe[0];
    }
  } else {
    // Scalar per-atom result (op produces a scalar from per-atom
    // inputs). Output is shape=[N].
    tailShape = [];
  }
  const tailLen = tailShape.reduce((a, b) => a * b, 1);
  const out = new Float64Array(N * tailLen);
  const outIm = isComplex ? new Float64Array(N * tailLen) : null;
  for (let i = 0; i < N; i++) {
    const r = perAtomResults[i];
    if (valueLib.isValue(r)) {
      const base = i * tailLen;
      // Flatten the per-atom Value's data into the output buffer.
      out.set(r.data, base);
      if (outIm && r.dtype === 'complex' && r.im) {
        outIm.set(r.im, base);
      }
    } else if (r instanceof Float64Array) {
      out.set(r, i * tailLen);
    } else if (Array.isArray(r)) {
      // Flatten the nested array atom-by-atom (slower path).
      _writeNested(out, i * tailLen, r);
    } else if (typeof r === 'number' || typeof r === 'boolean') {
      out[i] = +r;
    } else {
      throw new Error('ops.dispatch: unsupported per-atom result type ' + typeof r);
    }
  }
  if (outIm) {
    return valueLib.complexValue(out, outIm, [N, ...tailShape]);
  }
  return { shape: [N, ...tailShape], data: out };
}

// Recursive helper for the nested-array → flat write in _stackPerAtom.
function _writeNested(dst: Float64Array, offset: number, src: any): number {
  if (Array.isArray(src) || (src && src.BYTES_PER_ELEMENT !== undefined)) {
    for (let k = 0; k < src.length; k++) {
      offset = _writeNested(dst, offset, src[k]);
    }
    return offset;
  }
  dst[offset++] = +src;
  return offset;
}

// Unwrap an ArgInfo to its raw payload (value / measureType /
// callableType). Existing variant impls expect raw arg arrays; the
// typed-dispatch entry routes through this so impls stay unchanged.
// Variants that want the richer ArgInfo can read `ctx.argInfos`.
function _unwrapArgInfo(info: ArgInfo): any {
  if (info.kind === 'value')    return info.value;
  if (info.kind === 'measure')  return info.measureType;
  if (info.kind === 'function') return info.callableType;
  if (info.kind === 'kernel')   return info.callableType;
  return info;
}

// Typed-dispatch entry point. Accepts ArgInfo[] (rich arg
// representation including value, measure type, callable type, or
// IR axisStack context) and routes to the most-specific matching
// variant. Used by callers that want to dispatch on measure-arg
// types (P2 sample/batch/event shape) or IR-context (P3a axisStack)
// rather than just Value shapes.
//
// `dispatch` / `dispatchVariant` (raw-args form) keep their existing
// contract: raw Values in, raw result out. Callers wanting measure-
// arg or axisStack dispatch use `dispatchTyped` directly with
// ArgInfos. Variants that only constrain Value shapes are reachable
// from both entry points — the matcher auto-wraps raw args internally.
function dispatchTyped(name: string, argInfos: ArgInfo[], opts?: DispatchOpts): any {
  const decl = REGISTRY.get(name);
  if (!decl || !decl.variants || decl.variants.length === 0) return null;
  const v = _pickVariant(decl.variants, argInfos, opts || {});
  if (!v) return null;
  // Unwrap ArgInfo → raw arg for the impl. Existing impls expect raw
  // Values / measure types / callable types directly; new impls
  // that want axisStack or the full ArgInfo read ctx.argInfos.
  const rawArgs = argInfos.map(_unwrapArgInfo);
  return v.impl(rawArgs, Object.assign({}, opts || {}, { argInfos }));
}

// Try shape-pattern variant dispatch. Returns the impl's result on
// match; returns null (sentinel) when no variant matches. Callers
// that want a throw-on-no-match contract use `dispatch` instead;
// callers that want to fall through to a legacy path (e.g.
// `_maybeFastBroadcasted` falling to the per-cell `_broadcastApply`)
// use `dispatchVariant` and branch on null.
function dispatchVariant(name: string, args: any[], opts?: DispatchOpts): any {
  const decl = REGISTRY.get(name);
  if (!decl || !decl.variants || decl.variants.length === 0) return null;
  const safeOpts: DispatchOpts = opts || {};
  // Atom-aware first when opts.atomN is set (rank-2 atom-batched
  // would shadow rank-2 matrix exact-match otherwise).
  if (typeof safeOpts.atomN === 'number') {
    const ab = _pickVariantAtomAware(decl.variants, args, safeOpts);
    if (ab) return ab.variant.batched!(args, ab.N, safeOpts);
  }
  const v = _pickVariant(decl.variants, args, safeOpts);
  if (v) return v.impl(args, safeOpts);
  return null;
}

function dispatch(name: string, args: any[], opts?: DispatchOpts): any {
  const decl = REGISTRY.get(name);
  if (!decl) {
    throw new Error('ops.dispatch: no declaration for op \'' + name + '\'');
  }
  const safeOpts: DispatchOpts = opts || {};
  // Variants take priority. Most-specific match wins.
  if (decl.variants && decl.variants.length > 0) {
    // When the caller signals atom-batching (opts.atomN), try atom-
    // aware variants FIRST. This is the correct ordering because a
    // rank-2 atom-batched vector (shape=[N, n]) would otherwise match
    // a rank-2-matrix exact variant and run the wrong impl.
    if (typeof safeOpts.atomN === 'number') {
      const ab = _pickVariantAtomAware(decl.variants, args, safeOpts);
      if (ab) return ab.variant.batched!(args, ab.N, safeOpts);
    }
    const v = _pickVariant(decl.variants, args, safeOpts);
    if (v) return v.impl(args, safeOpts);
  }
  if (!decl.logical) {
    throw new Error('ops.dispatch: no variant matched and no logical fallback for op \'' +
      name + '\' (args: ' +
      args.map((a: any) => valueLib.isValue(a)
        ? 'shape=' + JSON.stringify(a.shape) + (a.t ? ',t=' + a.t : '')
        : typeof a).join(' / ') + ')');
  }
  const kind: OpKind = decl.kind || 'fixed-rank';

  // Rank-polymorphic and variadic ops bypass atom-batch detection.
  // The §2.1 shape contract still applies (Values keep their shape),
  // but the dispatcher just hands the inputs to `logical` as-is and
  // the op handles whatever rank/arity it received. Callers that
  // want atom-batched semantics over a rank-polymorphic op wrap
  // with explicit `broadcast(fn(op(_)), atom_batched)` —
  // engine-concepts §18.7.
  if (kind === 'rank-polymorphic' || kind === 'variadic') {
    return decl.logical(...args);
  }

  // Higher-order ops use `dispatchHigherOrder(name, irArgs, ctx)`.
  // Reaching `dispatch` (value-domain entry) means the caller routed
  // incorrectly — surface a clear error.
  if (kind === 'higher-order') {
    throw new Error('ops.dispatch: op \'' + name + '\' is higher-order; ' +
      'use ops.dispatchHigherOrder(name, irArgs, ctx) instead');
  }

  // Fixed-rank dispatch: classify each arg vs declared logical rank,
  // detect atom-batching by leading-axis convention, run logical or
  // batched accordingly.
  const argRanks = decl.argRanks;
  if (!argRanks) {
    throw new Error('ops.dispatch: op \'' + name +
      '\' is kind=fixed-rank but argRanks is missing');
  }
  if (args.length !== argRanks.length) {
    throw new Error('ops.dispatch: op \'' + name + '\' expects ' +
      argRanks.length + ' args, got ' + args.length);
  }
  // Classify each arg; collect the batch size.
  const batchedFlags: boolean[] = new Array(args.length);
  let N: number | null = null;
  for (let k = 0; k < args.length; k++) {
    const c = _classifyArg(args[k], argRanks[k]);
    batchedFlags[k] = c.batched;
    if (c.batched) {
      if (N === null) N = c.N!;
      else if (N !== c.N) {
        throw new Error('ops.dispatch: op \'' + name +
          '\' atom-batch size mismatch (arg ' + k + ' has N=' + c.N +
          ', earlier arg had N=' + N + ')');
      }
    }
  }
  // All atom-indep: one-shot.
  if (N === null) {
    return decl.logical(...args);
  }
  // Atom-batched. Try the fast-path first.
  if (decl.batched) {
    return decl.batched(args, N);
  }
  // Per-atom fallback.
  const perAtom: any[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const sliced = new Array(args.length);
    for (let k = 0; k < args.length; k++) {
      sliced[k] = _argAtAtom(args[k], batchedFlags[k], i);
    }
    perAtom[i] = decl.logical(...sliced);
  }
  return _stackPerAtom(perAtom, N);
}

// ---------------------------------------------------------------------
// Higher-order dispatch — for kind='higher-order' ops only
// ---------------------------------------------------------------------
//
// Higher-order ops (reduce / scan / filter / broadcast / aggregate)
// can't take pre-evaluated args because some args are callables
// (function-typed IR that resolves to {params, body}) and some are
// data IR that needs evaluation. The dispatch surface here takes
// the FULL IR node plus a context object carrying the engine's
// evaluator hooks:
//
//   ctx = {
//     env:           current evaluation environment,
//     evaluateExpr:  (ir, env) → value,
//     resolveFn:     (fnIR, env) → { params, body, paramKwargs? } | null,
//   }
//
// Passing the full IR (rather than just `ir.args`) lets ops that
// use kwargs / fields / other IR shape elements read what they
// need (broadcast accepts kwargs; reduce/scan/filter only need
// args). The op's `logical(ir, ctx)` then runs its own resolution
// + iteration against `ctx`. The dispatcher just routes; it
// doesn't try to understand callable shapes (each higher-order
// op's semantics is distinct — reduce vs scan vs filter vs
// broadcast vs aggregate).
//
// This separates the higher-order entry from the value-domain
// `dispatch(name, args)` so the type contracts stay clean: callers
// route based on what they have (evaluated values vs raw IR + ctx).

interface HigherOrderCtx {
  env: any;
  evaluateExpr: (ir: any, env: any) => any;
  resolveFn: (fnIR: any, env: any) => any;
}

function dispatchHigherOrder(name: string, ir: any, ctx: HigherOrderCtx): any {
  const decl = REGISTRY.get(name);
  if (!decl) {
    throw new Error('ops.dispatchHigherOrder: no declaration for op \'' + name + '\'');
  }
  if (decl.kind !== 'higher-order') {
    throw new Error('ops.dispatchHigherOrder: op \'' + name +
      '\' is not kind=higher-order (got ' + (decl.kind || 'fixed-rank') +
      '); use ops.dispatch instead');
  }
  if (!ctx || typeof ctx.evaluateExpr !== 'function'
      || typeof ctx.resolveFn !== 'function') {
    throw new Error('ops.dispatchHigherOrder: ctx must provide env, ' +
      'evaluateExpr, and resolveFn');
  }
  if (typeof decl.logical !== 'function') {
    throw new Error("ops.dispatchHigherOrder: op '" + name +
      "' has no logical impl");
  }
  return decl.logical(ir, ctx);
}

module.exports = {
  register,
  registerVariant,
  lookup,
  isDeclared,
  hasVariantFor,
  listDeclared,
  signatureOf,
  dispatch,
  dispatchVariant,
  dispatchTyped,
  dispatchHigherOrder,
  // ArgInfo constructors for typed dispatch (P2 / P3a integration):
  argInfoFromValue,
  argInfoFromMeasure,
  argInfoFromFunction,
  argInfoFromKernel,
  // Exported for the conformance harness:
  _classifyArg,
  _argAtAtom,
  _stackPerAtom,
  _matchArgPattern,
  _matchValueArg,
  _matchMeasureArg,
  _matchShape,
  _matchAxisStack,
  _matchVariant,
  _specificityScore,
  _pickVariant,
  _autoArgInfo,
};
