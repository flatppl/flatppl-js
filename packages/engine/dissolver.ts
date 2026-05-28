'use strict';

// =====================================================================
// dissolver.ts — term-rewriter for broadcast / aggregate dissolution
// =====================================================================
//
// Architectural background: engine-concepts §20 (cross-engine) and the
// "Broadcast / aggregate dissolution" section in
// packages/engine/ARCHITECTURE.md (JS-engine specifics).
//
// The dissolver rewrites higher-order constructs (broadcast, broadcasted,
// aggregate) in a binding's lifted IR into direct batched-op calls when
// the body is structurally a single op application of an inherently-
// batched primitive. The rewrite is a term-rewriting equivalence on
// FlatPIR; correctness follows from spec §04's
// `broadcasted(op)(args) ≡ broadcast(functionof(op(_,…)), args)` rule
// plus the observation that for op ∈ ARITH_OPS (and other batched
// primitives) `broadcasted(op) ≡ op` — the wrapper collapses.
//
// Phase 2 (this module's initial scope): single-op broadcast dissolution.
// Recognises
//     broadcast(functionof(op(refs…)), args…)         (positional)
//     broadcast(functionof(op(refs…)), name=arg, …)   (kwarg form)
// where `op` is in DISSOLVE_SAFE_OPS, the body args are exactly the
// functionof params in declared order (no closed-over scope, no
// argument swizzling), and the call's arity matches the params.
// Rewrites to a direct `op(args…)` call.
//
// Phases 3-6 will extend the pattern set:
//   3 — multi-op body dissolution (expression trees of safe ops)
//   4 — user-fn inlining (broadcast head referring to a user-defined
//       functionof binding gets inlined, then Phase 3 applies)
//   5 — aggregate dissolution (reduction axes become explicit
//       `reduce_along` ops)
//   6 — atom-axis unification (the atom dispatch path becomes a
//       degenerate case of the dissolver's broadcast-axis handling)
//
// The dissolver is conservative: when a broadcast doesn't match a
// dissolvable pattern, it is left as-is. The existing `_broadcastApply`
// runtime path handles the residual cases unchanged.

// ---------------------------------------------------------------------
// Dissolve-safe op set
// ---------------------------------------------------------------------
//
// These ops are genuinely *elementwise at any rank* through value-ops —
// `valueOps.<op>(A, B)` operates pointwise on flat row-major storage
// regardless of how many outer / inner axes the inputs carry. So when
// the broadcast args reach the dispatcher as flat tagged Values (the
// canonical post-C7 storage convention — engine-concepts §2.1, the
// outerRank tag), dissolving `broadcast(functionof(<op>(_,_)), A, B)`
// to a direct `<op>(A, B)` preserves spec semantics: the elementwise
// loop over flat storage is identical to the per-cell broadcast loop.
//
// Phase 2 keeps the set narrow on purpose. The risky ops (`mul`,
// `div`, `pow`, unary scalars like `exp` / `log`) are *not* universally
// elementwise:
//   - `mul` on rank-1 vectors does inner / outer product (per the
//     Klein-4 transpose tag), not elementwise multiply.
//   - `div` / `pow` have no value-ops elementwise impl; the JS scalar
//     fn returns NaN on Values.
//   - `exp` / `log` / `sqrt` / `sin` / … route through `broadcast1` in
//     ARITH_OPS_N, which assumes rank-0 or rank-1 scalar inputs —
//     higher-rank inputs (vector-per-cell broadcasts) silently break.
//
// Phase 3 will widen the set by:
//   1. Threading typeinfer's `inferredType` to the dissolver so the
//      arg ranks are known at lift time, OR
//   2. Adding per-op "elementwise-at-rank-k" metadata and routing the
//      dissolver to consult it, OR
//   3. Providing valueOps elementwise impls for the remaining scalar
//      ops (an `elementwise(fn, …)` helper over flat data) so the
//      dispatch is uniform.
//
// Until then, dissolving `add` / `sub` / `neg` / `pos` already
// captures the common dotted-binary cases that arise from
// `Y = A .+ B`, `Y = A .- B`, `Y = -A`, etc. — the perf-relevant
// surface for vector-of-scalar data arrays.
//
// Two tiers (Phase 3 type-aware widening):
//   - DISSOLVE_AT_ANY_RANK_OPS: safe regardless of the broadcast
//     args' shapes — value-ops handles them elementwise at any
//     rank.
//   - DISSOLVE_SCALAR_ONLY_OPS: safe only when ALL outer broadcast
//     args have `inferredType.kind === 'scalar'`. Includes
//     multiplicative arith (which has matrix semantics on rank-1)
//     and unary scalar maths (which route through `broadcast1`
//     and only handle rank-0 / rank-1 inputs correctly).
const DISSOLVE_AT_ANY_RANK_OPS: Set<string> = new Set([
  // Elementwise arith over flat row-major data (value-ops handles
  // any rank uniformly).
  'add', 'sub', 'neg', 'pos',
  // Declared fixed-rank ops from `ops-declarations.ts`. The
  // `ops.dispatch` dispatcher auto-detects atom-batching via the
  // leading-axis convention (rank == argRanks[k] + 1) and either
  // calls the op's `batched` fast-path or runs the per-atom
  // slicing fallback. Either way, `broadcast(<op>, args…) ≡
  // <op>(args…)` is sound — atom-axis unification (engine-concepts
  // §20.1; Phase 6 of the dissolution migration) means the atom
  // axis is just another outer axis the dispatcher handles
  // uniformly with the user-broadcast axis. Same-type guard on
  // the outer broadcast args still applies.
  'cross', 'self_outer', 'trace', 'det', 'logabsdet', 'diagmat',
  'inv', 'lower_cholesky', 'row_gram', 'col_gram',
]);

const DISSOLVE_SCALAR_ONLY_OPS: Set<string> = new Set([
  // Multiplicative arith (matrix semantics on rank-1 vectors).
  'mul', 'div', 'divide', 'mod', 'pow',
  // Unary scalar maths — broadcast1-routed, scalar-only.
  'abs', 'abs2', 'exp', 'log', 'log10', 'log1p', 'expm1', 'sqrt',
  'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
  'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
  'floor', 'ceil', 'round',
  // Special functions / link functions.
  'gamma', 'loggamma',
  'logit', 'invlogit', 'probit', 'invprobit',
  // Pairwise scalar reductions.
  'min', 'max',
  // Comparisons + predicates + logic — produce scalar booleans.
  'lt', 'le', 'gt', 'ge', 'equal', 'unequal',
  'isfinite', 'isinf', 'isnan', 'iszero',
  'land', 'lor', 'lxor', 'lnot', 'ifelse',
  // Scalar restrictors + complex constructors.
  'boolean', 'integer', 'real', 'imag', 'conj', 'complex', 'cis',
  // Identity (broadcast(identity, A) ≡ A; harmless dissolution).
  'identity',
]);

// Composite check: op is safe to dissolve at all? (Either tier.)
function _isSafeOp(name: string): boolean {
  return DISSOLVE_AT_ANY_RANK_OPS.has(name)
    || DISSOLVE_SCALAR_ONLY_OPS.has(name);
}

// Legacy alias retained for the structural unit tests that probe
// the safe-op set directly. Sums both tiers — equivalent to
// `_isSafeOp` for callers that just want "is this op dissolvable
// in principle".
const DISSOLVE_SAFE_OPS: Set<string> = new Set([
  ...DISSOLVE_AT_ANY_RANK_OPS,
  ...DISSOLVE_SCALAR_ONLY_OPS,
]);

// ---------------------------------------------------------------------
// Dissolution rewrites
// ---------------------------------------------------------------------

// Structural equality on FlatPIR type annotations. Used by the
// soundness check below: dissolving a broadcast to a direct call is
// only safe when all positional arg shapes match exactly (so
// elementwise valueOps semantics line up with the broadcast loop).
function _typesEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case 'scalar':
      return a.prim === b.prim;
    case 'array':
      if (a.rank !== b.rank) return false;
      if (!Array.isArray(a.shape) || !Array.isArray(b.shape)) return false;
      if (a.shape.length !== b.shape.length) return false;
      for (let i = 0; i < a.shape.length; i++) {
        if (a.shape[i] !== b.shape[i]) return false;
      }
      return _typesEqual(a.elem, b.elem);
    case 'record':
      if (!a.fields || !b.fields) return false;
      if (Object.keys(a.fields).length !== Object.keys(b.fields).length) return false;
      for (const k in a.fields) {
        if (!(k in b.fields)) return false;
        if (!_typesEqual(a.fields[k], b.fields[k])) return false;
      }
      return true;
    default:
      // Conservative: anything else (measure / kernel / tvar / failed /
      // deferred / any) — refuse equivalence so the dissolver leaves
      // the broadcast in place.
      return false;
  }
}

// Look up the inferredType + phase of a positional broadcast arg.
// Returns null if the arg is not a binding ref or if no usable type
// can be derived — the conservative response is "don't dissolve."
//
// Synthetic anon bindings created by `liftInlineSubexpressions` lack
// an `inferredType` (typeinfer runs in analyzer pass 7, before lift
// adds them). For anons whose IR has been dissolved to a direct
// safe-op call, we can derive the type lazily from the call's args.
// This is what `_resolveBindingType` does.
function _argTypeAndPhase(arg: any, bindings: any): { type: any; phase: any } | null {
  return _resolveExprType(arg, bindings, 0);
}

// Recursively resolve an IR expression's effective (type, phase).
//
// Handles three IR shapes:
//   - `ref`: looks up the binding's `inferredType` if present;
//     otherwise descends into the binding's IR (lazy resolution for
//     synthetic anon bindings created by lift after typeinfer).
//   - `call` to a DISSOLVE-safe op: type is the args' common type,
//     same-phase. Mirrors elementwise semantics — this is what
//     dissolution would produce if it folded the inline call out.
//   - `lit` / `const`: scalar real, fixed-phase. Permitted as a
//     constituent of a parent expression but doesn't anchor a
//     type on its own.
//
// Recursion is bounded to keep pathological cases cheap.
function _resolveExprType(
  expr: any,
  bindings: any,
  depth: number,
): { type: any; phase: any } | null {
  if (depth > 8) return null;
  if (!expr) return null;
  if (expr.kind === 'ref') {
    if (expr.ns !== 'self') return null;
    if (!bindings || !bindings.get) return null;
    const b = bindings.get(expr.name);
    if (!b) return null;
    const t = b.inferredType;
    if (t && t.kind !== 'deferred' && t.kind !== 'failed') {
      return { type: t, phase: b.phase };
    }
    if (b.ir) return _resolveExprType(b.ir, bindings, depth + 1);
    return null;
  }
  if (expr.kind === 'lit' || expr.kind === 'const') {
    // Scalar real, fixed-phase. Returned as a constituent type;
    // callers must combine with the other args' types — a pure
    // literal can't anchor an outer-arg uniformity check.
    return { type: { kind: 'scalar', prim: 'real' }, phase: 'fixed' };
  }
  if (expr.kind === 'call' && expr.op) {
    if (!_isSafeOp(expr.op)) return null;
    const argTPs: { type: any; phase: any }[] = [];
    const callArgs: any[] = expr.args || [];
    if (callArgs.length === 0) return null;
    let nonLiteralCount = 0;
    for (const a of callArgs) {
      const sub = _resolveExprType(a, bindings, depth + 1);
      if (!sub) return null;
      argTPs.push(sub);
      if (a && a.kind !== 'lit' && a.kind !== 'const') nonLiteralCount++;
    }
    // Need at least one non-literal arg to anchor the result type.
    if (nonLiteralCount === 0) return argTPs[0];  // all scalars
    // Skip literal contributors when checking uniformity (a literal
    // is scalar/fixed and broadcasts harmlessly via valueOps).
    let anchor: { type: any; phase: any } | null = null;
    for (let i = 0; i < argTPs.length; i++) {
      const a = callArgs[i];
      if (a && (a.kind === 'lit' || a.kind === 'const')) continue;
      if (anchor === null) anchor = argTPs[i];
      else {
        if (anchor.phase !== argTPs[i].phase) return null;
        if (!_typesEqual(anchor.type, argTPs[i].type)) return null;
      }
    }
    return anchor;
  }
  return null;
}

// Walk an IR expression bottom-up, dissolving any embedded broadcast
// forms that match a dissolvable pattern. Returns the (possibly new)
// IR; mutates nothing on the input.
//
// `bindings` is the post-lift map; the dissolver consults each
// candidate broadcast arg's inferredType + phase via the bindings
// lookup. Passing `null` disables the type-aware soundness check —
// useful for unit tests of the structural pattern matcher in
// isolation; production callers always pass the map.
function dissolveExpr(ir: any, bindings: any): any {
  if (!ir || typeof ir !== 'object') return ir;
  if (ir.kind !== 'call') return ir;
  // Walk children first so nested broadcasts see their dissolved
  // children. Only rebuild if any child changed (preserve object
  // identity in the common no-op case).
  let changed = false;
  let newArgs: any[] | null = null;
  if (Array.isArray(ir.args)) {
    newArgs = new Array(ir.args.length);
    for (let i = 0; i < ir.args.length; i++) {
      const w = dissolveExpr(ir.args[i], bindings);
      newArgs[i] = w;
      if (w !== ir.args[i]) changed = true;
    }
  }
  let newKwargs: Record<string, any> | null = null;
  if (ir.kwargs && typeof ir.kwargs === 'object') {
    newKwargs = {};
    for (const k in ir.kwargs) {
      const w = dissolveExpr(ir.kwargs[k], bindings);
      newKwargs[k] = w;
      if (w !== ir.kwargs[k]) changed = true;
    }
  }
  let newBody = ir.body;
  if (ir.body) {
    newBody = dissolveExpr(ir.body, bindings);
    if (newBody !== ir.body) changed = true;
  }
  let newFields: any[] | null = null;
  if (Array.isArray(ir.fields)) {
    newFields = new Array(ir.fields.length);
    for (let i = 0; i < ir.fields.length; i++) {
      const f = ir.fields[i];
      const wv = dissolveExpr(f && f.value, bindings);
      newFields[i] = (wv === f.value) ? f : { ...f, value: wv };
      if (wv !== f.value) changed = true;
    }
  }
  let walked = ir;
  if (changed) {
    walked = { ...ir };
    if (newArgs) walked.args = newArgs;
    if (newKwargs) walked.kwargs = newKwargs;
    if (newBody !== ir.body) walked.body = newBody;
    if (newFields) walked.fields = newFields;
  }
  // Pattern-match `broadcast(...)` and `aggregate(...)` at THIS
  // level after the children are dissolved.
  if (walked.op === 'broadcast') {
    const dissolved = _tryDissolveSingleOp(walked, bindings);
    if (dissolved) return dissolved;
  }
  if (walked.op === 'aggregate') {
    const dissolved = _tryDissolveAggregate(walked, bindings);
    if (dissolved) return dissolved;
  }
  return walked;
}

// =====================================================================
// Phase 5 — aggregate dissolution
// =====================================================================
//
// `aggregate(f_reduction, output_axes, body_expr)` admits a small
// set of structural rewrites into direct tensor-op calls. The
// runtime AGGREGATE_PATTERNS specialisers in `sampler-aggregate.ts`
// pattern-match these at evaluation time; Phase 5 lifts a subset of
// those matchers up to IR-time so the dissolved FlatPIR carries the
// direct op calls (and other backends — StableHLO / TF.js — can
// consume them without re-recognising the aggregate shape).
//
// Coverage (Phase 5 — narrow start):
//   - Matmul-family (4 transpose variants):
//       aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])  ≡  A · B
//     dissolves to `mul(<maybe transpose>(A), <maybe transpose>(B))`.
//   - Matvec-family (2 transpose variants):
//       aggregate(sum, [.i], A[.i, .j] * v[.j])          ≡  A · v
//     dissolves similarly.
//
// Both rewrites preserve spec semantics: `mul` on rank-2 matrices
// is matrix product per the Klein-4 transpose-tag dispatch in
// value-ops; a Klein-4 transpose is a free tag flip (no allocation)
// so emitting `transpose(A)` ahead of `mul` is cheap.
//
// Soundness gate: both source operands of the body's `mul` must be
// binding refs (or otherwise resolvable to a concrete inferredType
// via the lazy resolver). The runtime AGGREGATE_PATTERNS path
// remains as the cold fallback for everything else (outer-product,
// batched-matmul, pure-axis-reduction, non-matched shapes).

function _aggregateBodyClassifyA(
  fac: any, iName: string,
): { src: any; trans: boolean; jName: string } | null {
  if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
  if (!fac.args || fac.args.length !== 3) return null;
  const s0 = fac.args[1], s1 = fac.args[2];
  if (!s0 || s0.kind !== 'axis') return null;
  if (!s1 || s1.kind !== 'axis') return null;
  if (s0.name === iName) return { src: fac.args[0], trans: false, jName: s1.name };
  if (s1.name === iName) return { src: fac.args[0], trans: true, jName: s0.name };
  return null;
}

function _aggregateBodyClassifyB(
  fac: any, kName: string,
): { src: any; trans: boolean; jName: string } | null {
  if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
  if (!fac.args || fac.args.length !== 3) return null;
  const s0 = fac.args[1], s1 = fac.args[2];
  if (!s0 || s0.kind !== 'axis') return null;
  if (!s1 || s1.kind !== 'axis') return null;
  if (s1.name === kName) return { src: fac.args[0], trans: false, jName: s0.name };
  if (s0.name === kName) return { src: fac.args[0], trans: true, jName: s1.name };
  return null;
}

function _aggregateBodyClassifyV(
  fac: any, jName: string,
): any | null {
  if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
  if (!fac.args || fac.args.length !== 2) return null;
  const s0 = fac.args[1];
  if (!s0 || s0.kind !== 'axis') return null;
  if (s0.name !== jName) return null;
  return fac.args[0];
}

function _wrapTranspose(ir: any, doTranspose: boolean): any {
  if (!doTranspose) return ir;
  return { kind: 'call', op: 'transpose', args: [ir] };
}

// Verify a source IR resolves to a concrete inferredType — same
// gate the broadcast dissolver uses. Returns the resolved
// (type, phase) or null.
function _checkSourceType(ir: any, bindings: any): { type: any; phase: any } | null {
  return _resolveExprType(ir, bindings, 0);
}

// Inline a user-fn body into the aggregate's body position.
//
// Detects `aggregate(sum, [...], <user_fn>(args...))` where the body
// is a user-defined call (target: { ns: 'self', name }). Looks up
// the function's binding, retrieves its `functionof` body, and
// substitutes the functionof's placeholders with the call's args.
//
// Returns the rewritten body IR on success; null if:
//  - body isn't a user call,
//  - the target binding isn't a functionof,
//  - the call's arg count doesn't match the params,
//  - any placeholder occurrence in the body isn't substitutable
//    (the existing _substituteBody check fails — same gate the
//    broadcast Phase 4 inliner uses).
//
// This is the aggregate-side analogue of Phase 4's broadcast user-fn
// inlining (engine-concepts §20.9 fusion thread (c) sub-item 2).
// After inlining, the caller re-attempts the matmul/matvec
// recognisers on the inlined body.
function _inlineAggregateBody(bodyIR: any, bindings: any): any | null {
  if (!bodyIR || bodyIR.kind !== 'call') return null;
  // Must be a user-defined call (target set, no built-in op).
  if (!bodyIR.target || bodyIR.target.ns !== 'self' || bodyIR.op) return null;
  if (!bindings || !bindings.get) return null;

  const fnBinding = bindings.get(bodyIR.target.name);
  const fnIR = fnBinding && fnBinding.ir;
  if (!fnIR || fnIR.kind !== 'call' || fnIR.op !== 'functionof') return null;

  const params: string[] = Array.isArray(fnIR.params) ? fnIR.params : [];
  if (params.length === 0) return null;
  const fnBody = fnIR.body;
  if (!fnBody) return null;

  // Match call args to params. Aggregate body is a positional call
  // (the body is an EXPRESSION in scope of axis names — the user
  // doesn't write kwargs here in practice). Kwarg form bails.
  if (bodyIR.kwargs && Object.keys(bodyIR.kwargs).length > 0) return null;
  const callArgs: any[] = bodyIR.args || [];
  if (callArgs.length !== params.length) return null;

  // Substitute placeholders in the fn body with call args. Reuse the
  // existing _substituteBody machinery — it walks the IR and replaces
  // %local refs by index. For aggregate the substituted body lives
  // in axis-name scope, so it can contain `(get <src> <axis>)` etc.
  // which _substituteBody doesn't recognise. Walk the body ourselves
  // with a placeholder-aware visitor that handles ANY IR kind.
  const paramIndex: Map<string, number> = new Map();
  for (let i = 0; i < params.length; i++) paramIndex.set(params[i], i);
  return _substituteAllowingAxes(fnBody, paramIndex, callArgs);
}

// Recursively substitute `%local` refs in an IR expression with the
// corresponding positional args from `bcArgs`. Unlike _substituteBody
// (which gates on DISSOLVE_SAFE_OPS), this version is permissive
// about op names — the caller (`_tryDissolveAggregate`) will validate
// the resulting structure. Returns null if any placeholder can't be
// resolved.
function _substituteAllowingAxes(
  expr: any,
  paramIndex: Map<string, number>,
  bcArgs: any[],
): any | null {
  if (!expr) return expr;
  if (expr.kind === 'lit' || expr.kind === 'const' || expr.kind === 'axis') {
    return expr;
  }
  if (expr.kind === 'ref') {
    if (expr.ns === '%local') {
      const idx = paramIndex.get(expr.name);
      if (idx === undefined) return null;
      return bcArgs[idx];
    }
    return expr;
  }
  if (expr.kind === 'call') {
    const inArgs: any[] = expr.args || [];
    const outArgs: any[] = new Array(inArgs.length);
    for (let i = 0; i < inArgs.length; i++) {
      const sub = _substituteAllowingAxes(inArgs[i], paramIndex, bcArgs);
      if (sub === null) return null;
      outArgs[i] = sub;
    }
    const out: any = { kind: 'call' };
    if (expr.op)     out.op = expr.op;
    if (expr.target) out.target = expr.target;
    if (outArgs.length > 0) out.args = outArgs;
    if (expr.kwargs) {
      const outKwargs: Record<string, any> = {};
      for (const k in expr.kwargs) {
        const sub = _substituteAllowingAxes(expr.kwargs[k], paramIndex, bcArgs);
        if (sub === null) return null;
        outKwargs[k] = sub;
      }
      out.kwargs = outKwargs;
    }
    if (expr.loc) out.loc = expr.loc;
    return out;
  }
  return expr;
}

// Wrap a dissolved IR in a closed-form reduction-correction factor.
// For `sum`-reduction this is the identity; for `mean`-reduction we
// divide by the contraction-axis size (sum / k = mean × k). Returns
// null when the correction can't be expressed in closed form for
// the given reducer.
//
// Engineering note: emits `mul(lit, body)` rather than `divide(body,
// lit)` so the result routes through the variant registry's scalar-
// broadcast variant (mul(rank-0, rank-2) → scalar broadcast) — the
// existing dissolved path. Avoids introducing a new `divide` shape
// the registry doesn't yet cover.
function _applyReducerCorrection(
  reducer: string,
  body: any,
  contractionSize: number | '%dynamic',
): any | null {
  if (reducer === 'sum') return body;
  if (reducer === 'mean') {
    if (contractionSize === '%dynamic' || typeof contractionSize !== 'number') {
      // Can't compute the scalar correction without a static dim size.
      return null;
    }
    if (contractionSize <= 0) return null;
    const inv = 1 / contractionSize;
    return {
      kind: 'call', op: 'mul',
      args: [
        { kind: 'lit', value: inv, numType: 'real' },
        body,
      ],
    };
  }
  // 'prod' / 'var' / 'std' / 'max' / 'min' don't have a closed-form
  // matmul equivalent; refuse and leave the aggregate on the cold
  // path.
  return null;
}

// Resolve the contraction-axis size from a rank-2 array's inferred
// type and the transposition state of its get-indexing.
//   - trans=false → logical shape is [size(.i), size(.j)],
//     contraction axis is .j → shape[1].
//   - trans=true  → logical shape is [size(.j), size(.i)],
//     contraction axis is .j → shape[0].
function _contractionSizeFromA(
  tA: { type: any; phase: any },
  trans: boolean,
): number | '%dynamic' {
  const shape = tA && tA.type && tA.type.shape;
  if (!Array.isArray(shape) || shape.length < 2) return '%dynamic';
  const dim = trans ? shape[0] : shape[1];
  if (typeof dim === 'number') return dim;
  return '%dynamic';
}

function _tryDissolveAggregate(aggIR: any, bindings: any): any | null {
  if (!aggIR || aggIR.kind !== 'call' || aggIR.op !== 'aggregate') return null;
  const args = aggIR.args || [];
  if (args.length !== 3) return null;
  const fIR = args[0], axesIR = args[1];
  let bodyIR = args[2];
  // Reduction must be a closed-form recognised reducer. 'sum' is the
  // canonical case (direct matmul/matvec); 'mean' = sum / k where k
  // is the contraction-axis length (resolved from operand shape).
  // Other reductions (prod / var / std / min / max) don't have a
  // closed-form matmul equivalent and stay on the cold path.
  if (!fIR || fIR.kind !== 'ref') return null;
  const reducer = fIR.name;
  if (reducer !== 'sum' && reducer !== 'mean') return null;
  if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
  const outAxes = axesIR.args || [];
  for (const a of outAxes) {
    if (!a || a.kind !== 'axis') return null;
  }
  // User-fn body inlining (engine-concepts §20.9 fusion thread (c)
  // sub-item 2): if the body is a user-defined call, look up the
  // function's `functionof` binding and inline its body with
  // placeholders substituted. This lets `aggregate(sum, [.i, .k],
  // myMul(A[.i,.j], B[.j,.k]))` dissolve when `myMul = (a, b) ->
  // a * b` — after inlining, the body becomes `mul(A[.i,.j],
  // B[.j,.k])` and the existing matchers below recognise it.
  if (bodyIR && bodyIR.kind === 'call' && bodyIR.target
      && bodyIR.target.ns === 'self' && !bodyIR.op) {
    const inlined = _inlineAggregateBody(bodyIR, bindings);
    if (inlined) bodyIR = inlined;
  }
  // Body must be a `mul(<getA>, <getB>)` call.
  if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
  if (!bodyIR.args || bodyIR.args.length !== 2) return null;

  // Matmul / Outer-product: 2 output axes (.i, .k or .i, .j).
  if (outAxes.length === 2) {
    const iName = outAxes[0].name;
    const kName = outAxes[1].name;
    if (iName === kName) return null;
    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];

    // Matmul: body is mul of two-axis indexings sharing a reduction
    // axis (e.g. A[.i, .j] * B[.j, .k] reducing over .j).
    for (const [fA, fB] of [[f1, f2], [f2, f1]]) {
      const ca = _aggregateBodyClassifyA(fA, iName);
      const cb = _aggregateBodyClassifyB(fB, kName);
      if (!ca || !cb) continue;
      if (ca.jName !== cb.jName) continue;
      if (ca.jName === iName || ca.jName === kName) continue;
      const tA = _checkSourceType(ca.src, bindings);
      const tB = _checkSourceType(cb.src, bindings);
      if (!tA || !tB) return null;
      if (tA.phase !== tB.phase) return null;
      if (!tA.type || tA.type.kind !== 'array' || tA.type.rank !== 2) return null;
      if (!tB.type || tB.type.kind !== 'array' || tB.type.rank !== 2) return null;
      const A = _wrapTranspose(ca.src, ca.trans);
      const B = _wrapTranspose(cb.src, cb.trans);
      const mulCall: any = { kind: 'call', op: 'mul', args: [A, B] };
      // Apply the closed-form reducer correction. For 'sum' this is
      // identity; for 'mean' we wrap in `(1/k) * mulCall` where k is
      // the contraction-axis length (resolved from A's shape +
      // transposition state). Returns null on `mean` with dynamic
      // dim — falls back to the cold AGGREGATE_PATTERNS path.
      const k = _contractionSizeFromA(tA, ca.trans);
      const out = _applyReducerCorrection(reducer, mulCall, k);
      if (!out) return null;
      if (aggIR.loc) out.loc = aggIR.loc;
      return out;
    }

    // Outer product: body is mul of two 1-axis indexings with
    // distinct axes (e.g. u[.i] * v[.j]) — no reduction axis. Only
    // sum-reduction is meaningful here (no contraction axis means
    // 'mean' has no axis-length to divide by); mean falls back to
    // the cold path.
    if (reducer === 'sum') {
      for (const [fU, fV] of [[f1, f2], [f2, f1]]) {
        const uSrc = _aggregateBodyClassifyV(fU, iName);
        const vSrc = _aggregateBodyClassifyV(fV, kName);
        if (!uSrc || !vSrc) continue;
        const tU = _checkSourceType(uSrc, bindings);
        const tV = _checkSourceType(vSrc, bindings);
        if (!tU || !tV) return null;
        if (tU.phase !== tV.phase) return null;
        if (!tU.type || tU.type.kind !== 'array' || tU.type.rank !== 1) return null;
        if (!tV.type || tV.type.kind !== 'array' || tV.type.rank !== 1) return null;
        const out: any = {
          kind: 'call', op: 'mul',
          args: [uSrc, { kind: 'call', op: 'transpose', args: [vSrc] }],
        };
        if (aggIR.loc) out.loc = aggIR.loc;
        return out;
      }
    }
    return null;
  }

  // Matvec: 1 output axis (.i); body is mul of a 2-axis indexing (A)
  // and a 1-axis indexing (v), sharing a reduction axis.
  if (outAxes.length === 1) {
    const iName = outAxes[0].name;
    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];
    for (const [fA, fV] of [[f1, f2], [f2, f1]]) {
      const ca = _aggregateBodyClassifyA(fA, iName);
      if (!ca) continue;
      const vSrc = _aggregateBodyClassifyV(fV, ca.jName);
      if (!vSrc) continue;
      const tA = _checkSourceType(ca.src, bindings);
      const tV = _checkSourceType(vSrc, bindings);
      if (!tA || !tV) return null;
      if (tA.phase !== tV.phase) return null;
      if (!tA.type || tA.type.kind !== 'array' || tA.type.rank !== 2) return null;
      if (!tV.type || tV.type.kind !== 'array' || tV.type.rank !== 1) return null;
      const A = _wrapTranspose(ca.src, ca.trans);
      const mulCall: any = { kind: 'call', op: 'mul', args: [A, vSrc] };
      // Mean correction via the same closed-form factor as matmul.
      const k = _contractionSizeFromA(tA, ca.trans);
      const out = _applyReducerCorrection(reducer, mulCall, k);
      if (!out) return null;
      if (aggIR.loc) out.loc = aggIR.loc;
      return out;
    }
    return null;
  }

  return null;
}

// Recursively substitute `%local` refs in a body expression with the
// corresponding broadcast args, and verify every op-call in the body
// is in DISSOLVE_SAFE_OPS. Returns the substituted IR on success, or
// null when the body contains an unsupported construct.
//
// Constraints applied at every node:
//   - call.op must be in DISSOLVE_SAFE_OPS (no user-defined calls).
//   - call.args are recursively substituted; call.kwargs/fields not
//     supported (would imply a non-elementwise op).
//   - `%local` refs map to the broadcast arg via the params index.
//   - `self` refs are allowed ONLY when their target binding is
//     fixed-phase (held constant per cell — broadcast and direct
//     elementwise call produce the same result on a constant).
//   - Literals (`lit`, `const`) pass through unchanged.
//   - Anything else (nested functionof, broadcast, measure ops, …)
//     fails the check — return null.
function _substituteBody(
  expr: any,
  paramIndex: Map<string, number>,
  bcArgs: any[],
  bindings: any,
  argsAreAllScalar: boolean,
): any | null {
  if (!expr) return null;
  switch (expr.kind) {
    case 'lit':
    case 'const':
      return expr;
    case 'ref':
      if (expr.ns === '%local') {
        const idx = paramIndex.get(expr.name);
        if (idx === undefined) return null;  // unknown placeholder
        return bcArgs[idx];
      }
      if (expr.ns === 'self') {
        // Held constant: require the target binding be fixed-phase.
        if (!bindings || !bindings.get) return null;
        const b = bindings.get(expr.name);
        if (!b || b.phase !== 'fixed') return null;
        return expr;
      }
      return null;  // unknown namespace
    case 'call':
      if (!expr.op) return null;
      // Two-tier op-safety check:
      //   - rank-agnostic ops: always allowed in the body.
      //   - scalar-only ops: only allowed when ALL outer broadcast
      //     args are scalar (so the body runs entirely on scalars
      //     once substitution lands the runtime args in place).
      if (DISSOLVE_AT_ANY_RANK_OPS.has(expr.op)) {
        // OK regardless of arg ranks.
      } else if (DISSOLVE_SCALAR_ONLY_OPS.has(expr.op)) {
        if (!argsAreAllScalar) return null;
      } else {
        return null;  // unknown / unsafe op
      }
      if (expr.kwargs && Object.keys(expr.kwargs).length > 0) return null;
      if (Array.isArray(expr.fields) && expr.fields.length > 0) return null;
      if (expr.body) return null;  // nested functionof — bail
      const inArgs: any[] = expr.args || [];
      const outArgs: any[] = new Array(inArgs.length);
      for (let i = 0; i < inArgs.length; i++) {
        const sub = _substituteBody(
          inArgs[i], paramIndex, bcArgs, bindings, argsAreAllScalar);
        if (sub === null) return null;
        outArgs[i] = sub;
      }
      const out: any = { kind: 'call', op: expr.op, args: outArgs };
      if (expr.loc) out.loc = expr.loc;
      return out;
    default:
      return null;
  }
}

// Try Phase-2 / Phase-3 broadcast dissolution:
//
//   Phase 2: broadcast(functionof(<op>(_arg1_, _arg2_, …)), args…)
//     where the body is a SINGLE call with args being the placeholders
//     in declared order.
//
//   Phase 3 (this superset): the body may be an arbitrary expression
//     tree of DISSOLVE_SAFE_OPS, with `%local` refs at the leaves
//     mapping to the broadcast args and fixed-phase `self` refs held
//     constant. Each substitution preserves the spec broadcast
//     semantics because every op in the tree is elementwise-at-any-
//     rank under value-ops.
//
// On match, returns a substituted IR (the body's expression tree with
// placeholders replaced by the broadcast args). On no match, returns
// null and the caller leaves the broadcast IR in place.
function _tryDissolveSingleOp(bcIR: any, bindings: any): any | null {
  if (!bcIR || bcIR.kind !== 'call' || bcIR.op !== 'broadcast') return null;
  const bcArgs: any[] = bcIR.args || [];
  if (bcArgs.length < 1) return null;
  let head = bcArgs[0];
  // Phase 4 — user-fn inlining: when the head is a `self`-ref to a
  // user-defined `functionof` binding, look up its lifted IR and
  // inline. The dissolver then applies the same Phase-3 substitution
  // path that the inline-`functionof` form takes. The inlined fn must
  // itself be dissolvable (its body must satisfy the Phase-3 walker).
  if (head && head.kind === 'ref' && head.ns === 'self' && bindings && bindings.get) {
    const fnBinding = bindings.get(head.name);
    const fnIR = fnBinding && fnBinding.ir;
    if (fnIR && fnIR.kind === 'call' && fnIR.op === 'functionof') {
      // Visiting guard would matter if `fnIR.body` referenced its own
      // binding name — but `functionof` bodies only see `%local`
      // params after lift, so a self-reference would have to be
      // through another binding (mutual recursion). FlatPPL is DAG-
      // only at the spec level, so this can't form a cycle by
      // construction. Inline directly.
      head = fnIR;
    } else {
      return null;
    }
  }
  // Head must be a functionof (inline or post-inlined).
  if (!head || head.kind !== 'call' || head.op !== 'functionof') return null;
  const params: string[] = Array.isArray(head.params) ? head.params : [];
  if (params.length === 0) return null;
  const body = head.body;
  if (!body) return null;
  // Match broadcast's outer arity (positional OR kwarg form, not
  // mixed; mixed forms are rare in practice and the dissolver leaves
  // them to the runtime).
  const posArgs = bcArgs.slice(1);
  const kwargs: Record<string, any> = bcIR.kwargs || {};
  const kwNames = Object.keys(kwargs);
  let appliedArgs: any[];
  if (kwNames.length === 0) {
    if (posArgs.length !== params.length) return null;
    appliedArgs = posArgs;
  } else if (posArgs.length === 0) {
    // Kwarg form: head.paramKwargs records the surface kwarg name for
    // each param (in declared order). Map the caller's kwargs to
    // positional via that order.
    const kwOrder: string[] = Array.isArray(head.paramKwargs)
      ? head.paramKwargs : [];
    if (kwOrder.length !== params.length) return null;
    appliedArgs = new Array(params.length);
    for (let i = 0; i < params.length; i++) {
      const name = kwOrder[i];
      if (!(name in kwargs)) return null;
      appliedArgs[i] = kwargs[name];
    }
    // Every supplied kwarg must be consumed; an extra kwarg means the
    // caller passed something the head didn't declare — leave to the
    // runtime to report.
    for (const name of kwNames) {
      if (kwOrder.indexOf(name) === -1) return null;
    }
  } else {
    return null;  // mixed positional + kwarg — leave to runtime
  }
  // Soundness check: dissolving to a direct elementwise call is only
  // safe when all args have the SAME inferred type and the SAME phase.
  // Mixed-shape broadcasts (e.g. `alpha .+ beta .* x_data` where
  // alpha is scalar-parameterized and the RHS is vector-parameterized)
  // resolve at runtime as outer-broadcast loops (atom × intrinsic)
  // that valueOps' elementwise impl doesn't model — see the failing
  // bayesupdate / kernel-broadcast tests during Phase 2 sizing. Leaving
  // those broadcasts in place keeps `_broadcastApply` correct.
  //
  // Also tracks whether ALL outer args are scalar-typed — that
  // unlocks the SCALAR_ONLY ops (mul / exp / log / …) in the body
  // walker.
  //
  // The check is permissive when `bindings` is null (unit-test path
  // for the structural matcher in isolation). Production callers
  // always provide it.
  let argsAreAllScalar = false;
  if (bindings) {
    let firstTP: { type: any; phase: any } | null = null;
    argsAreAllScalar = true;
    for (let i = 0; i < appliedArgs.length; i++) {
      const tp = _argTypeAndPhase(appliedArgs[i], bindings);
      if (!tp) return null;  // non-ref arg or missing annotation
      if (firstTP === null) firstTP = tp;
      else {
        if (firstTP.phase !== tp.phase) return null;
        if (!_typesEqual(firstTP.type, tp.type)) return null;
      }
      if (!tp.type || tp.type.kind !== 'scalar') argsAreAllScalar = false;
    }
  }
  // Build the param-index map for the body walk.
  const paramIndex: Map<string, number> = new Map();
  for (let i = 0; i < params.length; i++) paramIndex.set(params[i], i);
  const substituted = _substituteBody(
    body, paramIndex, appliedArgs, bindings, argsAreAllScalar);
  if (substituted === null) return null;
  // Preserve the broadcast's source location on the outermost call so
  // diagnostics still point at the user-written site.
  if (bcIR.loc && substituted.kind === 'call' && !substituted.loc) {
    substituted.loc = bcIR.loc;
  }
  return substituted;
}

// =====================================================================
// Axis-stack annotation (P3a; engine-concepts §18.11 / §20.10.5 item 4)
// =====================================================================
//
// `propagateAxisStack` walks each binding's IR and annotates measure-
// op IR nodes with `axisStack` metadata: the outer iteration axes
// their variate carries from enclosing axis-introducing constructs
// (iid / kernel-broadcast / aggregate / value broadcast).
//
// The annotation is PURE METADATA — it doesn't change any IR
// structure or runtime semantics. Today no consumer reads it (fusion
// thread (b) will, when kernel-broadcast fusion lands). The pass
// exists to make the axis context EXPLICIT at the IR layer so the
// fusion work can lift the existing per-cell loops into batched
// primitives uniformly.
//
// **Pass scope (minimum-viable P3a).** Annotates the OUTERMOST
// measure-op IR node of each binding when the binding's RHS matches
// one of three axis-introducing shapes:
//
//   1. `iid(M, n)` — one iid axis of size `n`.
//   2. `broadcast(<head>, args…)` — one broadcast axis whose size
//      comes from the first non-head arg's outer shape (the
//      enforced "all args share leading axis" contract; spec §04).
//      The source label is 'kernel_broadcast' when the head resolves
//      to a kernel-typed binding, 'broadcast' otherwise (value head
//      or unresolved).
//   3. `aggregate(f, [axes…], expr)` — one entry per output axis;
//      `name` set to the axis name, `size` left as `'%dynamic'`
//      pending later refinement that resolves axis lengths via the
//      body's indexings.
//
// **Out of scope (future refinements)**:
//   - Nested annotation: if a binding's RHS is `iid(broadcast(K, …), n)`,
//     today we annotate the top-level `iid` with one entry. A future
//     pass can recurse into the inner `broadcast` and produce the
//     full stack `[{iid, n}, {kernel_broadcast, len}]`.
//   - Inlining through kernel bodies: when fusion (b) lands, the
//     pass extends to propagate axisStack INTO kernel `functionof`
//     bodies at materialise time.
//   - Atom axis: deliberately NOT in IR axisStack (engine-internal
//     concept; materialiser-time prepended).

function _isKernelHead(head: any, bindings: any): boolean {
  if (!head) return false;
  if (head.kind !== 'ref' || head.ns !== 'self') return false;
  if (!bindings || !bindings.get) return false;
  const b = bindings.get(head.name);
  if (!b) return false;
  // Either the binding is explicitly kernel-typed, OR its IR
  // constructs a measure (e.g. `kernelof(...)`). For the minimum-
  // viable pass we recognise only the type-driven case; the dissolver
  // already inlines `functionof` bodies, so a function head wouldn't
  // appear in a kernel-broadcast pattern.
  const t = b.inferredType;
  if (!t) return false;
  return t.kind === 'kernel';
}

function _argSizeIdentifier(arg: any, bindings: any): number | string {
  if (!arg) return '%dynamic';
  if (arg.kind === 'lit') {
    if (typeof arg.value === 'number') return arg.value;
    return '%dynamic';
  }
  if (arg.kind === 'ref' && arg.ns === 'self') {
    // Bindings whose inferred type is array[1] of known size give us
    // the leading dim; otherwise carry the binding name as a symbolic
    // size identifier so consumers can resolve at materialise time.
    if (bindings && bindings.get) {
      const b = bindings.get(arg.name);
      const t = b && b.inferredType;
      if (t && t.kind === 'array' && Array.isArray(t.shape) && t.shape.length >= 1) {
        const d0 = t.shape[0];
        if (typeof d0 === 'number') return d0;
      }
    }
    return arg.name;
  }
  return '%dynamic';
}

// Build the axisStack entry for a single measure-op IR node. Returns
// null if the IR doesn't match a recognised axis-introducing shape.
//
// Recurses into nested measure-op IR: `iid(broadcast(K, args), n)`
// produces the full 2-entry stack [{iid, n}, {kernel_broadcast,
// len(args)}] on the outer iid. The recursion is bounded by IR depth;
// pathological inputs are clamped at MAX_NEST_DEPTH and treated as
// unrecognised (return null) to keep the pass linear-time.
const _AXIS_STACK_MAX_DEPTH = 8;
function _axisStackForIR(ir: any, bindings: any, depth: number = 0): any[] | null {
  if (depth > _AXIS_STACK_MAX_DEPTH) return null;
  if (!ir || ir.kind !== 'call' || !ir.op) return null;
  const args: any[] = ir.args || [];

  if (ir.op === 'iid') {
    // iid(M, n) — second positional is the size (literal or ref).
    if (args.length < 2) return null;
    const n = args[1];
    const size = _argSizeIdentifier(n, bindings);
    const entry = { source: 'iid', size };
    // Descend into the inner measure's IR if it's itself an axis-
    // introducing shape, OR if it's a `self` ref to a binding whose
    // IR has axisStack already populated (the outer dissolveBindings
    // walk runs first; populating ALL bindings, so by the time we
    // get here a referenced inner iid's stack is on the binding's
    // IR). Prepend our entry to the inner stack.
    const innerStack = _innerMeasureStack(args[0], bindings, depth);
    return innerStack ? [entry].concat(innerStack) : [entry];
  }

  if (ir.op === 'broadcast') {
    // broadcast(<head>, args…) or broadcast(<head>, name=arg, …).
    // Head is args[0]; remaining positionals or kwargs are the
    // broadcast args. The outer axis size matches the first non-
    // head arg's leading dim.
    if (args.length < 1) return null;
    const head = args[0];
    let firstArg: any = null;
    if (args.length >= 2) firstArg = args[1];
    else if (ir.kwargs) {
      const keys = Object.keys(ir.kwargs);
      if (keys.length > 0) firstArg = ir.kwargs[keys[0]];
    }
    if (!firstArg) return null;
    const size = _argSizeIdentifier(firstArg, bindings);
    const source = _isKernelHead(head, bindings) ? 'kernel_broadcast' : 'broadcast';
    const entry: any = { source, size };
    if (firstArg.kind === 'ref' && firstArg.ns === 'self') entry.name = firstArg.name;
    // For kernel-broadcast the inner measure context is the kernel's
    // body (in a separate binding); recursing into a `self`-ref to a
    // kernel binding would require materialise-time inlining (future
    // fusion (b) work). For value-broadcast there's no inner measure
    // — the head is a function producing values. Either way no
    // within-IR recursion applies; the entry stands alone.
    return [entry];
  }

  if (ir.op === 'aggregate') {
    // aggregate(f, output_axes, body). One axisStack entry per output
    // axis; size is %dynamic pending axis-length resolution.
    if (args.length < 3) return null;
    const axesIR = args[1];
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const axes: any[] = axesIR.args || [];
    const stack: any[] = [];
    for (const ax of axes) {
      if (!ax || ax.kind !== 'axis') return null;
      stack.push({ source: 'aggregate', size: '%dynamic', name: ax.name });
    }
    return stack.length > 0 ? stack : null;
  }

  return null;
}

// Resolve the inner measure's axisStack for the `iid(M, n)` recursion.
// Handles two shapes:
//   1. M is an inline measure-op IR (e.g. `iid(broadcast(K, …), n)`
//      where the outer iid's first arg is itself an axis-introducing
//      call). Recurse via _axisStackForIR.
//   2. M is a `self` ref to another binding that's already been
//      annotated by an earlier propagateAxisStack iteration (the
//      pass runs to a fixed point). Read the binding's IR axisStack
//      if present.
// Returns null when no inner stack applies — caller treats the outer
// entry as standalone.
function _innerMeasureStack(measureArg: any, bindings: any, depth: number): any[] | null {
  if (!measureArg) return null;
  if (measureArg.kind === 'call') {
    return _axisStackForIR(measureArg, bindings, depth + 1);
  }
  if (measureArg.kind === 'ref' && measureArg.ns === 'self' && bindings && bindings.get) {
    const b = bindings.get(measureArg.name);
    if (b && b.ir && Array.isArray(b.ir.axisStack) && b.ir.axisStack.length > 0) {
      return b.ir.axisStack.slice();
    }
  }
  return null;
}

// Compare two axisStack arrays for content equality.
function _axisStacksEqual(a: any[], b: any[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].source !== b[i].source || a[i].size !== b[i].size || a[i].name !== b[i].name) {
      return false;
    }
  }
  return true;
}

// Annotate measure-op IR nodes with axisStack metadata. The pass
// iterates to a fixed point so cross-binding references settle: an
// `iid(M_ref, n)` binding inherits M_ref's axisStack via the inner-
// measure recursion, which requires M_ref's binding to be annotated
// first. Iteration order within a pass is bindings-map order; later
// iterations pick up entries that an earlier iteration produced.
// Annotation attaches as a NEW field on a fresh copy of the IR node
// (no in-place mutation), so other consumers holding references to
// the pre-annotation IR stay unaffected.
function propagateAxisStack(bindings: any): any {
  if (!bindings) return bindings;
  const MAX_ITERS = 5;
  for (let iter = 0; iter < MAX_ITERS; iter++) {
    let changed = false;
    for (const [name, b] of bindings) {
      if (!b || !b.ir) continue;
      const stack = _axisStackForIR(b.ir, bindings);
      if (!stack) continue;
      if (Array.isArray(b.ir.axisStack) && _axisStacksEqual(b.ir.axisStack, stack)) {
        continue;
      }
      const newIR = { ...b.ir, axisStack: stack };
      bindings.set(name, { ...b, ir: newIR });
      changed = true;
    }
    if (!changed) break;
  }
  return bindings;
}

// Walk a bindings map (post-liftInlineSubexpressions output) and
// dissolve each binding's `.ir` field in place. Returns the same map
// for caller convenience. Bindings without a cached IR are left alone.
function dissolveBindings(bindings: any): any {
  if (!bindings) return bindings;
  // Fixed-point iteration: dissolving a leaf anon binding may unlock
  // a parent (e.g. `(A .* B) .+ C` — the outer .+ can only dissolve
  // once `__anon0 = A .* B` has a derivable type). Capped at a small
  // number of iterations; the dissolution rules are monotone so
  // convergence is guaranteed.
  const MAX_ITERS = 5;
  for (let iter = 0; iter < MAX_ITERS; iter++) {
    let changed = false;
    for (const [name, b] of bindings) {
      if (!b || !b.ir) continue;
      const dissolved = dissolveExpr(b.ir, bindings);
      if (dissolved !== b.ir) {
        bindings.set(name, { ...b, ir: dissolved });
        changed = true;
      }
    }
    if (!changed) break;
  }
  // After dissolution settles, annotate measure-op IR nodes with
  // axisStack metadata (P3a). Pure annotation; doesn't affect any
  // runtime semantics.
  propagateAxisStack(bindings);
  return bindings;
}

module.exports = {
  DISSOLVE_SAFE_OPS,
  DISSOLVE_AT_ANY_RANK_OPS,
  DISSOLVE_SCALAR_ONLY_OPS,
  dissolveExpr,
  dissolveBindings,
  propagateAxisStack,
  _tryDissolveSingleOp,
  _tryDissolveAggregate,
  _axisStackForIR,
};
