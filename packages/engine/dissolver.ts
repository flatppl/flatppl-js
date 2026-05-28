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
const DISSOLVE_SAFE_OPS: Set<string> = new Set([
  'add', 'sub', 'neg', 'pos',
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
// Returns null if the arg is not a binding ref or if the binding
// lookup yields no usable annotation — the conservative response is
// "don't dissolve."
function _argTypeAndPhase(arg: any, bindings: any): { type: any; phase: any } | null {
  if (!arg) return null;
  if (arg.kind !== 'ref') return null;
  if (arg.ns !== 'self') return null;
  const b = bindings && bindings.get && bindings.get(arg.name);
  if (!b) return null;
  const t = b.inferredType;
  const ph = b.phase;
  if (!t || t.kind === 'deferred' || t.kind === 'failed') return null;
  return { type: t, phase: ph };
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
  // Pattern-match `broadcast(...)` at THIS level after the children
  // are dissolved. We don't try to dissolve aggregate / broadcasted
  // yet — those are Phase 5 / Phase 3 respectively.
  if (walked.op === 'broadcast') {
    const dissolved = _tryDissolveSingleOp(walked, bindings);
    if (dissolved) return dissolved;
  }
  return walked;
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
      if (!expr.op || !DISSOLVE_SAFE_OPS.has(expr.op)) return null;
      if (expr.kwargs && Object.keys(expr.kwargs).length > 0) return null;
      if (Array.isArray(expr.fields) && expr.fields.length > 0) return null;
      if (expr.body) return null;  // nested functionof — bail
      const inArgs: any[] = expr.args || [];
      const outArgs: any[] = new Array(inArgs.length);
      for (let i = 0; i < inArgs.length; i++) {
        const sub = _substituteBody(inArgs[i], paramIndex, bcArgs, bindings);
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
  // The check is permissive when `bindings` is null (unit-test path
  // for the structural matcher in isolation). Production callers
  // always provide it.
  if (bindings) {
    let firstTP: { type: any; phase: any } | null = null;
    for (let i = 0; i < appliedArgs.length; i++) {
      const tp = _argTypeAndPhase(appliedArgs[i], bindings);
      if (!tp) return null;  // non-ref arg or missing annotation
      if (firstTP === null) firstTP = tp;
      else {
        if (firstTP.phase !== tp.phase) return null;
        if (!_typesEqual(firstTP.type, tp.type)) return null;
      }
    }
  }
  // Build the param-index map for the body walk.
  const paramIndex: Map<string, number> = new Map();
  for (let i = 0; i < params.length; i++) paramIndex.set(params[i], i);
  const substituted = _substituteBody(body, paramIndex, appliedArgs, bindings);
  if (substituted === null) return null;
  // Preserve the broadcast's source location on the outermost call so
  // diagnostics still point at the user-written site.
  if (bcIR.loc && substituted.kind === 'call' && !substituted.loc) {
    substituted.loc = bcIR.loc;
  }
  return substituted;
}

// Walk a bindings map (post-liftInlineSubexpressions output) and
// dissolve each binding's `.ir` field in place. Returns the same map
// for caller convenience. Bindings without a cached IR are left alone.
function dissolveBindings(bindings: any): any {
  if (!bindings) return bindings;
  for (const [name, b] of bindings) {
    if (!b || !b.ir) continue;
    const dissolved = dissolveExpr(b.ir, bindings);
    if (dissolved !== b.ir) {
      bindings.set(name, { ...b, ir: dissolved });
    }
  }
  return bindings;
}

module.exports = {
  DISSOLVE_SAFE_OPS,
  dissolveExpr,
  dissolveBindings,
  _tryDissolveSingleOp,
};
