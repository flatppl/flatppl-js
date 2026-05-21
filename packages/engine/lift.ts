'use strict';

// lift.js — inline-subexpression lifting for the orchestrator.
// =====================================================================
//
// liftInlineSubexpressions(bindings) is the analyzer-output rewrite
// pass that hoists anonymous inline measure / kernel / value
// subexpressions into named synthetic bindings, so the downstream
// derivation classifier only ever sees self-refs to named bindings.
// canonicalizeImplicitBoundaries / bfsImplicitElementofLeavesAst
// normalise implicit `elementof` parameter boundaries before lifting;
// argSignature / opUsesValueKwargs / inferSyntheticType drive the
// per-op arg-position typing the lift visitor needs; isEvaluable is
// the static deterministic-evaluability predicate the classifier and
// lift share.
//
// A leaf w.r.t. the split: depends on lower (lowerExpr), signatures
// (signatureOf), and ir-shared (parseSetIR, EVALUABLE_OPS,
// SAMPLEABLE_DISTRIBUTIONS) — never on derivations or the
// orchestrator core, so there is no back-edge.

const { lowerExpr } = require('./lower.ts');
const { signatureOf } = require('./signatures.ts');
const {
  parseSetIR,
  EVALUABLE_OPS,
  SAMPLEABLE_DISTRIBUTIONS,
} = require('./ir-shared.ts');

// =====================================================================
// Inline-subexpression lifting
//
// FlatPPL operators have positional argument-type expectations: draw()
// expects a measure, weighted() expects (value, measure), normalize()
// expects a measure, etc. A user can supply either a bare reference to
// a binding or an inline expression in any of those positions, and the
// language semantics treat the two as equivalent — caching aside, an
// intermediate binding is just a name for a sub-expression.
//
// Rather than have every classifier branch handle inline forms one by
// one (and re-bug each time a new op is added), we run a single AST
// rewrite that lifts every non-trivial inline subexpression in a
// measure-arg position to a synthetic anonymous binding `__anon_N`,
// replacing the in-place AST with a reference. After lifting, every
// measure-typed argument is a bare Identifier; the existing classifier
// handles all forms uniformly. Inline `draw(<...>)` in a value slot
// gets the same treatment so the surviving value expression is
// evaluable end-to-end.
//
// Mutability: every binding's RHS is deep-cloned before being walked,
// so the original bindings map (and its AST nodes) are untouched.
// Calling buildDerivations twice on the same bindings is therefore
// idempotent — each call sees pristine input.
//
// Synthetic bindings carry `synthetic: true` so downstream layers
// (DAG render, plot pane) can choose to display them differently or
// hide them entirely.
// =====================================================================

/**
 * Argument-type signature for a known op, given its positional arity.
 *
 * Returns an array of expected types (one per arg position) or null if
 * the op isn't recognised. Types are:
 *   'measure'           — must be a measure-typed expression
 *   'value'             — must be a value-typed expression
 *   'value-or-measure'  — lawof's argument: either is acceptable
 *
 * Distribution constructors are positional-empty by convention (their
 * params come via kwargs, all of type 'value'); their kwargs are
 * handled separately in the visitor.
 */
function argSignature(op: string, numArgs: number): string[] | null {
  if (op === 'draw')                              return ['measure'];
  if (op === 'weighted' || op === 'logweighted')  return ['value', 'measure'];
  if (op === 'normalize')                         return ['measure'];
  if (op === 'superpose')                         return Array(numArgs).fill('measure');
  // NOTE: `ifelse` is intentionally NOT given a measure-arg
  // signature. ifelse is dual (value- OR measure-valued); forcing
  // its branch slots to 'measure' would mis-hoist value-valued
  // ifelse subexpressions (`ifelse(c, x+1, x-1)`) into bogus anon
  // measures. Measure-valued ifelse is recognised at classify time
  // (classifyIfelse) on NAMED measure-binding branches; inline-
  // measure branches are a documented deferral.
  if (op === 'lawof')                             return ['value-or-measure'];
  if (op === 'iid') {
    // iid(<measure>, n, m, ...): first arg measure-typed, rest values.
    const sig = ['measure'];
    for (let i = 1; i < numArgs; i++) sig.push('value');
    return sig;
  }
  if (op === 'truncate') {
    // truncate(<measure>, <set>): first arg measure-typed; second is a
    // set expression (named set / interval call) that we don't lift
    // — the classifier reads it raw via parseSetIR.
    return ['measure', 'value'];
  }
  if (op === 'pushfwd') {
    // pushfwd(<function>, <measure>): function position is value-shaped
    // for lifting (so inline fn(...) gets lifted to anon bindings the
    // way liftValue handles draw / logdensityof; functions in arg
    // position aren't lifted further than that). Measure position is
    // measure-typed.
    return ['value', 'measure'];
  }
  if (op === 'joint') {
    // Positional joint(M1, M2, ...): all args are measure-typed. The
    // kwarg form (joint(name1 = M1, ...)) is handled separately via
    // the isRecordLike branch — its kwarg values get liftMeasure'd
    // there. Setting a 'measure'-per-position signature here covers
    // the positional surface so that inline measure expressions
    // (e.g. `joint(Normal(0,1), Exp(1))`) get lifted to named anon
    // bindings before the classifier reads them.
    return Array(numArgs).fill('measure');
  }
  if (op === 'totalmass' || op === 'logdensityof') {
    // totalmass(<measure>) / logdensityof(<measure>, <obs>): first arg
    // measure-typed so a bare inline distribution gets lifted to a
    // named binding. logdensityof's obs is value-typed (already lifted
    // when it's a draw / logdensityof itself by liftValue's rules).
    const sig = ['measure'];
    for (let i = 1; i < numArgs; i++) sig.push('value');
    return sig;
  }
  if (op === 'jointchain' || op === 'kchain') {
    // jointchain(M, K1, K2, ...) / kchain(M, K1, K2, ...): every
    // positional arg is measure-typed (the first a base measure or
    // closed kernel; the rest non-nullary kernels). Lifting them as
    // measures hoists inline measure/kernel args to anon bindings
    // so classifyJointchain sees uniform refs (first-class kind;
    // the legacy inlineChainOps consumer is gone).
    return Array(numArgs).fill('measure');
  }
  if (SAMPLEABLE_DISTRIBUTIONS.has(op))           return Array(numArgs).fill('value');
  if (EVALUABLE_OPS.has(op))                      return Array(numArgs).fill('value');
  return null;
}

/**
 * True when an op accepts only value-typed kwargs (so the visitor
 * recurses into them without treating them as measure positions).
 * Currently every op we know about that takes kwargs uses them for
 * value parameters — distribution constructors and arithmetic.
 */
function opUsesValueKwargs(op: string) {
  return SAMPLEABLE_DISTRIBUTIONS.has(op) || EVALUABLE_OPS.has(op);
}

/**
 * Map an AST RHS to the binding `type` the analyzer would assign.
 * Mirrors classifyStatement in analyzer.js for the subset that can
 * appear after lifting; we don't redirect to the analyzer because
 * pulling it in here would create a cycle.
 */
function inferSyntheticType(astNode: any) {
  if (!astNode) return 'call';
  if (astNode.type === 'CallExpr' && astNode.callee && astNode.callee.type === 'Identifier') {
    switch (astNode.callee.name) {
      case 'draw':       return 'draw';
      case 'lawof':      return 'lawof';
      case 'functionof': return 'functionof';
      case 'kernelof':   return 'kernelof';
      case 'fn':         return 'fn';
    }
    return 'call';
  }
  switch (astNode.type) {
    case 'NumberLiteral':
    case 'StringLiteral':
    case 'BoolLiteral':
    case 'ArrayLiteral':
    case 'TupleLiteral': return 'literal';
  }
  return 'call';
}

/**
 * Walk every binding's RHS AST, lifting non-trivial subexpressions
 * in measure-arg positions (and inline `draw(...)` in value-arg
 * positions) to fresh synthetic bindings. Returns a new bindings Map
 * containing the originals with their RHS rewritten in-place, plus
 * one entry per synthesized anonymous binding.
 *
 * The pass is idempotent: rerunning it on the output is a no-op
 * because every measure-arg position is already a bare Identifier
 * after the first pass.
 */
// Sentinel prefix for substitution-map keys that target Placeholder
// nodes (vs. Identifier nodes). When a callable's boundary kwarg is
// declared as `par = _par_`, the body uses a Placeholder named 'par';
// substituteIdents reads this prefix to distinguish placeholder
// substitutions from identifier substitutions of the same name.
const PLACEHOLDER_SUB_PREFIX = '@placeholder:';

/**
 * Canonicalise no-kwargs functionof / kernelof to explicit-kwargs
 * form before any downstream consumer touches the AST or its
 * lowered IR.
 *
 * Per spec §04 sec:functionof: a single-argument functionof traces
 * its body's ancestor subgraph back to all parametric-phase leaves
 * (elementof bindings); those leaves become the inputs of the
 * reified callable. This pass realises that as an AST rewrite —
 * `f = functionof(body)` becomes `f = functionof(body, leaf=leaf, ...)`
 * for every elementof leaf found by transitive walk.
 *
 * One canonical place keeps three consumers in lockstep:
 *   1. lower._lowerReification    → reads the AST kwargs into ir.params.
 *   2. orchestrator.signatureOf   → reads ir.params verbatim.
 *   3. orchestrator.inlineOnce    → reads fnAst's KeywordArgs to build
 *                                   the substitution map.
 *
 * Before this pass existed, each of (2) and (3) had to re-derive the
 * implicit boundaries independently — see git history for the bugs
 * that came from the views drifting.
 *
 * Untouched cases:
 *   - functionof / kernelof with at least one boundary kwarg: the
 *     user has declared the signature; we don't bolt on extras.
 *   - fn(...): uses placeholder holes, not elementof refs, so the
 *     spec rule doesn't apply.
 *   - bindings without a parseable CallExpr AST.
 *   - bodies with no reachable elementof leaves (the function is
 *     truly parameterless; signatureOf still produces inputs:[]).
 *
 * Returns a fresh Map. The original bindings are not mutated; for
 * affected bindings, we deep-clone the node so the original AST stays
 * pristine for editor diagnostics and round-trip rendering.
 */
function canonicalizeImplicitBoundaries(bindings: any) {
  if (!bindings || bindings.size === 0) return bindings;
  const out: Map<string, any> = new Map(bindings);

  for (const [name, b] of bindings) {
    if (b.type !== 'functionof' && b.type !== 'kernelof') continue;
    if (!b.node || !b.node.value) continue;
    const callExpr = b.node.value;
    if (callExpr.type !== 'CallExpr') continue;
    const args = callExpr.args || [];
    if (args.length === 0) continue;
    // Already has explicit boundary kwargs → don't auto-promote.
    if (args.slice(1).some((a: any) => a && a.type === 'KeywordArg')) continue;

    const bodyAst = args[0];
    if (!bodyAst) continue;
    const leaves = bfsImplicitElementofLeavesAst(bodyAst, bindings);
    if (leaves.length === 0) continue;

    // Synthesize KeywordArgs for each parametric leaf. Both surface
    // name and value Identifier are the leaf's own binding name —
    // body refs to that name resolve normally through self-scope,
    // and the boundary kwarg's surface name matches the spec's
    // "leaf nodes become the inputs of the reified callable" wording.
    const newKwargs = leaves.map((leafName: any) => ({
      type: 'KeywordArg',
      name: leafName,
      value: { type: 'Identifier', name: leafName, loc: bodyAst.loc || null },
      loc: bodyAst.loc || null,
    }));

    const newCallExpr = { ...callExpr, args: [args[0], ...newKwargs] };
    out.set(name, {
      ...b,
      // Mark the rewrite so downstream tooling (diagnostics, round-
      // trip) can distinguish it from a user-authored kwarg list.
      implicitBoundaries: leaves.slice(),
      node: { ...b.node, value: newCallExpr },
    });
  }
  return out;
}

/**
 * BFS the AST `bodyAst` through binding refs, collecting parametric-
 * phase elementof leaves in visit order. Same traversal shape as
 * signatureOf used to do internally; centralised here so the
 * canonicalize pass is the single source of truth.
 */
function bfsImplicitElementofLeavesAst(bodyAst: any, bindings: any) {
  const seen = new Set();
  const leaves: any[] = [];
  const queue: any[] = [];
  collectIds(bodyAst, queue);
  while (queue.length > 0) {
    const name = queue.shift();
    if (seen.has(name)) continue;
    seen.add(name);
    const b = bindings.get(name);
    if (!b) continue;
    if (b.type === 'input' && b.phase === 'parameterized') {
      leaves.push(name);
      continue;
    }
    if (b.phase === 'fixed') continue;  // closed over per spec
    if (b.node && b.node.value) collectIds(b.node.value, queue);
  }
  return leaves;
  function collectIds(node: any, into: any[]) {
    if (node == null || typeof node !== 'object') return;
    if (Array.isArray(node)) { for (const c of node) collectIds(c, into); return; }
    if (node.type === 'Identifier') into.push(node.name);
    for (const k in node) collectIds(node[k], into);
  }
}

function liftInlineSubexpressions(bindings: any) {
  // Canonicalise implicit-boundary functionof / kernelof first.
  // After this, the lifted IR carries explicit ir.params, so every
  // downstream consumer (signatureOf, inlineOnce, _lowerReification)
  // sees a uniform shape regardless of whether the user wrote
  // explicit kwargs.
  bindings = canonicalizeImplicitBoundaries(bindings);
  const out: Map<string, any> = new Map(bindings);
  let counter = 0;
  function freshName() {
    let n: string;
    do { n = '__anon' + (counter++); } while (out.has(n));
    return n;
  }
  function makeIdent(name: string, loc: any) {
    return { type: 'Identifier', name, loc: loc || null };
  }
  function makeSyntheticBinding(name: string, ast: any) {
    return {
      name,
      names: [name],
      line: ast.loc && ast.loc.start ? ast.loc.start.line : -1,
      rhs: '',
      type: inferSyntheticType(ast),
      deps: [], callDeps: [],
      node: { value: ast, names: [makeIdent(name, ast.loc)], loc: ast.loc, type: 'AssignStatement' },
      nameLoc: ast.loc,
      synthetic: true,
    };
  }

  // Deep-clone each binding's RHS before walking so the lift's
  // mutations stay local; the caller's bindings map is untouched.
  // We lift TWO ASTs per binding when present:
  //   - node.value      (the user-written RHS; preserved by the
  //                      analyzer for round-trip and source-located
  //                      diagnostics)
  //   - effectiveValue  (the analyzer's rewriter-resolved canonical
  //                      RHS; for a disintegration delegate it's
  //                      the delegate target's RHS, for a synthesized
  //                      disintegration it's the synthesized rewrite,
  //                      otherwise undefined)
  // classifyDerivation reads effectiveValue when present, so it
  // needs the lifted shape too.
  for (const [name, binding] of bindings) {
    if (!binding.node || !binding.node.value) continue;
    let cloned = cloneAst(binding.node.value);
    cloned = inlineUserCall(cloned);
    visit(cloned);
    // Re-run user-call inlining after visit(): visit() lifts each
    // positional arg to a named anonymous binding, exposing further
    // inlinable user calls. inlineUserCall is idempotent when no
    // further rewrite applies, so this is a no-op otherwise.
    cloned = inlineUserCall(cloned);
    let effLifted = binding.effectiveValue;
    if (effLifted) {
      effLifted = cloneAst(effLifted);
      effLifted = inlineUserCall(effLifted);
      visit(effLifted);
      effLifted = inlineUserCall(effLifted);
    }
    // Refresh binding.type from the rewritten AST head. Rewrites that
    // change the head (e.g. pushfwd → lawof, jointchain → joint) need
    // the binding type to follow, so the classifier's special-type
    // branches (binding.type === 'lawof' / 'draw' / …) still fire.
    // Only narrow from the generic 'call' type — bindings the analyzer
    // already classified as a specific type stay as-is.
    let newType = binding.type;
    if (binding.type === 'call') {
      const rewrittenType = inferSyntheticType(cloned);
      if (rewrittenType !== 'call') newType = rewrittenType;
    }
    out.set(name, {
      ...binding,
      type: newType,
      node: { ...binding.node, value: cloned },
      effectiveValue: effLifted,
    });
  }

  // Post-pass: cache the lowered IR alongside the AST on every binding
  // (including the synthesized anonymous bindings inserted during the
  // user-call / chain-op inlining loop). Classifiers and other
  // downstream passes read this rather than re-lowering on every
  // call. The IR comes from the *effective* AST when the analyzer's
  // rewriter set one (disintegration delegates and synthesized plans),
  // otherwise from the literal RHS — same precedence the classifier
  // used to compute on demand.
  for (const [name, b] of out) {
    if (!b || !b.node || !b.node.value) continue;
    let ir = null;
    try { ir = lowerExpr(b.effectiveValue || b.node.value); } catch (_) { ir = null; }
    out.set(name, { ...b, ir });
  }
  return out;

  function cloneAst(node: any): any {
    if (node == null || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(cloneAst);
    const copy: Record<string, any> = {};
    for (const k in node) copy[k] = cloneAst(node[k]);
    return copy;
  }

  // -- Visitor ----------------------------------------------------------
  // visit() walks INTO a node, lifting children if needed. The lift
  // helpers (liftMeasure / liftValue / liftMeasureOrValue) are mutually
  // recursive with visit() through their internal `visit(astArg)` call
  // — that's how deep nesting flattens out one anon at a time.

  function visit(astNode: any) {
    if (!astNode) return;
    if (astNode.type === 'BinaryExpr') {
      astNode.left  = liftValue(astNode.left);
      astNode.right = liftValue(astNode.right);
      return;
    }
    if (astNode.type === 'UnaryExpr') {
      astNode.operand = liftValue(astNode.operand);
      return;
    }
    if (astNode.type === 'ArrayLiteral' || astNode.type === 'TupleLiteral') {
      // Per spec §03 arrays/tuples hold values, so each element is
      // value-typed; liftValue knows to lift inline `draw(...)` calls
      // to anonymous variate bindings while leaving literals and
      // arithmetic in place. Lifting here means the classifier can
      // recognise an array of variate refs and emit a tuple-shaped
      // measure for plotting.
      const elems = astNode.elements || [];
      for (let i = 0; i < elems.length; i++) {
        elems[i] = liftValue(elems[i]);
      }
      return;
    }
    if (astNode.type !== 'CallExpr') return;
    if (!astNode.callee || astNode.callee.type !== 'Identifier') return;

    const op = astNode.callee.name;
    const numArgs = astNode.args ? astNode.args.length : 0;
    const sig = argSignature(op, numArgs);

    // record/joint/jointchain fields: each kwarg value gets lifted
    // to a synthetic binding so the classifier can read them as bare
    // refs. record/jointchain fields are values; joint fields are
    // measures. All pass through liftMeasure (= "lift any non-
    // trivial expression to an anon binding") — the value/measure
    // distinction lives at type inference, not at lifting.
    const isRecordLike = op === 'record' || op === 'joint'
                       || op === 'jointchain';

    if (astNode.args) {
      for (let i = 0; i < astNode.args.length; i++) {
        const a = astNode.args[i];
        if (a && a.type === 'KeywordArg') {
          if (isRecordLike)             a.value = liftMeasure(a.value);
          else if (opUsesValueKwargs(op)) a.value = liftValue(a.value);
          continue;
        }
        const expected = sig ? sig[i] : null;
        if      (expected === 'measure')          astNode.args[i] = liftMeasure(a);
        else if (expected === 'value-or-measure') astNode.args[i] = liftMeasureOrValue(a);
        else                                      astNode.args[i] = liftValue(a);
      }
    }
    if (astNode.kwargs && opUsesValueKwargs(op)) {
      for (const k in astNode.kwargs) astNode.kwargs[k] = liftValue(astNode.kwargs[k]);
    }
  }

  // True iff `astNode` (or a descendant) is a Hole (`_`) or Placeholder
  // (`_name_`). Lifting an expression containing such a marker into a
  // module-level anon binding would be invalid: holes / placeholders
  // are local to the enclosing fn / functionof / kernelof scope and
  // can't be referenced from outside it. The lifter therefore bails
  // on any expression carrying one. Common case: `fn(record(a = _,
  // b = 2 * _))` — without this guard, the record's kwarg lifting
  // would pull each hole-containing kwarg into a separate anon
  // binding and clear the function's parameter list.
  function containsHoleOrPlaceholder(astNode: any): boolean {
    if (!astNode || typeof astNode !== 'object') return false;
    if (astNode.type === 'Hole' || astNode.type === 'Placeholder') return true;
    for (const k of Object.keys(astNode)) {
      const c = astNode[k];
      if (Array.isArray(c)) {
        for (const x of c) if (containsHoleOrPlaceholder(x)) return true;
      } else if (c && typeof c === 'object' && k !== 'loc') {
        if (containsHoleOrPlaceholder(c)) return true;
      }
    }
    return false;
  }

  function liftMeasure(astArg: any) {
    if (!astArg) return astArg;
    astArg = inlineUserCall(astArg);
    visit(astArg);
    if (astArg.type === 'Identifier') return astArg;
    if (containsHoleOrPlaceholder(astArg)) return astArg;
    const name = freshName();
    out.set(name, makeSyntheticBinding(name, astArg));
    return makeIdent(name, astArg.loc);
  }

  function liftValue(astArg: any) {
    if (!astArg) return astArg;
    astArg = inlineUserCall(astArg);
    visit(astArg);
    // Two value-position calls produce non-evaluable JS results so
    // they must be lifted to their own anon bindings:
    //   - draw(...)        — needs a fresh sample step
    //   - logdensityof(M,x)— needs traceeval.walk over M's expanded IR
    // Everything else (literals, identifiers, arithmetic) stays in
    // place for the IR evaluator.
    if (astArg.type === 'CallExpr' && astArg.callee
        && astArg.callee.type === 'Identifier'
        && (astArg.callee.name === 'draw'
            || astArg.callee.name === 'logdensityof')) {
      if (containsHoleOrPlaceholder(astArg)) return astArg;
      const name = freshName();
      out.set(name, makeSyntheticBinding(name, astArg));
      return makeIdent(name, astArg.loc);
    }
    return astArg;
  }

  function liftMeasureOrValue(astArg: any) {
    // lawof's argument can be either; lift any non-trivial expression
    // for uniformity (visit recurses into it first to handle nested
    // measure-arg positions).
    if (!astArg) return astArg;
    astArg = inlineUserCall(astArg);
    visit(astArg);
    if (astArg.type === 'Identifier') return astArg;
    if (containsHoleOrPlaceholder(astArg)) return astArg;
    const name = freshName();
    out.set(name, makeSyntheticBinding(name, astArg));
    return makeIdent(name, astArg.loc);
  }

  // -- User-call inlining ----------------------------------------------
  //
  // When a binding's RHS or any sub-expression is a call to a
  // user-defined function/kernel (`a = f_a(par = beta1)`), we inline
  // the function's body with parameter refs substituted by the call's
  // arguments. The result replaces the original CallExpr node and
  // gets re-walked by the lift pass — so nested user calls,
  // measure-algebra inside the body, etc. all flatten out uniformly.
  //
  // Per spec §sec:functionof: `functionof(f(a, b), a=a, b=b) ≡ f`.
  // Our implementation realises this by AST-level beta reduction,
  // bottoming out on the function's body. Bodies are deep-cloned
  // per call site so different invocations don't share mutated trees.

  function inlineUserCall(astArg: any) {
    // Iterate to a fixed point — inlined body might itself be (or
    // contain at its root) another user call (chained: f then g),
    // a jointchain that rewrites to a joint of further user calls,
    // a relabel that surfaces a record(...), an fchain that unrolls
    // to a tower of nested user calls, or a densityof that lowers
    // to exp(logdensityof(...)).
    let prev = null;
    while (astArg !== prev) {
      prev = astArg;
      astArg = inlineOnce(astArg);
      // jointchain/kchain are now a first-class derivation kind
      // (classifyJointchain + matJointchain + expandMeasureIR); the
      // legacy inlineChainOps AST rewrite has been retired.
      astArg = inlineRelabel(astArg);
      astArg = inlineFchain(astArg);
      astArg = inlineDensityof(astArg);
      // pushfwd is now a first-class measure-op (classifyPushfwd +
      // matPushfwd in materialiser + density.walkPushfwd). The legacy
      // AST-rewrite to lawof(f(draw(M))) is no longer needed; sampling
      // and density both consult the proper pushfwd derivation
      // structure. Only the f-position lift remains — inline
      // fn / functionof shapes get hoisted to anon bindings so the
      // classifier sees a clean self-ref.
      astArg = inlinePushfwdLift(astArg);
      astArg = inlineBijectionLift(astArg);
      astArg = inlineFilterLift(astArg);
      astArg = inlineBroadcasted(astArg);
    }
    return astArg;
  }

  /**
   * Rewrite `broadcasted(f)(args...)` to `broadcast(fn(f(_, _, ...)), args...)`
   * per spec §04: broadcasted(f)(args) ≡ broadcast(f, args). We
   * synthesize a `fn(...)` wrapping a call to f with one hole per
   * positional arg, so broadcast's value evaluator (which expects
   * its first arg to be a function-shaped IR) can apply f per
   * element — even when f is a built-in operator like `add` that
   * wouldn't otherwise have a functionof body to walk.
   *
   * Handles two surface shapes:
   *   * Direct: `broadcasted(f)(x, y)` — callee is a `broadcasted` call.
   *   * Via binding: `bc = broadcasted(f); bc(x, y)` — callee is an
   *     Identifier whose binding's RHS is a `broadcasted(...)` call.
   *
   * Mirrors inlineFchain in shape.
   */
  function inlineBroadcasted(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr' || !astArg.callee) return astArg;
    let bcCall: any = null;
    if (astArg.callee.type === 'CallExpr'
        && astArg.callee.callee
        && astArg.callee.callee.type === 'Identifier'
        && astArg.callee.callee.name === 'broadcasted') {
      bcCall = astArg.callee;
    } else if (astArg.callee.type === 'Identifier') {
      const target = out.get(astArg.callee.name);
      const targetAst = target && (target.effectiveValue || (target.node && target.node.value));
      if (targetAst && targetAst.type === 'CallExpr' && targetAst.callee
          && targetAst.callee.type === 'Identifier'
          && targetAst.callee.name === 'broadcasted') {
        bcCall = targetAst;
      }
    }
    if (!bcCall) return astArg;
    const fns = (bcCall.args || []).filter((a: any) => a && a.type !== 'KeywordArg');
    if (fns.length !== 1) return astArg;
    const fArg = fns[0];

    // Count user-supplied positional args; build that many holes inside
    // the synthesized fn. Kwargs aren't supported through broadcasted
    // today (the user would write them through plain broadcast).
    const callerArgs = astArg.args || [];
    const posArgs = callerArgs.filter((a: any) => a && a.type !== 'KeywordArg');
    if (posArgs.length === 0) return astArg;
    const holes = new Array(posArgs.length);
    for (let i = 0; i < posArgs.length; i++) {
      holes[i] = { type: 'Hole', loc: astArg.loc };
    }
    const fnExpr = {
      type: 'CallExpr',
      callee: makeIdent('fn', astArg.loc),
      args: [{
        type: 'CallExpr',
        callee: cloneAst(fArg),
        args: holes,
        loc: astArg.loc,
      }],
      loc: astArg.loc,
    };
    return {
      type: 'CallExpr',
      callee: makeIdent('broadcast', astArg.loc),
      args: [fnExpr].concat(posArgs.map(cloneAst)),
      loc: astArg.loc,
    };
  }

  /**
   * Lift the predicate arg of `filter(pred, data)` to a named binding
   * if it's an inline function expression. Per-element application
   * happens at evaluation time via the env's __resolveFnBody hook
   * (orchestrator.pre-eval attaches it); the AST keeps the surface
   * `filter(pred_ref, data)` shape so analyzer / typer see the
   * normal form.
   */
  /**
   * Lift the function arg of `pushfwd(f, M)` to a named binding if
   * it's an inline function expression. Mirrors inlineFilterLift's
   * shape: classifyPushfwd expects a self-ref to a fn / functionof /
   * kernelof / bijection binding; inline `fn(...)` / `functionof(...)`
   * shapes get hoisted to anon bindings here.
   */
  function inlinePushfwdLift(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'pushfwd') return astArg;
    if (!astArg.args || astArg.args.length !== 2) return astArg;
    let fArg = astArg.args[0];
    if (fArg.type === 'KeywordArg') return astArg;
    if (fArg.type === 'Identifier') return astArg;
    visit(fArg);
    if (fArg.type === 'Identifier') {
      astArg.args[0] = fArg;
      return astArg;
    }
    const n = freshName();
    out.set(n, makeSyntheticBinding(n, fArg));
    astArg.args[0] = makeIdent(n, astArg.loc);
    return astArg;
  }

  /**
   * Lift the three function args of `bijection(f, f_inv, logvolume)`
   * to named bindings. f and f_inv must be functions; logvolume may
   * be a function OR a literal scalar (`0` for volume-preserving
   * maps per spec §06 — we accept any non-CallExpr in the third slot
   * and treat it as a constant). Inline fn / functionof shapes get
   * hoisted to anon bindings; bare identifiers stay as-is.
   */
  function inlineBijectionLift(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'bijection') return astArg;
    if (!astArg.args || astArg.args.length !== 3) return astArg;
    for (let i = 0; i < 3; i++) {
      let a = astArg.args[i];
      if (a.type === 'KeywordArg') return astArg;
      if (a.type === 'Identifier') continue;
      // Scalar literal for logvolume slot is fine to keep inline.
      if (i === 2 && (a.type === 'NumberLiteral' || a.type === 'BoolLiteral')) continue;
      visit(a);
      if (a.type === 'Identifier') { astArg.args[i] = a; continue; }
      const n = freshName();
      out.set(n, makeSyntheticBinding(n, a));
      astArg.args[i] = makeIdent(n, astArg.loc);
    }
    return astArg;
  }

  function inlineFilterLift(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'filter') return astArg;
    if (!astArg.args || astArg.args.length !== 2) return astArg;
    let fArg = astArg.args[0];
    if (fArg.type === 'KeywordArg') return astArg;
    if (fArg.type === 'Identifier') return astArg;
    // Lift inline fn(...) to an anon binding.
    visit(fArg);
    if (fArg.type === 'Identifier') {
      astArg.args[0] = fArg;
      return astArg;
    }
    const n = freshName();
    out.set(n, makeSyntheticBinding(n, fArg));
    astArg.args[0] = makeIdent(n, astArg.loc);
    return astArg;
  }

  /**
   * Lower `densityof(M, x)` to `exp(logdensityof(M, x))` per spec
   * §sec:posterior — densityof is sugar over logdensityof, and
   * keeping logdensityof first-class avoids a second density-eval
   * primitive in the worker. The exp(...) wraps the logdensityof
   * call as a normal evaluate node, so once logdensityof has a
   * derivation, densityof inherits the cascade for free.
   */
  function inlineDensityof(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'densityof') return astArg;
    const loc = astArg.loc;
    const inner = {
      type: 'CallExpr',
      callee: { type: 'Identifier', name: 'logdensityof', loc },
      args: (astArg.args || []).map(cloneAst),
      loc,
    };
    return {
      type: 'CallExpr',
      callee: { type: 'Identifier', name: 'exp', loc },
      args: [inner],
      loc,
    };
  }

  /**
   * Rewrite an applied fchain — `fchain(f1, …, fN)(args)` or
   * `pipeline(args)` where `pipeline = fchain(…)` — to the equivalent
   * tower `fN(… f2(f1(args)))`, per spec §sec:design line 526-532.
   *
   * We only resolve fchain when its components are Identifier refs to
   * function bindings; inline `fn(…)` / `functionof(…)` components
   * would produce a CallExpr-on-CallExpr application that inlineOnce
   * doesn't handle. Those bindings are lifted to anons by the visit
   * pass before fchain is reached, so this is the common path.
   *
   * Returns the input unchanged when the call isn't an applied fchain
   * (so the binding falls through, and a real fchain that isn't
   * applied just classifies as unsupported — fchain bindings are
   * function values, not measures).
   */
  function inlineFchain(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee) return astArg;
    let fchainCall: any = null;
    if (astArg.callee.type === 'CallExpr'
        && astArg.callee.callee
        && astArg.callee.callee.type === 'Identifier'
        && astArg.callee.callee.name === 'fchain') {
      fchainCall = astArg.callee;
    } else if (astArg.callee.type === 'Identifier') {
      const target = out.get(astArg.callee.name);
      const targetAst = target && (target.effectiveValue || (target.node && target.node.value));
      if (targetAst && targetAst.type === 'CallExpr' && targetAst.callee
          && targetAst.callee.type === 'Identifier'
          && targetAst.callee.name === 'fchain') {
        fchainCall = targetAst;
      }
    }
    if (!fchainCall) return astArg;
    const fns = (fchainCall.args || []).filter((a: any) => a && a.type !== 'KeywordArg');
    if (fns.length === 0) return astArg;
    // Build f1(args), then f2(f1(args)), …, fN(…). Each fn[i] becomes
    // a callee — must be an Identifier so inlineOnce can substitute.
    for (const f of fns) {
      if (!f || f.type !== 'Identifier') return astArg;
    }
    const callerArgs = (astArg.args || []).map(cloneAst);
    let result: any = {
      type: 'CallExpr',
      callee: cloneAst(fns[0]),
      args: callerArgs,
      loc: astArg.loc,
    };
    for (let i = 1; i < fns.length; i++) {
      result = {
        type: 'CallExpr',
        callee: cloneAst(fns[i]),
        args: [result],
        loc: astArg.loc,
      };
    }
    return result;
  }

  /**
   * Rewrite `relabel(value, names)` per spec §sec:design line 482-507
   * to its equivalent `record(name1=val1, ...)` construction. Three
   * value shapes:
   *
   *   relabel(<scalar-or-non-record>, [n])     → record(n = arg)
   *   relabel(<array-typed>, [n1, n2, ...])    → record(n1 = arg[1], ...)
   *   relabel(<record-typed>, [n1, n2, ...])   → record(n1 = old_v1, ...)
   *
   * The names array must be a literal `[StringLiteral, ...]` so we can
   * resolve them statically. The arg may be inline (ArrayLiteral,
   * record(...) call, scalar literal) or an Identifier pointing at
   * another binding — for binding refs we synthesise indexed/field
   * access (arg[i] / arg.<old_field>) and let the lowerer + classifier
   * handle the resulting record uniformly.
   *
   * Anything else returns the input unchanged so the binding falls
   * through (and ends up "not derivable" if no other classifier handles
   * it).
   */
  function inlineRelabel(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'relabel') return astArg;
    if (!Array.isArray(astArg.args) || astArg.args.length !== 2) return astArg;
    const argExpr = astArg.args[0];
    const namesExpr = astArg.args[1];
    if (!argExpr || !namesExpr) return astArg;

    // Names must be a literal [StringLiteral, ...] — anything dynamic
    // doesn't admit a static rewrite.
    if (namesExpr.type !== 'ArrayLiteral') return astArg;
    const elems = namesExpr.elements || namesExpr.elems || [];
    const names: any[] = [];
    for (const el of elems) {
      if (!el || el.type !== 'StringLiteral') return astArg;
      names.push(el.value);
    }
    if (names.length === 0) return astArg;

    const loc = astArg.loc;
    const kw = (name: string, value: any) => ({ type: 'KeywordArg', name, value, loc });
    const recordCall = (kwargs: any) => ({
      type: 'CallExpr',
      callee: { type: 'Identifier', name: 'record', loc },
      args: kwargs, loc,
    });

    // Inline ArrayLiteral source: `relabel([a, b, c], ["x", "y", "z"])`
    // → `record(x = a, y = b, z = c)`.
    if (argExpr.type === 'ArrayLiteral') {
      const argElems = argExpr.elements || argExpr.elems || [];
      if (argElems.length !== names.length) return astArg;
      return recordCall(names.map((n: string, i: number) => kw(n, argElems[i])));
    }

    // Inline record(...) source: rename by position.
    if (argExpr.type === 'CallExpr' && argExpr.callee
        && argExpr.callee.type === 'Identifier' && argExpr.callee.name === 'record'
        && Array.isArray(argExpr.args)) {
      const recArgs = argExpr.args.filter((a: any) => a && a.type === 'KeywordArg');
      if (recArgs.length !== names.length) return astArg;
      return recordCall(names.map((n: string, i: number) => kw(n, recArgs[i].value)));
    }

    // Identifier pointing at another binding. Look up the binding's
    // RHS to determine its structural kind:
    //   - inline ArrayLiteral RHS  → reuse element exprs directly.
    //   - inline record(...) RHS   → reuse kwarg-value exprs directly.
    //   - anything else            → bail out (not statically rewritable).
    //
    // We *don't* synthesise IndexExpr / FieldAccess on the binding name
    // — those lower to (get …) / (get_field …) calls that no measure-op
    // classifier handles, which would cascade-prune the relabel binding.
    // Reading the source RHS directly keeps the result in record-of-refs
    // shape, the only form classifyRecordOrJoint accepts.
    if (argExpr.type === 'Identifier') {
      const target = out.get(argExpr.name);
      const targetAst = target && (target.effectiveValue || (target.node && target.node.value));
      if (targetAst && targetAst.type === 'ArrayLiteral') {
        const targetElems = targetAst.elements || targetAst.elems || [];
        if (targetElems.length !== names.length) return astArg;
        return recordCall(names.map((n: string, i: number) => kw(n, targetElems[i])));
      }
      if (targetAst && targetAst.type === 'CallExpr' && targetAst.callee
          && targetAst.callee.type === 'Identifier' && targetAst.callee.name === 'record'
          && Array.isArray(targetAst.args)) {
        const targetKwargs = targetAst.args.filter((a: any) => a && a.type === 'KeywordArg');
        if (targetKwargs.length !== names.length) return astArg;
        // Reuse the source record's kwarg VALUES verbatim (they're
        // already lifted to Identifier refs after the source binding
        // was visited, or are inline exprs ready to be lifted by the
        // outer visit pass). Synthesising FieldAccess (src.a) instead
        // would force the lowerer to introduce get_field nodes that
        // no measure-op classifier handles, breaking the cascade.
        return recordCall(names.map((n: string, i: number) => kw(n, targetKwargs[i].value)));
      }
      // Identifier pointing at neither — fall through.
    }

    // Single-name scalar wrap: relabel(<anything>, [name]) → record(name = arg)
    if (names.length === 1) {
      return recordCall([kw(names[0], argExpr)]);
    }

    return astArg;
  }

  function inlineOnce(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    const fnName = astArg.callee.name;
    // Use `out`, not `bindings`, so synthesized anon bindings created
    // during the lift pass are visible to inlineOnce too.
    const fnBinding = out.get(fnName);
    if (!fnBinding) return astArg;
    // We only inline real reified callables. fn and kernelof are
    // lowered to functionof in the IR (see lower.js); for AST-level
    // inlining we still see the surface forms, so we accept all
    // three. The substitution rules below assume named (kwarg)
    // parameters, which works because:
    //   - functionof: surface form is already named.
    //   - kernelof: same surface shape.
    //   - fn(...): surface form has positional `_` holes; per spec
    //     §sec:fn line 593-595 each hole is normatively named
    //     `arg1`, `arg2`, …, so positional call args bind in order
    //     and `argN=` kwargs bind to the N-th hole. Handled below.
    // Bijection is semantically `f` — calling it dispatches to the
    // forward function. Rewrite `b(x...)` to `<f-ref>(x...)` and
    // recurse so inlineOnce inlines f's body. f is the first arg of
    // the bijection call; inlineBijectionLift has hoisted any inline
    // shape to an anon binding, so it's always an Identifier here.
    if (fnBinding.type === 'bijection') {
      const bijAst = fnBinding.node && fnBinding.node.value;
      const fIdent = bijAst && bijAst.args && bijAst.args[0];
      if (fIdent && fIdent.type === 'Identifier') {
        const rewritten = Object.assign({}, astArg, {
          callee: makeIdent(fIdent.name, astArg.loc),
        });
        return inlineOnce(rewritten);
      }
      return astArg;
    }
    if (fnBinding.type !== 'functionof' && fnBinding.type !== 'kernelof'
        && fnBinding.type !== 'fn') {
      return astArg;
    }
    const fnAst = fnBinding.node && fnBinding.node.value;
    if (!fnAst || fnAst.type !== 'CallExpr' || !fnAst.args || fnAst.args.length === 0) {
      return astArg;
    }

    // The function's body is the first positional arg.
    const bodyAst = fnAst.args[0];
    if (!bodyAst || bodyAst.type === 'KeywordArg') return astArg;

    // fn(<body>) at the AST level: substitute holes positionally.
    // (lower.js handles the equivalent IR-level lowering for type
    // inference, but we work on AST here for the orchestrator's
    // inlining.)
    if (fnBinding.type === 'fn') {
      const numHoles = countHoles(bodyAst);
      const positional = new Array(numHoles).fill(undefined);
      let posIdx = 0;
      for (const a of (astArg.args || [])) {
        if (a.type === 'KeywordArg') {
          const m = /^arg(\d+)$/.exec(a.name);
          if (m) {
            const idx = parseInt(m[1], 10) - 1;
            if (idx >= 0 && idx < numHoles) positional[idx] = a.value;
          }
        } else if (posIdx < numHoles) {
          positional[posIdx++] = a;
        }
      }
      return substituteHolesPositional(cloneAst(bodyAst), positional);
    }

    // Build the substitution: surface kwarg name → call's arg AST.
    // Then map surface → internal (the placeholder/binding name used
    // in the body via %local) so we can substitute body identifiers.
    //
    // Surface order is the order of kwargs in the function's
    // declaration. The boundary kwarg's VALUE side determines what
    // node the body refers to:
    //   - Identifier (e.g. `theta1 = theta1`)   → body has
    //     Identifier 'theta1'; substitute by name.
    //   - Placeholder (e.g. `par = _par_`)      → body has
    //     Placeholder 'par'; substitute via a sentinel-prefixed key
    //     so it doesn't collide with same-named Identifiers in the
    //     body (substituteIdents reads both shapes).
    //   - anything else (complex boundary expr) → use the surface
    //     name as a fallback; real boundary surgery deferred.
    const surfaceOrder: any[] = [];
    const internalForSurface: Record<string, any> = {};
    for (let i = 1; i < fnAst.args.length; i++) {
      const a = fnAst.args[i];
      if (a.type !== 'KeywordArg') continue;
      surfaceOrder.push(a.name);
      if (a.value && a.value.type === 'Identifier') {
        internalForSurface[a.name] = a.value.name;
      } else if (a.value && a.value.type === 'Placeholder') {
        internalForSurface[a.name] = PLACEHOLDER_SUB_PREFIX + a.value.name;
      } else {
        // Boundary kwarg value is a complex expression
        // (e.g. functionof(body, theta = some_call(x))). The spec's
        // boundary-substitution semantics replace `theta` in the body
        // with the *substituted* form of `some_call(x)`, but doing so
        // correctly requires walking the body for refs to a synthetic
        // boundary node — work we haven't implemented yet.
        // Bail out of inlining rather than silently produce a wrong
        // substitution: leave the user-call as-is so the classifier
        // surfaces a clean "unsupported" outcome (no derivation)
        // instead of an IR with unbound %local refs that crash later.
        return astArg;
      }
    }

    // Implicit boundaries (spec §04 sec:functionof) are already
    // materialised by canonicalizeImplicitBoundaries (called at the
    // top of liftInlineSubexpressions). That pass rewrites the
    // function's AST so args[1+] carries explicit KeywordArgs for
    // every parametric leaf, and the loop above picks them up. So
    // surfaceOrder is correctly populated here regardless of whether
    // the user wrote kwargs or relied on the implicit form.

    // Auto-splatting (spec §sec:calling-convention lines 99-102):
    // `f(record(a=x, b=y))` and `f(some_record_value)` are
    // equivalent to `f(a=x, b=y)`. We detect two splat sources:
    //   - inline `record(...)` calls — splat their KeywordArg
    //     children directly.
    //   - Identifier ref to a record-typed binding — synthesize a
    //     FieldAccess per surface kwarg name.
    // Splat fires only when the call has exactly one positional arg,
    // surfaceOrder has more than one slot, and the arg's record fields
    // cover those slots (otherwise leave the call alone — the type
    // checker already raised "missing argument" or arg-mismatch).
    let callArgs = astArg.args || [];
    if (callArgs.length === 1
        && callArgs[0].type !== 'KeywordArg'
        && surfaceOrder.length >= 1) {
      const arg0 = callArgs[0];
      let splatted: any = null;
      if (arg0.type === 'CallExpr' && arg0.callee
          && arg0.callee.type === 'Identifier'
          && arg0.callee.name === 'record') {
        const fieldNames = new Set<string>();
        for (const f of (arg0.args || [])) {
          if (f.type === 'KeywordArg') fieldNames.add(f.name);
        }
        if (surfaceOrder.every((n: string) => fieldNames.has(n))) {
          splatted = (arg0.args || []).filter((f: any) => f.type === 'KeywordArg');
        }
      } else if (arg0.type === 'Identifier') {
        const recBinding = out.get(arg0.name);
        const t = recBinding && recBinding.inferredType;
        if (t && t.kind === 'record' && t.fields
            && surfaceOrder.every((n: string) => n in t.fields)) {
          splatted = surfaceOrder.map((name: string) => ({
            type: 'KeywordArg',
            name,
            value: { type: 'FieldAccess', object: cloneAst(arg0), field: name, loc: arg0.loc },
            loc: arg0.loc,
          }));
        }
      }
      if (splatted) callArgs = splatted;
    }

    // Walk the call's args. KeywordArg → match by surface name;
    // positional → match by surfaceOrder.
    const argMap = Object.create(null);
    let posIdx = 0;
    for (const a of callArgs) {
      if (a.type === 'KeywordArg') {
        const internal = internalForSurface[a.name];
        if (internal) argMap[internal] = a.value;
      } else {
        const surface = surfaceOrder[posIdx++];
        const internal = surface ? internalForSurface[surface] : null;
        if (internal) argMap[internal] = a;
      }
    }

    // Closure walk: find every transitive ancestor of the body that
    // (1) isn't a boundary itself and (2) isn't fixed-phase (per spec
    // §sec:functionof line 322-323, fixed ancestors are closed over,
    // not copied). For each closure binding, allocate a fresh
    // synthetic name and synthesize a substituted copy of its RHS.
    // After this, refs to closure bindings inside the body resolve
    // to the synthesized copies; refs to boundaries resolve to call
    // args; refs to closed-over fixed bindings resolve to outer-scope
    // refs unchanged.
    //
    // For value functions where boundaries are placeholders that
    // appear only in the body's direct AST (no transitive refs in
    // other bindings), the closure is empty and this collapses to a
    // pure body-level substitution — same as the pre-closure-walk
    // behaviour.
    const boundaries = new Set<string>();
    for (const k in argMap) boundaries.add(k);

    const closure = computeClosure(bodyAst, boundaries);

    // Allocate fresh synthetic names per closure binding and add
    // them to the substitution map. We use freshName() here so
    // closure-synthesized bindings share the same monotonically-
    // increasing __anon counter as inline-lifted anons; this keeps
    // generated names stable across runs and avoids perturbing other
    // anons' indices when no synthesis happens (closure empty).
    for (const origName of closure) {
      argMap[origName] = makeIdent(freshName(), bodyAst.loc);
    }

    // Synthesize each closure binding's RHS with substitution applied.
    // Each synthesized binding's RHS is a clone-with-substitutions of
    // the original — including refs to other closure members, which
    // resolve via argMap to their fresh names. So the synthesized
    // closure forms a self-contained subgraph with the call args at
    // its leaves.
    //
    // The cloned RHS may contain user calls (e.g., `a = f_a(par=beta1)`
    // clones with substitution to `__anon_a = f_a(par=__anon_beta1)`,
    // which still has a user call in it) and inline subexpressions
    // that haven't been lifted to anons. Run the same inline-and-lift
    // pipeline on each clone — same machinery the main loop applies
    // to user bindings — so the synthesized closure is fully ready
    // for classification.
    for (const origName of closure) {
      // Mirror computeClosure: prefer the post-lift form. A lifted
      // anon (e.g. __anon3 = Normal(mu=theta1, sigma=theta2)) only
      // lives in `out`, and skipping it leaves the body referencing
      // the fresh closure name with no binding behind it.
      const orig = out.get(origName) || bindings.get(origName);
      if (!orig || !orig.node || !orig.node.value) continue;
      let newRhs = substituteIdents(cloneAst(orig.node.value), argMap);
      newRhs = inlineUserCall(newRhs);
      visit(newRhs);
      const fresh = argMap[origName].name;
      out.set(fresh, makeSyntheticBinding(fresh, newRhs));
    }

    // Substitute identifiers in a deep-cloned body. Different call
    // sites get independent ASTs.
    return substituteIdents(cloneAst(bodyAst), argMap);
  }

  /**
   * Compute the transitive ancestor closure of `bodyAst`'s refs,
   * stopping at boundary names and at fixed-phase bindings (which
   * are closed over per spec §sec:functionof). Returns a Set of
   * binding names that need to be cloned-with-substitution at this
   * call site.
   *
   * Op-agnostic: just follows refs through binding RHS ASTs.
   * `lawof`, `draw`, `weighted`, `iid`, `record`, etc. all walk
   * through their operands without special-casing — substitution
   * propagates through any tree shape.
   */
  function computeClosure(bodyAst: any, boundaries: Set<string>) {
    const closure = new Set<string>();
    const visiting = new Set<string>();

    function walk(name: string) {
      if (closure.has(name) || visiting.has(name)) return;
      if (boundaries.has(name)) return;          // boundary, will be substituted
      // Look up in the post-lift map first so lift-introduced anons
      // (the Normal / iid extracted from a functionof body) are seen
      // by the walk too — without this, refs inside such an anon
      // back to a boundary name like `theta1` aren't reached by
      // substituteIdents and stay as the outer stochastic ref. Fall
      // back to the analyzer's pre-lift bindings for any names the
      // lift hasn't processed yet (the loop is in source order, so
      // earlier-defined bindings are always in `out`).
      const b = out.get(name) || bindings.get(name);
      if (!b) return;                            // unknown name (built-in, etc.)
      if (b.phase === 'fixed') return;            // closed over per spec
      visiting.add(name);
      closure.add(name);
      collectRefsAst(b.node && b.node.value);
      visiting.delete(name);
    }
    function collectRefsAst(node: any) {
      if (node == null || typeof node !== 'object') return;
      if (Array.isArray(node)) { for (const c of node) collectRefsAst(c); return; }
      if (node.type === 'Identifier') walk(node.name);
      for (const k in node) collectRefsAst(node[k]);
    }
    collectRefsAst(bodyAst);
    return Array.from(closure);
  }

  function substituteIdents(ast: any, sub: any): any {
    if (ast == null || typeof ast !== 'object') return ast;
    if (Array.isArray(ast)) return ast.map((c: any) => substituteIdents(c, sub));
    if (ast.type === 'Identifier' && sub[ast.name]) {
      // Replace with a clone of the substitute so mutations to either
      // side don't bleed across the boundary.
      return cloneAst(sub[ast.name]);
    }
    if (ast.type === 'Placeholder' && sub[PLACEHOLDER_SUB_PREFIX + ast.name]) {
      // Placeholder boundary applied — substitute with the call's arg.
      // Sentinel-prefixed key avoids collision with same-named
      // Identifier substitutions when both shapes appear in the body.
      return cloneAst(sub[PLACEHOLDER_SUB_PREFIX + ast.name]);
    }
    const out: Record<string, any> = {};
    for (const k in ast) out[k] = substituteIdents(ast[k], sub);
    return out;
  }

  // Walk the AST in reading order, replacing each Hole with the
  // corresponding positional arg. Holes whose `positional[i]` is
  // undefined stay as Hole (unbound — caller's responsibility to
  // flag the missing arg).
  function substituteHolesPositional(ast: any, positional: any[]) {
    let i = 0;
    function walk(node: any): any {
      if (node == null || typeof node !== 'object') return node;
      if (Array.isArray(node)) return node.map(walk);
      if (node.type === 'Hole') {
        const arg = positional[i++];
        return arg ? cloneAst(arg) : node;
      }
      const out: Record<string, any> = {};
      for (const k in node) out[k] = walk(node[k]);
      return out;
    }
    return walk(ast);
  }

  function countHoles(ast: any) {
    let n = 0;
    (function walk(node: any) {
      if (node == null || typeof node !== 'object') return;
      if (Array.isArray(node)) { for (const c of node) walk(c); return; }
      if (node.type === 'Hole') { n++; return; }
      for (const k in node) walk(node[k]);
    })(ast);
    return n;
  }
}

/**
 * Whether the worker's evaluateExpr can compute this IR end-to-end
 * given a numeric env. Conservative — returns false for anything we're
 * not 100% sure the evaluator handles, so the orchestrator can short-
 * circuit before involving the worker.
 */
function isEvaluable(ir: any): boolean {
  if (!ir) return false;
  switch (ir.kind) {
    case 'lit':         return typeof ir.value === 'number' || typeof ir.value === 'boolean';
    case 'const':       // pi, inf, im — evaluator resolves these.
                        return true;
    case 'ref':         // resolved against env at evaluation time.
                        return true;
    case 'call':
      if (!ir.op || !EVALUABLE_OPS.has(ir.op)) return false;
      // rand(state, measure) — the measure arg is a measure IR passed
      // verbatim to the trace evaluator, NOT a value expression. So we
      // only require the state (first) arg to be evaluable; whether
      // the measure is sampleable is the trace evaluator's call. Same
      // logic for any future state-threaded primitive that takes a
      // measure literal (none today besides rand).
      if (ir.op === 'rand') {
        const args = ir.args || [];
        if (args.length !== 2) return false;
        return isEvaluable(args[0]);
      }
      // All args / kwargs / fields must themselves be evaluable.
      // record IR uses `fields: [{name, value}, ...]` instead of args,
      // so we walk that shape too — a record(a=x, b=y) is evaluable
      // only if every field value is.
      if (ir.args) {
        for (const a of ir.args) if (!isEvaluable(a)) return false;
      }
      if (ir.kwargs) {
        for (const k in ir.kwargs) if (!isEvaluable(ir.kwargs[k])) return false;
      }
      if (Array.isArray(ir.fields)) {
        for (const f of ir.fields) if (!isEvaluable(f && f.value)) return false;
      }
      return true;
    default:
      return false;
  }
}

module.exports = {
  argSignature,
  opUsesValueKwargs,
  inferSyntheticType,
  PLACEHOLDER_SUB_PREFIX,
  canonicalizeImplicitBoundaries,
  bfsImplicitElementofLeavesAst,
  liftInlineSubexpressions,
  isEvaluable,
};
