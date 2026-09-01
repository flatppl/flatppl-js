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

import type { IRNode } from './engine-types';

const { lowerExpr } = require('./lower.ts');
const { signatureOf } = require('./signatures.ts');
const {
  parseSetIR,
  EVALUABLE_OPS,
  SAMPLEABLE_DISTRIBUTIONS,
  resolveCallableAlias,
} = require('./ir-shared.ts');
const { MEASURE_PRODUCING, BUILTIN_FUNCTIONS } = require('./builtins.ts');

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
  if (op === 'likelihoodof') {
    // likelihoodof(<kernel>, <obs>): kernel position is value-shaped
    // for lifting (an inline fn / functionof / kernelof / fchain
    // gets hoisted to an anon binding so classifyLikelihoodof sees
    // a clean self-ref). Obs is value-typed.
    return Array(numArgs).fill('value');
  }
  if (op === 'bayesupdate') {
    // bayesupdate(<likelihood>, <prior>): both arg positions are
    // measure-shaped at the consumption-side (likelihoods are
    // density objects classified as measures-of-real for typeinfer
    // purposes; priors are measures). Lifting both as 'measure'
    // hoists inline likelihoodof(...) / inline measure expressions
    // into anon bindings so classifyBayesupdate (derivations.ts)
    // sees self-refs at args[0] (the L ref) and args[1] (the prior
    // ref), matching its required shape. Without this, restrict-
    // expanded forms with inline `likelihoodof(...)` at args[0]
    // failed to classify.
    return Array(numArgs).fill('measure');
  }
  if (op === 'fchain') {
    // fchain(f1, f2, ...): every positional arg is a function value.
    // Function-value lifting works the same as pushfwd's function arg
    // — inline `fn(...)` / `functionof(...)` get lifted to anon
    // bindings so the chain composition sees uniform refs.
    return Array(numArgs).fill('value');
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
      // Phase 5.1 Session 5e — recognise synthetic bijection bindings
      // emitted by `inlineMvNormalLift`. Without this case the lift
      // post-pass + buildDerivations' bijection-construction loop
      // (derivations.ts:159-179, gated on binding.type === 'bijection')
      // skip the synthetic binding entirely, leaving binding.bijection
      // unpopulated and registryName + paramIRs unattached.
      case 'bijection':  return 'bijection';
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

// Square matrix ops whose ARGUMENT shape certifies the result is square, so
// the affine lifts can discover D from the argument when the op's own result
// type is %dynamic. ONLY genuinely shape-preserving ops belong here. NOT
// `transpose`/`adjoint`: those swap dims, so a `transpose([3,2])` is `[2,3]`.
// A square transpose still routes via a concrete-[D,D] inferredType instead.
const SHAPE_PRESERVING_SQUARE_OPS = new Set([
  'lower_cholesky', 'cholesky', 'inv']);

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

// =====================================================================
// C5: Broadcast-of-broadcast fusion (engine-concepts §05)
// =====================================================================
// Collapse `broadcast(K, outer…)` where K is a binding whose IR is
// `functionof(broadcast(<bareDist>, p₁,…), kw)` (after peeling an
// optional `lawof`) and each inner arg pᵢ is EXACTLY the i-th kernel
// parameter (1:1 match, no closed-over refs).  Rewrites to
// `broadcast(<bareDist>, outer…)` so it routes through the C2/C3
// multi-axis kernel-broadcast executor rather than the composite path.
//
// Fires ONLY on the clean 1:1 case.  Any partial or closed-over
// inner arg leaves the call untouched (composite executor handles it).
function tryFuseBroadcastOfBroadcast(callIR: any, bindings: Map<string, any>): any {
  if (!callIR || callIR.op !== 'broadcast') return null;
  const head = callIR.args && callIR.args[0];
  if (!head || head.kind !== 'ref') return null;
  const kb = bindings.get(head.name);
  if (!kb || !kb.ir || kb.ir.op !== 'functionof') return null;
  let body = kb.ir.body;
  if (body && body.op === 'lawof') body = body.args && body.args[0];
  if (!body || body.op !== 'broadcast') return null;
  const innerHead = body.args && body.args[0];
  // innerHead must be a bare-distribution ref (not a user binding, not a
  // %local-namespace placeholder — a bare distribution head is never local).
  if (!innerHead || innerHead.kind !== 'ref' || bindings.has(innerHead.name)
      || innerHead.ns === '%local') return null;
  const params: any[] = kb.ir.params || [];
  const innerArgs: any[] = body.args ? body.args.slice(1) : [];
  // innerArgs must be exactly the params, in order, as %local refs.
  if (innerArgs.length !== params.length || params.length === 0) return null;
  for (let i = 0; i < params.length; i++) {
    const a = innerArgs[i];
    if (!a || a.kind !== 'ref' || a.name !== params[i]) return null;
  }
  const outerArgs = callIR.args.slice(1);
  // broadcast(Dist) with no data axes is degenerate; don't fuse to it.
  if (outerArgs.length === 0) return null;
  return { kind: 'call', op: 'broadcast', loc: callIR.loc,
    args: [innerHead, ...outerArgs] };
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
  // Phase 5.1 Session 5e — separate name families for the synthetic
  // bijection-binding and iid-base-binding produced by the MvNormal
  // lowering helper. Counter is shared so collisions across families
  // are structurally impossible.
  function freshBijName() {
    let n: string;
    do { n = '__bij' + (counter++); } while (out.has(n));
    return n;
  }
  function freshIidName() {
    let n: string;
    do { n = '__iid' + (counter++); } while (out.has(n));
    return n;
  }
  // Module-name set hoisted up here (it was previously built after
  // the binding loop) so `inlineMvNormalLift` can call `lowerExpr`
  // on the synthetic-binding's muIR / covIR with the same
  // lower-context the post-pass uses. Module-typed bindings are
  // entered into `out` at the top of this function via the spread
  // from `bindings`; they aren't created by lift itself.
  const moduleNames = new Set<string>();
  for (const [bname, b] of out) {
    if (b && b.type === 'module') moduleNames.add(bname);
  }
  // All binding names (mirrors pir.lowerToModule): `_lowerReification`
  // consults this to tell a real-node boundary cut (body refs stay
  // `self`) from a pure formal (placeholder semantics). `out` already
  // holds every user binding at this point; lift-synthesized anons are
  // never boundary-kwarg names.
  const bindingNames = new Set<string>(out.keys());
  const liftLowerCtx = { localScope: null, moduleNames, bindingNames };
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

  // Metricsum-lift counter state + helpers. Declared at the top of the
  // function scope so the binding loop's first call into inlineUser
  // Call → inlineMetricsumLift → freshGDownName doesn't hit a TDZ
  // violation on `gDownCounter`. (Function declarations hoist; `let`
  // bindings do not — they enter scope at their textual position.)
  let gDownCounter = 0;
  function freshGDownName() {
    let n: string;
    do { n = '__g_down_' + (gDownCounter++); } while (out.has(n));
    return n;
  }
  let lowerInternalAxisCounter = 0;
  function freshLowerInternalAxisName() {
    return '__ms_lo_' + (lowerInternalAxisCounter++);
  }
  let upperOutputAxisCounter = 0;
  function freshUpperOutputAxisName() {
    return '__ms_up_' + (upperOutputAxisCounter++);
  }
  // Mixed-variance pre-mix cascade names (one per cascade step in
  // Form-B metricsum lift; engine-concepts §23). Distinct counter from
  // freshGDownName / freshName so multiple metricsum calls in the same
  // module produce non-overlapping mixed names without polluting the
  // generic anon namespace.
  let mixedCounter = 0;
  function freshMixedName() {
    let n: string;
    do { n = '__ms_mixed_' + (mixedCounter++); } while (out.has(n));
    return n;
  }
  // Output-raise cascade names (one per lower-variance output axis).
  let raisedCounter = 0;
  function freshRaisedName() {
    let n: string;
    do { n = '__ms_raised_' + (raisedCounter++); } while (out.has(n));
    return n;
  }
  // Hoisted composed tensor bases (`(p1 + p2)[.mu_]`) — the cascade keys
  // on a name, so a non-Identifier base needs one.
  let msBaseCounter = 0;
  function freshMsBaseName() {
    let n: string;
    do { n = '__ms_base_' + (msBaseCounter++); } while (out.has(n));
    return n;
  }
  // Hidden binding for the main aggregate when a raise cascade follows
  // (so the user's RHS gets redirected to the final raise step).
  let mainCounter = 0;
  function freshMainName() {
    let n: string;
    do { n = '__ms_main_' + (mainCounter++); } while (out.has(n));
    return n;
  }
  function makeAxisRef(name: string, loc: any) {
    // Bare AxisRef — variance marker stripped. AST shape mirrors the
    // ast.AxisRef ctor (variance is omitted when undefined, so we
    // never emit `variance: undefined` keys).
    return { type: 'AxisRef', name, loc: loc || null };
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
  //
  // The lowerExpr call must see `moduleNames` so `mod.X` lowers to
  // `(%ref mod X)` per spec §11 (not to `get_field`). After
  // re-lowering we run alias-resolution on the `.ir` fields so
  // `breit_wigner = hepphys.X` style aliases canonicalise (the
  // analyzer's earlier alias-resolution pass operated on `.rhs` of
  // loweredModule; lift's re-lower bypasses that, so we run the
  // same canonicalisation on `.ir` here to keep the two views in
  // lockstep).
  // moduleNames + liftLowerCtx are hoisted to the top of this function
  // (Phase 5.1 Session 5e) so the inlineMvNormalLift helper can lower
  // muIR / covIR with the same lower-context. The post-pass below
  // continues to use them unchanged.
  for (const [name, b] of out) {
    if (!b || !b.node || !b.node.value) continue;
    let ir = null;
    try { ir = lowerExpr(b.effectiveValue || b.node.value, liftLowerCtx); } catch (_) { ir = null; }
    out.set(name, { ...b, ir });
  }
  const { resolveAliasesOnBindings } = require('./alias-resolution.ts');
  resolveAliasesOnBindings(out, 'ir');
  // C5 fusion pass: collapse broadcast(K, outer…) where K is a
  // functionof whose body is broadcast(<bareDist>, params…) and every
  // inner arg is exactly the matching kernel param (1:1, in order).
  // Runs after alias-resolution so K refs are fully resolved.
  for (const [, b] of out) {
    if (!b || !b.ir) continue;
    const fused = tryFuseBroadcastOfBroadcast(b.ir, out);
    if (fused) b.ir = fused;
  }
  // Rewrite the state half of a composite `rand` to the value-domain
  // `rand_succ` successor (engine-concepts §11). After the IR-cache
  // loop + alias-resolution, so every `b.ir` and inner-measure ref resolves.
  rewriteCompositeRandSucc(out);
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
          if (op === 'joint')             a.value = liftJointComponent(a.value);
          else if (isRecordLike)           a.value = liftMeasure(a.value);
          else if (opUsesValueKwargs(op)) a.value = liftValue(a.value);
          // `broadcast(f, name=[lits or refs], …)` (kwarg form of the
          // surface broadcast surfaces, including dot-notation
          // f.(name=A, …)). Per spec §04 every kwarg value is a
          // collection; inline ArrayLiteral collection args block the
          // classifier (the lowered IR uses `vector(...)` which
          // ir-shared deliberately omits from EVALUABLE_OPS — see
          // below for the matching positional-args block). Hoist
          // those to anon bindings so the classifier sees bare refs.
          if (op === 'broadcast' && a.value
              && a.value.type === 'ArrayLiteral'
              && !containsHoleOrPlaceholder(a.value)) {
            const name = freshName();
            out.set(name, makeSyntheticBinding(name, a.value));
            a.value = makeIdent(name, a.value.loc);
          }
          continue;
        }
        const expected = sig ? sig[i] : null;
        if      (op === 'joint' && expected === 'measure') astNode.args[i] = liftJointComponent(a);
        else if (expected === 'measure')          astNode.args[i] = liftMeasure(a);
        else if (expected === 'value-or-measure') astNode.args[i] = liftMeasureOrValue(a);
        else                                      astNode.args[i] = liftValue(a);
        // `broadcast(f, [lit_or_ref, …], X, …)` and dot-call /
        // dotted-operator surfaces all lower through the same
        // `broadcast` CallExpr (parser.broadcastCall). When a non-head
        // positional arg is an inline ArrayLiteral (the Ref-wrap idiom
        // `[C]` or a literal collection `[1.0, 2.0, …]`) the lowered
        // IR carries it as `vector(<elems>)` — which is deliberately
        // omitted from `EVALUABLE_OPS` (stochastic-element arrays
        // keep the `kind:'array'` / `kind:'tuple'` classifier path).
        // Without lifting the array literal to its own anon binding,
        // the broadcast's `isEvaluable` recursion sees a non-
        // evaluable `vector(...)` arg and the classifier yields no
        // derivation for the parent binding. Hoist the array literal
        // so the broadcast's arg becomes a bare ref to an anon that
        // the classifier handles via the kind:'array' / kind:'tuple'
        // paths (see derivations.ts line 741+).
        //
        // i > 0 because args[0] is the callable (function / kernel /
        // distribution ref); inline ArrayLiteral there would be a
        // type error.
        if (op === 'broadcast' && i > 0 && astNode.args[i]
            && astNode.args[i].type === 'ArrayLiteral'
            && !containsHoleOrPlaceholder(astNode.args[i])) {
          const name = freshName();
          out.set(name, makeSyntheticBinding(name, astNode.args[i]));
          astNode.args[i] = makeIdent(name, astNode.args[i].loc);
        }
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

  // True iff `astNode` is itself a `fn(...)` / `functionof(...)` /
  // `kernelof(...)` CallExpr — a CLOSED reifier by construction (parser
  // desugaring / the surface grammar): every Hole/Placeholder inside its
  // own subtree is one of ITS OWN formals, never an outer scope's.
  function isClosedReifierCall(astNode: any): boolean {
    return !!astNode && astNode.type === 'CallExpr' && astNode.callee
      && astNode.callee.type === 'Identifier'
      && (astNode.callee.name === 'fn'
          || astNode.callee.name === 'functionof'
          || astNode.callee.name === 'kernelof');
  }

  function liftMeasure(astArg: any) {
    if (!astArg) return astArg;
    astArg = inlineUserCall(astArg);
    return finishLiftMeasure(astArg);
  }

  // A `joint` component keeps an inline `kernelof`/`functionof` reification
  // AS-IS rather than routing it through `liftMeasure`'s generic hoist. A
  // hoisted synthetic binding never gets an `inferredType` (typeinfer runs
  // over the pre-lift module; synthetic bindings created here don't exist
  // yet), and the applied-density hoist (`_jointComponentAsMeasure`) tells a
  // kernel component from a measure component by reading that type off the
  // resolved binding — an untyped synthetic binding reads as a bare measure,
  // silently dropping the component's kernel-ness. That is the inline
  // `joint(p = K1, q = kernelof(u, z = z))` shape: hoisting `q` erased its
  // kernel type and the fan-out scored as if `q` were absent. Left inline,
  // `_jointComponentAsMeasure` reads the CallExpr head directly instead —
  // exactly the same shape a named `K2 = kernelof(u, z = z)` binding gives
  // it, just without the lookup.
  function liftJointComponent(astArg: any) {
    if (!astArg) return astArg;
    astArg = inlineUserCall(astArg);
    if (astArg.type === 'CallExpr' && astArg.callee && astArg.callee.type === 'Identifier'
        && (astArg.callee.name === 'kernelof' || astArg.callee.name === 'functionof')) {
      visit(astArg);
      return astArg;
    }
    return finishLiftMeasure(astArg);
  }

  function finishLiftMeasure(astArg: any) {
    visit(astArg);
    if (astArg.type === 'Identifier') return astArg;
    // `weighted(w, M)` / `logweighted(g, M)` (spec §06: the weight/
    // log-weight is a constant OR a function of the variate): an INLINE
    // closed-reifier weight (`x -> w(x)`, `fn(...)`) carries its OWN
    // Hole/Placeholder formals — those don't escape past the reifier, so
    // they must not block lifting the ENCLOSING weighted/logweighted call
    // to its own anon binding (#307's classifier, derivations.ts's
    // `_classifyWeightedByFunction`, needs a NAMED base to substitute
    // the weight's parameter into; an inline weight that stays embedded
    // never reaches a named base and fails classification — "no
    // derivation"). Scoped to exactly this shape: EVERY other
    // `containsHoleOrPlaceholder` call site (jointchain/kchain kernel
    // args, record/joint kwarg fields, broadcast array literals, and
    // weighted/logweighted's own BASE slot) keeps the original
    // conservative check unchanged.
    if (astArg.type === 'CallExpr' && astArg.callee && astArg.callee.type === 'Identifier'
        && (astArg.callee.name === 'weighted' || astArg.callee.name === 'logweighted')
        && Array.isArray(astArg.args) && astArg.args.length === 2) {
      const weightArg = astArg.args[0];
      const baseArg = astArg.args[1];
      const weightEscapes = isClosedReifierCall(weightArg)
        ? false : containsHoleOrPlaceholder(weightArg);
      if (weightEscapes || containsHoleOrPlaceholder(baseArg)) return astArg;
    } else if (containsHoleOrPlaceholder(astArg)) {
      return astArg;
    }
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
    //   - logdensityof(M,x)— needs density.ts over M's expanded IR
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
      // The explicit matrix-vector affine spelling `pushfwd(x -> L * x + b,
      // iid(…, D))` routes to the affine registry, per spec §06's
      // known-bijection list. Must run BEFORE inlinePushfwdLift, which
      // hoists the f-position functionof to an anon binding.
      astArg = inlinePushfwdAffineLift(astArg);
      astArg = inlinePushfwdLift(astArg);
      astArg = inlineBijectionLift(astArg);
      // Phase 5.1 Session 5e — lower MvNormal CallExpr to the
      // canonical pushfwd-of-iid decomposition (engine-concepts §22).
      // Emits two synthetic bindings (__bij_N + __iid_N) and rewrites
      // the MvNormal call in place to `pushfwd(__bij_N, __iid_N)`.
      // Gate: literal-ArrayLiteral mu + cov (or one-level refs to
      // ArrayLiteral bindings). Non-resolvable cases fall through to
      // the AST-preserved matMvNormal path.
      astArg = inlineMvNormalLift(astArg);
      // P3 — lower a surviving non-scalar `locscale(base, shift, scale)` to
      // the canonical affine-registry pushfwd (mirrors inlineMvNormalLift).
      // Scalar locscale never reaches here — the analyzer pre-pass expands
      // it; only detected-non-scalar locscale with an iid base survives to
      // this routing.
      astArg = inlineLocscaleAffineLift(astArg);
      // Metric-aware Einstein summation (spec §04 §sec:metricsum). The
      // surface shorthand `metric: result[output_indices] := expr`
      // desugars (in the parser) to `metricsum(metric, [output_axes], expr)`
      // with variance-marked axes (`.name^` / `.name_`). The lift pass
      // rewrites every metricsum() call to a sum-`aggregate(...)` with
      // metric / inv(metric) factor insertions per the spec lowering
      // ("Each `_` axis name in `expr` becomes an `inv(metric)`
      // contraction; each `_` output axis becomes a `metric` contraction
      // after the sum, raising the result to all-upper canonical
      // storage"). After this pass the IR contains no `metricsum`
      // nodes — only the aggregate the rest of the engine already
      // handles.
      astArg = inlineMetricsumLift(astArg);
      astArg = inlineFilterLift(astArg);
      astArg = inlineLikelihoodofLift(astArg);
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
    let builtinSourceName: string | null = null;
    if (fArg.type === 'Identifier') {
      if (bindings.has(fArg.name)) return astArg;
      // Spec §06's own example, `pushfwd(exp, mu)`: a bare BUILTIN
      // function name at f-position desugars to `fn(exp(_))`, the
      // same shape the explicit-fn spelling below produces —
      // classifyPushfwd then sees the usual self-ref-to-a-functionof
      // binding. An unbound name that ISN'T a known builtin is left
      // alone (undefined-variable diagnostics are the analyzer's job).
      if (!BUILTIN_FUNCTIONS.has(fArg.name)) return astArg;
      builtinSourceName = fArg.name;
      fArg = {
        type: 'CallExpr',
        callee: makeIdent('fn', fArg.loc),
        args: [{
          type: 'CallExpr',
          callee: makeIdent(fArg.name, fArg.loc),
          args: [makeHole(fArg.loc)],
          loc: fArg.loc,
        }],
        loc: fArg.loc,
      };
    } else {
      // No further lifting for a non-Identifier, non-KeywordArg fArg
      // beyond recursing into it — `visit` mutates in place and can
      // never turn a CallExpr into a bare Identifier, so there is no
      // shape left to special-case here.
      visit(fArg);
    }
    const n = freshName();
    const synth: any = makeSyntheticBinding(n, fArg);
    // Surfaced by buildDerivations' reversed-pushfwd diagnostic so the
    // message names the user's own spelling (`exp`) instead of this
    // lift-synthesized anon binding name.
    if (builtinSourceName) synth.builtinSourceName = builtinSourceName;
    out.set(n, synth);
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

  /**
   * Phase 5.1 Session 5e — lower `MvNormal(mu = ..., cov = ...)` calls
   * to the canonical pushfwd-of-iid decomposition per engine-concepts
   * §22:
   *
   *   MvNormal(mu, cov)  →  pushfwd(<bij>, <iid>)
   *
   * where:
   *   __bij_N = bijection(fn(_), fn(_), 0.0)
   *             // additionally marked via the binding's
   *             // .__mvnormalLowering field, which buildDerivations'
   *             // bijection-construction loop reads to populate
   *             // binding.bijection.registryName = 'affine' and
   *             // binding.bijection.paramIRs = {L: lower_cholesky(cov),
   *             //                                b: mu}.
   *   __iid_N = iid(Normal(mu = 0.0, sigma = 1.0), D)
   *
   * Gate (conservative MVP — see ast_construction_recipes RECIPE 4):
   *   - kwarg form `MvNormal(mu = X, cov = Y)`. Positional 2-arg form
   *     is uncommon in production code; defer.
   *   - X and Y both resolve to ArrayLiteral via __resolveLiteralArrayRef
   *     (literal at the call site OR one-level ref to an ArrayLiteral
   *     binding). D = X.elements.length; covLit dimensions must match
   *     [D, D].
   *   - Element types of X and Y are NOT constrained — `[base, base+1]`
   *     is admissible. resolveIRToValue at materialise time handles
   *     non-literal elements (as long as they're atom-independent).
   *
   * When the gate doesn't fire (non-literal mu / cov, deeper ref
   * chains, MvNormal as a kernel-broadcast head etc.), `astArg` is
   * returned unchanged and the existing matMvNormal path handles the
   * MvNormal IR at materialise time as before.
   *
   * After Session 5e lands, matMvNormal is structurally unreachable
   * for any binding whose lift gate fires (the IR contains no MvNormal
   * node), so matPushfwd's vector-base + registry fast path (Session
   * 5d) is the sole materialiser for the rewritten cases.
   */
  function inlineMvNormalLift(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'MvNormal') return astArg;
    // Parse kwargs, or the positional form (5h-A): spec §08 defines the
    // parameter order — `MvNormal(mu, cov)` — so two positional args map
    // to (mu, cov). Mixed forms beyond that stay un-lowered.
    const kw: Record<string, any> = {};
    const pos: any[] = [];
    for (const a of (astArg.args || [])) {
      if (a && a.type === 'KeywordArg') kw[a.name] = a.value;
      else if (a) pos.push(a);
    }
    const muAst = kw.mu != null ? kw.mu : (pos.length === 2 ? pos[0] : null);
    const covAst = kw.cov != null ? kw.cov : (pos.length === 2 ? pos[1] : null);
    if (!muAst || !covAst) return astArg;
    // Discover the event dim D from mu (5f-1 static paths): (a) literal /
    // ref-to-literal / inline-rowstack-literal, OR (b) a named ref whose
    // binding carries a statically-known inferredType array (shape[0] a
    // concrete integer). mu must be rank-1 — a matrix-form mean
    // (`rowstack([[2.0]])`, shape [1,1]) is REJECTED here (spec §08: mu
    // is a VECTOR), surfacing as the classifier's clean refusal rather
    // than lowering to a [1,1] param the density guard would misread as
    // atom-batched (Session 5f adversarial-verify Issue 1).
    const D = __discoveredMvNormalD(muAst, out, 1);
    if (D != null && (!Number.isInteger(D) || D < 1)) return astArg;
    // 5h-A dynamic-D: when D is NOT statically known, the iid count
    // becomes the runtime expression `lengthof(mu)` — classifyIid's
    // deferred-count pass resolves it through fixedValues once shapes
    // are computable. Statically gate only what must hold for the
    // REWRITE to be meaningful: mu must be rank-1 wherever a rank is
    // visible at all (the literal/ref guards inside
    // __discoveredMvNormalD already rejected a visible rank-2 mu by
    // returning null — distinguish that REJECT from "no info" below).
    let countAst: any = null;
    if (D != null) {
      countAst = makeNumLit(D, astArg.loc);
    } else {
      // No static D. Reject the shapes the static paths POSITIVELY
      // identified as wrong-rank (a literal/ref matrix-form mean);
      // accept an Identifier whose rank is unknown or confirms rank-1
      // with a dynamic length — but ONLY when it names a REAL module
      // binding. A kernel-boundary placeholder / formal (`kernelof(
      // joint(loc = MvNormal(mu = m, …), …), m = m)` — `m` unbound)
      // belongs to the composite-body recognizers (engine-concepts
      // §21/§22.2(d); teaching them the lowered pushfwd form is 5h-B),
      // and `lengthof(<formal>)` could never resolve anyway.
      if (__mvNormalRankConflict(muAst, out, 1)) return astArg;
      let muRefAst: any = null;
      if (muAst.type === 'Identifier') {
        // Must name a REAL module binding — a bare formal skips (above).
        if (out.has(muAst.name)) muRefAst = makeIdent(muAst.name, astArg.loc);
      } else if (muAst.type === 'CallExpr' || muAst.type === 'ArrayLiteral'
                 || muAst.type === 'IndexExpr') {
        // An `IndexExpr` mu is a §06 `ksuperpose` family row
        // (`mus[i, :]`), which ksuperpose-expand synthesises per
        // component; it hoists like any other inline mu.
        //
        // Inline dynamic-shape mu (`MvNormal(mu = fill(0.0, k), …)`):
        // hoist it to an anon binding so the count expression has a
        // name to measure. Skip if it contains formals/placeholders —
        // those can't escape the enclosing reification scope (the
        // composite recognizers own that shape).
        if (!__containsFormalRef(muAst, out)) {
          const muAnon = freshName();
          out.set(muAnon, makeSyntheticBinding(muAnon, muAst));
          muRefAst = makeIdent(muAnon, astArg.loc);
        }
      }
      if (!muRefAst) return astArg;   // formal-carrying mu → composite scope
      countAst = {
        type: 'CallExpr',
        callee: makeIdent('lengthof', astArg.loc),
        args: [muRefAst],
        loc: astArg.loc,
      };
    }
    // Cov gating (5h-A relaxation): a STATICALLY VISIBLE shape conflict
    // (literal non-square, ref with rank ≠ 2 or shape[1] ≠ D) still
    // refuses the rewrite; shapes that are merely UNKNOWN (inline
    // `diagmat(...)`, `eye(<param>)`, dynamic refs) lower anyway — the
    // affine registry resolves mu/cov at materialise time and the
    // Cholesky/matvec shape checks fail loudly on a real mismatch,
    // which is exactly the runtime contract the retired matMvNormal
    // terminal had.
    const covLit = __resolveLiteralArrayRef(covAst, out);
    if (covLit) {
      const rows = covLit.elements.length;
      if (D != null && rows !== D) return astArg;
      for (const row of covLit.elements) {
        // Defensive: the parser never emits a falsy element inside an
        // ArrayLiteral, so no surface program reaches this.
        /* c8 ignore start */
        if (!row) return astArg;
        /* c8 ignore stop */
        // A row spelled as a nested literal exposes its length, so a
        // non-square matrix is a VISIBLE conflict and refuses. A row
        // that is any other expression (an Identifier naming a vector,
        // a call) exposes nothing — per the relaxation above that is
        // UNKNOWN, not wrong, so keep lowering and let the affine
        // registry's runtime shape checks reject a real mismatch.
        // Refusing here instead left `m` unsamplable while
        // `logdensityof(m, …)` still scored through another path.
        if (row.type !== 'ArrayLiteral') continue;
        if (row.elements.length !== rows) return astArg;
      }
    } else if (covAst.type === 'Identifier') {
      const cb = out.get(covAst.name);
      if (!cb) return astArg;   // formal / placeholder cov → composite scope
      const ct = cb.inferredType;
      if (ct && ct.kind === 'array') {
        if (ct.rank !== 2) return astArg;            // visibly not a matrix
        if (D != null && Number.isInteger(ct.shape[1])
            && ct.shape[1] !== D) return astArg;     // visibly mismatched
      } else if (ct && ct.kind !== 'deferred' && ct.kind !== 'any') {
        return astArg;                               // visibly not an array
      }
    }
    // (inline non-ref cov with unknown shape: lower — runtime checks own it)
    // Emit the synthetic bijection binding `__bij_N`. AST shape is the
    // same identity-stub form used in test/bijection-registry-
    // dispatch.test.ts — `fn(_)` for forward AND inverse, scalar 0
    // for log-volume. The bijection registry's affine entry provides
    // the real implementation; the AST stubs satisfy the additive-
    // invariant (resolveFnBody must still find a callable body even
    // when registryName is set).
    //
    // The fn(_) sub-expressions MUST be pre-hoisted to anon bindings
    // so buildDerivations' bijection-construction loop
    // (derivations.ts:159-179) sees args[0] and args[1] as
    // Identifiers. Synthetic bindings inserted into `out` during the
    // inlineUserCall loop bypass the per-binding visit() pass that
    // would normally hoist them via inlineBijectionLift.
    const fwdAnon = freshName();
    out.set(fwdAnon, makeSyntheticBinding(fwdAnon, makeFnHole(astArg.loc)));
    const invAnon = freshName();
    out.set(invAnon, makeSyntheticBinding(invAnon, makeFnHole(astArg.loc)));
    const bijAst = {
      type: 'CallExpr',
      callee: makeIdent('bijection', astArg.loc),
      args: [
        makeIdent(fwdAnon, astArg.loc),
        makeIdent(invAnon, astArg.loc),
        makeNumLit(0, astArg.loc),
      ],
      loc: astArg.loc,
    };
    const bijName = freshBijName();
    const bijBinding = makeSyntheticBinding(bijName, bijAst);
    // STEP-1 patched inferSyntheticType to return 'bijection' for this
    // AST head; if the patch is in place, bijBinding.type ==='bijection'.
    // Side-channel marker for buildDerivations' bijection-construction
    // loop (derivations.ts:159-179) to forward registryName +
    // paramIRs onto binding.bijection. The marker carries the lowered
    // IR for mu and cov so the materialiser-time resolveIRToValue
    // pass evaluates them through their actual binding refs (closed
    // over the user's surface bindings, not a copy).
    (bijBinding as any).__mvnormalLowering = {
      muIR:  lowerExpr(cloneAst(muAst),  liftLowerCtx),
      covIR: lowerExpr(cloneAst(covAst), liftLowerCtx),
    };
    out.set(bijName, bijBinding);
    // Emit the synthetic iid base binding `__iid_N`. The inner
    // `Normal(mu=0, sigma=1)` MUST be pre-hoisted to an anon binding
    // so classifyIid's `resolveMeasureBaseName(ast.args[0], bindings)`
    // sees an Identifier (the only shape it admits besides
    // `lawof(<ref>)`). Same pattern as fwd/inv fn stubs above.
    const normalAst = {
      type: 'CallExpr',
      callee: makeIdent('Normal', astArg.loc),
      args: [
        { type: 'KeywordArg', name: 'mu',
          value: makeNumLit(0, astArg.loc),
          loc: astArg.loc },
        { type: 'KeywordArg', name: 'sigma',
          value: makeNumLit(1, astArg.loc),
          loc: astArg.loc },
      ],
      loc: astArg.loc,
    };
    const normalAnon = freshName();
    out.set(normalAnon, makeSyntheticBinding(normalAnon, normalAst));
    const iidAst = {
      type: 'CallExpr',
      callee: makeIdent('iid', astArg.loc),
      args: [
        makeIdent(normalAnon, astArg.loc),
        countAst,
      ],
      loc: astArg.loc,
    };
    const iidName = freshIidName();
    out.set(iidName, makeSyntheticBinding(iidName, iidAst));
    // Rewrite the original MvNormal call IN PLACE — the caller's
    // slot (e.g. the user's binding X = MvNormal(...)) now references
    // pushfwd(__bij_N, __iid_N).
    astArg.callee = makeIdent('pushfwd', astArg.loc);
    astArg.args = [
      makeIdent(bijName, astArg.loc),
      makeIdent(iidName, astArg.loc),
    ];
    if (astArg.kwargs) delete astArg.kwargs;
    return astArg;
  }

  /**
   * Static event dim D of an affine map's [D,D] scale matrix, or null when it
   * is not statically known. A literal / ref-to-literal matrix resolves via
   * `__discoveredMvNormalD`; a named shape-preserving square op like
   * `Lc = lower_cholesky(cov)` has a %dynamic result type, so D comes from
   * the op's (concrete-shape) argument instead.
   *
   * Shared by the `locscale` and `pushfwd` affine lifts — one discovery rule
   * so both spellings route on identical static evidence.
   */
  function __affineScaleD(e: any): number | null {
    const direct = __discoveredMvNormalD(e, out, 2);
    if (direct != null) return direct;
    if (e && e.type === 'Identifier') {
      const b = out.get(e.name);
      const rhs = b && b.node && b.node.value;
      if (rhs && rhs.type === 'CallExpr' && rhs.callee
          && rhs.callee.type === 'Identifier'
          && SHAPE_PRESERVING_SQUARE_OPS.has(rhs.callee.name)
          && Array.isArray(rhs.args) && rhs.args.length >= 1) {
        return __discoveredMvNormalD(rhs.args[0], out, 2);
      }
    }
    return null;
  }

  /**
   * Static iid count of an affine pushforward's base — `iid(<dist>, K)` with
   * a NumberLiteral size, or a ref chain to such. Null when K is not
   * statically an integer, in which case the caller routes on the scale
   * shape alone and the count reconciles at materialise time.
   */
  function __affineBaseIidCount(e: any): number | null {
    if (!e) return null;
    if (e.type === 'CallExpr' && e.callee && e.callee.type === 'Identifier'
        && e.callee.name === 'iid' && Array.isArray(e.args)) {
      const positional = e.args.filter((a: any) => !(a && a.type === 'KeywordArg'));
      const sizeArg = positional[1];
      if (sizeArg && sizeArg.type === 'NumberLiteral'
          && Number.isInteger(sizeArg.value)) {
        return sizeArg.value;
      }
      return null;
    }
    if (e.type === 'Identifier') {
      const b = out.get(e.name);
      const rhs = b && b.node && b.node.value;
      if (rhs) return __affineBaseIidCount(rhs);
    }
    /* c8 ignore start */
    // Defensive tail: both callers gate on the base being an iid call or a
    // ref chain to one (locscale through the analyzer pre-pass, pushfwd
    // through __affineBaseIsIid), both resolved above. This falls through
    // only for a binding with no node.value.
    return null;
    /* c8 ignore stop */
  }

  /**
   * True when an affine pushforward's base is structurally an `iid(...)` —
   * the call itself, or a ref chain to it. The affine registry's density path
   * requires a base that scores as `iid(<scalar>, D)`, and unlike `locscale`
   * (whose analyzer pre-pass rejects a non-iid base up front) the explicit
   * `pushfwd` spelling accepts any measure, so the lift must check here.
   */
  function __affineBaseIsIid(e: any): boolean {
    if (!e) return false;
    if (e.type === 'CallExpr') {
      return !!e.callee && e.callee.type === 'Identifier'
        && e.callee.name === 'iid';
    }
    if (e.type === 'Identifier') {
      const b = out.get(e.name);
      const rhs = b && b.node && b.node.value;
      return !!rhs && __affineBaseIsIid(rhs);
    }
    /* c8 ignore start */
    // Defensive tail: a measure-position argument is a call or a ref, both
    // resolved above.
    return false;
    /* c8 ignore stop */
  }

  /**
   * P3 — lower a surviving non-scalar `locscale(base, shift, scale)` to the
   * canonical affine-registry pushfwd, mirroring `inlineMvNormalLift`. The
   * analyzer pre-pass leaves vector/matrix locscale unexpanded (only iid-base
   * forms reach here; other bases are diagnosed in the pre-pass), so the call
   * arrives as `locscale(<base>, <shift [D]>, <scale [D,D]>)`.
   *
   * Differences from MvNormal:
   *   - positional args [base, shift, scale], NOT kwargs mu/cov.
   *   - the user's own base (args[0]) is the pushfwd base — we do NOT
   *     synthesize a fresh iid(Normal,0,1). (The registry density path
   *     requires the base to score as iid(<scalar>, D); the analyzer
   *     pre-pass guarantees that for the forms that reach here.)
   *   - the marker carries `scale` as L DIRECTLY (no lower_cholesky wrap) —
   *     locscale's matrix IS the affine matrix, unlike MvNormal's cov.
   *
   * Static shape gate (same conservative discovery as MvNormal): scale must
   * be a rank-2 [D,D], shift a rank-1 [D]. When D can't be determined
   * statically the call is returned unchanged (the analyzer already emits a
   * diagnostic for detected-non-scalar locscale it can't route, so an
   * unhandled locscale never survives to materialisation).
   */
  function inlineLocscaleAffineLift(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'locscale') return astArg;
    const args = astArg.args || [];
    if (args.length !== 3) return astArg;
    if (args.some((a: any) => a && a.type === 'KeywordArg')) return astArg;
    const baseAst = args[0], shiftAst = args[1], scaleAst = args[2];
    // Static shape gate (mirror MvNormal's cov/mu discovery): scale [D,D],
    // shift [D]. A non-square scale, or one whose D cannot be discovered,
    // falls to the buildDerivations safety net rather than reaching the
    // affine bijection.
    const D = __affineScaleD(scaleAst);
    if (D == null || !Number.isInteger(D) || D < 1) return astArg;
    const shiftD = __discoveredMvNormalD(shiftAst, out, 1);
    if (shiftD !== D) return astArg;
    // Square-confirm scale: literal-form gets the full square-[D,D]
    // structural check exactly as inlineMvNormalLift confirms cov; a ref to a
    // statically-square rank-2 array is confirmed via its inferredType; a ref
    // to a shape-preserving square op (lower_cholesky etc.) was already
    // square-confirmed by discoverScaleD reading the op's concrete-[D,D]
    // argument, so it passes here.
    const scaleLit = __resolveLiteralArrayRef(scaleAst, out);
    if (scaleLit) {
      if (scaleLit.elements.length !== D) return astArg;
      for (const row of scaleLit.elements) {
        if (!row || row.type !== 'ArrayLiteral'
            || row.elements.length !== D) return astArg;
      }
    } else if (scaleAst.type === 'Identifier') {
      const cb = out.get(scaleAst.name);
      const ct = cb && cb.inferredType;
      const isSquareRefType = ct && ct.kind === 'array' && ct.rank === 2
        && Number.isInteger(ct.shape[1]) && ct.shape[1] === D;
      // A shape-preserving square-op ref (lower_cholesky etc.) has a
      // %dynamic-shape inferredType but was square-confirmed via its
      // argument in discoverScaleD; admit it.
      const rhs = cb && cb.node && cb.node.value;
      const isSquareOpRef = rhs && rhs.type === 'CallExpr' && rhs.callee
        && rhs.callee.type === 'Identifier'
        && SHAPE_PRESERVING_SQUARE_OPS.has(rhs.callee.name);
      // Defensive: discoverScaleD only returns a valid D for an Identifier
      // scale via the square-op path (→ isSquareOpRef) or via a square rank-2
      // inferredType (→ isSquareRefType), so a named ref that reaches here with
      // valid D already satisfies one of the two predicates.
      /* c8 ignore start */
      if (!isSquareRefType && !isSquareOpRef) {
        return astArg;   // not a statically square [D,D] ref → fallback
      }
      /* c8 ignore stop */
    // Defensive: a non-literal, non-Identifier scaleAst cannot pass
    // discoverScaleD (it resolves D only from ArrayLiterals, which take the
    // `if (scaleLit)` branch, or from Identifiers), so D is null and the early
    // `return astArg` at the D-check fires before this else.
    /* c8 ignore start */
    } else {
      return astArg;     // scale is neither literal nor a named ref → fallback
    }
    /* c8 ignore stop */
    // Reconcile the iid BASE dimension against the scale dimension D: the
    // affine map scale@x + shift needs a length-D base. Discover the base's
    // static iid count K — base is `iid(<dist>, K)` with a NumberLiteral size
    // arg, or a 1-level ref to such (resolved through `out`). When K is
    // statically known and !== D, DO NOT route: return unchanged so the
    // unrouted `op:'locscale'` reaches the buildDerivations safety net, which
    // throws the clean locscale-tagged error rather than letting a
    // length-mismatched base reach the registry density (a cryptic
    // "value exhausted"). When K can't be determined statically, route as
    // before.
    const baseK = __affineBaseIidCount(baseAst);
    if (baseK != null && baseK !== D) return astArg;   // → buildDerivations safety net
    // Synthetic bijection stub (identical to MvNormal): fn-hole fwd/inv +
    // scalar-0 log-volume, pre-hoisted to anon bindings so derivations'
    // bijection-construction loop sees Identifiers.
    const fwdAnon = freshName();
    out.set(fwdAnon, makeSyntheticBinding(fwdAnon, makeFnHole(astArg.loc)));
    const invAnon = freshName();
    out.set(invAnon, makeSyntheticBinding(invAnon, makeFnHole(astArg.loc)));
    const bijAst = {
      type: 'CallExpr',
      callee: makeIdent('bijection', astArg.loc),
      args: [
        makeIdent(fwdAnon, astArg.loc),
        makeIdent(invAnon, astArg.loc),
        makeNumLit(0, astArg.loc),
      ],
      loc: astArg.loc,
    };
    const bijName = freshBijName();
    const bijBinding = makeSyntheticBinding(bijName, bijAst);
    // Side-channel marker for buildDerivations' bijection-construction loop
    // to forward registryName + paramIRs onto binding.bijection. L is the
    // scale matrix DIRECTLY (no lower_cholesky wrap); b is the shift vector.
    (bijBinding as any).__locscaleAffineLowering = {
      LIR: lowerExpr(cloneAst(scaleAst), liftLowerCtx),   // scale matrix directly
      bIR: lowerExpr(cloneAst(shiftAst), liftLowerCtx),   // shift vector
    };
    out.set(bijName, bijBinding);
    // Base ref: the user's own base. classifyPushfwd needs an Identifier,
    // and (the load-bearing part) the registry density path requires the
    // base to classify as `iid(<ref-to-scalar-dist>, D)` — so the iid's
    // inner distribution CALL (`Normal(0,1)`) must itself be hoisted to an
    // anon ref, exactly as MvNormal's synthetic iid base is. `liftMeasure`
    // does both: it visits the base (recursively hoisting the inner
    // distribution call to its own anon so classifyIid's
    // resolveMeasureBaseName sees an Identifier) and returns a bare ref to
    // the (possibly newly-hoisted) base binding.
    const baseRef = liftMeasure(baseAst);
    astArg.callee = makeIdent('pushfwd', astArg.loc);
    astArg.args = [makeIdent(bijName, astArg.loc), baseRef];
    if (astArg.kwargs) delete astArg.kwargs;
    return astArg;
  }

  /**
   * Route the EXPLICIT matrix-vector affine spelling
   * `pushfwd(x -> L * x + b, iid(<scalar dist>, D))` through the same
   * affine-registry pushfwd `locscale` and `MvNormal` already lower to.
   *
   * Spec §06 "Engine contract for `pushfwd` density evaluation" case 1 lists
   * "matrix-vector affine maps such as `mu + lower_cholesky(cov) * _`" among
   * the built-in bijections every conforming engine must recognize by name,
   * so this spelling owes an analytic density. Recognizing it here also
   * keeps the sample side on the registry's atom-batched forward, which
   * consumes the base's `[N, D]` atom batch whole.
   *
   * Gate — every condition static, else the call is returned unchanged:
   *   - f resolves (inline, or through a 1-level named ref) to
   *     `functionof(<body>, p = <p>)` with exactly one parameter;
   *   - body is `L * p + b` or `b + L * p`, with `p` the multiplicand's
   *     RIGHT operand (a matrix-vector product, not `p * L`);
   *   - L is a square `[D,D]` and b a `[D]` vector, by the same discovery
   *     locscale's scale and shift use;
   *   - the base's static iid count, when known, is D.
   *
   * A body that is not a matrix-vector affine map falls through to
   * `__routeScalarAffinePushfwd`, which covers the scalar-factor spellings
   * (`s * x + b` and friends) by synthesising `L = s * eye(D)`.
   *
   * A call that fails both gates keeps its plain-pushfwd lowering: sampling
   * runs f's body over the base atoms, and density reports the §06 case-3
   * static error unless the user annotated f with `bijection`.
   */
  function inlinePushfwdAffineLift(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'pushfwd') return astArg;
    const args = astArg.args || [];
    if (args.length !== 2) return astArg;
    if (args.some((a: any) => a && a.type === 'KeywordArg')) return astArg;
    const fAst = args[0], baseAst = args[1];
    const fn = __resolveFunctionofRef(fAst);
    if (!fn) return astArg;
    const affine = __matchAffineBody(fn.body, fn.param);
    // `2.0 * x + b` matches the matrix body shape too — only the scale tells
    // the two apart — so a statically-scalar scale goes to the scalar route,
    // which synthesises `L = s * eye(D)`. Everything else keeps the
    // matrix-vector gate below.
    if (!affine || __staticNonzeroScalar(affine.scaleAst)) {
      return __routeScalarAffinePushfwd(astArg, baseAst, fn);
    }
    const D = __affineScaleD(affine.scaleAst);
    if (D == null || !Number.isInteger(D) || D < 1) return astArg;
    if (__discoveredMvNormalD(affine.shiftAst, out, 1) !== D) return astArg;
    // Square-confirm the scale. __affineScaleD reads only the LEADING dim, so
    // a [D, k] scale reaches here with D rows and a wrong column count; a
    // non-endomorphism is not a bijection and must not route. A literal is
    // confirmed structurally; a named rank-2 ref is confirmed through its
    // inferredType; a named shape-preserving square op was already confirmed
    // by __affineScaleD reading the op's concrete-[D,D] argument.
    const scaleLit = __resolveLiteralArrayRef(affine.scaleAst, out);
    if (scaleLit) {
      if (scaleLit.elements.length !== D) return astArg;
      for (const row of scaleLit.elements) {
        if (!row || row.type !== 'ArrayLiteral'
            || row.elements.length !== D) return astArg;
      }
    } else if (affine.scaleAst.type === 'Identifier') {
      const sb = out.get(affine.scaleAst.name);
      const st = sb && sb.inferredType;
      const isSquareRefType = st && st.kind === 'array' && st.rank === 2
        && Number.isInteger(st.shape[1]) && st.shape[1] === D;
      const rhs = sb && sb.node && sb.node.value;
      const isSquareOpRef = rhs && rhs.type === 'CallExpr' && rhs.callee
        && rhs.callee.type === 'Identifier'
        && SHAPE_PRESERVING_SQUARE_OPS.has(rhs.callee.name);
      if (!isSquareRefType && !isSquareOpRef) return astArg;
    /* c8 ignore start */
    // Defensive: __affineScaleD resolves D only from an ArrayLiteral (which
    // takes the scaleLit branch) or an Identifier, so a scale of any other
    // shape — an inline `lower_cholesky(...)` call, say — leaves D null and
    // returns at the D check above, before this else.
    } else {
      return astArg;
    }
    /* c8 ignore stop */
    // The registry's density path scores the base as `iid(<scalar>, D)`, and
    // the affine map L@x + b needs a length-D variate. Route only an iid
    // base, and only when its statically-known count is D — a count that
    // disagrees, or a non-iid base, keeps the plain-pushfwd lowering, which
    // reports on its own terms instead of being handed a wrong variate.
    if (!__affineBaseIsIid(baseAst)) return astArg;
    const baseK = __affineBaseIidCount(baseAst);
    if (baseK != null && baseK !== D) return astArg;
    // L is the user's own matrix (no lower_cholesky wrap — unlike MvNormal,
    // which is given cov); b is the shift vector.
    return __emitAffineRegistryPushfwd(astArg, baseAst,
      affine.scaleAst, affine.shiftAst);
  }

  /**
   * Route `pushfwd(f, <iid base>)` to the affine registry, given the affine
   * map's scale matrix and shift vector as ASTs. Mutates `astArg` in place and
   * returns it.
   *
   * Shared by the matrix-vector spelling (`x -> L * x + b`, where LAst/bAst are
   * the user's own operands) and the scalar spelling (`x -> s * x + b`, where
   * LAst is a synthesised `s * eye(D)`). Both owe an analytic density under
   * spec §06 "Engine contract for `pushfwd` density evaluation" case 1, and
   * both want the registry's atom-batched forward on the sample side.
   */
  function __emitAffineRegistryPushfwd(astArg: any, baseAst: any,
                                       LAst: any, bAst: any) {
    // Synthetic bijection stub + registry marker, identical in shape to the
    // locscale and MvNormal lowerings: fn-hole forward / inverse and a
    // scalar-0 log-volume, pre-hoisted so derivations' bijection-construction
    // loop sees Identifiers.
    const fwdAnon = freshName();
    out.set(fwdAnon, makeSyntheticBinding(fwdAnon, makeFnHole(astArg.loc)));
    const invAnon = freshName();
    out.set(invAnon, makeSyntheticBinding(invAnon, makeFnHole(astArg.loc)));
    const bijAst = {
      type: 'CallExpr',
      callee: makeIdent('bijection', astArg.loc),
      args: [
        makeIdent(fwdAnon, astArg.loc),
        makeIdent(invAnon, astArg.loc),
        makeNumLit(0, astArg.loc),
      ],
      loc: astArg.loc,
    };
    const bijName = freshBijName();
    const bijBinding = makeSyntheticBinding(bijName, bijAst);
    (bijBinding as any).__pushfwdAffineLowering = {
      LIR: lowerExpr(cloneAst(LAst), liftLowerCtx),
      bIR: lowerExpr(cloneAst(bAst), liftLowerCtx),
    };
    out.set(bijName, bijBinding);
    // The registry density path requires the base to classify as
    // `iid(<ref-to-scalar-dist>, D)`, so liftMeasure hoists the iid's inner
    // distribution call to its own anon ref and hands back a bare ref.
    const baseRef = liftMeasure(baseAst);
    astArg.args = [makeIdent(bijName, astArg.loc), baseRef];
    return astArg;
  }

  /**
   * Resolve an f-position argument to `{ param, body }` when it is a
   * single-parameter `functionof(<body>, p = <p>)` — the shape both the
   * lambda syntax `x -> …` and an explicit `functionof` parse to — either
   * inline or through a 1-level named ref. Null for anything else.
   */
  function __resolveFunctionofRef(ast: any): { param: string; body: any } | null {
    let e = ast;
    if (e && e.type === 'Identifier') {
      const b = out.get(e.name);
      e = b && b.node && b.node.value;
    }
    if (!e || e.type !== 'CallExpr' || !e.callee
        || e.callee.type !== 'Identifier'
        || e.callee.name !== 'functionof') return null;
    const fargs = e.args || [];
    const positional = fargs.filter((a: any) => !(a && a.type === 'KeywordArg'));
    const kw = fargs.filter((a: any) => a && a.type === 'KeywordArg');
    if (positional.length !== 1 || kw.length !== 1) return null;
    if (!kw[0].name) return null;
    return { param: kw[0].name, body: positional[0] };
  }

  /**
   * Match an affine body `L * p + b` / `b + L * p` over the parameter named
   * `param`, returning `{ scaleAst, shiftAst }`. The parameter must be the
   * multiplicand's RIGHT operand: `L * p` is the matrix-vector product the
   * affine registry implements, while `p * L` is a different map. The
   * parameter must not appear anywhere else in the body — a second
   * occurrence makes the map non-affine in p.
   */
  function __matchAffineBody(body: any, param: string):
      { scaleAst: any; shiftAst: any } | null {
    if (!body || body.type !== 'BinaryExpr' || body.op !== '+') return null;
    for (const [mulSide, shiftSide] of [
      [body.left, body.right], [body.right, body.left]] as any[]) {
      if (!mulSide || mulSide.type !== 'BinaryExpr' || mulSide.op !== '*') continue;
      if (!__isParamRef(mulSide.right, param)) continue;
      if (__mentionsParam(mulSide.left, param)) continue;
      if (__mentionsParam(shiftSide, param)) continue;
      return { scaleAst: mulSide.left, shiftAst: shiftSide };
    }
    return null;
  }

  // A leaf naming the function's parameter. The lambda surface spells it as
  // a Placeholder; an explicit `functionof` body may name it as an ordinary
  // Identifier.
  function __isParamRef(ast: any, param: string): boolean {
    return !!ast && (ast.type === 'Placeholder' || ast.type === 'Identifier')
      && ast.name === param;
  }

  function __mentionsParam(ast: any, param: string): boolean {
    let found = false;
    (function walk(n: any) {
      if (found || !n || typeof n !== 'object') return;
      if (Array.isArray(n)) { for (const x of n) walk(x); return; }
      if (__isParamRef(n, param)) { found = true; return; }
      for (const k of Object.keys(n)) {
        if (k === 'loc') continue;
        walk(n[k]);
      }
    })(ast);
    return found;
  }

  /**
   * Route the SCALAR-affine pushforward spelling over a vector base —
   * `pushfwd(x -> s * x + b, iid(<scalar dist>, D))` and its sign- and
   * shift-free variants — through the same affine-registry pushfwd the
   * matrix-vector spelling uses, by synthesising `L = s * eye(D)` and
   * `b = zeros(D)` where the user wrote no matrix or no shift.
   *
   * Spec §06 "Engine contract for `pushfwd` density evaluation" case 1 requires
   * every conforming engine to recognize "affine maps composed from
   * `add`/`sub`/`neg`/`mul`/`divide`" by name, so this spelling owes an
   * analytic density. Without the route, density consumed one scalar from the
   * length-D variate ("scalar leaf has no entry to consume") for a rescale, and
   * threw "per-atom or vector-valued bijections not yet supported here" for a
   * vector shift.
   *
   * Gate — every condition static, else the call is returned unchanged:
   *   - the body matches `__matchScalarAffineBody`;
   *   - the base is an `iid(...)` whose count D is a statically-known integer,
   *     since D is what `eye(D)` / `zeros(D)` are synthesised from;
   *   - the scalar factor, when present, is statically scalar and nonzero (a
   *     zero factor is a singular map, not a bijection);
   *   - the shift, when present, is a statically-discovered length-D vector.
   *     §07 `add` takes "scalars or arrays of same shape", so a SCALAR shift
   *     over a vector base is not a legal map and must not route.
   */
  function __routeScalarAffinePushfwd(astArg: any, baseAst: any, fn: any) {
    const m = __matchScalarAffineBody(fn.body, fn.param);
    if (!m) return astArg;
    if (!__affineBaseIsIid(baseAst)) return astArg;
    const D = __affineBaseIidCount(baseAst);
    if (D == null || !Number.isInteger(D) || D < 1) return astArg;
    if (m.scaleAst && !__staticNonzeroScalar(m.scaleAst)) return astArg;
    if (m.shiftAst && __discoveredMvNormalD(m.shiftAst, out, 1) !== D) {
      return astArg;
    }
    const loc = astArg.loc;
    // Every spelling lowers to `L = <scalar factor> * eye(D)`, so the registry
    // sees one [D,D] shape whichever way the user wrote the rescale. The sign
    // of the linear part rides on the factor, so L never needs a matrix
    // negation and an unscaled negated map gets the literal factor -1.
    let factor: any;
    if (!m.scaleAst) {
      factor = makeNumLit(m.negate ? -1 : 1, loc);
    } else if (m.negate) {
      factor = { type: 'UnaryExpr', op: '-', operand: m.scaleAst, loc };
    } else {
      factor = m.scaleAst;
    }
    const LAst = {
      type: 'BinaryExpr', op: '*', left: factor,
      right: {
        type: 'CallExpr', callee: makeIdent('eye', loc),
        args: [makeNumLit(D, loc)], loc,
      },
      loc,
    };
    const zerosD = {
      type: 'CallExpr', callee: makeIdent('zeros', loc),
      args: [makeNumLit(D, loc)], loc,
    };
    let bAst: any;
    if (!m.shiftAst) {
      bAst = zerosD;
    } else if (m.negateShift) {
      // `zeros(D) - b`, not `-b`: materialise resolves the shift through
      // `ir-shared.resolveIRToValue`, whose `neg` (and `mul`) fold a
      // vector-valued binding ref to NaN / null, while `sub` coerces it.
      bAst = { type: 'BinaryExpr', op: '-', left: zerosD, right: m.shiftAst, loc };
    } else {
      bAst = m.shiftAst;
    }
    return __emitAffineRegistryPushfwd(astArg, baseAst, LAst, bAst);
  }

  /**
   * Match a scalar-affine body over the parameter named `param` — a scalar
   * rescale of the parameter plus a vector shift, in any of these spellings:
   *
   *   x          s * x      x * s          (each optionally negated)
   *   <above> + b      b + <above>      <above> - b      b - <above>
   *
   * Returns `{ scaleAst, negate, shiftAst, negateShift }`, where a null
   * `scaleAst` means an unscaled parameter and a null `shiftAst` a shift-free
   * map. `negate` negates the linear part (`-x`, `b - x`); `negateShift`
   * negates the shift (`x - b`). The caller owns every shape and value check —
   * this matcher only decides the SYNTACTIC form.
   *
   * The parameter must occur exactly once: `s * x + x` is not affine in x. A
   * dotted (broadcast) operator is refused — `.+` / `.*` lower to a
   * `broadcast(...)` call, a different map from the scalar-array arithmetic
   * §07 gives `add` and `mul`.
   *
   * §06 case 1 also admits `divide`, but `x / s` is NOT matched: typeinfer's
   * `divide` carries a scalars-only `(real, real) → real` signature (it is
   * absent from typeinfer's `BINARY_ARITH_OPS`), so a vector-over-scalar
   * division types as a scalar and the query is refused before the lift is
   * consulted. Routing it would emit a correct number alongside a static
   * error. Reinstating the form needs the typeinfer fix first.
   */
  function __matchScalarAffineBody(body: any, param: string): {
    scaleAst: any; negate: boolean;
    shiftAst: any; negateShift: boolean;
  } | null {
    // A null body cannot mention the parameter, so this also covers it.
    if (!__mentionsParam(body, param)) return null;
    if (body.broadcast) return null;
    // Peel the optional outer additive layer first: after this, `linear` holds
    // the (possibly negated) scalar-rescale part and the shift is settled.
    let linear = body, shiftAst: any = null, negateShift = false;
    let shiftNegatesLinear = false;
    if (body.type === 'BinaryExpr' && (body.op === '+' || body.op === '-')) {
      const leftHasParam = __mentionsParam(body.left, param);
      const rightHasParam = __mentionsParam(body.right, param);
      if (leftHasParam === rightHasParam) return null;   // both, or neither
      if (leftHasParam) {
        linear = body.left;
        shiftAst = body.right;
        negateShift = body.op === '-';
      } else {
        linear = body.right;
        shiftAst = body.left;
        // `b - <linear>` negates the linear part, not the shift.
        shiftNegatesLinear = body.op === '-';
      }
    }
    // Then peel any unary minuses off the linear part; each flips the sign.
    let negate = shiftNegatesLinear;
    while (linear && linear.type === 'UnaryExpr' && linear.op === '-') {
      if (linear.broadcast) return null;
      negate = !negate;
      linear = linear.operand;
    }
    /* c8 ignore start */
    // Defensive: the body is non-null (it mentions the parameter) and a peeled
    // `UnaryExpr` always carries an operand, so only a malformed AST lands
    // here. The guard keeps `.type` below off a null.
    if (!linear) return null;
    /* c8 ignore stop */
    // Finally the rescale itself: a bare parameter, or one scalar factor.
    if (__isParamRef(linear, param)) {
      return { scaleAst: null, negate, shiftAst, negateShift };
    }
    if (linear.type !== 'BinaryExpr' || linear.broadcast) return null;
    if (linear.op !== '*') return null;
    for (const [pSide, factor] of [
      [linear.left, linear.right], [linear.right, linear.left]] as any[]) {
      if (!__isParamRef(pSide, param)) continue;
      if (__mentionsParam(factor, param)) continue;
      return { scaleAst: factor, negate, shiftAst, negateShift };
    }
    return null;
  }

  /**
   * True when `e` is statically a scalar whose value is not zero — a literal
   * (with any number of leading unary minuses) or a ref whose inferredType is
   * `kind: 'scalar'`. A zero factor makes the affine map singular, so it is not
   * a bijection and must not reach the registry; a scalar-typed ref's value is
   * unknown here, exactly as a named matrix scale's singularity is.
   */
  function __staticNonzeroScalar(e: any): boolean {
    let inner = e;
    while (inner && inner.type === 'UnaryExpr' && inner.op === '-') {
      inner = inner.operand;
    }
    /* c8 ignore start */
    // Defensive: every caller passes a real expression node, and a peeled
    // `UnaryExpr` always carries an operand.
    if (!inner) return false;
    /* c8 ignore stop */
    if (inner.type === 'NumberLiteral') {
      return typeof inner.value === 'number' && Number.isFinite(inner.value)
        && inner.value !== 0;
    }
    if (inner.type === 'Identifier') {
      const b = out.get(inner.name);
      const t = b && b.inferredType;
      return !!t && t.kind === 'scalar';
    }
    return false;
  }

  /**
   * Rewrite `metricsum(metric, [output_axes_with_variance], expr)` to
   * `aggregate(sum, [output_axes_no_variance], wrapped_expr)` per spec
   * §04 §sec:metricsum lowering rule. Hoists a single synthetic
   * `__g_down_N = inv(<metric>)` binding when the body contains any
   * lower-variance axis occurrence. Mutates the call in place so the
   * caller's binding slot now references the aggregate.
   *
   * Rewrite recipe (spec §sec:metricsum "Lowering to `aggregate`"):
   *   - For each lower-variance occurrence `.X_` of a tensor access in
   *     `expr`, replace it with a fresh internal axis `.X__lo_N` (still
   *     indexing the stored all-upper tensor) and append a factor
   *     `__g_down[.X__lo_N, .X]` to the body. `.X` is the body's running
   *     bare-name axis after variance strip; the factor contracts the
   *     fresh internal index against it.
   *   - For each lower-variance output axis `.X_`, rename it to a fresh
   *     output upper axis `.X__up_N` and append a factor
   *     `metric[.X, .X__up_N]` to the body — this raises the
   *     lower-component result tensor to all-upper canonical storage
   *     (spec: "the actual result returned is automatically raised to
   *     all-upper canonical storage").
   *   - Strip variance markers from every other axis occurrence (upper-
   *     variance occurrences just drop the `^`).
   *   - Reduce via `sum` (per spec equivalence: `metricsum(eye(n), ...)`
   *     is equivalent to `aggregate(sum, ...)` with bare axes — and
   *     non-eye metrics are handled exactly by the inserted factors).
   *
   * The body is rewritten via deep walk; nested `aggregate(...)` and
   * `metricsum(...)` calls are skipped (their axes are in a separate
   * lexical scope per spec §05). Fresh axis names use a `__` prefix that
   * users can't write at the source level — the spec axis-name grammar
   * forbids leading underscore — so collisions with user axes are
   * structurally impossible.
   */
  function inlineMetricsumLift(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'metricsum') return astArg;
    if (!astArg.args || astArg.args.length !== 3) return astArg;
    const metricArg = astArg.args[0];
    const axesArg   = astArg.args[1];
    const bodyArg   = astArg.args[2];
    if (!metricArg) return astArg;
    if (!axesArg || axesArg.type !== 'ArrayLiteral') return astArg;
    if (!bodyArg) return astArg;
    const outputAxes = axesArg.elements || [];

    // ─── STEP 1: stable metric Identifier + runtime symmetry guard ────
    // The metric flows into the optional `inv(metric)` hoist AND into
    // the output-raise cascade. We need a bare Identifier so synthetic
    // bindings reference a single stable name.
    let metricRef: any;
    if (metricArg.type === 'Identifier') {
      metricRef = metricArg;
    } else {
      const mName = freshName();
      out.set(mName, makeSyntheticBinding(mName, metricArg));
      metricRef = makeIdent(mName, metricArg.loc);
    }
    // Spec §sec:metricsum mandates symmetric metrics. The engine can't
    // check this statically (the metric may be sampled / parameterised
    // / non-deterministic), so we wrap metricRef once with a runtime-
    // guard `_ms_check_symmetric(metric)` (engine-concepts §23). The
    // guard is a validating passthrough: returns the metric unchanged
    // on success, throws a metricsum-attributed Error on shape or
    // symmetry violation. Wrapping once at this point covers both the
    // inv() hoist AND every raise factor — those downstream uses
    // reference the checked metric Ident, so the check fires exactly
    // once per metricsum call.
    const checkedMetricCall = {
      type: 'CallExpr',
      callee: makeIdent('_ms_check_symmetric', astArg.loc),
      args: [makeIdent(metricRef.name, metricRef.loc)],
      loc: astArg.loc,
    };
    const checkedMetricName = freshName();
    out.set(checkedMetricName,
      makeSyntheticBinding(checkedMetricName, checkedMetricCall));
    metricRef = makeIdent(checkedMetricName, astArg.loc);

    // ─── STEP 2: probe body for any lower-variance axis ───────────
    // Decides whether `__g_down = inv(metric)` is needed. Stops at
    // nested aggregate / metricsum boundaries (separate scope per
    // spec §05).
    let needsInverseMetric = false;
    function probeLower(n: any) {
      if (!n || typeof n !== 'object' || needsInverseMetric) return;
      if (Array.isArray(n)) { for (const c of n) probeLower(c); return; }
      if (n.type === 'AxisRef' && n.variance === 'lower') {
        needsInverseMetric = true;
        return;
      }
      if (n.type === 'CallExpr' && n.callee && n.callee.type === 'Identifier'
          && (n.callee.name === 'aggregate' || n.callee.name === 'metricsum')) {
        return;
      }
      for (const k of Object.keys(n)) {
        if (k === 'loc') continue;
        probeLower(n[k]);
      }
    }
    probeLower(bodyArg);

    // ─── STEP 3: hoist `__g_down = inv(metric)` (once per metricsum) ─
    let gDownIdent: any = null;
    if (needsInverseMetric) {
      const gDownName = freshGDownName();
      const invCall = {
        type: 'CallExpr',
        callee: makeIdent('inv', astArg.loc),
        args: [makeIdent(metricRef.name, metricRef.loc)],
        loc: astArg.loc,
      };
      out.set(gDownName, makeSyntheticBinding(gDownName, invCall));
      gDownIdent = makeIdent(gDownName, astArg.loc);
    }

    // ─── STEP 4/5: per-(source, variance-pattern) mixed-variance cascade ─
    // For each body IndexExpr `T[axisRefs...]` with at least one lower
    // variance axis, emit a CASCADE of K aggregate bindings (K = number of
    // lower axes), one per lower position (left-to-right). Each step
    // contracts one axis with __g_down via a clean matmul-shaped
    // aggregate that the dissolver's matmul-family pattern matcher
    // (aggregate-patterns.classifyMatmulBody) recognises and specialises
    // into a `mul(prev, __g_down)` IR for rank-2 / rank-1 cases.
    //
    // CSE: a single Map<sourceName + variance-pattern, finalMixedIdent>
    // ensures two accesses of the same tensor with the same pattern share
    // a single cascade. Engine-concepts §23 describes the design.
    const mixedCSE = new Map<string, any>();

    function buildCSEKey(sourceName: string, indexList: any[]): string {
      const partFragments: string[] = [];
      for (const idx of indexList) {
        if (idx && idx.type === 'AxisRef') {
          partFragments.push(idx.name + '|' + (idx.variance || 'bare'));
        } else {
          // Non-axis index slot (literal, expression). Include a stable
          // tag so two accesses with different non-axis indices don't CSE.
          partFragments.push('@' + (idx && idx.type || '?'));
        }
      }
      return sourceName + '||' + partFragments.join(',');
    }

    function emitMixedCascade(sourceIdent: any, indexList: any[], anchorLoc: any): any {
      const cseKey = buildCSEKey(sourceIdent.name, indexList);
      const cached = mixedCSE.get(cseKey);
      if (cached) return cached;
      let prevIdent: any = sourceIdent;
      // Cascade one lower position at a time (left-to-right). At each
      // step the contracted axis is `.__ms_lo_N` (internal); the output
      // axis takes the user-given name (.j, .rho, etc.) so the OUTER
      // aggregate body indexes the mixed binding with the user's axis
      // names directly.
      for (let p = 0; p < indexList.length; p++) {
        const ax = indexList[p];
        if (!ax || ax.type !== 'AxisRef' || ax.variance !== 'lower') continue;
        const internalAxis = freshLowerInternalAxisName();
        // Aggregate output axes: full index list with variance markers
        // stripped (and any earlier lower-variance positions already
        // bare from previous cascade steps; their names persist via
        // identity-on-rebuild here).
        const stepOutputAxes = indexList.map((idx: any) => {
          if (idx && idx.type === 'AxisRef') {
            return makeAxisRef(idx.name, idx.loc || anchorLoc);
          }
          return idx;
        });
        // Body: prev[<indices with internal at position p>] * __g_down[.internal, .name_at_p]
        const stepBodyIndices = indexList.map((idx: any, k: number) => {
          if (k === p) return makeAxisRef(internalAxis, ax.loc || anchorLoc);
          if (idx && idx.type === 'AxisRef') {
            return makeAxisRef(idx.name, idx.loc || anchorLoc);
          }
          return idx;
        });
        const prevAccess = {
          type: 'IndexExpr',
          object: makeIdent(prevIdent.name, anchorLoc),
          indices: stepBodyIndices,
          loc: anchorLoc,
          indexOp: 'get',
        };
        const gDownFactor = {
          type: 'IndexExpr',
          object: makeIdent(gDownIdent!.name, anchorLoc),
          indices: [
            makeAxisRef(internalAxis, anchorLoc),
            makeAxisRef(ax.name, anchorLoc),
          ],
          loc: anchorLoc,
          indexOp: 'get',
        };
        const stepBody = {
          type: 'BinaryExpr',
          op: '*',
          left: prevAccess,
          right: gDownFactor,
          loc: anchorLoc,
        };
        const stepAggregate = {
          type: 'CallExpr',
          callee: makeIdent('aggregate', anchorLoc),
          args: [
            makeIdent('sum', anchorLoc),
            { type: 'ArrayLiteral', elements: stepOutputAxes, loc: anchorLoc },
            stepBody,
          ],
          loc: anchorLoc,
        };
        const stepName = freshMixedName();
        out.set(stepName, makeSyntheticBinding(stepName, stepAggregate));
        prevIdent = makeIdent(stepName, anchorLoc);
      }
      mixedCSE.set(cseKey, prevIdent);
      return prevIdent;
    }

    // Hoist a composed (non-Identifier) tensor base into its own
    // synthetic binding so emitMixedCascade has a name to key on.
    // Structurally identical bases share one binding, which also keeps
    // the cascade CSE effective across repeated accesses.
    const baseCSE = new Map<string, any>();
    function containsAxisRef(n: any): boolean {
      if (!n || typeof n !== 'object') return false;
      if (Array.isArray(n)) return n.some(containsAxisRef);
      if (n.type === 'AxisRef') return true;
      for (const k of Object.keys(n)) {
        if (k === 'loc') continue;
        if (containsAxisRef(n[k])) return true;
      }
      return false;
    }
    function baseStructuralKey(n: any): string {
      return JSON.stringify(n, (k, v) => (k === 'loc' ? undefined : v));
    }
    function hoistComposedBase(objectNode: any, anchorLoc: any): any {
      const key = baseStructuralKey(objectNode);
      const cached = baseCSE.get(key);
      if (cached) return cached;
      const name = freshMsBaseName();
      out.set(name, makeSyntheticBinding(name, objectNode));
      const ident = makeIdent(name, anchorLoc);
      baseCSE.set(key, ident);
      return ident;
    }

    // ─── STEP 6: body rewrite ──────────────────────────────────────
    // Deep-walk the body, cloning as we go (the source AST stays
    // untouched). For each IndexExpr with any lower-variance index,
    // emit (or look up) a mixed cascade and replace the IndexExpr with
    // `__ms_mixed_N[bare_axes]`. Other AxisRef occurrences lose their
    // variance markers. Nested aggregate/metricsum subtrees pass
    // through verbatim.
    function rewriteBody(n: any): any {
      if (!n) return n;
      if (Array.isArray(n)) return n.map(rewriteBody);
      if (typeof n !== 'object') return n;
      // IndexExpr with a lower-variance axis. A non-Identifier source
      // is hoisted first; skipping the cascade instead would silently
      // drop the inv(metric) contraction the spec mandates.
      if (n.type === 'IndexExpr' && n.object && Array.isArray(n.indices)) {
        let hasLower = false;
        for (const idx of n.indices) {
          if (idx && idx.type === 'AxisRef' && idx.variance === 'lower') {
            hasLower = true; break;
          }
        }
        if (hasLower && gDownIdent) {
          let sourceIdent = n.object;
          if (sourceIdent.type !== 'Identifier') {
            // A base mentioning an axis of this metricsum cannot leave
            // the aggregate scope, so it cannot be hoisted.
            if (containsAxisRef(sourceIdent)) {
              throw new Error(
                'metricsum: a tensor base indexed with a lower-variance axis '
                + 'may not itself mention an axis name');
            }
            sourceIdent = hoistComposedBase(cloneAst(sourceIdent), n.loc);
          }
          const mixedIdent = emitMixedCascade(sourceIdent, n.indices, n.loc);
          const bareIndices = n.indices.map((idx: any) => {
            if (idx && idx.type === 'AxisRef') {
              return makeAxisRef(idx.name, idx.loc);
            }
            return rewriteBody(idx);
          });
          return {
            type: 'IndexExpr',
            object: makeIdent(mixedIdent.name, n.loc),
            indices: bareIndices,
            loc: n.loc,
            indexOp: n.indexOp || 'get',
          };
        }
      }
      // Standalone AxisRef (not handled by the IndexExpr case above):
      // strip variance marker. The standalone case shows up in output-
      // axis arrays handled outside this walker; here it's defensive.
      if (n.type === 'AxisRef') {
        return makeAxisRef(n.name, n.loc);
      }
      // Nested aggregate / metricsum: pass through without descending.
      if (n.type === 'CallExpr' && n.callee && n.callee.type === 'Identifier'
          && (n.callee.name === 'aggregate' || n.callee.name === 'metricsum')) {
        return n;
      }
      // Default: deep-clone with recursive rewrite.
      const rebuilt: any = {};
      for (const k of Object.keys(n)) {
        if (k === 'loc') { rebuilt[k] = n[k]; continue; }
        rebuilt[k] = rewriteBody(n[k]);
      }
      return rebuilt;
    }
    const rewrittenBody = rewriteBody(bodyArg);

    // ─── STEP 7: partition output axes for the main aggregate ──────
    // Strip variance markers from every output axis. The bare names
    // stay on the main aggregate's output. Lower-variance axes are
    // recorded for the post-main raise cascade — the raise cascade
    // renames each lower-variance output axis (one at a time) via a
    // metric contraction.
    const mainOutputAxes: any[] = [];
    const lowerOutputs: { origName: string; loc: any }[] = [];
    for (const ax of outputAxes) {
      if (!ax || ax.type !== 'AxisRef') {
        mainOutputAxes.push(ax);
        continue;
      }
      mainOutputAxes.push(makeAxisRef(ax.name, ax.loc));
      if (ax.variance === 'lower') {
        lowerOutputs.push({ origName: ax.name, loc: ax.loc });
      }
    }

    // ─── STEP 8: build the main aggregate ─────────────────────────
    // Body has only bare-axis indexing (no metric factors interspersed).
    // Output axes are the user's axis names with variance markers
    // stripped. When no lower-variance output axis exists, we mutate the
    // metricsum call IN PLACE to the aggregate (same return shape as
    // Form-A's degenerate path; the caller-side handling is unchanged
    // for this case).
    if (lowerOutputs.length === 0) {
      astArg.callee = makeIdent('aggregate', astArg.loc);
      astArg.args = [
        makeIdent('sum', astArg.loc),
        { type: 'ArrayLiteral', elements: mainOutputAxes, loc: axesArg.loc },
        rewrittenBody,
      ];
      if (astArg.kwargs) delete astArg.kwargs;
      return astArg;
    }

    // ─── STEP 9: hoist main aggregate as a hidden binding, emit raises ──
    // When at least one output axis is lower-variance, the main
    // aggregate produces the body's mixed-variance components; a per-
    // axis raise cascade then transforms those into the all-upper
    // canonical storage layout (spec §sec:metricsum: "the actual result
    // returned by metricsum is automatically raised to all-upper
    // canonical storage"). The user's binding RHS gets redirected to
    // the final raise step's Identifier — this matches the
    // inlineMvNormalLift pattern (return a replacement node from the
    // inline-lift; caller's `astArg = inlineXxxLift(astArg)` accepts
    // the new value).
    const mainAggregate = {
      type: 'CallExpr',
      callee: makeIdent('aggregate', astArg.loc),
      args: [
        makeIdent('sum', astArg.loc),
        { type: 'ArrayLiteral', elements: mainOutputAxes, loc: axesArg.loc },
        rewrittenBody,
      ],
      loc: astArg.loc,
    };
    const mainName = freshMainName();
    out.set(mainName, makeSyntheticBinding(mainName, mainAggregate));

    // Raise cascade — one synthetic aggregate per lower-variance output.
    // Each step renames a single lower-variance axis (.X) to a fresh
    // upper axis (.X_up_N) by contracting with `metric[.X, .X_up_N]`.
    let prevIdent: any = makeIdent(mainName, astArg.loc);
    let prevAxes: { name: string; loc: any }[] = mainOutputAxes
      .filter((ax: any) => ax && ax.type === 'AxisRef')
      .map((ax: any) => ({ name: ax.name, loc: ax.loc }));

    for (const { origName, loc } of lowerOutputs) {
      const freshUp = freshUpperOutputAxisName();
      // Raise step output axes: prev's axes with origName replaced by
      // freshUp (the new upper-storage axis).
      const raiseOutputAxes = prevAxes.map((ax) =>
        ax.name === origName
          ? makeAxisRef(freshUp, ax.loc || loc)
          : makeAxisRef(ax.name, ax.loc || loc)
      );
      // Raise body: prev[<prev's axes verbatim>] * metric[.origName, .freshUp]
      const prevAccess = {
        type: 'IndexExpr',
        object: makeIdent(prevIdent.name, loc),
        indices: prevAxes.map((ax) => makeAxisRef(ax.name, ax.loc || loc)),
        loc: loc,
        indexOp: 'get',
      };
      const metricFactor = {
        type: 'IndexExpr',
        object: makeIdent(metricRef.name, loc),
        indices: [
          makeAxisRef(origName, loc),
          makeAxisRef(freshUp, loc),
        ],
        loc: loc,
        indexOp: 'get',
      };
      const raiseBody = {
        type: 'BinaryExpr',
        op: '*',
        left: prevAccess,
        right: metricFactor,
        loc: loc,
      };
      const raiseAggregate = {
        type: 'CallExpr',
        callee: makeIdent('aggregate', loc),
        args: [
          makeIdent('sum', loc),
          { type: 'ArrayLiteral', elements: raiseOutputAxes, loc: loc },
          raiseBody,
        ],
        loc: loc,
      };
      const raiseName = freshRaisedName();
      out.set(raiseName, makeSyntheticBinding(raiseName, raiseAggregate));
      prevIdent = makeIdent(raiseName, loc);
      prevAxes = raiseOutputAxes.map((ax: any) => ({ name: ax.name, loc: ax.loc }));
    }

    // Return the final raise Ident — the caller redirects the user's
    // binding RHS to this.
    return prevIdent;
  }

  // Helpers used by inlineMvNormalLift (small AST builders + literal
  // array resolver). Scoped inside liftInlineSubexpressions so they
  // share `counter`, `out`, `makeIdent`, etc.
  function makeNumLit(value: any, loc: any) {
    return {
      type: 'NumberLiteral',
      value: +value,
      raw: String(value),
      loc: loc || null,
    };
  }
  function makeHole(loc: any) {
    return { type: 'Hole', loc: loc || null };
  }
  function makeFnHole(loc: any) {
    // `fn(_)` — identity-stub used as the bijection's forward / inverse.
    return {
      type: 'CallExpr',
      callee: makeIdent('fn', loc),
      args: [makeHole(loc)],
      loc: loc || null,
    };
  }
  function __resolveLiteralArrayRef(ast: any, bindings: any): any {
    // Returns the ArrayLiteral AST node when `ast` is one OR is a
    // direct ref to a binding whose RHS is one. Otherwise null. No
    // deep resolution — non-literal bindings fall through to the
    // matMvNormal safety path.
    //
    // Also unwraps one level of `rowstack(<ArrayLiteral>)`: spec §03
    // requires the user to wrap matrix literals in `rowstack(...)` to
    // commit a row-major interpretation; the lift treats the wrapped
    // form as semantically identical to a bare ArrayLiteral for the
    // purpose of resolving the per-row literal structure (the
    // resulting Value is the same row-major flat storage either way).
    if (!ast) return null;
    if (ast.type === 'ArrayLiteral') return ast;
    if (ast.type === 'CallExpr'
        && ast.callee && ast.callee.type === 'Identifier'
        && ast.callee.name === 'rowstack'
        && Array.isArray(ast.args) && ast.args.length === 1
        && ast.args[0] && ast.args[0].type === 'ArrayLiteral') {
      return ast.args[0];
    }
    if (ast.type === 'Identifier') {
      const b = bindings.get(ast.name);
      if (b && b.node && b.node.value
          && b.node.value.type === 'ArrayLiteral') {
        return b.node.value;
      }
      // Note: a ref to a `rowstack(<ArrayLiteral>)` binding is NOT
      // unwrapped here — this helper only resolves per-row LITERAL
      // structure. Ref-form rowstack is instead covered by
      // `__discoveredMvNormalD` path (b) via the binding's inferredType
      // (`array(2,[D,D])`), so 5f-1 lowers it through the registry
      // without needing literal-structure access.
    }
    return null;
  }

  // 5f-1: discover the static event dim D of an MvNormal mu/cov argument,
  // requiring the argument to have rank `expectedRank` (1 for mu, 2 for
  // cov). Order:
  //   (a) literal / ref-to-literal / inline-rowstack-literal — via
  //       `__resolveLiteralArrayRef`; D = number of (outer) elements,
  //       BUT only when the literal's nesting matches expectedRank (a
  //       rank-1 mu must have scalar elements; a rank-2 cov must have
  //       ArrayLiteral rows). This rejects a matrix-form mean
  //       `rowstack([[2.0]])` (a [1,1] matrix where a [1] vector is
  //       wanted), which would otherwise lower to a [1,1] param the
  //       density guard later misreads as atom-batched (5f Issue 1).
  //   (b) a named Identifier ref whose binding carries a statically-known
  //       inferredType array of EXACTLY rank `expectedRank` — D =
  //       shape[0] (the leading dim is the event dim for both a [D]
  //       vector and a [D,D] matrix). Requires shape[0] a concrete
  //       integer, NOT '%dynamic'.
  // Returns null when D is not statically known — the gate then falls
  // through to the matMvNormal terminal materialiser. inferredType is
  // populated on every binding at lift time (analyzer copies it from the
  // typeinfer pass onto the bindings buildDerivations hands to lift).
  function __discoveredMvNormalD(ast: any, bindings: any,
                                 expectedRank: number): number | null {
    if (!ast) return null;
    const lit = __resolveLiteralArrayRef(ast, bindings);       // path (a)
    if (lit && lit.type === 'ArrayLiteral') {
      const firstElemIsArray = lit.elements.length > 0
        && lit.elements[0] && lit.elements[0].type === 'ArrayLiteral';
      // rank-1 wants scalar elements; rank-2 wants ArrayLiteral rows.
      if (expectedRank === 1 && firstElemIsArray) return null;
      if (expectedRank === 2 && !firstElemIsArray) return null;
      return lit.elements.length;
    }
    if (ast.type === 'Identifier') {                            // path (b)
      const b = bindings.get(ast.name);
      const t = b && b.inferredType;
      if (t && t.kind === 'array' && t.rank === expectedRank
          && Array.isArray(t.shape) && Number.isInteger(t.shape[0])) {
        return t.shape[0];
      }
    }
    return null;
  }

  // 5h-A: true when the AST contains an Identifier that names neither a
  // module binding nor a builtin — a reification formal / boundary
  // placeholder (`m` in `kernelof(…, m = m)`, `_x_`-class names). Such an
  // expression cannot be hoisted to a module-level anon (the name only
  // has per-call-site meaning inside the enclosing reified scope).
  function __containsFormalRef(ast: any, bindings: any): boolean {
    const builtins = require('./builtins.ts');
    let found = false;
    (function walk(n: any) {
      if (found || !n || typeof n !== 'object') return;
      if (Array.isArray(n)) { for (const x of n) walk(x); return; }
      if (n.type === 'Identifier') {
        if (!bindings.has(n.name) && !builtins.ALL_KNOWN.has(n.name)) found = true;
        return;
      }
      // Callee identifiers are op names, not value refs — walk args only.
      if (n.type === 'CallExpr') { walk(n.args); return; }
      for (const k in n) { if (k !== 'loc' && k !== 'callee') walk(n[k]); }
    })(ast);
    return found;
  }

  // 5h-A: POSITIVE rank-conflict detector — true only when the argument's
  // structure is statically VISIBLE and contradicts `expectedRank`
  // (a nested ArrayLiteral where rank-1 is wanted; a ref whose
  // inferredType is an array of the wrong rank). "Unknown" (inline
  // expressions, dynamic/deferred types) is NOT a conflict — the
  // dynamic-D path lowers those and lets the materialise-time shape
  // checks own validity. __discoveredMvNormalD conflates conflict with
  // unknown (both return null), so the dynamic gate needs this split.
  function __mvNormalRankConflict(ast: any, bindings: any,
                                  expectedRank: number): boolean {
    if (!ast) return false;
    const lit = __resolveLiteralArrayRef(ast, bindings);
    if (lit && lit.type === 'ArrayLiteral') {
      const firstElemIsArray = lit.elements.length > 0
        && lit.elements[0] && lit.elements[0].type === 'ArrayLiteral';
      if (expectedRank === 1 && firstElemIsArray) return true;
      if (expectedRank === 2 && !firstElemIsArray) return true;
      return false;
    }
    if (ast.type === 'Identifier') {
      const b = bindings.get(ast.name);
      const t = b && b.inferredType;
      if (t && t.kind === 'array' && t.rank !== expectedRank) return true;
    }
    return false;
  }

  /**
   * Lift the kernel arg of `likelihoodof(K, obs)` to a named binding
   * when it's an inline function expression (fn / functionof /
   * kernelof / fchain). classifyLikelihoodof + classifyBayesupdate
   * both require K to be a self-ref to a function-like binding so
   * they can walk the binding graph. The general `liftValue` /
   * `liftMeasure` paths refuse to hoist function-shaped expressions
   * because they typically contain placeholders / holes that can't
   * escape the enclosing scope; the op-specific lift here knows the
   * kernel-arg position safely contains them and hoists the whole
   * function expression as a unit.
   */
  function inlineLikelihoodofLift(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    if (astArg.callee.name !== 'likelihoodof') return astArg;
    if (!astArg.args || astArg.args.length !== 2) return astArg;
    let kArg = astArg.args[0];
    if (kArg.type === 'KeywordArg') return astArg;
    if (kArg.type === 'Identifier') return astArg;
    visit(kArg);
    if (kArg.type === 'Identifier') {
      astArg.args[0] = kArg;
      return astArg;
    }
    const n = freshName();
    out.set(n, makeSyntheticBinding(n, kArg));
    astArg.args[0] = makeIdent(n, astArg.loc);
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
  // Whether an AST arg in relabel's first position denotes a measure (or
  // kernel) rather than a value. A direct call to a measure-producing op /
  // distribution, or an Identifier whose bound RHS is one (peeling a nested
  // relabel). Conservative: anything else is treated as a value, preserving
  // the existing array/record/scalar → record rewrite.
  function isMeasureLikeArg(node: any): boolean {
    if (!node) return false;
    if (node.type === 'CallExpr' && node.callee && node.callee.type === 'Identifier') {
      const op = node.callee.name;
      if (op === 'relabel' && Array.isArray(node.args) && node.args.length >= 1) {
        return isMeasureLikeArg(node.args[0]);
      }
      return MEASURE_PRODUCING.has(op);
    }
    if (node.type === 'Identifier') {
      const target = out.get(node.name);
      const targetAst = target && (target.effectiveValue || (target.node && target.node.value));
      if (targetAst && targetAst !== node) return isMeasureLikeArg(targetAst);
    }
    return false;
  }

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

    // relabel of a MEASURE (or kernel) is the output-side axis renaming of
    // spec §04 — it stays a `relabel(...)` measure node so the density
    // walkers peel it transparently (density.walkAcc / mat-density
    // asScalarFactor). Only relabel of a VALUE rewrites to `record(...)`.
    // Without this guard the single-name scalar-wrap below turns
    // `relabel(Normal(…), ["x"])` into `record(x = Normal(…))`, which the
    // iid/joint density walker then treats as a named-field joint and tries
    // to consume a field `x` from each scalar observation — wrong, and the
    // source of "cannot consume named field 'x' from non-record value".
    if (isMeasureLikeArg(argExpr)) return astArg;

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

  // A reified callable that (directly or mutually) references itself can
  // NOT be beta-reduced to a fixed point — inlining its body re-introduces
  // a call to itself, so the inlineUserCall fixpoint loop would spin
  // forever. FlatPPL is a static flat graph; recursion is not a valid
  // program (the analyzer flags the cycle separately). inlineOnce refuses
  // to expand a recursive callable, leaving the call un-inlined so the
  // fixpoint converges and the pipeline (lift → derivations → DAG viz)
  // stays responsive. Memoised: the recursion property of a callable is
  // stable across the lift. A reachability walk from `fnName` through every
  // referenced binding (visited-guarded) is recursive iff it reaches
  // `fnName` again — catches direct and mutual recursion. For an acyclic
  // (valid) program no callable is ever reachable from itself, so this
  // never refuses a legitimate inline.
  function isRecursiveCallable(fnName: string): boolean {
    // Cache on the (hoisted) function object — fresh per liftInline-
    // Subexpressions call, and free of the temporal-dead-zone a textually-
    // late `const` would hit (the visit pass calls inlineOnce before this
    // point in the function body is reached).
    const cache: Map<string, boolean> = (isRecursiveCallable as any)._cache
      || ((isRecursiveCallable as any)._cache = new Map<string, boolean>());
    if (cache.has(fnName)) return cache.get(fnName)!;
    const visited = new Set<string>();
    const stack = [fnName];
    let recursive = false;
    while (stack.length > 0 && !recursive) {
      const cur = stack.pop()!;
      if (visited.has(cur)) continue;
      visited.add(cur);
      const b = out.get(cur) || bindings.get(cur);
      const v = b && (b.effectiveValue || (b.node && b.node.value));
      if (!v) continue;
      const refs: string[] = [];
      (function collect(node: any) {
        if (node == null || typeof node !== 'object') return;
        if (Array.isArray(node)) { for (const c of node) collect(c); return; }
        if (node.type === 'Identifier' && typeof node.name === 'string') refs.push(node.name);
        for (const k in node) collect(node[k]);
      })(v);
      for (const rn of refs) {
        if (rn === fnName) { recursive = true; break; }
        if (!visited.has(rn)) stack.push(rn);
      }
    }
    cache.set(fnName, recursive);
    return recursive;
  }

  function inlineOnce(astArg: any) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    // Expression-headed call (spec §11 / §05 Postfix Call*): the callee
    // is an INLINE reification applied directly — `functionof(e, p=a)(2.5)`,
    // `(x -> 2*x)(3.0)` (the parser desugars lambdas to functionof
    // CallExprs), `fn(2*_)(3.0)`. Treat the callee AST as fnAst with a
    // synthetic binding-type from its head — the entire named-callable
    // substitution machinery below applies unchanged (no name ⇒ no
    // recursion possible). Other expression callees (chained `f(x)(y)`
    // where f(x) yields a callable) are left un-inlined — they lower to
    // the callee-form IR and surface a clean classification refusal.
    if (astArg.callee && astArg.callee.type === 'CallExpr'
        && astArg.callee.callee && astArg.callee.callee.type === 'Identifier'
        && (astArg.callee.callee.name === 'functionof'
          || astArg.callee.callee.name === 'kernelof'
          || astArg.callee.callee.name === 'fn')) {
      return _inlineApplication(astArg,
        { type: astArg.callee.callee.name }, astArg.callee);
    }
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    // Callable ALIAS (spec §04 "Aliasing is just assignment"): resolve the
    // call head through any chain of bare aliases (`g = f; b = g(x)`) to the
    // canonical callable, then recurse so the bijection / inlining logic
    // below operates on the REAL callable. Without this a call THROUGH an
    // alias never inlines (the alias's binding-type is the fallback 'call')
    // and the binding gets no derivation; the cross-module idiom
    // `f_b = common.f_b` reduces to this after linking. `resolveCallableAlias`
    // is the ONE alias-chain follow (shared with signatureOf), cycle-guarded.
    const fnName = resolveCallableAlias(astArg.callee.name, out);
    if (fnName !== astArg.callee.name) {
      return inlineOnce(Object.assign({}, astArg, { callee: makeIdent(fnName, astArg.loc) }));
    }
    // Use `out`, not `bindings`, so synthesized anon bindings created
    // during the lift pass are visible to inlineOnce too.
    const fnBinding = out.get(fnName);
    if (!fnBinding) return astArg;
    // Recursive callable → can't beta-reduce; leave the call un-inlined so
    // the fixpoint converges (see isRecursiveCallable above).
    if (isRecursiveCallable(fnName)) return astArg;
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
    // Applying a kernel `joint` (spec §06 "Uniform kernel extension" + the
    // `joint` entry: "a kernel that fans a single input out to all component
    // kernels"). `joint` is not a reified callable, so the machinery above does
    // not reach it — rewrite the fan-out into the equivalent single
    // reification first, then inline THAT. One reification means ONE closure
    // synthesis, which is what makes the retained semantics fall out; see
    // `_jointFanoutAsFunctionof`.
    if (fnBinding.type === 'call') {
      const synth = _jointFanoutAsFunctionof(fnBinding);
      if (synth) return _inlineApplication(astArg, { type: 'functionof' }, synth);
      return astArg;
    }
    if (fnBinding.type !== 'functionof' && fnBinding.type !== 'kernelof'
        && fnBinding.type !== 'fn') {
      return astArg;
    }
    // Use effectiveValue when the analyzer synthesised one (e.g. a
    // disintegrate-derived kernel/prior pair, where the literal RHS
    // is `disintegrate(selector, M)` but the effective form is the
    // synthesised `kernelof(body, kw=...)`). Without this, inlineOnce
    // would grab `disintegrate`'s first arg — the selector array
    // literal — and treat it as the kernel body, producing wildly
    // wrong inlinings (the restrict-expand complement-route bug).
    const fnAst = (fnBinding.effectiveValue
      || (fnBinding.node && fnBinding.node.value));
    return _inlineApplication(astArg, fnBinding, fnAst);
  }

  // ── applied kernel `joint`: the commuting identity ────────────────────
  //
  // `joint(p = kernelof(a1, z = z), q = kernelof(a2, z = z))` is a kernel, and
  // §06's ancestry rule governs its output measure at each input point:
  // components whose traces share a stochastic node yield the correlated
  // record law, components sharing none yield the product. So the fan-out
  // COMMUTES with the reification — it equals
  // `kernelof(record(p = a1, q = a2), z = z)` — and applying it must synthesize
  // the shared sub-DAG ONCE.
  //
  // Rewriting the application per component instead (`joint(p = K1(v), q =
  // K2(v))`) would give each component its own closure copy, hence its own
  // `u`, hence the PRODUCT of the marginals — a different measure (the probe
  // scores −3.0310242469693 instead of −3.3871832107434). So the rewrite hoists
  // the boundary to a single enclosing `functionof` over a body that is the
  // measure-`joint` of the components' laws, and hands that to
  // `_inlineApplication`. The resulting `joint(p = lawof(â1), q = lawof(â2))`
  // over one synthesized `û` is the ordinary measure-joint shape, so the
  // ancestry rule, the singular-joint refusal and the sampler all apply to it
  // unchanged.
  //
  // The identity is LICENSED ONLY WHEN THE MERGED BOUNDARY SET IS STILL AN
  // ALL-OR-NONE COMPLETE CUT FOR EVERY COMPONENT — the premise the maths doc
  // states as "(each all-or-none complete for its own component)", resting on
  // §04 "Specifying reification boundaries". It holds for matched boundary sets,
  // for disjoint non-nested ones, and for a measure component (empty set). It
  // FAILS when a boundary node of component i is an INTERIOR ancestor of
  // component j: merging then cuts a path j deliberately marginalizes over, so
  // j's own input goes dead and j is scored at the wrong law. That shape needs
  // no ruling to be wrong — it violates the marginalization floor (the i-th
  // marginal of the composed law must equal K_i at the same input), which both
  // candidate conventions satisfy — so it must refuse, not answer.
  //
  // Returns null — leaving the call un-inlined, so the classifier refuses
  // rather than lowering something else — for every shape this rewrite cannot
  // resolve: a non-`joint` binding, an all-measure `joint` (not callable), a
  // component whose reification boundary is a placeholder or a complex
  // expression, an inline (unlifted) reification component, a nested boundary
  // scope (above), and a boundary naming conflict in either direction.
  function _jointFanoutAsFunctionof(fnBinding: any) {
    const jAst = fnBinding.node && fnBinding.node.value;
    if (!jAst || jAst.type !== 'CallExpr' || !jAst.callee
        || jAst.callee.type !== 'Identifier' || jAst.callee.name !== 'joint') {
      return null;
    }
    const outerOfSurface = new Map<string, string>();
    const surfaceOfOuter = new Map<string, string>();
    const newArgs: any[] = [];
    const kernelReps: any[] = [];
    const measureReps: any[] = [];
    for (const a of (jAst.args || [])) {
      const isKw = a.type === 'KeywordArg';
      const rep = _jointComponentAsMeasure(isKw ? a.value : a, isKw ? a.name : undefined);
      if (!rep) return null;
      if (!rep.boundaries) measureReps.push(rep);
      if (rep.boundaries) {
        kernelReps.push(rep);
        for (const [surface, outer] of rep.boundaries) {
          const prevOuter = outerOfSurface.get(surface);
          const prevSurface = surfaceOfOuter.get(outer);
          // One input name must denote one node, and one node must be reached
          // through one input name. The first is §06's boundary-naming clause.
          // The second rejects one LEGAL shape too — trace-disjoint components
          // binding one node under two names — because a single substitution
          // map cannot give one node two fed values; recorded as a gap in
          // flatppl-dev/TODO-flatppl-js.md.
          if (prevOuter !== undefined && prevOuter !== outer) return null;
          if (prevSurface !== undefined && prevSurface !== surface) return null;
          outerOfSurface.set(surface, outer);
          surfaceOfOuter.set(outer, surface);
        }
      }
      newArgs.push(isKw
        ? { type: 'KeywordArg', name: a.name, value: rep.measureAst, loc: a.loc }
        : rep.measureAst);
    }
    if (kernelReps.length === 0) return null;
    // The complete-cut check.
    for (const rj of kernelReps) {
      const interior = _reachableAncestors(rj.measureAst,
        new Set(rj.boundaries.values()));
      for (const ri of kernelReps) {
        if (ri === rj) continue;
        for (const outer of ri.boundaries.values()) {
          if (interior.has(outer)) return null;
        }
      }
    }
    // A measure component binds nothing (§06's nullary case), so its coordinate
    // reads the AMBIENT parameter — never the fed input. The hoist has one
    // substitution map for the whole body, so a measure component that reaches a
    // hoisted boundary node would be rewritten with the fed value: that is the
    // W1 bug (kernel-joint-w1-maths.md §9 item 2, "the substitution map … must
    // never rewrite nodes inside a measure component"). Refuse instead.
    //
    // The illegal sharing shape is already a located static error in typeinfer;
    // this covers the LEGAL non-sharing parameterized case too (§5, third
    // bullet), which the maths doc scores only when the ambient parameter is
    // bound — a value this rewrite cannot see, so refusing is the correct
    // behavior rather than a gap.
    for (const rm of measureReps) {
      const reach = _reachableAncestors(rm.measureAst, new Set<string>());
      for (const outer of outerOfSurface.values()) if (reach.has(outer)) return null;
    }
    const boundaryArgs: any[] = [];
    for (const [surface, outer] of outerOfSurface) {
      boundaryArgs.push({
        type: 'KeywordArg', name: surface,
        value: makeIdent(outer, jAst.loc), loc: jAst.loc,
      });
    }
    return {
      type: 'CallExpr', callee: makeIdent('functionof', jAst.loc),
      args: [
        { type: 'CallExpr', callee: makeIdent('joint', jAst.loc), args: newArgs, loc: jAst.loc },
        ...boundaryArgs,
      ],
      loc: jAst.loc,
    };
  }

  // Every binding name `ast` reaches transitively through identifier refs,
  // stopping at (and excluding) `stopAt` — the INTERIOR of a reification whose
  // boundary is `stopAt`. Substituting any of these names rewrites the
  // reification's interior, which is what the complete-cut check must detect.
  //
  // Deliberately NOT `computeClosure`: that walk reports the boundary-DEPENDENT
  // subset (the members needing a fresh copy), and a member left out of it is
  // shared by name — so a merged substitution still rewrites it. `v ~ Normal(0,1)`
  // (not descended from the boundary) with `a2 ~ Normal(v + z, 1)` and a boundary
  // at `z` is the case: `v` is reachable but not boundary-dependent, so
  // `computeClosure` omits it while `argMap[v]` would still substitute it.
  function _reachableAncestors(ast: any, stopAt: Set<string>): Set<string> {
    const found = new Set<string>();
    const idents = (node: any, acc: Set<string>) => {
      if (node == null || typeof node !== 'object') return;
      if (Array.isArray(node)) { for (const c of node) idents(c, acc); return; }
      if (node.type === 'Identifier' && typeof node.name === 'string') acc.add(node.name);
      for (const k in node) if (k !== 'loc') idents(node[k], acc);
    };
    const walk = (name: string) => {
      if (stopAt.has(name) || found.has(name)) return;
      const b = out.get(name) || bindings.get(name);
      if (!b) return;
      found.add(name);
      const rhs = b.node && b.node.value;
      if (!rhs) return;
      const refs = new Set<string>();
      idents(rhs, refs);
      for (const r of refs) walk(r);
    };
    const seeds = new Set<string>();
    idents(ast, seeds);
    for (const s of seeds) walk(s);
    return found;
  }

  // One `joint` component, expressed as the measure it contributes to the
  // hoisted body, plus the reification boundary it declares (null for a
  // measure component — §06's nullary case, which "ignores the input").
  function _jointComponentAsMeasure(compAst: any, compLabel?: string) {
    if (!compAst) return null;
    const inlineHead = compAst.type === 'CallExpr' && compAst.callee
      && compAst.callee.type === 'Identifier' && compAst.callee.name;
    let kAst: any;
    if (compAst.type === 'Identifier') {
      const name = resolveCallableAlias(compAst.name, out);
      const b = out.get(name) || bindings.get(name);
      const t = b && b.inferredType;
      if (!t || t.kind !== 'kernel') {
        return { measureAst: cloneAst(compAst), boundaries: null };
      }
      kAst = b.effectiveValue || (b.node && b.node.value);
    } else if (inlineHead === 'kernelof' || inlineHead === 'functionof') {
      // Written out verbatim as a joint component, not bound to a name first.
      // Spec §06's uniform kernel extension does not distinguish a named
      // kernel from an inline one, so this hoists the same way the Identifier
      // branch above does — the literal CallExpr already IS the reification,
      // no binding lookup needed.
      kAst = compAst;
    } else {
      // `fn(...)` written out verbatim (not bound to a name first) would need
      // its own boundary hoist that this reader does not perform — refuse
      // rather than mistake it for a measure (a silent null here surfaces
      // only as the fanout's generic "no derivation" fallback, which does not
      // say why). Named `fn(...)` kernel components hit the Identifier branch
      // above and resolve through the binding's effective value the same as
      // any other kernel; only the inline literal reaches this refusal.
      if (inlineHead === 'fn') {
        const where = compLabel ? `joint kwarg "${compLabel}"` : 'a joint component';
        const err: any = new Error(
          `${where} is an inline fn(...) reification used directly as a kernel `
          + 'in a joint fan-out (spec §06 "Uniform kernel extension"); the '
          + 'fan-out hoist only reads kernelof(...) / functionof(...) inline, '
          + 'or a name bound to any kernel. Bind it to a name first, e.g. '
          + `k = fn(...); joint(${compLabel ? compLabel + ' = k' : 'k'}, ...).`);
        err.code = 'JOINT_FANOUT_INLINE_FN_COMPONENT';
        throw err;
      }
      return { measureAst: cloneAst(compAst), boundaries: null };
    }
    if (!kAst || kAst.type !== 'CallExpr' || !kAst.callee
        || kAst.callee.type !== 'Identifier') return null;
    const head = kAst.callee.name;
    if (head !== 'kernelof' && head !== 'functionof') return null;
    const bodyAst = kAst.args && kAst.args[0];
    if (!bodyAst || bodyAst.type === 'KeywordArg') return null;
    const boundaries = new Map<string, string>();
    for (let i = 1; i < kAst.args.length; i++) {
      const kw = kAst.args[i];
      if (kw.type !== 'KeywordArg') return null;
      if (!kw.value || kw.value.type !== 'Identifier') return null;
      boundaries.set(kw.name, kw.value.name);
    }
    if (boundaries.size === 0) return null;
    // `kernelof(x, kw…)` reifies a VALUE, so the component's output measure is
    // that value's law — the same wrapping lower.ts applies when it rewrites
    // `kernelof(x, kw)` to `functionof(lawof(x), kw)`. A `functionof` typed as
    // a kernel already has a measure body.
    const measureAst = head === 'kernelof'
      ? {
        type: 'CallExpr', callee: makeIdent('lawof', bodyAst.loc),
        args: [cloneAst(bodyAst)], loc: bodyAst.loc,
      }
      : cloneAst(bodyAst);
    return { measureAst, boundaries };
  }

  // Apply a reified callable's AST (`fnAst` — a functionof / kernelof /
  // fn CallExpr) to a call site (`astArg`), substituting boundary args +
  // synthesizing the closure. Shared by the named path (inlineOnce
  // resolves fnAst from the callee binding) and the expression-headed
  // path (spec §11 — fnAst IS the inline callee). `fnBinding` carries at
  // least the callable's `type`; synthetic for the inline case.
  function _inlineApplication(astArg: any, fnBinding: any, fnAst: any) {
    if (!fnAst || fnAst.type !== 'CallExpr' || !fnAst.args || fnAst.args.length === 0) {
      return astArg;
    }
    // After picking effectiveValue we may now be looking at a
    // `kernelof(...)` AST (synthesised form). The body-substitution
    // rules below assume the same call-positional structure that
    // functionof / kernelof / fn use: args[0] = body, args[1..] =
    // boundary kwargs. Both functionof and the synthesised kernelof
    // satisfy this — but if a future synthesis emits a different
    // head, we'd need additional dispatch here.
    if (fnAst.callee && fnAst.callee.type === 'Identifier') {
      const head = fnAst.callee.name;
      if (head !== 'functionof' && head !== 'kernelof' && head !== 'fn') {
        // Unsupported synthesised shape — bail rather than mis-inline.
        return astArg;
      }
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

    // Auto-splatting (spec §04 sec:calling-convention):
    // `f(record(a=x, b=y))` and `f(some_record_value)` are
    // equivalent to `f(a=x, b=y)`, and likewise for `table(...)` /
    // table columns. We detect two splat sources:
    //   - inline `record(...)` / `table(...)` calls — splat their
    //     KeywordArg children directly.
    //   - Identifier ref to a record- or table-typed binding — synthesize
    //     a FieldAccess per RECORD FIELD / TABLE COLUMN.
    // A sole positional record ALWAYS splats — §04: "A sole positional
    // record or table therefore always splats: whether its field or
    // column names match the callable's argument names decides only
    // whether the call is valid, never whether the splat occurs." So the
    // splat is NOT conditioned on the fields covering surfaceOrder, and a
    // mismatch must never fall back to binding the whole record to
    // surfaceOrder[0] — that fallback was the name-conditioned reading
    // §04 now rules out, and it silently accepted `f(pars)` against a
    // one-input `f`. typeinfer raises the located §04 error; here we bail
    // out of inlining (same idiom as the complex-boundary bail above), so
    // the classifier reports an unsupported outcome instead of emitting a
    // body with unsubstituted boundary identifiers.
    // The keyword spelling `f(pars = <record>)` is how a record is passed
    // as one ordinary argument; it is a KeywordArg and never reaches here.
    //
    // §04's single-input carve-out (design#78) does not apply here either:
    // `fnAst` is always a user-reified callable (functionof / kernelof / fn),
    // and every carve-out-exempt callable is a §07 builtin, which this pass
    // never inlines. See the note at typeinfer's matching check.
    let callArgs = astArg.args || [];
    if (callArgs.length === 1
        && callArgs[0].type !== 'KeywordArg'
        && surfaceOrder.length >= 1) {
      const arg0 = callArgs[0];
      let splatted: any = null;
      if (arg0.type === 'CallExpr' && arg0.callee
          && arg0.callee.type === 'Identifier'
          && (arg0.callee.name === 'record' || arg0.callee.name === 'table')) {
        splatted = (arg0.args || []).filter((f: any) => f.type === 'KeywordArg');
      } else if (arg0.type === 'Identifier') {
        const recBinding = out.get(arg0.name);
        const t = recBinding && recBinding.inferredType;
        // An unknown or non-record/table type is not a splat source we can
        // expand — leave the call alone. That is an unresolved-type bail,
        // not a name-conditioned choice. Records and table columns both
        // splat (spec §04); `t.colname` reads a table column the same way
        // `r.name` reads a record field (spec §03), so the same FieldAccess
        // synthesis covers both.
        const names = (t && t.kind === 'record' && t.fields) ? Object.keys(t.fields)
                    : (t && t.kind === 'table' && t.columns) ? Object.keys(t.columns)
                    : null;
        if (names) {
          splatted = names.map((name: string) => ({
            type: 'KeywordArg',
            name,
            value: { type: 'FieldAccess', object: cloneAst(arg0), field: name, loc: arg0.loc },
            loc: arg0.loc,
          }));
        }
      }
      if (splatted) {
        const splatNames = new Set(splatted.map((f: any) => f.name));
        if (!surfaceOrder.every((n: string) => splatNames.has(n))) return astArg;
        callArgs = splatted;
      }
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
      const synth: any = makeSyntheticBinding(fresh, newRhs);
      // Which model binding this copy stands for. A closure copy carries no
      // `inferredType` (typeinfer ran before it existed), so downstream passes
      // that must tell a copied VARIATE from a copied constructor measure have
      // only its IR to go on — and only a copy created HERE has a boundary
      // substitution behind it. clm's `_namesADraw` keys on this marker so the
      // reasoning stays confined to reification closures.
      synth.closureOf = origName;
      out.set(fresh, synth);
    }

    // Substitute identifiers in a deep-cloned body. Different call
    // sites get independent ASTs.
    const inlined = substituteIdents(cloneAst(bodyAst), argMap);
    // A `kernelof` reifies its argument to a MEASURE (the law of the
    // reified value), so applying it yields that measure — NOT the bare
    // value. At the AST surface level `kernelof(x, kw)`'s body arg is the
    // raw value `x`, so the inlined application must be wrapped in
    // `lawof(...)` to be measure-valued. This mirrors lower.ts's IR-level
    // `kernelof(x, kw) → functionof(lawof(x), kw)` lowering. Without it,
    // `generator(pars)` (with `generator = kernelof(x, …)`) inlined to the variate `x`
    // instead of `lawof(x)`, so e.g. `iid(generator(pars), n)` became the
    // malformed `iid(draw(<dist>), n)` (an iid over a variate, not a
    // measure) and never classified. functionof / fn yield the value
    // itself and are NOT wrapped (the fn branch returned earlier).
    if (fnAst.callee && fnAst.callee.type === 'Identifier'
        && fnAst.callee.name === 'kernelof') {
      return {
        type: 'CallExpr', callee: makeIdent('lawof', astArg.loc),
        args: [inlined], loc: astArg.loc,
      };
    }
    return inlined;
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
    // Phase 1 — reachable non-boundary, non-fixed ancestors + all identifier
    // refs in each one's RHS subtree (not just top-level — the fixpoint needs
    // every nested ref). (Same walk as before: skip boundaries and
    // fixed-phase bindings; look up post-lift `out` first, then `bindings`.)
    const reachable = new Set<string>();
    const directRefs = new Map<string, Set<string>>();
    const visiting = new Set<string>();

    function identsOf(node: any, acc: Set<string>) {
      if (node == null || typeof node !== 'object') return;
      if (Array.isArray(node)) { for (const c of node) identsOf(c, acc); return; }
      if (node.type === 'Identifier') acc.add(node.name);
      // Placeholder refs (arrow-style `par -> ...` boundaries) are a
      // distinct node shape from Identifier — substituteIdents matches
      // them via the sentinel-prefixed key (see PLACEHOLDER_SUB_PREFIX
      // above), so boundary-dependency detection must use the same key
      // or a Placeholder-boundary ref is invisible to `boundaries.has(r)`
      // and every downstream member wrongly stays shared.
      if (node.type === 'Placeholder') acc.add(PLACEHOLDER_SUB_PREFIX + node.name);
      for (const k in node) identsOf(node[k], acc);
    }
    function walk(name: string) {
      if (reachable.has(name) || visiting.has(name)) return;
      if (boundaries.has(name)) return;          // boundary → substituted directly
      const b = out.get(name) || bindings.get(name);
      if (!b) return;                            // built-in / unknown
      if (b.phase === 'fixed') return;           // closed over per spec (shared)
      visiting.add(name);
      reachable.add(name);
      const refs = new Set<string>();
      identsOf(b.node && b.node.value, refs);
      directRefs.set(name, refs);
      for (const r of refs) walk(r);
      visiting.delete(name);
    }
    const bodyRefs = new Set<string>();
    identsOf(bodyAst, bodyRefs);
    for (const r of bodyRefs) walk(r);

    // Phase 2 — fixpoint: a reachable member is COPIED iff its RHS subtree
    // references a boundary directly, or references another already-copied
    // member. Boundary-independent members are left out → shared (their refs
    // in the inlined body keep the original name → resolve to the outer
    // binding, so a point-override patches the one instance the body uses).
    // This is the #265 fix: only members that must thread a substituted
    // boundary get a fresh, independent copy.
    const copy = new Set<string>();
    let changed = true;
    while (changed) {
      changed = false;
      for (const name of reachable) {
        if (copy.has(name)) continue;
        const refs = directRefs.get(name);
        if (!refs) continue;
        let dep = false;
        for (const r of refs) {
          if (boundaries.has(r) || copy.has(r)) { dep = true; break; }
        }
        if (dep) { copy.add(name); changed = true; }
      }
    }
    return Array.from(copy);
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

// Whether `ir` is a (possibly nested) literal numeric/boolean array —
// `vector(lit, lit, ...)` at any rank, e.g. the rank-2 `[[0.0]]`
// kernel-param / observed-data literals a flattened nested-iid density
// lowering emits (spec §03 "vectors of vectors"; determiniser: a
// composed/nested `iid` peels to a single `sum(broadcast(
// builtin_logdensityof, K, <rank-d size-1 array-of-records>,
// <rank-d obs>))`, whose params/obs are rank-d bracket literals, not
// flat rank-1 ones). Recurses through nested `vector(...)` calls so a
// literal array of ANY nesting depth qualifies, not just one level —
// the original one-level-only check rejected `[[0.0]]` (its single
// element is itself a `vector` call, not a `lit`/`const`), which
// silently dropped the containing binding from the derivation graph
// (`isEvaluable` returning false ⇒ `classifyDerivation` returns null)
// instead of scoring it.
function _isLiteralNumericVector(ir: any): boolean {
  if (!ir || ir.kind !== 'call' || ir.op !== 'vector' || !Array.isArray(ir.args)) {
    return false;
  }
  for (const a of ir.args) {
    if (!a) return false;
    if (a.kind === 'lit' || a.kind === 'const') continue;
    if (_isLiteralNumericVector(a)) continue;
    return false;
  }
  return true;
}

/**
 * Whether the worker's evaluateExpr can compute this IR end-to-end
 * given a numeric env. Conservative — returns false for anything we're
 * not 100% sure the evaluator handles, so the orchestrator can short-
 * circuit before involving the worker.
 */
function isEvaluable(ir: IRNode | null | undefined): boolean {
  if (!ir) return false;
  switch (ir.kind) {
    case 'lit':         return typeof ir.value === 'number' || typeof ir.value === 'boolean';
    case 'const':       // pi, inf, im — evaluator resolves these.
                        return true;
    case 'ref':         // resolved against env at evaluation time.
                        return true;
    case 'axis':        // resolved against the enclosing aggregate's axis env.
                        return true;
    case 'call':
      // Special case: a `vector(...)` whose elements are ALL pure
      // literals / consts (no refs) is a constant integer/real array
      // and IS evaluable as a value. The general `vector` op stays
      // out of EVALUABLE_OPS because vectors-with-stochastic-refs
      // (`[mu, 1.0]`) need the array-derivation path; but
      // shape-fold emits literal-only vectors (e.g. for
      // `indicesof0(C)` → `vector(lit_0, lit_1, lit_2)`) that the
      // aggregate body in fusion (a) Step 2 indexes via `get(vec,
      // .j)`. Treating literal-only vectors as evaluable lets that
      // pattern classify cleanly.
      if (ir.op === 'vector' && Array.isArray(ir.args) && _isLiteralNumericVector(ir)) {
        return true;
      }
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
      // rand_succ(state) — the value-domain composite-rand successor
      // (split lane 1). One arg, the rngstate (itself evaluable).
      // Synthesised by the lift rand_succ rewrite; never surface syntax.
      if (ir.op === 'rand_succ') {
        const args = ir.args || [];
        if (args.length !== 1) return false;
        return isEvaluable(args[0]);
      }
      // aggregate(f_reduction, output_axes, expr) — the second arg is
      // always a `vector(...)` of axis refs by construction (the
      // analyzer enforces). vector is deliberately NOT in EVALUABLE_OPS
      // (stochastic-element arrays must stay in the kind:'array' path),
      // so we skip the output_axes arg here and just check the
      // reduction ref (always evaluable) + the expr (the user-written
      // contraction body, which carries axis refs but otherwise looks
      // like a normal value expression).
      if (ir.op === 'aggregate') {
        const args = ir.args || [];
        if (args.length !== 3) return false;
        return isEvaluable(args[0]) && isEvaluable(args[2]);
      }
      // broadcast(f, args, kwargs) — args[0] is a CALLABLE (a function
      // ref OR an inline `functionof(...)` produced by the dotted-
      // operator lowering, e.g. `.+` → `broadcast(functionof(_a + _b),
      // …)`). functionof isn't in EVALUABLE_OPS (it's a measure/
      // callable shape, not a value), so naively recursing into args[0]
      // would reject the whole call. Same structural exception
      // aggregate uses for its reduction ref: skip the callable, only
      // require the collection / scalar args + kwargs to be evaluable.
      // Kernel-broadcast (head is a distribution constructor) lands in
      // classifyKernelBroadcast before this path can fire, so we
      // don't have to distinguish it here.
      if (ir.op === 'broadcast') {
        const args = ir.args || [];
        if (args.length < 1) return false;
        for (let i = 1; i < args.length; i++) {
          if (!isEvaluable(args[i])) return false;
        }
        if (ir.kwargs) {
          for (const k in ir.kwargs) if (!isEvaluable(ir.kwargs[k])) return false;
        }
        return true;
      }
      // x in S (spec §07 membership) — args[1] is a SET descriptor
      // (interval(...) / a named set), NOT a value expression. interval and
      // the set constructors aren't in EVALUABLE_OPS, so naively recursing
      // into the set arg would reject the whole call (same structural
      // exception aggregate/broadcast use for their non-value args). Require
      // only the tested value (args[0]) be evaluable; the set arg is resolved
      // when membership is computed (sampler.evaluateCall `in` case).
      if (ir.op === 'in') {
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

// =====================================================================
// Composite-`rand` successor rewrite (engine-concepts §11)
// =====================================================================

// Shared gate for the two halves of a composite `rand` tuple_get. Given a
// binding RHS `tuple_get(<self-ref to a rand binding>, idx)` at the
// expected index, validate it, pick up the rand's state arg, decompose the
// measure (`iid(M, size)` → (M, size); bare M → (M, 1)), and classify the
// inner as a leaf distribution vs a composite measure. Returns
// `{ stateIR, fromIR, countIR, isComposite }` or null (not a composite-rand
// tuple_get at `idx`: not a tuple_get, wrong index, rand binding missing,
// or inner not a self-ref). `isComposite` is false when the inner resolves
// to a known leaf distribution — then the draw stays on the batched-leaf /
// pre-eval path and the state half stays threaded via sampleLeafN.
//
// Used by BOTH the index-0 draw classifier (derivations.classifyRandSample)
// and the index-1 successor rewrite (rewriteCompositeRandSucc below), so the
// decompose + leaf gate can never drift between the draw and successor
// halves (drift would be a silent correctness bug — a leaf successor wrongly
// key-split, or a composite draw wrongly walked).
function classifyRandTuple(rhsIR: any, bindings: any, idx: number): any {
  if (!rhsIR || rhsIR.op !== 'tuple_get'
      || !Array.isArray(rhsIR.args) || rhsIR.args.length !== 2) return null;
  const src: any = rhsIR.args[0];
  const idxNode: any = rhsIR.args[1];
  if (!src || src.kind !== 'ref' || src.ns !== 'self') return null;
  if (!idxNode || idxNode.kind !== 'lit' || idxNode.value !== idx) return null;

  const randB = bindings.get(src.name);
  const randIR = randB && randB.ir;
  if (!randIR || randIR.op !== 'rand' || !Array.isArray(randIR.args)
      || randIR.args.length !== 2) return null;
  const stateIR = randIR.args[0];

  // Decompose the measure arg (iid inline or a lift-hoisted anon ref).
  // The resolved IR is inspected for the iid shape ONLY — the no-iid
  // fallback hands back the ORIGINAL node, so a by-name ref to a non-iid
  // composite measure (`v, s2 = rand(state, model_dist)` over an
  // applied-kernel / lawof binding) keeps its ref and classifies as a
  // count-1 randsample instead of failing the must-be-a-ref check below
  // and falling onto the per-draw evaluator (which cannot resolve a
  // measure ref — "no resolveMeasureRef was supplied").
  const measureArg: any = randIR.args[1];
  let measureIR: any = measureArg;
  if (measureIR && measureIR.kind === 'ref' && measureIR.ns === 'self') {
    const mb = bindings.get(measureIR.name);
    if (mb && mb.ir) measureIR = mb.ir;
  }
  let fromIR: any, countIR: any;
  if (measureIR && measureIR.kind === 'call' && measureIR.op === 'iid'
      && Array.isArray(measureIR.args) && measureIR.args.length === 2) {
    fromIR = measureIR.args[0];
    countIR = measureIR.args[1];
  } else {
    fromIR = measureArg;
    countIR = { kind: 'lit', value: 1, numType: 'integer' };
  }
  // The inner measure must be a named binding (lift hoists inline inner
  // measures to anons). Inline non-ref inner isn't expected post-lift.
  if (!fromIR || fromIR.kind !== 'ref' || fromIR.ns !== 'self') return null;

  // Leaf gate: resolve the inner-measure ref to its IR, following ref
  // chains to a fixed point. alias-resolution (run immediately before this
  // pass) normally collapses measure-alias chains to a direct call, but we
  // loop (cycle-guarded) so the gate is sound even if one survives — a leaf
  // must NEVER be misclassified composite and wrongly key-split. A
  // known-distribution inner is a LEAF; anything else is composite.
  let innerIR: any = fromIR;
  const seen = new Set<string>();
  while (innerIR && innerIR.kind === 'ref' && innerIR.ns === 'self'
         && !seen.has(innerIR.name)) {
    seen.add(innerIR.name);
    const ib = bindings.get(innerIR.name);
    if (!ib || !ib.ir) break;
    innerIR = ib.ir;
  }
  const isComposite = !(innerIR && innerIR.kind === 'call'
                        && SAMPLEABLE_DISTRIBUTIONS.has(innerIR.op));

  return { stateIR, fromIR, countIR, isComposite };
}

// Post-lift binding-IR rewrite: the STATE half of a COMPOSITE `rand`
// (`tuple_get(<rand>, 1)`) becomes a pure value-domain successor op
// `rand_succ(stateIR)`. This MUST mutate the binding IR (not a derivation):
// the consumers of a composite successor — FixedValues._compute and
// orchestrator.resolveIRToValue — both read `binding.ir`, and the raw
// `tuple_get(rand, 1)` over a composite would route through the per-draw
// walker and throw (a throw FixedValues._compute swallows into a SILENT
// unresolved). Rewriting `binding.ir` once here makes every consumer agree
// and converts 'state half of a tuple' into 'pure successor value' exactly
// once. Leaf state-halves are left untouched so they keep threading the
// counter-advanced successor via sampleLeafN (bit-for-bit builtin_sample ≡
// rand + iid preserved). Runs after the IR-caching post-pass loop populates
// every `b.ir` and after alias-resolution, so inner-measure refs resolve.
function rewriteCompositeRandSucc(out: any): void {
  for (const [name, b] of out) {
    if (!b || !b.ir) continue;
    const t = classifyRandTuple(b.ir, out, 1);   // 1 = the STATE half
    if (!t || !t.isComposite) continue;           // leaf state-half stays threaded
    out.set(name, { ...b, ir: {
      kind: 'call', op: 'rand_succ', args: [t.stateIR], loc: b.ir.loc,
    } });
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
  classifyRandTuple,
};
