'use strict';

// FlatPPL main-thread orchestrator: walks an analyzed bindings map and
// builds an executable "chain" of sample/evaluate steps that the
// sampler-worker can run end-to-end via its `sampleChain` /
// `densityFromChain` messages.
//
// What this module does
// =====================
//
// Given a target binding name and the map produced by
// `analyzer.analyze(ast, source).bindings`, `buildSampleChain(target,
// bindings)` returns a topological list of small "step" records:
//
//   [
//     { name: 'mu', kind: 'sample',   ir: <Normal IR>     },
//     { name: 's',  kind: 'evaluate', ir: <add(mu, 1) IR> },
//     { name: 'y',  kind: 'sample',   ir: <Normal IR refs mu, s> },
//   ]
//
// At sampling time the worker walks this list, drawing or evaluating
// per step, threading the per-draw env (`{name → number}`) through. The
// last step's value is the target's drawn value; repeat N times for N
// samples.
//
// What it explicitly does NOT do
// ==============================
//
//  * It does not execute anything — no RNG, no stdlib. It only inspects
//    AST nodes and lowers them via `lower.js`. The actual sampling
//    happens in the worker.
//  * It does not yet handle reified scopes (lawof, functionof, kernelof,
//    fn, modules), bayesupdate, weighted measures, or any non-scalar
//    binding. Encountering one short-circuits with `unsupported`.
//  * It does not validate worker-side distribution availability; it
//    refuses bindings whose RHS isn't a known/supported distribution
//    or a deterministic numeric expression. The worker's REGISTRY is
//    the ground truth — but a too-eager orchestrator would just push
//    the failure into a "Plot tab errored out" state, which is worse
//    UX than "Plot tab disabled".
//
// Why a separate file vs. extending lower.js or worker.js
// =======================================================
//
// `lower.js` is a pure AST→IR translation; it has no notion of the
// bindings map. `worker.js` is transport-agnostic execution; it has no
// AST. The orchestrator straddles both — it consumes analyzer output
// (AST + bindings) and produces input for the worker. Keeping it
// separate also keeps the DAG visualizer's existing dependency on
// lower.js minimal: the visualizer can choose whether to import the
// orchestrator at all.

const { lowerExpr } = require('./lower');
const { isMeasureExpr } = require('./analyzer');
const { MEASURE_PRODUCING } = require('./builtins');
const { quantileSorted } = require('./histogram');

// Distributions the worker's REGISTRY currently implements. Hardcoded
// here to avoid pulling sampler.js (and stdlib) into the main bundle.
// Mirrored in sampler.js's REGISTRY; if you add one there, add it here
// too. The orchestrator gates on this list — if a binding's RHS is a
// distribution we don't list, the chain comes back unsupported instead
// of failing later in the worker.
const SAMPLEABLE_DISTRIBUTIONS = new Set([
  'Normal', 'Exponential', 'LogNormal', 'Beta', 'Gamma',
  'Cauchy', 'StudentT', 'Bernoulli', 'Binomial', 'Poisson',
  // Dirac is degenerate (zero entropy): the sampler emits the
  // 'value' kwarg verbatim N times. Listed here so measure-alias
  // bindings like `m = Dirac(value = 5)` get classified 'skip' and
  // resolved to a sample step at the target rather than failing
  // SAMPLEABLE_DISTRIBUTIONS gate. Identity rewrite for
  // `draw(Dirac(value=e))` lives in classifyForChain — at the
  // draw site we re-route to evaluate(e) rather than sampling.
  'Dirac',
]);

// Subset of the above whose density is over the counting reference (a
// pmf, integer atoms). Used by the worker to switch between KDE and
// integer-histogram density estimation.
const DISCRETE_DISTRIBUTIONS = new Set([
  'Bernoulli', 'Binomial', 'Poisson',
]);

// Deterministic builtins whose call IRs the worker's evaluateExpr knows
// how to compute. Mirrors the operator desugaring in lower.js plus a
// small catalogue of safe scalar functions. Anything else lowered as
// `(call <op> ...)` is treated as unsupported, so a stray `joint(...)`
// or `disintegrate(...)` doesn't silently get scheduled.
//
// Keep in sync with sampler.js's evaluateExpr handler set. When the
// evaluator gains a new builtin (e.g. `pow`), add it both there and
// here.
const EVALUABLE_OPS = new Set([
  // Operator desugaring (BIN_OP_MAP / UN_OP_MAP in lower.js).
  // This list mirrors sampler.js's ARITH_OPS exactly. Extend both
  // sides together when adding ops (the static gate must match the
  // worker's evaluator).
  'add', 'sub', 'mul', 'div', 'mod', 'neg', 'pos',
  'abs', 'abs2', 'exp', 'log', 'log10', 'sqrt',
  'sin', 'cos',
  'floor', 'ceil', 'round',
  'pow',
  // Comparisons → boolean.
  'lt', 'le', 'gt', 'ge', 'equal', 'unequal',
  // Predicates → boolean.
  'isfinite', 'isinf', 'isnan', 'iszero',
  // Logic / conditionals.
  'land', 'lor', 'lxor', 'lnot', 'ifelse',
  // Reductions over arrays (sampler.js implements the runtime ops). The
  // static gate for these is conservative: only mark the binding
  // evaluable when the operand is a static array (kind: 'array'
  // derivation) — handled by the array-evaluable check downstream.
  // Generic ref-to-stochastic-array isn't evaluable in the per-i
  // worker model since each atom's value would itself be an array.
  // Note: 'vector' deliberately omitted — leaves like `[mu, 1.0]`
  // (with stochastic refs) must NOT classify as evaluable, so the
  // existing array-derivation special case keeps owning that path.
  'sum', 'mean', 'prod', 'length', 'maximum', 'minimum', 'var',
  // Engine-internal projection emitted by the analyzer's multi-LHS
  // rewriter (`a, b = rand(...)`). sampler.evaluateCall handles it.
  'tuple_get', 'tuple',
  // Field access: lowered from surface `obj.field` and from `record(
  // a=x, b=y)` constructors. Both are pure value computations the
  // evaluator handles.
  'get_field', 'record',
  // Random-number primitives (spec §sec:random). All three are
  // ordinary value-typed functions whose phase propagates from
  // their inputs. sampler.evaluateCall dispatches each.
  'rnginit', 'rngstate', 'rand',
]);

/**
 * Build an execution chain for sampling `targetName`.
 *
 * @param {string} targetName  binding to sample
 * @param {Map<string, BindingInfo>} bindings  from analyzer.analyze()
 * @returns {{
 *   chain?: Array<{ name: string, kind: 'sample'|'evaluate', ir: object }>,
 *   discrete?: boolean,        // true iff target is a discrete-distribution draw
 *   unsupported?: { reason: string },
 * }}
 */
function buildSampleChain(targetName, bindings) {
  if (!bindings || !bindings.has(targetName)) {
    return { unsupported: { reason: `unknown binding '${targetName}'` } };
  }

  const visited = new Set();    // names already placed into `order`
  const visiting = new Set();   // names currently on the DFS stack (cycle guard)
  const order = [];             // topologically-ordered chain steps

  // Track per-binding diagnostics so the first hit aborts cleanly.
  let unsupported = null;

  function visit(name) {
    if (unsupported) return;
    if (visited.has(name)) return;
    if (visiting.has(name)) {
      // Cycle. Bindings aren't supposed to be self-referential, but
      // defensive — better than an infinite recurse.
      unsupported = { reason: `cyclic dependency through '${name}'` };
      return;
    }
    const binding = bindings.get(name);
    if (!binding) {
      // A dep references a name not in `bindings`. Could be a builtin
      // (pi, true, …) — the lowering will produce a `const` or `lit`
      // node and the evaluator handles it. Could also be a free var
      // the analyzer flagged with a warning. Either way, we don't
      // need to add a chain step for it; the lowered IR for whoever
      // referenced it can stand on its own.
      return;
    }
    visiting.add(name);

    // Recurse into deps first so they appear earlier in the chain.
    for (const dep of binding.deps) visit(dep);
    if (unsupported) return;

    // Lower this binding's RHS expression. Bindings the analyzer has
    // rewritten (multi-LHS, disintegrate) carry an `effectiveValue`
    // AST that's the per-name view; lower that when present so the
    // chain sees `tuple_get(...)` for `random_data, rstate2 = rand(...)`
    // and the synthesised kernel/prior for disintegrate, not the raw
    // user-written RHS shared across the group.
    let rhsIR;
    try {
      rhsIR = lowerExpr(binding.effectiveValue || binding.node.value);
    } catch (e) {
      unsupported = { reason: `cannot lower '${name}': ${e.message}` };
      return;
    }

    // Classify the step. Four shapes are supported today:
    //   1. draw(<dist-call>)        → sample step on the inner dist IR
    //   2. draw(<ref-to-measure>)   → sample step using the resolved
    //                                  underlying dist IR (alias chase)
    //   3. literal/numeric          → evaluate step (lit IR)
    //   4. deterministic arithmetic → evaluate step (lowered RHS)
    // A fifth, "skip", covers measure-alias bindings (like
    //   `m = Normal(...)`) that downstream draws inline. They produce
    //   no chain step of their own; their deps are still walked so any
    //   stochastic parents inside the alias body land in the chain.
    const stepKind = classifyForChain(binding, rhsIR, bindings);
    if (!stepKind) {
      unsupported = {
        reason: `binding '${name}' (type=${binding.type}) is not chainable for sampling yet`,
      };
      return;
    }

    if (stepKind.kind === 'sample') {
      order.push({ name, kind: 'sample', ir: stepKind.distIR });
    } else if (stepKind.kind === 'evaluate') {
      // irOverride lets a classifier (e.g. the draw(Dirac) identity
      // rewrite) substitute a different IR than the binding's
      // literal RHS. Default: use rhsIR verbatim.
      order.push({ name, kind: 'evaluate', ir: stepKind.irOverride || rhsIR });
    }
    // 'skip' contributes nothing to the chain — its deps were already
    // walked above. This is the alias case.

    visiting.delete(name);
    visited.add(name);
  }

  visit(targetName);
  if (unsupported) return { unsupported };

  // If the target was classified 'skip' (a measure-alias binding like
  //   theta1_dist = Normal(0, 1)
  // — its deps got walked but no chain step was pushed for the target
  // itself), promote it now to a sample step using its lowered RHS.
  // The user is asking for samples *from this measure*, so we synthesise
  // the step that does exactly that. Any upstream stochastic params the
  // alias references are already in the chain via the dep walk above.
  const targetAppeared = order.some(s => s.name === targetName);
  if (!targetAppeared) {
    const targetBinding = bindings.get(targetName);
    if (targetBinding && targetBinding.node && targetBinding.node.value) {
      let targetIR;
      try { targetIR = lowerExpr(targetBinding.node.value); } catch (_) { targetIR = null; }
      // Same canonicalisation as resolveMeasure / classifyForChain,
      // so a target like `m = lawof(observed_data)` (where
      // observed_data is fixed-phase) promotes to a sample step on
      // Dirac(value=observed_data).
      targetIR = normalizeMeasureIR(targetIR, bindings);
      if (targetIR && targetIR.kind === 'call' && targetIR.op
          && SAMPLEABLE_DISTRIBUTIONS.has(targetIR.op)) {
        order.push({ name: targetName, kind: 'sample', ir: targetIR });
      } else {
        return { unsupported: { reason: `target '${targetName}' produced no chain step` } };
      }
    }
  }

  // Mark whether the target's leaf distribution is discrete so the
  // density estimator picks histogram over KDE.
  const lastStep = order[order.length - 1];
  let discrete = false;
  if (lastStep && lastStep.kind === 'sample' && lastStep.ir && lastStep.ir.op) {
    discrete = DISCRETE_DISTRIBUTIONS.has(lastStep.ir.op);
  }

  return { chain: order, discrete };
}

/**
 * Canonicalise measure-construction IRs so downstream classification,
 * sampling, and the viewer's plot dispatch all see a single normalized
 * shape per equivalence class. Pure: input IR is not mutated.
 *
 *   lawof(e)              ≡ Dirac(value = e)   ONLY when e is fixed-phase.
 *      (For deterministic e the law is a point mass at e. For
 *      stochastic e — e.g. `lawof(draw(m))` — the spec identity is
 *      lawof(draw(m)) ≡ m, NOT Dirac(value=draw_result), so we skip
 *      the rewrite. Spec §sec:variate-measure + §sec:lawof.)
 *
 *   Dirac(e)              ≡ Dirac(value = e)
 *      (Positional argument bound to the kwarg name per spec
 *      §sec:calling-convention: built-in callables accept both
 *      positional and keyword forms, with identical semantics.
 *      Purely syntactic — no phase check needed.)
 *
 * Applied at every entry point that classifies measure IRs
 * (classifyForChain, resolveMeasure, target-promotion,
 * classifyDerivation). After this point, fixed-phase lawof and
 * positional-Dirac don't appear as distinct measure surface forms —
 * only Dirac(value=...) remains, and the Dirac sampler / viewer text
 * path handles it uniformly.
 *
 * @param ir       IR node to (possibly) rewrite.
 * @param bindings Optional bindings map. When supplied, enables the
 *                 lawof rewrite by letting us check the phase of a
 *                 ref-arg. Without it, lawof passes through unchanged.
 */
function normalizeMeasureIR(ir, bindings) {
  if (!ir || ir.kind !== 'call') return ir;
  if (ir.op === 'lawof'
      && Array.isArray(ir.args) && ir.args.length === 1
      && (!ir.kwargs || Object.keys(ir.kwargs).length === 0)) {
    if (isFixedPhaseValueIR(ir.args[0], bindings)) {
      return { kind: 'call', op: 'Dirac',
               kwargs: { value: ir.args[0] }, loc: ir.loc };
    }
  }
  if (ir.op === 'Dirac'
      && (!ir.kwargs || !Object.prototype.hasOwnProperty.call(ir.kwargs, 'value'))
      && Array.isArray(ir.args) && ir.args.length === 1) {
    return { kind: 'call', op: 'Dirac',
             kwargs: { value: ir.args[0] }, loc: ir.loc };
  }
  return ir;
}

// Conservative "this IR denotes a deterministic value" predicate used
// by normalizeMeasureIR's lawof rewrite. Literals and named constants
// are always fixed; refs are fixed iff they point at a binding with
// phase='fixed'. Anything else (calls, missing bindings) returns
// false — the rewrite skips them and the lawof stays in its original
// form for downstream phase-aware dispatch.
function isFixedPhaseValueIR(ir, bindings) {
  if (!ir) return false;
  if (ir.kind === 'lit' || ir.kind === 'const') return true;
  if (ir.kind === 'ref' && ir.ns === 'self' && bindings) {
    const b = bindings.get(ir.name);
    return !!(b && b.phase === 'fixed');
  }
  return false;
}

/**
 * Decide how a single binding contributes to the chain.
 * Returns null if not chainable, otherwise one of:
 *   { kind: 'sample',   distIR }  — sample from distIR per draw
 *   { kind: 'evaluate' }          — call evaluateExpr on the lowered RHS
 *   { kind: 'skip' }              — measure alias; deps walked, no
 *                                    chain step produced for this name
 */
function classifyForChain(binding, rhsIR, bindings) {
  // Canonicalise lawof / positional-Dirac into Dirac(value=...) so
  // every branch below can reason in a single normalized form.
  rhsIR = normalizeMeasureIR(rhsIR, bindings);
  // Stochastic binding: `y = draw(...)`. The lowered RHS is a
  // (call draw <args>); we want the args[0] (the dist-call IR) for
  // the sample step so the worker doesn't have to special-case 'draw'.
  // `args[0]` may be either:
  //   * a direct distribution call: draw(Normal(0, 1))
  //   * a ref to a measure alias:   draw(theta1_dist)   where
  //     theta1_dist = Normal(0, 1) lives one (or more) hops away.
  // resolveMeasure handles both, chasing through alias chains until
  // it bottoms out on a sampleable dist call.
  if (binding.type === 'draw') {
    if (!rhsIR || rhsIR.kind !== 'call' || rhsIR.op !== 'draw') return null;
    const inner = (rhsIR.args && rhsIR.args[0]) || null;
    if (!inner) return null;
    const distIR = resolveMeasure(inner, bindings, new Set());
    if (!distIR) return null;
    // Identity rewrite for degenerate (zero-entropy) measures:
    //   draw(Dirac(value = e)) ≡ e
    // (lawof / positional-Dirac forms are already canonicalised to
    // Dirac(value=...) by resolveMeasure → normalizeMeasureIR.)
    // Re-route the binding from a sample step on a degenerate measure
    // to an evaluate step on the value IR; the worker evaluates e
    // (per atom, with refs from upstream) rather than spinning up a
    // degenerate sampler. Phase analysis still classifies the binding
    // 'stochastic' by the strict structural rule (any draw ancestor →
    // stochastic), but the runtime values are correct and downstream
    // rendering treats it equivalently to e.
    if (distIR.kind === 'call' && distIR.op === 'Dirac'
        && distIR.kwargs && distIR.kwargs.value) {
      return { kind: 'evaluate', irOverride: distIR.kwargs.value };
    }
    return { kind: 'sample', distIR };
  }

  // Measure-alias binding: e.g. `theta1_dist = Normal(0, 1)`. The
  // analyzer classifies this as type='call'. It's *not* itself a
  // scalar — it constructs a measure that downstream draws sample
  // from. We don't add it to the chain (no scalar value to thread)
  // but we still want its deps walked, so the right answer is 'skip'.
  // Detection: the lowered RHS is a (call <DistName> ...) with
  // DistName in SAMPLEABLE_DISTRIBUTIONS. Anything else under
  // type='call' falls through to the deterministic-arithmetic path.
  //
  // type='lawof' is admitted alongside type='call' here, since after
  // normalizeMeasureIR a `lawof(e)` binding is shaped exactly like
  // `Dirac(value=e)` — same skip-then-promote-on-target flow.
  if ((binding.type === 'call' || binding.type === 'lawof')
      && rhsIR && rhsIR.kind === 'call' && rhsIR.op
      && SAMPLEABLE_DISTRIBUTIONS.has(rhsIR.op)) {
    return { kind: 'skip' };
  }

  // Deterministic literal binding: `pi_over_2 = pi / 2` etc. Either:
  //   - lit / const node (constant directly)
  //   - call to an EVALUABLE_OPS op
  // We accept either shape via the evaluator on the worker.
  if (binding.type === 'literal' || binding.type === 'call') {
    if (isEvaluable(rhsIR)) return { kind: 'evaluate' };
    return null;
  }

  // Inputs (elementof) are external boundary values — they MUST be
  // supplied via the env passed alongside the chain. The chain itself
  // doesn't need a step for them.
  if (binding.type === 'input') {
    return null; // handled by caller via env, not a chain step
  }

  // Reifications, modules, joints, likelihoods, bayesupdate, … all
  // unsupported in this iteration.
  return null;
}

/**
 * Resolve a measure-typed expression to a concrete distribution IR.
 * Walks through `(ref self <name>)` aliases by looking up bindings
 * and lowering their RHS, until we land on a `(call <Dist> ...)`
 * whose op is sampleable. Returns the dist IR on success, null
 * otherwise. The IR returned is fresh (lowered each call) so callers
 * are free to embed it without worrying about aliasing.
 *
 * @param {object} ir   IR node — a `call` (potentially a dist call)
 *                      or a `ref` we should chase
 * @param {Map}    bindings  binding map for ref lookup
 * @param {Set<string>} seen  cycle guard — names currently being chased
 * @returns {object | null}
 */
function resolveMeasure(ir, bindings, seen) {
  if (!ir) return null;
  // Canonicalise on the way in: lawof of fixed-phase value and
  // positional-Dirac become Dirac(value=...) so the SAMPLEABLE check
  // below — and every downstream consumer of the returned IR — sees
  // a single shape per measure-equivalence class.
  ir = normalizeMeasureIR(ir, bindings);
  if (ir.kind === 'call' && ir.op && SAMPLEABLE_DISTRIBUTIONS.has(ir.op)) {
    return ir;
  }
  if (ir.kind === 'ref' && ir.ns === 'self') {
    if (seen.has(ir.name)) return null; // cycle in alias chain
    seen.add(ir.name);
    const b = bindings.get(ir.name);
    if (!b || !b.node || !b.node.value) return null;
    let bIR;
    try { bIR = lowerExpr(b.node.value); } catch (_) { return null; }
    return resolveMeasure(bIR, bindings, seen);
  }
  return null;
}

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
function argSignature(op, numArgs) {
  if (op === 'draw')                              return ['measure'];
  if (op === 'weighted' || op === 'logweighted')  return ['value', 'measure'];
  if (op === 'normalize')                         return ['measure'];
  if (op === 'superpose')                         return Array(numArgs).fill('measure');
  if (op === 'lawof')                             return ['value-or-measure'];
  if (op === 'iid') {
    // iid(<measure>, n, m, ...): first arg measure-typed, rest values.
    const sig = ['measure'];
    for (let i = 1; i < numArgs; i++) sig.push('value');
    return sig;
  }
  if (op === 'jointchain' || op === 'chain') {
    // jointchain(M, K1, K2, ...) / chain(M, K1, K2, ...): every
    // positional arg is measure-typed (the first a base measure or
    // closed kernel; the rest non-nullary kernels). Lifting them lets
    // inlineChainOps find them as named bindings to walk.
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
function opUsesValueKwargs(op) {
  return SAMPLEABLE_DISTRIBUTIONS.has(op) || EVALUABLE_OPS.has(op);
}

/**
 * Map an AST RHS to the binding `type` the analyzer would assign.
 * Mirrors classifyStatement in analyzer.js for the subset that can
 * appear after lifting; we don't redirect to the analyzer because
 * pulling it in here would create a cycle.
 */
function inferSyntheticType(astNode) {
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

function liftInlineSubexpressions(bindings) {
  const out = new Map(bindings);
  let counter = 0;
  function freshName() {
    let n;
    do { n = '__anon' + (counter++); } while (out.has(n));
    return n;
  }
  function makeIdent(name, loc) {
    return { type: 'Identifier', name, loc: loc || null };
  }
  function makeSyntheticBinding(name, ast) {
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
    // Re-run user-call / chain-op inlining after visit. visit() lifts
    // each positional arg to a named anonymous binding, so a top-level
    // chain op like jointchain(Exp(1), fn(...)) — whose args were
    // inline expressions before visit — now has Identifier args and
    // can be rewritten by inlineChainOps. inlineUserCall is idempotent
    // when no further rewrite applies, so this is a no-op for bindings
    // that don't benefit from a second pass.
    cloned = inlineUserCall(cloned);
    let effLifted = binding.effectiveValue;
    if (effLifted) {
      effLifted = cloneAst(effLifted);
      effLifted = inlineUserCall(effLifted);
      visit(effLifted);
      effLifted = inlineUserCall(effLifted);
    }
    out.set(name, {
      ...binding,
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

  function cloneAst(node) {
    if (node == null || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(cloneAst);
    const copy = {};
    for (const k in node) copy[k] = cloneAst(node[k]);
    return copy;
  }

  // -- Visitor ----------------------------------------------------------
  // visit() walks INTO a node, lifting children if needed. The lift
  // helpers (liftMeasure / liftValue / liftMeasureOrValue) are mutually
  // recursive with visit() through their internal `visit(astArg)` call
  // — that's how deep nesting flattens out one anon at a time.

  function visit(astNode) {
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
  function containsHoleOrPlaceholder(astNode) {
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

  function liftMeasure(astArg) {
    if (!astArg) return astArg;
    astArg = inlineUserCall(astArg);
    visit(astArg);
    if (astArg.type === 'Identifier') return astArg;
    if (containsHoleOrPlaceholder(astArg)) return astArg;
    const name = freshName();
    out.set(name, makeSyntheticBinding(name, astArg));
    return makeIdent(name, astArg.loc);
  }

  function liftValue(astArg) {
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

  function liftMeasureOrValue(astArg) {
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

  function inlineUserCall(astArg) {
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
      astArg = inlineChainOps(astArg);
      astArg = inlineRelabel(astArg);
      astArg = inlineFchain(astArg);
      astArg = inlineDensityof(astArg);
    }
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
  function inlineDensityof(astArg) {
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
  function inlineFchain(astArg) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee) return astArg;
    let fchainCall = null;
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
    const fns = (fchainCall.args || []).filter(a => a && a.type !== 'KeywordArg');
    if (fns.length === 0) return astArg;
    // Build f1(args), then f2(f1(args)), …, fN(…). Each fn[i] becomes
    // a callee — must be an Identifier so inlineOnce can substitute.
    for (const f of fns) {
      if (!f || f.type !== 'Identifier') return astArg;
    }
    const callerArgs = (astArg.args || []).map(cloneAst);
    let result = {
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
  function inlineRelabel(astArg) {
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
    const names = [];
    for (const el of elems) {
      if (!el || el.type !== 'StringLiteral') return astArg;
      names.push(el.value);
    }
    if (names.length === 0) return astArg;

    const loc = astArg.loc;
    const kw = (name, value) => ({ type: 'KeywordArg', name, value, loc });
    const recordCall = (kwargs) => ({
      type: 'CallExpr',
      callee: { type: 'Identifier', name: 'record', loc },
      args: kwargs, loc,
    });

    // Inline ArrayLiteral source: `relabel([a, b, c], ["x", "y", "z"])`
    // → `record(x = a, y = b, z = c)`.
    if (argExpr.type === 'ArrayLiteral') {
      const argElems = argExpr.elements || argExpr.elems || [];
      if (argElems.length !== names.length) return astArg;
      return recordCall(names.map((n, i) => kw(n, argElems[i])));
    }

    // Inline record(...) source: rename by position.
    if (argExpr.type === 'CallExpr' && argExpr.callee
        && argExpr.callee.type === 'Identifier' && argExpr.callee.name === 'record'
        && Array.isArray(argExpr.args)) {
      const recArgs = argExpr.args.filter(a => a && a.type === 'KeywordArg');
      if (recArgs.length !== names.length) return astArg;
      return recordCall(names.map((n, i) => kw(n, recArgs[i].value)));
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
        return recordCall(names.map((n, i) => kw(n, targetElems[i])));
      }
      if (targetAst && targetAst.type === 'CallExpr' && targetAst.callee
          && targetAst.callee.type === 'Identifier' && targetAst.callee.name === 'record'
          && Array.isArray(targetAst.args)) {
        const targetKwargs = targetAst.args.filter(a => a && a.type === 'KeywordArg');
        if (targetKwargs.length !== names.length) return astArg;
        // Reuse the source record's kwarg VALUES verbatim (they're
        // already lifted to Identifier refs after the source binding
        // was visited, or are inline exprs ready to be lifted by the
        // outer visit pass). Synthesising FieldAccess (src.a) instead
        // would force the lowerer to introduce get_field nodes that
        // no measure-op classifier handles, breaking the cascade.
        return recordCall(names.map((n, i) => kw(n, targetKwargs[i].value)));
      }
      // Identifier pointing at neither — fall through.
    }

    // Single-name scalar wrap: relabel(<anything>, [name]) → record(name = arg)
    if (names.length === 1) {
      return recordCall([kw(names[0], argExpr)]);
    }

    return astArg;
  }

  /**
   * Rewrite `jointchain(P, K)` and `chain(P, K)` at the AST level.
   *
   *   jointchain(P, K)  →  joint(P_fields..., K_body_fields...)
   *   chain(P, K)       →  K_body  (P's fields marginalized away)
   *
   * Mechanics: extract P's record-field map; map K's surface kwargs
   * to P's fields by name; build a synthetic kernel-call AST and
   * recursively inline it via inlineOnce (which performs the closure
   * walk + substitution per applyCallable). The result is the
   * substituted body — typically a ref to a synthesized record
   * measure. We then extract that body's fields and combine.
   *
   * Constraints (Phase 1):
   *   - P must be a record-shaped measure (lawof(record(...)) or
   *     joint(...)). Fields extracted from its RHS AST.
   *   - K must be functionof / kernelof / fn whose body resolves to
   *     a record-shaped measure (joint, lawof(record), etc.).
   *   - Surface kwarg names of K must match field names of P.
   *
   * Anything outside these shapes returns the input unchanged so the
   * binding falls through to the generic classifier (and ends up
   * unsupported, which is the right behaviour for forms we don't yet
   * structurally handle).
   */
  function inlineChainOps(astArg) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    const op = astArg.callee.name;
    if (op !== 'jointchain' && op !== 'chain') return astArg;
    if (!astArg.args || astArg.args.length !== 2) return astArg;

    // Keyword-shorthand jointchain: jointchain(a = M, b = K) per spec
    // §sec:jointchain is equivalent to
    //   jointchain(relabel(M, ["a"]), relabel(K, ["b"]))
    // i.e. each kwarg names a component output. We lower to the same
    // stochastic-node form as the positional case, but emit a
    // record(...) so the kind='record' classifier picks it up.
    //
    //   jointchain(a = M, b = K)  ≡  record(a = draw(M), b = draw(K(_a)))
    //
    // chain (which marginalises) doesn't fit cleanly into the kwarg
    // shorthand and isn't covered by this branch.
    if (op === 'jointchain'
        && astArg.args[0].type === 'KeywordArg'
        && astArg.args[1].type === 'KeywordArg') {
      const fieldA = astArg.args[0];
      const fieldB = astArg.args[1];
      const Mexpr = fieldA.value;
      const Kexpr = fieldB.value;
      // Lift inline M and K to anon bindings (they may already be
      // Identifiers if visit() got there first).
      const Mref = Mexpr.type === 'Identifier' ? Mexpr : (function() {
        const n = freshName();
        out.set(n, makeSyntheticBinding(n, Mexpr));
        return makeIdent(n, astArg.loc);
      }());
      const Kref = Kexpr.type === 'Identifier' ? Kexpr : (function() {
        const n = freshName();
        out.set(n, makeSyntheticBinding(n, Kexpr));
        return makeIdent(n, astArg.loc);
      }());
      // Synthesise _a = draw(M).
      const aName = freshName();
      out.set(aName, makeSyntheticBinding(aName, {
        type: 'CallExpr',
        callee: makeIdent('draw', astArg.loc),
        args: [Mref],
        loc: astArg.loc,
      }));
      // Apply K positionally to _a (inlineOnce handles fn / functionof
      // / kernelof bodies).
      const kCall = {
        type: 'CallExpr',
        callee: Kref,
        args: [makeIdent(aName, astArg.loc)],
        loc: astArg.loc,
      };
      const appliedK = inlineOnce(kCall);
      if (!appliedK || appliedK === kCall) return astArg;
      const bName = freshName();
      const drawB = {
        type: 'CallExpr',
        callee: makeIdent('draw', astArg.loc),
        args: [appliedK],
        loc: astArg.loc,
      };
      visit(drawB);
      out.set(bName, makeSyntheticBinding(bName, drawB));
      // record(a = _a, b = _b).
      return {
        type: 'CallExpr',
        callee: makeIdent('record', astArg.loc),
        args: [
          { type: 'KeywordArg', name: fieldA.name, value: makeIdent(aName, astArg.loc), loc: astArg.loc },
          { type: 'KeywordArg', name: fieldB.name, value: makeIdent(bName, astArg.loc), loc: astArg.loc },
        ],
        loc: astArg.loc,
      };
    }

    // Lift inline P / K expressions to anon bindings so the lookup
    // below finds them in `out`. The kwarg branch above does the
    // same for its M/K — without lifting here, positional
    // jointchain(Exp(1), fn(Normal(1, _))) would fail the
    // Identifier check and fall through to "unsupported", leaving
    // the binding unplottable.
    let Parg = astArg.args[0], Karg = astArg.args[1];
    if (Parg.type !== 'Identifier') {
      const n = freshName();
      out.set(n, makeSyntheticBinding(n, Parg));
      Parg = makeIdent(n, astArg.loc);
    }
    if (Karg.type !== 'Identifier') {
      const n = freshName();
      out.set(n, makeSyntheticBinding(n, Karg));
      Karg = makeIdent(n, astArg.loc);
    }

    // Look up via `out`, not the read-only `bindings` input — args
    // may have been lifted to synthesized anon bindings during the
    // visit pass and those live in `out`. Originals are present in
    // both (out is a copy of bindings at function entry).
    const Pbinding = out.get(Parg.name);
    const Kbinding = out.get(Karg.name);
    if (!Pbinding || !Kbinding) return astArg;
    if (Kbinding.type !== 'functionof' && Kbinding.type !== 'kernelof'
        && Kbinding.type !== 'fn') return astArg;

    const Pfields = extractRecordFields(Pbinding.node && Pbinding.node.value);

    // Positional non-record case: jointchain(M, K) where M is a
    // scalar (or array) measure. Per spec §sec:jointchain the
    // result variate is cat(M_var, K(M_var)_var), so for two
    // scalar components the output is a 2-element array measure.
    // Equivalence we lower to:
    //   _a = draw(P_ref)            (alias-style; reuses P's atoms)
    //   _b = draw(K_applied_to_a)   (applies K positionally to _a)
    //   result = [_a, _b]            (ArrayLiteral → tuple kind)
    // Restricted to two-arg jointchain for now (chain not handled,
    // since chain marginalises and we don't have that materialiser).
    if (!Pfields) {
      if (op !== 'jointchain') return astArg;
      const aName = freshName();
      const drawA = {
        type: 'CallExpr',
        callee: makeIdent('draw', astArg.loc),
        args: [makeIdent(Parg.name, astArg.loc)],
        loc: astArg.loc,
      };
      out.set(aName, makeSyntheticBinding(aName, drawA));
      // Apply K positionally to _a. inlineOnce produces the body's
      // substituted AST inline (e.g. fn(Normal(1, _)) applied to _a
      // becomes Normal(1, _a)); lift visits it so any nested inline
      // measure expressions get their own anons.
      const kCall = {
        type: 'CallExpr',
        callee: makeIdent(Karg.name, astArg.loc),
        args: [makeIdent(aName, astArg.loc)],
        loc: astArg.loc,
      };
      const appliedK = inlineOnce(kCall);
      if (!appliedK || appliedK === kCall) return astArg;  // K not inlineable
      const bName = freshName();
      const drawB = {
        type: 'CallExpr',
        callee: makeIdent('draw', astArg.loc),
        args: [appliedK],
        loc: astArg.loc,
      };
      visit(drawB);
      out.set(bName, makeSyntheticBinding(bName, drawB));
      return {
        type: 'ArrayLiteral',
        elements: [makeIdent(aName, astArg.loc), makeIdent(bName, astArg.loc)],
        loc: astArg.loc,
      };
    }

    // Map K's surface kwargs to P's field refs by name. K's surface
    // kwargs are read from K's RHS args (everything after the body).
    const Kast = Kbinding.node && Kbinding.node.value;
    if (!Kast || Kast.type !== 'CallExpr' || !Kast.args || Kast.args.length === 0) return astArg;
    const callKwargs = [];
    for (let i = 1; i < Kast.args.length; i++) {
      const a = Kast.args[i];
      if (a.type !== 'KeywordArg') continue;
      const fieldRef = Pfields[a.name];
      if (!fieldRef) return astArg;   // K's param doesn't match a P field
      callKwargs.push({ type: 'KeywordArg', name: a.name, value: fieldRef, loc: astArg.loc });
    }
    // Build a synthetic kernel-call AST and inline it (delegates to
    // applyCallable's closure walk + synthesis).
    const synthCall = {
      type: 'CallExpr', callee: Karg, args: callKwargs, loc: astArg.loc,
    };
    const appliedBody = inlineOnce(synthCall);
    if (!appliedBody || appliedBody.type !== 'Identifier') return astArg;

    // appliedBody is a ref to the synthesized body binding. It should
    // resolve to a record-shaped measure; extract its fields.
    const synthBodyBinding = out.get(appliedBody.name);
    const Kfields = extractRecordFields(synthBodyBinding && synthBodyBinding.node && synthBodyBinding.node.value);
    if (!Kfields) return astArg;

    // Build the result AST. For jointchain combine; for chain take
    // only K's fields.
    const fields = op === 'jointchain' ? { ...Pfields, ...Kfields } : { ...Kfields };
    const args = Object.keys(fields).map(name => ({
      type: 'KeywordArg', name, value: fields[name], loc: astArg.loc,
    }));
    return {
      type: 'CallExpr',
      callee: { type: 'Identifier', name: 'joint', loc: astArg.callee.loc },
      args,
      loc: astArg.loc,
    };
  }

  /**
   * Walk a measure-typed AST node back to its record(...) or
   * joint(...) constructor and return its field map (name → AST
   * value). Chases `lawof` once if present (per spec, lawof of a
   * record is the joint measure of those fields). Returns null when
   * the expression isn't a recognisable record-shaped form.
   */
  function extractRecordFields(astNode) {
    if (!astNode) return null;
    let v = astNode;
    // Follow Identifier refs through the lifted bindings map. After
    // visit() runs, e.g. `lawof(record(...))` may have its inner
    // record lifted to an anonymous binding, leaving us with
    // `lawof(__anon_record)`. Dereference through `out` so we can
    // still see the underlying record/joint shape.
    if (v.type === 'Identifier') {
      const inner = out.get(v.name);
      if (!inner || !inner.node || !inner.node.value) return null;
      return extractRecordFields(inner.node.value);
    }
    if (v.type === 'CallExpr' && v.callee && v.callee.type === 'Identifier') {
      if (v.callee.name === 'lawof' && v.args && v.args.length === 1) {
        return extractRecordFields(v.args[0]);
      }
      if (v.callee.name === 'record' || v.callee.name === 'joint') {
        const fields = {};
        for (const a of (v.args || [])) {
          if (a.type === 'KeywordArg' && a.value) fields[a.name] = a.value;
        }
        return fields;
      }
    }
    return null;
  }

  function inlineOnce(astArg) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    const fnName = astArg.callee.name;
    // Use `out`, not `bindings`, so synthesized anon bindings created
    // during the lift pass are visible to inlineOnce too (mirrors the
    // change in inlineChainOps above).
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
    const surfaceOrder = [];
    const internalForSurface = {};
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
      let splatted = null;
      if (arg0.type === 'CallExpr' && arg0.callee
          && arg0.callee.type === 'Identifier'
          && arg0.callee.name === 'record') {
        const fieldNames = new Set();
        for (const f of (arg0.args || [])) {
          if (f.type === 'KeywordArg') fieldNames.add(f.name);
        }
        if (surfaceOrder.every(n => fieldNames.has(n))) {
          splatted = (arg0.args || []).filter(f => f.type === 'KeywordArg');
        }
      } else if (arg0.type === 'Identifier') {
        const recBinding = out.get(arg0.name);
        const t = recBinding && recBinding.inferredType;
        if (t && t.kind === 'record' && t.fields
            && surfaceOrder.every(n => n in t.fields)) {
          splatted = surfaceOrder.map(name => ({
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
    const boundaries = new Set();
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
  function computeClosure(bodyAst, boundaries) {
    const closure = new Set();
    const visiting = new Set();

    function walk(name) {
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
    function collectRefsAst(node) {
      if (node == null || typeof node !== 'object') return;
      if (Array.isArray(node)) { for (const c of node) collectRefsAst(c); return; }
      if (node.type === 'Identifier') walk(node.name);
      for (const k in node) collectRefsAst(node[k]);
    }
    collectRefsAst(bodyAst);
    return Array.from(closure);
  }

  function substituteIdents(ast, sub) {
    if (ast == null || typeof ast !== 'object') return ast;
    if (Array.isArray(ast)) return ast.map(c => substituteIdents(c, sub));
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
    const out = {};
    for (const k in ast) out[k] = substituteIdents(ast[k], sub);
    return out;
  }

  // Walk the AST in reading order, replacing each Hole with the
  // corresponding positional arg. Holes whose `positional[i]` is
  // undefined stay as Hole (unbound — caller's responsibility to
  // flag the missing arg).
  function substituteHolesPositional(ast, positional) {
    let i = 0;
    function walk(node) {
      if (node == null || typeof node !== 'object') return node;
      if (Array.isArray(node)) return node.map(walk);
      if (node.type === 'Hole') {
        const arg = positional[i++];
        return arg ? cloneAst(arg) : node;
      }
      const out = {};
      for (const k in node) out[k] = walk(node[k]);
      return out;
    }
    return walk(ast);
  }

  function countHoles(ast) {
    let n = 0;
    (function walk(node) {
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
function isEvaluable(ir) {
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

// =====================================================================
// Derivations: a per-binding description of how to compute its samples.
//
// Where buildSampleChain produces a topologically-ordered execution
// plan for a single target, buildDerivations produces a *dictionary*
// covering every binding the orchestrator can sample. The main thread
// uses it to back a content-addressed sample cache: when the user
// clicks a node, we recursively materialise its samples (and cache
// them), reusing cached arrays for any deps already computed.
//
// Derivation kinds:
//   { kind: 'sample',   distIR }           — sample N from distIR per i
//                                            (kwargs may have refs to
//                                             other binding names —
//                                             those are resolved via
//                                             the cache at compute time)
//   { kind: 'alias',    from: '<name>' }   — share another binding's
//                                            sample array, no fresh draws.
//                                            Used for variates
//                                              theta1 = draw(theta1_dist)
//                                            and lawof aliases
//                                              x = lawof(y)
//   { kind: 'evaluate', ir }               — element-wise deterministic
//                                            compute, e.g. s = mu + 1
//
// The variate-vs-measure semantics live entirely in the alias rule:
// `theta1 = draw(theta1_dist)` becomes alias→theta1_dist, so theta1
// and theta1_dist literally share their cached Float64Array. There is
// never a "second draw" that happens to look statistically the same;
// they are the same array.
//
// Bindings that can't be derived (reified scopes, modules, multivariate
// laws like lawof(record(...)), unsupported distributions) are omitted
// from the result. The visualizer treats absence of a derivation as
// "not plottable".
// =====================================================================

/**
 * Build a derivation dictionary for every chainable binding.
 *
 * @param {Map<string, BindingInfo>} bindings  from analyzer.analyze()
 * @returns {{
 *   derivations: { [name: string]: object },  // alias / sample / evaluate
 *   discrete:    { [name: string]: boolean },  // resolved-leaf discreteness
 * }}
 */
function buildDerivations(bindings) {
  // Pre-pass: lift inline subexpressions so every measure-arg position
  // is a bare ref and every value-arg is evaluable. After lifting, the
  // classifier below handles all forms uniformly — there's no special
  // case for inline weighted/normalize/superpose/draw inside another
  // measure expression.
  bindings = liftInlineSubexpressions(bindings);

  const derivations = Object.create(null);

  // Initial classification — every binding considered independently.
  // We resolve cross-binding ref validity in a follow-up pass so a
  // dropped derivation can cascade: if A depends on B and B becomes
  // unsupported, A also drops.
  for (const [name, binding] of bindings) {
    const d = classifyDerivation(binding, bindings);
    if (d) derivations[name] = d;
  }

  // Fixed-phase pre-evaluation. Walk fixed-phase bindings in topo
  // order and try to compute each one's value end-to-end via the
  // sampler's evaluator (which now handles rnginit / rand / rngstate
  // / tuple_get on top of the existing arithmetic). Two outputs:
  //
  //   - fixedValues: name → JS value. Exposed so consumers (worker,
  //     viewer) can resolve refs to fixed bindings as global env
  //     entries rather than as per-atom slices, which is the only
  //     correct semantics for non-scalar fixed values (e.g. a
  //     length-10 array from `rand(rstate, iid(Normal, 10))`).
  //
  //   - derivation overrides:
  //       * numeric JS array → reclassify as { kind: 'array', values }
  //         so the existing array-plot path renders it (mirrors
  //         the treatment of literal arrays like `[1, 2, 3]`).
  //       * opaque values (rngstate, plain JS objects, non-numeric
  //         arrays) → drop the derivation so the viewer reports
  //         "not plottable" cleanly. The value remains in
  //         fixedValues so downstream evaluators can resolve refs.
  //       * scalar numbers → keep the existing 'evaluate' kind. The
  //         worker's evaluateN runs N iterations with the scalar in
  //         env, producing a Float64Array(N) of the same scalar —
  //         the right shape for a fixed-phase scalar plotted as a
  //         delta. (No reclassification needed: existing path works.)
  //
  // The pass is iterative: each round evaluates any binding whose
  // deps are already in fixedValues, until no progress. Bindings
  // we can't evaluate (refs to non-fixed names, ops outside the
  // evaluator) silently stay at their original classification.
  const fixedValues = new Map();
  const samplerLib = require('./sampler');
  // resolveMeasureRef closure threaded through evaluateExpr → evaluateRand
  // → traceeval. When traceeval hits a `(ref self <name>)` for a
  // measure operand it consults this to recover the measure IR.
  //
  // Two paths here. For named bindings that classify as a measure
  // derivation (sample / record / iid / weighted / alias), use
  // expandMeasureIR — this canonicalises through the derivation
  // graph, turning e.g. `prior = lawof(record(theta1=draw(M1),
  // theta2=draw(M2)))` into the sampleable `joint(theta1=M1,
  // theta2=M2)` shape that traceeval can walk directly. For
  // anonymous lift-introduced bindings or any case expandMeasureIR
  // can't resolve, fall back to the raw lowered IR — those tend to
  // already be primitive distribution calls that traceeval handles.
  function resolveMeasureRef(refName) {
    const expanded = expandMeasureIR(refName, derivations);
    if (expanded) return expanded;
    const b = bindings.get(refName);
    return (b && b.ir) || null;
  }

  // True when a binding's value is a measure. Two ways to know:
  //   1. typeinfer: inferredType.kind in {measure, function, kernel}.
  //   2. lift-introduced synthetic anons that don't carry inferredType
  //      yet but whose IR head is a measure-producing op — exactly
  //      the `MEASURE_PRODUCING` set the surface analyzer uses, so we
  //      reuse it rather than maintaining a parallel list.
  function isMeasureBinding(b) {
    if (!b) return false;
    const t = b.inferredType;
    if (t && (t.kind === 'measure' || t.kind === 'function' || t.kind === 'kernel')) return true;
    if (b.synthetic && b.ir && b.ir.kind === 'call' && b.ir.op
        && MEASURE_PRODUCING.has(b.ir.op)) return true;
    return false;
  }

  let progress = true;
  while (progress) {
    progress = false;
    for (const [name, binding] of bindings) {
      if (fixedValues.has(name)) continue;
      // Pre-eval is restricted to fixed-phase bindings. Lift-
      // synthesised anonymous bindings have phase=undefined; we
      // include them so anonymous *value* bindings (a lifted scalar
      // expression) are computed too. Anonymous *measure* bindings
      // (Normal, iid, ...) will be tried but evaluateExpr will throw
      // on them and we'll silently skip; that's fine — they're
      // resolved later via resolveMeasureRef.
      if (binding.phase != null && binding.phase !== 'fixed') continue;

      // Use the post-lift cached IR if present (set by the lift
      // pass at the bottom of liftInlineSubexpressions); fall back
      // to lowering on demand for bindings the lift never touched.
      const ir = binding.ir
        || (function () {
          try { return lowerExpr(binding.effectiveValue || binding.node.value); }
          catch (_) { return null; }
        })();
      if (!ir) continue;

      // Collect every self-ref in the IR transitively through measure
      // bindings: value-typed refs nested inside a measure subtree
      // (e.g. distribution params like `Normal(mu=ref(rp_field))`)
      // need env entries because traceeval calls evaluateExpr on
      // them at sample time. Measure-typed refs themselves are
      // skipped here — they're resolved via resolveMeasureRef.
      const env = { __resolveMeasureRef: resolveMeasureRef };
      let depsReady = true;
      const seenMeasure = new Set();
      const valueRefs = new Set();

      function collectFor(walkIr) {
        const refs = collectSelfRefs(walkIr);
        for (const r of refs) {
          if (!bindings.has(r)) continue;
          const dep = bindings.get(r);
          // A binding the orchestrator has classified as a measure
          // derivation: recurse through the canonical sampleable
          // expansion (what traceeval actually walks at runtime).
          // expandMeasureIR resolves variate aliases to their
          // distribution and rewrites lawof(record(...)) → joint(...).
          // The resulting tree contains exactly the value refs
          // traceeval will look up in env at sample time.
          if (derivations[r]) {
            if (seenMeasure.has(r)) continue;
            seenMeasure.add(r);
            const expanded = expandMeasureIR(r, derivations);
            if (expanded) { collectFor(expanded); continue; }
            // Not a sample-shape derivation — fall through to
            // value-binding treatment below.
          }
          if (isMeasureBinding(dep)) {
            // Synthetic anon with measure-construction IR but no
            // derivation entry (e.g. dropped during cascade-prune):
            // recurse through the raw IR and hope distribution
            // params reach value bindings we can resolve.
            if (seenMeasure.has(r)) continue;
            seenMeasure.add(r);
            if (dep.ir) collectFor(dep.ir);
            continue;
          }
          valueRefs.add(r);
        }
      }
      collectFor(ir);

      for (const r of valueRefs) {
        const dep = bindings.get(r);
        if (dep.phase != null && dep.phase !== 'fixed') { depsReady = false; break; }
        if (!fixedValues.has(r))                         { depsReady = false; break; }
        env[r] = fixedValues.get(r);
      }
      if (!depsReady) continue;

      // The synthesised disintegrate effectiveValue can be a measure
      // expression (e.g. `lawof(...)`); evaluating that as a value is
      // a category error. Skip cleanly — sampler.evaluateExpr would
      // throw, but iterating across all bindings per analyze means
      // catching cheaply is easier than detecting up front. Same for
      // anonymous bindings whose lifted IR is a measure construction
      // (Normal, etc.).
      let value;
      try {
        value = samplerLib.evaluateExpr(ir, env);
      } catch (_) { continue; }
      fixedValues.set(name, value);
      progress = true;
      // No derivation reclassification here — fixedValues IS the
      // source of truth for the binding's value. The viewer's
      // getMeasure short-circuits any binding present in fixedValues
      // to the appropriate measure shape (Float64Array for numeric
      // arrays / scalars, fields-SoA for records, null for opaque
      // rngstate). Existing derivation kinds set by classifyDerivation
      // (sample / iid / record / weighted / alias / array / …) stay
      // unchanged — the viewer's per-kind paths still apply for
      // bindings without fixedValues entries.
    }
  }

  // Cascade-prune: drop any derivation whose refs aren't satisfiable.
  // Runs AFTER pre-eval so refs to fixed-phase value bindings (whose
  // derivations were dropped because the value is opaque / a record)
  // count as resolvable through fixedValues — without this, a
  // sample derivation like `Normal(mu=get_field(ref(rp), "theta1"))`
  // gets pruned the moment pre-eval drops rp's derivation.
  let changed = true;
  while (changed) {
    changed = false;
    for (const name of Object.keys(derivations)) {
      if (!derivationRefsValid(derivations[name], derivations, bindings, fixedValues)) {
        delete derivations[name];
        changed = true;
      }
    }
  }

  // Discrete map: walk through aliases to find each binding's leaf
  // sample step. evaluate-only bindings inherit the discreteness of
  // their inputs naively, but we treat them as continuous — arithmetic
  // on integer-valued samples produces fractional values via mul/div,
  // and even when it doesn't (a + 1) the user usually wants continuous
  // FD bins for a generic "transformed" view. Toggle-ability via opts
  // is a future refinement.
  const discrete = Object.create(null);
  for (const name of Object.keys(derivations)) {
    discrete[name] = isDiscreteAt(name, derivations);
  }

  // Expose the post-lift bindings alongside derivations so consumers
  // (the viewer's profile-plot path) can call signatureOf without
  // re-running the lift pass. Backward-compatible: existing callers
  // that destructure just { derivations, discrete } are unaffected.
  return { derivations, discrete, bindings, fixedValues };
}

/**
 * Classify a single binding into one of the three derivation kinds,
 * or null if it isn't sample-able under our current support set.
 *
 * The 'draw' case is the interesting one: it can resolve to either an
 * inline distribution call or an alias to another binding (the
 * underlying measure). When the inner is a ref, we emit an alias —
 * NOT a sample. This is what gives variates and their measures the
 * same cached samples.
 */
function classifyDerivation(binding, bindings) {
  if (!binding || !binding.node || !binding.node.value) return null;

  // Read the lowered IR cached by liftInlineSubexpressions. The IR is
  // the canonical "what does this binding compute?" view — surface
  // forms like kernelof and fn have already been lowered to
  // functionof, so the classifier reads one shape per construct
  // instead of pattern-matching every surface variant.
  //
  // The legacy AST is still kept on the binding (binding.node.value
  // and binding.effectiveValue) for source-located helpers that need
  // language-level type judgements (isMeasureExpr,
  // resolveMeasureBaseName) and for things like rename refactoring.
  const rhsIR = binding.ir;
  const rhsAst = binding.effectiveValue || binding.node.value;
  if (!rhsIR) return null;

  if (binding.type === 'draw') {
    if (!rhsIR || rhsIR.kind !== 'call' || rhsIR.op !== 'draw') return null;
    const inner = (rhsIR.args && rhsIR.args[0]) || null;
    if (!inner) return null;
    // draw(<ref>): alias. The samples of the variate ARE the samples
    // of the underlying measure; no extra RNG consumption.
    if (inner.kind === 'ref' && inner.ns === 'self') {
      if (!bindings.has(inner.name)) return null;
      return { kind: 'alias', from: inner.name };
    }
    // draw(<inline-dist-call>): treat the inline dist as if it were
    // a freshly-named anonymous measure binding. We sample directly.
    if (inner.kind === 'call' && inner.op && SAMPLEABLE_DISTRIBUTIONS.has(inner.op)) {
      return { kind: 'sample', distIR: inner };
    }
    return null;
  }

  // 'bayesupdate' produces an importance-reweighted version of the
  // prior: posterior atoms ARE the prior atoms, with logWeights
  // shifted by per-atom log-likelihood. Per spec §sec:bayesupdate,
  //   bayesupdate(L, prior)  ≡  logweighted(fn(logdensityof(L, _)), prior)
  // and per spec §sec:likelihoodof,
  //   logdensityof(likelihoodof(K, obs), theta)  ≡  logdensityof(K(theta), obs)
  // So per atom i: logw_i = logdensityof(K_body[θ_i], obs), evaluated
  // by traceeval.walk on K's body with env carrying the prior's atom
  // and tally='clamped'. We carry that out at materialise time
  // rather than synthesising an intermediate logweighted IR — the
  // walker already implements the lowered primitive.
  if (binding.type === 'bayesupdate') {
    return classifyBayesupdate(binding, bindings);
  }

  // 'lawof' is the dual of 'draw' for our purposes: lawof(<ref>) is
  // the measure that ref's variate is drawn from, so its samples
  // coincide with the ref's samples. lawof(<complex expr>) (e.g.
  // lawof(record(...))) is multivariate and unplottable today.
  if (binding.type === 'lawof') {
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op === 'lawof'
        && rhsIR.args && rhsIR.args.length === 1) {
      const arg = rhsIR.args[0];
      if (arg.kind === 'ref' && arg.ns === 'self' && bindings.has(arg.name)) {
        return { kind: 'alias', from: arg.name };
      }
    }
    return null;
  }

  if (binding.type === 'call' || binding.type === 'literal') {
    // Canonicalise lawof / positional-Dirac before the SAMPLEABLE
    // check, so e.g. `m = Dirac(observed_data)` (positional) and
    // `m = lawof(some_value_binding)` (with value_binding fixed)
    // both classify on Dirac(value=...) — the engine's single
    // canonical form for point-mass measures.
    const normalizedRhsIR = normalizeMeasureIR(rhsIR, bindings);
    // Measure construction: call to a sampleable distribution.
    if (normalizedRhsIR && normalizedRhsIR.kind === 'call' && normalizedRhsIR.op
        && SAMPLEABLE_DISTRIBUTIONS.has(normalizedRhsIR.op)) {
      // Dirac(value = ref-to-binding) is mathematically a plain
      // alias — same equivalence class as lawof(ref-to-binding) —
      // so classify as 'alias' for the lighter, sampler-free path.
      // getMeasure recursively follows the alias chain to the
      // source binding's measure object; sampling never runs and
      // the Dirac REGISTRY's scalar-only limitation is sidestepped.
      // (Without this, `m = Dirac(observed_data)` would hit the
      // sample path with refArrays missing per-atom values for the
      // literal-array source, producing garbage samples.)
      if (normalizedRhsIR.op === 'Dirac'
          && normalizedRhsIR.kwargs && normalizedRhsIR.kwargs.value
          && normalizedRhsIR.kwargs.value.kind === 'ref'
          && normalizedRhsIR.kwargs.value.ns === 'self'
          && bindings.has(normalizedRhsIR.kwargs.value.name)) {
        return { kind: 'alias', from: normalizedRhsIR.kwargs.value.name };
      }
      return { kind: 'sample', distIR: normalizedRhsIR };
    }

    // Measure-algebra ops dispatch through MEASURE_OP_CLASSIFIERS
    // below. Each entry is one tightly-scoped handler that decides the
    // derivation kind (or returns null). New ops add one entry — no
    // edits to this dispatch loop.
    //
    // Operand type-checking still uses the original AST via
    // isMeasureExpr, since "this expression denotes a measure" isn't
    // determinable from bare IR shape (lawof / draw / certain
    // combinators are involved). The lowered IR tells us *which* op
    // we're matching; the AST tells us which operands are measures.
    const ast = binding.node.value;
    if (rhsIR && rhsIR.kind === 'call' && MEASURE_OP_CLASSIFIERS[rhsIR.op]) {
      const result = MEASURE_OP_CLASSIFIERS[rhsIR.op](rhsIR, ast, bindings);
      if (result) return result;
    }
    // Numeric array literal: lowered to (call vector lit lit ...).
    // Treated as static data, not samples — the cache stores the
    // values verbatim (length = array length, not SAMPLE_COUNT) and
    // the plot panel renders an index/value step plot rather than a
    // histogram. We accept only the simplest shape (every entry a
    // numeric lit) to keep the typing trivial; deeper shapes (nested
    // arrays, refs, computed entries) can be added later.
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op === 'vector'
        && Array.isArray(rhsIR.args) && rhsIR.args.length > 0) {
      const values = [];
      let allNumericLits = true;
      for (const a of rhsIR.args) {
        if (a && a.kind === 'lit' && typeof a.value === 'number') {
          values.push(a.value);
        } else {
          allNumericLits = false;
          break;
        }
      }
      if (allNumericLits) return { kind: 'array', values };

      // Array literal whose elements are all self-refs to other
      // bindings — typically the result of liftInlineSubexpressions
      // turning `[draw(M_a), draw(M_b)]` into `[__anon_a, __anon_b]`.
      // Per spec §03/§06, this represents a value of array type whose
      // law is the array-shaped joint of the components' measures —
      // we materialise it as a tuple measure (struct-of-arrays
      // analogue of recordMeasure, but positional). Each ref must
      // have a derivation.
      let allRefs = true;
      const elems = [];
      for (const a of rhsIR.args) {
        if (a && a.kind === 'ref' && a.ns === 'self') {
          elems.push(a.name);
        } else {
          allRefs = false;
          break;
        }
      }
      if (allRefs && elems.length > 0) return { kind: 'tuple', elems };
    }
    // Bare ref to another binding: alias. Common after liftInline
    // hoists a measure-typed RHS into an anon and the user binding
    // becomes `name = ref(__anonN)` (e.g. `expected_obs =
    // forward_kernel(rand_pars)` lifts the substituted joint body
    // out, leaving expected_obs as a bare ref to the joint anon).
    // Without this, the evaluable fallthrough below mis-classifies it
    // as kind:'evaluate' and the per-atom evaluator chokes when the
    // ref target is a measure rather than a value.
    if (rhsIR && rhsIR.kind === 'ref' && rhsIR.ns === 'self'
        && bindings.has(rhsIR.name)) {
      return { kind: 'alias', from: rhsIR.name };
    }
    // Deterministic arithmetic on cached samples.
    if (isEvaluable(rhsIR)) {
      return { kind: 'evaluate', ir: rhsIR };
    }
    return null;
  }

  // Reifications, modules, inputs, joints, likelihoods, bayesupdate: unsupported.
  return null;
}

// =====================================================================
// Measure-algebra op classifiers
// =====================================================================
//
// One handler per IR op whose classification is non-trivial (the
// distribution leaves go through the `sample` shortcut above). Each
// handler receives:
//   irCall    — the binding's lowered IR (a call node with the matching op)
//   ast       — the original RHS AST (for isMeasureExpr operand checks)
//   bindings  — the post-lift bindings map
// and returns a derivation record `{ kind, ... }` or `null` for "not
// classifiable in this shape — fall through to the next attempt".
//
// Adding a new measure op (pushfwd, truncate, relabel, …) is one entry
// in MEASURE_OP_CLASSIFIERS plus the corresponding handler function;
// no edits to the dispatch loop in classifyDerivation.

function classifyWeighted(rhsIR, ast, bindings) {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 2) return null;
  const weightAst = ast.args[0];
  const baseAst   = ast.args[1];
  // After liftInlineSubexpressions the weight slot is either a literal,
  // a ref, or an evaluable arithmetic tree; inline draws have been
  // lifted to synthetic anonymous variates already.
  const weightExpr = rhsIR.args[0];
  const baseName = resolveMeasureBaseName(baseAst, bindings);
  if (baseName == null) return null;
  if (isMeasureExpr(weightAst, bindings)) return null;
  const w = resolveConstant(weightExpr, bindings, new Set());
  if (w != null) {
    if (!(w > 0) || !Number.isFinite(w)) return null;
    return { kind: 'weighted', from: baseName, logShift: Math.log(w) };
  }
  if (isEvaluable(weightExpr)) {
    return { kind: 'weighted', from: baseName, weightIR: weightExpr, isLog: false };
  }
  return null;
}

function classifyLogWeighted(rhsIR, ast, bindings) {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 2) return null;
  const weightAst = ast.args[0];
  const baseAst   = ast.args[1];
  const lwExpr = rhsIR.args[0];
  const baseName = resolveMeasureBaseName(baseAst, bindings);
  if (baseName == null) return null;
  if (isMeasureExpr(weightAst, bindings)) return null;
  const lw = resolveConstant(lwExpr, bindings, new Set());
  if (lw != null) {
    if (!Number.isFinite(lw)) return null;
    return { kind: 'weighted', from: baseName, logShift: lw };
  }
  if (isEvaluable(lwExpr)) {
    return { kind: 'weighted', from: baseName, weightIR: lwExpr, isLog: true };
  }
  return null;
}

function classifyNormalize(rhsIR, ast, bindings) {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 1) return null;
  const baseAst = ast.args[0];
  const baseName = resolveMeasureBaseName(baseAst, bindings);
  if (baseName == null) return null;
  return { kind: 'normalize', from: baseName };
}

function classifySuperpose(rhsIR, ast, bindings) {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length < 1) return null;
  const fromNames = [];
  for (let i = 0; i < rhsIR.args.length; i++) {
    const baseName = resolveMeasureBaseName(ast.args[i], bindings);
    if (baseName == null) return null;
    fromNames.push(baseName);
  }
  return { kind: 'superpose', fromNames };
}

// `record` builds a record-typed value; `joint` builds a measure over
// a record. Both share IR shape (call with `fields:[{name,value},…]`)
// and the same SoA empirical-measure layout downstream — typeinfer
// records the value-vs-measure distinction, the derivation kind unifies.
function classifyRecordOrJoint(rhsIR /*, ast, bindings */) {
  if (!Array.isArray(rhsIR.fields) || rhsIR.fields.length === 0) return null;
  const fields = {};
  for (const f of rhsIR.fields) {
    if (!f.value || f.value.kind !== 'ref' || f.value.ns !== 'self') return null;
    fields[f.name] = f.value.name;
  }
  return { kind: 'record', fields };
}

function classifyIid(rhsIR, ast, bindings) {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length < 2) return null;
  const baseName = resolveMeasureBaseName(ast.args[0], bindings);
  if (baseName == null) return null;
  const dims = [];
  for (let i = 1; i < rhsIR.args.length; i++) {
    const n = resolveConstant(rhsIR.args[i], bindings, new Set());
    if (n == null || !Number.isInteger(n) || n <= 0) return null;
    dims.push(n);
  }
  return { kind: 'iid', from: baseName, dims };
}

// `logdensityof(M, x)` — per spec §sec:posterior, evaluate M's
// log-density at x. Result is REAL (a value, not a measure), but the
// classifier dispatch lives here uniformly: the materialiser computes
// per-prior-atom values via traceeval.walk + tally='clamped', so each
// prior atom θ_i contributes logp = logdensityof(M[θ_i], x). This is
// the same primitive that drives bayesupdate's reweight, just exposed
// as a scalar binding rather than folded into a posterior.
//
// Supported shape (Phase 1):
//   - M is a self-ref to a measure binding (sample / record / iid /
//     algebraic combinator chain — anything expandMeasureIR handles).
//   - x is resolvable to a concrete JS value (literal, array binding,
//     record literal, …) via resolveIRToValue. Variate observations
//     (x is itself a variate) are deferred — they require encoding
//     the observation into refArrays per atom, an extra path the
//     materialiser doesn't yet take.
function classifyLogdensityof(rhsIR, ast, bindings) {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 2) return null;
  const Mref   = rhsIR.args[0];
  const obsIR  = rhsIR.args[1];
  if (!isSelfRef(Mref)) return null;
  if (!bindings.has(Mref.name)) return null;
  const obsValue = resolveIRToValue(obsIR, bindings, new Set());
  if (obsValue === RESOLVE_FAIL) return null;
  return { kind: 'logdensityof', measureName: Mref.name, obsValue };
}

const MEASURE_OP_CLASSIFIERS = {
  weighted:     classifyWeighted,
  logweighted:  classifyLogWeighted,
  normalize:    classifyNormalize,
  superpose:    classifySuperpose,
  record:       classifyRecordOrJoint,
  joint:        classifyRecordOrJoint,
  iid:          classifyIid,
  logdensityof: classifyLogdensityof,
};

/**
 * Whether a derivation's outgoing references are satisfiable —
 * i.e. every 'self' ref it contains points at a binding that itself
 * has a derivation. Aliases / weighted / normalize just check the
 * target.
 */
function derivationRefsValid(d, derivations, bindings, fixedValues) {
  // A name is "resolvable downstream" if there's a derivation for it
  // (the materialiser knows how to compute samples) OR it has a
  // fixed-phase value the worker resolves through its session env.
  // The viewer's collectRefArrays already drops fixed-phase refs from
  // refArrays, so a binding whose only deps are fixed values can
  // still sample correctly via session env. Without this, a Normal(
  // mu=get_field(ref(rp), "theta1"), …) classified as 'sample' would
  // cascade-prune the moment the orchestrator dropped rp's
  // derivation (it's a record, not numeric — pre-eval drops those).
  function resolvable(name) {
    if (Object.prototype.hasOwnProperty.call(derivations, name)) return true;
    if (fixedValues && fixedValues.has(name)) return true;
    return false;
  }

  if (d.kind === 'alias' || d.kind === 'normalize') {
    return resolvable(d.from);
  }
  if (d.kind === 'weighted') {
    if (!resolvable(d.from)) return false;
    // Per-atom path also depends on every binding referenced by its
    // weight expression — those need derivations of their own so the
    // visualPanel can build refArrays for evaluateN.
    if (d.weightIR) {
      for (const r of collectSelfRefs(d.weightIR)) {
        if (!resolvable(r)) return false;
      }
    }
    return true;
  }
  // Superpose: every component must be resolvable.
  if (d.kind === 'superpose') {
    for (const n of d.fromNames) {
      if (!resolvable(n)) return false;
    }
    return true;
  }
  // Record: every field's source binding must be resolvable.
  if (d.kind === 'record') {
    for (const k in d.fields) {
      if (!resolvable(d.fields[k])) return false;
    }
    return true;
  }
  // Tuple: every positional element binding must be resolvable.
  if (d.kind === 'tuple') {
    for (const n of d.elems) {
      if (!resolvable(n)) return false;
    }
    return true;
  }
  // iid: the inner measure must be resolvable.
  if (d.kind === 'iid') {
    return resolvable(d.from);
  }
  if (d.kind === 'bayesupdate') {
    if (!resolvable(d.from)) return false;
    if (d.bodyName) {
      if (!resolvable(d.bodyName)) return false;
      return true;
    }
    if (d.bodyIR) {
      for (const r of collectSelfRefs(d.bodyIR)) {
        if (!resolvable(r)) return false;
      }
      return true;
    }
    return false;
  }
  // Static array literals carry no refs by construction.
  if (d.kind === 'array') return true;
  if (d.kind === 'logdensityof') {
    return resolvable(d.measureName);
  }
  const ir = d.kind === 'sample' ? d.distIR : d.ir;
  const refs = collectSelfRefs(ir);
  for (const r of refs) {
    if (!resolvable(r)) return false;
  }
  return true;
}

/**
 * Resolve an IR node to a constant numeric value, or null if it
 * doesn't reduce. Handles literal numerics, the named built-in
 * constants the evaluator knows, and refs to bindings whose RHS
 * itself reduces to a constant. Used by `weighted` / `logweighted`
 * derivations to pre-compute the log-shift at classification time
 * rather than at sample-render time. Cycle-guarded.
 */
/**
 * Resolve a measure-typed AST argument (the measure operand of
 * weighted / normalize / superpose, etc.) to a binding name we can
 * alias the new derivation to. Returns null when the argument isn't a
 * shape we currently support.
 *
 * Accepts:
 *   - Identifier(<name>) where <name>'s binding is a measure (per
 *     spec §sec:measure-algebra; uses isMeasureExpr to be robust to
 *     lawof bindings, alias chains, distribution constructors,
 *     weighted/normalize/superpose results, etc.)
 *   - CallExpr `lawof(<ident>)` — the spec's identity law gives
 *     `lawof(draw(m)) = m`, and our empirical-measure cache treats
 *     a variate and its underlying measure as the same atoms +
 *     weights, so we alias to the inner ident's cached measure
 *     directly. This is the spec-correct way to lift a value into a
 *     measure on the fly.
 *
 * Inline measure constructions (e.g. `weighted(0.5, Normal(0, 1))`,
 * or chains like `weighted(0.5, normalize(m))`) need anonymous
 * intermediate derivations and are deferred — the user can split
 * them into named bindings for now.
 */
function resolveMeasureBaseName(astNode, bindings) {
  if (!astNode) return null;
  if (astNode.type === 'Identifier' && bindings.has(astNode.name)) {
    return isMeasureExpr(astNode, bindings) ? astNode.name : null;
  }
  if (astNode.type === 'CallExpr'
      && astNode.callee && astNode.callee.type === 'Identifier'
      && astNode.callee.name === 'lawof'
      && Array.isArray(astNode.args) && astNode.args.length === 1) {
    const inner = astNode.args[0];
    if (inner && inner.type === 'Identifier' && bindings.has(inner.name)) {
      return inner.name;
    }
  }
  return null;
}

function resolveConstant(ir, bindings, seen) {
  if (!ir) return null;
  if (ir.kind === 'lit') {
    if (typeof ir.value === 'number' && Number.isFinite(ir.value)) return ir.value;
    return null;
  }
  if (ir.kind === 'const') {
    if (ir.name === 'pi')  return Math.PI;
    if (ir.name === 'e')   return Math.E;
    if (ir.name === 'inf') return Infinity;
    return null;
  }
  if (ir.kind === 'ref' && ir.ns === 'self') {
    if (seen.has(ir.name)) return null;
    seen.add(ir.name);
    const b = bindings.get(ir.name);
    if (!b || !b.node || !b.node.value) return null;
    let bIR;
    try { bIR = lowerExpr(b.node.value); } catch (_) { return null; }
    return resolveConstant(bIR, bindings, seen);
  }
  // Constant-fold small arithmetic. Crucially, the parser lowers a
  // negative literal `-3.5` to `(call neg (lit 3.5))`, so without this
  // we'd fail to recognise plain negative numbers as constants. The
  // operator set matches EVALUABLE_OPS so the language's evaluator
  // semantics agree at this level.
  if (ir.kind === 'call' && ir.op && Array.isArray(ir.args)) {
    const args = ir.args.map(a => resolveConstant(a, bindings, seen));
    if (args.some(v => v == null)) return null;
    switch (ir.op) {
      case 'neg': return args.length === 1 ? -args[0] : null;
      case 'pos': return args.length === 1 ?  args[0] : null;
      case 'add': return args.length === 2 ? args[0] + args[1] : null;
      case 'sub': return args.length === 2 ? args[0] - args[1] : null;
      case 'mul': return args.length === 2 ? args[0] * args[1] : null;
      case 'div': return args.length === 2 ? args[0] / args[1] : null;
      default: return null;
    }
  }
  return null;
}

function isDiscreteAt(name, derivations, visited) {
  visited = visited || new Set();
  if (visited.has(name)) return false; // cycle guard
  visited.add(name);
  const d = derivations[name];
  if (!d) return false;
  if (d.kind === 'alias')     return isDiscreteAt(d.from, derivations, visited);
  if (d.kind === 'weighted')  return isDiscreteAt(d.from, derivations, visited);
  if (d.kind === 'normalize') return isDiscreteAt(d.from, derivations, visited);
  if (d.kind === 'sample')    return DISCRETE_DISTRIBUTIONS.has(d.distIR.op);
  if (d.kind === 'superpose') {
    // A superposition is discrete only if every component is. Mixed
    // discrete/continuous superpositions don't have a clean
    // histogram representation; treating them as continuous (FD
    // bins) is the safer default.
    if (d.fromNames.length === 0) return false;
    for (const n of d.fromNames) {
      if (!isDiscreteAt(n, derivations, new Set(visited))) return false;
    }
    return true;
  }
  return false; // evaluate — see comment in buildDerivations.
}

/**
 * Walk through alias chains to find the underlying sample step's IR.
 * Used to surface the analytical density opportunity for measure
 * bindings: if a binding's leaf step is a sample step with all-literal
 * kwargs, the analytical PDF/PMF from stdlib is callable on that IR.
 *
 * Returns null if the chain doesn't bottom out on a sample step
 * (e.g. it's an evaluate-only binding) or if a cycle is hit.
 */
function leafSampleIR(name, derivations, visited) {
  visited = visited || new Set();
  if (visited.has(name)) return null;
  visited.add(name);
  const d = derivations[name];
  if (!d) return null;
  if (d.kind === 'alias')   return leafSampleIR(d.from, derivations, visited);
  if (d.kind === 'sample')  return d.distIR;
  return null;
}

/**
 * Expand a binding's derivation into a self-contained measure IR
 * suitable for traceeval.walk. Walks the derivation graph,
 * substituting measure refs with their referenced derivations until
 * every internal ref points at a value (not a measure) — those value
 * refs are the names callers need to populate refArrays for during
 * the walk.
 *
 * Used by the visualPanel's bayesupdate materialiser: the kernel
 * body of a likelihood (e.g. `obs_dist`) typically has been lifted
 * by liftInlineSubexpressions into a chain of anonymous measure
 * bindings (record → iid → leaf-distribution). For density
 * evaluation we don't want to materialise samples for each anon —
 * we want one self-contained IR the walker can recurse into. This
 * function does that reconstruction by reading the derivation graph
 * (which already encodes structure of joint/iid/weighted/sample/
 * alias measures) and emitting the corresponding IR call shape.
 *
 * Returns null if the derivation chain hits an unsupported kind
 * (e.g. evaluate, normalize, superpose) — a measure needs to bottom
 * out at sample / alias / sample-via-alias for density evaluation
 * to work today. evaluate-typed bindings are deterministic
 * transforms (no density without a Jacobian, see project notes).
 */
function expandMeasureIR(name, derivations, visited) {
  visited = visited || new Set();
  if (visited.has(name)) return null;
  const next = new Set(visited); next.add(name);
  const d = derivations[name];
  if (!d) return null;
  switch (d.kind) {
    case 'alias':
      return expandMeasureIR(d.from, derivations, next);
    case 'sample':
      // Leaf distribution call — return the distIR verbatim. Refs
      // in its kwargs are value refs (per-i params).
      return d.distIR;
    case 'iid': {
      const inner = expandMeasureIR(d.from, derivations, next);
      if (!inner) return null;
      // dims is a multi-dim shape; flatten to a single iid count.
      // The walker's iid case handles the n-shape uniformly via
      // observed length — multi-dim observations would need to be
      // flattened to match (1D arrays). For typical bayesupdate the
      // dims are 1D and obs is a flat array, which already matches.
      const total = d.dims.reduce((a, b) => a * b, 1);
      return {
        kind: 'call', op: 'iid',
        args: [inner, { kind: 'lit', value: total }],
      };
    }
    case 'record': {
      const fields = [];
      for (const k in d.fields) {
        const inner = expandMeasureIR(d.fields[k], derivations, next);
        if (!inner) return null;
        fields.push({ name: k, value: inner });
      }
      // Use 'joint' op (the measure form). 'record' and 'joint'
      // share the IR shape and the walker treats them equivalently.
      return { kind: 'call', op: 'joint', fields };
    }
    case 'weighted': {
      const inner = expandMeasureIR(d.from, derivations, next);
      if (!inner) return null;
      if (d.weightIR) {
        // Per-i weight expression — the walker resolves its refs
        // through env at evaluation time.
        return {
          kind: 'call',
          op: d.isLog ? 'logweighted' : 'weighted',
          args: [d.weightIR, inner],
        };
      }
      // Constant log-shift was pre-computed; surface as logweighted
      // with a lit weight so the walker just adds it.
      return {
        kind: 'call', op: 'logweighted',
        args: [{ kind: 'lit', value: d.logShift }, inner],
      };
    }
    // evaluate / array / normalize / superpose / iid-of-iid / etc.
    // are not measures-with-densities we can score today.
    default:
      return null;
  }
}

/**
 * Walk a measure IR and replace every measure-position ref with the
 * expanded IR of the binding it points to. Value-position refs (the
 * ones that appear in distribution kwargs as per-i parameters) are
 * left as-is — the walker resolves them via env / refArrays at
 * materialise time.
 *
 * Used when the kernel body of a bayesupdate is an inline expression
 * (e.g. `record(obs = obs)` written directly inside `kernelof(...)`),
 * not a binding name. classifyBayesupdate stores the lowered IR; the
 * visualPanel calls this at materialise time to inline the measure
 * refs through the now-built derivation graph, producing the same
 * fully-self-contained IR shape that expandMeasureIR(name, ...)
 * would have produced.
 *
 * Measure-position slots recognised:
 *   - joint / record fields' value
 *   - iid's first arg (the inner measure)
 *   - weighted / logweighted's second arg (the base measure)
 */
function expandMeasureRefsInIR(ir, derivations, visited) {
  visited = visited || new Set();
  if (!ir) return ir;
  // Top-level ref to a measure binding: expand via the same
  // measure-IR reconstructor used for refs in measure-arg positions.
  // Without this branch, a body that's just `(ref self some_dist)` —
  // typical for `forward_kernel = functionof(obs_dist, …)` where
  // obs_dist is a measure binding — falls through unchanged and
  // downstream consumers (materialiseConcreteMeasure, traceeval.walk)
  // reject the bare ref.
  if (ir.kind === 'ref' && ir.ns === 'self') {
    const expanded = expandMeasureIR(ir.name, derivations, visited);
    return expanded || ir;
  }
  if (ir.kind !== 'call') return ir;
  // `lawof(M)` is a no-op once M is in measure position. The
  // disintegrate rewriter wraps a kernel body's record/joint output
  // in lawof to express "the measure of this value", but for density
  // evaluation by traceeval we want the underlying measure structure
  // — record / joint / iid / weighted / leaf distribution. Peel it
  // before recursing so the resulting IR is one of those shapes.
  if (ir.op === 'lawof' && Array.isArray(ir.args) && ir.args.length === 1) {
    return expandMeasureRefsInIR(ir.args[0], derivations, visited);
  }
  const out = { ...ir };
  if (Array.isArray(ir.fields)) {
    out.fields = ir.fields.map(f => ({
      ...f,
      value: expandMeasurePos(f.value, derivations, visited),
    }));
  }
  if (Array.isArray(ir.args)) {
    if (ir.op === 'iid' && ir.args.length === 2) {
      out.args = [
        expandMeasurePos(ir.args[0], derivations, visited),
        ir.args[1],
      ];
    } else if ((ir.op === 'weighted' || ir.op === 'logweighted') && ir.args.length === 2) {
      out.args = [
        ir.args[0],
        expandMeasurePos(ir.args[1], derivations, visited),
      ];
    }
  }
  return out;
}

function expandMeasurePos(node, derivations, visited) {
  if (node && node.kind === 'ref' && node.ns === 'self') {
    const expanded = expandMeasureIR(node.name, derivations, visited);
    return expanded || node;
  }
  if (node && node.kind === 'call') {
    return expandMeasureRefsInIR(node, derivations, visited);
  }
  return node;
}

// =====================================================================
// bayesupdate classification + obs-AST resolution
// =====================================================================
//
// bayesupdate(L, prior) is detected at the AST level. We resolve the
// chain L → likelihoodof(K, obs) → K → functionof(body, kw...) and
// build a derivation that carries:
//   - `from`:     prior's binding name (provides the atoms; their
//                 samples and shape are reused unchanged)
//   - `bodyName`: name of the kernel body's measure binding. The
//                 visualPanel uses expandMeasureIR(bodyName) to
//                 reconstruct a self-contained measure IR by
//                 walking that binding's derivation chain (record /
//                 iid / weighted / sample / alias).
//   - `obsValue`: a JS value structure mirroring the body's variate
//                 space (number / array / record), built by walking
//                 the obs argument's AST and resolving identifier
//                 refs through the bindings map.
//
// The visualPanel materialiser uses this to issue one
// `worker.logDensityN` call: refArrays are populated from the prior's
// record fields plus any inner-binding samples the body refers to;
// observed = d.obsValue; tally='clamped'. Per-atom log-likelihoods
// come back, and the posterior is a copy of the prior's empirical
// measure with logWeights += those log-likelihoods.
//
// Why classify here and not as an AST rewrite to logweighted? The
// spec lowering `bayesupdate(L, prior) → logweighted(fn(logdensityof(L, _)), prior)`
// works mathematically, but realising it as an IR would require
// extending the evaluator to call traceeval.walk for a
// `logdensityof` op inside a logweighted weightIR. Doing the
// dispatch at the derivation layer is the same in spirit (one
// primitive — the trace walker — handles all density evaluation),
// without introducing a new IR-evaluator call. Future work: lift
// this into a true AST rewrite once we have a worker primitive that
// directly evaluates `logdensityof` calls inside arithmetic IR.
function classifyBayesupdate(binding, bindings) {
  // Walk the L→K chain through cached IR rather than AST. The lowerer
  // canonicalises kernelof → functionof and fn → functionof, so we
  // only need to check for op === 'functionof' here regardless of
  // which surface keyword the user wrote.
  const ir = binding.ir;
  if (!isCallOp(ir, 'bayesupdate', 2)) return null;
  const Lref = ir.args[0];
  const priorRef = ir.args[1];
  if (!isSelfRef(Lref) || !isSelfRef(priorRef)) return null;
  if (!bindings.has(priorRef.name)) return null;

  // Resolve L → likelihoodof(K, obs) at IR level.
  const Lbinding = bindings.get(Lref.name);
  const Lir = Lbinding && Lbinding.ir;
  if (!isCallOp(Lir, 'likelihoodof', 2)) return null;
  const Kref = Lir.args[0];
  const obsIR = Lir.args[1];
  if (!isSelfRef(Kref)) return null;

  // Resolve K → functionof(body, kw=...). Both kernelof and fn lower
  // to functionof, so the IR shape is uniform. The lowerer's
  // _lowerReification stores the body as `Kir.body` (NOT `args[0]`;
  // see lower.js _lowerReification — params/paramKwargs/body sit at
  // the top of the IR node, no `args` array).
  const Kbinding = bindings.get(Kref.name);
  const Kir = Kbinding && Kbinding.ir;
  if (!Kir || Kir.kind !== 'call' || Kir.op !== 'functionof' || !Kir.body) return null;

  // The body has two shapes:
  //   - (ref self <name>) → store bodyName, visualPanel expands via
  //     expandMeasureIR(bodyName, derivations).
  //   - inline call IR → store as bodyIR, visualPanel expands measure
  //     refs in it via expandMeasureRefsInIR(bodyIR, derivations).
  // Both paths converge on the same expanded measure IR for the
  // walker; they differ only in WHERE the body roots in the binding
  // graph.
  const bodyIRArg = Kir.body;
  let bodyName = null;
  let bodyIR = null;
  if (isSelfRef(bodyIRArg)) {
    if (!bindings.has(bodyIRArg.name)) return null;
    bodyName = bodyIRArg.name;
  } else if (bodyIRArg && bodyIRArg.kind === 'call') {
    bodyIR = bodyIRArg;
  } else {
    return null;
  }

  // Resolve obs to a concrete JS value. The trace walker clamps the
  // matching variate sites with this. Translation from IR (literals,
  // record-of-fields, refs to literal-array bindings, etc.) is done
  // by resolveIRToValue; for an Identifier-pointing-at-a-literal-array
  // case we still bottom out via the original binding's AST since the
  // value lives there as a primitive node tree.
  const obsValue = resolveIRToValue(obsIR, bindings, new Set());
  if (obsValue === RESOLVE_FAIL) return null;

  return {
    kind: 'bayesupdate',
    from: priorRef.name,
    bodyName,
    bodyIR,
    obsValue,
  };
}

function isCallOp(ir, op, expectedArgCount) {
  if (!ir || ir.kind !== 'call' || ir.op !== op || !Array.isArray(ir.args)) return false;
  if (expectedArgCount !== null && ir.args.length !== expectedArgCount) return false;
  return true;
}

function isSelfRef(ir) {
  return !!ir && ir.kind === 'ref' && ir.ns === 'self';
}

const RESOLVE_FAIL = Symbol('orchestrator.resolveIRToValue.FAIL');

/**
 * Convert a lowered IR expression to a concrete JS value (number,
 * array of values, plain object) — used to materialise the `observed`
 * argument for bayesupdate's kernel-body density evaluation. Resolves
 * self-refs through the bindings map (recursively, with cycle guard)
 * so an IR like `(call vector (lit 1.2) (lit 3.4) ...)` (which is what
 * `[1.2, 3.4, ...]` lowers to) reduces to a plain JS array.
 *
 * Recognised IR shapes:
 *   - { kind: 'lit', value: <number> }    → number
 *   - { kind: 'call', op: 'vector', args }→ array of resolved elements
 *   - { kind: 'call', op: 'record',
 *       fields: [{name, value}, ...] }    → plain object keyed by field
 *   - { kind: 'call', op: 'neg', args }   → negative
 *   - { kind: 'ref', ns: 'self', name }   → resolve through binding.ir
 *
 * Anything else returns the RESOLVE_FAIL sentinel; callers must
 * propagate that as a "no derivation" outcome rather than try to
 * coerce a partial value.
 */
function resolveIRToValue(ir, bindings, seen) {
  if (!ir || typeof ir !== 'object') return RESOLVE_FAIL;
  if (ir.kind === 'lit' && typeof ir.value === 'number') return ir.value;
  if (ir.kind === 'ref' && ir.ns === 'self') {
    if (seen.has(ir.name)) return RESOLVE_FAIL;
    const b = bindings.get(ir.name);
    if (!b || !b.ir) return RESOLVE_FAIL;
    const next = new Set(seen); next.add(ir.name);
    return resolveIRToValue(b.ir, bindings, next);
  }
  if (ir.kind === 'call') {
    if (ir.op === 'vector' && Array.isArray(ir.args)) {
      const out = new Array(ir.args.length);
      for (let i = 0; i < ir.args.length; i++) {
        const v = resolveIRToValue(ir.args[i], bindings, seen);
        if (v === RESOLVE_FAIL) return RESOLVE_FAIL;
        out[i] = v;
      }
      return out;
    }
    if (ir.op === 'record' && Array.isArray(ir.fields)) {
      const out = {};
      for (const f of ir.fields) {
        const v = resolveIRToValue(f.value, bindings, seen);
        if (v === RESOLVE_FAIL) return RESOLVE_FAIL;
        out[f.name] = v;
      }
      return out;
    }
    if (ir.op === 'neg' && Array.isArray(ir.args) && ir.args.length === 1) {
      const v = resolveIRToValue(ir.args[0], bindings, seen);
      if (v === RESOLVE_FAIL) return RESOLVE_FAIL;
      return -v;
    }
  }
  return RESOLVE_FAIL;
}

/**
 * Collect the names of every (ref self <name>) inside an IR subtree.
 * Used by the worker / main thread to gather upstream sample arrays
 * before drawing or evaluating. Doesn't follow into reified scopes —
 * those introduce their own scope and their bodies aren't part of the
 * outer binding's data dependencies for sampling.
 */
function collectSelfRefs(ir) {
  const seen = new Set();
  walk(ir);
  return seen;
  function walk(node) {
    if (!node || typeof node !== 'object') return;
    if (node.kind === 'ref' && node.ns === 'self') seen.add(node.name);
    if (node.args)   for (const a of node.args)            walk(a);
    if (node.kwargs) for (const k in node.kwargs)          walk(node.kwargs[k]);
    // joint/record IRs use `fields: [{ name, value }, ...]` instead
    // of args/kwargs. Walk values so refs inside joint fields don't
    // get missed.
    if (Array.isArray(node.fields)) for (const f of node.fields) walk(f && f.value);
    if (node.body)                                          walk(node.body);
    // Reified-scope params/paramKwargs are name lists, not IRs.
  }
}

// =====================================================================
// Callable introspection (for the profile-plot UI)
// =====================================================================
//
// signatureOf(name, bindings) returns the canonical input/output
// signature of any reified callable — function, kernel, or
// likelihood — by combining typeinfer's structural types
// (binding.inferredType) with the lowering's paramSources backrefs
// (binding.ir.paramSources, populated in lower.js _lowerReification).
//
// distributeAxes(signature) flattens the input types into the list
// of atomic scalar leaves the profile-plot UI uses for its axis
// dropdown — distributing cartprod (records / tuples) and cartpow
// (static-shape arrays) recursively.
//
// Both return plain JS objects; no mutation, no global state. The
// viewer treats them as a stable shape contract.
//
// Likelihood handling:
//   L = likelihoodof(K, obs) is treated as a function with K's input
//   signature and a real-valued output (the log-likelihood at obs).
//   We dereference to K to get the structural inputs but rewrite the
//   output type and tag kind='likelihood' so the profile evaluator
//   uses logDensityN rather than evaluateExpr.
function signatureOf(name, bindings) {
  if (!bindings) return null;
  const b = bindings.get(name);
  if (!b) return null;

  if (b.type === 'likelihood') return signatureOfLikelihood(b, bindings);

  if (b.type !== 'functionof' && b.type !== 'fn' && b.type !== 'kernelof') {
    return null;
  }
  const ir = b.ir;
  if (!ir || ir.op !== 'functionof') return null;

  // kind comes from inferredType when typeinfer resolved it
  // (function: value body, kernel: measure body). For bindings whose
  // inferredType is deferred — disintegrate-derived bindings, synthesised
  // analyses, etc. — fall back to inspecting the body: a self-ref body
  // pointing at a measure-typed binding makes this a kernel; same for
  // a body that's itself a measure-op call. Otherwise (or when the
  // body resolution fails) default to function.
  const t = b.inferredType;
  let kind;
  if (t && t.kind === 'function')      kind = 'function';
  else if (t && t.kind === 'kernel')   kind = 'kernel';
  else if (b.type === 'kernelof')      kind = 'kernel';
  else                                  kind = bodyImpliesKernel(ir.body, bindings) ? 'kernel' : 'function';

  const params      = ir.params      || [];
  const paramKwargs = ir.paramKwargs || [];
  const sources     = ir.paramSources|| [];
  const inputTypes  = (t && t.inputs)|| [];

  const inputs = [];
  for (let i = 0; i < params.length; i++) {
    // typeinfer's inferReification stores boundary types via
    // expr.kwargs lookup, but the lowered functionof IR doesn't carry
    // the boundary expressions, so all reification params come back
    // typed 'any'. Recover the actual boundary type from the
    // paramSources backref: a binding source's type is its
    // inferredType (e.g. theta1 → real); a placeholder source's
    // type is the corresponding `<name> = elementof(<set>)`
    // binding's inferredType.
    const inferredHere = inputTypes[i] && inputTypes[i].type;
    const fromSource   = resolveSourceType(sources[i] || null, bindings);
    inputs.push({
      paramName: params[i],
      kwargName: paramKwargs[i],
      type:      fromSource || inferredHere || null,
      source:    sources[i] || null,
    });
  }
  return { kind, inputs, output: { type: t && t.result }, body: ir.body || null };
}

// Decide if a reified body returns a measure (→ kernel) by walking
// the body IR. Used as a fallback when typeinfer left the binding
// type as 'deferred'. Conservative: only returns true when we can
// see a measure-shaped op or a self-ref to a measure-typed binding.
const KNOWN_MEASURE_OPS = new Set([
  'joint', 'record', 'iid', 'weighted', 'logweighted', 'normalize',
  'superpose', 'lawof', 'pushfwd', 'truncate', 'mixture',
  // leaf distributions are measures too — we don't enumerate them
  // here; they'd need the full SAMPLEABLE_DISTRIBUTIONS set, but a
  // reified callable typically has its leaf wrapped in a record /
  // joint / lawof anyway.
]);
function bodyImpliesKernel(body, bindings) {
  if (!body) return false;
  if (body.kind === 'call') {
    if (KNOWN_MEASURE_OPS.has(body.op))                  return true;
    if (SAMPLEABLE_DISTRIBUTIONS.has(body.op))           return true;
  }
  if (body.kind === 'ref' && body.ns === 'self' && bindings) {
    const target = bindings.get(body.name);
    if (target && target.inferredType
        && target.inferredType.kind === 'measure') {
      return true;
    }
    // Some derived measure bindings have inferredType='deferred' too;
    // recurse into THEIR body / IR if it's a call we recognise.
    if (target && target.ir) return bodyImpliesKernel(target.ir, bindings);
  }
  return false;
}

// Resolve a paramSources entry to the value type it references.
// Returns null if the source can't be resolved (e.g. placeholder
// without a corresponding elementof binding) — caller falls back
// to typeinfer's per-call type (typically 'any').
function resolveSourceType(source, bindings) {
  if (!source || !bindings) return null;
  const target = bindings.get(source.name);
  if (!target || !target.inferredType) return null;
  // Both binding-source and placeholder-source ultimately want the
  // value type of the bound expression — for a placeholder the
  // typeinfer pass walks elementof's set and tags the binding with
  // the corresponding value type, so the lookup is uniform.
  return target.inferredType;
}

// Likelihood inspection: walk likelihoodof(K, obs) → resolve K to a
// kernel binding → reuse its signature with the output overridden to
// REAL (the log-likelihood at obs) and obsValue stored alongside.
function signatureOfLikelihood(b, bindings) {
  const ir = b.ir;
  if (!ir || ir.op !== 'likelihoodof' || !Array.isArray(ir.args) || ir.args.length !== 2) {
    return null;
  }
  const Kref  = ir.args[0];
  const obsIR = ir.args[1];
  if (!isSelfRef(Kref) || !bindings.has(Kref.name)) return null;
  const inner = signatureOf(Kref.name, bindings);
  if (!inner) return null;
  const obsValue = resolveIRToValue(obsIR, bindings, new Set());
  return {
    kind: 'likelihood',
    inputs: inner.inputs,
    output: { type: { kind: 'scalar', prim: 'real' } },
    obsValue: obsValue === RESOLVE_FAIL ? null : obsValue,
    kernelName: Kref.name,
    // Likelihood evaluation reuses the kernel's body (peeled of any
    // lawof wrapper) at logdensity-mode evaluation time. Carrying the
    // body here lets the viewer dispatch without re-resolving the
    // kernel ref.
    body: inner.body,
  };
}

// Distribute a signature's inputs over their structural shape: one
// axis per scalar leaf. Records emit one axis per field
// (recursively, dot-separated), tuples per element (1-indexed),
// static-shape arrays per slot. Dynamic-shape arrays surface as a
// single non-scalar axis (the UI either rejects them or the user
// has to constrain shape elsewhere).
//
// Each axis carries:
//   - key:        unique stable id within the signature
//                 (e.g. "theta", "theta.mu", "obs[1]", "x[2,3].phi")
//   - label:      same string, used as display
//   - kwargName:  the input this axis belongs to (for fixed-value
//                 substitution at evaluation time)
//   - path:       array of segments — strings for record/tuple-named
//                 fields, numbers (1-indexed) for tuple/array slots
//   - leafType:   the scalar / dynamic-array type at the leaf
//   - source:     paramSources entry for the input (the *whole*
//                 input — UI may need to drill in for record sources)
function distributeAxes(signature) {
  if (!signature || !Array.isArray(signature.inputs)) return [];
  const out = [];
  for (const input of signature.inputs) {
    walkType(input.type || null, [], (path, leafType) => {
      const label = formatAxisLabel(input.kwargName, path);
      out.push({
        key: label,
        label,
        kwargName: input.kwargName,
        path,
        leafType,
        source: input.source,
      });
    });
  }
  return out;
}

function walkType(type, path, emit) {
  if (!type) return;
  if (type.kind === 'scalar') return emit(path, type);
  // Unrestricted placeholder boundaries (`fn(_)` / `_par_ =
  // elementof(anything)`) infer to 'any'. From the UI's standpoint
  // these are unknown scalars — emit a single axis so the user can
  // still profile-sweep along the input. Defaults fall back to the
  // generic-real handling.
  if (type.kind === 'any' || type.kind === 'deferred' || type.kind === 'var') {
    return emit(path, type);
  }
  if (type.kind === 'record' && type.fields) {
    for (const f in type.fields) walkType(type.fields[f], path.concat([f]), emit);
    return;
  }
  if (type.kind === 'tuple' && Array.isArray(type.elems)) {
    for (let i = 0; i < type.elems.length; i++) {
      walkType(type.elems[i], path.concat([i + 1]), emit);
    }
    return;
  }
  if (type.kind === 'array' && Array.isArray(type.shape)) {
    if (type.shape.some(d => d === '%dynamic')) {
      // Dynamic shape: surface a single axis at this level, marked
      // by leafType so the UI can refuse to plot or ask the user
      // for a concrete shape via a preset.
      return emit(path, type);
    }
    walkArraySlots(type, path, emit);
    return;
  }
  // function / kernel / measure / failed / set: not axis-emitting
  // (the UI shouldn't be asked to sweep them).
}

function walkArraySlots(arrayType, path, emit) {
  const dims = arrayType.shape;
  let total = 1;
  for (const d of dims) total *= d;
  for (let s = 0; s < total; s++) {
    let r = s;
    const idx = new Array(dims.length);
    for (let d = dims.length - 1; d >= 0; d--) {
      idx[d] = (r % dims[d]) + 1;  // FlatPPL is 1-indexed
      r = Math.floor(r / dims[d]);
    }
    walkType(arrayType.elem, path.concat([{ idx }]), emit);
  }
}

/**
 * Walk an IR replacing every (ref %local <name>) with a literal of
 * env[name] when env contains a value for that name. Used by the
 * kernel-plot path to substitute preset parameter values into a
 * kernel body before sampling it as a concrete measure.
 *
 * Leaves self-refs intact (those go through the normal materialiser
 * path via expandMeasureRefsInIR + the binding's derivation) and
 * leaves %local refs not in env intact too (defensive — caller
 * should populate env for every param before calling this).
 */
function substituteLocals(ir, env) {
  if (ir == null || typeof ir !== 'object') return ir;
  if (Array.isArray(ir)) return ir.map(function(x) { return substituteLocals(x, env); });
  if (ir.kind === 'ref' && ir.ns === '%local'
      && Object.prototype.hasOwnProperty.call(env, ir.name)) {
    return { kind: 'lit', value: env[ir.name], numType: 'real', loc: ir.loc };
  }
  const out = {};
  for (const k in ir) out[k] = substituteLocals(ir[k], env);
  return out;
}

/**
 * Sibling of substituteLocals for `(ref self <name>)` references.
 * Used by the kernel-sample plot path: after inlineForProfile pulls
 * in deterministic deps and rewrites the kernel's input parameters
 * to %local refs, any remaining self-refs name *captured stochastic
 * or fixed bindings from the outer scope* (e.g. a `sigma ~ Exp(1)`
 * referenced inside the reified kernel body). Those don't get
 * inlined because they aren't deterministic; the caller materializes
 * each captured binding via getMeasure and feeds samples[0] (or the
 * fixed value) in here as `env[<name>] = <scalar>`.
 *
 * Leaves self-refs not in env intact. Any binding still self-ref'd
 * after this call will be the next thing the worker complains about
 * with an "unbound self reference" error — surface it loudly so we
 * fix the missing case rather than silently emit wrong samples.
 */
function substituteSelfRefs(ir, env) {
  if (ir == null || typeof ir !== 'object') return ir;
  if (Array.isArray(ir)) return ir.map(function(x) { return substituteSelfRefs(x, env); });
  if (ir.kind === 'ref' && ir.ns === 'self'
      && Object.prototype.hasOwnProperty.call(env, ir.name)) {
    return { kind: 'lit', value: env[ir.name], numType: 'real', loc: ir.loc };
  }
  const out = {};
  for (const k in ir) out[k] = substituteSelfRefs(ir[k], env);
  return out;
}

function formatAxisLabel(kwargName, path) {
  let s = kwargName || '';
  for (const seg of path) {
    if (typeof seg === 'string')          s += '.' + seg;
    else if (typeof seg === 'number')     s += '[' + seg + ']';
    else if (seg && Array.isArray(seg.idx)) s += '[' + seg.idx.join(',') + ']';
  }
  return s;
}

/**
 * Enumerate the scalar leaves of an output type. Mirrors
 * distributeAxes but for the OUTPUT side of a callable: each
 * leaf is a scalar component the viewer's profile plot can
 * select as its evaluated y-value.
 *
 * For a scalar output type returns a single entry with empty path;
 * the caller reads it as "there's only one output, no Output:
 * dropdown needed".
 *
 *   record { a: real, b: integer }   → [{path:['a'], label:'a',  leafType:real},
 *                                       {path:['b'], label:'b',  leafType:integer}]
 *   tuple (real, real)               → [{path:[1],  label:'[1]', leafType:real},
 *                                       {path:[2],  label:'[2]', leafType:real}]
 *   array<real, [3]>                  → [{path:[{idx:[1]}], label:'[1]', …}, …]
 *   real                              → [{path:[], label:'', leafType:real}]
 *
 * Path segments use the same shape as distributeAxes' input paths
 * — string field name, integer (1-indexed) tuple/array slot, or
 * `{idx:[…]}` for multi-dim array indices. extractOutputIR
 * consumes the same path shape to pull the matching sub-IR out
 * of the body.
 */
function enumerateOutputLeaves(outputType) {
  if (!outputType) return [];
  const out = [];
  walkType(outputType, [], (path, leafType) => {
    out.push({
      key: formatAxisLabel('', path) || '<scalar>',
      label: formatAxisLabel('', path),
      path: path.slice(),
      leafType,
    });
  });
  return out;
}

/**
 * Extract the sub-IR of a body expression at the given output
 * path. Reverses the path navigation enumerateOutputLeaves built:
 * record path segments (field names) descend through `body.fields`,
 * tuple / array integer indices descend through `body.args`,
 * array `{idx:[…]}` segments traverse `body.args` flattened in
 * row-major order.
 *
 *   body = record(a = X, b = Y)        → path ['a']  →  X
 *   body = (call tuple X Y)             → path [1]    →  X
 *   body = (call vector X Y Z)          → path [2]    →  Y  (1-indexed)
 *
 * Returns the original body when path is empty (scalar output).
 * Returns null if the path can't be resolved (e.g. body shape
 * doesn't match the type's shape — should never happen on a
 * type-checked module, but defensive).
 */
function extractOutputIR(bodyIR, path) {
  if (!path || path.length === 0) return bodyIR || null;
  let cur = bodyIR;
  for (const seg of path) {
    if (!cur) return null;
    if (typeof seg === 'string') {
      // Record field. body.fields = [{name, value}, …]
      if (!Array.isArray(cur.fields)) return null;
      const f = cur.fields.find(x => x && x.name === seg);
      if (!f) return null;
      cur = f.value;
      continue;
    }
    if (typeof seg === 'number') {
      // Tuple / 1-D array slot, 1-indexed.
      if (!Array.isArray(cur.args)) return null;
      cur = cur.args[seg - 1];
      continue;
    }
    if (seg && Array.isArray(seg.idx)) {
      // Multi-dim array. Flatten the index in row-major order
      // matching walkArraySlots' (FlatPPL 1-indexed) traversal.
      // Without the array's static shape here we can't fully
      // generalise, but the common case is a single positional
      // index into a 1-D vector body.
      if (!Array.isArray(cur.args)) return null;
      if (seg.idx.length === 1) {
        cur = cur.args[seg.idx[0] - 1];
        continue;
      }
      // For multi-dim arrays we'd need the shape from the type to
      // do the same row-major flatten as walkArraySlots; defer
      // until a concrete model exercises this.
      return null;
    }
    return null;
  }
  return cur || null;
}

/**
 * Resolve the value-set of a paramSources entry to a structural
 * descriptor the profile-plot UI can use to pick an axis range.
 *
 * Returns one of:
 *
 *   { kind: 'interval', lo, hi }   — bounded set; viewer uses [lo, hi]
 *   { kind: 'reals' / 'posreals' / 'nonnegreals' / 'unitinterval' }
 *   { kind: 'integers' / 'posintegers' / 'nonnegintegers' / 'booleans' }
 *   { kind: 'empirical', name }    — binding ref; viewer materialises
 *                                    the binding and computes a 4-σ
 *                                    quantile range
 *   null                           — couldn't resolve; UI falls back
 *                                    to default range for the leaf type
 */
function resolveAxisBaseSet(source, bindings) {
  if (!source) return null;
  // Anonymous placeholder boundaries (`par = _par_`) aren't bound to
  // any elementof — per spec they're equivalent to
  // elementof(anything), so we have no set restriction to surface.
  // The viewer falls back to its leaf-type-based default range.
  if (source.kind === 'placeholder') return null;
  if (source.kind === 'binding') {
    if (!bindings) return { kind: 'empirical', name: source.name };
    const target = bindings.get(source.name);
    if (!target) return { kind: 'empirical', name: source.name };
    // elementof bindings (`x_set = elementof(reals)` /
    // `x_set = elementof(interval(0, 1))`) carry a structural set
    // restriction; surface it so the viewer can use the bounds
    // directly. The analyzer marks these as type='input'.
    if (target.type === 'input') {
      const ir = target.ir
        || (target.effectiveValue && lowerSafe(target.effectiveValue))
        || (target.node && target.node.value && lowerSafe(target.node.value));
      if (ir && ir.kind === 'call' && ir.op === 'elementof'
          && Array.isArray(ir.args) && ir.args.length === 1) {
        const setDescr = parseSetIR(ir.args[0]);
        if (setDescr) return setDescr;
      }
    }
    // Anything else (variates, derived deterministic bindings):
    // there's no static set, but the binding has empirical samples
    // we can quantile-clip into a range at materialise time.
    return { kind: 'empirical', name: source.name };
  }
  return null;
}

function lowerSafe(ast) {
  try { return lowerExpr(ast); } catch (_) { return null; }
}

const NAMED_SETS = {
  reals:           { kind: 'reals' },
  posreals:        { kind: 'posreals' },
  nonnegreals:     { kind: 'nonnegreals' },
  unitinterval:    { kind: 'interval', lo: 0, hi: 1 },
  integers:        { kind: 'integers' },
  posintegers:     { kind: 'posintegers' },
  nonnegintegers:  { kind: 'nonnegintegers' },
  booleans:        { kind: 'booleans' },
};

function parseSetIR(setIR) {
  if (!setIR) return null;
  if (setIR.kind === 'const' && NAMED_SETS[setIR.name])
    return NAMED_SETS[setIR.name];
  if (setIR.kind === 'ref' && setIR.ns === 'self' && NAMED_SETS[setIR.name])
    return NAMED_SETS[setIR.name];
  if (setIR.kind === 'call' && setIR.op === 'interval'
      && Array.isArray(setIR.args) && setIR.args.length === 2) {
    const lo = setIR.args[0], hi = setIR.args[1];
    if (lo && lo.kind === 'lit' && typeof lo.value === 'number'
        && hi && hi.kind === 'lit' && typeof hi.value === 'number') {
      return { kind: 'interval', lo: lo.value, hi: hi.value };
    }
  }
  return null;
}

/**
 * Find record bindings that look like preset points for a callable's
 * input signature. A "preset point" is any global record(...) binding
 * whose kwarg shape matches the callable's input kwargs (spec §03
 * value types: "Any literal (or fixed, in general) global binding
 * `some_name = record(name1=val1, ...)` can be interpreted as a
 * possibly suitable input"). The profile-plot UI uses this to
 * populate its preset-point dropdown — selecting one fills fixedEnv
 * with its values for non-swept axes.
 *
 * Match rule (Phase 1 — strict, top-level scalars only):
 *   - b.ir.op === 'record'
 *   - the set of record kwarg names equals the set of signature
 *     input kwargNames (no missing inputs, no extra record fields)
 *   - every value is constant-resolvable to a finite number via
 *     resolveConstant, after first unwrapping any fixed(...) marker.
 *     resolveConstant folds literals, named constants, and simple
 *     arithmetic (e.g. `-3.5` lowers to `neg(lit 3.5)`).
 *
 * Returns an array of { name, values, fixedNames } where:
 *   - values    : kwargName → JS number (held-constant + sweepable)
 *   - fixedNames: Set<kwargName> for kwargs wrapped in `fixed(...)` —
 *                 the spec's "hold constant during optimization" hint.
 *                 Tooling uses this to e.g. exclude these kwargs from
 *                 the x-axis sweep selector.
 *
 * Future work (deferred): unify nested record / array preset
 * shapes against record-input signatures.
 */
function findMatchingPresets(signature, bindings) {
  if (!signature || !bindings || !Array.isArray(signature.inputs)) return [];
  const expected = new Set();
  for (const inp of signature.inputs) {
    if (inp.kwargName) expected.add(inp.kwargName);
  }
  if (expected.size === 0) return [];
  const out = [];
  for (const [name, b] of bindings) {
    if (!b || !b.ir || b.ir.kind !== 'call' || b.ir.op !== 'record') continue;
    // record's IR carries fields (FIELD_FORM in lower.js), not kwargs.
    // Each field's value is typically a ref to an anon-lifted binding
    // (the lift pre-pass moves literals into __anon* bindings);
    // resolveConstant chases refs through the bindings map to recover
    // the underlying value. Before constant-folding, peek through any
    // `fixed(...)` wrapper so the hint doesn't block the match.
    const fields = Array.isArray(b.ir.fields) ? b.ir.fields : [];
    if (fields.length !== expected.size) continue;
    let allMatch = true;
    const values = {};
    const fixedNames = new Set();
    for (const f of fields) {
      if (!expected.has(f.name)) { allMatch = false; break; }
      let inner = f.value;
      // Unwrap fixed(...) at the top of the field value. The wrapper
      // may be a direct call or a ref to a lifted __anon binding
      // whose IR is the fixed() call. resolveConstantInner handles
      // both because the IR-level resolveConstant chases refs.
      if (inner && inner.kind === 'call' && inner.op === 'fixed'
          && Array.isArray(inner.args) && inner.args.length === 1) {
        fixedNames.add(f.name);
        inner = inner.args[0];
      } else if (inner && inner.kind === 'ref' && inner.ns === 'self') {
        const refTarget = bindings.get(inner.name);
        if (refTarget && refTarget.ir && refTarget.ir.kind === 'call'
            && refTarget.ir.op === 'fixed'
            && Array.isArray(refTarget.ir.args)
            && refTarget.ir.args.length === 1) {
          fixedNames.add(f.name);
          inner = refTarget.ir.args[0];
        }
      }
      const v = resolveConstant(inner, bindings, new Set());
      if (v == null) { allMatch = false; break; }
      values[f.name] = v;
    }
    if (!allMatch) continue;
    out.push({ name, values, fixedNames });
  }
  return out;
}

/**
 * Find cartprod bindings that look like preset domains for a
 * callable's input signature. A "preset domain" is any global
 * cartprod(...) binding whose kwarg shape matches the callable's
 * input kwargs (spec §03 value types: "Any literal/fixed global
 * binding like `some_name = cartprod(name1=some_set, ...)` can be
 * interpreted as a possibly suitable domain"). The viewer uses
 * this to populate a "Domain" dropdown — selecting one sets the
 * x-axis range per kwarg, or falls back to the per-binding auto-
 * fit for kwargs whose field is a bare set name rather than a
 * bounded interval.
 *
 * Match rule:
 *   - b.ir.op === 'cartprod'
 *   - the set of cartprod kwarg names equals the set of signature
 *     input kwargNames
 *   - every field is one of
 *       (a) `interval(lo, hi)` with constant-resolvable numeric
 *           bounds (lo < hi), OR
 *       (b) a bare named-set reference: `reals`, `posreals`,
 *           `nonnegreals`, `unitinterval`, `integers`,
 *           `posintegers`, `nonnegintegers`, `booleans`.
 *     (a) contributes a {lo, hi} entry to `ranges`; (b) does not —
 *     the named set is recorded in `setNames` so tooling can display
 *     it, but it's unbounded for axis-fit purposes, and the viewer
 *     uses the per-axis auto-fit instead.
 *
 * Returns an array of { name, ranges, setNames } where:
 *   - ranges:   kwargName → { lo, hi }       (interval bounds only)
 *   - setNames: kwargName → 'reals' | …      (named-set fields only)
 *
 * Future work (deferred): cartpow for vector inputs; unwrapping
 * fixed(...) wrappers around set fields.
 */
const NAMED_SET_NAMES = new Set([
  'reals', 'posreals', 'nonnegreals', 'unitinterval',
  'integers', 'posintegers', 'nonnegintegers', 'booleans',
]);

function findMatchingDomains(signature, bindings) {
  if (!signature || !bindings || !Array.isArray(signature.inputs)) return [];
  const expected = new Set();
  for (const inp of signature.inputs) {
    if (inp.kwargName) expected.add(inp.kwargName);
  }
  if (expected.size === 0) return [];
  const out = [];
  for (const [name, b] of bindings) {
    if (!b || !b.ir || b.ir.kind !== 'call' || b.ir.op !== 'cartprod') continue;
    const fields = Array.isArray(b.ir.fields) ? b.ir.fields : [];
    if (fields.length !== expected.size) continue;
    let allMatch = true;
    const ranges = {};
    const setNames = {};
    for (const f of fields) {
      if (!expected.has(f.name)) { allMatch = false; break; }
      // Chase a single ref through lifted __anon bindings so the
      // surface form `cartprod(x = reals)` matches whether the lift
      // pass moved the value out or not.
      let inner = f.value;
      if (inner && inner.kind === 'ref' && inner.ns === 'self') {
        const refTarget = bindings.get(inner.name);
        if (refTarget && refTarget.ir) inner = refTarget.ir;
      }
      // (a) interval(lo, hi) with literal-resolvable bounds.
      if (inner && inner.kind === 'call' && inner.op === 'interval'
          && Array.isArray(inner.args) && inner.args.length === 2) {
        const lo = resolveConstant(inner.args[0], bindings, new Set());
        const hi = resolveConstant(inner.args[1], bindings, new Set());
        if (lo == null || hi == null || !(lo < hi)) { allMatch = false; break; }
        ranges[f.name] = { lo, hi };
        continue;
      }
      // (b) bare named-set reference (parser emits these as const
      // refs in the IR — see builtins.SETS).
      const setName = inner && (inner.kind === 'const' || inner.kind === 'ref')
        ? inner.name : null;
      if (setName && NAMED_SET_NAMES.has(setName)) {
        setNames[f.name] = setName;
        continue;
      }
      allMatch = false; break;
    }
    if (!allMatch) continue;
    out.push({ name, ranges, setNames });
  }
  return out;
}

/**
 * Compute a 4-σ-equivalent central quantile range from a sample
 * array. Returns [lo, hi] or null for an empty input.
 *
 * 4-σ on a unit Gaussian covers central probability erf(4/√2) ≈
 * 0.999937, leaving a tail of ~3.17e-5 per side. With sample sizes
 * typical of the visualizer (5000 / 100000) this is essentially
 * min/max; under heavy-tailed empirical distributions it drops the
 * thinnest tails. Used by the profile-plot UI to set an axis range
 * from a binding-source backref's empirical samples.
 */
function fourSigmaQuantileRange(samples) {
  if (!samples || samples.length === 0) return null;
  if (samples.length === 1) return [samples[0], samples[0]];
  const sorted = Float64Array.from(samples);
  sorted.sort();
  // Two-sided tail mass for ±4σ on a unit Normal: (1 - erf(4/√2)) / 2
  // ≈ 3.1671241833e-5. We use the exact constant rather than
  // computing it inline — it's spec-stable per our docs.
  const ALPHA = 3.1671241833e-5;
  return [
    quantileSorted(sorted, ALPHA),
    quantileSorted(sorted, 1 - ALPHA),
  ];
}

/**
 * Substitute IR for the profile-plot evaluator. Two transformations:
 *
 *   1. (ref self <name>) where <name> is a swept input parameter →
 *      (ref %local <name>). The body uses %local refs for its own
 *      params; transitive deps that surface a self-ref to the same
 *      param need to be rewritten so they pick up the swept value
 *      from the worker's env.
 *
 *   2. (ref self <name>) where <name>'s derivation is evaluate-kind
 *      → substitute the binding's lowered IR inline (recursively
 *      processed). Pulls deterministic transforms (e.g. `a = c *
 *      theta1` or `b = abs(theta1) * theta2`) into the body so the
 *      swept axis propagates through them. Constants and other
 *      truly-self-referential bindings (literals, prior atoms) are
 *      left as-is for the viewer's pre-materialise step to bind via
 *      fixedEnv.
 *
 * Used by the profile-plot UI before sending IR to worker.profileN.
 * Without this pass, sweeping `theta1` through a kernel body whose
 * `mu = a` (with `a = c * theta1`) leaves `a` materialised at a
 * single fixed value — the plot shows a flat line because the swept
 * axis doesn't reach the leaf distributions.
 */
function inlineForProfile(ir, paramNames, bindings, derivations) {
  if (!ir) return ir;
  const paramSet = new Set(paramNames || []);
  const visiting = new Set();
  return walk(ir);

  function walk(node) {
    if (node == null || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(walk);
    if (node.kind === 'ref' && node.ns === 'self') {
      // Swept input → %local rewrite.
      if (paramSet.has(node.name)) {
        return { ...node, ns: '%local' };
      }
      // Evaluable binding → inline. Cycle guard: if we're already
      // expanding this name, leave the ref intact (the cycle would
      // be the analyzer's bug, not ours to mask).
      if (derivations && Object.prototype.hasOwnProperty.call(derivations, node.name)
          && derivations[node.name].kind === 'evaluate'
          && !visiting.has(node.name)) {
        const target = bindings && bindings.get(node.name);
        if (target && target.ir) {
          visiting.add(node.name);
          const expanded = walk(target.ir);
          visiting.delete(node.name);
          return expanded;
        }
      }
      // Constant / stochastic / opaque ref — leave for fixedEnv.
      return node;
    }
    // Recurse into structural children: args, fields, kwargs, body.
    const out = { ...node };
    if (Array.isArray(node.args))   out.args   = node.args.map(walk);
    if (Array.isArray(node.fields)) out.fields = node.fields.map(f => ({ ...f, value: walk(f.value) }));
    if (node.kwargs && typeof node.kwargs === 'object') {
      out.kwargs = {};
      for (const k in node.kwargs) out.kwargs[k] = walk(node.kwargs[k]);
    }
    if (node.body) out.body = walk(node.body);
    return out;
  }
}

module.exports = {
  buildSampleChain,
  buildDerivations,
  liftInlineSubexpressions,
  collectSelfRefs,
  leafSampleIR,
  expandMeasureIR,
  expandMeasureRefsInIR,
  signatureOf,
  distributeAxes,
  enumerateOutputLeaves,
  extractOutputIR,
  inlineForProfile,
  substituteLocals,
  substituteSelfRefs,
  resolveAxisBaseSet,
  fourSigmaQuantileRange,
  findMatchingPresets,
  findMatchingDomains,
  // Internal — exported for tests and for visualPanel.js to mirror the
  // gating rules locally if it wants a quick "is this plottable?" check
  // without re-running the full builder.
  SAMPLEABLE_DISTRIBUTIONS,
  DISCRETE_DISTRIBUTIONS,
  EVALUABLE_OPS,
  _internal: { classifyForChain, isEvaluable, classifyDerivation, isDiscreteAt },
};
