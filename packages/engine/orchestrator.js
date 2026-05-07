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

// Distributions the worker's REGISTRY currently implements. Hardcoded
// here to avoid pulling sampler.js (and stdlib) into the main bundle.
// Mirrored in sampler.js's REGISTRY; if you add one there, add it here
// too. The orchestrator gates on this list — if a binding's RHS is a
// distribution we don't list, the chain comes back unsupported instead
// of failing later in the worker.
const SAMPLEABLE_DISTRIBUTIONS = new Set([
  'Normal', 'Exponential', 'LogNormal', 'Beta', 'Gamma',
  'Cauchy', 'StudentT', 'Bernoulli', 'Binomial', 'Poisson',
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
  'add', 'sub', 'mul', 'div', 'neg', 'pos',
  'abs', 'exp', 'log', 'log10', 'sqrt',
  'sin', 'cos', 'tan',
  'floor', 'ceil', 'round',
  'pow',
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

    // Lower this binding's RHS expression.
    let rhsIR;
    try {
      rhsIR = lowerExpr(binding.node.value);
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
      order.push({ name, kind: 'evaluate', ir: rhsIR });
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
 * Decide how a single binding contributes to the chain.
 * Returns null if not chainable, otherwise one of:
 *   { kind: 'sample',   distIR }  — sample from distIR per draw
 *   { kind: 'evaluate' }          — call evaluateExpr on the lowered RHS
 *   { kind: 'skip' }              — measure alias; deps walked, no
 *                                    chain step produced for this name
 */
function classifyForChain(binding, rhsIR, bindings) {
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
  if (binding.type === 'call' && rhsIR && rhsIR.kind === 'call' && rhsIR.op
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
    let effLifted = binding.effectiveValue;
    if (effLifted) {
      effLifted = cloneAst(effLifted);
      effLifted = inlineUserCall(effLifted);
      visit(effLifted);
    }
    out.set(name, {
      ...binding,
      node: { ...binding.node, value: cloned },
      effectiveValue: effLifted,
    });
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
    if (astNode.type !== 'CallExpr') return;
    if (!astNode.callee || astNode.callee.type !== 'Identifier') return;

    const op = astNode.callee.name;
    const numArgs = astNode.args ? astNode.args.length : 0;
    const sig = argSignature(op, numArgs);

    // record/joint/jointchain fields: each kwarg value gets lifted to
    // a synthetic binding so the classifier can read them as bare
    // refs. record/jointchain fields are values; joint fields are
    // measures. Both pass through liftMeasure (= "lift non-trivial
    // expression to anon binding") — the distinction lives at type
    // inference, not at lifting.
    const isRecordLike = op === 'record' || op === 'joint' || op === 'jointchain';

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

  function liftMeasure(astArg) {
    if (!astArg) return astArg;
    astArg = inlineUserCall(astArg);
    visit(astArg);
    if (astArg.type === 'Identifier') return astArg;
    const name = freshName();
    out.set(name, makeSyntheticBinding(name, astArg));
    return makeIdent(name, astArg.loc);
  }

  function liftValue(astArg) {
    if (!astArg) return astArg;
    astArg = inlineUserCall(astArg);
    visit(astArg);
    // Only inline `draw(...)` is lifted from a value position — every
    // other value-typed expression (literals, identifiers, arithmetic)
    // is evaluable in place.
    if (astArg.type === 'CallExpr' && astArg.callee
        && astArg.callee.type === 'Identifier' && astArg.callee.name === 'draw') {
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
    // contain at its root) another user call (chained: f then g).
    let prev = null;
    while (astArg !== prev) {
      prev = astArg;
      astArg = inlineOnce(astArg);
    }
    return astArg;
  }

  function inlineOnce(astArg) {
    if (!astArg || astArg.type !== 'CallExpr') return astArg;
    if (!astArg.callee || astArg.callee.type !== 'Identifier') return astArg;
    const fnName = astArg.callee.name;
    const fnBinding = bindings.get(fnName);
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
    // declaration. Internal name is the kwarg's value when it's a
    // bare Identifier (e.g. `par = _par_` → surface 'par', internal '_par_').
    // Anything else (boundary that's a complex expression) — skip;
    // those need real surgery, deferred to a later commit.
    const surfaceOrder = [];
    const internalForSurface = {};
    for (let i = 1; i < fnAst.args.length; i++) {
      const a = fnAst.args[i];
      if (a.type !== 'KeywordArg') continue;
      surfaceOrder.push(a.name);
      if (a.value && a.value.type === 'Identifier') {
        internalForSurface[a.name] = a.value.name;
      } else {
        // Non-trivial boundary (e.g. functionof(body, theta=theta_expr)
        // where theta_expr isn't a bare ref). For now we use the
        // surface name as the internal name — body refs to it will
        // miss the substitution and fall through to outer-scope refs.
        // Real boundary substitution comes later.
        internalForSurface[a.name] = a.name;
      }
    }

    // Walk the call's args. KeywordArg → match by surface name;
    // positional → match by surfaceOrder.
    const argMap = Object.create(null);
    let posIdx = 0;
    for (const a of (astArg.args || [])) {
      if (a.type === 'KeywordArg') {
        const internal = internalForSurface[a.name];
        if (internal) argMap[internal] = a.value;
      } else {
        const surface = surfaceOrder[posIdx++];
        const internal = surface ? internalForSurface[surface] : null;
        if (internal) argMap[internal] = a;
      }
    }

    // Substitute identifiers in a deep-cloned body. Different call
    // sites get independent ASTs.
    return substituteIdents(cloneAst(bodyAst), argMap);
  }

  function substituteIdents(ast, sub) {
    if (ast == null || typeof ast !== 'object') return ast;
    if (Array.isArray(ast)) return ast.map(c => substituteIdents(c, sub));
    if (ast.type === 'Identifier' && sub[ast.name]) {
      // Replace with a clone of the substitute so mutations to either
      // side don't bleed across the boundary.
      return cloneAst(sub[ast.name]);
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
      // All args / kwargs must themselves be evaluable.
      if (ir.args) {
        for (const a of ir.args) if (!isEvaluable(a)) return false;
      }
      if (ir.kwargs) {
        for (const k in ir.kwargs) if (!isEvaluable(ir.kwargs[k])) return false;
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

  // Drop derivations whose refs point to unsatisfiable names. Iterate
  // until stable; one pass isn't enough because removing A might leave
  // B's refs stranded, and so on.
  let changed = true;
  while (changed) {
    changed = false;
    for (const name of Object.keys(derivations)) {
      if (!derivationRefsValid(derivations[name], derivations, bindings)) {
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

  return { derivations, discrete };
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

  // Use the analyzer's rewriter-resolved RHS when it's set. This
  // is the canonical "what does this binding actually compute?" AST:
  //   - Disintegration delegate plan: the delegate target's RHS.
  //   - Disintegration synthesized plan: the synthesized rewrite.
  //   - Otherwise: same as node.value.
  // Reading it uniformly means the orchestrator classifies all forms
  // through one path — no per-rewrite-kind special cases.
  const rhsAst = binding.effectiveValue || binding.node.value;
  let rhsIR;
  try { rhsIR = lowerExpr(rhsAst); } catch (_) { return null; }

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
    // Measure construction: call to a sampleable distribution.
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op
        && SAMPLEABLE_DISTRIBUTIONS.has(rhsIR.op)) {
      return { kind: 'sample', distIR: rhsIR };
    }

    // Measure-algebra ops require *measures* as their measure-typed
    // arguments — passing a value (e.g. `weighted(0.5, theta1)` where
    // theta1 is a draw) is a type error per spec §sec:measure-algebra.
    // We use isMeasureExpr on the original AST args (not the lowered
    // IR) because that helper already encodes all the special cases
    // (lawof / draw / MEASURE_PRODUCING). The orchestrator's lowered-IR
    // matching tells us which OP we're looking at; the AST tells us
    // which OPERANDS are actually measures.
    const ast = binding.node.value;

    // Density reweighting: weighted(<value-expr>, <measure-expr>).
    //   - constant value-expr → uniform log-shift, precomputed here.
    //   - per-atom value-expr → store the IR; evaluate at materialise
    //                           time and add log(w_i) to logWeights[i].
    // Per spec §sec:measure-algebra the first argument MUST be a value
    // (non-negative real). A measure-typed weight is a type error and
    // gets rejected; a value-typed expression that we can't sample (no
    // derivation in scope) also gets rejected.
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op === 'weighted'
        && Array.isArray(rhsIR.args) && rhsIR.args.length === 2) {
      const weightAst  = ast.args[0];
      const baseAst    = ast.args[1];
      // After liftInlineSubexpressions the weight slot's IR is already
      // either a literal/ref or an evaluable arithmetic tree — any
      // inline draws have been lifted to synthetic anonymous variates.
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

    // Log-density reweighting: logweighted(<value-expr>, <measure-expr>).
    // Same two paths as weighted, but the user has already supplied
    // log-weights — we add them in directly with no log() call. Negative
    // and even -Infinity values are valid (probability 0).
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op === 'logweighted'
        && Array.isArray(rhsIR.args) && rhsIR.args.length === 2) {
      const weightAst = ast.args[0];
      const baseAst   = ast.args[1];
      const lwExpr    = rhsIR.args[0];
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

    // Normalisation: normalize(<measure-expr>). Subtracts logSumExp
    // from each weight so the result is a probability measure.
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op === 'normalize'
        && Array.isArray(rhsIR.args) && rhsIR.args.length === 1) {
      const baseAst = ast.args[0];
      const baseName = resolveMeasureBaseName(baseAst, bindings);
      if (baseName == null) return null;
      return { kind: 'normalize', from: baseName };
    }

    // Additive superposition: superpose(<measure-expr>, ...).
    // Per spec §sec:additive-superposition the result is generally
    // not normalised — totals add. Components must be measures (a
    // value summand is a type error).
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op === 'superpose'
        && Array.isArray(rhsIR.args) && rhsIR.args.length >= 1) {
      const fromNames = [];
      for (let i = 0; i < rhsIR.args.length; i++) {
        const argAst = ast.args[i];
        const baseName = resolveMeasureBaseName(argAst, bindings);
        if (baseName == null) return null;
        fromNames.push(baseName);
      }
      return { kind: 'superpose', fromNames };
    }
    // record(name1=val1, ...) builds a record-typed value at every atom.
    // joint(name1=M1, ...) builds a measure over a record (joint of
    // measures). At the EmpiricalMeasure level both produce the same
    // SoA shape (record fields → per-field sub-measures); the
    // type-system distinction (value vs measure) is recorded by
    // typeinfer, not by the derivation kind. We unify them here as
    // `kind: 'record'` and let the materialiser combine the per-field
    // sub-measures.
    //
    // Spec §03 line 126: `record(t)` ↔ `table(r)` auto-conversion is
    // free at this layer — both have the same SoA shape.
    if (rhsIR && rhsIR.kind === 'call' && (rhsIR.op === 'record' || rhsIR.op === 'joint')
        && rhsIR.fields && rhsIR.fields.length > 0) {
      const fields = {};
      for (const f of rhsIR.fields) {
        // Per-field operand: must be a binding ref after lifting.
        // `record` accepts value refs; `joint` accepts measure refs.
        // Either way it's a `(ref self <name>)` to a derivable binding.
        if (!f.value || f.value.kind !== 'ref' || f.value.ns !== 'self') return null;
        fields[f.name] = f.value.name;
      }
      return { kind: 'record', fields };
    }
    // iid(M, n, ...): per spec §sec:iid, a measure over arrays of
    // shape [n, ...] of M-domain values. After lifting, M is a bare
    // ref to a measure binding; each dim is a literal positive
    // integer (refs to integer constants are pinned to %dynamic at
    // type-inference but we need a concrete number to allocate
    // sample buffers — folding via resolveConstant keeps this
    // simple). 1-D iid is the common case (`iid(Normal(0,1), 10)`);
    // higher-rank stays straightforward.
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op === 'iid'
        && Array.isArray(rhsIR.args) && rhsIR.args.length >= 2) {
      const baseAst = ast.args[0];
      const baseName = resolveMeasureBaseName(baseAst, bindings);
      if (baseName == null) return null;
      const dims = [];
      for (let i = 1; i < rhsIR.args.length; i++) {
        const n = resolveConstant(rhsIR.args[i], bindings, new Set());
        if (n == null || !Number.isInteger(n) || n <= 0) return null;
        dims.push(n);
      }
      return { kind: 'iid', from: baseName, dims };
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

/**
 * Whether a derivation's outgoing references are satisfiable —
 * i.e. every 'self' ref it contains points at a binding that itself
 * has a derivation. Aliases / weighted / normalize just check the
 * target.
 */
function derivationRefsValid(d, derivations, bindings) {
  if (d.kind === 'alias' || d.kind === 'normalize') {
    return Object.prototype.hasOwnProperty.call(derivations, d.from);
  }
  if (d.kind === 'weighted') {
    if (!Object.prototype.hasOwnProperty.call(derivations, d.from)) return false;
    // Per-atom path also depends on every binding referenced by its
    // weight expression — those need derivations of their own so the
    // visualPanel can build refArrays for evaluateN.
    if (d.weightIR) {
      for (const r of collectSelfRefs(d.weightIR)) {
        if (!Object.prototype.hasOwnProperty.call(derivations, r)) return false;
      }
    }
    return true;
  }
  // Superpose: every component must be derivable. Empty/missing
  // components were already rejected by classifyDerivation, so we
  // only need the recursive check here.
  if (d.kind === 'superpose') {
    for (const n of d.fromNames) {
      if (!Object.prototype.hasOwnProperty.call(derivations, n)) return false;
    }
    return true;
  }
  // Record: every field's source binding must be derivable.
  if (d.kind === 'record') {
    for (const k in d.fields) {
      if (!Object.prototype.hasOwnProperty.call(derivations, d.fields[k])) return false;
    }
    return true;
  }
  // iid: the inner measure must be derivable.
  if (d.kind === 'iid') {
    return Object.prototype.hasOwnProperty.call(derivations, d.from);
  }
  // Static array literals carry no refs by construction.
  if (d.kind === 'array') return true;
  const ir = d.kind === 'sample' ? d.distIR : d.ir;
  const refs = collectSelfRefs(ir);
  for (const r of refs) {
    if (!Object.prototype.hasOwnProperty.call(derivations, r)) return false;
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
    if (node.body)                                          walk(node.body);
    // Reified-scope params/paramKwargs are name lists, not IRs.
  }
}

module.exports = {
  buildSampleChain,
  buildDerivations,
  collectSelfRefs,
  leafSampleIR,
  // Internal — exported for tests and for visualPanel.js to mirror the
  // gating rules locally if it wants a quick "is this plottable?" check
  // without re-running the full builder.
  SAMPLEABLE_DISTRIBUTIONS,
  DISCRETE_DISTRIBUTIONS,
  EVALUABLE_OPS,
  _internal: { classifyForChain, isEvaluable, classifyDerivation, isDiscreteAt },
};
