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

// Facade re-bind of the leaf IR utilities now living in ir-shared.js.
// ir-shared is the dependency ROOT of the orchestrator split (depends
// only on lower/analyzer + a lazy sampler require); re-binding the
// names here keeps existing internal callers and module.exports
// resolving unchanged.
const {
  resolveMeasureBaseName,
  resolveConstant,
  isCallOp,
  isSelfRef,
  resolveIRToValue,
  valueToPlain,
  collectSelfRefs,
  lowerSafe,
  NAMED_SETS,
  parseSetIR,
  NAMED_SET_NAMES,
  SAMPLEABLE_DISTRIBUTIONS,
  DISCRETE_DISTRIBUTIONS,
  EVALUABLE_OPS,
  normalizeMeasureIR,
  isFixedPhaseValueIR,
} = require('./ir-shared');

// Facade re-bind of the profile-plot UI support now living in
// profile-plan.js. These have zero internal callers in the
// orchestrator (only reached via the public API); the re-bind exists
// solely so module.exports keeps resolving them unchanged.
const {
  resolveAxisBaseSet,
  findMatchingPresets,
  findMatchingDomains,
  fourSigmaQuantileRange,
  inlineForProfile,
} = require('./profile-plan');

// Facade re-bind of the callable-introspection support now living in
// signatures.js (signatureOf / distributeAxes and their helpers).
// Re-bound here so module.exports keeps re-exporting them with the
// public API byte-identical; signatures.js is a leaf depending only
// on ir-shared.
const {
  signatureOf,
  KNOWN_MEASURE_OPS,
  bodyImpliesKernel,
  resolveSourceType,
  signatureOfLikelihood,
  distributeAxes,
  walkType,
  walkArraySlots,
  substituteLocals,
  formatAxisLabel,
  enumerateOutputLeaves,
  extractOutputIR,
} = require('./signatures');

// Facade re-bind of the inline-subexpression lifting pass now living
// in lift.js. buildDerivations (still here / moving to derivations.js)
// calls liftInlineSubexpressions; the classifier shares isEvaluable.
// lift.js is a leaf (lower + signatures + ir-shared only), so the
// re-bind is a one-way edge with no cycle.
const {
  argSignature,
  opUsesValueKwargs,
  inferSyntheticType,
  PLACEHOLDER_SUB_PREFIX,
  canonicalizeImplicitBoundaries,
  bfsImplicitElementofLeavesAst,
  liftInlineSubexpressions,
  isEvaluable,
} = require('./lift');

// Facade re-bind of the derivation builder + classifiers now living in
// derivations.js. The cluster is a leaf w.r.t. the orchestrator core
// (it never calls buildSampleChain / classifyForChain / resolveMeasure),
// so this is a one-way edge with no cycle. Re-bound here so the public
// API (buildDerivations, expandMeasureIR, the _internal test hooks, …)
// keeps resolving byte-identically.
const {
  buildDerivations,
  classifyDerivation,
  classifyWeighted,
  classifyLogWeighted,
  classifyNormalize,
  classifySuperpose,
  classifyRecordOrJoint,
  classifyIid,
  classifyKernelBroadcast,
  classifyLogdensityof,
  classifyTotalmass,
  classifyTruncate,
  classifyPushfwd,
  classifyJointchain,
  MEASURE_OP_CLASSIFIERS,
  derivationRefsValid,
  isDiscreteAt,
  leafSampleIR,
  resolveBijectionMeta,
  expandMeasureIR,
  implicitKernelSignature,
  implicitFunctionSignature,
  _expandMeasureIRStructural,
  expandMeasureRefsInIR,
  expandMeasurePos,
  classifyBayesupdate,
} = require('./derivations');

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

  // type='input' covers both phases of boundary value:
  //   - parameterized (elementof): supplied via env at chain-eval time.
  //   - fixed (external / load_data): supplied at module-init time.
  // In neither case does the chain need a step — the value is
  // pre-bound by the caller. Phase doesn't matter for this decision,
  // only that the binding is a boundary input rather than a value
  // we have to compute.
  if (binding.type === 'input') {
    return null;
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


module.exports = {
  buildSampleChain,
  buildDerivations,
  liftInlineSubexpressions,
  canonicalizeImplicitBoundaries,
  collectSelfRefs,
  leafSampleIR,
  expandMeasureIR,
  expandMeasureRefsInIR,
  resolveIRToValue,
  implicitKernelSignature,
  implicitFunctionSignature,
  signatureOf,
  distributeAxes,
  enumerateOutputLeaves,
  extractOutputIR,
  inlineForProfile,
  substituteLocals,
  resolveAxisBaseSet,
  parseSetIR,
  fourSigmaQuantileRange,
  findMatchingPresets,
  findMatchingDomains,
  // Internal — exported for tests and for visualPanel.js to mirror the
  // gating rules locally if it wants a quick "is this plottable?" check
  // without re-running the full builder.
  SAMPLEABLE_DISTRIBUTIONS,
  DISCRETE_DISTRIBUTIONS,
  EVALUABLE_OPS,
  _internal: {
    classifyForChain, isEvaluable, classifyDerivation, isDiscreteAt,
    // First-class jointchain classifier (exported for a direct
    // structural unit test).
    classifyJointchain,
  },
};
