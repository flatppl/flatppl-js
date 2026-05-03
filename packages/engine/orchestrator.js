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
  // This list mirrors sampler.js's ARITH_OPS exactly. Comparisons,
  // log/exp/sqrt/abs, etc. are NOT here yet — extend both sides
  // together if you add them.
  'add', 'sub', 'mul', 'div', 'neg', 'pos',
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
  let rhsIR;
  try { rhsIR = lowerExpr(binding.node.value); } catch (_) { return null; }

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
 * has a derivation. Aliases just check the target.
 */
function derivationRefsValid(d, derivations, bindings) {
  if (d.kind === 'alias') {
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

function isDiscreteAt(name, derivations, visited) {
  visited = visited || new Set();
  if (visited.has(name)) return false; // cycle guard
  visited.add(name);
  const d = derivations[name];
  if (!d) return false;
  if (d.kind === 'alias')   return isDiscreteAt(d.from, derivations, visited);
  if (d.kind === 'sample')  return DISCRETE_DISTRIBUTIONS.has(d.distIR.op);
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
