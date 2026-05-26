'use strict';

// derivations.js — the per-binding derivation builder + classifiers.
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

// Leaf w.r.t. the orchestrator core: this cluster never calls back
// into buildSampleChain / classifyForChain / resolveMeasure. Its
// cross-module deps are all leaves (lower, analyzer, builtins,
// ir-shared, lift, signatures), so the orchestrator's facade
// re-bind is a one-way edge.

import type {
  BindingInfo,
  Derivation,
  DerivationAlias,
  DerivationArray,
  DerivationBase,
  DerivationBayesupdate,
  DerivationBroadcastLogdensity,
  DerivationEvaluate,
  DerivationIid,
  DerivationJointchain,
  DerivationKernelBroadcast,
  DerivationLogdensityof,
  DerivationMvNormal,
  DerivationNormalize,
  DerivationPushfwd,
  DerivationRecord,
  DerivationSample,
  DerivationSelect,
  DerivationSuperpose,
  DerivationTotalmass,
  DerivationTruncate,
  DerivationTuple,
  DerivationWeighted,
  IRNode,
} from './engine-types';

const { lowerExpr } = require('./lower.ts');
const { isMeasureExpr } = require('./analyzer.ts');
const { MEASURE_PRODUCING } = require('./builtins.ts');
const { isEvaluable, liftInlineSubexpressions } = require('./lift.ts');
const { signatureOf, substituteLocals } = require('./signatures.ts');
const {
  collectSelfRefs,
  isCallOp,
  isSelfRef,
  resolveConstant,
  resolveIRToValue,
  resolveMeasureBaseName,
  parseSetIR,
  normalizeMeasureIR,
  SAMPLEABLE_DISTRIBUTIONS,
  DISCRETE_DISTRIBUTIONS,
} = require('./ir-shared.ts');

/**
 * Build a derivation dictionary for every chainable binding.
 *
 * @param {Map<string, BindingInfo>} bindings  from analyzer.analyze()
 * @returns {{
 *   derivations: { [name: string]: object },  // alias / sample / evaluate
 *   discrete:    { [name: string]: boolean },  // resolved-leaf discreteness
 * }}
 */
function buildDerivations(bindings: Map<string, BindingInfo>) {
  // Pre-pass: lift inline subexpressions so every measure-arg position
  // is a bare ref and every value-arg is evaluable. After lifting, the
  // classifier below handles all forms uniformly — there's no special
  // case for inline weighted/normalize/superpose/draw inside another
  // measure expression.
  bindings = liftInlineSubexpressions(bindings);

  // After the lift, record bijection metadata on bijection-typed
  // bindings. The classifier and downstream code (matPushfwd's
  // resolveFnBody, density.walkPushfwd) consult
  // `binding.bijection = { fName, fInvName, logVolume }`. fName /
  // fInvName point at lifted function bindings; logVolume is either
  // `{ kind: 'fn', name }` (function binding) or `{ kind: 'scalar',
  // value }` (literal scalar — for volume-preserving maps).
  for (const [, binding] of bindings) {
    if (binding.type !== 'bijection') continue;
    const ast = binding.node && binding.node.value;
    if (!ast || ast.type !== 'CallExpr' || !Array.isArray(ast.args)
        || ast.args.length !== 3) continue;
    const fA = ast.args[0], fIA = ast.args[1], lvA = ast.args[2];
    if (!fA || fA.type !== 'Identifier') continue;
    if (!fIA || fIA.type !== 'Identifier') continue;
    let logVolume;
    if (lvA.type === 'Identifier') {
      logVolume = { kind: 'fn', name: lvA.name };
    } else if (lvA.type === 'NumberLiteral') {
      logVolume = { kind: 'scalar', value: +lvA.value };
    } else {
      // inlineBijectionLift should have lifted any non-trivial shape;
      // unrecognised shape leaves the bijection without metadata and
      // density-side dispatch reports a clear error.
      continue;
    }
    binding.bijection = { fName: fA.name, fInvName: fIA.name, logVolume };
  }

  const derivations = Object.create(null);

  // Initial classification — every binding considered independently.
  // We resolve cross-binding ref validity in a follow-up pass so a
  // dropped derivation can cascade: if A depends on B and B becomes
  // unsupported, A also drops.
  for (const [name, binding] of bindings) {
    // fixedValues hasn't been populated yet — pass undefined so
    // classifyDerivation falls back to its legacy AST-walk for
    // constant resolution. A second classification pass below
    // re-tries unclassified bindings with the populated fixedValues
    // so classifiers like classifyIid can pick up binding-ref size
    // arguments.
    const d = classifyDerivation(binding, bindings);
    if (d) {
      derivations[name] = d;
    }
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
  const samplerLib = require('./sampler.ts');
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
  function resolveMeasureRef(refName: any) {
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
  function isMeasureBinding(b: any) {
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

      // Collect self-refs into TWO buckets based on where they're
      // reached: "value-context" refs that must be in fixedValues
      // before we can evaluate (the classic gate), and "measure-
      // context" refs reached through a measure subtree (e.g.
      // `rand(rstate, ref d)` where d's expansion has
      // `Normal(mu = ref a, sigma = ref b)`). The latter never block
      // pre-eval — they're resolved lazily at evaluation time via
      // __resolveValueRef, which threads rng state through any
      // stochastic ancestor sampling. This is what makes
      //   data, _ = rand(rstate, lawof(obs))
      // pre-eval successfully even when obs's distribution params
      // depend on stochastic ancestors (theta1, theta2): rand owns
      // the state thread, and the resolver hands it the values it
      // needs by walking back through the binding graph the same
      // way traceeval would.
      const env: any = { __resolveMeasureRef: resolveMeasureRef };
      let depsReady = true;
      const seenMeasure = new Set();
      const valueRefs = new Set<string>();
      const deferredRefs = new Set();

      function collectFor(walkIr: any, inMeasureContext: any) {
        const refs = collectSelfRefs(walkIr);
        for (const r of refs) {
          if (!bindings.has(r)) continue;
          const dep = bindings.get(r);
          // Function-typed bindings (fn / functionof / kernelof) aren't
          // looked up as values during evaluation — the ones that flow
          // through evaluator dispatch (filter, broadcast, etc.) get
          // resolved via env.__resolveFnBody at the call site, not
          // through env[name]. Skip them from the value-ref gate so
          // pre-eval doesn't block on them being in fixedValues
          // (they never will be — there's no value to set). But the
          // FUNCTION'S BODY may reference fixed-phase value bindings
          // (e.g. `fn(_ > threshold)` closes over `threshold`); we
          // need those values preloaded into env before evaluating
          // the surrounding call (filter, broadcast). Recurse into
          // the body's refs.
          if (dep && (dep.type === 'fn' || dep.type === 'functionof'
                      || dep.type === 'kernelof')) {
            if (dep.ir && dep.ir.kind === 'call'
                && dep.ir.op === 'functionof' && dep.ir.body) {
              // Walk the body, but only follow refs to value-typed
              // bindings (not refs to the function's own parameters,
              // which are %local). collectSelfRefs already filters
              // for ns === 'self'.
              collectFor(dep.ir.body, inMeasureContext);
            }
            continue;
          }
          // A binding the orchestrator has classified as a measure
          // derivation: recurse through the canonical sampleable
          // expansion (what traceeval actually walks at runtime).
          // expandMeasureIR resolves variate aliases to their
          // distribution and rewrites lawof(record(...)) → joint(...).
          // The resulting tree contains exactly the value refs
          // traceeval will look up in env at sample time — and we
          // mark anything beyond this point as measure-context.
          if (derivations[r]) {
            if (seenMeasure.has(r)) continue;
            seenMeasure.add(r);
            const expanded = expandMeasureIR(r, derivations);
            if (expanded) { collectFor(expanded, true); continue; }
            // Not a sample-shape derivation — fall through to
            // value-binding treatment below.
          }
          if (dep && isMeasureBinding(dep)) {
            // Synthetic anon with measure-construction IR but no
            // derivation entry (e.g. dropped during cascade-prune):
            // recurse through the raw IR and hope distribution
            // params reach value bindings we can resolve.
            if (seenMeasure.has(r)) continue;
            seenMeasure.add(r);
            if (dep.ir) collectFor(dep.ir, true);
            continue;
          }
          if (inMeasureContext) deferredRefs.add(r);
          else                  valueRefs.add(r);
        }
      }
      collectFor(ir, false);

      for (const r of valueRefs) {
        const dep = bindings.get(r);
        if (dep && dep.phase != null && dep.phase !== 'fixed') { depsReady = false; break; }
        if (!fixedValues.has(r))                                { depsReady = false; break; }
        env[r] = fixedValues.get(r);
      }
      if (!depsReady) continue;

      // Build a state-threading resolver closed over the local env
      // and the binding graph. Called from traceeval (via
      // evaluateRand → opts.resolveValueRef) whenever a measure-
      // context ref isn't already in env. Two cases:
      //   - measure-shaped derivation (a draw, an iid, an alias, …):
      //     expandMeasureIR + traceeval.walk samples it through the
      //     same recursive walker. State threads through.
      //   - deterministic derivation (a = c * theta1, etc.): inline
      //     the binding's IR, recursively pre-fill its own refs
      //     through this same resolver, then evaluateExpr.
      // env is the SAME object the outer evaluateExpr will consult,
      // so resolved values are cached implicitly — two refs to the
      // same name share one draw.
      const traceeval = require('./traceeval.ts');
      function localResolveValueRef(refName: any, state: any) {
        if (env[refName] !== undefined) return [env[refName], state];
        if (fixedValues.has(refName)) {
          env[refName] = fixedValues.get(refName);
          return [env[refName], state];
        }
        const dep = bindings.get(refName);
        if (!dep) throw new Error(`resolveValueRef: unknown binding '${refName}'`);
        const d = derivations[refName];
        if (d && (d.kind === 'sample' || d.kind === 'alias'
                  || d.kind === 'iid' || d.kind === 'record'
                  || d.kind === 'weighted')) {
          const measureIR = expandMeasureIR(refName, derivations);
          if (!measureIR) {
            throw new Error(`resolveValueRef: cannot expand measure for '${refName}'`);
          }
          const r = traceeval.walk(state, measureIR, env, {
            resolveMeasureRef,
            resolveValueRef: localResolveValueRef,
          });
          env[refName] = r.value;
          return [r.value, r.state];
        }
        // Deterministic binding (evaluate-kind, lifted anon, …):
        // walk its own refs through this resolver first, then evaluate.
        const innerIR = (d && d.kind === 'evaluate' && d.ir) || dep.ir;
        if (!innerIR) {
          throw new Error(`resolveValueRef: no IR for '${refName}'`);
        }
        // Pre-fill nested refs depth-first by calling ourselves.
        const inner = collectSelfRefs(innerIR);
        for (const n of inner) {
          if (env[n] !== undefined) continue;
          const sub = localResolveValueRef(n, state);
          state = sub[1];
        }
        const v = samplerLib.evaluateExpr(innerIR, env);
        env[refName] = v;
        return [v, state];
      }
      env.__resolveValueRef = localResolveValueRef;

      // filter(pred, data) needs to walk pred's body per element of
      // data. The sampler's evaluateCall picks up this hook to
      // resolve a binding name to its (body IR + parameter name)
      // pair when the binding is a unary fn / functionof / kernelof.
      env.__resolveFnBody = function (bname: any) {
        const fb = bindings.get(bname);
        if (!fb || (fb.type !== 'fn' && fb.type !== 'functionof'
                    && fb.type !== 'kernelof')) {
          return null;
        }
        const fIR = fb.ir;
        if (!fIR || fIR.kind !== 'call' || fIR.op !== 'functionof'
            || !Array.isArray(fIR.params) || !fIR.body) {
          return null;
        }
        // Return all params + surface kwarg names so higher-order
        // callers can name both slots. broadcast's kwargs form needs
        // surface names (paramKwargs); filter / reduce / scan use
        // internal params for env-keying.
        return {
          body:        fIR.body,
          params:      fIR.params,
          paramKwargs: fIR.paramKwargs,
          paramName:   fIR.params[0],
        };
      };

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

  // Second classification pass — pre-eval is now finished and
  // `fixedValues` is populated, so classifiers that depend on
  // constant-resolution of a fixed-phase binding (e.g. classifyIid
  // resolving `iid(M, n)` where `n = lengthof(data)` is a fixed-phase
  // binding) can succeed here even though the first pass at line 145
  // ran with an empty fixedValues map. Only re-classify bindings that
  // didn't already get a derivation in pass 1 — pre-eval may have
  // overridden some derivations (e.g. converting a `vector(...)`
  // sample call to an `array` kind), which we don't want to clobber.
  for (const [name, binding] of bindings) {
    if (derivations[name]) continue;
    const d = classifyDerivation(binding, bindings, fixedValues);
    if (d) derivations[name] = d;
  }

  // Classification diagnostics. "No derivation" is a heavily
  // overloaded state — inputs (`elementof`), callables (`functionof`,
  // `kernelof`, `bijection`), likelihood objects, and parameterized-
  // /stochastic-phase variates all legitimately have none, and the
  // cascade-prune below routinely (by design) drops parameterized
  // stochastic derivations that the viewer then re-plots via the
  // implicit-`kernelof` escape hatch. So a "dropped derivation" is
  // NOT a failure signal — testing proved it false-positives on every
  // ordinary `x ~ Normal(mu = elementof, …)` model.
  //
  // The one UNAMBIGUOUS silent-failure mode is the fixed-phase dead
  // end: a fixed-phase VALUE computation that ends with neither a
  // fixedValues entry (pre-eval gave up) nor a derivation (classifier
  // gave up). A deterministic expression that produces nothing is an
  // engine gap, not a modelling choice. (Fixed phase rules out draws;
  // callable / measure-object binding types are excluded — those are
  // legitimately underived.) This precisely names the root cause —
  // e.g. `bcadd = broadcasted(add)` in the explicit-`broadcasted` +
  // `disintegrate` model whose whole downstream graph silently
  // vanished — instead of the user hitting a confusing plot-time
  // error far from the cause. The broader stochastic-side overloading
  // is real debt tracked for the derivation-kind unification refactor.
  const diagnostics: any[] = [];
  const OBJECT_BINDING_TYPES = new Set([
    'input', 'functionof', 'fn', 'kernelof', 'bijection', 'lawof',
    'likelihood',
  ]);
  function bindingLoc(name: any) {
    const b = bindings.get(name);
    return (b && b.node && b.node.loc) || undefined;
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

  // Fixed-phase dead end (mode b). A fixed-phase value computation
  // must end up either pre-evaluated (fixedValues) or classified
  // (derivations); neither means the engine silently gave up on a
  // deterministic computation.
  for (const [name, b] of bindings) {
    if (!b || b.phase !== 'fixed') continue;
    if (OBJECT_BINDING_TYPES.has(b.type)) continue;     // legit underived
    if (derivations[name]) continue;
    if (fixedValues.has(name)) continue;
    diagnostics.push({
      severity: 'error',
      message: `Fixed-phase binding '${name}' produced no value: the engine `
        + `could neither evaluate it (pre-eval) nor classify it `
        + `(derivation). This is an engine gap — the expression is `
        + `deterministic but unsupported. Plotting '${name}' or anything `
        + `depending on it will fail.`,
      loc: bindingLoc(name),
    });
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
  return { derivations, discrete, bindings, fixedValues, diagnostics };
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
function classifyDerivation(
  binding: BindingInfo, bindings: Map<string, BindingInfo>, fixedValues?: any,
): Derivation | null {
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
  // coincide with the ref's samples.
  //
  // The binding.type === 'lawof' tag identifies surface-level
  // lawof(...) bindings AND `disintegrate(...)` result prior bindings
  // (the analyzer tags both with type='lawof' in pass 3). The
  // disintegrate-result prior's effective RHS may be a non-lawof
  // measure expression — e.g. `joint(inner = inner)` synthesised
  // by the joint-shape decomposition. So when the IR isn't a
  // straight lawof, fall through to the MEASURE_OP_CLASSIFIERS
  // dispatch (joint, jointchain, etc.) below so the prior gets the
  // appropriate derivation kind.
  if (binding.type === 'lawof') {
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op === 'lawof'
        && rhsIR.args && rhsIR.args.length === 1) {
      const arg = rhsIR.args[0];
      if (arg.kind === 'ref' && arg.ns === 'self' && bindings.has(arg.name)) {
        return { kind: 'alias', from: arg.name };
      }
      // lawof(<non-ref>) — falls through to the generic dispatch.
    }
    // Fall through to MEASURE_OP_CLASSIFIERS for non-lawof IR shapes
    // produced by disintegrate's effective-value rewrite.
  }

  if (binding.type === 'call' || binding.type === 'literal' || binding.type === 'lawof') {
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

    // Multivariate sampleable distributions go through dedicated kind
    // handlers (matMvNormal etc.) — they produce vector atoms
    // (shape=[N, n]) rather than scalar atoms, and use closed-form
    // density walkers (walkMvNormal etc.) instead of the per-leaf
    // logpdf dispatch in walkLeaf. 
    if (normalizedRhsIR && normalizedRhsIR.kind === 'call'
        && normalizedRhsIR.op === 'MvNormal') {
      return { kind: 'mvnormal', distIR: normalizedRhsIR };
    }
    if (normalizedRhsIR && normalizedRhsIR.kind === 'call'
        && normalizedRhsIR.op === 'Dirichlet') {
      return { kind: 'dirichlet', distIR: normalizedRhsIR };
    }
    if (normalizedRhsIR && normalizedRhsIR.kind === 'call'
        && normalizedRhsIR.op === 'Multinomial') {
      return { kind: 'multinomial', distIR: normalizedRhsIR };
    }
    if (normalizedRhsIR && normalizedRhsIR.kind === 'call'
        && normalizedRhsIR.op === 'Wishart') {
      return { kind: 'wishart', distIR: normalizedRhsIR };
    }
    if (normalizedRhsIR && normalizedRhsIR.kind === 'call'
        && normalizedRhsIR.op === 'InverseWishart') {
      return { kind: 'inversewishart', distIR: normalizedRhsIR };
    }
    if (normalizedRhsIR && normalizedRhsIR.kind === 'call'
        && normalizedRhsIR.op === 'LKJCholesky') {
      return { kind: 'lkjcholesky', distIR: normalizedRhsIR };
    }
    if (normalizedRhsIR && normalizedRhsIR.kind === 'call'
        && normalizedRhsIR.op === 'LKJ') {
      return { kind: 'lkj', distIR: normalizedRhsIR };
    }
    if (normalizedRhsIR && normalizedRhsIR.kind === 'call'
        && normalizedRhsIR.op === 'BinnedPoissonProcess') {
      return { kind: 'binnedpoissonprocess', distIR: normalizedRhsIR };
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
    if (rhsIR && rhsIR.kind === 'call' && rhsIR.op != null
        && (MEASURE_OP_CLASSIFIERS as any)[rhsIR.op]) {
      const result = (MEASURE_OP_CLASSIFIERS as any)[rhsIR.op](
        rhsIR, ast, bindings, fixedValues);
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
      const values: any[] = [];
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
      const elems: any[] = [];
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

function classifyWeighted(
  rhsIR: IRNode, ast: any, bindings: any, fixedValues?: any,
): DerivationWeighted | null {
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
  const w = resolveConstant(weightExpr, bindings, new Set(), fixedValues);
  if (w != null) {
    if (!(w > 0) || !Number.isFinite(w)) return null;
    return { kind: 'weighted', from: baseName, logShift: Math.log(w) };
  }
  if (isEvaluable(weightExpr)) {
    return { kind: 'weighted', from: baseName, weightIR: weightExpr, isLog: false };
  }
  return null;
}

function classifyLogWeighted(
  rhsIR: IRNode, ast: any, bindings: any, fixedValues?: any,
): DerivationWeighted | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 2) return null;
  const weightAst = ast.args[0];
  const baseAst   = ast.args[1];
  const lwExpr = rhsIR.args[0];
  const baseName = resolveMeasureBaseName(baseAst, bindings);
  if (baseName == null) return null;
  if (isMeasureExpr(weightAst, bindings)) return null;
  const lw = resolveConstant(lwExpr, bindings, new Set(), fixedValues);
  if (lw != null) {
    if (!Number.isFinite(lw)) return null;
    return { kind: 'weighted', from: baseName, logShift: lw };
  }
  if (isEvaluable(lwExpr)) {
    return { kind: 'weighted', from: baseName, weightIR: lwExpr, isLog: true };
  }
  return null;
}

function classifyNormalize(rhsIR: IRNode, ast: any, bindings: any): DerivationNormalize | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 1) return null;
  const baseAst = ast.args[0];
  const baseName = resolveMeasureBaseName(baseAst, bindings);
  if (baseName == null) return null;
  return { kind: 'normalize', from: baseName };
}

function classifySuperpose(rhsIR: IRNode, ast: any, bindings: any): DerivationSuperpose | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length < 1) return null;
  const fromNames: any[] = [];
  for (let i = 0; i < rhsIR.args.length; i++) {
    const baseName = resolveMeasureBaseName(ast.args[i], bindings);
    if (baseName == null) return null;
    fromNames.push(baseName);
  }
  return { kind: 'superpose', fromNames };
}

// Resolve a condition IR to a Bernoulli success-probability value-IR
// (the closed-form selector weight: P(true)=p, P(false)=1−p).
// Follows self-refs / draw / lawof down to a `Bernoulli(p)` call and
// returns its `p` value-IR. Returns null when P(true) isn't closed-
// form (comparisons of continuous RVs, arbitrary boolean expressions,
// …) — classifyIfelse then declines, leaving the MC-estimated-weight
// fallback as a documented follow-up (engine-concepts §11).
function resolveBernoulliP(ir: any, bindings: any, seen: any) {
  if (!ir || seen.size > 64) return null;
  if (ir.kind === 'ref' && ir.ns === 'self') {
    if (seen.has(ir.name)) return null;
    seen.add(ir.name);
    const b = bindings.get(ir.name);
    if (!b || !b.ir) return null;
    return resolveBernoulliP(b.ir, bindings, seen);
  }
  if (ir.kind === 'call') {
    if ((ir.op === 'draw' || ir.op === 'lawof')
        && Array.isArray(ir.args) && ir.args.length === 1) {
      return resolveBernoulliP(ir.args[0], bindings, seen);
    }
    if (ir.op === 'Bernoulli') {
      if (ir.kwargs && ir.kwargs.p) return ir.kwargs.p;
      if (Array.isArray(ir.args) && ir.args.length === 1) return ir.args[0];
    }
  }
  return null;
}

// Resolve an index IR to a Categorical selector: { pIR, base } where
// pIR is the probability-vector value-IR and base is 1 (Categorical,
// spec 1-based) or 0 (Categorical0). Follows self-refs / draw / lawof
// to the Categorical call. null when not a closed-form Categorical
// index (→ classifyStochasticIndex declines; plain value indexing
// stays a deterministic `get`).
function resolveCategoricalP(ir: any, bindings: any, seen: any) {
  if (!ir || seen.size > 64) return null;
  if (ir.kind === 'ref' && ir.ns === 'self') {
    if (seen.has(ir.name)) return null;
    seen.add(ir.name);
    const b = bindings.get(ir.name);
    if (!b || !b.ir) return null;
    return resolveCategoricalP(b.ir, bindings, seen);
  }
  if (ir.kind === 'call') {
    if ((ir.op === 'draw' || ir.op === 'lawof')
        && Array.isArray(ir.args) && ir.args.length === 1) {
      return resolveCategoricalP(ir.args[0], bindings, seen);
    }
    if (ir.op === 'Categorical' || ir.op === 'Categorical0') {
      const base = ir.op === 'Categorical' ? 1 : 0;
      if (ir.kwargs && ir.kwargs.p) return { pIR: ir.kwargs.p, base };
      if (Array.isArray(ir.args) && ir.args.length === 1) {
        return { pIR: ir.args[0], base };
      }
    }
  }
  return null;
}

// Stochastic-phase array indexing — the draw-style spelling of a
// discrete mixture (engine-concepts §11):
//
//   i  ~ Categorical(w)
//   xs = [draw(M1), draw(M2), …]      # a `tuple` of variates
//   x  = xs[i]                         # get(xs, i), i stochastic
//
// is exactly the K-branch select: branches = xs's component measures,
// selector = i, weight_k = w_k. Recognised here so it rides the SAME
// core as ifelse/superpose/mixture (no parallel path). Declines
// (null) unless the container is a vector/tuple of self-refs AND the
// index resolves to a closed-form Categorical — plain deterministic
// `xs[k]` indexing is untouched.
function classifyStochasticIndex(rhsIR: IRNode, ast: any, bindings: any): DerivationSelect | null {
  if (rhsIR.op !== 'get' || !Array.isArray(rhsIR.args)
      || rhsIR.args.length !== 2) return null;
  const containerIR = rhsIR.args[0];
  const indexIR = rhsIR.args[1];
  if (!containerIR || containerIR.kind !== 'ref' || containerIR.ns !== 'self') {
    return null;
  }
  const cb = bindings.get(containerIR.name);
  if (!cb || !cb.ir || cb.ir.kind !== 'call' || cb.ir.op !== 'vector'
      || !Array.isArray(cb.ir.args) || cb.ir.args.length === 0) return null;
  const branches: any[] = [];
  for (const el of cb.ir.args) {
    if (!el || el.kind !== 'ref' || el.ns !== 'self') return null;
    branches.push({ ref: el.name });
  }
  const cat = resolveCategoricalP(indexIR, bindings, new Set());
  if (!cat) return null;
  const K = branches.length;
  const selectorRef = (indexIR.kind === 'ref' && indexIR.ns === 'self')
    ? indexIR.name : null;
  // Per-branch log-weights from the Categorical pmf. A literal weight
  // vector folds to per-element log lits; otherwise index the weight
  // IR per branch (base-aware: Categorical 1-based, Categorical0 0).
  const pIR = cat.pIR;
  const litVec = (pIR.kind === 'call' && pIR.op === 'vector'
    && Array.isArray(pIR.args) && pIR.args.length === K) ? pIR.args : null;
  const logweightIRs: any[] = [];
  for (let k = 0; k < K; k++) {
    const elem = litVec
      ? litVec[k]
      : { kind: 'call', op: 'get',
          args: [pIR, { kind: 'lit', value: cat.base + k }] };
    logweightIRs.push({ kind: 'call', op: 'log', args: [elem] });
  }
  return {
    kind: 'select',
    branches,
    logweightIRs,
    selectorRef,
    selectorBase: cat.base,
    marginalize: true,
    mode: 'mixture',
  };
}

// ifelse(cond, a, b) over MEASURES — the 2-branch discrete-selector
// mixture (engine-concepts §11). Classifies to the shared `select`
// kind: branch a is taken when cond is true (prob p), b when false
// (prob 1−p), so the marginal (selector-anonymous) density is the
// exact mixture logsumexp([log p + logp_a, log(1−p) + logp_b]).
//
// Scope (first pass): branches must be NAMED measure bindings (the
// canonical `a = Normal(…); b = Normal(…); m = ifelse(c, a, b)`
// form) and the condition must resolve to a Bernoulli probability
// (closed-form weight). Inline-measure branches and non-closed-form
// conditions are documented deferrals; classifyIfelse returns null
// for them, so value-valued ifelse stays on the evaluator path
// untouched.
function classifyIfelse(rhsIR: IRNode, ast: any, bindings: any): DerivationSelect | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 3) return null;
  if (!ast || !Array.isArray(ast.args) || ast.args.length !== 3) return null;
  // A branch is either a NAMED measure binding (resolveMeasureBaseName)
  // or an INLINE sampleable-distribution leaf call — the common
  // `ifelse(c, Normal(0,1), Normal(5,1))` form. Inline *composite*
  // measures stay unsupported here (use named bindings for those);
  // value-valued ifelse resolves to null on both and stays on the
  // evaluator path. Unified branch shape: { ref } | { ir }.
  const resolveBranch = (astArg: any, irArg: any) => {
    const nm = resolveMeasureBaseName(astArg, bindings);
    if (nm != null) return { ref: nm };
    if (irArg && irArg.kind === 'call'
        && SAMPLEABLE_DISTRIBUTIONS.has(irArg.op)) return { ir: irArg };
    return null;
  };
  const aB = resolveBranch(ast.args[1], rhsIR.args[1]);
  const bB = resolveBranch(ast.args[2], rhsIR.args[2]);
  if (!aB || !bB) return null;
  const call = (op: any, args: any): IRNode => ({ kind: 'call', op, args });
  const lit1 = { kind: 'lit', value: 1 };
  // Realised selector for SAMPLING (matSelect): the condition binding
  // — a {0,1} Bernoulli variate. Density only needs the per-branch
  // log-weights (selector marginalised); generation needs the
  // per-atom realised condition. When the condition isn't a bare
  // self-ref, selectorRef is null → no materialisable selector.
  const cond = rhsIR.args[0];
  const selectorRef = (cond && cond.kind === 'ref' && cond.ns === 'self')
    ? cond.name : null;
  const pIR = resolveBernoulliP(rhsIR.args[0], bindings, new Set());
  if (pIR == null) {
    // Non-closed-form condition (comparisons of continuous RVs,
    // arbitrary boolean expressions, …). The mixture is still
    // STRUCTURALLY exact — only the selector probability P(true)
    // lacks a closed form. If the condition is a materialisable
    // {0,1} self-ref we estimate P(true) ONCE from its sampled
    // ensemble at materialisation time (engine-concepts §11
    // MC-weight selector): density then uses the constant logweights
    // [log p̂, log(1−p̂)], the exact discrete-mixture form with an
    // estimated weight rather than an estimated structure. Sampling
    // never needed P(true) at all (matSelect gathers by the realised
    // condition). Without a materialisable selector we can do
    // neither → decline, so value-valued ifelse and opaque
    // conditions stay on the evaluator path untouched.
    if (selectorRef == null) return null;
    return {
      kind: 'select',
      branches: [aB, bB],
      // Weights deferred to materialisation: p̂_0 = P(cond TRUE) =
      // empirical frequency of the {0,1} selector being truthy
      // (branch 0 = the TRUE branch, matching matSelect's sel?0:1
      // gather), p̂_1 = 1 − p̂_0. The materialiser's runtime-weight
      // resolver fills the select node's logweights from this spec.
      logweightIRs: null,
      runtimeWeights: { ref: selectorRef, K: 2, base: 0 },
      selectorRef,
      marginalize: true,
      mode: 'mixture',
    };
  }
  return {
    kind: 'select',
    branches: [aB, bB],
    // log P(true)=log p ; log P(false)=log(1−p). Constant in the
    // observation point; walkSelect evaluates these per atom.
    logweightIRs: [
      call('log', [pIR]),
      call('log', [call('sub', [lit1, pIR])]),
    ],
    selectorRef,
    marginalize: true,
    mode: 'mixture',
  };
}

// `record` builds a record-typed value; `joint` builds a measure over
// a record. Both share IR shape (call with `fields:[{name,value},…]`)
// and the same SoA empirical-measure layout downstream — typeinfer
// records the value-vs-measure distinction, the derivation kind unifies.
//
// Positional joint (`joint(M1, M2, ...)`) is the same measure-algebra
// construction (independent product) but produces a positional shape
// rather than a named-field record. Per spec §06: "all components
// must have the same shape class — all scalars yields a vector, all
// vectors yields a concatenated vector, all records (with distinct
// fields) yields a merged record." Today we map all-scalar positional
// joint to the same `tuple` derivation kind used for array literals
// of measure refs; downstream matTuple materialises a positional
// EmpiricalMeasure (SoA across the components).
function classifyRecordOrJoint(rhsIR: any /*, ast, bindings */): DerivationRecord | DerivationTuple | null {
  if (Array.isArray(rhsIR.fields) && rhsIR.fields.length > 0) {
    const fields: Record<string, any> = {};
    for (const f of rhsIR.fields) {
      if (!f.value || f.value.kind !== 'ref' || f.value.ns !== 'self') return null;
      fields[f.name] = f.value.name;
    }
    return { kind: 'record', fields };
  }
  if (Array.isArray(rhsIR.args) && rhsIR.args.length > 0) {
    // Positional joint: every arg must already be a self-ref to a
    // measure binding (liftInlineSubexpressions lifts inline measure
    // expressions into anon bindings before classification).
    const elems: any[] = [];
    for (const a of rhsIR.args) {
      if (!a || a.kind !== 'ref' || a.ns !== 'self') return null;
      elems.push(a.name);
    }
    return { kind: 'tuple', elems };
  }
  return null;
}

function classifyIid(
  rhsIR: IRNode, ast: any, bindings: any, fixedValues?: any,
): DerivationIid | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length < 2) return null;
  const baseName = resolveMeasureBaseName(ast.args[0], bindings);
  if (baseName == null) return null;
  // New spec: iid(M, size) where size is a positive int or vector of
  // positive ints. Legacy form accepts variadic positional ints. The
  // single-vector-literal case (lowered as `vector(...)` IRCall) is
  // unpacked into per-axis dims. `fixedValues` is consulted by
  // resolveConstant so a binding ref like `n = lengthof(data)` resolves
  // through the pre-eval cache instead of needing a constant-fold
  // entry per surface op.
  const dims: number[] = [];
  const tail = rhsIR.args.slice(1);
  let nodes: IRNode[] = tail;
  if (tail.length === 1
    && tail[0].kind === 'call'
    && (tail[0] as any).op === 'vector'
    && Array.isArray((tail[0] as any).args)) {
    nodes = (tail[0] as any).args;
  }
  for (const node of nodes) {
    const n = resolveConstant(node, bindings, new Set(), fixedValues);
    if (n == null || !Number.isInteger(n) || n <= 0) return null;
    dims.push(n);
  }
  return { kind: 'iid', from: baseName, dims };
}

// Stochastic kernel-broadcast: `broadcast(K, c1, c2, …)` where K is a
// distribution kernel → array-valued independent-product measure
// (spec §04). v1 scope: arg0 is a sampleable-distribution constructor
// used directly as the kernel (`broadcast(Normal, means, sigmas)`);
// the collection args bind to the distribution's parameters,
// positionally or by kwarg. The deterministic value-broadcast
// (`broadcast(f, …)` with f a function) returns null here and is
// handled as an ordinary value binding, not a measure.
//
// `fn(Dist(…))` / `kernelof` / multi-axis collections are documented
// follow-ups (TODO §04). Per-element shape resolution + sampling
// happens in matKernelBroadcast (length K is data-driven, resolved at
// materialise time — unlike iid's static integer dims).
function classifyKernelBroadcast(rhsIR: IRNode, ast: any, bindings: any): DerivationKernelBroadcast | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length < 1) return null;
  const k = rhsIR.args[0];
  // Bare distribution-constructor kernel, not shadowed by a binding.
  if (!k || k.kind !== 'ref' || k.ns !== 'self'
      || !SAMPLEABLE_DISTRIBUTIONS.has(k.name) || bindings.has(k.name)) {
    return null;
  }
  const argIRs = rhsIR.args.slice(1);
  const kwargIRs = rhsIR.kwargs ? Object.assign({}, rhsIR.kwargs) : null;
  if (argIRs.length === 0 && (!kwargIRs || Object.keys(kwargIRs).length === 0)) {
    return null;   // no parameter inputs → not a broadcast
  }
  return { kind: 'kernelbroadcast', distOp: k.name, argIRs: argIRs, kwargIRs: kwargIRs };
}

// broadcast(logdensityof, M, points) — evaluate a measure's density
// at MANY points. Reference (eager, per-point) realisation: the
// abstract lowering says logdensityof(M,_) is a batched closed-form
// expression in the point and broadcast maps it; flatppl-js is the
// EAGER engine (engine-concepts §11), so we map the trusted
// single-point logdensityof over the points — tractable M ⇒ no
// sampling. The principled FlatPIR-codegen path can later replace
// this with tests + this reference under it.
function classifyBroadcastLogdensity(rhsIR: IRNode, ast: any, bindings: any): DerivationBroadcastLogdensity | null {
  if (rhsIR.op !== 'broadcast' || !Array.isArray(rhsIR.args)
      || rhsIR.args.length !== 3) return null;
  const fIR = rhsIR.args[0];
  const mIR = rhsIR.args[1];
  const pIR = rhsIR.args[2];
  // First arg must be the bare `logdensityof` builtin (not a
  // user-shadowed binding).
  if (!fIR || fIR.kind !== 'ref' || fIR.ns !== 'self'
      || fIR.name !== 'logdensityof' || bindings.has('logdensityof')) {
    return null;
  }
  if (!isSelfRef(mIR) || !bindings.has(mIR.name)) return null;
  return { kind: 'broadcast_logdensity', measureName: mIR.name, pointsIR: pIR };
}

// `broadcast` is overloaded: stochastic kernel-broadcast
// (broadcast(Normal, mus, sigmas)) vs. broadcast(logdensityof, M,
// pts). Try the logdensity form first; fall back to kernel-broadcast
// (which only matches a bare SAMPLEABLE head, so the two never
// collide).
function classifyBroadcast(rhsIR: IRNode, ast: any, bindings: any): DerivationBroadcastLogdensity | DerivationKernelBroadcast | null {
  return classifyBroadcastLogdensity(rhsIR, ast, bindings)
      || classifyKernelBroadcast(rhsIR, ast, bindings);
}

// `logdensityof(M, x)` — per spec §sec:posterior, evaluate M's
// log-density at x. Result is REAL (a value, not a measure), but the
// classifier dispatch lives here uniformly: the materialiser computes
// per-prior-atom values via traceeval.walk + tally='clamped', so each
// prior atom θ_i contributes logp = logdensityof(M[θ_i], x). This is
// the same primitive that drives bayesupdate's reweight, just exposed
// as a scalar binding rather than folded into a posterior.
//
// Supported shape:
//   - M is a self-ref to a measure binding (sample / record / iid /
//     algebraic combinator chain — anything expandMeasureIR handles).
//   - x is resolvable to a concrete JS value (literal, array binding,
//     record literal, …) via resolveIRToValue. Variate observations
//     (x is itself a variate) are deferred — they require encoding
//     the observation into refArrays per atom, an extra path the
//     materialiser doesn't yet take.
function classifyLogdensityof(rhsIR: IRNode, ast: any, bindings: any): DerivationLogdensityof | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 2) return null;
  const Mref   = rhsIR.args[0];
  const obsIR  = rhsIR.args[1];
  if (!isSelfRef(Mref)) return null;
  if (!bindings.has(Mref.name)) return null;
  // Hold the obs IR — the materialiser resolves it to a concrete JS
  // value at sample time, consulting fixedValues for any binding
  // refs. Classification cares only that an obs argument exists in
  // a recognisable shape; eager value resolution at classify time
  // forces a pre-eval-vs-classify ordering dance that we no longer
  // need.
  return { kind: 'logdensityof', measureName: Mref.name, obsIR };
}

/**
 * Classify `totalmass(M)` (spec §06) as a derivation that surfaces
 * the measure's tracked totalmass as a per-atom scalar value. The
 * materialiser reads M's `logTotalmass` and broadcasts `exp(...)` to
 * N atoms. Supported when M is a self-ref to a measure binding the
 * orchestrator can materialise (anything expandMeasureIR handles).
 */
function classifyTotalmass(rhsIR: IRNode, ast: any, bindings: any): DerivationTotalmass | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 1) return null;
  const Mref = rhsIR.args[0];
  if (!isSelfRef(Mref)) return null;
  if (!bindings.has(Mref.name)) return null;
  return { kind: 'totalmass', measureName: Mref.name };
}

/**
 * Classify `truncate(M, S)` (spec §06): restricts the support of
 * measure M to set S, with ν(A) = M(A ∩ S). Per spec, truncate does
 * NOT normalize — the resulting measure carries M(S) as its
 * totalmass, which the materialiser surfaces via logTotalmass.
 *
 * Supported shape:
 *   - M is a self-ref to a measure binding (anything resolveMeasureBaseName
 *     accepts; the materialiser walks the parent measure for samples).
 *   - S is a literal set expression parseSetIR can lift to a structural
 *     descriptor: interval(lo, hi) with literal bounds, or one of the
 *     named real / integer / boolean sets. Dynamic sets defer to a
 *     future pass — they'd require per-atom set membership evaluation.
 */
function classifyTruncate(rhsIR: IRNode, ast: any, bindings: any): DerivationTruncate | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 2) return null;
  const baseName = resolveMeasureBaseName(ast.args[0], bindings);
  if (baseName == null) return null;
  const setDescr = parseSetIR(rhsIR.args[1], bindings);
  if (setDescr == null) return null;
  return { kind: 'truncate', from: baseName, setDescr };
}

// pushfwd(f, M) — first-class measure-op classifier. Per spec §06:
// pushforward of measure M through function f. The result is a measure
// whose variate is `f(x)` for x ~ M.
//
// Sampling (matPushfwd): one batched call to evaluateExprN over M's
// per-atom samples. Density (density.walkPushfwd): score M at
// f_inv(y), subtract logvolume(f_inv(y)) — requires the f arg to be
// a `bijection(...)` annotation (otherwise density isn't tractable in
// general and we throw a clear error).
//
// Supported f bindings: fn / functionof / kernelof / bijection. The
// pushfwd's f-position lift signature (see signatureOf) lifts inline
// fn / functionof shapes to anon bindings, so by the time we classify
// here both args are self-refs.
function classifyPushfwd(rhsIR: IRNode, ast: any, bindings: any): DerivationPushfwd | null {
  if (!Array.isArray(rhsIR.args) || rhsIR.args.length !== 2) return null;
  const fIR = rhsIR.args[0];
  const mIR = rhsIR.args[1];
  if (!isSelfRef(fIR) || !isSelfRef(mIR)) return null;
  const fBinding = bindings.get(fIR.name);
  const mBinding = bindings.get(mIR.name);
  if (!fBinding || !mBinding) return null;
  // f must be a function-typed binding so matPushfwd can find a body
  // to evaluate. bijection-annotated functions are themselves
  // functionof-shaped (the underlying f's body); they classify here
  // identically and density-side dispatch reads the bijection metadata
  // separately via opts.resolveBijection.
  if (fBinding.type !== 'fn' && fBinding.type !== 'functionof'
      && fBinding.type !== 'kernelof' && fBinding.type !== 'bijection') {
    return null;
  }
  return { kind: 'pushfwd', from: mIR.name, fnRef: fIR.name };
}

// ---------------------------------------------------------------------
// jointchain / kchain first-class derivation kind (consume/rest
// consolidation — flatppl-dev TODO §06). This is the ONLY path: the
// legacy `inlineChainOps` AST-rewrite and the transitional migration
// flag have been deleted. jointchain/kchain IR always reaches
// `classifyJointchain` → `kind:'jointchain'`, materialised by
// `matJointchain` and scored via `expandMeasureIR`'s jointchain case
// on the proven consume/rest spine.

/**
 * Classify `jointchain(...)` / `kchain(...)` into a first-class
 * `kind:'jointchain'` derivation with an EXPLICIT step structure —
 * no AST rewrite, no surface-kwarg-name matching (the fragility class
 * `inlineChainOps` suffers from).
 *
 *   { kind:'jointchain',
 *     marginalize: bool,                 // kchain ⇒ true (keep last only)
 *     labels: [string]|null,             // kwarg form ⇒ record-shaped
 *     steps: [
 *       { var, role:'base',   ref, kernel:bool },   // step 0 (M, or
 *                                                   //   kernel-first)
 *       { var, role:'kernel', ref, inputs:[var…] }, // step i≥1: K_i on
 *       … ] }                                       //   cat(prior vars)
 *
 * Mirrors the spec stochastic-node equivalence
 * `a~M1; b~K2(a); c~K3([a,b])` (§06). Kernel application is recorded
 * structurally (`ref` + `inputs`), never by inlining K's body.
 *
 * Covers positional (2-arg, N-ary), kwarg, kernel-first, record-prior
 * multi-param (auto-splat), and inline-functionof kernels. Returns
 * null only for genuinely unsupported shapes (a clear "cannot
 * classify" → the binding surfaces an error rather than being
 * silently mis-handled).
 */
function classifyJointchain(rhsIR: any, ast: any, bindings?: any, opts?: any): DerivationJointchain | null {
  if (!rhsIR || rhsIR.kind !== 'call'
      || (rhsIR.op !== 'jointchain' && rhsIR.op !== 'kchain')) return null;
  const marginalize = (rhsIR.op === 'kchain');

  // IR-driven (not AST/ref-only). Each component is uniformly one of:
  //   - a self-ref to a measure or kernel binding, or
  //   - an INLINE callable IR — `functionof` (a kernel; `fn`/`kernelof`
  //     lower to functionof) which the lift leaves in place because it
  //     contains a hole (liftMeasure:989), or an inline measure call
  //     (op ∈ MEASURE_PRODUCING). Reading the IR resolves the
  //     `liftMeasure` hole asymmetry that left the kernel inline while
  //     the measure was hoisted to a ref (the `kchain(Exponential(1),
  //     fn(Normal(0,_)))` case).
  // Kwarg form lowers to `fields:[{name,value}]` (FIELD_FORMS);
  // positional to `args:[…]`.
  let comps, labels;
  if (Array.isArray(rhsIR.fields) && rhsIR.fields.length >= 2) {
    labels = rhsIR.fields.map((f: any) => f.name);
    comps = rhsIR.fields.map((f: any) => f.value);
  } else if (Array.isArray(rhsIR.args) && rhsIR.args.length >= 2) {
    labels = null;
    comps = rhsIR.args;
  } else {
    return null;
  }

  // Classify one component IR into a node descriptor:
  //   { ref, isKernel }            self-ref to a binding
  //   { kernelIR }                 inline functionof
  //   { measureIR }                inline measure call
  // or null if it's none of these.
  const describe = (ir: any) => {
    if (!ir) return null;
    if (ir.kind === 'ref' && ir.ns === 'self' && bindings.has(ir.name)) {
      const b = bindings.get(ir.name);
      const isKernel = !!b && (b.type === 'functionof'
        || b.type === 'kernelof' || b.type === 'fn');
      const isMeasure = !isKernel && (
        (b && b.ir && b.ir.kind === 'call' && MEASURE_PRODUCING.has(b.ir.op))
        || isMeasureExpr(b && b.node && b.node.value, bindings));
      if (!isKernel && !isMeasure) return null;
      return { ref: ir.name, isKernel };
    }
    if (ir.kind === 'call' && ir.op === 'functionof') return { kernelIR: ir };
    if (ir.kind === 'call' && MEASURE_PRODUCING.has(ir.op)) {
      return { measureIR: ir };
    }
    return null;
  };

  const steps: any[] = [];
  for (let i = 0; i < comps.length; i++) {
    const d = describe(comps[i]);
    if (!d) return null;                            // shape not covered
    const v = labels ? labels[i] : ('s' + i);
    const isKernelComp = !!(d.kernelIR || d.isKernel);
    if (i === 0) {
      // Base: a measure, or (kernel-first) a kernel.
      const step: any = { var: v, role: 'base', kernel: isKernelComp };
      if (d.ref != null) step.ref = d.ref;
      else if (d.kernelIR) step.kernelIR = d.kernelIR;
      else step.measureIR = d.measureIR;
      steps.push(step);
    } else {
      if (!isKernelComp) return null;               // K_i must be a kernel
      const step: any = { var: v, role: 'kernel', inputs: steps.map((s) => s.var) };
      if (d.ref != null) step.ref = d.ref;
      else step.kernelIR = d.kernelIR;
      steps.push(step);
    }
  }
  return { kind: 'jointchain', marginalize, labels, steps };
}

const MEASURE_OP_CLASSIFIERS = {
  weighted:     classifyWeighted,
  logweighted:  classifyLogWeighted,
  normalize:    classifyNormalize,
  superpose:    classifySuperpose,
  ifelse:       classifyIfelse,
  get:          classifyStochasticIndex,
  record:       classifyRecordOrJoint,
  joint:        classifyRecordOrJoint,
  iid:          classifyIid,
  broadcast:    classifyBroadcast,
  logdensityof: classifyLogdensityof,
  totalmass:    classifyTotalmass,
  truncate:     classifyTruncate,
  pushfwd:      classifyPushfwd,
  jointchain:   classifyJointchain,
  kchain:       classifyJointchain,
};

/**
 * Whether a derivation's outgoing references are satisfiable —
 * i.e. every 'self' ref it contains points at a binding that itself
 * has a derivation. Aliases / weighted / normalize just check the
 * target.
 */
function derivationRefsValid(d: DerivationBase, derivations: any, bindings: Map<string, BindingInfo>, fixedValues: any) {
  // A name is "resolvable downstream" if there's a derivation for it
  // (the materialiser knows how to compute samples) OR it has a
  // fixed-phase value the worker resolves through its session env.
  // The viewer's collectRefArrays already drops fixed-phase refs from
  // refArrays, so a binding whose only deps are fixed values can
  // still sample correctly via session env. Without this, a Normal(
  // mu=get_field(ref(rp), "theta1"), …) classified as 'sample' would
  // cascade-prune the moment the orchestrator dropped rp's
  // derivation (it's a record, not numeric — pre-eval drops those).
  function resolvable(name: any) {
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
  // Select (ifelse / mixture): every NAMED branch must be resolvable;
  // inline-IR branches ({ ir }) are self-contained.
  if (d.kind === 'select') {
    if (!Array.isArray(d.branches) || d.branches.length === 0) return false;
    for (const b of d.branches) {
      if (b && b.ref != null && !resolvable(b.ref)) return false;
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
  // kernelbroadcast: every self-ref in the parameter inputs must be
  // resolvable (the distribution kernel itself is a builtin).
  if (d.kind === 'kernelbroadcast') {
    const irs = (d.argIRs || []).concat(
      d.kwargIRs ? Object.keys(d.kwargIRs).map((k) => d.kwargIRs[k]) : []);
    for (const ir of irs) {
      for (const r of collectSelfRefs(ir)) {
        if (!resolvable(r)) return false;
      }
    }
    return true;
  }
  // broadcast(logdensityof, M, pts): the measure must be resolvable
  // (the points expression is a fixed-phase value resolved at
  // materialise time).
  if (d.kind === 'broadcast_logdensity') {
    return resolvable(d.measureName);
  }
  // pushfwd: the base measure must be resolvable. f is a function
  // binding referenced by name; we trust the binding map.
  if (d.kind === 'pushfwd') {
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
  if (d.kind === 'totalmass') {
    return resolvable(d.measureName);
  }
  if (d.kind === 'truncate') {
    return resolvable(d.from);
  }
  const ir = d.kind === 'sample' ? d.distIR : d.ir;
  const refs = collectSelfRefs(ir);
  for (const r of refs) {
    if (!resolvable(r)) return false;
  }
  return true;
}

function isDiscreteAt(name: any, derivations: any, visited?: any) {
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
  if (d.kind === 'select') {
    // Same rule as superpose: a mixture/ifelse is discrete only if
    // every branch is (mixed support ⇒ treat as continuous). Named
    // branch ⇒ recurse; inline-IR leaf branch ⇒ check the dist op.
    if (!Array.isArray(d.branches) || d.branches.length === 0) return false;
    for (const b of d.branches) {
      if (!b) return false;
      if (b.ref != null) {
        if (!isDiscreteAt(b.ref, derivations, new Set(visited))) return false;
      } else if (b.ir) {
        if (!DISCRETE_DISTRIBUTIONS.has(b.ir.op)) return false;
      } else return false;
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
function leafSampleIR(name: any, derivations: any, visited: any) {
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
/**
 * Resolve a bijection binding's metadata into call-ready form for
 * density.walkPushfwd. Reads the f_inv and logvolume bindings to
 * extract their body+paramName (or, for a scalar logvolume, the
 * literal value). Returns null when the metadata can't be resolved
 * (missing binding, non-functionof IR) — callers treat null as "not
 * available, density not tractable for this binding".
 */
function resolveBijectionMeta(bij: any, bindings: any) {
  // Read body + paramName from a functionof binding. arity=1 returns
  // a fn-with-param; arity=0 returns a fn-with-null-paramName (a
  // closed-form constant — `fn(log(2.0))` is the canonical example).
  function fnBodyOf(bindingName: any, allowConst: any) {
    const b = bindings.get(bindingName);
    if (!b || !b.ir || b.ir.kind !== 'call' || b.ir.op !== 'functionof') return null;
    const params = b.ir.params || [];
    if (!b.ir.body) return null;
    if (params.length === 1) return { body: b.ir.body, paramName: params[0] };
    if (params.length === 0 && allowConst) return { body: b.ir.body, paramName: null };
    return null;
  }
  // f_inv must be a true 1-arg function — y is its input.
  const fInv = fnBodyOf(bij.fInvName, false);
  if (!fInv) return null;
  let logVolume;
  if (bij.logVolume.kind === 'scalar') {
    logVolume = { kind: 'scalar', value: bij.logVolume.value };
  } else {
    // logvolume may be a function of x OR a constant (per spec §06).
    const lvBody = fnBodyOf(bij.logVolume.name, true);
    if (!lvBody) return null;
    logVolume = { kind: 'fn', body: lvBody.body, paramName: lvBody.paramName };
  }
  return { fInv, logVolume };
}

// Closed-form total mass (in log) of an already-expanded measure IR,
// or null when it isn't closed-form here (data-dependent weights,
// truncate, pushfwd, …). Mirrors the measure-algebra mass rules and
// is used to lower `normalize(M)` to `logweighted(−log Z, M)` so the
// normalized-mixture density needs no opts/worker plumbing and reuses
// walkLogWeighted (engine-concepts §11; "totalmass is a first-class
// node concern"). All stdlib leaf distributions are normalized (unit
// mass); weighted/superpose/iid compose multiplicatively/additively.
function closedFormLogTotalmass(ir: any, bindings: any): any {
  if (!ir || ir.kind !== 'call') return null;
  const op = ir.op;
  if (op === 'MvNormal' || SAMPLEABLE_DISTRIBUTIONS.has(op)) return 0;
  if (op === 'normalize') return 0;
  if (op === 'logweighted') {
    const g = resolveConstant(ir.args[0], bindings || new Map(), new Set());
    if (g == null || !Number.isFinite(g)) return null;
    const b: any = closedFormLogTotalmass(ir.args[1], bindings);
    return b == null ? null : g + b;
  }
  if (op === 'weighted') {
    const w = resolveConstant(ir.args[0], bindings || new Map(), new Set());
    if (w == null || !(w > 0) || !Number.isFinite(w)) return null;
    const b: any = closedFormLogTotalmass(ir.args[1], bindings);
    return b == null ? null : Math.log(w) + b;
  }
  if (op === 'select') {
    const br = ir.branches || [];
    if (br.length === 0) return null;
    const terms: any[] = [];
    for (let k = 0; k < br.length; k++) {
      const b = closedFormLogTotalmass(br[k], bindings);
      if (b == null) return null;
      let lw = 0;
      if (ir.logweights) {
        lw = resolveConstant(ir.logweights[k], bindings || new Map(), new Set());
        if (lw == null || !Number.isFinite(lw)) return null;
      }
      terms.push(lw + b);
    }
    let m = -Infinity;
    for (const t of terms) if (t > m) m = t;
    if (!Number.isFinite(m)) return m;
    let s = 0;
    for (const t of terms) s += Math.exp(t - m);
    return m + Math.log(s);
  }
  if (op === 'joint' || op === 'record') {
    const comps = Array.isArray(ir.fields) ? ir.fields.map((f: any) => f.value)
      : (Array.isArray(ir.args) ? ir.args : null);
    if (!comps) return null;
    let acc = 0;
    for (const c of comps) {
      const t = closedFormLogTotalmass(c, bindings);
      if (t == null) return null;
      acc += t;
    }
    return acc;
  }
  if (op === 'iid' && Array.isArray(ir.args) && ir.args.length === 2) {
    const inner: any = closedFormLogTotalmass(ir.args[0], bindings);
    if (inner == null) return null;
    const n = resolveConstant(ir.args[1], bindings || new Map(), new Set());
    if (n == null || !Number.isFinite(n)) return null;
    return n * inner;
  }
  // truncate / pushfwd / jointchain / unknown — not closed-form here.
  return null;
}

function expandMeasureIR(name: string, derivations: any, visited?: any, bindings?: any): IRNode | null {
  visited = visited || new Set();
  if (visited.has(name)) return null;
  const next = new Set(visited); next.add(name);
  const d = derivations && derivations[name];
  if (d) {
    switch (d.kind) {
      case 'alias':
        return expandMeasureIR(d.from, derivations, next, bindings);
      case 'sample':
        // Leaf distribution call — return the distIR verbatim. Refs
        // in its kwargs are value refs (per-i params).
        return d.distIR;
      case 'mvnormal':
        // Multivariate sampleable distribution. Same
        // treatment as 'sample': return the IR verbatim; the density
        // walker has a dedicated handler keyed on the op name
        // (walkMvNormal in density.js OP_HANDLERS).
        return d.distIR;
      case 'iid': {
        const inner: any = expandMeasureIR(d.from, derivations, next, bindings);
        if (!inner) return null;
        // dims is a multi-dim shape; flatten to a single iid count.
        // The walker's iid case handles the n-shape uniformly via
        // observed length — multi-dim observations would need to be
        // flattened to match (1D arrays). For typical bayesupdate the
        // dims are 1D and obs is a flat array, which already matches.
        const total = d.dims.reduce((a: any, b: any) => a * b, 1);
        return {
          kind: 'call', op: 'iid',
          args: [inner, { kind: 'lit', value: total }],
        };
      }
      case 'record': {
        const fields: any[] = [];
        for (const k in d.fields) {
          const inner = expandMeasureIR(d.fields[k], derivations, next, bindings);
          if (!inner) return null;
          // Attach the source binding name alongside the expanded
          // value. Density-side env-threading uses this as a second
          // env key so refs to the source binding (e.g. an anon
          // produced by kernel substitution) resolve to the OBSERVED
          // field value, not the per-atom prior sample. Without this,
          // jointchain rewrites whose substituted bodies ref the
          // source-binding anons (as opposed to the surface field
          // names) get the wrong density. See
          // flatppl-dev/flatppl-engine-concepts.md §5 (env-threading).
          fields.push({ name: k, value: inner, source: d.fields[k] });
        }
        // Use 'joint' op (the measure form). 'record' and 'joint'
        // share the IR shape and the walker treats them equivalently.
        return { kind: 'call', op: 'joint', fields };
      }
      case 'tuple': {
        // Positional joint(M1, M2, ...) — args = [inner_M1, inner_M2, ...].
        // Walker dispatches the positional-args branch of walkJoint.
        const argsIR: any[] = [];
        for (const n of d.elems) {
          const inner = expandMeasureIR(n, derivations, next, bindings);
          if (!inner) return null;
          argsIR.push(inner);
        }
        return { kind: 'call', op: 'joint', args: argsIR };
      }
      case 'superpose': {
        // Additive superposition ν = Σ_k M_k (spec §06). Canonicalise
        // to the discrete-selector `select` IR (engine-concepts §11):
        // density = logsumexp_k logp_{M_k} = log Σ p_k — the *raw*
        // (un-normalised) sum, so `logweights:null` (each component
        // self-carries any weighted()/normalize() factor via its own
        // expanded sub-IR). All components share one variate space
        // (spec), so every branch consumes the same observation; the
        // walker asserts identical consumption. This is the discrete
        // sibling of the kchain MC marginal — but EXACT (no −logN).
        const branches: any[] = [];
        for (const n of d.fromNames) {
          const inner = expandMeasureIR(n, derivations, next, bindings);
          if (!inner) return null;
          branches.push(inner);
        }
        if (branches.length === 0) return null;
        return { kind: 'call', op: 'select', branches, logweights: null };
      }
      case 'select': {
        // Weighted discrete-selector mixture (ifelse today; explicit
        // mixture / xs[i] later). Same canonical `select` IR as the
        // superpose case, but with explicit per-branch log-weight
        // value-IRs (e.g. ifelse ⇒ [log p, log(1−p)] from the
        // Bernoulli condition). Marginalising selector ⇒ walkSelect
        // logsumexp_k(logw_k + logp_branch_k) — the exact mixture
        // density.
        const branches: any[] = [];
        for (const b of d.branches) {
          let inner = null;
          if (b && b.ref != null) {
            inner = expandMeasureIR(b.ref, derivations, next, bindings);
          } else if (b && b.ir) {
            inner = _expandMeasureIRStructural(b.ir, derivations, next, bindings);
          }
          if (!inner) return null;
          branches.push(inner);
        }
        if (branches.length === 0) return null;
        return {
          kind: 'call', op: 'select', branches,
          logweights: d.logweightIRs || null,
          // Unresolved runtime-weight spec (non-closed-form selector,
          // engine-concepts §11). expandMeasureIR is pure — it cannot
          // reduce the selector ensemble — so it carries the spec
          // through for the materialiser's runtime-weight resolver to
          // turn into literal logweights before density evaluation.
          weightsFrom: d.runtimeWeights || null,
          // Retain-mode hint (engine-concepts §11): the selector
          // binding name, plus its base (Categorical 1-based,
          // Categorical0 0-based; absent ⇒ Bernoulli/ifelse and the
          // walker uses sel?0:1). walkSelect checks the env overlay
          // at evaluation time: if the selector value is in scope
          // (because an enclosing joint already consumed it from
          // the observation and threaded it through), walkSelect
          // routes to the matching branch and scores `logw_k +
          // logp_branch_k` — the joint/retain density `log P(i=k) +
          // log p_{x|i}(x_obs | i=k)`. Absent in env ⇒ marginalising
          // mixture density (the current logsumexp behaviour); same
          // node, branch chosen by env-threading.
          selectorName: d.selectorRef || null,
          selectorBase: (d.selectorBase != null) ? d.selectorBase : null,
        };
      }
      case 'jointchain': {
        // First-class jointchain/kchain (consume/rest consolidation,
        // steps 2c + 2b-ext). Canonicalise the EXPLICIT step structure
        // into the self-contained measure IR the proven env-threaded
        // walkJoint already scores — the structural, robust analogue
        // of what inlineChainOps did fragilely by surface-kwarg-name
        // matching. Left-associative per spec §06; the i-th kernel
        // takes the cat of all prior step variates as its single arg
        // (`b~K2(a)` for one prior, `c~K3([a,b])` for ≥2), realised by
        // rewiring the kernel's param ref to ref(prior_0) / vector(ref
        // prior_0, …) over the prior step-var names.
        const steps = d.steps || [];
        if (steps.length < 2) return null;
        const base = steps[0];
        if (base.kernel || base.ref == null) return null;
        const baseIR: any = expandMeasureIR(base.ref, derivations, next, bindings);
        if (!baseIR) return null;
        const vname = (i: any) => (d.labels && d.labels[i]) || ('s' + i);

        // Resolve a kernel step to { params, body } and EXPAND the
        // body. The "closure walk" the legacy inlineChainOps did by
        // hand IS just expandMeasureIR: when a `functionof` body is a
        // ref to a measure binding, expandMeasureIR follows it into
        // the self-contained measure IR where the kernel's boundary
        // params surface as leaf refs (e.g. `functionof(obs_dist,
        // theta1=theta1, theta2=theta2)` ⇒ body expands to
        // `joint(y = Normal(mu = ref theta1, sigma = ref theta2))`).
        // So kernel application = expand the body, then bind each
        // param to the prior variate: a NAMED param that matches a
        // prior field resolves for free by walkJoint's overlay
        // env-threading (its leaf ref already carries that name); a
        // lone HOLE/placeholder param (the `fn(…_…)` case, single
        // param, no matching prior field) is rewired to the prior cat.
        const kernelExpand = (kstep: any) => {
          let f = kstep.kernelIR;
          if (!f && kstep.ref != null) {
            const kb = bindings && bindings.get(kstep.ref);
            if (kb && kb.ir && kb.ir.kind === 'call'
                && kb.ir.op === 'functionof') f = kb.ir;
          }
          if (!f || !f.body || !Array.isArray(f.params)
              || f.params.length === 0) return null;
          let body = f.body;
          if (body.kind === 'ref' && body.ns === 'self') {
            body = expandMeasureIR(body.name, derivations, next, bindings);
            if (!body) return null;
          }
          return { params: f.params, body };
        };
        // Spread a (record/joint) measure IR into its named field
        // descriptors (preserving `source` for env-threading); a
        // scalar measure IR contributes one field under `fallback`.
        const spreadFields = (ir: any, fallback: any, src: any) => {
          if (ir && ir.kind === 'call'
              && (ir.op === 'joint' || ir.op === 'record')
              && Array.isArray(ir.fields)) {
            return ir.fields.map((fl: any) => ({
              name: fl.name, value: fl.value,
              source: fl.source != null ? fl.source : fl.name,
            }));
          }
          return [{ name: fallback, value: ir, source: src }];
        };
        // Rewire a lone hole/placeholder param to the prior cat: one
        // prior ⇒ ref(prior_0); ≥2 ⇒ vector(ref prior_0, …).
        const rewireHole = (body: any, param: any, priorNames: any) => {
          const sub: any = (node: any) => {
            if (node == null || typeof node !== 'object') return node;
            if (Array.isArray(node)) return node.map(sub);
            if (node.kind === 'ref'
                && (node.ns === '%local' || node.ns === 'self')
                && node.name === param) {
              if (priorNames.length === 1) {
                return { kind: 'ref', ns: 'self', name: priorNames[0],
                  loc: node.loc };
              }
              return { kind: 'call', op: 'vector',
                args: priorNames.map((nm: any) => ({ kind: 'ref', ns: 'self',
                  name: nm, loc: node.loc })) };
            }
            const out: Record<string, any> = {};
            for (const k in node) out[k] = sub(node[k]);
            return out;
          };
          return sub(body);
        };
        // Bind a kernel's expanded body to the available prior field
        // names. Named params already matching a prior field thread
        // for free; a single unmatched param is a hole bound to the
        // prior cat; an unmatched param in a multi-param kernel is
        // unsupported (clean null).
        const bindKernel = (ke: any, priorNames: any) => {
          let body = ke.body;
          for (const p of ke.params) {
            if (priorNames.indexOf(p) === -1) {
              if (ke.params.length !== 1) return null;
              body = rewireHole(body, p, priorNames);
            }
          }
          return body;
        };

        // Flatten all step variates left-associatively into one joint.
        const outFields = spreadFields(baseIR, vname(0), base.ref);
        const priorNames = outFields.map((f: any) => f.name);

        if (d.marginalize) {
          // kchain: marginal of the LAST step's variate(s); the prior
          // is integrated out by matLogdensityof's isChain MC
          // (logsumexp−logN over the prior atoms).
          const ke = kernelExpand(steps[steps.length - 1]);
          if (!ke) return null;
          if (steps.length === 2) {
            // Proven 2-step path. A HOLE param binds to the BASE
            // BINDING ref — a single materialisable scalar prior that
            // matLogdensityof resolves per-atom (the synthetic spread
            // name `s0` is NOT a binding, so it must not be the
            // rewire target here). NAMED params (record-kchain)
            // already ref the base record's draw bindings — leave
            // them for matLogdensityof + isChain.
            let body = ke.body;
            for (const p of ke.params) {
              if (priorNames.indexOf(p) === -1) {
                if (ke.params.length !== 1) return null;
                body = rewireHole(body, p, [base.ref]);
              }
            }
            return body;
          }
          // N-ary (>2) chain-associativity recursion (engine-concepts
          // §6): kchain(M,K₁,…,K_{n-1}) ≡ a 2-step kchain whose prior
          // is the RETAINED (n−1)-joint history jointchain(M,K₁,…,
          // K_{n-2}). We return ONLY the final kernel body with its
          // hole rewired to the cat over that history's variate names
          // (`rewireHole`'s ≥2 vector branch — the SAME helper the
          // retain branch below uses, so marginalize/retain share one
          // hole-rewire path, no parallel logic). matLogdensityof
          // materialises that joint history ONCE and binds its variate
          // columns as the per-atom refArrays the cat consumes, then
          // logsumexp−logN — the SAME estimator as the 2-step case,
          // lifted to a joint prior. The prior-variate names are the
          // deterministic positional `vname(i)` (= `s{i}`, labels
          // null); matLogdensityof rebuilds the same names by position
          // from the materialised history (shared `vname` convention).
          // Labelled/record-prior N-ary stays a clean deferral.
          if (d.labels) return null;
          for (let i = 1; i < steps.length - 1; i++) {
            const ik = kernelExpand(steps[i]);
            if (!ik) return null;
            const ib = bindKernel(ik, priorNames.slice());
            if (!ib) return null;
            const ifs = spreadFields(ib, vname(i), null);
            for (const fl of ifs) {
              outFields.push(fl);
              priorNames.push(fl.name);
            }
          }
          const kb = bindKernel(ke, priorNames.slice());
          if (!kb) return null;
          return kb;
        }

        // jointchain: ∏ conditional densities. Record/labelled (or a
        // record-shaped base) ⇒ walkJoint `fields` + `source`/name
        // overlay env-threading. Positional scalar base
        // (tuple-observed) ⇒ walkJoint positional `args`, which
        // threads each consumed scalar under `s{i}` so the kernel
        // body's rewired `ref(s0)` / `vector(ref s0, ref s1, …)`
        // resolves to the observed prior. N-ary positional is just
        // the loop generalisation of the 2-step funnel form: each
        // kernel K_i takes the cat of prior variates [s0..s_{i-1}]
        // (spec §06 `c~K3([a,b])`), realised by `bindKernel`'s
        // existing rewireHole — single-prior collapses to `ref(s0)`,
        // ≥2 priors to a vector(ref s_k, …). The density walker's
        // positional loop (density.js walkJointFieldsOrPositional)
        // already iterates `ir.args.length` without bound, so this
        // commit just lifts the structural cap at the expand step.
        const baseIsRecord = baseIR.kind === 'call'
          && (baseIR.op === 'joint' || baseIR.op === 'record')
          && Array.isArray(baseIR.fields);
        if (!d.labels && !baseIsRecord) {
          const positionalArgs: any = [baseIR];
          const positionalPriorNames = ['s0'];
          for (let i = 1; i < steps.length; i++) {
            const ke = kernelExpand(steps[i]);
            if (!ke) return null;
            const kb = bindKernel(ke, positionalPriorNames.slice());
            if (!kb) return null;
            positionalArgs.push(kb);
            positionalPriorNames.push('s' + i);
          }
          return { kind: 'call', op: 'joint', args: positionalArgs };
        }
        for (let i = 1; i < steps.length; i++) {
          const ke = kernelExpand(steps[i]);
          if (!ke) return null;
          const kb = bindKernel(ke, priorNames.slice());
          if (!kb) return null;
          const kfs = spreadFields(kb, vname(i), null);
          for (const fl of kfs) {
            outFields.push(fl);
            priorNames.push(fl.name);
          }
        }
        return { kind: 'call', op: 'joint', fields: outFields };
      }
      case 'weighted': {
        const inner: any = expandMeasureIR(d.from, derivations, next, bindings);
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
      case 'normalize': {
        // normalize(M) = M / totalmass(M). Lower to
        // logweighted(−log Z, expand(M)) when Z is closed-form (the
        // canonical normalized mixture
        // normalize(superpose(weighted(w_k, M_k))) has Z = Σ w_k, so
        // a probability mixture has Z=1 ⇒ a 0-shift no-op; an
        // unnormalized base shifts every atom by −log Z). Reuses
        // walkLogWeighted — no opts/worker plumbing, exact density.
        const inner: any = expandMeasureIR(d.from, derivations, next, bindings);
        if (!inner) return null;
        const logZ = closedFormLogTotalmass(inner, bindings);
        if (logZ != null && Number.isFinite(logZ)) {
          return {
            kind: 'call', op: 'logweighted',
            args: [{ kind: 'lit', value: -logZ }, inner],
          };
        }
        // Z not closed-form here (truncate base, data-dependent
        // weights, …): emit the normalize IR; walkNormalize falls
        // back to opts.measureLogTotalmass (default 0) — a documented
        // limitation, not silently wrong for the common closed cases.
        return { kind: 'call', op: 'normalize', args: [inner] };
      }
      case 'pushfwd': {
        // pushfwd(f, M): the f-arg surfaces as a self-ref to the
        // function binding (so call-site recognition by name stays
        // possible). When f is a `bijection(...)` annotation, attach
        // the resolved metadata (f_inv body + paramName, logvolume
        // body+paramName OR scalar) as a side-property on the call IR
        // so density.walkPushfwd can compute the pushforward density
        // without round-tripping through a resolver callback.
        const inner: any = expandMeasureIR(d.from, derivations, next, bindings);
        if (!inner) return null;
        const out: any = {
          kind: 'call', op: 'pushfwd',
          args: [{ kind: 'ref', ns: 'self', name: d.fnRef }, inner],
        };
        if (bindings) {
          const fBinding = bindings.get(d.fnRef);
          const bijMeta = fBinding && fBinding.bijection
            ? resolveBijectionMeta(fBinding.bijection, bindings) : null;
          if (bijMeta) out.bijection = bijMeta;
        }
        return out;
      }
      // evaluate / array / normalize / superpose / iid-of-iid / etc.
      // are not measures-with-densities we can score today.
    }
  }
  // Structural fallback: buildDerivations prunes any derivation whose
  // distIR depends on a parameterized binding (so top-level plot of
  // an unbound parameter says "Not plottable" cleanly). The kernel-
  // sample path substitutes those parameters via env at materialise
  // time, so it still needs the structural shape. When the caller
  // passes `bindings`, walk binding.ir directly using the same
  // measure-shape vocabulary as the derivation-based path above.
  if (bindings) {
    const b = bindings.get(name);
    if (!b || !b.ir) return null;
    return _expandMeasureIRStructural(b.ir, derivations, next, bindings);
  }
  return null;
}

/**
 * Synthesize a kernel signature for a stochastic binding whose
 * derivation was pruned because it depends on parameterized
 * (elementof) ancestors. Conceptually: treat the user clicking on
 * `x` (a stochastic node with open inputs) as if they'd written
 * `kernelof(x)` with no boundary kwargs — per spec §04, that
 * reifies x as a kernel whose inputs are x's elementof leaves.
 *
 * Used by the viewer to surface an Inputs dropdown directly on a
 * stochastic binding rather than the current "Not plottable"
 * dead-end. The binding's type stays whatever it was (draw, …),
 * so colorForBinding keeps painting the node in its original
 * binding-type color.
 *
 * Returns a signature in the same shape as signatureOf would produce
 * for an explicit `kernelof` binding, or `null` when there are no
 * parameterized ancestors (the regular plot path handles those).
 *
 * The body IR is recovered via expandMeasureIR with the bindings
 * fallback so it works even when the orchestrator pruned the chain.
 * Inputs are derived by walking the body's self-refs and keeping
 * the ones whose target binding has type='input' (elementof /
 * external).
 */
function implicitKernelSignature(name: any, bindings: any, derivations: any) {
  if (!bindings) return null;
  // Fire for anything that produces samples or measures: either
  // stochastic phase (a draw / iid draw whose value varies) or a
  // measure-typed binding with open parametric inputs (e.g.
  // `m = iid(Normal(mu, 1), 3)` — phase=parameterized but the
  // binding itself IS a measure). Deterministic value bindings
  // (mu2 = mu^2) take the symmetric implicitFunctionSignature
  // path; without this gate, this helper would build a "kernel"
  // whose body is pow(mu, 2) and the kernel-sample sampler then
  // errors because pow isn't a distribution.
  const subject = bindings.get(name);
  if (!subject) return null;
  const isMeasureLike = subject.phase === 'stochastic'
    || (subject.inferredType && subject.inferredType.kind === 'measure');
  if (!isMeasureLike) return null;
  const body = expandMeasureIR(name, derivations, undefined, bindings);
  if (!body) return null;

  // BFS through the body's self-refs to find PARAMETRIC-phase leaves.
  // Per spec §04 sec:functionof: only elementof leaves (parameterized
  // phase) become kernel inputs. external(...) / load_data(...) are
  // closed over despite sharing binding.type='input' with elementof.
  // We walk transitively because the body may refer to evaluable
  // intermediates (e.g. `resolution = 2.5 + 0.3 * mu`) that hide the
  // actual parametric leaf — same logic as signatureOf's auto-promote.
  const seen = new Set();
  const queue = Array.from(collectSelfRefs(body));
  const elementofRefs: any[] = [];
  while (queue.length > 0) {
    const refName = queue.shift();
    if (seen.has(refName)) continue;
    seen.add(refName);
    const target = bindings.get(refName);
    if (!target) continue;
    if (target.type === 'input' && target.phase === 'parameterized') {
      elementofRefs.push(refName);
      continue;
    }
    // Non-leaf: descend into its IR. Fixed-phase input bindings
    // (external / load_data) have no .ir to walk, so they drop out
    // here silently — exactly the spec's "closed over" semantics.
    if (target.ir) {
      for (const inner of collectSelfRefs(target.ir)) queue.push(inner);
    }
  }
  const inputs: any[] = [];
  for (const refName of elementofRefs) {
    const target = bindings.get(refName);
    inputs.push({
      paramName: refName,
      kwargName: refName,
      type: (target && target.inferredType) || null,
      source: { kind: 'binding', name: refName },
    });
  }
  if (inputs.length === 0) return null;

  return {
    kind: 'kernel',
    inputs,
    output: { type: null },
    body,
    // Tag for callers that want to render slightly differently
    // (current viewer doesn't — same kernel-sample render path).
    implicit: true,
  };
}

/**
 * Synthesize a function signature for a deterministic (parametric-
 * phase) binding whose derivation was pruned by buildDerivations.
 * The symmetric counterpart to implicitKernelSignature: that helper
 * reifies a stochastic binding as `kernelof(x)` with no boundary
 * kwargs (parametric leaves as inputs); this one reifies a value
 * binding as `functionof(v)` with the same auto-boundary semantics.
 *
 * Conceptually: clicking on `mu2 = mu^2` (with mu = elementof(reals))
 * is equivalent to plotting `functionof(mu2)` — a function whose
 * single input is the elementof leaf and whose body computes mu^2.
 * The viewer's profile-plot pipeline then evaluates the body at a
 * range of mu values.
 *
 * Returns null when the binding isn't a parametric-phase value
 * binding, has no .ir, or has no parametric ancestors (the regular
 * fixed-value or function-binding path handles those).
 */
function implicitFunctionSignature(name: any, bindings: any, derivations: any) {
  if (!bindings) return null;
  const subject = bindings.get(name);
  if (!subject || subject.phase !== 'parameterized' || !subject.ir) return null;
  // Measure-typed parameterized bindings go through implicitKernel
  // (they sample, not evaluate). This branch is only for value-
  // typed (scalar / array / record) bindings.
  if (subject.inferredType && subject.inferredType.kind === 'measure') return null;
  // No need to filter callables / elementof here:
  //   - Callables (functionof / kernelof / fn / likelihood) have
  //     phase='fixed', already excluded by the early phase check.
  //   - An elementof leaf (subject is `mu = elementof(reals)`) has
  //     phase='parameterized' but its body has no parametric self-
  //     refs to surface, so the BFS below produces inputs.length===0
  //     and we return null naturally.

  // Body is the binding's lowered IR. Unlike the kernel path, we
  // don't call expandMeasureIR — value bindings aren't measure
  // expressions, and the profile evaluator walks the IR via
  // evaluateExpr (after inlineForProfile rewrites `ref self <input>`
  // → `ref %local <input>`).
  const body = subject.ir;

  // BFS for parametric leaves, same shape as implicitKernelSignature.
  const seen = new Set();
  const queue = Array.from(collectSelfRefs(body));
  const elementofRefs: any[] = [];
  while (queue.length > 0) {
    const refName = queue.shift();
    if (seen.has(refName)) continue;
    seen.add(refName);
    const target = bindings.get(refName);
    if (!target) continue;
    if (target.type === 'input' && target.phase === 'parameterized') {
      elementofRefs.push(refName);
      continue;
    }
    if (target.ir) {
      for (const inner of collectSelfRefs(target.ir)) queue.push(inner);
    }
  }
  const inputs: any[] = [];
  for (const refName of elementofRefs) {
    const target = bindings.get(refName);
    inputs.push({
      paramName: refName,
      kwargName: refName,
      type: (target && target.inferredType) || null,
      source: { kind: 'binding', name: refName },
    });
  }
  if (inputs.length === 0) return null;

  return {
    kind: 'function',
    inputs,
    output: { type: subject.inferredType || null },
    body,
    implicit: true,
  };
}

function _expandMeasureIRStructural(ir: any, derivations: any, visited: any, bindings: any): any {
  if (!ir) return null;
  if (ir.kind === 'ref' && ir.ns === 'self') {
    return expandMeasureIR(ir.name, derivations, visited, bindings);
  }
  if (ir.kind !== 'call') return ir;
  // draw / lawof are pass-throughs at the structural level — the
  // measure they wrap is the measure we want.
  if ((ir.op === 'draw' || ir.op === 'lawof')
      && Array.isArray(ir.args) && ir.args.length === 1) {
    return _expandMeasureIRStructural(ir.args[0], derivations, visited, bindings);
  }
  // iid(measure, n, ...) — recurse on the measure operand.
  if (ir.op === 'iid' && Array.isArray(ir.args) && ir.args.length >= 2) {
    const inner: any = _expandMeasureIRStructural(ir.args[0], derivations, visited, bindings);
    if (!inner) return null;
    return { kind: 'call', op: 'iid', args: [inner].concat(ir.args.slice(1)), loc: ir.loc };
  }
  // record / joint / jointchain — recurse on each field's measure.
  if ((ir.op === 'record' || ir.op === 'joint' || ir.op === 'jointchain')
      && Array.isArray(ir.fields)) {
    const fields = ir.fields.map((f: any) => ({
      ...f, value: _expandMeasureIRStructural(f.value, derivations, visited, bindings),
    }));
    return { kind: 'call', op: ir.op, fields, loc: ir.loc };
  }
  // weighted / logweighted — recurse on the measure-position arg.
  if ((ir.op === 'weighted' || ir.op === 'logweighted')
      && Array.isArray(ir.args) && ir.args.length === 2) {
    const inner: any = _expandMeasureIRStructural(ir.args[1], derivations, visited, bindings);
    if (!inner) return null;
    return { kind: 'call', op: ir.op, args: [ir.args[0], inner], loc: ir.loc };
  }
  // Sampleable distribution call — return as-is. Refs inside the
  // kwargs are value-position refs to per-atom parameters; caller
  // resolves them via substituteLocals; refArrays handles captured
  // self-refs per-atom at sampleN time.
  if (SAMPLEABLE_DISTRIBUTIONS.has(ir.op)) return ir;
  // Anything else (normalize / superpose / truncate / pushfwd /
  // unknown ops): return as-is. Downstream materialise will report
  // a precise error if the shape isn't supported.
  return ir;
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
function expandMeasureRefsInIR(ir: IRNode | null, derivations: any, visited: any): IRNode | null {
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
    out.fields = ir.fields.map((f: any) => ({
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

function expandMeasurePos(node: any, derivations: any, visited: any) {
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
//   - `obsIR`:    the obs argument's lowered IR. The viewer resolves
//                 it to a concrete JS value at materialise time via
//                 resolveIRToValue + fixedValues — same lookup the
//                 rest of the viewer uses for any binding ref, no
//                 separate eager-materialisation pass at classify
//                 time.
//
// The visualPanel materialiser uses this to issue one
// `worker.logDensityN` call: refArrays are populated from the prior's
// record fields plus any inner-binding samples the body refers to;
// observed comes from resolveIRToValue(d.obsIR, …); tally='clamped'. Per-atom log-likelihoods
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
function classifyBayesupdate(binding: any, bindings: any): DerivationBayesupdate | null {
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

  // Hold the obs IR; resolution to a JS value happens at materialise
  // time via resolveIRToValue + fixedValues. The classifier cares
  // only about the structural shape — does this binding look like a
  // bayesupdate over a likelihood of a kernel? — not about WHAT the
  // observation is.
  return {
    kind: 'bayesupdate',
    from: priorRef.name,
    bodyName,
    bodyIR,
    obsIR,
  };
}

module.exports = {
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
};
