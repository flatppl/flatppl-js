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
const { isEvaluable, liftInlineSubexpressions, classifyRandTuple } = require('./lift.ts');
const { dissolveBindings } = require('./dissolver.ts');
const { signatureOf, substituteLocals } = require('./signatures.ts');
const { FixedValues } = require('./fixed-values.ts');
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
  VECTOR_OUTPUT_DISTRIBUTIONS,
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
// ---------------------------------------------------------------------
// Callable-like binding-type predicate (canonical).
//
// Every classifier site that needs to recognise "this binding holds a
// function-like callable" consults this predicate instead of inlining
// its own `b.type === 'fn' || b.type === 'functionof' || ...` chain.
// Without a single source of truth the sets drifted across sites:
// fchain was missing from every callable-acceptance check, bijection
// was missing from some, etc.
//
// **Included**: every analyzer-tagged callable producer that can hold
// a function or kernel that's referenceable elsewhere (broadcast head,
// pushfwd map, filter / reduce / scan head, etc.). Specifically:
//   - 'fn'         — fn(...) hole-lifted anonymous function
//   - 'functionof' — explicit functionof / lambda
//   - 'kernelof'   — kernelof — reifies a stochastic sub-DAG as a kernel
//   - 'bijection'  — bijection(f, finv, logvol) — annotated function
//   - 'fchain'     — fchain(f1, f2, ...) — composition; surface-level
//                    fchain bindings whose value is the composed function
//
// **Excluded**: any binding type whose value is a measure or a non-
// callable object. The `isKernel` predicate elsewhere is intentionally
// narrower (kernel-producing only) and stays separate.
function isCallableLikeBindingType(t: string | undefined): boolean {
  return t === 'fn' || t === 'functionof' || t === 'kernelof'
      || t === 'bijection' || t === 'fchain';
}

function buildDerivations(bindings: Map<string, BindingInfo>) {
  // Pre-pass: lift inline subexpressions so every measure-arg position
  // is a bare ref and every value-arg is evaluable. After lifting, the
  // classifier below handles all forms uniformly — there's no special
  // case for inline weighted/normalize/superpose/draw inside another
  // measure expression.
  bindings = liftInlineSubexpressions(bindings);

  // Post-lift: dissolve broadcast / aggregate constructs whose body is
  // an inherently-batched single op (engine-concepts §20 / Phase 2 of
  // the dissolution migration). The dissolver rewrites
  //   broadcast(functionof(<safe_op>(_, _, …)), args…)
  // to a direct `<safe_op>(args…)` call. Downstream classifiers and
  // evaluators see the dissolved form, so dotted-binary surfaces
  // (`A .+ B`, `A .* B`, `.exp(X)`, …) bypass per-cell iteration and
  // route through ARITH_OPS_N's batched-broadcast path. Non-dissolvable
  // broadcasts (kernel-broadcast, table row-dispatch, recursive user-
  // fns) stay as-is — the existing `_broadcastApply` cold path handles
  // them unchanged.
  bindings = dissolveBindings(bindings);

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
    binding.bijection = { fName: fA.name, fInvName: fIA.name, logVolume } as {
      fName: string; fInvName: string; logVolume: any;
      registryName?: string;
      paramIRs?: Record<string, any>;
    };
    // Phase 5.1 Session 5e — synthetic MvNormal-lowering marker.
    //
    // `lift.inlineMvNormalLift` rewrites `MvNormal(mu, cov)` IR to
    // `pushfwd(<bij>, <iid>)` and emits a synthetic bijection binding
    // marked via `__mvnormalLowering = {muIR, covIR}`. The loop above
    // populates the AST-path metadata (fName / fInvName / logVolume)
    // from the synthetic `bijection(fn(_), fn(_), 0.0)` stubs as for
    // any user-written bijection; THIS block layers the §22 registry
    // contract on top — additively, per the load-bearing invariant
    // documented in resolveBijectionMeta (line ~2090).
    //
    // The marker is the single source of truth: lift recognises the
    // MvNormal shape and attaches __mvnormalLowering; the construction
    // loop here forwards it into binding.bijection.{registryName,
    // paramIRs} without itself trying to recognise MvNormal patterns.
    // Loose coupling: lift owns AST recognition, derivations owns the
    // registry-binding contract.
    //
    // paramIRs.L: lower_cholesky is an EVALUABLE_OPS member
    // (ir-shared.ts), so matPushfwd's resolveIRToValue pass evaluates
    // this call on demand at materialise time — same numerical path
    // matMvNormal's L computation already uses.
    if ((binding as any).__mvnormalLowering) {
      const m = (binding as any).__mvnormalLowering;
      binding.bijection.registryName = 'affine';
      binding.bijection.paramIRs = {
        L: { kind: 'call', op: 'lower_cholesky', args: [m.covIR] },
        b: m.muIR,
      };
    }
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

  // Fixed-phase value resolution is DEMAND-DRIVEN (engine-concepts §17.4).
  // `fixedValues` is a lazy, memoised, cycle-guarded resolver (FixedValues
  // — see fixed-values.ts), NOT an eager map. A binding's fixed value is
  // computed only when a consumer first asks for it — a shape const-eval
  // during pass-2 classification, the dead-end diagnostic below, or a
  // worker-env push at materialise time — then cached. The per-binding
  // evaluation logic (formerly the eager `while (progress)` sweep here)
  // lives verbatim in FixedValues._compute. `fixedValues` exposes a
  // Map-compatible surface (.has / .get / iterate) so the ~100 downstream
  // consumers that read it as a Map are untouched; only WHEN values are
  // computed changed (eager-sweep → first-demand).
  //
  // `resolveMeasureRef` and `isMeasureBinding` are defined / available
  // here (they also consult `derivations` / `expandMeasureIR`) and are
  // injected into the resolver below.
  const samplerLib = require('./sampler.ts');
  // resolveMeasureRef closure threaded through evaluateExpr → evaluateRand
  // → the measure walker (sampler.walk). When the walker hits a
  // `(ref self <name>)` for a measure operand it consults this to recover
  // the measure IR.
  //
  // Two paths here. For named bindings that classify as a measure
  // derivation (sample / record / iid / weighted / alias), use
  // expandMeasureIR — this canonicalises through the derivation
  // graph, turning e.g. `prior = lawof(record(theta1=draw(M1),
  // theta2=draw(M2)))` into the sampleable `joint(theta1=M1,
  // theta2=M2)` shape that the walker can sample directly. For
  // anonymous lift-introduced bindings or any case expandMeasureIR
  // can't resolve, fall back to the raw lowered IR — those tend to
  // already be primitive distribution calls that the walker handles.
  function resolveMeasureRef(refName: any) {
    const expanded = expandMeasureIR(refName, derivations);
    if (expanded) return expanded;
    const b = bindings.get(refName);
    return (b && b.ir) || null;
  }

  const fixedValues = new FixedValues({
    bindings,
    derivations,
    resolveMeasureRef,
    isMeasureBinding,
    samplerLib,
    expandMeasureIR,
    collectSelfRefs,
    lowerExpr,
  });

  // Second classification pass. Classifiers that depend on
  // constant-resolution of a fixed-phase binding (e.g. classifyIid
  // resolving `iid(M, n)` where `n = lengthof(data)` is a fixed-phase
  // binding) can succeed here even though the first pass ran with the
  // derivations table still incomplete: `fixedValues` is the lazy
  // resolver (FixedValues), so `resolveConstant(n, …, fixedValues)`
  // computes `n` (and only `n`'s subgraph) ON DEMAND now that pass-1
  // derivations exist. Only re-classify bindings that didn't already get
  // a derivation in pass 1 — a pass-1 classification may be load-bearing
  // (e.g. a `vector(...)`
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
  // Binding types that are "legitimately underived" — they hold
  // first-class objects (callables, measures, likelihoods, raw
  // inputs) whose value isn't a sample-able derivation. The
  // dead-end check below skips these. Callable types route through
  // `isCallableLikeBindingType` so adding a new callable producer
  // (e.g. a future first-class `fchain`-via-aggregate combinator)
  // updates one place, not seven.
  function _isObjectBindingType(t: string): boolean {
    if (t === 'input' || t === 'lawof' || t === 'likelihood') return true;
    return isCallableLikeBindingType(t);
  }
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
  // must end up either resolvable (fixedValues) or classified
  // (derivations); neither means the engine silently gave up on a
  // deterministic computation.
  //
  // Demand-driven note (§17.4): with the lazy resolver this `.has(name)`
  // is the ONE intentional bounded forcing point. It resolves only
  // bindings that are fixed-phase AND not object-typed AND have no
  // derivation — exactly the set the old eager sweep would have left
  // without a value (an engine gap). Bindings with a derivation are
  // skipped above, so this never force-resolves the common case. The
  // headline laziness win (`A = load_huge_matrix(); B = expensive(A)`
  // never displayed) holds whenever A/B carry a derivation or are
  // object-typed; a *bare* underived fixed-phase value binding is still
  // resolved here, same as before. Fully eliminating that residual would
  // need a reachability-from-plotted-target gate not available at build
  // time — deferred (TODO §06).
  for (const [name, b] of bindings) {
    if (!b || b.phase !== 'fixed') continue;
    if (_isObjectBindingType(b.type)) continue;         // legit underived
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
  // by density.ts (logDensityConsumeN) on K's body with env carrying the
  // prior's atom (tally='clamped'). We carry that out at materialise time
  // rather than synthesising an intermediate logweighted IR — density.ts
  // already implements the lowered primitive.
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
    //
    // Post-5f, an `MvNormal` IR node only survives to this branch when
    // the §22 lift gate SKIPPED it (static-D MvNormal is already
    // `pushfwd(affine, iid)` by the time we classify — no MvNormal node
    // left). So `kind='mvnormal'` is now the gate-skip fallback for:
    // dynamic-shape cov, matrix-form mean (rank-1 guard), and positional
    // form. It dispatches to matMvNormal (materialiser.ts), the §22
    // terminal materialiser. Intentional; retirement gated on 5h
    // (dynamic-D iid routing). See lift.inlineMvNormalLift +
    // mat-multivariate.matMvNormal's docstring.
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
    // User-call to a kernel-first jointchain/kchain binding —
    // `applied_chain = chain(theta = 0.5)`. lift.ts inlines the
    // analogous case for kernelof bindings (the call's IR becomes
    // the kernel body with args substituted into the placeholders);
    // for chains the body isn't a single expression, it's the chain
    // structure. We instead synthesise a fresh jointchain derivation
    // with the chain's first step replaced by an APPLIED measure
    // (K0's body with the chain's residual inputs substituted by the
    // call's kwargs). Subsequent steps are inherited verbatim;
    // matJointchain then materialises as a closed-first chain.
    // (Engine-concepts §19; spec §06 "uniform kernel extension" —
    // applying a kernel binds its inputs and yields a measure.)
    {
      const appliedChain = classifyAppliedChain(rhsIR, bindings);
      if (appliedChain) return appliedChain;
    }
    // `samples, _ = rand(state, iid(<composite M>, n))` destructures to
    // `tuple_get(<rand>, 0)`. The DRAW of a forward composite measure
    // (lawof of a broadcast/aggregate, a pushfwd, …) can't be sampled by
    // the per-draw measure walker (sampler.walk), but the batched materialiser can —
    // so route it there on demand (engine-concepts §17.4 stage 2). Leaf
    // rand stays on the existing batched-leaf path (sampleLeafN via
    // pre-eval); the gate lives in classifyRandSample. Checked before the
    // generic `evaluate` fallback below (tuple_get is otherwise evaluable).
    {
      const randSample = classifyRandSample(rhsIR, bindings, fixedValues);
      if (randSample) return randSample;
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
  // Spec §06: `weighted(weight, base)` accepts a FUNCTION of the
  // variate as weight. Check this BEFORE the evaluable-arith fall-
  // through — a self-ref to a function-typed binding is
  // syntactically evaluable but semantically a callable (no
  // per-atom value to read); we need to substitute the function's
  // parameter with the base ref so the body evaluates per atom.
  const fnDeriv = _classifyWeightedByFunction(weightExpr, baseName, bindings);
  if (fnDeriv) return fnDeriv;
  if (isEvaluable(weightExpr)) {
    return { kind: 'weighted', from: baseName, weightIR: weightExpr, isLog: false };
  }
  return null;
}

// Recognise a function-of-variate weight: weightExpr is a self-ref
// to a callable-layer binding whose functionof body has a SINGLE
// parameter (the variate). Returns a synthesised weightIR with the
// parameter substituted by `(%ref self <baseName>)`. Returns null
// when the shape doesn't match.
//
// Why single-parameter: spec §06 says weight is "a function of the
// variate x of M". Multi-parameter functions imply additional inputs
// the engine doesn't have a binding source for in this context;
// they're out of scope for now (could be lifted to additional
// per-atom refs in a follow-up).
function _classifyWeightedByFunction(
  weightExpr: any, baseName: string, bindings: any,
): DerivationWeighted | null {
  if (!weightExpr || weightExpr.kind !== 'ref' || weightExpr.ns !== 'self') return null;
  const fnBinding = bindings.get(weightExpr.name);
  if (!fnBinding) return null;
  const fnIR = fnBinding.ir;
  if (!fnIR || fnIR.kind !== 'call' || fnIR.op !== 'functionof'
      || !Array.isArray(fnIR.params) || fnIR.params.length !== 1
      || !fnIR.body) return null;
  // Callable-layer check (function-layer or kernel-layer per engine-
  // concepts §19.2). Aliases to standard-module functions pass too
  // — the alias-resolution pass canonicalises their RHS to a module-
  // namespaced ref, and the inferred function type is set by typeinfer's
  // cross-module-ref resolution.
  const ic = fnBinding.inferredType && fnBinding.inferredType.kind;
  if (ic !== 'function' && ic !== 'kernel') return null;
  const paramName = fnIR.params[0];
  // Replace every `(ref %local <paramName>)` in the body with
  // `(ref self <baseName>)`. mapIR's identity-preserving rebuild
  // means any sub-tree that doesn't mention the parameter shares
  // references with the original — the binding's body keeps its
  // IR intact for other consumers (signatures, dag).
  const { mapIR } = require('./ir-walk.ts');
  const synth = mapIR(fnIR.body, (n: any) => {
    if (n && n.kind === 'ref' && n.ns === '%local' && n.name === paramName) {
      return { kind: 'ref', ns: 'self', name: baseName, loc: n.loc };
    }
    return n;
  });
  return { kind: 'weighted', from: baseName, weightIR: synth, isLog: false };
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

// Demand-driven composite `rand` draw (engine-concepts §17.4 stage 2).
//
// `samples, _ = rand(state, iid(M, count))` lowers (multi-LHS) to
// `samples = tuple_get(%mlhs, 0)` / `_ = tuple_get(%mlhs, 1)` with
// `%mlhs = rand(state, iid(M, count))`. This classifier intercepts the
// DRAW half — `tuple_get(<rand>, 0)` — and, when `M` is a COMPOSITE
// measure, routes it to the batched materialiser (kind `randsample`)
// instead of the per-draw measure walker (sampler.walk) that `evaluate`
// would use.
//
// Why: the walker can sample leaf distributions and simple measure
// algebra, but not a forward composite like `lawof(<broadcast over a
// stochastic iid vector>)` — it has no walker for `aggregate` /
// arbitrary value ops in measure position. The materialiser already
// samples such a measure correctly (it materialises the whole ancestor
// DAG); we just need `count` independent draws, which is exactly
// "materialise M at sampleCount = count" (each atom = one iid draw).
//
// Leaf gate: a single known-distribution inner stays on the existing
// path (pre-eval computes it via the batched `sampleLeafN`, preserving
// the bit-for-bit `builtin_sample ≡ rand+iid` invariant from stage 1).
// The fixedValues short-circuit in the materialiser keeps that path
// authoritative even if a randsample derivation were also present, but
// gating here keeps intent honest and avoids dead derivations.
//
// Scope: the DRAW half (index 0). The composite STATE half
// (`tuple_get(…,1)` — a successor rngstate for chaining a second rand) is
// handled UPSTREAM by lift's `rewriteCompositeRandSucc`, which rewrites the
// binding IR to the value-domain `rand_succ` op (so it arrives here as an
// ordinary evaluable, classified `kind:'evaluate'`); the leaf state half
// stays threaded via sampleLeafN. The decompose + leaf gate is the shared
// `classifyRandTuple` (lift.ts) so the draw and successor halves can't drift.
function classifyRandSample(
  rhsIR: IRNode, bindings: any, fixedValues?: any,
): any {
  const t = classifyRandTuple(rhsIR, bindings, 0);   // 0 = the DRAW half
  if (!t || !t.isComposite) return null;             // leaf draw stays on the batched path
  // Resolve the iid count (literal, or a fixed-phase binding ref via the
  // pre-eval cache). Null until fixedValues is populated → returning
  // null defers classification to the post-pre-eval pass.
  const count = resolveConstant(t.countIR, bindings, new Set(), fixedValues);
  if (count == null || !Number.isInteger(count) || count <= 0) return null;

  return { kind: 'randsample', from: t.fromIR.name, count, stateIR: t.stateIR };
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
  if (!k || k.kind !== 'ref' || k.ns !== 'self') return null;
  // Bare distribution-constructor kernel, not shadowed by a binding.
  const isBareDist = SAMPLEABLE_DISTRIBUTIONS.has(k.name) && !bindings.has(k.name);
  // Phase 5.1 Session 4: bare vector-output dist (MvNormal etc.).
  // Falls outside SAMPLEABLE_DISTRIBUTIONS because the worker's
  // sampleN is scalar-only by design — engine-concepts §22.2(a). The
  // runtime (matKernelBroadcast) dispatches these to a per-cell
  // materialiser path that consumes the bijection registry directly.
  const isBareVectorDist = VECTOR_OUTPUT_DISTRIBUTIONS.has(k.name)
    && !bindings.has(k.name);
  // Composite kernel-of-iid binding — fusion (b) Phase F shape.
  // Recognise broadcast(<user-kernel-binding>, …) when the binding
  // is `kernelof(iid(<BuiltinDist>, n), kw)` (lowered to
  // `functionof(lawof(iid(<BuiltinDist>, n)), kw)`). The runtime
  // (matKernelBroadcast) detects and handles this case.
  const isIidComposite = !isBareDist && !isBareVectorDist
    && _isIidCompositeKernelBinding(k.name, bindings);
  // Phase 4.2: joint-bodied user kernels — body shape is
  // `lawof(joint(<components>))` (positional or keyword), with each
  // component an anon-ref to a sampleable distribution. matKernel-
  // Broadcast dispatches to `_executeJointComposite` via the
  // COMPOSITE_BODY_RECOGNIZERS table.
  const isJointComposite = !isBareDist && !isBareVectorDist && !isIidComposite
    && _isJointCompositeKernelBinding(k.name, bindings);
  // Phase 4.3: jointchain-bodied user kernels — body shape is
  // `lawof(jointchain(<base>, <K_1>, …))`. Markov chain where each
  // step depends on the previous step's variate. matKernelBroadcast
  // dispatches to `_executeJointChainComposite` via the
  // COMPOSITE_BODY_RECOGNIZERS table.
  const isJointChainComposite = !isBareDist && !isBareVectorDist
    && !isIidComposite && !isJointComposite
    && _isJointChainCompositeKernelBinding(k.name, bindings);
  // Phase 4.4: nested-broadcast-bodied user kernels — body shape is
  // `lawof(broadcast(<bare_dist>, <kwargs>))`. Outer kernel-broadcast
  // dispatches to `_executeNestedBroadcastComposite`.
  const isNestedBroadcastComposite = !isBareDist && !isBareVectorDist
    && !isIidComposite && !isJointComposite && !isJointChainComposite
    && _isNestedBroadcastCompositeKernelBinding(k.name, bindings);
  // engine-concepts §21 5th kind: generative-bodied user kernels — body
  // shape is `lawof(<value-expr>)` where the value-expr closes over an
  // internal draw (a hoisted `draw(<DistCall>)` that isn't a boundary).
  // Tried LAST (most permissive) so it never shadows the four measure-
  // construction recognisers. matKernelBroadcast dispatches to
  // `_executeGenerativeComposite` via the COMPOSITE_BODY_RECOGNIZERS table.
  // Without this gate the binding falls through to the silent `null` below
  // (no derivation → the variate never materialises).
  const isGenerativeComposite = !isBareDist && !isBareVectorDist
    && !isIidComposite && !isJointComposite && !isJointChainComposite
    && !isNestedBroadcastComposite
    && _isGenerativeCompositeKernelBinding(k.name, bindings);
  if (!isBareDist && !isBareVectorDist && !isIidComposite && !isJointComposite
      && !isJointChainComposite && !isNestedBroadcastComposite
      && !isGenerativeComposite) return null;
  const argIRs = rhsIR.args.slice(1);
  const kwargIRs = rhsIR.kwargs ? Object.assign({}, rhsIR.kwargs) : null;
  if (argIRs.length === 0 && (!kwargIRs || Object.keys(kwargIRs).length === 0)) {
    return null;   // no parameter inputs → not a broadcast
  }
  return { kind: 'kernelbroadcast', distOp: k.name, argIRs: argIRs, kwargIRs: kwargIRs };
}

// Delegate to the shared classifier in `kernel-broadcast-shape.ts`.
// P7 (LANDED 2026-05-30) hoisted the structural recognition into a
// single source of truth; both classify-time (`derivations`) and
// runtime (`mat-broadcast`) consumers share it.
const _kernelBroadcastShape = require('./kernel-broadcast-shape.ts');
function _isIidCompositeKernelBinding(name: string, bindings: any): boolean {
  return _kernelBroadcastShape.isIidCompositeKernelBinding(name, bindings);
}
function _isJointCompositeKernelBinding(name: string, bindings: any): boolean {
  return _kernelBroadcastShape.isJointCompositeKernelBinding(name, bindings);
}
function _isJointChainCompositeKernelBinding(name: string, bindings: any): boolean {
  return _kernelBroadcastShape.isJointChainCompositeKernelBinding(name, bindings);
}
function _isNestedBroadcastCompositeKernelBinding(name: string, bindings: any): boolean {
  return _kernelBroadcastShape.isNestedBroadcastCompositeKernelBinding(name, bindings);
}
function _isGenerativeCompositeKernelBinding(name: string, bindings: any): boolean {
  return _kernelBroadcastShape.isGenerativeCompositeKernelBinding(name, bindings);
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

// broadcast(fn(logdensityof(M, _)), pts) — the idiomatic broadcasting
// surface (and its dot-sugar fn(logdensityof(M,_)).(pts)) for grid
// density evaluation. Lowers to a 2-arg broadcast whose head is an
// inline functionof of one parameter whose body is EXACTLY
// logdensityof(<self-measure>, <that param>). Recognise that precise
// shape and emit the SAME broadcast_logdensity derivation the bare
// 3-arg broadcast(logdensityof, M, pts) form produces, so it reuses
// matBroadcastLogdensity verbatim. Anything richer (a transformed
// point, a scaled/offset density, multiple params) does NOT match and
// falls through unchanged.
function classifyBroadcastFnLogdensity(
  rhsIR: IRNode, ast: any, bindings: any,
): DerivationBroadcastLogdensity | null {
  if (rhsIR.op !== 'broadcast' || !Array.isArray(rhsIR.args)
      || rhsIR.args.length !== 2) return null;
  const headIR: any = rhsIR.args[0];
  const pIR: any = rhsIR.args[1];
  // Head: inline functionof with exactly one parameter.
  if (!headIR || headIR.kind !== 'call' || headIR.op !== 'functionof'
      || !Array.isArray(headIR.params) || headIR.params.length !== 1) {
    return null;
  }
  const paramName = headIR.params[0];
  const body: any = headIR.body;
  // Body: exactly logdensityof(<measure>, <param>).
  if (!body || body.kind !== 'call' || body.op !== 'logdensityof'
      || !Array.isArray(body.args) || body.args.length !== 2) {
    return null;
  }
  const mIR: any = body.args[0];
  const xIR: any = body.args[1];
  if (!isSelfRef(mIR) || !bindings.has(mIR.name)) return null;
  // The point must be the functionof's own parameter, untouched — not
  // a transformed expression. Match by name (ns is the local param ns).
  if (!xIR || xIR.kind !== 'ref' || xIR.name !== paramName) return null;
  return { kind: 'broadcast_logdensity', measureName: mIR.name, pointsIR: pIR };
}

// `broadcast` is overloaded: stochastic kernel-broadcast
// (broadcast(Normal, mus, sigmas)) vs. broadcast(logdensityof, M,
// pts). Try the logdensity form first; fall back to kernel-broadcast
// (which only matches a bare SAMPLEABLE head, so the two never
// collide).
function classifyBroadcast(rhsIR: IRNode, ast: any, bindings: any): DerivationBroadcastLogdensity | DerivationKernelBroadcast | null {
  return classifyBroadcastLogdensity(rhsIR, ast, bindings)
      || classifyBroadcastFnLogdensity(rhsIR, ast, bindings)
      || classifyKernelBroadcast(rhsIR, ast, bindings);
}

// `logdensityof(M, x)` — per spec §sec:posterior, evaluate M's
// log-density at x. Result is REAL (a value, not a measure), but the
// classifier dispatch lives here uniformly: the materialiser computes
// per-prior-atom values via density.ts (logDensityConsumeN, tally='clamped'), so each
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
  // Callable-like binding-type check via the canonical predicate.
  // pushfwd(f, M) accepts every function-like callable producer
  // (fn / functionof / kernelof / bijection / fchain) — fchain
  // bindings as the map were missing before the consolidation;
  // routing through the predicate closes the gap.
  if (!isCallableLikeBindingType(fBinding.type)) {
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
      // Kernel detection by producer tag (functionof/kernelof/fn) OR
      // by inferredType (a kernel-first jointchain/kchain binding
      // whose inferred type is kernelType). Engine-concepts §19 +
      // §19.5: the layer is read from the type system; the producer
      // tag's traditional set isn't authoritative for ordinary-call
      // bindings that produce a kernel-typed value.
      const isKernel = !!b && (
        b.type === 'functionof' || b.type === 'kernelof' || b.type === 'fn'
        || (b.inferredType && b.inferredType.kind === 'kernel'));
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

/**
 * Recognise a user-call whose target is a kernel-first jointchain or
 * kchain binding and synthesise an applied-chain derivation.
 *
 * Pattern: `applied = chain(theta = 0.5)` where `chain = jointchain(
 * K0, K1)` is kernel-first. Today's lift.ts inlines analogous
 * user-calls for kernelof bindings by substituting the kernel body
 * into the call site; jointchain bodies aren't a single expression
 * so the analogous inlining isn't well-defined. We instead operate
 * at the derivation level:
 *
 *   1. Look up the chain binding and re-classify it (the chain
 *      derivation may already be built, but we re-derive to keep
 *      classifier dependencies one-directional).
 *   2. The chain must be kernel-first (else the call is meaningless
 *      — closed-first chains have no inputs to bind).
 *   3. Substitute the kwargs into the first step's kernel body:
 *      walk K0's body, replace `(ref %local <param>)` for each bound
 *      param with the corresponding kwarg IR. The result is a closed
 *      measure expression (K0's body returns a measure).
 *   4. Synthesise a new jointchain derivation whose step 0 is the
 *      substituted measure (role: 'base', kernel: false) and whose
 *      steps 1..N are the original chain's remaining steps.
 *   5. matJointchain materialises the result as an ordinary closed-
 *      first chain.
 *
 * Returns null for shapes outside this pattern: non-self refs,
 * targets that aren't chains, closed-first chains (no inputs to
 * apply), or chains whose first step's kernel body isn't a
 * functionof we can substitute into (e.g. inline measure IR in the
 * base step — covered by a follow-up).
 */
function classifyAppliedChain(rhsIR: any, bindings: any): any {
  if (!rhsIR || rhsIR.kind !== 'call' || !rhsIR.target) return null;
  if (rhsIR.target.ns !== 'self') return null;
  const targetName = rhsIR.target.name;
  if (!bindings.has(targetName)) return null;
  const target = bindings.get(targetName);
  if (!target || !target.ir || target.ir.kind !== 'call') return null;
  if (target.ir.op !== 'jointchain' && target.ir.op !== 'kchain') return null;
  // Re-classify the chain to get its step structure.
  const chainDeriv = classifyJointchain(target.ir, target.node && target.node.value, bindings);
  if (!chainDeriv || !Array.isArray(chainDeriv.steps) || chainDeriv.steps.length < 2) {
    return null;
  }
  // Only kernel-first chains can be applied (closed-first has no
  // inputs to bind).
  const baseStep = chainDeriv.steps[0];
  if (!baseStep.kernel) return null;
  // Extract K0's body. The base must be a ref to a functionof binding;
  // the inline `kernelIR` case is handled by a separate code path
  // (not yet wired — follow-up).
  if (!baseStep.ref) return null;
  const k0bind = bindings.get(baseStep.ref);
  if (!k0bind || !k0bind.ir || k0bind.ir.kind !== 'call'
      || k0bind.ir.op !== 'functionof') return null;
  const k0body = k0bind.ir.body;
  const k0params = k0bind.ir.params || [];
  if (!k0body || k0params.length === 0) return null;
  // Every param must be bound by a kwarg (full application only;
  // partial application is a follow-up that would need typeinfer
  // residual-input bookkeeping).
  const kwargs = rhsIR.kwargs || {};
  for (const p of k0params) {
    if (!Object.prototype.hasOwnProperty.call(kwargs, p)) return null;
  }
  // Substitute params with kwargs. Walk the body IR, replace any
  // (ref %local <param>) with the corresponding kwarg IR. Other refs
  // — self refs, other %local refs — pass through unchanged.
  function subst(node: any): any {
    if (node == null || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(subst);
    if (node.kind === 'ref' && node.ns === '%local'
        && k0params.indexOf(node.name) !== -1) {
      return kwargs[node.name];
    }
    const out: any = {};
    for (const k in node) {
      if (Object.prototype.hasOwnProperty.call(node, k)) out[k] = subst(node[k]);
    }
    return out;
  }
  const substitutedBody = subst(k0body);
  // Synthesise the applied chain derivation. Step 0 is now closed
  // (role: 'base', kernel: false). Steps 1..N are inherited verbatim.
  const newSteps = [
    { var: baseStep.var, role: 'base', kernel: false, measureIR: substitutedBody },
    ...chainDeriv.steps.slice(1),
  ];
  return {
    kind: 'jointchain',
    marginalize: chainDeriv.marginalize,
    labels: chainDeriv.labels,
    steps: newSteps,
  };
}

// Lebesgue(support = interval(a, b)) — the canonical continuous
// reference measure restricted to a finite interval. Spec §06: total
// mass equals the support's Lebesgue measure (= b − a for a 1-D
// interval). Sampling shape coincides with `Uniform(support =
// interval(a, b))` (same atom positions), so the derivation reuses
// the standard 'sample' kind with a synthetic Uniform distIR — but
// carries the correct `logTotalmass = log(b − a)` so a downstream
// `totalmass(M)` reads the spec-canonical value, not 1. `normalize`
// consumers cancel it out; `weighted(<function>, Lebesgue(...))`
// composes it with the per-atom function weights.
//
// Bounds resolution. classifyLebesgueInterval consults `resolveConstant`
// for each interval endpoint:
//   - both literal → log(b − a) attached to the derivation.
//   - either unresolvable (stochastic / refs-to-elementof) → the
//     totalmass shift becomes per-atom and currently isn't tracked;
//     the derivation drops it with `logTotalmass = undefined` and
//     downstream sees the default 0. Documented as an open follow-up;
//     in practice Lebesgue's interval bounds are literals.
//
// Higher-dim Lebesgue supports (`cartpow(interval, n)`, `cartprod(...)`)
// are a follow-up — each adds its own multi-axis sampling path with
// a corresponding totalmass formula. Tracked in TODO §06.
function classifyLebesgueInterval(rhsIR: IRNode, ast: any, bindings: any, fixedValues?: any): any {
  void ast;
  if (!rhsIR.kwargs) return null;
  const support = rhsIR.kwargs.support;
  if (!support || support.kind !== 'call' || support.op !== 'interval') return null;
  if (!Array.isArray(support.args) || support.args.length !== 2) return null;
  const aIR = support.args[0], bIR = support.args[1];
  // Synthetic Uniform sample distIR — same atom positions as
  // Lebesgue, the (b − a) factor lives in the derivation's
  // logTotalmass.
  const distIR = {
    kind: 'call', op: 'Uniform',
    kwargs: { support },
    loc: rhsIR.loc,
  };
  const out: any = { kind: 'sample', distIR, discrete: false };
  const aVal = resolveConstant(aIR, bindings, new Set(), fixedValues);
  const bVal = resolveConstant(bIR, bindings, new Set(), fixedValues);
  if (typeof aVal === 'number' && typeof bVal === 'number'
      && Number.isFinite(aVal) && Number.isFinite(bVal) && bVal > aVal) {
    out.logTotalmass = Math.log(bVal - aVal);
  }
  // else: per-atom or improper-support case — totalmass tracking
  // deferred (downstream sees default 0).
  return out;
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
  Lebesgue:     classifyLebesgueInterval,
};

/**
 * Whether a derivation's outgoing references are satisfiable —
 * i.e. every 'self' ref it contains points at a binding that itself
 * has a derivation. Aliases / weighted / normalize just check the
 * target.
 */
// True when a binding's value is a measure (not a plottable value).
// One owner, two consumers: the fixed-value resolver (so measure-context
// recursion never treats a measure as a value ref) and cascade-prune (a
// measure binding is resolvable downstream only via a derivation, never
// as a fixed value). Two ways to know: typeinfer's inferredType, or a
// lift-introduced synthetic anon whose IR head is measure-producing (the
// `MEASURE_PRODUCING` set the surface analyzer uses, reused rather than
// maintaining a parallel list).
function isMeasureBinding(b: any): boolean {
  if (!b) return false;
  const t = b.inferredType;
  if (t && (t.kind === 'measure' || t.kind === 'function' || t.kind === 'kernel')) return true;
  if (b.synthetic && b.ir && b.ir.kind === 'call' && b.ir.op
      && MEASURE_PRODUCING.has(b.ir.op)) return true;
  return false;
}

function derivationRefsValid(d: DerivationBase, derivations: any, bindings: Map<string, BindingInfo>, fixedValues: any) {
  // A name is "resolvable downstream" if there's a derivation for it
  // (the materialiser knows how to compute samples) OR it is a
  // fixed-phase VALUE the worker resolves through its session env.
  // The viewer's collectRefArrays already drops fixed-phase refs from
  // refArrays, so a binding whose only deps are fixed values can
  // still sample correctly via session env. Without this, a Normal(
  // mu=get_field(ref(rp), "theta1"), …) classified as 'sample' would
  // cascade-prune the moment the orchestrator dropped rp's
  // derivation (it's a record, not numeric — pre-eval drops those).
  //
  // Demand-driven (§17.4), but FAITHFUL to the old prune. Three cases:
  //   1. Has a derivation → resolvable (the materialiser computes it).
  //      First, and cheapest — no value resolution. This is the headline
  //      laziness win: a never-displayed `B = expensive(A)` where A/B
  //      carry derivations returns here WITHOUT resolving A.
  //   2. A measure binding with no derivation → NOT resolvable (measures
  //      are resolvable only via a derivation, so dropping one must
  //      cascade to dependents).
  //   3. A NON-measure fixed-phase / lift-anon binding with no derivation
  //      → confirm it actually HAS a value (`fixedValues.has`, which
  //      resolves on demand). This restores the old behavior exactly: a
  //      dependent of an UNEVALUABLE fixed binding (an engine-gap dead
  //      end) still cascade-prunes rather than surviving to fail at
  //      materialise. The forcing is bounded — underived fixed bindings
  //      are the same small set the dead-end diagnostic resolves anyway —
  //      so it costs no laziness beyond what's already paid, while a
  //      derivation-having (case 1) binding is never resolved here.
  function resolvable(name: any) {
    if (Object.prototype.hasOwnProperty.call(derivations, name)) return true;
    const b = bindings.get(name);
    if (!b || isMeasureBinding(b)) return false;
    if (b.phase != null && b.phase !== 'fixed') return false;
    return !!(fixedValues && fixedValues.has(name));
  }

  // Refs in CALLABLE-HEAD positions (args[0] of broadcast / aggregate
  // / pushfwd / etc.) are resolved at evaluation time via the env's
  // `__resolveFnBody` hook, NOT through derivations / fixedValues —
  // so they're "resolvable" via a different mechanism. Collect these
  // separately so the IR-refs walk below can skip them.
  //
  // Without this, an `evaluate`-classified `broadcast(<user-fn-ref>,
  // …)` (e.g. `Y = polyeval.([C], X)`) would be pruned the moment
  // the refs-valid sweep walked `<user-fn-ref>` and found polyeval
  // missing from both derivations and fixedValues. The materialiser's
  // broadcast handler resolves it correctly at evaluation time; the
  // cascade-prune sweep just needs to know.
  //
  // Narrow to: refs of a callable BINDING (fn / functionof /
  // kernelof / bijection) USED in a callable-head IR position. We
  // can't use type alone (a kernelof binding is also legitimately
  // used as a measure source in `logdensityof(k, x)` — see the
  // orchestrator test for that case, which must still cascade-prune
  // when k isn't materialisable as a measure).
  function _collectCallableHeadRefs(ir: any, seen?: Set<string>): Set<string> {
    if (!seen) seen = new Set();
    if (!ir || typeof ir !== 'object') return seen;
    if (ir.kind === 'call' && ir.op
        && (ir.op === 'broadcast' || ir.op === 'aggregate')
        && Array.isArray(ir.args) && ir.args.length > 0) {
      const head = ir.args[0];
      if (head && head.kind === 'ref' && head.ns === 'self') {
        const b = bindings.get(head.name);
        // Callable-like binding-type check via the canonical
        // predicate — recognises every function/kernel-like
        // callable producer (fn / functionof / kernelof / bijection
        // / fchain) as a legitimate head ref. Adding fchain here
        // surfaces `broadcast(myFchain, xs)` and similar correctly.
        if (b && isCallableLikeBindingType(b.type)) {
          seen.add(head.name);
        }
        // Aggregate's first arg is the REDUCER (sum / mean / prod /
        // …) — a builtin reducer reference rather than a user binding.
        // Builtins aren't in `bindings`, so the `b` check above
        // wouldn't have added them; but the cascade-prune sweep would
        // still see the ref via `collectSelfRefs` and try to resolve
        // it. Skip unconditionally for aggregate heads — the reducer
        // is resolved by the aggregate evaluator at run time, not by
        // a binding lookup.
        if (ir.op === 'aggregate') seen.add(head.name);
      }
    }
    // Recurse — broadcasts can nest inside other expression trees.
    if (ir.args)    for (const a of ir.args)            _collectCallableHeadRefs(a, seen);
    if (ir.kwargs)  for (const k in ir.kwargs)          _collectCallableHeadRefs(ir.kwargs[k], seen);
    if (Array.isArray(ir.fields)) for (const f of ir.fields) _collectCallableHeadRefs(f && f.value, seen);
    if (ir.body)                                        _collectCallableHeadRefs(ir.body, seen);
    return seen;
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
  const callableHeads = _collectCallableHeadRefs(ir);
  for (const r of refs) {
    // Callable-head refs (broadcast / aggregate heads bound to a
    // callable binding) resolve via __resolveFnBody at eval time,
    // not via derivations / fixedValues. Skip them in the cascade-
    // prune check.
    if (callableHeads.has(r)) continue;
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
 * suitable for the measure walker (sampler.walk). Walks the derivation graph,
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
  // Phase 5.1 Sessions 5c+5d — additive registry-driven fast path.
  //
  // Engine-concepts §22 architectural reframe: multivariate dists
  // decompose as `pushfwd(known_bijection, iid(scalar, D))`. When the
  // forward / inverse / logvolume functions of a bijection binding
  // implement a closed-form transform that's also in the open
  // bijection-registry (`bijection-registry.ts`), the producer marks
  // the binding with two parallel additive fields:
  //
  //   - `binding.bijection.registryName = '<name>'` — the registry-entry
  //     key (e.g. 'affine'). Session 5c (`188ffb5`) added forwarding.
  //   - `binding.bijection.paramIRs = { <name>: <IRNode>, ... }` — the
  //     parameter IRs the registry entry consumes at materialise time
  //     (e.g. `{L: <chol-IR>, b: <mu-IR>}` for 'affine'). Session 5d
  //     adds forwarding.
  //
  // Downstream consumers (Session 5d) read `ir.bijection.registryName`
  // and `ir.bijection.paramIRs` together to dispatch through the
  // registry's atom-batched fast paths. matPushfwd vector-base resolves
  // each paramIR via orchestrator.resolveIRToValue, then calls
  // entry.atomBatchedForward; walkPushfwd vector-base does the same on
  // entry.atomBatchedInverse + entry.logDetJ.
  //
  // Invariant: registryName + paramIRs are PURELY ADDITIVE. `fName` /
  // `fInvName` / `logVolume` MUST remain present and valid even when
  // registryName is set — the registry path is an OPTIMISATION over the
  // AST path, not a REPLACEMENT. matPushfwd's existing resolveFnBody →
  // fName walk continues to resolve a callable body for every bijection
  // binding regardless of registryName presence. This eliminates the
  // "degenerate binding" risk surface: callers that don't opt into the
  // registry path continue to work identically to pre-§22.
  //
  // Producer contract (Session 5e+): when `registryName` is set,
  // `paramIRs` MUST also be set. Consumers reject loudly if registryName
  // is present without paramIRs — that's a caller bug. The two fields
  // are emitted together by the lift-time MvNormal lowering pass.
  //
  // Plumbing: bij is binding.bijection (built in buildDerivations'
  // construction loop at lines 159-179); we forward registryName +
  // paramIRs here so they reach ir.bijection via expandMeasureIR (call
  // site in the pushfwd case, search the file for
  // `out.bijection = bijMeta`) and become visible to density.walkPushfwd.
  const out: any = { fInv, logVolume };
  if (bij.registryName) out.registryName = bij.registryName;
  if (bij.paramIRs)     out.paramIRs    = bij.paramIRs;
  return out;
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

/**
 * Unified measure-IR expansion (engine-concepts §17.4).
 *
 * Single canonical entry point. Replaces the four-way maze of
 * `expandMeasureIR` (by-name) + `expandMeasureRefsInIR` (by-IR) +
 * `_expandMeasureIRStructural` (binding-IR fallback) +
 * `expandMeasurePos` (dispatcher) — each of which handled overlapping
 * but subtly-different cases with optional `bindings` plumbing that
 * was load-bearing for correctness.
 *
 * Dispatch rules:
 *  - `input` is a **string** or **ref node** → look up the binding by
 *    name. Dispatch per derivation kind; if no derivation case fires,
 *    structurally walk binding.ir (the safety net for bindings that
 *    were pruned out of the derivation graph — e.g. depending on a
 *    parameterized ancestor).
 *  - `input` is a **call node** → walk structurally: peel `draw` /
 *    `lawof`, recurse on measure-position slots (joint fields, iid
 *    inner, weighted base, jointchain fields), pass through
 *    distribution leaves and unknown ops verbatim. Refs encountered
 *    during the walk recurse via the by-name path.
 *
 * `ctx` carries the lookup tables. `bindings` is required for
 * structural fallback (a binding whose derivation was pruned) to
 * fire — callers that have access to `bindings` should always pass
 * it. The backwards-compatible shims `expandMeasureIR` /
 * `expandMeasureRefsInIR` accept the old positional argument shape.
 */
function expandMeasure(input: any, ctx: any, visited?: Set<string>): IRNode | null {
  visited = visited || new Set<string>();
  if (input == null) return null;
  if (typeof input === 'string') {
    return _expandByName(input, ctx, visited);
  }
  if (input.kind === 'ref' && input.ns === 'self') {
    return _expandByName(input.name, ctx, visited);
  }
  if (input.kind !== 'call') return input;
  return _expandStructural(input, ctx, visited);
}

/**
 * Resolve a binding by name → its canonical measure IR. Walks the
 * derivation graph by kind; falls back to structurally walking
 * `binding.ir` when no derivation case fires (or no derivation
 * exists). Maintains a `visited` set keyed by binding name to break
 * cycles.
 */
function _expandByName(name: string, ctx: any, visited: Set<string>): IRNode | null {
  if (visited.has(name)) return null;
  const next = new Set(visited); next.add(name);
  const derivations = ctx && ctx.derivations;
  const bindings = ctx && ctx.bindings;
  const d = derivations && derivations[name];
  if (d) {
    switch (d.kind) {
      case 'alias':
        return _expandByName(d.from, ctx, next);
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
            inner = _expandByName(b.ref, ctx, next);
          } else if (b && b.ir) {
            inner = _expandStructural(b.ir, ctx, next);
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
      case 'kernelbroadcast': {
        // broadcast(Dist, args…) — array-valued independent product
        // measure (spec §04 sec:higher-order). The density walker's
        // walkBroadcast handler scores the resulting independent
        // product as Σ_j logpdf(y[j]; params_j); we rebuild the
        // canonical broadcast IR shape here so density / sample
        // consumers see the same self-contained measure IR as for
        // any other measure-algebra op. Previously this case
        // fell through to the structural fallback below, which
        // worked only when callers passed `bindings` — a silent
        // dependency that bit kernel-broadcast expansion in
        // kernel-body density paths.
        const head: any = { kind: 'ref', ns: 'self', name: d.distOp };
        const ir: any = { kind: 'call', op: 'broadcast', args: [head] };
        if (d.kwargIRs && Object.keys(d.kwargIRs).length > 0) {
          ir.kwargs = Object.assign({}, d.kwargIRs);
        } else if (Array.isArray(d.argIRs)) {
          for (let i = 0; i < d.argIRs.length; i++) ir.args.push(d.argIRs[i]);
        }
        return ir;
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
    return _expandStructural(b.ir, ctx, next);
  }
  return null;
}

/**
 * Structural walk of a measure-position IR. Unifies the former
 * `_expandMeasureIRStructural` (binding-IR safety net) and
 * `expandMeasureRefsInIR` (inline kernel-body expansion) into one
 * walker — they handled overlapping cases with subtle differences
 * (lawof unwrap, jointchain field recursion, sampleable
 * pass-through) that drifted as the engine grew. Now they share a
 * single set of structural rules.
 *
 * Rules:
 *   - refs in measure position → `_expandByName(ref.name, ctx)`
 *   - `draw(M)` / `lawof(M)` → unwrap recursively
 *   - `record` / `joint` / `jointchain` with fields → recurse on each
 *     field value (preserving `source` for env-threading)
 *   - `iid(M, …)` → recurse on args[0]
 *   - `weighted(w, M)` / `logweighted(g, M)` → recurse on args[1]
 *   - everything else (sampleable distributions, select, broadcast,
 *     normalize, superpose, truncate, pushfwd, unknown ops) → return
 *     unchanged. Refs in their kwargs are value-position (per-i
 *     params resolved at materialise / density time).
 */
function _expandStructural(ir: any, ctx: any, visited: Set<string>): any {
  if (!ir) return null;
  if (ir.kind === 'ref' && ir.ns === 'self') {
    // A ref encountered structurally: try to expand by name; if no
    // expansion is available (no derivation case + no bindings
    // fallback), keep the ref unchanged. This is the by-IR
    // contract — the caller's IR stays well-formed even when some
    // refs can't be resolved (e.g. the kernel-sample path expects
    // refs to be substituted by inlineForProfile / substituteLocals
    // downstream, not by the expansion walker). The by-name path
    // (expandMeasure(name, …)) returns null instead — caller can
    // tell whether the binding was resolvable.
    const expanded = _expandByName(ir.name, ctx, visited);
    return expanded || ir;
  }
  if (ir.kind !== 'call') return ir;
  if ((ir.op === 'draw' || ir.op === 'lawof')
      && Array.isArray(ir.args) && ir.args.length === 1) {
    return _expandStructural(ir.args[0], ctx, visited);
  }
  if (ir.op === 'iid' && Array.isArray(ir.args) && ir.args.length >= 2) {
    const inner: any = _expandStructural(ir.args[0], ctx, visited);
    if (!inner) return null;
    return { ...ir, args: [inner].concat(ir.args.slice(1)) };
  }
  if ((ir.op === 'record' || ir.op === 'joint' || ir.op === 'jointchain')
      && Array.isArray(ir.fields)) {
    const fields = ir.fields.map((f: any) => ({
      ...f, value: _expandStructural(f.value, ctx, visited),
    }));
    return { ...ir, fields };
  }
  if ((ir.op === 'weighted' || ir.op === 'logweighted')
      && Array.isArray(ir.args) && ir.args.length === 2) {
    const inner: any = _expandStructural(ir.args[1], ctx, visited);
    if (!inner) return null;
    return { ...ir, args: [ir.args[0], inner] };
  }
  // Sampleable distribution / select / broadcast / superpose / etc.:
  // pass through unchanged. Their kwargs / args hold value-position
  // refs that the density / sample walker resolves per-atom; we
  // don't substitute those at expansion time.
  return ir;
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

/**
 * Backwards-compatible shim. New code should use
 * `expandMeasure(input, ctx, visited)` directly. This wrapper exists
 * because the materialiser and a handful of tests call
 * `expandMeasureIR(name, derivations, visited, bindings)` with the
 * positional argument order — keeping it lets the unification be a
 * pure refactor with no API churn.
 *
 * Resolves a binding **by name** to its canonical measure IR.
 */
function expandMeasureIR(name: string, derivations: any, visited?: any, bindings?: any): IRNode | null {
  return expandMeasure(name, { derivations, bindings }, visited);
}

/**
 * Backwards-compatible shim. Walks an **inline measure-position IR**
 * via the same unified expander. Equivalent to the old
 * `expandMeasureRefsInIR` + `expandMeasurePos` + the
 * `_expandMeasureIRStructural` safety net, all collapsed into
 * `expandMeasure`'s call-IR branch.
 */
function expandMeasureRefsInIR(ir: IRNode | null, derivations: any, visited?: any, bindings?: any): IRNode | null {
  return expandMeasure(ir, { derivations, bindings }, visited);
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
// extending the evaluator to call density.ts (logDensityConsumeN) for a
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
  // Record the kernel's parametric inputs (the reified boundary names).
  // matBayesupdate must FEED these from the prior's atoms — per the spec
  // lowering bayesupdate(L,prior) = logweighted(fn(logdensityof(L,_)),prior),
  // the prior's variate IS the kernel's parametric input — rather than let
  // prepareDensityRefs re-materialise a like-named module binding via
  // getMeasure (the boundary-conflation bug, audit §3 / H1/H6). paramKwargs
  // are the call-site names the prior fields map onto.
  return {
    kind: 'bayesupdate',
    from: priorRef.name,
    bodyName,
    bodyIR,
    obsIR,
    paramKwargs: Array.isArray(Kir.paramKwargs) ? Kir.paramKwargs.slice()
      : (Array.isArray(Kir.params) ? Kir.params.slice() : []),
    params: Array.isArray(Kir.params) ? Kir.params.slice() : [],
  };
}

module.exports = {
  buildDerivations,
  classifyDerivation,
  // Canonical callable-binding-type predicate — see comment at its
  // definition for the included / excluded set.
  isCallableLikeBindingType,
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
  expandMeasure,
  expandMeasureIR,
  expandMeasureRefsInIR,
  implicitKernelSignature,
  implicitFunctionSignature,
  classifyBayesupdate,
};
