// @flatppl/viewer — plot plan builder + concrete materialiser —
//
// buildPlotPlan inspects a binding's derivation + signature to
// produce the routing record renderers consume (mode: 'analytical'
// vs 'chain' vs 'samples'; axes; presets; domains). materialise-
// ConcreteMeasure substitutes user-set kwargs into a sampleable
// measure IR for the kernel-sample plot path.

import type { Ctx, Plan } from './types';
import { collectRefArrays } from './engine-facade.js';
import { sendWorker } from './worker.js';
import { resolveMeasureAlias } from './util.js';
// esbuild rewrites this CommonJS require at bundle time; declare it for
// the type-only tsc pass (the viewer tsconfig omits node types on
// purpose — this is browser code). The require here is a pre-existing
// lazy-load of engine-facade; not refactored as part of the typecheck
// cleanup.
declare function require(id: string): any;
export function buildPlotPlan(ctx: Ctx, binding: any /*, bindingsMap */): Plan | null {
  if (!binding || !ctx.derivationsState) return null;
  const name = binding.name;

  // Callable bindings (function / kernel / fn / likelihood) don't
  // get a derivation kind — they're functions, not random
  // variables. They take the profile-plot path: sweep one input
  // axis, hold the rest fixed, evaluate the body per point. The
  // engine's signatureOf + distributeAxes (orchestrator.js) shape
  // the input cartprod / cartpow into a flat list of scalar
  // axes; the UI layer here picks the default sweep axis +
  // default range and dispatches to worker.profileN.
  // Callable-layer bindings (function ∪ kernel: functionof / fn /
  // kernelof / bijection / fchain — and, in Phase 2, kernel-chain)
  // plus likelihood objects all route through the profile-plot path.
  // The set is grep-able via the producer tags; the predicate
  // `isCallableLayerBinding` reads inferredType.kind for the
  // type-driven view (engine-concepts §19.2).
  if (binding.type === 'functionof' || binding.type === 'fn'
      || binding.type === 'kernelof' || binding.type === 'bijection'
      || binding.type === 'fchain'
      || binding.type === 'likelihood') {
    if (!ctx.derivationsState.bindings) return null;
    const sig = FlatPPLEngine.orchestrator.signatureOf(name, ctx.derivationsState.bindings);
    if (!sig || !sig.body) return null;
    const axes = FlatPPLEngine.orchestrator.distributeAxes(sig);
    if (axes.length === 0) return null;
    const presets = FlatPPLEngine.orchestrator.findMatchingPresets(
      sig, ctx.derivationsState.bindings);
    const domains = FlatPPLEngine.orchestrator.findMatchingDomains(
      sig, ctx.derivationsState.bindings);
    // On-demand specialize the output type at this synthetic call
    // site: scope = {paramName → input type}. typeinfer's
    // inferExprInScope handles polymorphic bodies — module-level
    // inference saw inputs as `any`, but here we have concrete
    // types from sig.inputs[i].type (which signatureOf already
    // resolved through paramSources). For multi-output bodies
    // (record/tuple/array of scalars), enumerateOutputLeaves
    // gives one entry per scalar leaf the user can pick from.
    let outputs: any[] = [];
    try {
      const paramTypes = new Map();
      for (let ii = 0; ii < sig.inputs.length; ii++) {
        paramTypes.set(sig.inputs[ii].paramName,
                       sig.inputs[ii].type || { kind: 'any' });
      }
      const specOutType = sig.body && ctx.currentLoweredModule
        ? FlatPPLEngine.typeinfer.inferExprInScope(
            ctx.currentLoweredModule, sig.body, paramTypes)
        : (sig.output && sig.output.type) || null;
      outputs = FlatPPLEngine.orchestrator.enumerateOutputLeaves(specOutType);
    } catch (_) {
      // Fall back to module-level type if specialization fails.
      outputs = FlatPPLEngine.orchestrator.enumerateOutputLeaves(
        sig.output && sig.output.type);
    }
    // Default to the first leaf — single entry with empty path
    // for scalar outputs, so the existing pipeline works
    // unchanged.
    const outputKey = outputs.length > 0 ? outputs[0].key : null;
    // Kernels (sig.kind === 'kernel') don't get a swept-axis
    // profile plot — there's nothing to "sweep" without an
    // observation. Instead we treat them like other measure
    // bindings: pick a preset (or auto-defaults), substitute
    // those into the kernel body, sample N times, and show the
    // resulting empirical measure as a histogram / corner plot.
    if (sig.kind === 'kernel') {
      return {
        name: name,
        mode: 'kernel-sample',
        signature: sig,
        axes: axes,
        matchedPresets: presets,
        presetName: null,            // null = "auto", string = named preset
        // Per-binding override for the auto pseudo-preset.
        // Auto's "values" depend on the binding's signature
        // (type defaults / cached source samples), so they
        // can't be shared module-wide. Reset when the user
        // navigates to a different binding (the plan is
        // rebuilt). Named-preset overrides live in the
        // module-wide presetOverrides map instead.
        //   null | { values: {kwarg: val} }
        autoOverride: null,
        // Domain selector state: same shape as the inputs side,
        // but driving x-axis range per kwarg from cartprod(...)
        // bindings. domainAutoOverride is the per-binding override
        // for the auto pseudo-domain (same lifetime as
        // autoOverride). Named domain overrides live module-wide
        // in domainOverrides.
        matchedDomains: domains,
        domainName: null,
        domainAutoOverride: null,
      };
    }
    return {
      name: name,
      mode: 'profile',
      signature: sig,
      axes: axes,
      sweepKey: axes[0].key,
      matchedPresets: presets,
      presetName: null,
      outputs: outputs,
      outputKey: outputKey,
      autoOverride: null,
      matchedDomains: domains,
      domainName: null,
      domainAutoOverride: null,
    };
  }

  const d = ctx.derivationsState.derivations[name];
  // A binding with no derivation can still be plottable when it has a
  // concrete fixed VALUE (typically a record / array from rand). The
  // phase-driven dispatch below routes those by inferredType alone.
  //
  // Demand-driven note (§17.4): the `fixedValues.has(name)` test in the
  // `if` below is intentionally VALUE-PRESENCE, NOT a phase check — it
  // distinguishes a fixed binding with a materialisable value from an
  // opaque one (e.g. rngstate: fixed-phase but no plottable value) that
  // must fall through to the implicit-kernel / Not-plottable branch. So
  // it stays `.has`, not `binding.phase === 'fixed'`. Under the lazy
  // resolver this resolves only THIS plotted binding (the user asked to
  // plot it — demand-driven by definition, not a build-time sweep).
  const fixedValues = ctx.derivationsState.fixedValues;
  // Or — and this is the implicit-kernelof escape hatch — a
  // stochastic binding can have its derivation pruned because
  // its distIR depends on a parameterized (elementof) ancestor.
  // Per spec §04, clicking on `x` is equivalent to plotting
  // `kernelof(x)` with no boundary kwargs: a kernel whose inputs
  // are x's elementof leaves. We synthesise that signature and
  // route through the kernel-sample plan shape — the user gets
  // the same Inputs dropdown they'd see on an explicit
  // `kernel = kernelof(x, mu = mu)` binding.
  if (!d && !(fixedValues && fixedValues.has(name))) {
    // Pass the LIFTED bindings (derivationsState.bindings, populated
    // by buildDerivations → liftInlineSubexpressions). The unlifted
    // currentBindings don't carry `.ir`, so the structural fallback
    // in expandMeasureIR can't walk them.
    //
    // Dispatch by phase:
    //   stochastic   → implicit kernel (synthesise `kernelof(x)` with
    //                  parametric leaves as inputs; kernel-sample plan).
    //   parameterized → implicit function (synthesise `functionof(x)`
    //                  with parametric leaves as inputs; profile plan).
    // Fixed-phase bindings with no fixedValue entry shouldn't reach
    // here (they'd be in fixedValues or have a derivation); fall
    // through to "Not plottable".
    if (binding.phase === 'stochastic') {
      const implicitSig = FlatPPLEngine.orchestrator.implicitKernelSignature(
        name, ctx.derivationsState.bindings, ctx.derivationsState.derivations);
      if (implicitSig && implicitSig.inputs.length > 0) {
        const iAxes = FlatPPLEngine.orchestrator.distributeAxes(implicitSig);
        if (iAxes.length > 0) {
          const iPresets = FlatPPLEngine.orchestrator.findMatchingPresets(
            implicitSig, ctx.derivationsState.bindings);
          const iDomains = FlatPPLEngine.orchestrator.findMatchingDomains(
            implicitSig, ctx.derivationsState.bindings);
          return {
            name: name,
            mode: 'kernel-sample',
            signature: implicitSig,
            axes: iAxes,
            matchedPresets: iPresets,
            presetName: null,
            autoOverride: null,
            matchedDomains: iDomains,
            domainName: null,
            domainAutoOverride: null,
          };
        }
      }
    } else if (binding.phase === 'parameterized') {
      const implicitFnSig = FlatPPLEngine.orchestrator.implicitFunctionSignature(
        name, ctx.derivationsState.bindings, ctx.derivationsState.derivations);
      if (implicitFnSig && implicitFnSig.inputs.length > 0) {
        const fAxes = FlatPPLEngine.orchestrator.distributeAxes(implicitFnSig);
        if (fAxes.length > 0) {
          const fPresets = FlatPPLEngine.orchestrator.findMatchingPresets(
            implicitFnSig, ctx.derivationsState.bindings);
          const fDomains = FlatPPLEngine.orchestrator.findMatchingDomains(
            implicitFnSig, ctx.derivationsState.bindings);
          const fOutputs = FlatPPLEngine.orchestrator.enumerateOutputLeaves(
            implicitFnSig.output && implicitFnSig.output.type);
          const fOutputKey = fOutputs.length > 0 ? fOutputs[0].key : null;
          return {
            name: name,
            mode: 'profile',
            signature: implicitFnSig,
            axes: fAxes,
            sweepKey: fAxes[0].key,
            matchedPresets: fPresets,
            presetName: null,
            outputs: fOutputs,
            outputKey: fOutputKey,
            autoOverride: null,
            matchedDomains: fDomains,
            domainName: null,
            domainAutoOverride: null,
          };
        }
      }
    }
    return null;
  }
  const discrete = !!ctx.derivationsState.discrete[name];

  // Phase-driven dispatch (per spec §sec:phases):
  //   'stochastic'   → atoms vary across i; histogram / corner plot.
  //   'fixed'        → compile-time-determinate object. Sub-cases by
  //                    inferredType.kind:
  //                      value type (scalar/record/tuple/array)
  //                        → render the value as text. Scalars
  //                          additionally fall through to histogram
  //                          when the per-atom samples differ
  //                          (engine-side broadcast, e.g. lp_obs).
  //                      measure type → atoms come from sampling the
  //                        fixed measure; histogram still applies.
  //   'parameterized' → handled via callable / input bindings above.
  // Records/tuples with phase='fixed' get text directly (no
  // measureIsConstant walk — phase has already classified them as
  // deterministic).
  const phase = binding.phase;
  const inferredType = binding.inferredType;
  const typeKind = inferredType && inferredType.kind;

  // Resolve through measure-equivalence aliases — applies
  // regardless of phase. The principle is "plot by what the
  // binding IS, not how it was constructed":
  //
  //   m = lawof(observed_data)         → alias to observed_data
  //   m = Dirac(observed_data)         → alias to observed_data
  //                                       (engine promotes Dirac-
  //                                        of-ref to alias kind)
  //   y = draw(m)  for any of the above → alias to m → … →
  //                                        observed_data
  //
  // All produce per-atom values identical to observed_data's,
  // so all should render identically. Use the source binding's
  // plan, but tag it with the original name so colorForBinding
  // picks up the alias's own binding-type color (lawof-blue,
  // measure-grey, draw-purple, …) instead of the underlying
  // value's color (literal pink, etc.). For non-aliased
  // bindings (the common case — Normal samples, posterior,
  // function bindings, etc.) resolveMeasureAlias returns null
  // and we fall through to the regular dispatch below.
  const sourceName = resolveMeasureAlias(name, ctx.derivationsState.derivations,
                                       ctx.currentBindings);
  if (sourceName && sourceName !== name) {
    const sourceBinding = ctx.currentBindings!.get(sourceName);
    if (sourceBinding) {
      const sourcePlan = buildPlotPlan(ctx, sourceBinding);
      if (sourcePlan) {
        const aliased = Object.assign({}, sourcePlan);
        aliased.name = name;
        return aliased;
      }
    }
  }

  if (phase === 'fixed') {
    // Opaque value-typed bindings — rngstate today, future
    // engine-internal types in the same vein — have no useful
    // visual representation. Drop them out of the plot pipeline
    // here (the alternative — falling through to samples mode —
    // produced an empty histogram of NaN values when the
    // per-atom evaluator coerced the opaque object to a Float64
    // entry).
    if (typeKind === 'rngstate') return null;
    if (typeKind === 'record' || typeKind === 'tuple') {
      return { name: name, mode: 'fixed-record' };
    }
    // Static numeric arrays take the dedicated step-plot / matrix
    // path. (kind:'array' derivation also implies phase='fixed' and
    // inferredType=array.) Routing keys on the binding's STATIC
    // type+phase+derivation-kind triple alone — `inferBroadcast`
    // tightened in typeinfer.ts (see test/typeinfer.test.ts
    // 'broadcast: user-defined callable resolves cell-type via
    // callee result') so dotted-broadcast results
    // (`Y = polyeval.([C], X)`, `tau = (bkg ./ dbkg) .^ 2`) now
    // type as `array(rank, shape, real)` and flow through here
    // naturally, no runtime-value fallback needed.
    if ((d && d.kind === 'array') || typeKind === 'array') {
      // Rank-2 fixed array → heatmap. Axis-length literalness is
      // not required at the type level — the runtime measure carries
      // an `intrinsicShape` field that the renderer reads at draw
      // time (see materialiser's fixedValueToMeasure). Static literal
      // axes are still forwarded when available, as a hint; the
      // renderer prefers the measure's shape when both are present.
      // Falls through to the 1D step plot for rank-1 arrays.
      if (typeKind === 'array' && inferredType.rank === 2) {
        const out: any = { name: name, mode: 'matrix' };
        if (Array.isArray(inferredType.shape)
            && inferredType.shape.length === 2
            && typeof inferredType.shape[0] === 'number'
            && typeof inferredType.shape[1] === 'number') {
          out.shape = [inferredType.shape[0], inferredType.shape[1]];
        }
        return out;
      }
      return { name: name, mode: 'array' };
    }
    if (typeKind === 'scalar') {
      return { name: name, mode: 'fixed-scalar', discrete: discrete };
    }
    // Falls through (typeKind === 'measure' / 'any' / 'deferred'):
    // sample-driven render below.
  } else {
    // phase='stochastic' (or unknown) — keep the sample path.
    if (d && d.kind === 'array') {
      return { name: name, mode: 'array' };
    }
  }

  // Variates never get a density overlay — see rule 1 above.
  // For measures, the overlay is the analytical PDF/PMF when the
  // resolved leaf has all-literal kwargs (closed-form marginal).
  //
  // "Is this a variate?" semantically = stochastic phase. Today
  // only `draw(...)` / `~` produce stochastic-phase value
  // bindings, so binding.type === 'draw' happens to match — but
  // phase is the spec-grounded discriminator and protects against
  // any future syntactic form that also yields a variate. (A
  // measure with stochastic ancestors will still reach this
  // branch and be filtered by the all-literal-kwargs gate below,
  // not the phase check; that's intentional.)
  let analyticalIR = null;
  if (binding.phase !== 'stochastic') {
    const leafIR = FlatPPLEngine.orchestrator.leafSampleIR(name, ctx.derivationsState.derivations);
    if (leafIR && leafIR.kind === 'call' && leafIR.op
        && (!leafIR.args || leafIR.args.length === 0)) {
      let allLit = true;
      const kw = leafIR.kwargs || {};
      for (const k in kw) {
        if (kw[k].kind !== 'lit') { allLit = false; break; }
      }
      if (allLit) analyticalIR = leafIR;
    }
  }
  return { name: name, mode: 'samples', discrete: discrete, analyticalIR: analyticalIR };
}

// Materialise a concrete (closed, no-refs) measure IR — used by
// the kernel-plot path after substituting placeholder values into
// the kernel body. The viewer doesn't case on FlatPPL operations
// itself; all measure-algebra dispatch lives in the engine. This
// function builds a matCtx mirroring engine-facade.getMeasure's
// shape and delegates to `materialiseMeasureIR`, the engine's
// IR-direct measure entry. Per-op routing (lawof / iid / joint /
// record / broadcast(Dist) / select / leaf-dist) is handled there.
export function materialiseConcreteMeasure(ctx: Ctx, ir: any, count: number, seed: number | null): Promise<any> {
  const matCtx: any = {
    derivations: ctx.derivationsState!.derivations,
    bindings:    ctx.derivationsState!.bindings,
    fixedValues: ctx.derivationsState!.fixedValues,
    getMeasure:  function(n: any) {
      return require('./engine-facade.js').getMeasure(ctx, n);
    },
    sendWorker:  function(m: any) { return sendWorker(ctx, m); },
    sampleCount: count,
    rootSeed:    seed != null ? seed : ctx.rootSeed,
    rejectionBudget: ctx.REJECTION_BUDGET,
  };
  return FlatPPLEngine.materialiser.materialiseMeasureIR(ir, matCtx);
}

