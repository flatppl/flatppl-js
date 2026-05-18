// @flatppl/viewer — plot plan builder + concrete materialiser (Phase 4f).
//
// buildPlotPlan inspects a binding's derivation + signature to
// produce the routing record renderers consume (mode: 'analytical'
// vs 'chain' vs 'samples'; axes; presets; domains). materialise-
// ConcreteMeasure substitutes user-set kwargs into a sampleable
// measure IR for the kernel-sample plot path.

import { collectRefArrays } from './engine-facade.js';
import { sendWorker } from './worker.js';
import { resolveMeasureAlias } from './util.js';
export function buildPlotPlan(ctx, binding /*, bindingsMap */) {
  if (!binding || !ctx.derivationsState) return null;
  var name = binding.name;

  // Callable bindings (function / kernel / fn / likelihood) don't
  // get a derivation kind — they're functions, not random
  // variables. They take the profile-plot path: sweep one input
  // axis, hold the rest fixed, evaluate the body per point. The
  // engine's signatureOf + distributeAxes (orchestrator.js) shape
  // the input cartprod / cartpow into a flat list of scalar
  // axes; the UI layer here picks the default sweep axis +
  // default range and dispatches to worker.profileN.
  if (binding.type === 'functionof' || binding.type === 'fn'
      || binding.type === 'kernelof' || binding.type === 'likelihood') {
    if (!ctx.derivationsState.bindings) return null;
    var sig = FlatPPLEngine.orchestrator.signatureOf(name, ctx.derivationsState.bindings);
    if (!sig || !sig.body) return null;
    var axes = FlatPPLEngine.orchestrator.distributeAxes(sig);
    if (axes.length === 0) return null;
    var presets = FlatPPLEngine.orchestrator.findMatchingPresets(
      sig, ctx.derivationsState.bindings);
    var domains = FlatPPLEngine.orchestrator.findMatchingDomains(
      sig, ctx.derivationsState.bindings);
    // On-demand specialize the output type at this synthetic call
    // site: scope = {paramName → input type}. typeinfer's
    // inferExprInScope handles polymorphic bodies — module-level
    // inference saw inputs as `any`, but here we have concrete
    // types from sig.inputs[i].type (which signatureOf already
    // resolved through paramSources). For multi-output bodies
    // (record/tuple/array of scalars), enumerateOutputLeaves
    // gives one entry per scalar leaf the user can pick from.
    var outputs = [];
    try {
      var paramTypes = new Map();
      for (var ii = 0; ii < sig.inputs.length; ii++) {
        paramTypes.set(sig.inputs[ii].paramName,
                       sig.inputs[ii].type || { kind: 'any' });
      }
      var specOutType = sig.body && ctx.currentLoweredModule
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
    var outputKey = outputs.length > 0 ? outputs[0].key : null;
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

  var d = ctx.derivationsState.derivations[name];
  // A binding with no derivation can still be plottable when the
  // orchestrator's pre-eval pass put a value in fixedValues
  // (typically a record / array from rand). The phase-driven
  // dispatch below routes those by inferredType alone.
  var fixedValues = ctx.derivationsState.fixedValues;
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
      var implicitSig = FlatPPLEngine.orchestrator.implicitKernelSignature(
        name, ctx.derivationsState.bindings, ctx.derivationsState.derivations);
      if (implicitSig && implicitSig.inputs.length > 0) {
        var iAxes = FlatPPLEngine.orchestrator.distributeAxes(implicitSig);
        if (iAxes.length > 0) {
          var iPresets = FlatPPLEngine.orchestrator.findMatchingPresets(
            implicitSig, ctx.derivationsState.bindings);
          var iDomains = FlatPPLEngine.orchestrator.findMatchingDomains(
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
      var implicitFnSig = FlatPPLEngine.orchestrator.implicitFunctionSignature(
        name, ctx.derivationsState.bindings, ctx.derivationsState.derivations);
      if (implicitFnSig && implicitFnSig.inputs.length > 0) {
        var fAxes = FlatPPLEngine.orchestrator.distributeAxes(implicitFnSig);
        if (fAxes.length > 0) {
          var fPresets = FlatPPLEngine.orchestrator.findMatchingPresets(
            implicitFnSig, ctx.derivationsState.bindings);
          var fDomains = FlatPPLEngine.orchestrator.findMatchingDomains(
            implicitFnSig, ctx.derivationsState.bindings);
          var fOutputs = FlatPPLEngine.orchestrator.enumerateOutputLeaves(
            implicitFnSig.output && implicitFnSig.output.type);
          var fOutputKey = fOutputs.length > 0 ? fOutputs[0].key : null;
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
  var discrete = !!ctx.derivationsState.discrete[name];

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
  var phase = binding.phase;
  var inferredType = binding.inferredType;
  var typeKind = inferredType && inferredType.kind;

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
  var sourceName = resolveMeasureAlias(name, ctx.derivationsState.derivations,
                                       ctx.currentBindings);
  if (sourceName && sourceName !== name) {
    var sourceBinding = ctx.currentBindings.get(sourceName);
    if (sourceBinding) {
      var sourcePlan = buildPlotPlan(ctx, sourceBinding);
      if (sourcePlan) {
        var aliased = Object.assign({}, sourcePlan);
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
    // Static numeric arrays still take the dedicated step-plot
    // path. (kind:'array' derivation also implies phase='fixed'
    // and inferredType=array.)
    //
    // Ground-truth fallback: route by WHAT THE BINDING IS, not by
    // the static type. A fixed-phase binding whose pre-evaluated
    // value (orchestrator fixedValues) is a flat numeric/boolean
    // vector IS an array value — even when inferredType came back
    // 'deferred' because the producing expression isn't covered by
    // typeinfer. The canonical case: `tau = (bkg ./ dbkg) .^ 2`
    // (dotted-broadcast typeinfer is intentionally loose, TODO
    // §07), which materialises byte-identically to the literal
    // `dbkg = [3.0, 7.0]` yet, without this, fell through to the
    // scalar-sample path and rendered as an empty 2-point
    // histogram. Tightening dotted-broadcast typeinfer would also
    // fix it at the source; this fallback hardens the viewer
    // against every present and future loose-typeinfer case.
    var fvMap = ctx.derivationsState.fixedValues;
    var fvVal = fvMap && fvMap.has(name) ? fvMap.get(name) : undefined;
    var isFlatNumericVec = Array.isArray(fvVal) && fvVal.length > 0
      && fvVal.every(function (e) {
        return typeof e === 'number' || typeof e === 'boolean';
      });
    if ((d && d.kind === 'array') || typeKind === 'array'
        || isFlatNumericVec) {
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
  var analyticalIR = null;
  if (binding.phase !== 'stochastic') {
    var leafIR = FlatPPLEngine.orchestrator.leafSampleIR(name, ctx.derivationsState.derivations);
    if (leafIR && leafIR.kind === 'call' && leafIR.op
        && (!leafIR.args || leafIR.args.length === 0)) {
      var allLit = true;
      var kw = leafIR.kwargs || {};
      for (var k in kw) {
        if (kw[k].kind !== 'lit') { allLit = false; break; }
      }
      if (allLit) analyticalIR = leafIR;
    }
  }
  return { name: name, mode: 'samples', discrete: discrete, analyticalIR: analyticalIR };
}

export function materialiseConcreteMeasure(ctx, ir, count, seed) {
  if (!ir) return Promise.reject(new Error('materialiseConcreteMeasure: null IR'));
  if (ir.kind !== 'call') {
    return Promise.reject(new Error(
      "materialiseConcreteMeasure: non-call IR (kind '" + ir.kind + "')"));
  }
  if (ir.op === 'lawof' && Array.isArray(ir.args) && ir.args.length === 1) {
    return materialiseConcreteMeasure(ctx, ir.args[0], count, seed);
  }
  if (ir.op === 'iid' && Array.isArray(ir.args) && ir.args.length >= 2) {
    var inner = ir.args[0];
    var dims = [];
    for (var di = 1; di < ir.args.length; di++) {
      var d = ir.args[di];
      if (!d || d.kind !== 'lit' || !Number.isInteger(d.value)) {
        return Promise.reject(new Error('materialiseConcreteMeasure: iid dim must be integer literal'));
      }
      dims.push(d.value);
    }
    var k = dims.reduce(function(p, n) { return p * n; }, 1);
    // Leaf-distribution inner: use sampleN's `repeat` so the per-
    // atom refArrays line up — atom i gets refArrays[i], then k
    // independent draws share it. Mirrors getMeasure's iid path.
    // Naive recursion with count*k would mis-index refArrays
    // (only `count` entries available, repeated k times by the
    // atom index — out-of-bounds for i >= count).
    var SAMPLEABLE = FlatPPLEngine.orchestrator.SAMPLEABLE_DISTRIBUTIONS;
    if (inner.kind === 'call' && SAMPLEABLE && SAMPLEABLE.has(inner.op)) {
      return collectRefArrays(ctx, inner).then(function(refArrays) {
        return sendWorker(ctx, {
          type: 'sampleN', ir: inner, count: count, repeat: k,
          refArrays: refArrays, seed: seed,
        });
      }).then(function(reply) {
        return FlatPPLEngine.empirical.arrayMeasure(reply.samples, dims, null);
      });
    }
    // Non-leaf inner (nested iid / record / joint inside iid).
    // The recursive form keeps the structure but doesn't handle
    // captured refs correctly under expansion — flag if we ever
    // hit it in practice. Today's kernel-sample bodies all
    // bottom out at leaf distributions after the IR pipeline.
    return materialiseConcreteMeasure(ctx, inner, count * k, seed).then(function(innerM) {
      return FlatPPLEngine.empirical.arrayMeasure(innerM.samples, dims, null);
    });
  }
  if ((ir.op === 'joint' || ir.op === 'record') && Array.isArray(ir.fields)) {
    var fieldNames = ir.fields.map(function(f) { return f.name; });
    var fieldIRs = ir.fields.map(function(f) { return f.value; });
    return Promise.all(fieldIRs.map(function(v, i) {
      return materialiseConcreteMeasure(ctx, v, count,
        seed != null ? (seed ^ (i + 1) * 0x9e3779b1) : null);
    })).then(function(subs) {
      var fields = {};
      for (var i = 0; i < fieldNames.length; i++) fields[fieldNames[i]] = subs[i];
      return FlatPPLEngine.empirical.recordMeasure(fields, null);
    });
  }
  // Leaf distribution (or unrecognised op — sampleN throws if
  // it's not in the registry). Captured self-refs in the dist's
  // kwargs (e.g. `Normal(mu = lit, sigma = pow(ref self sqrt_sigma, 2))`
  // after substituteLocals) are resolved per-atom via refArrays
  // — same mechanism getMeasure uses for closed-measure
  // sampling. Fixed-phase refs flow through the worker's session
  // env, so collectRefArrays drops them.
  return collectRefArrays(ctx, ir).then(function(refArrays) {
    return sendWorker(ctx, {
      type: 'sampleN', ir: ir, count: count, seed: seed,
      refArrays: refArrays,
    });
  }).then(function(reply) {
    // Phase 8: hand-built Measures populate `.value` for
    // consistency with materialiser-produced ones.
    var data = reply.samples;
    return {
      samples: data,
      value: { shape: [data.length], data: data },
      logWeights: null,
    };
  });
}
