'use strict';

// @flatppl/engine — posterior-predictive sampling for bayesupdate posteriors.
//
// The posterior predictive of a bayesupdate model is `kchain(posterior,
// forward_kernel)` (spec §06 "Dependent composition": the documented
// `prior_predictive = kchain(prior, forward_kernel)` with the posterior
// substituted for the prior). Equivalently, it is the law of the forward
// (observation) variate when the latent parameters are distributed as the
// posterior rather than the prior.
//
// Rather than re-deriving replicated observations by hand (decoding iid /
// locscale / pushfwd shapes and calling the worker's low-level sample
// primitives), this builds the predictive by REUSING the generic materialiser:
// it pre-seeds a sub-context's `getMeasure` cache with the posterior parameter
// columns (keyed by latent name) and then materialises the forward variate.
// The materialiser samples the forward variate per posterior atom, handling ANY
// sampleable measure shape — leaf distributions, iid, truncate, superpose,
// nested composites, arbitrary bijections — so a PPC is produced for every
// bayesupdate model built with the standard `kernelof`/`likelihoodof`/
// `bayesupdate` construct, not just a shape whitelist.
//
// Weight handling: leaf-sample materialisation does not propagate importance
// weights (the worker's sample primitives return `logWeights: null`), so the
// posterior's top-level `logWeights` are re-attached onto the predictive draws
// here — broadcast atom-major across each posterior atom's expanded slots
// (vector variate and/or iid replicates).

const orchestrator = require('./orchestrator.ts');
const materialiser = require('./materialiser.ts');
const { scalarMeasureN } = require('./materialiser-shared.ts');

// ── Types ───────────────────────────────────────────────────────────────────

// Local interfaces (NOT `export`ed): an ESM `export` statement flips esbuild's
// module classification to ESM, which breaks the CJS `module.exports` at the
// bottom of this file in the browser engine bundle ("module is not defined").
// The runtime contract is the `module.exports` below; consumers read the result
// as `any` via FlatPPLEngine.posteriorPredictive.
interface YRepField {
  samples: Float64Array;
  logWeights: Float64Array | null;
}

interface PPCField {
  yRep: YRepField;
  observed: number[];
}

interface PPC {
  fields: Record<string, PPCField>;
}

// ── Observed-value resolution ────────────────────────────────────────────────

// Extract the observed values for a field from the resolved observation object.
// For a record observation (`record(y = y_data)`) the field is looked up by
// name; a bare observation (scalar/vector) is returned as-is.
function resolveObserved(obs: any, name: string): number[] {
  try {
    if (obs && typeof obs === 'object' && !Array.isArray(obs) && name in obs) {
      const raw = obs[name];
      return Array.isArray(raw) ? raw.map(Number) : [Number(raw)];
    }
    if (obs !== null && obs !== undefined) {
      return Array.isArray(obs) ? obs.map(Number) : [Number(obs)];
    }
  } catch (_) { /* fall through */ }
  return [];
}

// ── Main export ────────────────────────────────────────────────────────────

/**
 * Build a posterior-predictive check for a bayesupdate derivation.
 *
 * Supports any forward kernel whose output variate the engine can sample:
 * record outputs (`lawof(record(field = ...))`) yield one PPC field per record
 * field; a bare-variate output yields a single field. The inner distribution of
 * each field may be any sampleable measure (leaf, iid, truncate, superpose,
 * composite, pushfwd) — the generic materialiser handles it.
 *
 * Returns null (never throws) when:
 *  - `d` is not a bayesupdate or lacks required fields
 *  - `d.paramKwargs` is empty
 *  - the body is not a `lawof(...)`
 *  - a posterior parameter column is missing
 *  - a field's forward variate fails to materialise
 */
async function buildPosteriorPredictive(
  d: any,
  ctx: any,
  posteriorMeasure: any,
): Promise<PPC | null> {

  // Guard: must be a classified bayesupdate with a body and non-empty params.
  if (!d || d.kind !== 'bayesupdate') return null;
  if (!d.bodyIR && !d.bodyName) return null;
  if (!Array.isArray(d.paramKwargs) || d.paramKwargs.length === 0) return null;

  // Resolve the body IR (the forward kernel's output law). When bodyIR is
  // available use it directly; when only bodyName is available, resolve it via
  // the orchestrator.
  let bodyIR: any = d.bodyIR;
  if (!bodyIR && d.bodyName) {
    try {
      bodyIR = orchestrator.expandMeasureIR(d.bodyName, ctx.derivations, ctx.bindings);
    } catch (_) {
      return null;
    }
  }
  if (!bodyIR) return null;

  // The body must be `lawof(<measure>)`. The measure is either a `record(...)`
  // (per-field PPC) or a bare variate (single-field PPC).
  if (bodyIR.kind !== 'call' || bodyIR.op !== 'lawof') return null;
  if (!Array.isArray(bodyIR.args) || bodyIR.args.length < 1) return null;
  const outIR = bodyIR.args[0];
  if (!outIR) return null;

  // Enumerate the forward fields to materialise.
  let fieldSpecs: Array<{ name: string; ir: any }>;
  if (outIR.kind === 'call' && outIR.op === 'record' && Array.isArray(outIR.fields)
      && outIR.fields.length > 0) {
    fieldSpecs = outIR.fields.map((f: any) => ({ name: f.name, ir: f.value }));
  } else {
    // Bare variate — a single predictive field. Name it after the referenced
    // binding when available, else a neutral default.
    const nm = (outIR.kind === 'ref' && outIR.name) ? outIR.name : 'observed';
    fieldSpecs = [{ name: nm, ir: outIR }];
  }

  // Resolve the observed value(s) once (record or bare).
  let obs: any = null;
  if (d.obsIR) {
    try {
      obs = orchestrator.resolveIRToValue(d.obsIR, ctx.bindings, ctx.fixedValues);
    } catch (_) {
      obs = null;
    }
  }

  // Posterior atom count: all parameter columns share the same length.
  const firstParam = d.paramKwargs[0];
  if (!posteriorMeasure || !posteriorMeasure.fields
      || !posteriorMeasure.fields[firstParam]
      || !posteriorMeasure.fields[firstParam].samples) return null;
  const count: number = posteriorMeasure.fields[firstParam].samples.length;
  const postLW: Float64Array | null = posteriorMeasure.logWeights || null;

  // Sub-context whose getMeasure cache is PRE-SEEDED with the posterior
  // parameter columns, so materialising a forward variate draws its latents
  // from the posterior instead of re-sampling the prior. Mirrors the
  // getMeasure-cache idiom used by mcmcRun / the test ctx factory. Leaf columns
  // carry logWeights: null per the composite-measure invariant; the posterior
  // weight is re-attached below.
  const cache = new Map<string, any>();
  const subCtx: any = {
    ...ctx,
    // The materialiser reads these; the caller's PPC matCtx may omit them
    // (it only needs to sample a named posterior via sendWorker). Default them
    // so the forward-variate materialisation is self-sufficient. sampleCount
    // matches the posterior atom count so any non-seeded root draws align.
    sampleCount: ctx.sampleCount != null ? ctx.sampleCount : count,
    rootSeed: ctx.rootSeed != null ? ctx.rootSeed : ctx.rootKey,
    rootKey: ctx.rootKey,
    marginalizationCount: ctx.marginalizationCount != null ? ctx.marginalizationCount : 64,
    getMeasure(n: string) {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, subCtx);
      cache.set(n, m);
      return m;
    },
  };
  for (const p of d.paramKwargs) {
    const f = posteriorMeasure.fields[p];
    if (!f || !f.samples) return null;
    cache.set(p, Promise.resolve(scalarMeasureN(f.samples, { logWeights: null })));
  }

  const fields: Record<string, PPCField> = {};

  for (const spec of fieldSpecs) {
    // Materialise the forward variate with the posterior latents bound. A field
    // that is a self-ref resolves through the (pre-seeded) getMeasure cache; an
    // inline measure IR is materialised directly.
    let fieldMeasure: any;
    try {
      fieldMeasure = (spec.ir && spec.ir.kind === 'ref' && spec.ir.ns === 'self')
        ? await subCtx.getMeasure(spec.ir.name)
        : await materialiser.materialiseMeasureIR(spec.ir, subCtx);
    } catch (_) {
      return null;
    }
    if (!fieldMeasure || !(fieldMeasure.samples instanceof Float64Array)) return null;
    const yrSamples: Float64Array = fieldMeasure.samples;

    // Re-attach the posterior weights (leaf-sample materialisation drops them).
    // Each posterior atom i expands to `perAtom` predictive slots (vector
    // variate and/or iid replicates), so broadcast atom-major.
    const total = yrSamples.length;
    const perAtom = (total > 0 && count > 0) ? Math.max(1, Math.round(total / count)) : 1;
    let logWeights: Float64Array | null = null;
    if (postLW) {
      logWeights = new Float64Array(total);
      for (let i = 0; i < count; i++) {
        const w = postLW[i];
        const base = i * perAtom;
        for (let r = 0; r < perAtom; r++) {
          const idx = base + r;
          if (idx < total) logWeights[idx] = w;
        }
      }
    }

    fields[spec.name] = {
      yRep: { samples: yrSamples, logWeights },
      observed: resolveObserved(obs, spec.name),
    };
  }

  return { fields };
}

module.exports = { buildPosteriorPredictive };
