'use strict';

// @flatppl/engine — posterior-predictive sampling for bayesupdate likelihoods.
//
// For a classified `bayesupdate` derivation whose likelihood kernel body is
// `lawof(record(field = iid(Dist, n), ...))`, builds a PPC by feeding each
// posterior parameter draw-column through the worker's `sampleN` primitive
// to produce replicated observations `y_rep` at each posterior atom.
//
// Composite/transform inner distributions are handled as follows:
//   - `locscale(M, shift, scale)` — desugared by the analyzer to
//     `pushfwd(bijection(fwd, inv, lv), M)`. Handled explicitly (Task 2):
//     sample the leaf base M with `sampleN`, then apply the affine forward
//     function via `evaluateN`. The two-worker-call approach lets the worker
//     handle the base distribution with its own primitives (leafs, randNFn)
//     and apply the transform element-wise over the resulting sample column.
//   - Other pushfwd shapes (non-locscale bijections) → whole build returns null.
//   - truncate, superpose, nested iid, etc. → whole build returns null.
//
// Not yet generalised (documented, not silently ignored):
//   - pushfwd with a non-locscale bijection (e.g. exp/log, user bijection)
//   - truncate inner distributions
//   - superpose / select / nested iid
// These fields cause the whole build to return null with no throw.

const orchestrator = require('./orchestrator.ts');
const { SAMPLEABLE_DISTRIBUTIONS } = require('./ir-shared.ts');

// ── Types ───────────────────────────────────────────────────────────────────

export interface YRepField {
  samples: Float64Array;
  logWeights: Float64Array | null;
}

export interface PPCField {
  yRep: YRepField;
  observed: number[];
}

export interface PPC {
  fields: Record<string, PPCField>;
}

// ── Local IR walk ─────────────────────────────────────────────────────────

// Collect ns:'self' ref names from an IR node.
// Mirrors the selfRefs walk in packages/viewer/src/generated-quantities.ts
// (covers: callee, args, kwargs, fields[*].value, body).
function selfRefs(ir: any, acc: Set<string>): Set<string> {
  if (!ir || typeof ir !== 'object') return acc;
  if (ir.kind === 'ref' && ir.ns === 'self' && ir.name) acc.add(ir.name);
  if (ir.callee) selfRefs(ir.callee, acc);
  if (Array.isArray(ir.args)) ir.args.forEach((a: any) => selfRefs(a, acc));
  if (ir.kwargs) for (const k in ir.kwargs) selfRefs(ir.kwargs[k], acc);
  if (Array.isArray(ir.fields)) {
    ir.fields.forEach((f: any) => { if (f && f.value) selfRefs(f.value, acc); });
  }
  if (Array.isArray(ir.assigns)) {
    ir.assigns.forEach((a: any) => { if (a && a.value) selfRefs(a.value, acc); });
  }
  if (ir.body) selfRefs(ir.body, acc);
  if (Array.isArray(ir.branches)) {
    ir.branches.forEach((b: any) => {
      if (!b || typeof b !== 'object') return;
      if (b.ir) selfRefs(b.ir, acc);
      else if (b.kind) selfRefs(b, acc);
    });
  }
  if (ir.selector) selfRefs(ir.selector, acc);
  if (Array.isArray(ir.logweights)) {
    ir.logweights.forEach((w: any) => { if (w) selfRefs(w, acc); });
  }
  return acc;
}

// ── IR ref resolution ─────────────────────────────────────────────────────

// Resolve a self-ref (or a `draw(ref(...))` wrapper) through the bindings
// map down to a concrete measure/dist IR. The intent: for a field like
// `ref('y')` where `y` is a `draw` binding with IR `draw(ref('__anon3'))`,
// and `__anon3` is `iid(ref('__anon2'), 5)`, we want to land on
// `iid(ref('__anon2'), 5)` — the actual measure structure.
//
// Resolution rules (in order):
//  1. A `ref(ns:'self', name)` → look up binding.ir, continue from there.
//  2. A `draw(ref(...))` → peel the draw, continue with the inner ref.
//  3. Any other call IR (iid, Normal, etc.) → return as-is.
//  4. Cycle or missing binding → return null.
function resolveToMeasureIR(
  ir: any,
  bindings: any,
  seen: Set<string>,
): any {
  if (!ir || typeof ir !== 'object') return null;

  // Rule 2: peel draw(ref(...)) — a draw-binding's IR form
  if (ir.kind === 'call' && ir.op === 'draw'
      && Array.isArray(ir.args) && ir.args.length >= 1) {
    return resolveToMeasureIR(ir.args[0], bindings, seen);
  }

  // Rule 1: chase a self-ref to its binding's IR
  if (ir.kind === 'ref' && ir.ns === 'self') {
    const name: string = ir.name;
    if (seen.has(name)) return null; // cycle guard
    seen.add(name);
    const binding = bindings && bindings.get && bindings.get(name);
    if (!binding || !binding.ir) return null;
    return resolveToMeasureIR(binding.ir, bindings, seen);
  }

  // Rule 3: a concrete IR node — return as-is
  return ir;
}

// ── Main export ────────────────────────────────────────────────────────────

/**
 * Build a posterior-predictive check for a bayesupdate derivation whose
 * likelihood kernel body is `lawof(record(...))` with iid fields.
 *
 * Supported inner distribution shapes:
 *  - Any leaf in SAMPLEABLE_DISTRIBUTIONS — sampled directly via `sampleN`.
 *  - pushfwd(bijection, leaf) — the locscale expansion; sampled in two steps:
 *    `sampleN` the leaf base, then `evaluateN` the bijection forward body.
 *
 * Returns null (never throws) when:
 *  - `d` is not a bayesupdate or lacks required fields
 *  - `d.paramKwargs` is empty
 *  - the body is not a `lawof(record(...))` shape
 *  - any field's inner distribution is a non-sampleable composite
 *    (truncate, superpose, nested iid, non-leaf-base pushfwd, …)
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

  // Resolve the body IR. When bodyIR is available use it directly; when only
  // bodyName is available, resolve it via the orchestrator (expandMeasureIR).
  let bodyIR: any = d.bodyIR;
  if (!bodyIR && d.bodyName) {
    try {
      bodyIR = orchestrator.expandMeasureIR(d.bodyName, ctx.derivations, ctx.bindings);
    } catch (_) {
      return null;
    }
  }
  if (!bodyIR) return null;

  // The body must be `lawof(record(...))`.
  // Shape: call { op:'lawof', args:[call { op:'record', fields:[{name,value}] }] }
  if (bodyIR.kind !== 'call' || bodyIR.op !== 'lawof') return null;
  if (!Array.isArray(bodyIR.args) || bodyIR.args.length < 1) return null;
  const recordIR = bodyIR.args[0];
  if (!recordIR || recordIR.kind !== 'call' || recordIR.op !== 'record') return null;
  if (!Array.isArray(recordIR.fields) || recordIR.fields.length === 0) return null;

  // Resolve the observed record so we can extract per-field observed values.
  let obs: any = null;
  if (d.obsIR) {
    try {
      obs = orchestrator.resolveIRToValue(d.obsIR, ctx.bindings, ctx.fixedValues);
    } catch (_) {
      obs = null;
    }
  }

  // Posterior atom count: all field columns share the same length.
  if (!posteriorMeasure || !posteriorMeasure.fields
      || !posteriorMeasure.fields[d.paramKwargs[0]]
      || !posteriorMeasure.fields[d.paramKwargs[0]].samples) return null;
  const count: number = posteriorMeasure.fields[d.paramKwargs[0]].samples.length;

  const fields: Record<string, PPCField> = {};

  for (let fieldIndex = 0; fieldIndex < recordIR.fields.length; fieldIndex++) {
    const fieldEntry = recordIR.fields[fieldIndex];
    const name: string = fieldEntry.name;
    const mIR: any = fieldEntry.value; // the field's measure IR

    // Resolve the field's measure IR through any draw/alias/ref chain.
    // E.g. `ref('y')` → binding IR `draw(ref('__anon3'))` → `iid(ref('__anon2'), 5)`
    const resolvedMIR: any = resolveToMeasureIR(mIR, ctx.bindings, new Set<string>());

    // Determine innerDist and repeat count.
    // Shape A: iid(innerDist, n)  — where innerDist may itself be a ref
    // Shape B: bare dist (treat as repeat=1)
    let innerDist: any;
    let repeat: number;

    if (resolvedMIR && resolvedMIR.kind === 'call' && resolvedMIR.op === 'iid'
        && Array.isArray(resolvedMIR.args) && resolvedMIR.args.length >= 2) {
      // Resolve the inner dist — it may be a ref to a named dist binding.
      const rawInnerDistIR = resolvedMIR.args[0];
      const resolvedInner = resolveToMeasureIR(rawInnerDistIR, ctx.bindings, new Set<string>());
      innerDist = resolvedInner;
      if (!innerDist) return null;
      // Resolve repeat count
      try {
        const repeatVal = orchestrator.resolveIRToValue(
          resolvedMIR.args[1], ctx.bindings, ctx.fixedValues,
        );
        repeat = typeof repeatVal === 'number' ? (repeatVal | 0) : 1;
      } catch (_) {
        repeat = 1;
      }
    } else {
      // Bare dist: treat as repeat=1.
      innerDist = resolvedMIR;
      repeat = 1;
    }

    // Guard: innerDist must be sampleable — either a leaf distribution or
    // a pushfwd over a leaf (the locscale case). Other composites
    // (truncate, superpose, nested iid, …) return null for the whole build.
    if (!innerDist || innerDist.kind !== 'call' || !innerDist.op) return null;

    // seed is (rootKey XOR fieldIndex) as uint32.
    const seed = (((ctx.rootKey | 0) ^ fieldIndex) >>> 0);

    let yrSamples: Float64Array;

    if (SAMPLEABLE_DISTRIBUTIONS.has(innerDist.op)) {
      // ── Leaf distribution path (Task 1) ────────────────────────────────
      // Build refArrays: include only params that appear as self-refs in innerDist.
      const distSelfRefs = selfRefs(innerDist, new Set<string>());
      const refArrays: Record<string, Float64Array> = {};
      for (const p of d.paramKwargs) {
        if (distSelfRefs.has(p) && posteriorMeasure.fields[p]) {
          refArrays[p] = posteriorMeasure.fields[p].samples;
        }
      }

      const reply = await ctx.sendWorker({
        type: 'sampleN',
        ir: innerDist,
        count,
        repeat,
        refArrays,
        seed,
      });
      yrSamples = reply.samples;

    } else if (innerDist.op === 'pushfwd') {
      // ── Pushfwd path (Task 2: locscale and similar affine transforms) ──
      //
      // locscale(M, shift, scale) is desugared by the analyzer to
      // pushfwd(bijection(fwd, inv, lv), M) before derivation classification.
      // M is a leaf distribution (e.g. StudentT(nu)).
      //
      // Strategy: sample from the base distribution M with `sampleN`,
      // then apply the bijection's forward function body via `evaluateN`.
      // The two-call approach lets the worker use its leaf-distribution
      // primitives (randNFn fast paths, etc.) for the base, and its
      // arithmetic evaluator for the transform.
      //
      // Args layout: pushfwd(bijRef, baseDistIR)
      if (!Array.isArray(innerDist.args) || innerDist.args.length < 2) return null;
      const bijRefIR: any = innerDist.args[0]; // ref to the bijection binding
      const rawBaseIR: any = innerDist.args[1]; // base dist IR (may be a ref)

      // Resolve the base dist through any ref chain.
      const baseDist: any = resolveToMeasureIR(rawBaseIR, ctx.bindings, new Set<string>());
      if (!baseDist || baseDist.kind !== 'call' || !baseDist.op) return null;
      if (!SAMPLEABLE_DISTRIBUTIONS.has(baseDist.op)) return null;

      // Resolve the bijection binding to get the forward function body.
      // The bijection binding carries `{ bijection: { fName, fInvName, logVolume } }`.
      let bijBindingName: string | null = null;
      if (bijRefIR.kind === 'ref' && bijRefIR.ns === 'self') {
        bijBindingName = bijRefIR.name;
      } else {
        // Non-self-ref bijection arg — not a locscale synthetic pattern; decline.
        return null;
      }
      const bijBinding = ctx.bindings && ctx.bindings.get && ctx.bindings.get(bijBindingName);
      if (!bijBinding || !bijBinding.bijection || !bijBinding.bijection.fName) return null;

      const fwdBinding = ctx.bindings.get(bijBinding.bijection.fName);
      if (!fwdBinding || !fwdBinding.ir) return null;
      const fwdIR = fwdBinding.ir;
      if (fwdIR.kind !== 'call' || fwdIR.op !== 'functionof'
          || !Array.isArray(fwdIR.params) || !fwdIR.body) return null;
      const fwdParamName: string = fwdIR.params[0]; // e.g. 'x'
      const fwdBody: any = fwdIR.body;

      // Step 1: sample from the base distribution.
      // refArrays for the base dist: self-refs in baseDist that are posterior params.
      const baseSelfRefs = selfRefs(baseDist, new Set<string>());
      const baseRefArrays: Record<string, Float64Array> = {};
      for (const p of d.paramKwargs) {
        if (baseSelfRefs.has(p) && posteriorMeasure.fields[p]) {
          baseRefArrays[p] = posteriorMeasure.fields[p].samples;
        }
      }
      const baseReply = await ctx.sendWorker({
        type: 'sampleN',
        ir: baseDist,
        count,
        repeat,
        refArrays: baseRefArrays,
        seed,
      });
      const baseZ: Float64Array = baseReply.samples; // length: count * repeat

      // Step 2: apply the forward transform via evaluateN.
      // Collect self-refs in the forward body (excluding the function param).
      const fwdBodyRefs = selfRefs(fwdBody, new Set<string>());
      fwdBodyRefs.delete(fwdParamName); // param is not a self-ref, but guard anyway

      // Build refArrays for the forward body:
      //   - fwdParamName → base samples (already count*repeat length)
      //   - other self-refs (e.g. mu1, sigma1) → posterior columns tiled
      //     atom-major to count*repeat length (atom i's value is shared across
      //     its `repeat` inner draws, matching the base sample layout).
      const fwdRefArrays: Record<string, any> = {};
      fwdRefArrays[fwdParamName] = baseZ;
      for (const p of d.paramKwargs) {
        if (fwdBodyRefs.has(p) && posteriorMeasure.fields[p]) {
          const col: Float64Array = posteriorMeasure.fields[p].samples;
          if (repeat === 1) {
            fwdRefArrays[p] = col;
          } else {
            // Tile atom-major: atom i's posterior value is shared across
            // its `repeat` base-sample positions [i*repeat … (i+1)*repeat).
            const tiled = new Float64Array(count * repeat);
            for (let i = 0; i < count; i++) {
              const v = col[i];
              const base = i * repeat;
              for (let r = 0; r < repeat; r++) tiled[base + r] = v;
            }
            fwdRefArrays[p] = tiled;
          }
        }
      }

      const fwdReply = await ctx.sendWorker({
        type: 'evaluateN',
        ir: fwdBody,
        count: count * repeat,
        refArrays: fwdRefArrays,
      });
      yrSamples = fwdReply.samples || fwdReply.out;

    } else {
      // Composite inner distribution not yet supported (truncate, superpose, …).
      // Return null for the whole build.
      return null;
    }

    // Build logWeights for y_rep: broadcast posteriorMeasure.logWeights[i]
    // across each atom's `repeat` block.
    let logWeights: Float64Array | null;
    if (posteriorMeasure.logWeights == null) {
      logWeights = null;
    } else {
      const pw: Float64Array = posteriorMeasure.logWeights;
      logWeights = new Float64Array(count * repeat);
      for (let i = 0; i < count; i++) {
        const w = pw[i];
        for (let r = 0; r < repeat; r++) {
          logWeights[i * repeat + r] = w;
        }
      }
    }

    // Resolve observed values for this field.
    let observed: number[];
    try {
      if (obs && typeof obs === 'object' && !Array.isArray(obs) && name in obs) {
        const raw = obs[name];
        observed = Array.isArray(raw) ? raw.map(Number) : [Number(raw)];
      } else if (obs !== null && obs !== undefined) {
        observed = Array.isArray(obs) ? obs.map(Number) : [Number(obs)];
      } else {
        observed = [];
      }
    } catch (_) {
      observed = [];
    }

    fields[name] = { yRep: { samples: yrSamples, logWeights }, observed };
  }

  return { fields };
}

module.exports = { buildPosteriorPredictive };
