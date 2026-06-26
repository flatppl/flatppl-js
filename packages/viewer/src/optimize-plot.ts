// @flatppl/viewer — "Find maximum" action for function / likelihood plots.
//
// Drives the engine optimizer (FlatPPLEngine.optimizer, CMA-ES + FD polish)
// over a profile plot's FREE scalar inputs (the elementof leaves the plot is
// parameterised by, minus any pinned with `fixed(...)`), starting from the
// current pivot. The argmax is written back as the modified pivot via the
// override store (exactly like a manual slice edit), so Save/Discard work
// unchanged.
//
// The objective is evaluated in a single batched worker call per generation:
// the optimizer proposes a cloud of K candidate points; we feed each free
// input as a length-K refArray column and post evaluateN (function plots) or
// logDensityN (density/likelihood plots — per-θ params via refArrays scored
// against the shared observation, the normal likelihood mode). This reuses
// render-profile's IR setup (CLM lowerMeasure + substituteBoundaryValues),
// generalised from one swept axis to all free axes.
//
// Scope (v1, TODO §"Plot optimizer"): closed-form function + density targets,
// scalar real / interval / posreals / nonnegreals inputs. Marginalised
// (Monte-Carlo) targets and array/per-slot inputs are the next steps.

import type { Ctx } from './types';
import { tryGetMeasure } from './engine-facade.js';
import {
  activeDomainRangesFor, activeFixedNamesFor, activeInputDomain,
  activeInputValues, ensureOverrideFor, setOverrideFor,
} from './overrides.js';
import { showPlotMessage } from './render-frame.js';
import { defaultRangeForLeafType, esc, formatScalar } from './util.js';
import { sendWorker } from './worker.js';

/** Map an engine SetDescriptor to an optimizer Domain. */
function domainFromDescriptor(desc: any): any {
  const kind = desc && desc.kind;
  if (kind === 'interval') return { kind: 'interval', lo: +desc.lo, hi: +desc.hi };
  if (kind === 'unitinterval') return { kind: 'interval', lo: 0, hi: 1 };
  if (kind === 'posreals') return { kind: 'posreals' };
  if (kind === 'nonnegreals') return { kind: 'nonnegreals' };
  // reals / integers / unknown → unconstrained (integers handled by rounding).
  return { kind: 'real' };
}

/** Result of computeMaximum: the fit plus the per-kwarg argmax (rounded for
 *  integer axes), or a refusal carrying a user-facing message. */
type MaxResult =
  | { ok: true; fit: any;
      free: Array<{ kwarg: string; param: string; axisKey: string; slot: number; isInt: boolean }>;
      modeValues: Record<string, number | number[]>;
      modeSds: Record<string, number> }
  | { ok: false; message: string; hint?: boolean };

/**
 * Optimise a profile plan's free scalar inputs from the current pivot and
 * return the argmax (CMA-ES + FD polish) — WITHOUT touching the UI or the
 * override store. Shared by two callers:
 *   - runFindMaximum (the ⛰ button) writes the result as the modified pivot;
 *   - populateModeCache caches it as the labelled `auto (MLE)` default point.
 * Refusals (no free inputs / un-lowerable / marginalised) come back as
 * { ok:false, message } so each caller decides whether to surface them.
 */
export async function computeMaximum(ctx: Ctx, plan: any): Promise<MaxResult> {
  const sig = plan.signature;
  const mode = sig.kind === 'function' ? 'function' : 'logdensity';
  const dst = ctx.derivationsState!;

  // --- 1. Free scalar inputs (minus fixed(...)-pinned) -----------------
  const fixedNames = activeFixedNamesFor(ctx, plan);
  const vals = activeInputValues(ctx, plan);
  const doms = activeInputDomain(ctx, plan);
  const ranges = activeDomainRangesFor(ctx, plan);

  // One optimiser dimension per free SCALAR LEAF: a scalar input is one dim;
  // a 1-D array input of scalars (e.g. `theta` in a hierarchical model) is one
  // dim per slot. `slot = -1` marks a scalar input; `slot = j` an array slot.
  // `paramArrayLen` records each array param's full length so evalCloud can
  // rebuild the per-atom vector (held slots — fixed or non-finite — keep the
  // pivot value). axisKey matches distributeAxes' key (`theta[3]`) so the
  // per-axis mode-cache curvature lookup lines up with the plot's sweep axis.
  const free: Array<{ kwarg: string; param: string; axisKey: string; slot: number; isInt: boolean }> = [];
  const x0: number[] = [];
  const domains: any[] = [];
  const scales: number[] = [];
  const paramArrayLen: Record<string, number> = {};
  const scaleFor = (kwarg: string, leafType: any) => {
    const r = ranges[kwarg];
    let s = r && Number.isFinite(r.hi - r.lo) ? Math.abs(r.hi - r.lo) : 0;
    if (!(s > 0)) { const dr = defaultRangeForLeafType(leafType); s = dr ? Math.abs(dr[1] - dr[0]) : 0; }
    return s;
  };
  for (let i = 0; i < sig.inputs.length; i++) {
    const inp = sig.inputs[i];
    if (fixedNames && fixedNames.has(inp.kwargName)) continue;   // whole input fixed(...)
    const t = inp.type;
    if (t && t.kind === 'scalar') {
      const v = vals[inp.kwargName];
      if (typeof v !== 'number' || !Number.isFinite(v)) continue;
      free.push({ kwarg: inp.kwargName, param: inp.paramName, axisKey: inp.kwargName, slot: -1,
        isInt: !!(t.prim === 'integer') });
      x0.push(v);
      domains.push(domainFromDescriptor(doms[inp.kwargName]));
      scales.push(scaleFor(inp.kwargName, t));
    } else if (t && t.kind === 'array' && t.elem && t.elem.kind === 'scalar'
               && Array.isArray(t.shape) && t.shape.length === 1) {
      const J = t.shape[0] | 0;
      const arr = FlatPPLEngine.orchestrator.valueToPlain(vals[inp.kwargName]);
      if (!Array.isArray(arr)) continue;
      paramArrayLen[inp.paramName] = J;
      const leafType = { kind: 'scalar', prim: t.elem.prim };
      const dom = domainFromDescriptor(doms[inp.kwargName]);
      const sc = scaleFor(inp.kwargName, leafType);
      for (let j = 0; j < J; j++) {
        const vj = arr[j];
        if (typeof vj !== 'number' || !Number.isFinite(vj)) continue;   // held at pivot
        free.push({ kwarg: inp.kwargName, param: inp.paramName, axisKey: inp.kwargName + '[' + (j + 1) + ']',
          slot: j, isInt: !!(t.elem.prim === 'integer') });
        x0.push(vj);
        domains.push(dom);
        scales.push(sc);
      }
    }
    // record / multi-dim array inputs: not yet optimised (held at the pivot).
  }

  if (free.length === 0) {
    return { ok: false, hint: true, message: 'Find maximum: no free scalar inputs '
      + 'to optimize (all inputs are <code>fixed(...)</code> or non-scalar).' };
  }

  // --- 2. Objective IR (CLM lowering; free inputs left unbaked) ---------
  let ir = sig.body;
  if (plan.outputs && plan.outputs.length > 1 && plan.outputKey) {
    for (let oj = 0; oj < plan.outputs.length; oj++) {
      if (plan.outputs[oj].key === plan.outputKey) {
        const ex = FlatPPLEngine.orchestrator.extractOutputIR(ir, plan.outputs[oj].path);
        if (ex) ir = ex;
        break;
      }
    }
  }
  const paramNames = sig.inputs.map((inp: any) => inp.paramName);
  const freeParams = free.map((f) => f.param);
  const freeParamSet = new Set(freeParams);
  const boundaries: Record<string, any> = {};
  for (const pn of paramNames) if (!freeParamSet.has(pn)) boundaries[pn] = true;

  let clmNode: any = null, clmErr: any = null;
  try {
    clmNode = FlatPPLEngine.clm.lowerMeasure(ir,
      { derivations: dst.derivations, bindings: dst.bindings, fixedValues: dst.fixedValues },
      { boundaries, freeInputs: freeParams });
  } catch (e) { clmErr = e; }
  if (!clmNode) {
    return { ok: false, message: 'Find maximum: cannot lower <strong>' + esc(plan.name)
      + '</strong> to a canonical measure'
      + (clmErr ? ' — ' + esc(clmErr.message || String(clmErr)) : '') + '.' };
  }
  ir = clmNode.body;

  // Marginalised (Monte-Carlo) targets: deferred to the next step (the K×M
  // density batch). Fail gracefully rather than silently mis-optimising.
  if (clmNode.mc) {
    return { ok: false, hint: true, message: 'Find maximum: this target is internally '
      + 'marginalised (Monte-Carlo). Maximisation of marginalised targets lands in the '
      + 'next step; closed-form functions and likelihoods work today.' };
  }

  // --- 3. Bake boundary (non-free) values + body self-refs --------------
  const fixedEnv: Record<string, any> = {};
  for (let i = 0; i < sig.inputs.length; i++) {
    const inp = sig.inputs[i];
    if (freeParamSet.has(inp.paramName)) continue;
    if (Object.prototype.hasOwnProperty.call(vals, inp.kwargName)) {
      fixedEnv[inp.paramName] = vals[inp.kwargName];
    }
  }
  // Random-phase parents referenced in the body → bake samples[0] (mirrors
  // render-profile), so a stochastic NUISANCE parent is held at one draw while
  // we optimise. CRITICAL: exclude the FREE inputs. classifyProfileSelfRefs
  // returns every stochastic self-ref in the body — including the very params
  // we are optimising (they have priors, so they look random-phase). Baking
  // those to samples[0] replaces the free refs with a constant prior draw (the
  // pivot); refArrays then has nothing to drive and the objective is FLAT at
  // the pivot → the optimizer returns its incumbent. The free inputs must stay
  // as live refs, fed per-atom by refArrays in evalCloud.
  const { perAtomNames: allSelfRefs } =
    FlatPPLEngine.materialiser.classifyProfileSelfRefs(ir, dst.bindings, dst.fixedValues);
  const selfRefs = allSelfRefs.filter((n: string) => !freeParamSet.has(n));
  const selfMeasures = await Promise.all(
    selfRefs.map((n: string) => tryGetMeasure(ctx, n)));
  for (let i = 0; i < selfRefs.length; i++) {
    const m = selfMeasures[i];
    if (m && m.samples && m.samples.length > 0) fixedEnv[selfRefs[i]] = m.samples[0];
  }
  // Fixed-phase self-refs in the body (observed data, literal arrays, fixed
  // reductions) are baked in too, so the objective IR is fully self-contained
  // and does not depend on the worker's session env surviving the optimiser's
  // many round-trips. (The free inputs above are deliberately NOT baked — they
  // are the search variables, fed per-atom by refArrays in evalCloud.)
  const fvMap = dst.fixedValues;
  if (fvMap) {
    for (const n of FlatPPLEngine.orchestrator.collectSelfRefs(ir)) {
      if (fvMap.has(n) && !Object.prototype.hasOwnProperty.call(fixedEnv, n)) {
        // valueToPlain: fixedValues holds engine Value objects ({shape, data});
        // substituteBoundaryValues → envValueToIR needs plain arrays/scalars, or
        // it mis-bakes an array Value as a record(shape, data) and downstream
        // broadcast/dot-call rejects it.
        fixedEnv[n] = FlatPPLEngine.orchestrator.valueToPlain(fvMap.get(n));
      }
    }
  }
  // Build a per-term list of (irBaked, observed). For a joint_likelihood sig
  // (sig.terms present) each term's body is lowered and baked independently;
  // the CLM-lowered `ir` above is NOT used (it was clmNode.body from a single
  // composite CLM node). For a single likelihood (sig.terms absent) we wrap the
  // existing `ir` + `sig.obsIR` into a one-element list so evalCloud is uniform.
  // The shared fixedEnv (non-free boundary values + self-refs) is valid for
  // every term: baking the full fixedEnv over a term that doesn't reference a
  // name is a no-op.
  const evalTerms: Array<{ irBaked: any; observed: any }> = (
    sig.terms
      ? (sig.terms as Array<{ body: any; obsIR: any }>)
      : [{ body: ir, obsIR: sig.obsIR }]
  ).map((term) => {
    const tBaked = FlatPPLEngine.orchestrator.substituteBoundaryValues(term.body, fixedEnv);
    const tObs = term.obsIR != null
      ? FlatPPLEngine.orchestrator.resolveIRToValue(term.obsIR, dst.bindings, dst.fixedValues)
      : undefined;
    return { irBaked: tBaked, observed: tObs };
  });

  // Pivot vectors for array params (held slots that aren't free dims keep these
  // values per atom). Plain JS arrays via valueToPlain.
  const pivotVec: Record<string, number[]> = {};
  for (const p in paramArrayLen) {
    const kw = free.find((f) => f.param === p)!.kwarg;
    const pv = FlatPPLEngine.orchestrator.valueToPlain(vals[kw]);
    if (Array.isArray(pv)) pivotVec[p] = pv;
  }

  // --- 4. Batched cloud objective over the worker ----------------------
  const evalCloud = async (cloud: number[][]): Promise<number[]> => {
    const K = cloud.length;
    // Group free dims by param. A scalar param → one Float64Array(K) refArray;
    // an array param → a K×J Value, each atom's vector assembled from its free
    // slots (held slots from the pivot). The density's per-atom value evaluator
    // drives the param ref from these per-atom entries (walkLeaf), so the whole
    // vector varies across the cloud — exactly what the multi-dim optimiser needs.
    const refArrays: Record<string, any> = {};
    for (const p in paramArrayLen) {
      const J = paramArrayLen[p];
      const data = new Float64Array(K * J);
      const pv = pivotVec[p];
      if (pv) for (let k = 0; k < K; k++) for (let j = 0; j < J; j++) data[k * J + j] = pv[j];
      refArrays[p] = { shape: [K, J], data };
    }
    for (let d = 0; d < free.length; d++) {
      const f = free[d];
      if (f.slot < 0) {
        const col = new Float64Array(K);
        for (let k = 0; k < K; k++) col[k] = cloud[k][d];
        refArrays[f.param] = col;
      } else {
        const J = paramArrayLen[f.param];
        const data = (refArrays[f.param] as any).data as Float64Array;
        for (let k = 0; k < K; k++) data[k * J + f.slot] = cloud[k][d];
      }
    }
    // Sum log-densities over all terms. For a single likelihood (evalTerms has
    // one entry) this is identical to the former single logDensityN call.
    // For a joint_likelihood (multiple terms) the per-atom results are summed;
    // any term that is out-of-support (non-finite) propagates −Infinity as the
    // worst-case value, matching the single-term behaviour.
    const res = new Array(K).fill(0);
    for (const t of evalTerms) {
      const msg: any = mode === 'function'
        ? { type: 'evaluateN', ir: t.irBaked, refArrays, count: K }
        : { type: 'logDensityN', ir: t.irBaked, observed: t.observed, refArrays, count: K };
      const reply: any = await sendWorker(ctx, msg);
      const out = reply && reply.samples;
      for (let k = 0; k < K; k++) {
        const v = out ? out[k] : NaN;
        res[k] += Number.isFinite(v) ? v : -Infinity; // out-of-support → worst
      }
    }
    return res;
  };

  // --- 5. Optimize ------------------------------------------------------
  // No setEnv push: every fixed value the objective needs is baked into
  // irBaked above, so the worker's session env (which a concurrent plot
  // rebuild may reset mid-run) is irrelevant to this objective.
  const fit = await FlatPPLEngine.optimizer.optimize({
    evalCloud, x0, domains, scales,
    opts: { seed: 0xf1a7, starts: free.length <= 4 ? 3 : 2 },
  });

  // modeValues keyed by KWARG (scalar → number, array → number[] starting from
  // the pivot vector so any held slot keeps its value). modeSds keyed by AXIS
  // KEY (`mu`, `theta[3]`) — the per-axis curvature width the plot uses to frame
  // each sweep axis.
  const modeValues: Record<string, any> = {};
  const modeSds: Record<string, number> = {};
  for (const p in paramArrayLen) {
    const kw = free.find((f) => f.param === p)!.kwarg;
    modeValues[kw] = (pivotVec[p] || new Array(paramArrayLen[p]).fill(0)).slice();
  }
  for (let d = 0; d < free.length; d++) {
    const f = free[d];
    const v = f.isInt ? Math.round(fit.mode[d]) : fit.mode[d];
    if (f.slot < 0) modeValues[f.kwarg] = v;
    else (modeValues[f.kwarg] as number[])[f.slot] = v;
    if (fit.sd && Number.isFinite(fit.sd[d]) && fit.sd[d] > 0) modeSds[f.axisKey] = fit.sd[d];
  }
  return { ok: true, fit, free, modeValues, modeSds };
}

/**
 * The ⛰ "find maximum" button: optimise the free inputs and write the argmax
 * as the modified pivot (Save to keep), then `rerender`. Refusals / errors
 * surface as a plot message.
 */
export async function runFindMaximum(ctx: Ctx, plan: any, rerender: () => void): Promise<void> {
  showPlotMessage(ctx, 'Finding maximum…', { hint: true });
  const r = await computeMaximum(ctx, plan);
  if (!r.ok) {
    showPlotMessage(ctx, r.message, r.hint ? { hint: true } : undefined);
    return;
  }
  const entry = ensureOverrideFor(ctx, plan);
  entry.values = Object.assign({}, entry.values || {}, r.modeValues);
  setOverrideFor(ctx, plan, entry);

  const mode = plan.signature.kind === 'function' ? 'function' : 'logdensity';
  const where = r.free.map((f) => {
    const mv = r.modeValues[f.kwarg];
    const val = f.slot < 0 ? mv : (Array.isArray(mv) ? mv[f.slot] : mv);
    return esc(f.axisKey) + ' = ' + formatScalar(val as number);
  }).join(', ');
  const noteBoundary = r.fit.boundaryActive && r.fit.boundaryActive.some((b: boolean) => b)
    ? ' (on a boundary)' : '';
  rerender();
  showPlotMessage(ctx, 'Maximum: ' + (mode === 'function' ? 'f' : 'log p') + ' = '
    + formatScalar(r.fit.value) + ' at ' + where + noteBoundary
    + ' — ' + r.fit.nBatches + ' batches. Save to keep.', { hint: true });
}

// Best-effort budget for the background MLE so a slow / pathological target
// can't stall the labelled `auto (MLE)` default — on timeout we just fall back
// to the normal prior-draw `auto` pivot.
const MLE_TIMEOUT_MS = 4000;

/**
 * Populate `ctx.modeCenterCache[plan.name]` with the MLE point for a
 * likelihood plot (obsIR-bearing), so buildPresetControl can offer a labelled
 * `auto (MLE)` default. Best-effort and fire-and-forget: it runs AFTER the
 * plot has been posted (the worker is FIFO, so the visible sweep goes first),
 * races a timeout, and on failure OR timeout marks the entry 'failed' so the
 * option simply never appears — the normal `auto` pivot remains. Idempotent
 * per binding (the cache is cleared on every source rebuild).
 */
export function populateModeCache(ctx: Ctx, plan: any, rerender: () => void): void {
  const sig = plan && plan.signature;
  if (!sig || (sig.obsIR == null && !sig.terms)) return;  // likelihoods: single obsIR, or joint terms
  if (!ctx.modeCenterCache) ctx.modeCenterCache = new Map();
  const name = plan.name;
  if (ctx.modeCenterCache.has(name)) return;        // pending / ready / failed already
  ctx.modeCenterCache.set(name, { status: 'pending' });

  let timer: any = null;
  const timeout = new Promise<{ kind: 'timeout' }>(function(res) {
    timer = setTimeout(function() { res({ kind: 'timeout' }); }, MLE_TIMEOUT_MS);
  });
  const work = computeMaximum(ctx, plan)
    .then(function(r) { return { kind: 'done' as const, r: r }; })
    .catch(function(e) { return { kind: 'error' as const, e: e }; });

  Promise.race([work, timeout]).then(function(outcome: any) {
    if (timer) clearTimeout(timer);
    const entry = ctx.modeCenterCache!.get(name);
    if (!entry || entry.status !== 'pending') return;   // superseded by a rebuild
    if (outcome.kind === 'done' && outcome.r.ok) {
      ctx.modeCenterCache!.set(name,
        { status: 'ready', values: outcome.r.modeValues, sd: outcome.r.modeSds });
      rerender();                                        // redraw so the row appears
    } else {
      const reason = outcome.kind === 'timeout' ? 'timeout'
        : outcome.kind === 'error' ? 'error'
        : (outcome.r && outcome.r.message) || 'refused';
      ctx.modeCenterCache!.set(name, { status: 'failed', reason: reason });
      // No rerender: nothing to show — the normal prior-draw `auto` stays.
    }
  }).catch(function() {
    // Best-effort: a throwing rerender must not surface as an unhandled
    // rejection. The cache entry is already settled; the normal auto pivot
    // is always available regardless.
  });
}
