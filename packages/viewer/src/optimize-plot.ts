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
      free: Array<{ kwarg: string; param: string; isInt: boolean }>;
      modeValues: Record<string, number> }
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

  const free: Array<{ kwarg: string; param: string; isInt: boolean }> = [];
  const x0: number[] = [];
  const domains: any[] = [];
  const scales: number[] = [];
  for (let i = 0; i < sig.inputs.length; i++) {
    const inp = sig.inputs[i];
    const isScalar = inp.type && inp.type.kind === 'scalar';
    if (!isScalar) continue;                       // array/record inputs: held fixed (v1)
    if (fixedNames && fixedNames.has(inp.kwargName)) continue;
    const v = vals[inp.kwargName];
    if (typeof v !== 'number' || !Number.isFinite(v)) continue;
    free.push({
      kwarg: inp.kwargName, param: inp.paramName,
      isInt: !!(inp.type.prim === 'integer'),
    });
    x0.push(v);
    domains.push(domainFromDescriptor(doms[inp.kwargName]));
    const r = ranges[inp.kwargName];
    let scale = r && Number.isFinite(r.hi - r.lo) ? Math.abs(r.hi - r.lo) : 0;
    if (!(scale > 0)) {
      const dr = defaultRangeForLeafType(inp.type);
      scale = dr ? Math.abs(dr[1] - dr[0]) : 0;
    }
    scales.push(scale);
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
  // render-profile). Fixed-phase self-refs ride the worker session env below.
  const { perAtomNames: selfRefs } =
    FlatPPLEngine.materialiser.classifyProfileSelfRefs(ir, dst.bindings, dst.fixedValues);
  const selfMeasures = await Promise.all(
    selfRefs.map((n: string) => tryGetMeasure(ctx, n)));
  for (let i = 0; i < selfRefs.length; i++) {
    const m = selfMeasures[i];
    if (m && m.samples && m.samples.length > 0) fixedEnv[selfRefs[i]] = m.samples[0];
  }
  const irBaked = FlatPPLEngine.orchestrator.substituteBoundaryValues(ir, fixedEnv);

  // Fixed-phase self-refs in the body (literal arrays, externals, fixed
  // reductions): push once into the worker session env (merge), as render-
  // profile does, before evaluating the cloud.
  const sessionEnv: Record<string, any> = {};
  const fvMap = dst.fixedValues;
  if (fvMap) {
    for (const n of FlatPPLEngine.orchestrator.collectSelfRefs(ir)) {
      if (fvMap.has(n)) sessionEnv[n] = fvMap.get(n);
    }
  }

  let observed: any;
  if (sig.obsIR != null) {
    observed = FlatPPLEngine.orchestrator.resolveIRToValue(sig.obsIR, dst.bindings, dst.fixedValues);
  }

  // --- 4. Batched cloud objective over the worker ----------------------
  const evalCloud = async (cloud: number[][]): Promise<number[]> => {
    const K = cloud.length;
    const refArrays: Record<string, Float64Array> = {};
    for (let j = 0; j < free.length; j++) {
      const col = new Float64Array(K);
      for (let k = 0; k < K; k++) col[k] = cloud[k][j];
      refArrays[free[j].param] = col;
    }
    // logDensityN in the normal (likelihood) mode: the N-axis indexes the
    // K candidate PARAMETER points (per-θ params via refArrays), scored
    // against the shared `observed` — exactly profileN's sweep, generalised
    // to a cloud. (NOT pointsBatched: that varies the observation instead.)
    const msg: any = mode === 'function'
      ? { type: 'evaluateN', ir: irBaked, refArrays, count: K }
      : { type: 'logDensityN', ir: irBaked, observed, refArrays, count: K };
    const reply: any = await sendWorker(ctx, msg);
    const out = reply && reply.samples;
    const res = new Array(K);
    for (let k = 0; k < K; k++) {
      const v = out ? out[k] : NaN;
      res[k] = Number.isFinite(v) ? v : -Infinity; // out-of-support → worst
    }
    return res;
  };

  // --- 5. Optimize ------------------------------------------------------
  if (Object.keys(sessionEnv).length > 0) {
    await sendWorker(ctx, { type: 'setEnv', env: sessionEnv, merge: true });
  }
  const fit = await FlatPPLEngine.optimizer.optimize({
    evalCloud, x0, domains, scales,
    opts: { seed: 0xf1a7, starts: free.length <= 4 ? 3 : 2 },
  });

  const modeValues: Record<string, number> = {};
  for (let j = 0; j < free.length; j++) {
    modeValues[free[j].kwarg] = free[j].isInt ? Math.round(fit.mode[j]) : fit.mode[j];
  }
  return { ok: true, fit, free, modeValues };
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
  const where = r.free.map((f) => esc(f.kwarg) + ' = '
    + formatScalar(r.modeValues[f.kwarg])).join(', ');
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
  if (!sig || sig.obsIR == null) return;            // likelihoods only — see note
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
      ctx.modeCenterCache!.set(name, { status: 'ready', values: outcome.r.modeValues });
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
