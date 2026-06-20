// @flatppl/viewer — "auto-fit domain" action for function / likelihood plots.
//
// Re-frames each sweepable axis's x-range around the CURRENT pivot. The
// existing auto-fit (resolveSweepRange) frames the prior / source set and is
// pivot-INDEPENDENT, so after the pivot moves — manually, or via the
// neighbouring "find maximum" button — the plotted window can drift off the
// interesting region. This action recomputes each axis's window via the
// engine's `pivotCenteredRange` (same auto width, re-centred on the pivot,
// support-aware) and writes it as the active domain override, exactly like a
// manual range edit (so Save / Discard work unchanged).
//
// The width math is the engine's pure `orchestrator.pivotCenteredRange`
// (unit-tested); this module is the async DOM glue that gathers the pivot,
// the per-axis auto width, and the support kind, then writes the override.

import type { Ctx } from './types';
import {
  activeInputDomain, activeInputValues, ensureDomainOverrideFor,
  resolveSweepRange, setDomainOverrideFor,
} from './overrides.js';
import { showPlotMessage } from './render-frame.js';

/**
 * Re-fit the active plan's domain so every sweepable axis is framed around
 * its current pivot value, then `rerender`. Resolves once the override is
 * written (errors are surfaced as a plot message by the caller).
 */
export async function runAutoDomain(ctx: Ctx, plan: any, rerender: () => void): Promise<void> {
  if (!plan.axes || plan.axes.length === 0) {
    showPlotMessage(ctx, 'Auto-fit domain: this plot has no input axes to fit.', { hint: true });
    return;
  }

  const vals = activeInputValues(ctx, plan);   // pivot value per kwarg
  const doms = activeInputDomain(ctx, plan);   // SetDescriptor per kwarg

  // One window per sweepable axis, deduped by kwarg: linked-cartpow array
  // axes (theta[0], theta[1], …) share a single interval stored under the
  // kwarg name, matching the render-side range lookup and the persist path.
  const seen = new Set<string>();
  const axes: any[] = [];
  for (const a of plan.axes) {
    const kw = a.kwargName || a.key;
    if (!kw || seen.has(kw)) continue;
    seen.add(kw);
    axes.push(a);
  }

  // resolveSweepRange may await the source measure (prior 4-σ width for
  // stochastic sources); fit every axis concurrently.
  const bases = await Promise.all(axes.map(function(a) { return resolveSweepRange(ctx, a); }));

  const entry = ensureDomainOverrideFor(ctx, plan);
  entry.ranges = Object.assign({}, entry.ranges || {});
  let nWritten = 0;
  for (let i = 0; i < axes.length; i++) {
    const a = axes[i];
    const kw = a.kwargName || a.key;
    const base = bases[i];
    if (!base || !Number.isFinite(base[0]) || !Number.isFinite(base[1])) continue;
    const center = vals[kw];
    if (typeof center !== 'number' || !Number.isFinite(center)) continue;
    const desc = doms[kw];
    const kind = (desc && desc.kind) || 'reals';
    const isInt = !!(a.leafType && a.leafType.prim === 'integer');
    const r = FlatPPLEngine.orchestrator.pivotCenteredRange(
      center, base[0], base[1], { kind: kind, isInt: isInt });
    entry.ranges[kw] = { lo: r[0], hi: r[1] };
    nWritten++;
  }

  if (nWritten === 0) {
    showPlotMessage(ctx, 'Auto-fit domain: no sweepable axis had a finite pivot '
      + 'to fit around.', { hint: true });
    return;
  }
  setDomainOverrideFor(ctx, plan, entry);

  // Drop cached auto-fits for this plan so a later "auto" domain reset
  // recomputes from the source rather than serving a stale window. (The
  // override we just wrote already shadows the cache for display.)
  if (ctx.profileRangeCache) {
    const prefix = plan.name + '|';
    const keys = Array.from(ctx.profileRangeCache.keys());
    for (let k = 0; k < keys.length; k++) {
      if (keys[k].indexOf(prefix) === 0) ctx.profileRangeCache.delete(keys[k]);
    }
  }

  rerender();
  showPlotMessage(ctx, 'Domain re-fit around the current point ('
    + nWritten + (nWritten === 1 ? ' axis' : ' axes') + '). Save to keep.', { hint: true });
}
