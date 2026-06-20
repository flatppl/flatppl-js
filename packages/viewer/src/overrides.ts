// @flatppl/viewer — preset + domain override stores —
//
// Persistent ctx.presetOverrides / ctx.domainOverrides maps and the
// helpers that read/mutate them. Stable across binding navigation;
// reconciled in rebuildDerivations against current source values.
//
// The `plan` parameter on these helpers is intentionally `any`: most
// touch fields shared only by ProfilePlan / KernelSamplePlan
// (presetName, matchedPresets, …), and some are called on the full
// Plan union where TS's narrowing-by-mode would reject those reads.
// Per-call narrowing happens at the renderer boundary (render-profile,
// render-kernel) — this file stays cross-mode.

import type { Ctx } from './types';
import { getMeasure } from './engine-facade.js';
import { arrayInputLength, defaultRangeForLeafType, defaultValueForLeafType, filterOverrideToAxes, rangeFromSetDescriptor } from './util.js';

// Sentinel preset name for the labelled `auto (MLE)` default point on
// likelihood plots. ':' can't appear in a user binding identifier (same
// trick as MODULE_TARGET), so it never collides with a real preset; its
// base values come from ctx.modeCenterCache rather than matchedPresets.
export const MLE_PRESET = ':mle';

export function overrideEntryFor(ctx: Ctx, plan: any) {
  if (plan.presetName == null) return plan.autoOverride;
  return ctx.presetOverrides.get(plan.presetName) || null;
}

export function hasOverrides(ctx: Ctx, plan: any) {
  const e = overrideEntryFor(ctx, plan);
  if (!e) return false;
  const v = e.values || {};
  for (const k in v) {
    if (Object.prototype.hasOwnProperty.call(v, k)) return true;
  }
  return false;
}

export function setOverrideFor(ctx: Ctx, plan: any, entry: any) {
  if (plan.presetName == null) {
    plan.autoOverride = entry;
    return;
  }
  if (entry) {
    ctx.presetOverrides.set(plan.presetName, entry);
  } else {
    ctx.presetOverrides.delete(plan.presetName);
  }
}

export function ensureOverrideFor(ctx: Ctx, plan: any) {
  const existing = overrideEntryFor(ctx, plan);
  if (existing) {
    existing.values = Object.assign({}, existing.values || {});
    return existing;
  }
  return { values: {} };
}

export function activePresetFor(ctx: Ctx, plan: any) {
  const baseValues = baseValuesFor(ctx, plan);
  const entry = overrideEntryFor(ctx, plan);
  if (!entry) return { values: baseValues };
  return {
    values: Object.assign({}, baseValues, entry.values || {}),
  };
}

export function baseValuesFor(ctx: Ctx, plan: any) {
  // `auto (MLE)`: base values are the cached MLE point. If it isn't ready
  // (pending / failed / cleared by a rebuild) return {} — effectiveInputValues
  // then falls through to the prior-draw `auto`, so selecting MLE before it's
  // computed degrades gracefully rather than blanking the pivot.
  if (plan.presetName === MLE_PRESET) {
    const e = ctx.modeCenterCache && ctx.modeCenterCache.get(plan.name);
    return (e && e.status === 'ready' && e.values) ? e.values : {};
  }
  if (plan.presetName != null && plan.matchedPresets) {
    for (let i = 0; i < plan.matchedPresets.length; i++) {
      if (plan.matchedPresets[i].name === plan.presetName) {
        return plan.matchedPresets[i].values || {};
      }
    }
  }
  return {};
}

export function domainOverrideEntryFor(ctx: Ctx, plan: any) {
  if (plan.domainName == null) return plan.domainAutoOverride || null;
  return ctx.domainOverrides.get(plan.domainName) || null;
}

export function ensureDomainOverrideFor(ctx: Ctx, plan: any) {
  const existing = domainOverrideEntryFor(ctx, plan);
  if (existing) {
    existing.ranges = Object.assign({}, existing.ranges || {});
    return existing;
  }
  return { ranges: {} };
}

export function setDomainOverrideFor(ctx: Ctx, plan: any, entry: any) {
  if (plan.domainName == null) {
    plan.domainAutoOverride = entry;
    return;
  }
  if (entry) {
    ctx.domainOverrides.set(plan.domainName, entry);
  } else {
    ctx.domainOverrides.delete(plan.domainName);
  }
}

export function hasDomainOverrides(ctx: Ctx, plan: any) {
  const e = domainOverrideEntryFor(ctx, plan);
  if (!e || !e.ranges) return false;
  return Object.keys(e.ranges).length > 0;
}

export function baseRangesFor(ctx: Ctx, plan: any) {
  if (plan.domainName != null && plan.matchedDomains) {
    for (let i = 0; i < plan.matchedDomains.length; i++) {
      if (plan.matchedDomains[i].name === plan.domainName) {
        return plan.matchedDomains[i].ranges || {};
      }
    }
  }
  return {};
}

export function activeDomainRangesFor(ctx: Ctx, plan: any) {
  const base = baseRangesFor(ctx, plan);
  const entry = domainOverrideEntryFor(ctx, plan);
  if (!entry || !entry.ranges) return Object.assign({}, base);
  return Object.assign({}, base, entry.ranges);
}

/**
 * Effective per-kwarg input values for the active plan: merges
 * auto / base preset / user override via the engine helper. The
 * single source of "what's the current value of each kwarg" —
 * commitSliceX, persist, and the auto preset dropdown all consume
 * this rather than re-doing the merge by hand.
 */
export function activeInputValues(ctx: Ctx, plan: any): Record<string, any> {
  if (!plan || !plan.signature) return {};
  const bindings = ctx.derivationsState && ctx.derivationsState.bindings;
  const fixedValues = ctx.derivationsState && ctx.derivationsState.fixedValues;
  const cache = ctx.measureCache;
  function getAtomZero(name: string): any {
    if (!cache || !cache.has(name)) return null;
    const src = bindings && bindings.get(name);
    if (FlatPPLEngine.materialiser.isFunctionLikeBinding(src)) return null;
    // Demand-driven (§17.4): is-fixed-phase is a phase question — read
    // binding.phase, not `fixedValues.has` (which would force-resolve).
    if (src && src.phase === 'fixed') return null;
    const m = cache.get(name);
    if (!m || !m.samples || m.samples.length === 0) return null;
    return m.samples;
  }
  const baseValues = baseValuesFor(ctx, plan);
  const overrideEntry = overrideEntryFor(ctx, plan);
  const overrideValues = (overrideEntry && overrideEntry.values) || {};
  return FlatPPLEngine.orchestrator.effectiveInputValues(
    plan.signature, bindings, fixedValues, baseValues, overrideValues, getAtomZero);
}

/**
 * Effective per-kwarg input domain (SetDescriptor) for the active
 * plan: merges auto / base named-domain / user override via the
 * engine helper. The single source of "what's the current domain
 * for each kwarg" — commitRange / persist / the domain dropdown
 * all consume this.
 */
export function activeInputDomain(ctx: Ctx, plan: any): Record<string, any> {
  if (!plan || !plan.signature) return {};
  const bindings = ctx.derivationsState && ctx.derivationsState.bindings;
  const baseRanges = baseRangesFor(ctx, plan);
  // baseSetNames lives on plan.matchedDomains alongside ranges.
  let baseSetNames: Record<string, string> = {};
  if (plan.domainName != null && plan.matchedDomains) {
    for (let i = 0; i < plan.matchedDomains.length; i++) {
      if (plan.matchedDomains[i].name === plan.domainName) {
        baseSetNames = plan.matchedDomains[i].setNames || {};
        break;
      }
    }
  }
  const overrideEntry = domainOverrideEntryFor(ctx, plan);
  const overrideRanges = (overrideEntry && overrideEntry.ranges) || {};
  return FlatPPLEngine.orchestrator.effectiveInputDomain(
    plan.signature, bindings, baseRanges, baseSetNames, overrideRanges);
}

export function activeFixedNamesFor(ctx: Ctx, plan: any) {
  if (plan.presetName != null && plan.matchedPresets) {
    for (let i = 0; i < plan.matchedPresets.length; i++) {
      if (plan.matchedPresets[i].name === plan.presetName) {
        return plan.matchedPresets[i].fixedNames || new Set();
      }
    }
  }
  return new Set();
}

export function resolveSweepRange(ctx: Ctx, axis: any) {
  const descriptor = FlatPPLEngine.orchestrator.resolveAxisBaseSet(
    axis.source, ctx.derivationsState && ctx.derivationsState.bindings);
  if (descriptor && descriptor.kind === 'empirical') {
    return getMeasure(ctx, descriptor.name).then(function(m: any) {
      if (m && m.samples && m.samples.length > 0) {
        const range = FlatPPLEngine.orchestrator.fourSigmaQuantileRange(m.samples);
        if (range && range[0] < range[1]) return range;
      }
      return defaultRangeForLeafType(axis.leafType);
    }, function() {
      return defaultRangeForLeafType(axis.leafType);
    });
  }
  const fromDescriptor = rangeFromSetDescriptor(descriptor);
  if (fromDescriptor) return Promise.resolve(fromDescriptor);
  return Promise.resolve(defaultRangeForLeafType(axis.leafType));
}

export function applyRememberedSelections(ctx: Ctx, plan: any) {
  if (!plan) return;
  const mem = ctx.planMemoryByName.get(plan.name);
  if (!mem) return;
  const axisKwargs = new Set<any>();
  if (plan.axes) {
    for (let i = 0; i < plan.axes.length; i++) {
      if (plan.axes[i].kwargName) axisKwargs.add(plan.axes[i].kwargName);
    }
  }
  if (mem.sweepKey
      && plan.axes
      && plan.axes.some(function(a: any) { return a.key === mem.sweepKey; })) {
    plan.sweepKey = mem.sweepKey;
  }
  if (mem.outputKey
      && plan.outputs
      && plan.outputs.some(function(o: any) { return o.key === mem.outputKey; })) {
    plan.outputKey = mem.outputKey;
  }
  plan.autoOverride = filterOverrideToAxes(mem.autoOverride, axisKwargs, 'values');
  plan.domainAutoOverride = filterOverrideToAxes(mem.domainAutoOverride, axisKwargs, 'ranges');
  if (mem.presetName === MLE_PRESET) {
    // `auto (MLE)` isn't a matchedPreset; restore it directly. If its cache
    // entry was cleared (source edit), baseValuesFor falls through to the
    // prior-draw `auto`, so this is always safe.
    plan.presetName = MLE_PRESET;
  } else if (mem.presetName != null
      && plan.matchedPresets
      && plan.matchedPresets.some(function(p: any) { return p.name === mem.presetName; })) {
    plan.presetName = mem.presetName;
  }
  if (mem.domainName != null
      && plan.matchedDomains
      && plan.matchedDomains.some(function(d: any) { return d.name === mem.domainName; })) {
    plan.domainName = mem.domainName;
  }
}

export function rememberPlanSelections(ctx: Ctx, plan: any) {
  if (!plan || !plan.name) return;
  ctx.planMemoryByName.set(plan.name, {
    sweepKey: plan.sweepKey || null,
    outputKey: plan.outputKey || null,
    presetName: plan.presetName || null,
    domainName: plan.domainName || null,
    autoOverride: plan.autoOverride || null,
    domainAutoOverride: plan.domainAutoOverride || null,
  });
}

export function computeAutoValues(ctx: Ctx, plan: any) {
  // Delegates per-kwarg resolution to engine.orchestrator.compute-
  // AutoInputs — single source of truth for the array-vs-scalar
  // dispatch, per-kwarg dedup, and "atom-0 vs leaf-default"
  // fallback. The viewer's responsibility is the side-table
  // lookup that engine code can't reach into directly: the
  // measure cache + the function-like / fixed-phase filters
  // (a synchronous `getAtomZero(name)` callback).
  if (!plan || !plan.signature) return {};
  const bindings = ctx.derivationsState && ctx.derivationsState.bindings;
  const fixedValues = ctx.derivationsState && ctx.derivationsState.fixedValues;
  const cache = ctx.measureCache;
  function getAtomZero(name: string): any {
    if (!cache || !cache.has(name)) return null;
    // Skip function-like and fixed-phase sources (same latent guard
    // as before — fixed-phase array bindings collapse to samples[0]
    // under naive override). Demand-driven (§17.4): is-fixed-phase is a
    // phase question — read binding.phase, not `fixedValues.has`.
    const src = bindings && bindings.get(name);
    if (FlatPPLEngine.materialiser.isFunctionLikeBinding(src)) return null;
    if (src && src.phase === 'fixed') return null;
    const m = cache.get(name);
    if (!m || !m.samples || m.samples.length === 0) return null;
    // Always return the samples view; computeAutoInputs slices to
    // arrayLen for array-typed inputs and takes [0] for scalars.
    return m.samples;
  }
  return FlatPPLEngine.orchestrator.computeAutoInputs(
    plan.signature, bindings, fixedValues, getAtomZero);
}
