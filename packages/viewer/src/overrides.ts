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
  if (mem.presetName != null
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
  const out: Record<string, any> = {};
  const axes = plan.axes || [];
  const bindings = ctx.derivationsState && ctx.derivationsState.bindings;
  const fixedValues = ctx.derivationsState && ctx.derivationsState.fixedValues;
  // Group by kwargName: array-typed inputs (`walkType` emits one
  // axis per slot) share a kwarg — one source lookup, one entry in
  // `out` (as a J-element array). Without grouping, J slots would
  // overwrite each other with the same scalar.
  const inputs = (plan.signature && plan.signature.inputs) || [];
  const inputByKwarg: Record<string, any> = {};
  for (let k = 0; k < inputs.length; k++) inputByKwarg[inputs[k].kwargName] = inputs[k];
  const seen = new Set<string>();
  for (let i = 0; i < axes.length; i++) {
    const ax = axes[i];
    if (seen.has(ax.kwargName)) continue;
    seen.add(ax.kwargName);
    const inp = inputByKwarg[ax.kwargName];
    const arrayLen = inp ? arrayInputLength(inp.type) : null;
    const leafDef = defaultValueForLeafType(ax.leafType);
    let def: any = arrayLen != null ? new Array(arrayLen).fill(leafDef) : leafDef;
    if (ax.source && ax.source.kind === 'binding'
        && ctx.measureCache && ctx.measureCache.has(ax.source.name)) {
      // Latent guard: skip function-like, fixed-phase, or non-binding
      // sources. Fixed-phase array bindings collapse to samples[0]'s
      // first scalar element under the naive override (the same
      // failure mode we fixed in render-profile's profile-plot path
      // — flatppl-js commit e9984f3).
      const src = bindings && bindings.get(ax.source.name);
      const isFunctionLike = FlatPPLEngine.materialiser.isFunctionLikeBinding(src);
      const isFixedPhase = fixedValues && fixedValues.has(ax.source.name);
      if (!isFunctionLike && !isFixedPhase) {
        const m = ctx.measureCache.get(ax.source.name);
        if (m && m.samples && m.samples.length > 0) {
          if (arrayLen != null && m.samples.length >= arrayLen) {
            def = Array.from(m.samples.slice(0, arrayLen));
          } else {
            def = m.samples[0];
          }
        }
      }
    }
    out[ax.kwargName] = def;
  }
  return out;
}
