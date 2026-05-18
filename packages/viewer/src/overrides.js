// @flatppl/viewer — preset + domain override stores (Phase 4d).
//
// Persistent ctx.presetOverrides / ctx.domainOverrides maps and the
// helpers that read/mutate them. Stable across binding navigation;
// reconciled in rebuildDerivations against current source values.

export function overrideEntryFor(ctx, plan) {
  if (plan.presetName == null) return plan.autoOverride;
  return ctx.presetOverrides.get(plan.presetName) || null;
}

export function hasOverrides(ctx, plan) {
  var e = overrideEntryFor(ctx, plan);
  if (!e) return false;
  var v = e.values || {};
  for (var k in v) {
    if (Object.prototype.hasOwnProperty.call(v, k)) return true;
  }
  return false;
}

export function setOverrideFor(ctx, plan, entry) {
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

export function ensureOverrideFor(ctx, plan) {
  var existing = overrideEntryFor(ctx, plan);
  if (existing) {
    existing.values = Object.assign({}, existing.values || {});
    return existing;
  }
  return { values: {} };
}

export function activePresetFor(ctx, plan) {
  var baseValues = baseValuesFor(ctx, plan);
  var entry = overrideEntryFor(ctx, plan);
  if (!entry) return { values: baseValues };
  return {
    values: Object.assign({}, baseValues, entry.values || {}),
  };
}

export function baseValuesFor(ctx, plan) {
  if (plan.presetName != null && plan.matchedPresets) {
    for (var i = 0; i < plan.matchedPresets.length; i++) {
      if (plan.matchedPresets[i].name === plan.presetName) {
        return plan.matchedPresets[i].values || {};
      }
    }
  }
  return {};
}

export function domainOverrideEntryFor(ctx, plan) {
  if (plan.domainName == null) return plan.domainAutoOverride || null;
  return ctx.domainOverrides.get(plan.domainName) || null;
}

export function ensureDomainOverrideFor(ctx, plan) {
  var existing = domainOverrideEntryFor(ctx, plan);
  if (existing) {
    existing.ranges = Object.assign({}, existing.ranges || {});
    return existing;
  }
  return { ranges: {} };
}

export function setDomainOverrideFor(ctx, plan, entry) {
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

export function hasDomainOverrides(ctx, plan) {
  var e = domainOverrideEntryFor(ctx, plan);
  if (!e || !e.ranges) return false;
  return Object.keys(e.ranges).length > 0;
}

export function baseRangesFor(ctx, plan) {
  if (plan.domainName != null && plan.matchedDomains) {
    for (var i = 0; i < plan.matchedDomains.length; i++) {
      if (plan.matchedDomains[i].name === plan.domainName) {
        return plan.matchedDomains[i].ranges || {};
      }
    }
  }
  return {};
}

export function activeDomainRangesFor(ctx, plan) {
  var base = baseRangesFor(ctx, plan);
  var entry = domainOverrideEntryFor(ctx, plan);
  if (!entry || !entry.ranges) return Object.assign({}, base);
  return Object.assign({}, base, entry.ranges);
}

export function activeFixedNamesFor(ctx, plan) {
  if (plan.presetName != null && plan.matchedPresets) {
    for (var i = 0; i < plan.matchedPresets.length; i++) {
      if (plan.matchedPresets[i].name === plan.presetName) {
        return plan.matchedPresets[i].fixedNames || new Set();
      }
    }
  }
  return new Set();
}

export function resolveSweepRange(ctx, axis) {
  var descriptor = FlatPPLEngine.orchestrator.resolveAxisBaseSet(
    axis.source, ctx.derivationsState && ctx.derivationsState.bindings);
  if (descriptor && descriptor.kind === 'empirical') {
    return getMeasure(ctx, descriptor.name).then(function(m) {
      if (m && m.samples && m.samples.length > 0) {
        var range = FlatPPLEngine.orchestrator.fourSigmaQuantileRange(m.samples);
        if (range && range[0] < range[1]) return range;
      }
      return defaultRangeForLeafType(axis.leafType);
    }, function() {
      return defaultRangeForLeafType(axis.leafType);
    });
  }
  var fromDescriptor = rangeFromSetDescriptor(descriptor);
  if (fromDescriptor) return Promise.resolve(fromDescriptor);
  return Promise.resolve(defaultRangeForLeafType(axis.leafType));
}

export function applyRememberedSelections(ctx, plan) {
  if (!plan) return;
  var mem = ctx.planMemoryByName.get(plan.name);
  if (!mem) return;
  var axisKwargs = new Set();
  if (plan.axes) {
    for (var i = 0; i < plan.axes.length; i++) {
      if (plan.axes[i].kwargName) axisKwargs.add(plan.axes[i].kwargName);
    }
  }
  if (mem.sweepKey
      && plan.axes
      && plan.axes.some(function(a) { return a.key === mem.sweepKey; })) {
    plan.sweepKey = mem.sweepKey;
  }
  if (mem.outputKey
      && plan.outputs
      && plan.outputs.some(function(o) { return o.key === mem.outputKey; })) {
    plan.outputKey = mem.outputKey;
  }
  plan.autoOverride = filterOverrideToAxes(mem.autoOverride, axisKwargs, 'values');
  plan.domainAutoOverride = filterOverrideToAxes(mem.domainAutoOverride, axisKwargs, 'ranges');
  if (mem.presetName != null
      && plan.matchedPresets
      && plan.matchedPresets.some(function(p) { return p.name === mem.presetName; })) {
    plan.presetName = mem.presetName;
  }
  if (mem.domainName != null
      && plan.matchedDomains
      && plan.matchedDomains.some(function(d) { return d.name === mem.domainName; })) {
    plan.domainName = mem.domainName;
  }
}

export function rememberPlanSelections(ctx, plan) {
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

export function computeAutoValues(ctx, plan) {
  var out = {};
  var axes = plan.axes || [];
  for (var i = 0; i < axes.length; i++) {
    var ax = axes[i];
    var def = defaultValueForLeafType(ax.leafType);
    if (ax.source && ax.source.kind === 'binding'
        && ctx.measureCache && ctx.measureCache.has(ax.source.name)) {
      var m = ctx.measureCache.get(ax.source.name);
      if (m && m.samples && m.samples.length > 0) def = m.samples[0];
    }
    out[ax.kwargName] = def;
  }
  return out;
}
