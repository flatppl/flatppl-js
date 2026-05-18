// @flatppl/viewer — persist-to-source (Phase 4d).
//
// Builds source-text edits (preset binding lines, domain cartprod
// lines) and applies them via host.editSource. Pure of DOM until
// the host adapter dispatches; testable end-to-end against a
// captured editSource mock.

import { activePresetFor, computeAutoValues, domainOverrideEntryFor, hasDomainOverrides, hasOverrides } from './overrides.js';

export function formatScalarForSource(ctx, v) {
  if (typeof v === 'boolean') {
    // Boolean spelling follows the source-file variant: FlatPPL
    // and FlatPPJ use lowercase `true`/`false`; FlatPPY uses
    // capitalized `True`/`False`.
    if (ctx.currentVariantId === 'flatppy') return v ? 'True' : 'False';
    return v ? 'true' : 'false';
  }
  if (!Number.isFinite(v)) return String(v);
  return String(v);
}

export function canPersistActive(ctx, plan) {
  if (!hasOverrides(ctx, plan)) return false;
  if (!ctx.host || typeof ctx.host.editSource !== 'function') return false;
  if (typeof ctx.host.canPersist === 'function' && !ctx.host.canPersist()) return false;
  if (ctx.host.canPersist === false) return false;
  if (plan.presetName == null) {
    return typeof ctx.host.promptForName === 'function';
  }
  if (!ctx.currentBindings) return false;
  var b = ctx.currentBindings.get(plan.presetName);
  if (!b || !b.node || !b.node.value
      || b.node.value.type !== 'CallExpr'
      || !b.node.value.callee
      || b.node.value.callee.name !== 'record') return false;
  // Persist only when every field is a literal — possibly wrapped
  // in fixed(...) which is identity at runtime (spec §03) and just
  // a "hold constant" hint we preserve when rewriting. Anything
  // more structural (refs, nested calls) means the source isn't
  // edit-in-place writable; canPersist returns false so the
  // toolbar hides the button rather than offering a broken
  // write-back.
  var args = b.node.value.args || [];
  for (var i = 0; i < args.length; i++) {
    var a = args[i];
    if (a.type !== 'KeywordArg' || !a.value) return false;
    var v = a.value;
    if (v.type === 'CallExpr' && v.callee && v.callee.name === 'fixed'
        && Array.isArray(v.args) && v.args.length === 1) {
      v = v.args[0];
    }
    if (v.type !== 'NumberLiteral' && v.type !== 'BoolLiteral') return false;
  }
  return true;
}

export function buildPersistedPresetLine(ctx, plan) {
  var active = activePresetFor(ctx, plan);
  var b = ctx.currentBindings.get(plan.presetName);
  var srcArgs = b.node.value.args || [];
  var parts = [];
  for (var i = 0; i < srcArgs.length; i++) {
    var sa = srcArgs[i];
    var kwarg = sa.name;
    var srcVal = sa.value;
    var wasFixed = srcVal && srcVal.type === 'CallExpr'
                 && srcVal.callee && srcVal.callee.name === 'fixed';
    var innerSrc = wasFixed ? srcVal.args[0] : srcVal;
    var override = active.values
                && Object.prototype.hasOwnProperty.call(active.values, kwarg);
    var v = override ? active.values[kwarg]
                     : (innerSrc && innerSrc.value);
    var text = formatScalarForSource(ctx, v);
    if (wasFixed) text = 'fixed(' + text + ')';
    parts.push(kwarg + ' = ' + text);
  }
  return plan.presetName + ' = record(' + parts.join(', ') + ')';
}

export function persistActive(ctx, plan) {
  if (!canPersistActive(ctx, plan)) return;
  if (plan.presetName == null) {
    persistAutoAsNewBinding(ctx, plan);
  } else {
    persistNamedPreset(ctx, plan);
  }
}

export function persistNamedPreset(ctx, plan) {
  var b = ctx.currentBindings.get(plan.presetName);
  var newText = buildPersistedPresetLine(ctx, plan);
  try {
    ctx.host.editSource({
      range: {
        start: { line: b.node.loc.start.line, col: b.node.loc.start.col },
        end:   { line: b.node.loc.end.line,   col: b.node.loc.end.col },
      },
      newText: newText,
    });
  } catch (err) {
    console.error('[viewer] editSource (named persist) failed:', err);
  }
}

export function persistAutoAsNewBinding(ctx, plan) {
  if (typeof ctx.host.promptForName !== 'function'
      || typeof ctx.host.editSource !== 'function') {
    console.warn('[viewer] persist auto: ctx.host missing promptForName / editSource');
    return;
  }
  var autoValues = computeAutoValues(ctx, plan);
  var override = plan.autoOverride;
  var combined = Object.assign({}, autoValues, (override && override.values) || {});
  var parts = [];
  for (var k in combined) {
    if (!Object.prototype.hasOwnProperty.call(combined, k)) continue;
    var v = combined[k];
    if (!Number.isFinite(v)) continue;
    parts.push(k + ' = ' + formatScalarForSource(ctx, v));
  }
  if (parts.length === 0) return;
  var existingNames = [];
  if (ctx.currentBindings) ctx.currentBindings.forEach(function(_b, n) { existingNames.push(n); });
  var pairsText = parts.join(', ');
  var suggested = (plan.name || 'inputs') + '_default';
  Promise.resolve(ctx.host.promptForName({
    suggested: suggested,
    existingNames: existingNames,
  })).then(function(name) {
    if (!name) return;
    ctx.pendingPresetName = name;
    ctx.host.editSource({
      range: null,
      newText: name + ' = record(' + pairsText + ')',
    });
  }).catch(function(err) {
    console.error('[viewer] persistAutoAsNewBinding failed:', err);
  });
}

export function setFieldToSource(ctx, v) {
  if (v.type === 'Identifier') return v.name;
  // interval(NumLit, NumLit)
  return 'interval('
    + formatScalarForSource(ctx, v.args[0].value) + ', '
    + formatScalarForSource(ctx, v.args[1].value) + ')';
}

export function defaultSetSourceForKwarg(ctx, plan, kwargName) {
  if (!plan.axes) return 'reals';
  var matching = [];
  for (var i = 0; i < plan.axes.length; i++) {
    if (plan.axes[i].kwargName === kwargName) matching.push(plan.axes[i]);
  }
  if (matching.length !== 1) return 'reals';  // non-scalar — defer
  var bindings = ctx.derivationsState && ctx.derivationsState.bindings;
  var d = null;
  try {
    d = FlatPPLEngine.orchestrator.resolveAxisBaseSet(matching[0].source, bindings);
  } catch (_) { d = null; }
  if (!d) return 'reals';
  switch (d.kind) {
    case 'reals':           return 'reals';
    case 'posreals':        return 'posreals';
    case 'nonnegreals':     return 'nonnegreals';
    case 'integers':        return 'integers';
    case 'posintegers':     return 'posintegers';
    case 'nonnegintegers':  return 'nonnegintegers';
    case 'booleans':        return 'booleans';
    case 'interval':
      if (d.lo === 0 && d.hi === 1) return 'unitinterval';
      return 'interval('
        + formatScalarForSource(ctx, d.lo) + ', '
        + formatScalarForSource(ctx, d.hi) + ')';
    default:                return 'reals';  // empirical / unknown
  }
}

export function canPersistDomain(ctx, plan) {
  if (!hasDomainOverrides(ctx, plan)) return false;
  if (!ctx.host || typeof ctx.host.editSource !== 'function') return false;
  if (typeof ctx.host.canPersist === 'function' && !ctx.host.canPersist()) return false;
  if (ctx.host.canPersist === false) return false;
  if (plan.domainName == null) {
    return typeof ctx.host.promptForName === 'function';
  }
  if (!ctx.currentBindings) return false;
  var b = ctx.currentBindings.get(plan.domainName);
  if (!b || !b.node || !b.node.value
      || b.node.value.type !== 'CallExpr'
      || !b.node.value.callee
      || b.node.value.callee.name !== 'cartprod') return false;
  var args = b.node.value.args || [];
  for (var i = 0; i < args.length; i++) {
    var a = args[i];
    if (a.type !== 'KeywordArg' || !a.value) return false;
    if (!isPersistableSetField(a.value)) return false;
  }
  return true;
}

export function persistDomain(ctx, plan) {
  if (!canPersistDomain(ctx, plan)) return;
  if (plan.domainName == null) {
    persistAutoDomainAsNewBinding(ctx, plan);
  } else {
    persistNamedDomain(ctx, plan);
  }
}

export function buildPersistedDomainLine(ctx, plan) {
  var b = ctx.currentBindings.get(plan.domainName);
  var srcArgs = b.node.value.args || [];
  var override = domainOverrideEntryFor(ctx, plan);
  var or = (override && override.ranges) || {};
  var parts = [];
  for (var i = 0; i < srcArgs.length; i++) {
    var sa = srcArgs[i];
    var kwarg = sa.name;
    if (Object.prototype.hasOwnProperty.call(or, kwarg)) {
      parts.push(kwarg + ' = interval('
        + formatScalarForSource(ctx, or[kwarg].lo) + ', '
        + formatScalarForSource(ctx, or[kwarg].hi) + ')');
    } else {
      parts.push(kwarg + ' = ' + setFieldToSource(ctx, sa.value));
    }
  }
  return plan.domainName + ' = cartprod(' + parts.join(', ') + ')';
}

export function persistNamedDomain(ctx, plan) {
  var b = ctx.currentBindings.get(plan.domainName);
  var newText = buildPersistedDomainLine(ctx, plan);
  try {
    ctx.host.editSource({
      range: {
        start: { line: b.node.loc.start.line - 1, col: 0 },
        end:   { line: b.node.loc.end.line   - 1, col: 1000000 },
      },
      newText: newText,
    });
  } catch (err) {
    console.error('[viewer] persistNamedDomain failed:', err);
  }
}

export function persistAutoDomainAsNewBinding(ctx, plan) {
  if (typeof ctx.host.promptForName !== 'function'
      || typeof ctx.host.editSource !== 'function') {
    console.warn('[viewer] persist domain auto: ctx.host missing promptForName / editSource');
    return;
  }
  var override = plan.domainAutoOverride;
  var ranges = (override && override.ranges) || {};
  // Enumerate every signature input so the resulting cartprod has
  // full shape coverage. Per-kwarg precedence:
  //   1. user override range          → interval(lo, hi)
  //   2. auto-fit cached for this kwarg in profileRangeCache
  //      (the plot engine populated it when the user previously
  //      had this kwarg selected as sweep axis) → interval(lo, hi)
  //   3. natural base set from the input's source descriptor
  //      → bare named set (reals / posreals / …)
  // Step 2 means an axis the user looked at but never edited
  // still persists with its observed bounds rather than being
  // weakened to the natural set.
  var inputs = (plan.signature && plan.signature.inputs) || [];
  var parts = [];
  for (var i = 0; i < inputs.length; i++) {
    var kw = inputs[i].kwargName;
    if (!kw) continue;
    var r = Object.prototype.hasOwnProperty.call(ranges, kw) ? ranges[kw] : null;
    if (r && Number.isFinite(r.lo) && Number.isFinite(r.hi)) {
      parts.push(kw + ' = interval('
        + formatScalarForSource(ctx, r.lo) + ', '
        + formatScalarForSource(ctx, r.hi) + ')');
      continue;
    }
    var cached = ctx.profileRangeCache.get(
      plan.name + '|' + kw + '|D=' + (plan.domainName || ''));
    if (cached && Number.isFinite(cached.lo) && Number.isFinite(cached.hi)) {
      parts.push(kw + ' = interval('
        + formatScalarForSource(ctx, cached.lo) + ', '
        + formatScalarForSource(ctx, cached.hi) + ')');
      continue;
    }
    parts.push(kw + ' = ' + defaultSetSourceForKwarg(ctx, plan, kw));
  }
  if (parts.length === 0) return;
  var existingNames = [];
  if (ctx.currentBindings) ctx.currentBindings.forEach(function(_b, n) { existingNames.push(n); });
  var pairsText = parts.join(', ');
  var suggested = (plan.name || 'domain') + '_domain';
  Promise.resolve(ctx.host.promptForName({
    suggested: suggested,
    existingNames: existingNames,
  })).then(function(name) {
    if (!name) return;
    ctx.pendingDomainName = name;
    ctx.host.editSource({
      range: null,
      newText: name + ' = cartprod(' + pairsText + ')',
    });
  }).catch(function(err) {
    console.error('[viewer] persistAutoDomainAsNewBinding failed:', err);
  });
}
