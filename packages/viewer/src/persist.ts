// @flatppl/viewer — persist-to-source —
//
// Builds source-text edits (preset binding lines, domain cartprod
// lines) and applies them via host.editSource. Pure of DOM until
// the host adapter dispatches; testable end-to-end against a
// captured editSource mock.

import type { Ctx } from './types';
import { activePresetFor, computeAutoValues, domainOverrideEntryFor, hasDomainOverrides, hasOverrides } from './overrides.js';

import { isPersistableSetField } from './util.js';
export function formatScalarForSource(ctx: Ctx, v: unknown): string {
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

export function canPersistActive(ctx: Ctx, plan: any): boolean {
  if (!hasOverrides(ctx, plan)) return false;
  if (!ctx.host || typeof ctx.host.editSource !== 'function') return false;
  if (typeof ctx.host.canPersist === 'function' && !ctx.host.canPersist()) return false;
  if (ctx.host.canPersist === false) return false;
  if (plan.presetName == null) {
    return typeof ctx.host.promptForName === 'function';
  }
  if (!ctx.currentBindings) return false;
  const b = ctx.currentBindings.get(plan.presetName);
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
  const args = b.node.value.args || [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a.type !== 'KeywordArg' || !a.value) return false;
    let v = a.value;
    if (v.type === 'CallExpr' && v.callee && v.callee.name === 'fixed'
        && Array.isArray(v.args) && v.args.length === 1) {
      v = v.args[0];
    }
    if (v.type !== 'NumberLiteral' && v.type !== 'BoolLiteral') return false;
  }
  return true;
}

export function buildPersistedPresetLine(ctx: Ctx, plan: any): string {
  const active = activePresetFor(ctx, plan);
  const b = ctx.currentBindings!.get(plan.presetName);
  const srcArgs = b.node.value.args || [];
  const parts: string[] = [];
  for (let i = 0; i < srcArgs.length; i++) {
    const sa = srcArgs[i];
    const kwarg = sa.name;
    const srcVal = sa.value;
    const wasFixed = srcVal && srcVal.type === 'CallExpr'
                 && srcVal.callee && srcVal.callee.name === 'fixed';
    const innerSrc = wasFixed ? srcVal.args[0] : srcVal;
    const override = active.values
                && Object.prototype.hasOwnProperty.call(active.values, kwarg);
    const v = override ? active.values[kwarg]
                     : (innerSrc && innerSrc.value);
    let text = formatScalarForSource(ctx, v);
    if (wasFixed) text = 'fixed(' + text + ')';
    parts.push(kwarg + ' = ' + text);
  }
  return plan.presetName + ' = record(' + parts.join(', ') + ')';
}

export function persistActive(ctx: Ctx, plan: any): void {
  if (!canPersistActive(ctx, plan)) return;
  if (plan.presetName == null) {
    persistAutoAsNewBinding(ctx, plan);
  } else {
    persistNamedPreset(ctx, plan);
  }
}

export function persistNamedPreset(ctx: Ctx, plan: any): void {
  const b = ctx.currentBindings!.get(plan.presetName);
  const newText = buildPersistedPresetLine(ctx, plan);
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

export function persistAutoAsNewBinding(ctx: Ctx, plan: any): void {
  if (typeof ctx.host.promptForName !== 'function'
      || typeof ctx.host.editSource !== 'function') {
    console.warn('[viewer] persist auto: ctx.host missing promptForName / editSource');
    return;
  }
  const autoValues = computeAutoValues(ctx, plan);
  const override = plan.autoOverride;
  const combined = Object.assign({}, autoValues, (override && override.values) || {});
  const parts: string[] = [];
  for (const k in combined) {
    if (!Object.prototype.hasOwnProperty.call(combined, k)) continue;
    const v = combined[k];
    if (!Number.isFinite(v)) continue;
    parts.push(k + ' = ' + formatScalarForSource(ctx, v));
  }
  if (parts.length === 0) return;
  const existingNames: string[] = [];
  if (ctx.currentBindings) ctx.currentBindings.forEach(function(_b: unknown, n: string) { existingNames.push(n); });
  const pairsText = parts.join(', ');
  const suggested = (plan.name || 'inputs') + '_default';
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

export function setFieldToSource(ctx: Ctx, v: any): string {
  if (v.type === 'Identifier') return v.name;
  // interval(NumLit, NumLit)
  return 'interval('
    + formatScalarForSource(ctx, v.args[0].value) + ', '
    + formatScalarForSource(ctx, v.args[1].value) + ')';
}

export function defaultSetSourceForKwarg(ctx: Ctx, plan: any, kwargName: string): string {
  if (!plan.axes) return 'reals';
  const matching: any[] = [];
  for (let i = 0; i < plan.axes.length; i++) {
    if (plan.axes[i].kwargName === kwargName) matching.push(plan.axes[i]);
  }
  if (matching.length !== 1) return 'reals';  // non-scalar — defer
  const bindings = ctx.derivationsState && ctx.derivationsState.bindings;
  let d: any = null;
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

export function canPersistDomain(ctx: Ctx, plan: any): boolean {
  if (!hasDomainOverrides(ctx, plan)) return false;
  if (!ctx.host || typeof ctx.host.editSource !== 'function') return false;
  if (typeof ctx.host.canPersist === 'function' && !ctx.host.canPersist()) return false;
  if (ctx.host.canPersist === false) return false;
  if (plan.domainName == null) {
    return typeof ctx.host.promptForName === 'function';
  }
  if (!ctx.currentBindings) return false;
  const b = ctx.currentBindings.get(plan.domainName);
  if (!b || !b.node || !b.node.value
      || b.node.value.type !== 'CallExpr'
      || !b.node.value.callee
      || b.node.value.callee.name !== 'cartprod') return false;
  const args = b.node.value.args || [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a.type !== 'KeywordArg' || !a.value) return false;
    if (!isPersistableSetField(a.value)) return false;
  }
  return true;
}

export function persistDomain(ctx: Ctx, plan: any): void {
  if (!canPersistDomain(ctx, plan)) return;
  if (plan.domainName == null) {
    persistAutoDomainAsNewBinding(ctx, plan);
  } else {
    persistNamedDomain(ctx, plan);
  }
}

export function buildPersistedDomainLine(ctx: Ctx, plan: any): string {
  const b = ctx.currentBindings!.get(plan.domainName);
  const srcArgs = b.node.value.args || [];
  const override = domainOverrideEntryFor(ctx, plan);
  const or = (override && override.ranges) || {};
  const parts: string[] = [];
  for (let i = 0; i < srcArgs.length; i++) {
    const sa = srcArgs[i];
    const kwarg = sa.name;
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

export function persistNamedDomain(ctx: Ctx, plan: any): void {
  const b = ctx.currentBindings!.get(plan.domainName);
  const newText = buildPersistedDomainLine(ctx, plan);
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

export function persistAutoDomainAsNewBinding(ctx: Ctx, plan: any): void {
  if (typeof ctx.host.promptForName !== 'function'
      || typeof ctx.host.editSource !== 'function') {
    console.warn('[viewer] persist domain auto: ctx.host missing promptForName / editSource');
    return;
  }
  const override = plan.domainAutoOverride;
  const ranges = (override && override.ranges) || {};
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
  const inputs = (plan.signature && plan.signature.inputs) || [];
  const parts: string[] = [];
  for (let i = 0; i < inputs.length; i++) {
    const kw = inputs[i].kwargName;
    if (!kw) continue;
    const r = Object.prototype.hasOwnProperty.call(ranges, kw) ? ranges[kw] : null;
    if (r && Number.isFinite(r.lo) && Number.isFinite(r.hi)) {
      parts.push(kw + ' = interval('
        + formatScalarForSource(ctx, r.lo) + ', '
        + formatScalarForSource(ctx, r.hi) + ')');
      continue;
    }
    const cached = ctx.profileRangeCache.get(
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
  const existingNames: string[] = [];
  if (ctx.currentBindings) ctx.currentBindings.forEach(function(_b: unknown, n: string) { existingNames.push(n); });
  const pairsText = parts.join(', ');
  const suggested = (plan.name || 'domain') + '_domain';
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
