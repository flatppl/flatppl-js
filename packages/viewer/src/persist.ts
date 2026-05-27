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
    // Canonical FlatPPL boolean spelling (spec §05).
    return v ? 'true' : 'false';
  }
  if (!Number.isFinite(v)) return String(v);
  return String(v);
}

/**
 * Serialise a preset-point value into its FlatPPL source form.
 * Scalars round-trip through `formatScalarForSource`. Arrays
 * (plain Array or Float64Array) emit `[v1, v2, …]` literals so
 * an array-typed callable input (e.g. `theta` of shape `[J]`)
 * persists as `theta = [v1, …, vJ]` rather than dropping the
 * field or stringifying as `"1,2,3"`.
 */
export function formatValueForSource(ctx: Ctx, v: unknown): string | null {
  if (typeof v === 'boolean' || typeof v === 'number') {
    if (!Number.isFinite(v) && typeof v === 'number') return null;
    return formatScalarForSource(ctx, v);
  }
  if (Array.isArray(v) || (v != null && typeof v === 'object'
      && typeof (v as any).length === 'number'
      && (typeof (v as any).BYTES_PER_ELEMENT === 'number'
          || (v as any)[0] !== undefined))) {
    const parts: string[] = [];
    const len = (v as any).length;
    for (let i = 0; i < len; i++) {
      const e = (v as any)[i];
      if (typeof e !== 'number' || !Number.isFinite(e)) return null;
      parts.push(formatScalarForSource(ctx, e));
    }
    return '[' + parts.join(', ') + ']';
  }
  return null;
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
    if (v.type === 'NumberLiteral' || v.type === 'BoolLiteral') continue;
    // Array literal: every element must itself be a numeric literal
    // (no nested refs / calls — the preset must be self-contained
    // for edit-in-place persistence).
    if (v.type === 'ArrayLiteral' && Array.isArray(v.elements)) {
      let allLits = true;
      for (const el of v.elements) {
        if (!el || (el.type !== 'NumberLiteral' && el.type !== 'BoolLiteral')) {
          allLits = false; break;
        }
      }
      if (allLits) continue;
    }
    return false;
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
    let text: string;
    if (override) {
      // Override may be a scalar or an array (array-typed inputs
      // persist via formatValueForSource as `[v1, …]`).
      const t = formatValueForSource(ctx, active.values[kwarg]);
      text = (t != null)
        ? t
        // Fall through to existing scalar path if value isn't
        // representable (defensive — shouldn't happen for valid
        // override entries; the toolbar dropdown only writes
        // representable values).
        : formatScalarForSource(ctx, active.values[kwarg]);
    } else if (innerSrc && innerSrc.type === 'ArrayLiteral'
               && Array.isArray(innerSrc.elements)) {
      // Source-form array literal — preserve it verbatim by
      // formatting each element.
      const elemTexts: string[] = [];
      for (const el of innerSrc.elements) {
        if (el && (el.type === 'NumberLiteral' || el.type === 'BoolLiteral')) {
          elemTexts.push(formatScalarForSource(ctx, el.value));
        } else {
          elemTexts.push('null');   // unreachable per canPersistActive
        }
      }
      text = '[' + elemTexts.join(', ') + ']';
    } else {
      text = formatScalarForSource(ctx, innerSrc && innerSrc.value);
    }
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
    const text = formatValueForSource(ctx, combined[k]);
    if (text == null) continue;   // unrepresentable value — skip kwarg
    parts.push(k + ' = ' + text);
  }
  if (parts.length === 0) return;
  const existingNames: string[] = [];
  if (ctx.currentBindings) ctx.currentBindings.forEach(function(_b: unknown, n: string) { existingNames.push(n); });
  const pairsText = parts.join(', ');
  const suggested = (plan.name || 'inputs') + '_input';
  Promise.resolve(ctx.host.promptForName({
    suggested: suggested,
    existingNames: existingNames,
  })).then(function(name) {
    if (!name) return;
    ctx.pendingPresetName = name;
    // The save-as promotes the auto-pseudo-preset's values into a
    // named binding; the original autoOverride is now redundant
    // (the named preset captures it). Clear it so the toolbar
    // dropdown stops showing "auto (modified)" once the source
    // refresh switches to the new named preset. If editSource
    // never lands the user's auto-edits are also cleared, which
    // is acceptable: the unsaved state was the override; the
    // saved state is the persisted record.
    plan.autoOverride = null;
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
  if (v.type === 'CallExpr' && v.callee && v.callee.name === 'cartpow'
      && Array.isArray(v.args) && v.args.length === 2) {
    // cartpow(<base>, <n>) — array-typed input. Round-trip the base
    // recursively (interval / named set) and the int-literal count.
    return 'cartpow(' + setFieldToSource(ctx, v.args[0]) + ', '
      + formatScalarForSource(ctx, v.args[1].value) + ')';
  }
  // interval(NumLit, NumLit)
  return 'interval('
    + formatScalarForSource(ctx, v.args[0].value) + ', '
    + formatScalarForSource(ctx, v.args[1].value) + ')';
}

export function defaultSetSourceForKwarg(ctx: Ctx, plan: any, kwargName: string): string {
  // Engine's `computeAutoDomain` resolves the structured descriptor
  // (including cartpow wrap for array-typed inputs); this function
  // is now a pure formatter on top.
  if (!plan || !plan.signature) return 'reals';
  const bindings = ctx.derivationsState && ctx.derivationsState.bindings;
  const dom = FlatPPLEngine.orchestrator.computeAutoDomain(plan.signature, bindings);
  const descriptor = dom && dom[kwargName];
  if (!descriptor) return 'reals';
  return formatSetDescriptor(ctx, descriptor);
}

/**
 * Format an engine `SetDescriptor` (the return shape of
 * `resolveAxisBaseSet` / `computeAutoDomain`) into FlatPPL source
 * text. Pure formatter — keeps the viewer's source-emission
 * concerns separate from the engine's resolution logic.
 *
 * Descriptor kinds (mirrors engine spec):
 *   reals / posreals / nonnegreals / integers / posintegers /
 *   nonnegintegers / booleans                          → bareword
 *   interval (lo=0, hi=1)                              → unitinterval
 *   interval (lo, hi)                                  → interval(lo, hi)
 *   cartpow {base, n}                                  → cartpow(<base>, n)
 *   empirical / unknown                                → 'reals' (safe fallback)
 */
function formatSetDescriptor(ctx: Ctx, d: any): string {
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
    case 'cartpow':
      return 'cartpow(' + formatSetDescriptor(ctx, d.base) + ', '
        + formatScalarForSource(ctx, d.n) + ')';
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
      const r = or[kwarg];
      // Preserve the source field's array shape: if the original
      // field is `cartpow(<base>, n)`, the override (which is a
      // single interval per the linked-domain semantics) updates
      // the base while keeping the cartpow wrap. Otherwise emit
      // bare `interval(lo, hi)`.
      if (sa.value && sa.value.type === 'CallExpr' && sa.value.callee
          && sa.value.callee.name === 'cartpow'
          && Array.isArray(sa.value.args) && sa.value.args.length === 2
          && sa.value.args[1].type === 'NumberLiteral') {
        parts.push(kwarg + ' = cartpow(interval('
          + formatScalarForSource(ctx, r.lo) + ', '
          + formatScalarForSource(ctx, r.hi) + '), '
          + formatScalarForSource(ctx, sa.value.args[1].value) + ')');
      } else {
        parts.push(kwarg + ' = interval('
          + formatScalarForSource(ctx, r.lo) + ', '
          + formatScalarForSource(ctx, r.hi) + ')');
      }
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
  // Pre-resolve the engine's auto-domain descriptors. For array-
  // typed kwargs the descriptor is `{kind:'cartpow', base, n}`;
  // when a range override or cache hit applies to the kwarg we
  // substitute its `base` with the interval rather than collapsing
  // to a bare `interval(...)` (which would lose the array shape).
  const bindings = ctx.derivationsState && ctx.derivationsState.bindings;
  const autoDomain = FlatPPLEngine.orchestrator.computeAutoDomain(
    plan.signature, bindings) || {};
  const inputs = (plan.signature && plan.signature.inputs) || [];
  const parts: string[] = [];
  function formatRangeForKwarg(kw: string, lo: number, hi: number): string {
    const desc = autoDomain[kw];
    const intervalDesc: any =
      (lo === 0 && hi === 1) ? { kind: 'interval', lo: 0, hi: 1 }
                             : { kind: 'interval', lo, hi };
    if (desc && desc.kind === 'cartpow') {
      // Linked-cartpow domain: edit propagates to every slot. The
      // wrap preserves the source's array shape per spec §03.
      return formatSetDescriptor(ctx, {
        kind: 'cartpow', base: intervalDesc, n: desc.n,
      });
    }
    return formatSetDescriptor(ctx, intervalDesc);
  }
  for (let i = 0; i < inputs.length; i++) {
    const kw = inputs[i].kwargName;
    if (!kw) continue;
    const r = Object.prototype.hasOwnProperty.call(ranges, kw) ? ranges[kw] : null;
    if (r && Number.isFinite(r.lo) && Number.isFinite(r.hi)) {
      parts.push(kw + ' = ' + formatRangeForKwarg(kw, r.lo, r.hi));
      continue;
    }
    const cached = ctx.profileRangeCache.get(
      plan.name + '|' + kw + '|D=' + (plan.domainName || ''));
    if (cached && Number.isFinite(cached.lo) && Number.isFinite(cached.hi)) {
      parts.push(kw + ' = ' + formatRangeForKwarg(kw, cached.lo, cached.hi));
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
    // Same rationale as the preset save-as: promotes the auto-
    // domain override into a named binding; clear plan.domainAuto-
    // Override so the dropdown stops showing "auto (modified)"
    // once the source refresh switches to the new named domain.
    plan.domainAutoOverride = null;
    ctx.host.editSource({
      range: null,
      newText: name + ' = cartprod(' + pairsText + ')',
    });
  }).catch(function(err) {
    console.error('[viewer] persistAutoDomainAsNewBinding failed:', err);
  });
}
