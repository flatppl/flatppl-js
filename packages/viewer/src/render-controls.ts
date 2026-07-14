// @flatppl/viewer — preset + domain control popovers —
//
// buildPresetControl / buildDomainControl construct the
// codicon-buttoned popover UIs above the plot pane that let users
// override per-binding preset values / cartprod ranges, persist them
// to source, or reset to the source's declared values.

import type { Ctx } from './types';
import { computeAutoValues, hasDomainOverrides, hasOverrides, setDomainOverrideFor, setOverrideFor } from './overrides.js';
import { canPersistActive, canPersistDomain, persistActive, persistDomain } from './persist.js';
import { makeActionButton } from './render-frame.js';
import { domainBoundsText, presetValuesText } from './util.js';
import { clearForwardCaches, pinForwardSeed, rerollForwardSeed } from './forward-draws.js';

type ControlEntry = {
  name: string | null;
  modified: boolean;
  shortLabel: string;
  longLabel: string;
};
type OutsideClickHandler = ((ev: MouseEvent) => void) | null;

// `trailing`, when given, is placed immediately right of the dropdown and
// LEFT of the reset/save action group — so a caller's action button (e.g. the
// profile plot's ⛰ find-max) sits next to the selector, not after the icons.
export function buildPresetControl(ctx: Ctx, plan: any, onChange: () => void, trailing?: Node): DocumentFragment {
  const frag = document.createDocumentFragment();
  if (!plan.axes || plan.axes.length === 0) return frag;

  // One row per preset name (auto plus each named preset). The
  // "(modified)" tag is appended to the label when the active
  // override entry has values — there's no separate
  // "<name> (modified)" row anymore. Switching presets just
  // changes plan.presetName; the override store decides whether
  // the row reads as modified.
  const entries: ControlEntry[] = [];
  const presets = plan.matchedPresets || [];
  const autoValues = computeAutoValues(ctx, plan);

  function buildEntry(name: string | null, baseValues: any, isAuto: boolean, autoIsMle?: boolean): ControlEntry {
    const entryOverride = (name == null) ? plan.autoOverride : ctx.presetOverrides.get(name);
    const modified = !!(entryOverride && entryOverride.values
      && Object.keys(entryOverride.values).length > 0);
    const combined = Object.assign({}, baseValues, (entryOverride && entryOverride.values) || {});
    // One `auto` row; labelled `auto (MLE)` when its value is the converged MLE
    // (likelihood mode), plain `auto` otherwise — never both.
    const displayName = isAuto ? (autoIsMle ? 'auto (MLE)' : 'auto') : name;
    const tag = modified ? ' (modified)' : '';
    return {
      name: name,
      modified: modified,
      shortLabel: displayName + tag,
      longLabel: displayName + tag + ': ' + presetValuesText(combined),
    };
  }

  // Single `auto` entry. Its value is the converged MLE (mode) when the
  // background optimiser is ready for this likelihood, else the prior-draw
  // auto — matching the actual default pivot (overrides.baseValuesFor). There
  // is no separate `auto (MLE)` row: one auto, labelled (MLE) when MLE-backed.
  const mleCache = ctx.modeCenterCache && ctx.modeCenterCache.get(plan.name);
  const autoIsMle = !!(plan.signature && (plan.signature.obsIR != null || plan.signature.terms)
    && mleCache && mleCache.status === 'ready' && mleCache.values);
  entries.push(buildEntry(null, autoIsMle ? mleCache.values : autoValues, true, autoIsMle));
  for (let pi = 0; pi < presets.length; pi++) {
    entries.push(buildEntry(presets[pi].name, presets[pi].values || {}, false));
  }

  function isActive(entry: ControlEntry): boolean { return entry.name === plan.presetName; }
  let activeEntry: ControlEntry | null = null;
  for (let k = 0; k < entries.length; k++) {
    if (isActive(entries[k])) { activeEntry = entries[k]; break; }
  }
  if (!activeEntry) activeEntry = entries[0];

  const wrap = document.createElement('span');
  wrap.style.position = 'relative';
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.3em';

  const lbl = document.createElement('label');
  lbl.textContent = 'Inputs:';
  lbl.style.opacity = '0.6';

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
  btn.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
  btn.style.border = '1px solid var(--vscode-dropdown-border, #555)';
  btn.style.padding = '2px 6px';
  btn.style.fontSize = '1em';
  btn.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
  btn.style.cursor = 'pointer';
  btn.style.borderRadius = '2px';
  btn.textContent = activeEntry.shortLabel + '  ▾';
  btn.title = activeEntry.longLabel;

  const panel = document.createElement('div');
  panel.style.position = 'absolute';
  panel.style.top = 'calc(100% + 4px)';
  panel.style.left = '0';
  panel.style.zIndex = '50';
  panel.style.minWidth = '100%';
  panel.style.maxHeight = '20em';
  panel.style.overflowY = 'auto';
  panel.style.padding = '0.2em';
  panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
  panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
  panel.style.borderRadius = '3px';
  panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
  panel.style.display = 'none';
  panel.style.whiteSpace = 'nowrap';

  let outsideClickHandler: OutsideClickHandler = null;
  function closePanel(): void {
    panel.style.display = 'none';
    if (outsideClickHandler) {
      document.removeEventListener('mousedown', outsideClickHandler);
      outsideClickHandler = null;
    }
  }
  function openPanel(): void {
    panel.style.display = 'block';
    // Defer the outside-click attach so the same click that
    // opened the panel doesn't immediately close it.
    setTimeout(function() {
      outsideClickHandler = function(ev: MouseEvent) {
        if (!wrap.contains(ev.target as Node)) closePanel();
      };
      document.addEventListener('mousedown', outsideClickHandler);
    }, 0);
  }

  btn.addEventListener('click', function(ev) {
    ev.stopPropagation();
    if (panel.style.display === 'none') openPanel(); else closePanel();
  });

  entries.forEach(function(entry) {
    const row = document.createElement('div');
    row.textContent = entry.longLabel;
    row.style.padding = '0.25em 0.6em';
    row.style.cursor = 'pointer';
    row.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
    row.style.borderRadius = '2px';
    if (isActive(entry)) {
      row.style.background = 'rgba(13, 113, 199, 0.45)';
      row.style.color = '#fff';
    }
    row.addEventListener('mouseenter', function() {
      if (!isActive(entry)) row.style.background = 'rgba(255,255,255,0.06)';
    });
    row.addEventListener('mouseleave', function() {
      if (!isActive(entry)) row.style.background = '';
    });
    row.addEventListener('click', function(ev) {
      ev.stopPropagation();
      plan.presetName = entry.name;
      closePanel();
      onChange();
    });
    panel.appendChild(row);
  });

  wrap.appendChild(lbl);
  wrap.appendChild(btn);
  wrap.appendChild(panel);

  frag.appendChild(wrap);
  if (trailing) frag.appendChild(trailing);

  // Reset / save action buttons live in a tight inline-flex group
  // so the two icons read as a single control rather than each
  // inheriting the toolbar's wider gap.
  // Reset button — visible only when the active selection has
  // overrides. Clears the override entry (auto → plan.autoOverride
  // = null; named → presetOverrides.delete(name)) and re-renders
  // through onChange. The dropdown row's "(modified)" tag then
  // disappears with no further user action.
  if (hasOverrides(ctx, plan)) {
    const actionGroup = document.createElement('span');
    actionGroup.style.display = 'inline-flex';
    actionGroup.style.gap = '2px';
    const resetBtn = makeActionButton(ctx, 'discard', 'Reset preset to source values');
    resetBtn.addEventListener('click', function(ev) {
      ev.stopPropagation();
      setOverrideFor(ctx, plan, null);
      onChange();
    });
    actionGroup.appendChild(resetBtn);

    // Persist button — visible when the active selection is a
    // named preset with overrides AND the host supports
    // writing (web edit-mode on, or VS Code) AND the source RHS
    // is preset(<kwarg>=<literal>, …) with no non-literal
    // values. Hidden otherwise so the user never sees a
    // disabled-looking button.
    //   - named preset + overrides → 'save'    (overwrite RHS)
    //   - auto + overrides         → 'save-as' (append new binding)
    // canPersistActive enforces the host-capability split: 'save'
    // needs host.editSource; 'save-as' additionally needs
    // host.promptForName.
    if (canPersistActive(ctx, plan)) {
      const isSaveAs = (plan.presetName == null);
      const persistBtn = makeActionButton(ctx, 
        isSaveAs ? 'save-as' : 'save',
        isSaveAs
          ? 'Save as new preset binding'
          : 'Save overrides into preset'
      );
      persistBtn.addEventListener('click', function(ev) {
        ev.stopPropagation();
        persistActive(ctx, plan);
      });
      actionGroup.appendChild(persistBtn);
    }
    frag.appendChild(actionGroup);
  }
  return frag;
}

export function buildDomainControl(ctx: Ctx, plan: any, onChange: () => void, trailing?: Node): DocumentFragment {
  const frag = document.createDocumentFragment();
  if (!plan.axes || plan.axes.length === 0) return frag;

  const domains = plan.matchedDomains || [];
  // kwarg display order: take it from plan.signature.inputs so
  // every entry — including modifications — reads in the same
  // order as the source signature, regardless of which kwargs
  // got user overrides.
  const inputs = (plan.signature && plan.signature.inputs) || [];
  const kwargOrder: string[] = [];
  for (let ki = 0; ki < inputs.length; ki++) {
    if (inputs[ki].kwargName) kwargOrder.push(inputs[ki].kwargName);
  }

  function buildEntry(name: string | null, baseRanges: any, baseSetNames: any, isAuto: boolean): ControlEntry {
    const entryOverride = (name == null)
      ? plan.domainAutoOverride
      : ctx.domainOverrides.get(name);
    const modified = !!(entryOverride && entryOverride.ranges
      && Object.keys(entryOverride.ranges).length > 0);
    const combinedRanges = Object.assign({}, baseRanges,
      (entryOverride && entryOverride.ranges) || {});
    // User overrides shadow source named-set fields: drop those
    // entries from setNames so the kwarg renders with the
    // bounded interval rather than both.
    const combinedSetNames = Object.assign({}, baseSetNames);
    for (const k in combinedRanges) {
      if (Object.prototype.hasOwnProperty.call(combinedRanges, k)) {
        delete combinedSetNames[k];
      }
    }
    const displayName = isAuto ? 'auto' : name;
    const tag = modified ? ' (modified)' : '';
    return {
      name: name,
      modified: modified,
      shortLabel: displayName + tag,
      longLabel: displayName + tag + ': '
        + domainBoundsText(kwargOrder, combinedRanges, combinedSetNames),
    };
  }

  const entries: ControlEntry[] = [];
  entries.push(buildEntry(null, {}, {}, true));
  for (let di = 0; di < domains.length; di++) {
    entries.push(buildEntry(
      domains[di].name,
      domains[di].ranges || {},
      domains[di].setNames || {},
      false));
  }

  function isActive(entry: ControlEntry): boolean { return entry.name === plan.domainName; }
  let activeEntry: ControlEntry | null = null;
  for (let k = 0; k < entries.length; k++) {
    if (isActive(entries[k])) { activeEntry = entries[k]; break; }
  }
  if (!activeEntry) activeEntry = entries[0];

  const wrap = document.createElement('span');
  wrap.style.position = 'relative';
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.3em';

  const lbl = document.createElement('label');
  lbl.textContent = 'Domain:';
  lbl.style.opacity = '0.6';

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
  btn.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
  btn.style.border = '1px solid var(--vscode-dropdown-border, #555)';
  btn.style.padding = '2px 6px';
  btn.style.fontSize = '1em';
  btn.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
  btn.style.cursor = 'pointer';
  btn.style.borderRadius = '2px';
  btn.textContent = activeEntry.shortLabel + '  ▾';
  btn.title = activeEntry.longLabel;

  const panel = document.createElement('div');
  panel.style.position = 'absolute';
  panel.style.top = 'calc(100% + 4px)';
  panel.style.left = '0';
  panel.style.zIndex = '50';
  panel.style.minWidth = '100%';
  panel.style.maxHeight = '20em';
  panel.style.overflowY = 'auto';
  panel.style.padding = '0.2em';
  panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
  panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
  panel.style.borderRadius = '3px';
  panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
  panel.style.display = 'none';
  panel.style.whiteSpace = 'nowrap';

  let outsideClickHandler: OutsideClickHandler = null;
  function closePanel(): void {
    panel.style.display = 'none';
    if (outsideClickHandler) {
      document.removeEventListener('mousedown', outsideClickHandler);
      outsideClickHandler = null;
    }
  }
  function openPanel(): void {
    panel.style.display = 'block';
    setTimeout(function() {
      outsideClickHandler = function(ev: MouseEvent) {
        if (!wrap.contains(ev.target as Node)) closePanel();
      };
      document.addEventListener('mousedown', outsideClickHandler);
    }, 0);
  }

  btn.addEventListener('click', function(ev) {
    ev.stopPropagation();
    if (panel.style.display === 'none') openPanel(); else closePanel();
  });

  entries.forEach(function(entry) {
    const row = document.createElement('div');
    row.textContent = entry.longLabel;
    row.style.padding = '0.25em 0.6em';
    row.style.cursor = 'pointer';
    row.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
    row.style.borderRadius = '2px';
    if (isActive(entry)) {
      row.style.background = 'rgba(13, 113, 199, 0.45)';
      row.style.color = '#fff';
    }
    row.addEventListener('mouseenter', function() {
      if (!isActive(entry)) row.style.background = 'rgba(255,255,255,0.06)';
    });
    row.addEventListener('mouseleave', function() {
      if (!isActive(entry)) row.style.background = '';
    });
    row.addEventListener('click', function(ev) {
      ev.stopPropagation();
      plan.domainName = entry.name;
      closePanel();
      onChange();
    });
    panel.appendChild(row);
  });

  wrap.appendChild(lbl);
  wrap.appendChild(btn);
  wrap.appendChild(panel);
  frag.appendChild(wrap);
  if (trailing) frag.appendChild(trailing);

  // Reset / save / save-as icons mirror the Inputs control,
  // grouped in a tight inline-flex span so they read as one
  // pair rather than picking up the toolbar's wider gap.
  if (hasDomainOverrides(ctx, plan)) {
    const actionGroup = document.createElement('span');
    actionGroup.style.display = 'inline-flex';
    actionGroup.style.gap = '2px';
    const resetBtn = makeActionButton(ctx, 'discard', 'Reset domain to source ranges');
    resetBtn.addEventListener('click', function(ev) {
      ev.stopPropagation();
      setDomainOverrideFor(ctx, plan, null);
      onChange();
    });
    actionGroup.appendChild(resetBtn);

    if (canPersistDomain(ctx, plan)) {
      const isSaveAs = (plan.domainName == null);
      const persistBtn = makeActionButton(ctx, 
        isSaveAs ? 'save-as' : 'save',
        isSaveAs
          ? 'Save as new cartprod domain binding'
          : 'Save range overrides into cartprod'
      );
      persistBtn.addEventListener('click', function(ev) {
        ev.stopPropagation();
        persistDomain(ctx, plan);
      });
      actionGroup.appendChild(persistBtn);
    }
    frag.appendChild(actionGroup);
  }

  return frag;
}

// Global inference-backend selector for the header. Posterior (bayesupdate)
// measures default to importance sampling ('is'); this lets the user switch
// to the MCMC driver ('mh' / 'emcee'). Writes onto ctx.inferenceOpts.
//
// Sampling is DEFERRED behind an explicit "Sample" button: editing the sampler
// dropdown or any gear knob only mutates ctx.inferenceOpts and flags the button
// dirty — nothing re-samples until the user clicks Sample, which calls onChange
// (clears the measure cache and re-renders → re-draws). The same button re-runs
// with unchanged settings, so it doubles as a re-draw/re-roll. The engine reads
// ctx.inferenceOpts via the matCtx — see engine-facade.getMeasure.
export function buildInferenceControl(ctx: Ctx, onChange: () => void): HTMLElement {
  const opts = ctx.inferenceOpts;

  // `appliedSnapshot` records the config last committed via onChange, so
  // markDirty can tell a pending edit ("Sample ●", highlighted) from a clean
  // state ("Sample", used as a plain re-draw). Serialised over every knob.
  function snapshotOpts(): string {
    return JSON.stringify([opts.backend, opts.chains, opts.walkers, opts.warmup,
      opts.draws, opts.seed, opts.amisIters, opts.amisSamples,
      opts.smcParticles, opts.smcSteps, opts.smcCESS]);
  }
  let appliedSnapshot = snapshotOpts();

  function styleControl(el: HTMLElement) {
    el.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
    el.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
    el.style.border = '1px solid var(--vscode-dropdown-border, #555)';
    el.style.padding = '1px 4px';
    el.style.fontSize = '1em';
    el.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
    el.style.borderRadius = '2px';
    el.style.cursor = 'pointer';
  }

  const wrap = document.createElement('span');
  wrap.style.position = 'relative';
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.3em';

  const lbl = document.createElement('label');
  lbl.textContent = 'Sampler:';
  lbl.style.opacity = '0.6';

  const sel = document.createElement('select');
  for (const [v, t] of [['is', 'IS'], ['mh', 'MH'], ['emcee', 'emcee'], ['amis', 'AMIS'], ['smc', 'SMC'], ['elliptical-slice-sampler', 'ESS']]) {
    const o = document.createElement('option');
    o.value = v; o.textContent = t;
    sel.appendChild(o);
  }
  sel.value = opts.backend;
  styleControl(sel);
  sel.title = 'Posterior inference backend. IS = importance sampling (default); '
    + 'MH / emcee run MCMC; AMIS = adaptive multiple importance sampling; '
    + 'SMC = sequential Monte Carlo (robust on funnels; reports evidence); '
    + 'ESS = elliptical slice sampling (gradient- and tuning-free).';

  const gear = document.createElement('button');
  gear.type = 'button';
  gear.textContent = '⚙';
  gear.title = 'Sampler settings';
  styleControl(gear);

  // Advanced popup. Position:fixed (not absolute) so it escapes the header's
  // `overflow: hidden` clip and renders above the graph canvas's stacking
  // context; coordinates are set from the gear button's rect on open.
  const panel = document.createElement('div');
  panel.style.position = 'fixed';
  panel.style.zIndex = '9999';
  panel.style.padding = '0.5em';
  panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
  panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
  panel.style.borderRadius = '3px';
  panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
  panel.style.display = 'none';
  panel.style.fontSize = '0.9em';
  panel.style.minWidth = '12em';

  // Rows tagged with the backends they apply to, so the popup shows only the
  // knobs relevant to the current sampler (MCMC vs AMIS have different params).
  const rows: { el: HTMLElement; backends: string[] }[] = [];

  // One labelled number input row. `get` reads the current value, `set` writes
  // it back (empty string → null for the optional seed). `backends` lists which
  // samplers the row applies to. Returns the input so the caller can refresh it.
  function numRow(label: string, backends: string[], get: () => number | null, set: (v: number | null) => void,
                  opt?: { step?: number; min?: number; max?: number }): HTMLInputElement {
    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.justifyContent = 'space-between';
    row.style.alignItems = 'center';
    row.style.gap = '0.6em';
    row.style.margin = '0.15em 0';
    const rl = document.createElement('label');
    rl.textContent = label;
    rl.style.opacity = '0.7';
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.style.width = '5.5em';
    // Default integer step (counts); callers override for fractional knobs (the
    // spinner buttons then move by `step`, not to the nearest integer).
    inp.step = opt && opt.step != null ? String(opt.step) : '1';
    if (opt && opt.min != null) inp.min = String(opt.min);
    if (opt && opt.max != null) inp.max = String(opt.max);
    styleControl(inp);
    const cur = get();
    inp.value = cur == null ? '' : String(cur);
    inp.addEventListener('change', function () {
      const raw = inp.value.trim();
      set(raw === '' ? null : Number(raw));
      markDirty();
    });
    row.append(rl, inp);
    panel.appendChild(row);
    rows.push({ el: row, backends });
    return inp;
  }

  // The latent-count knob means "chains" for MH and "walkers" for emcee.
  const countRow = document.createElement('div');
  const countLabel = document.createElement('label');
  countLabel.style.opacity = '0.7';
  const countInput = document.createElement('input');
  countInput.type = 'number';
  countInput.style.width = '5.5em';
  styleControl(countInput);
  countRow.style.display = 'flex';
  countRow.style.justifyContent = 'space-between';
  countRow.style.alignItems = 'center';
  countRow.style.gap = '0.6em';
  countRow.style.margin = '0.15em 0';
  countRow.append(countLabel, countInput);

  function refreshCountRow() {
    if (opts.backend === 'emcee') {
      countLabel.textContent = 'walkers';
      countInput.value = opts.walkers == null ? '' : String(opts.walkers);
      countInput.placeholder = 'auto';
    } else {
      countLabel.textContent = 'chains';
      countInput.value = String(opts.chains);
      countInput.placeholder = '';
    }
  }
  countInput.addEventListener('change', function () {
    const raw = countInput.value.trim();
    if (opts.backend === 'emcee') opts.walkers = raw === '' ? null : Number(raw);
    else opts.chains = raw === '' ? 4 : Number(raw);
    markDirty();
  });

  numRow('draws', ['mh', 'emcee', 'elliptical-slice-sampler'], function () { return opts.draws; }, function (v) { opts.draws = v == null ? 1000 : v; });
  numRow('warmup', ['mh', 'emcee', 'elliptical-slice-sampler'], function () { return opts.warmup; }, function (v) { opts.warmup = v == null ? 1000 : v; });
  panel.appendChild(countRow);
  rows.push({ el: countRow, backends: ['mh', 'emcee', 'elliptical-slice-sampler'] });
  numRow('iterations', ['amis'], function () { return opts.amisIters; }, function (v) { opts.amisIters = v == null ? 30 : v; });
  numRow('samples/iter', ['amis'], function () { return opts.amisSamples; }, function (v) { opts.amisSamples = v == null ? 300 : v; });
  numRow('particles', ['smc'], function () { return opts.smcParticles; }, function (v) { opts.smcParticles = v == null ? 2000 : v; });
  numRow('chain', ['smc'], function () { return opts.smcSteps; }, function (v) { opts.smcSteps = v == null ? 12 : v; });
  numRow('CESS ratio', ['smc'], function () { return opts.smcCESS; }, function (v) { opts.smcCESS = v == null ? 0.7 : v; },
    { step: 0.05, min: 0.05, max: 0.99 });
  numRow('seed', ['mh', 'emcee', 'amis', 'smc', 'elliptical-slice-sampler'], function () { return opts.seed; }, function (v) { opts.seed = v; });
  refreshCountRow();

  // IS has no sampler knobs — disable the gear so the advanced panel is clearly
  // inert in the default mode. For MH/emcee/AMIS, show only the rows that apply
  // to the selected backend.
  function refreshEnabled() {
    const isIS = opts.backend === 'is';
    gear.disabled = isIS;
    gear.style.opacity = isIS ? '0.4' : '1';
    gear.style.cursor = isIS ? 'default' : 'pointer';
    if (isIS) panel.style.display = 'none';
    for (const r of rows) r.el.style.display = r.backends.indexOf(opts.backend) >= 0 ? 'flex' : 'none';
  }

  let outside: ((ev: MouseEvent) => void) | null = null;
  function closePanel() {
    panel.style.display = 'none';
    if (outside) { document.removeEventListener('mousedown', outside); outside = null; }
  }
  gear.addEventListener('click', function () {
    if (gear.disabled) return;
    if (panel.style.display === 'none') {
      refreshCountRow();
      // Anchor the fixed popup under the gear's right edge (viewport coords).
      const r = gear.getBoundingClientRect();
      panel.style.top = (r.bottom + 4) + 'px';
      panel.style.right = Math.max(4, window.innerWidth - r.right) + 'px';
      panel.style.display = 'block';
      outside = function (ev: MouseEvent) {
        if (!wrap.contains(ev.target as Node)) closePanel();
      };
      document.addEventListener('mousedown', outside);
    } else {
      closePanel();
    }
  });

  sel.addEventListener('change', function () {
    opts.backend = sel.value;
    refreshEnabled();
    refreshCountRow();
    markDirty();
  });

  // Explicit Sample / re-draw button. Highlighted ("Sample ●") whenever the
  // pending config differs from what was last drawn; a plain "Sample" otherwise
  // (still clickable, to re-run / re-roll with the current settings).
  const applyBtn = document.createElement('button');
  applyBtn.type = 'button';
  applyBtn.textContent = 'Sample';
  styleControl(applyBtn);
  applyBtn.style.fontWeight = '600';
  function refreshApply() {
    const dirty = snapshotOpts() !== appliedSnapshot;
    applyBtn.textContent = dirty ? 'Sample ●' : 'Sample';
    if (dirty) {
      applyBtn.style.background = 'var(--vscode-button-background, #0e639c)';
      applyBtn.style.color = 'var(--vscode-button-foreground, #ffffff)';
      applyBtn.style.borderColor = 'var(--vscode-button-background, #0e639c)';
      applyBtn.title = 'Apply the changed sampler settings and draw';
    } else {
      applyBtn.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
      applyBtn.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
      applyBtn.style.borderColor = 'var(--vscode-dropdown-border, #555)';
      applyBtn.title = 'Re-run the sampler with the current settings (re-draw)';
    }
  }
  function markDirty() { refreshApply(); }
  applyBtn.addEventListener('click', function () {
    onChange();
    appliedSnapshot = snapshotOpts();
    refreshApply();
  });

  refreshEnabled();
  refreshApply();
  wrap.append(lbl, sel, gear, applyBtn, panel);
  return wrap;
}

// Forward-sampling draw control for the plot toolbar. Steers the forward
// draw budget (SAMPLE_COUNT), inner marginalization count (M), and the
// forward PRNG seed — the settings that drive every prior / tractable-measure
// histogram. DISTINCT from buildInferenceControl, which steers the POSTERIOR
// sampler (inferenceOpts) and is gated on bayesupdate.
//
// Editing an input is DEFERRED behind the Sample button (same idiom as
// buildInferenceControl): a change only stages the ctx field + flags the
// button dirty ("Sample ●"); nothing re-draws until the user clicks Sample.
// Sample doubles as re-roll — dirty → apply staged edits (clear caches +
// re-render); clean → rerollForwardSeed (bump seed) so the draw visibly
// changes despite the deterministic forward PRNG.
export function buildDrawControl(ctx: Ctx, onChange: () => void): HTMLElement {
  function styleControl(el: HTMLElement) {
    el.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
    el.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
    el.style.border = '1px solid var(--vscode-dropdown-border, #555)';
    el.style.padding = '1px 4px';
    el.style.fontSize = '1em';
    el.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
    el.style.borderRadius = '2px';
    el.style.cursor = 'pointer';
  }

  const wrap = document.createElement('span');
  wrap.style.position = 'relative';
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.3em';

  // PENDING (staged) values, local to this control instance. ctx holds only
  // APPLIED values — editing an input touches these locals, never ctx, so a
  // staged-but-unapplied edit cannot leak into an unrelated re-render. The
  // control is rebuilt from ctx on every re-render (it is a toolbar thunk), so
  // uncommitted edits are simply re-seeded from the applied ctx values then.
  let pendingDraws = ctx.SAMPLE_COUNT;
  let pendingSeed = ctx.rootSeed;
  let pendingM = ctx.MARGINALIZATION_COUNT;
  function dirty(): boolean {
    return pendingDraws !== ctx.SAMPLE_COUNT
      || pendingSeed !== ctx.rootSeed
      || pendingM !== ctx.MARGINALIZATION_COUNT;
  }
  function markDirty() {
    const d = dirty();
    sample.textContent = d ? 'Sample ●' : 'Sample';
    sample.style.fontWeight = d ? 'bold' : 'normal';
  }

  const lbl = document.createElement('label');
  lbl.textContent = 'Draws:';
  lbl.style.opacity = '0.6';

  const draws = document.createElement('input');
  draws.type = 'number';
  draws.min = '1';
  draws.step = '1000';
  draws.style.width = '6em';
  draws.value = String(pendingDraws);
  draws.title = 'Forward draw budget for prior / tractable-measure histograms. '
    + 'Takes effect on the next Sample.';
  styleControl(draws);
  // Stage only: update the pending local, never ctx. Revert on invalid input.
  draws.addEventListener('change', function () {
    const v = Number(draws.value);
    if (Number.isFinite(v) && v > 0) pendingDraws = v | 0;
    draws.value = String(pendingDraws);
    markDirty();
  });

  const seedLbl = document.createElement('label');
  seedLbl.textContent = 'seed';
  seedLbl.style.opacity = '0.6';

  const seed = document.createElement('input');
  seed.type = 'number';
  seed.step = '1';
  seed.style.width = '4.5em';
  seed.value = String(pendingSeed);
  seed.title = 'Forward-sampling seed. Pinning it makes draws reproducible '
    + '(and the whole view, when no posterior seed is set separately). '
    + 'Takes effect on the next Sample.';
  styleControl(seed);
  seed.addEventListener('change', function () {
    const raw = seed.value.trim();
    if (raw !== '' && Number.isFinite(Number(raw))) pendingSeed = Number(raw) | 0;
    seed.value = String(pendingSeed);
    markDirty();
  });

  const sample = document.createElement('button');
  sample.type = 'button';
  sample.textContent = 'Sample';
  sample.title = 'Apply the draw settings and re-render. With no pending change, '
    + 're-rolls the forward seed to draw a fresh sample.';
  styleControl(sample);
  sample.addEventListener('click', function () {
    if (dirty()) {
      // Commit the staged pending values to ctx, then invalidate + re-render.
      ctx.SAMPLE_COUNT = pendingDraws;
      ctx.MARGINALIZATION_COUNT = pendingM;
      pinForwardSeed(ctx, pendingSeed);   // sets rootSeed (+ mirrors null posterior seed)
      clearForwardCaches(ctx);
    } else {
      // Clean → re-roll: bump the seed so the draw visibly changes.
      pendingSeed = rerollForwardSeed(ctx);  // bumps rootSeed + clears caches
      seed.value = String(pendingSeed);
    }
    markDirty();
    onChange();
  });

  const gear = document.createElement('button');
  gear.type = 'button';
  gear.textContent = '⚙';
  gear.title = 'Advanced draw settings';
  styleControl(gear);

  const panel = document.createElement('div');
  panel.style.position = 'fixed';
  panel.style.zIndex = '9999';
  panel.style.padding = '0.5em';
  panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
  panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
  panel.style.borderRadius = '3px';
  panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
  panel.style.display = 'none';
  panel.style.fontSize = '0.9em';
  panel.style.minWidth = '12em';

  const mRow = document.createElement('div');
  mRow.style.display = 'flex';
  mRow.style.justifyContent = 'space-between';
  mRow.style.alignItems = 'center';
  mRow.style.gap = '0.6em';
  const mLbl = document.createElement('label');
  mLbl.textContent = 'marginalization M';
  mLbl.style.opacity = '0.7';
  const mInput = document.createElement('input');
  mInput.type = 'number';
  mInput.min = '1';
  mInput.step = '1';
  mInput.style.width = '5.5em';
  mInput.value = String(pendingM);
  mInput.title = 'Monte Carlo count for marginalizing internal latent draws. '
    + 'Takes effect on the next Sample.';
  styleControl(mInput);
  // Stage only (same as Draws/seed): update the pending local; commit on Sample.
  mInput.addEventListener('change', function () {
    const v = Number(mInput.value);
    if (Number.isFinite(v) && v >= 1) pendingM = v | 0;
    mInput.value = String(pendingM);
    markDirty();
  });
  mRow.append(mLbl, mInput);
  panel.appendChild(mRow);

  gear.addEventListener('click', function () {
    if (panel.style.display === 'none') {
      const r = gear.getBoundingClientRect();
      panel.style.left = String(Math.round(r.left)) + 'px';
      panel.style.top = String(Math.round(r.bottom + 4)) + 'px';
      panel.style.display = 'block';
    } else {
      panel.style.display = 'none';
    }
  });

  wrap.append(lbl, draws, seedLbl, seed, sample, gear, panel);
  return wrap;
}
