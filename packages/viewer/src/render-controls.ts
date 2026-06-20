// @flatppl/viewer — preset + domain control popovers —
//
// buildPresetControl / buildDomainControl construct the
// codicon-buttoned popover UIs above the plot pane that let users
// override per-binding preset values / cartprod ranges, persist them
// to source, or reset to the source's declared values.

import type { Ctx } from './types';
import { MLE_PRESET, computeAutoValues, hasDomainOverrides, hasOverrides, setDomainOverrideFor, setOverrideFor } from './overrides.js';
import { canPersistActive, canPersistDomain, persistActive, persistDomain } from './persist.js';
import { makeActionButton } from './render-frame.js';
import { domainBoundsText, presetValuesText } from './util.js';

type ControlEntry = {
  name: string | null;
  modified: boolean;
  shortLabel: string;
  longLabel: string;
};
type OutsideClickHandler = ((ev: MouseEvent) => void) | null;

/**
 * The labelled `auto (MLE)` entry for likelihood plots, present only once the
 * background optimiser (populateModeCache) has a mode for this binding. On
 * failure / timeout the cache entry is 'failed' (or absent) and this returns
 * null, so the option simply doesn't appear and the normal prior-draw `auto`
 * stays the pivot. The MLE is a computed *base*, override-able like any preset
 * (a dragged point shows `auto (MLE) (modified)` with a reset button).
 */
function mleControlEntry(ctx: Ctx, plan: any): ControlEntry | null {
  const sig = plan && plan.signature;
  if (!sig || sig.obsIR == null) return null;            // likelihoods only
  const cached = ctx.modeCenterCache && ctx.modeCenterCache.get(plan.name);
  if (!cached || cached.status !== 'ready' || !cached.values) return null;
  const override = ctx.presetOverrides.get(MLE_PRESET);
  const modified = !!(override && override.values
    && Object.keys(override.values).length > 0);
  const combined = Object.assign({}, cached.values, (override && override.values) || {});
  const tag = modified ? ' (modified)' : '';
  return {
    name: MLE_PRESET,
    modified: modified,
    shortLabel: 'auto (MLE)' + tag,
    longLabel: 'auto (MLE)' + tag + ': ' + presetValuesText(combined),
  };
}

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

  function buildEntry(name: string | null, baseValues: any, isAuto: boolean): ControlEntry {
    const entryOverride = (name == null) ? plan.autoOverride : ctx.presetOverrides.get(name);
    const modified = !!(entryOverride && entryOverride.values
      && Object.keys(entryOverride.values).length > 0);
    const combined = Object.assign({}, baseValues, (entryOverride && entryOverride.values) || {});
    const displayName = isAuto ? 'auto' : name;
    const tag = modified ? ' (modified)' : '';
    return {
      name: name,
      modified: modified,
      shortLabel: displayName + tag,
      longLabel: displayName + tag + ': ' + presetValuesText(combined),
    };
  }

  entries.push(buildEntry(null, autoValues, true));
  // Labelled `auto (MLE)` sits right after `auto` when the background
  // optimiser has a mode for this likelihood; absent otherwise.
  const mleEntry = mleControlEntry(ctx, plan);
  if (mleEntry) entries.push(mleEntry);
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
