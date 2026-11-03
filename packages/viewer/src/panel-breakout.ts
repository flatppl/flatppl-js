// @ts-check
// @flatppl/viewer — seams for the web playground's pane breakout —
//
// Everything here is inert unless the host asks for it: the Pop out control
// exists only when the host adapter defines detachPanel, a placement changes
// only through the handle's setPanelPlacement, and snapshots are taken only
// on request. The VS Code webview does none of these, so its view is the one
// pinned by vscode-view-unchanged.test.ts.

import { PANEL_DOM, PANEL_ORDER } from './panels.js';
import type { PanelId, PanelPlacement } from './panels.js';
import { applyPanelLayout, setGraphEnabled, setMathEnabled, setPlotEnabled } from './render-frame.js';
import { updatePlotForBinding } from './render-plot.js';
import { renderDAG, updateBackBtn } from './dag.js';
import { rememberPlanSelections } from './overrides.js';
import { clearForwardCaches, setMarginalizationCount, setSampleCount } from './forward-draws.js';
import { $ } from './util.js';
import type { Ctx, PanelSnapshot } from './types';

const BREAKOUT_CSS = `
#view-controls { position: relative; }
#detach-btn {
  background: var(--vscode-button-secondaryBackground, #3a3d41);
  color: var(--vscode-button-secondaryForeground, #ccc);
  border: 1px solid var(--vscode-button-border, transparent);
  border-radius: 3px;
  padding: 2px 10px;
  font-size: 12px;
  cursor: pointer;
  font-family: var(--vscode-font-family, sans-serif);
  flex-shrink: 0;
}
#detach-btn:hover { background: var(--vscode-button-secondaryHoverBackground, #505355); }
#detach-btn:disabled { opacity: 0.5; cursor: default; }
#detach-menu {
  position: absolute; right: 0; top: calc(100% + 4px); z-index: 20;
  min-width: 14em; padding: 4px 0;
  background: var(--vscode-menu-background, #252526);
  color: var(--vscode-menu-foreground, #ccc);
  border: 1px solid var(--vscode-menu-border, #454545);
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
}
#detach-menu[hidden] { display: none; }
#detach-menu [role="menuitem"] {
  display: block; width: 100%; text-align: left;
  padding: 4px 12px; border: 0; background: none; color: inherit;
  font: inherit; font-size: 12px; cursor: pointer;
}
#detach-menu [role="menuitem"]:hover,
#detach-menu [role="menuitem"]:focus { background: var(--vscode-menu-selectionBackground, #094771); outline: none; }
#graph-toggle.detached, #plot-toggle.detached, #math-toggle.detached { font-style: italic; }
.viewer-sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}
`;

/** Inject the breakout styles once per document, and only for hosts that
 *  use the breakout, so the VS Code stylesheet stays byte-identical. */
function ensureBreakoutCss(): void {
  if (document.head.querySelector('style[data-flatppl-breakout-css]')) return;
  const style = document.createElement('style');
  style.setAttribute('data-flatppl-breakout-css', '');
  style.textContent = BREAKOUT_CSS;
  document.head.appendChild(style);
}

function allowed(ctx: Ctx, id: PanelId): boolean {
  return !ctx.panelAllowed || ctx.panelAllowed.has(id);
}

function placementOf(ctx: Ctx, id: PanelId): PanelPlacement {
  return ctx.panelPlacement?.[id] === 'detached' ? 'detached' : 'docked';
}

/** Panels the Pop out menu offers: allowed here and shown here. */
function detachable(ctx: Ctx): PanelId[] {
  return PANEL_ORDER.filter((id) => allowed(ctx, id) && placementOf(ctx, id) === 'docked');
}

/**
 * Add the Pop out menu button to #view-controls when the host can detach
 * panels. Returns its teardown, or a no-op when the host cannot.
 * Keyboard: Enter, Space or ArrowDown open the menu on its first item;
 * ArrowUp/ArrowDown/Home/End move; Escape closes and returns focus; Tab
 * closes.
 */
export function installDetachControl(ctx: Ctx): () => void {
  if (!ctx.host || typeof ctx.host.detachPanel !== 'function') return function () {};
  ensureBreakoutCss();
  const controls = $('view-controls');
  const button = document.createElement('button');
  button.id = 'detach-btn';
  button.type = 'button';
  button.textContent = 'Pop out';
  button.setAttribute('aria-haspopup', 'menu');
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-controls', 'detach-menu');
  button.setAttribute('title', 'Open a panel in a new window');
  const menu = document.createElement('div');
  menu.id = 'detach-menu';
  menu.setAttribute('role', 'menu');
  menu.setAttribute('aria-label', 'Open a panel in a new window');
  menu.hidden = true;
  controls.appendChild(button);
  controls.appendChild(menu);

  function items(): HTMLElement[] {
    return Array.from(menu.querySelectorAll('[role="menuitem"]')) as HTMLElement[];
  }
  function close(returnFocus: boolean): void {
    if (menu.hidden) return;
    menu.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    document.removeEventListener('pointerdown', onOutside, true);
    if (returnFocus) button.focus();
  }
  function open(): void {
    menu.replaceChildren();
    for (const id of detachable(ctx)) {
      const item = document.createElement('button');
      item.type = 'button';
      item.setAttribute('role', 'menuitem');
      item.setAttribute('tabindex', '-1');
      item.setAttribute('data-panel', id);
      item.textContent = 'Open ' + PANEL_DOM[id].label + ' in a new window';
      menu.appendChild(item);
    }
    if (!items().length) return;
    menu.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    document.addEventListener('pointerdown', onOutside, true);
    items()[0].focus();
  }
  function onOutside(ev: Event): void {
    const t = ev.target as Node | null;
    if (t && (menu.contains(t) || button.contains(t))) return;
    close(false);
  }
  function choose(id: PanelId): void {
    close(false);
    // The panel leaves this window, so focus goes to its toggle.
    try { ctx.host.detachPanel!(id); } finally { $(PANEL_DOM[id].toggle).focus(); }
  }
  function onButtonClick(): void { if (menu.hidden) open(); else close(true); }
  function onButtonKey(ev: KeyboardEvent): void {
    if (ev.key === 'ArrowDown' && menu.hidden) { ev.preventDefault(); open(); }
  }
  function onMenuClick(ev: Event): void {
    const item = (ev.target as HTMLElement | null)?.closest('[role="menuitem"]') as HTMLElement | null;
    const id = item && item.getAttribute('data-panel');
    if (id) choose(id as PanelId);
  }
  function onMenuKey(ev: KeyboardEvent): void {
    const list = items();
    const i = list.indexOf(document.activeElement as HTMLElement);
    let next = -1;
    if (ev.key === 'ArrowDown') next = (i + 1) % list.length;
    else if (ev.key === 'ArrowUp') next = (i - 1 + list.length) % list.length;
    else if (ev.key === 'Home') next = 0;
    else if (ev.key === 'End') next = list.length - 1;
    else if (ev.key === 'Escape') { ev.preventDefault(); close(true); return; }
    else if (ev.key === 'Tab') { close(false); return; }
    if (next >= 0) { ev.preventDefault(); list[next].focus(); }
  }
  button.addEventListener('click', onButtonClick);
  button.addEventListener('keydown', onButtonKey);
  menu.addEventListener('click', onMenuClick);
  menu.addEventListener('keydown', onMenuKey);
  ctx.refreshDetachControl = function () {
    (button as HTMLButtonElement).disabled = detachable(ctx).length === 0;
    if ((button as HTMLButtonElement).disabled) close(false);
  };
  ctx.refreshDetachControl();
  return function () {
    close(false);
    ctx.refreshDetachControl = null;
    button.remove();
    menu.remove();
  };
}

const SETTERS: Record<PanelId, (ctx: Ctx, on: boolean) => void> = {
  graph: setGraphEnabled,
  plot: setPlotEnabled,
  math: setMathEnabled,
};

/** Show a panel here again, or mark it as shown in another window. A
 *  detached panel is off in this mount, so it costs nothing here. */
export function setPanelPlacement(ctx: Ctx, id: PanelId, placement: PanelPlacement): void {
  if (!PANEL_DOM[id]) throw new RangeError('unknown panel: ' + id);
  if (placement !== 'docked' && placement !== 'detached') throw new RangeError('unknown placement: ' + placement);
  if (!allowed(ctx, id)) throw new RangeError('panel ' + id + ' is not part of this mount');
  ensureBreakoutCss();
  ctx.panelPlacement = { ...(ctx.panelPlacement || {}), [id]: placement };
  SETTERS[id](ctx, placement === 'docked');
  applyPanelLayout(ctx);
  if (ctx.refreshDetachControl) ctx.refreshDetachControl();
}

/** The toggle of a detached panel asks the host to bring it back.
 *  Returns true when it handled the click. */
export function redockFromToggle(ctx: Ctx, id: PanelId): boolean {
  if (placementOf(ctx, id) !== 'detached') return false;
  if (ctx.host && typeof ctx.host.redockPanel === 'function') ctx.host.redockPanel(id);
  return true;
}

/** Plain data that lets another mount show this panel as it is now. */
export function snapshotPanel(ctx: Ctx, id: PanelId): PanelSnapshot {
  if (id === 'graph') {
    const cy = ctx.cy;
    return structuredClone({
      graph: {
        collapsedReifications: Array.from(ctx.collapsedReifications || []) as string[],
        viewport: cy ? { zoom: cy.zoom(), pan: { ...cy.pan() } } : null,
        history: (ctx.history || []).map((h: any) => ({ targetName: h.targetName, path: h.path ?? null })),
      },
    });
  }
  if (id === 'plot') {
    // Capture edits made to the current plan since it was built.
    rememberPlanSelections(ctx, ctx.currentPlotPlan);
    return structuredClone({
      plot: {
        sampleCount: ctx.SAMPLE_COUNT,
        marginalizationCount: ctx.MARGINALIZATION_COUNT,
        rejectionBudget: ctx.REJECTION_BUDGET,
        rootSeed: ctx.rootSeed,
        inferenceOpts: { ...ctx.inferenceOpts },
        recordSelection: ctx.recordSelection || null,
        presetOverrides: Array.from(ctx.presetOverrides || new Map()),
        domainOverrides: Array.from(ctx.domainOverrides || new Map()),
        planMemory: Array.from(ctx.planMemoryByName || new Map()),
      },
    }) as PanelSnapshot;
  }
  if (id === 'math') {
    const el = document.getElementById('math-content');
    return { math: { scrollTop: el ? el.scrollTop : 0 } };
  }
  throw new RangeError('unknown panel: ' + id);
}

/** Apply a snapshot from snapshotPanel, taken in this or another mount. */
export function restorePanel(ctx: Ctx, id: PanelId, snapshot: PanelSnapshot): void {
  if (id === 'graph' && snapshot.graph) restoreGraph(ctx, snapshot.graph);
  else if (id === 'plot' && snapshot.plot) restorePlot(ctx, snapshot.plot);
  else if (id === 'math' && snapshot.math) {
    const el = document.getElementById('math-content');
    if (el) el.scrollTop = snapshot.math.scrollTop;
  } else {
    throw new RangeError('snapshot has no state for panel ' + id);
  }
}

function restoreGraph(ctx: Ctx, g: NonNullable<PanelSnapshot['graph']>): void {
  // History entries need the model to rebuild their sub-DAGs; without it
  // Back would render an entry with no data.
  if (!ctx.currentBindings) {
    throw new RangeError('restorePanel(\'graph\') needs a model: call update() first');
  }
  ctx.collapsedReifications = new Set(g.collapsedReifications);
  g.collapsedReifications.forEach((name) => ctx._reifSeen.add(name));
  // Same-model entries get their sub-DAG back; a cross-module entry keeps
  // data null, and Back sends it through host.openModule as before.
  ctx.history = g.history.map(function (h) {
    let data = null;
    if (h.path === (ctx.currentPath ?? null)) {
      data = h.targetName === ctx.MODULE_TARGET
        ? FlatPPLEngine.computeFullDAG(ctx.currentBindings, { linkedBindings: ctx.currentLinkedBindings })
        : FlatPPLEngine.computeSubDAG(ctx.currentBindings, h.targetName, { linkedBindings: ctx.currentLinkedBindings });
    }
    return { targetName: h.targetName, path: h.path, data };
  });
  if (ctx.currentState && ctx.cy) renderDAG(ctx, ctx.currentState.data);
  if (ctx.cy && g.viewport) {
    ctx.cy.zoom(g.viewport.zoom);
    ctx.cy.pan(g.viewport.pan);
    ctx.graphAutoFit = false;
  }
  updateBackBtn(ctx);
}

function restorePlot(ctx: Ctx, p: NonNullable<PanelSnapshot['plot']>): void {
  setSampleCount(ctx, p.sampleCount);
  setMarginalizationCount(ctx, p.marginalizationCount);
  const io = ctx.inferenceOpts || {};
  const drawsChanged = p.rejectionBudget !== ctx.REJECTION_BUDGET || p.rootSeed !== ctx.rootSeed
    || JSON.stringify(p.inferenceOpts) !== JSON.stringify(io);
  ctx.REJECTION_BUDGET = p.rejectionBudget;
  ctx.rootSeed = p.rootSeed;
  ctx.inferenceOpts = { ...p.inferenceOpts } as Ctx['inferenceOpts'];
  if (drawsChanged) clearForwardCaches(ctx);
  ctx.recordSelection = p.recordSelection as Ctx['recordSelection'];
  ctx.presetOverrides = new Map(p.presetOverrides);
  ctx.domainOverrides = new Map(p.domainOverrides);
  ctx.planMemoryByName = new Map(p.planMemory);
  // Rebuild the plan so the restored memory applies; dropping the current
  // plan first keeps updatePlotForBinding from writing its stale state back.
  const name = ctx.currentPlotBindingName;
  ctx.currentPlotPlan = null;
  if (name != null) updatePlotForBinding(ctx, name);
}
