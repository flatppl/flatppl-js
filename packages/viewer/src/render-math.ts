// @ts-check
// @flatppl/viewer — the math pane —
//
// Third panel of the vertical split (panels.ts): the current model as
// mathematical notation, one row per module-level binding, with the
// focused binding highlighted and every identifier a link back to its
// binding. The FlatPPL → mathematics conversion is flatppl-rust's job
// (`flatppl_wasm_api.render_math`, the `mathdoc` crate); this module
// owns the renderer loading, the pane's DOM and the navigation hooks.
// The request/response shapes, the row composition and markup, and the
// focus / model-key rules are the pure module math-view.ts. Design +
// contract: flatppl-dev/math-view-design.md.
//
// Renderer seam. The host names the wasm-pack `--target web` glue of
// flatppl_wasm_api through `__FLATPPL_CONFIG__.wasmApiUrl` (relative
// URLs resolve against the page; the `.wasm` sits next to the glue, as
// wasm-pack lays it out). It is imported lazily on the first render, so
// hosts and users who never open the pane pay nothing. A host that
// ships no artifact leaves the field unset and the pane says so — no
// probe, no broken button. A failed load offers a retry.
//
// One entry point, idempotent: renderMathForCurrent(ctx) is called on
// every focus change (updatePlotForBinding), when the pane is enabled,
// and after a source update that failed to parse. It renders the ANALYSED
// source — the one the DAG's bindings came from (ctx.analyzedSource) —
// so rows, their doc-comments and source lines describe one model; when
// the editor's source no longer parses the pane keeps that model and says
// so. It re-renders only when the model changed (memoised by
// mathModelKey); otherwise it just moves the focus highlight. A rendering
// failure keeps the last good rows on screen with a notice.

import { esc } from './util.js';
import { focusNode } from './dag.js';
import { updatePlotForBinding } from './render-plot.js';
import {
  buildMathRequest, composeMathRows, rowHtml, moduleDocHtml, notationHtml, focusedBindingName, mathModelKey, mathWasmUrl,
} from './math-view.js';
import type { MathResponse } from './math-view.js';
import type { Ctx } from './types';

// ---- renderer loading -------------------------------------------------

interface RendererState {
  status: 'unconfigured' | 'loading' | 'ready' | 'failed';
  render: ((requestJson: string) => string) | null;
  error: string | null;
}

function rendererState(ctx: Ctx): RendererState {
  if (!ctx.mathRenderer) {
    ctx.mathRenderer = { status: 'unconfigured', render: null, error: null };
  }
  return ctx.mathRenderer;
}

/** Kick off (once) the lazy import of the wasm glue; re-renders the
 *  pane when it settles either way. Returns false when the host
 *  configured no renderer. */
function ensureRenderer(ctx: Ctx): boolean {
  const st = rendererState(ctx);
  if (st.status !== 'unconfigured') return true;
  const url = mathWasmUrl(ctx.CONFIG, typeof document !== 'undefined' ? document.baseURI : undefined);
  if (!url) return false;
  st.status = 'loading';
  st.error = null;
  // The glue is an ES module; `import()` with a computed URL keeps the
  // bundler from trying to resolve it at build time. default() runs the
  // wasm init (fetches the .wasm next to the glue).
  import(/* @vite-ignore */ url).then(async function (mod: any) {
    if (mod && typeof mod.default === 'function') await mod.default();
    if (typeof mod.render_math !== 'function') {
      throw new Error('the wasm API at ' + url + ' has no render_math export');
    }
    st.render = mod.render_math;
    st.status = 'ready';
  }).catch(function (err: any) {
    st.status = 'failed';
    st.error = String(err && err.message || err);
  }).then(function () {
    // The pane may have been disposed or hidden meanwhile (disposeMathPane
    // clears mathEnabled), so this re-render is conditional.
    if (ctx.mathEnabled) renderMathForCurrent(ctx);
  });
  return true;
}

// ---- rendering --------------------------------------------------------

/** The pane's content element, or null once the viewer was disposed
 *  (callbacks such as the renderer load can outlive the skeleton, so no
 *  throwing `$()` here). */
function contentEl(ctx: Ctx): HTMLElement | null {
  if (!ctx.mathEnabled || typeof document === 'undefined') return null;
  return document.getElementById('math-content');
}

function showMathMessage(el: HTMLElement, html: string) {
  el.innerHTML = '<div class="math-empty">' + html + '</div>';
}

function bindingLine(ctx: Ctx, name: string): number | null {
  const b = ctx.currentBindings && ctx.currentBindings.get(name);
  return b && typeof b.line === 'number' && b.line >= 0 ? b.line : null;
}

/** The current model's memo key (the analysed source, not the editor's). */
function modelKey(ctx: Ctx): string {
  return mathModelKey(ctx.analyzedSource, ctx.currentPath, ctx.currentBundleSources);
}

/** The notice shown above the rows when the editor's source is not the
 *  model on screen (it failed to parse; the DAG keeps the last model too). */
function staleSourceNotice(ctx: Ctx): string | null {
  if (ctx.currentSource != null && ctx.analyzedSource != null && ctx.currentSource !== ctx.analyzedSource) {
    return 'The source does not parse; showing the last valid model.';
  }
  return null;
}

function buildRows(ctx: Ctx, el: HTMLElement, res: MathResponse, notice: string | null) {
  // Rust owns document layout, appendices and the legend. Keep the row renderer
  // below for hosts that still carry an older WASM artifact.
  if (res.document) {
    el.innerHTML = (notice ? '<div class="math-notice">' + notice + '</div>' : '') + res.document.html;
    const style = document.createElement('style');
    style.textContent = res.document.css;
    el.prepend(style);
    ctx.mathView = { key: modelKey(ctx), response: res, rowCount: res.bindings.length };
    updateFocus(ctx, el);
    return;
  }
  const { rows, moduleDiagnostics } = composeMathRows(res, {
    focus: focusedBindingName(ctx),
    lineOf: function (name) { return bindingLine(ctx, name); },
  });
  let h = '';
  if (notice) h += '<div class="math-notice">' + notice + '</div>';
  h += moduleDocHtml(res.doc);
  if (moduleDiagnostics.length) {
    h += '<ul class="math-module-diags">';
    for (const d of moduleDiagnostics) h += '<li>' + esc(d) + '</li>';
    h += '</ul>';
  }
  if (rows.length === 0) h += '<div class="math-empty">No bindings to show.</div>';
  for (const r of rows) h += rowHtml(r);
  h += notationHtml(res.notation);
  // SECURITY: markup helpers escape everything they interpolate except the
  // fragments our own Rust renderer produced (row and notation MathML, plus
  // sanitised doc-comment HTML); the notices above are esc()'d or constant.
  el.innerHTML = h;
  ctx.mathView = { key: modelKey(ctx), response: res, rowCount: rows.length };
  scrollFocusedIntoView(el);
}

function scrollFocusedIntoView(el: HTMLElement) {
  const f = el.querySelector('.math-row.focused, mtr[data-flatppl-binding].focused') as HTMLElement | null;
  if (f && typeof f.scrollIntoView === 'function') {
    try { f.scrollIntoView({ block: 'nearest' }); } catch (_) { f.scrollIntoView(); }
  }
}

/** Move the focus highlight without rebuilding the rows. */
function updateFocus(ctx: Ctx, el: HTMLElement) {
  const focus = focusedBindingName(ctx);
  const res = ctx.mathView && ctx.mathView.response;
  const rows = el.querySelectorAll('.math-row, mtr[data-flatppl-binding]');
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i] as HTMLElement;
    const name = r.getAttribute('data-flatppl-binding') || r.getAttribute('data-binding');
    const b = res && res.bindings.find(function (x) { return x.name === name; });
    const names = b && b.names && b.names.length ? b.names : (name ? [name] : []);
    r.classList.toggle('focused', focus !== null && names.indexOf(focus) !== -1);
  }
  scrollFocusedIntoView(el);
}

/** Render (or re-render) the math pane for the current model; cheap
 *  when only the focus moved. Safe to call whenever the pane is enabled. */
export function renderMathForCurrent(ctx: Ctx) {
  const el = contentEl(ctx);
  if (!el) return;
  if (!ctx.analyzedSource) {
    showMathMessage(el, ctx.currentSource
      ? 'The source does not parse yet.'
      : 'Load a model to see it as mathematics.');
    return;
  }
  if (!ensureRenderer(ctx)) {
    showMathMessage(el, 'Math view is not available in this build.');
    return;
  }
  const st = rendererState(ctx);
  if (st.status === 'loading') {
    showMathMessage(el, 'Loading the math renderer…');
    return;
  }
  if (st.status === 'failed') {
    showMathMessage(el, 'Math view could not load its renderer: ' + esc(st.error || 'unknown error')
      + ' <a href="#" class="math-retry">Retry</a>');
    return;
  }
  const key = modelKey(ctx);
  const notice = staleSourceNotice(ctx);
  if (ctx.mathView && ctx.mathView.key === key) {
    updateFocus(ctx, el);
    // The stale-source notice is the one thing that can change while the
    // model on screen stays the same.
    const shown = el.querySelector('.math-notice.stale');
    if (notice && !shown) el.insertAdjacentHTML('afterbegin', '<div class="math-notice stale">' + notice + '</div>');
    else if (!notice && shown) shown.remove();
    return;
  }
  let res: MathResponse;
  try {
    res = JSON.parse(st.render!(JSON.stringify(buildMathRequest({
      source: ctx.analyzedSource,
      path: ctx.currentPath,
      bundleSources: ctx.currentBundleSources,
    }))));
  } catch (err: any) {
    const msg = esc(String(err && err.message || err));
    if (ctx.mathView && ctx.mathView.response) {
      // Keep the last good rendering visible and say why it is stale.
      buildRows(ctx, el, ctx.mathView.response, 'Math view is out of date: ' + msg);
      ctx.mathView.key = key;   // don't retry until the model changes again
    } else {
      showMathMessage(el, 'Math view could not render this model: ' + msg);
      ctx.mathView = { key, response: null, rowCount: 0 };
    }
    return;
  }
  buildRows(ctx, el, res, notice ? '<span class="stale">' + notice + '</span>' : null);
}

/** Forget the pane's state on viewer teardown, so a renderer load that
 *  settles later neither throws nor paints into a successor viewer. */
export function disposeMathPane(ctx: Ctx) {
  ctx.mathEnabled = false;
  ctx.mathRenderer = null;
  ctx.mathView = null;
}

// ---- navigation ------------------------------------------------------

/** Click wiring for the pane (event delegation, installed once at mount),
 *  the DAG's own gestures:
 *    click on an identifier / row → select that binding (plot pane follows)
 *    double-click                  → drill the sub-DAG down to it
 *    Ctrl/Cmd+click                → jump to the binding's source line
 *  Identifiers carry `data-flatppl-ref` on their outermost MathML element
 *  (an <mi>, or an <msub> for a subscripted symbol), so the lookup walks
 *  up from the click target to the nearest carrier. The failed-load
 *  message's Retry link is handled here too. */
export function installMathPaneNavigation(ctx: Ctx) {
  const el = document.getElementById('math-content');
  if (!el) return;
  function bindingAt(ev: MouseEvent): string | null {
    const target = ev.target as Element | null;
    if (!target) return null;
    const refEl = target.closest('[data-flatppl-ref]');
    const rowEl = target.closest('mtr[data-flatppl-binding], .math-row');
    const name = refEl ? refEl.getAttribute('data-flatppl-ref')
      : rowEl && (rowEl.getAttribute('data-flatppl-binding') || rowEl.getAttribute('data-binding'));
    return name && ctx.currentBindings && ctx.currentBindings.has(name) ? name : null;
  }
  el.addEventListener('click', function (ev: MouseEvent) {
    const target = ev.target as Element | null;
    if (target && target.closest('.math-retry')) {
      ev.preventDefault();
      ctx.mathRenderer = null;
      renderMathForCurrent(ctx);
      return;
    }
    const name = bindingAt(ev);
    if (!name) return;
    ev.preventDefault();
    if (ev.ctrlKey || ev.metaKey) {
      const line = bindingLine(ctx, name);
      if (line !== null && ctx.host && typeof ctx.host.revealSourceLine === 'function') {
        ctx.host.revealSourceLine(line, name);
      }
      return;
    }
    updatePlotForBinding(ctx, name);
  });
  el.addEventListener('dblclick', function (ev: MouseEvent) {
    const name = bindingAt(ev);
    if (!name || ev.ctrlKey || ev.metaKey) return;
    ev.preventDefault();
    focusNode(ctx, name, true);
  });
}
