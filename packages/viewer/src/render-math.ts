// @ts-check
// @flatppl/viewer — the math pane —
//
// Third panel of the vertical split (panels.ts): the current model as
// mathematical notation, one row per module-level binding, with the
// focused binding highlighted and every identifier a link back to its
// binding. The FlatPPL → mathematics conversion is flatppl-rust's job
// (`flatppl_wasm_api.render_math`, the `mathdoc` crate); this module
// owns the renderer loading, the pane's DOM and the navigation hooks.
// The request/response shapes and the row composition are the pure
// module math-view.ts. Design + contract: flatppl-dev/math-view-design.md.
//
// Renderer seam. The host names the wasm-pack `--target web` glue of
// flatppl_wasm_api through `__FLATPPL_CONFIG__.wasmApiUrl` (relative
// URLs resolve against the page; the `.wasm` sits next to the glue, as
// wasm-pack lays it out). It is imported lazily on the first render, so
// hosts and users who never open the pane pay nothing. A host that
// ships no artifact leaves the field unset and the pane says so — no
// probe, no broken button.
//
// One entry point, idempotent: renderMathForCurrent(ctx) is called on
// every focus change (updatePlotForBinding) and when the pane is
// enabled. It re-renders only when the model changed (memoised by
// source + path + bundle); otherwise it just moves the focus highlight.
// A rendering failure keeps the last good rows on screen with a notice,
// mirroring how the DAG keeps the previous bindings on a parse error.

import { $, esc } from './util.js';
import { renderDoc } from './markdown.js';
import { focusNode } from './dag.js';
import { buildMathRequest, composeMathRows } from './math-view.js';
import type { MathResponse, MathRow } from './math-view.js';
import type { Ctx } from './types';

// ---- renderer loading -------------------------------------------------

type Renderer = (requestJson: string) => string;

interface RendererState {
  status: 'unconfigured' | 'loading' | 'ready' | 'failed';
  render: Renderer | null;
  error: string | null;
}

function rendererState(ctx: Ctx): RendererState {
  if (!ctx.mathRenderer) {
    ctx.mathRenderer = { status: 'unconfigured', render: null, error: null };
  }
  return ctx.mathRenderer;
}

/** Resolve the configured glue URL against the page, or null when the
 *  host configured none. */
export function mathWasmUrl(config: { wasmApiUrl?: string } | null | undefined): string | null {
  const raw = config && typeof config.wasmApiUrl === 'string' ? config.wasmApiUrl.trim() : '';
  if (!raw) return null;
  try {
    return new URL(raw, (typeof document !== 'undefined' && document.baseURI) || undefined).href;
  } catch (_) {
    return raw;
  }
}

/** Kick off (once) the lazy import of the wasm glue; re-renders the
 *  pane when it settles either way. */
function ensureRenderer(ctx: Ctx) {
  const st = rendererState(ctx);
  if (st.status !== 'unconfigured') return;
  const url = mathWasmUrl(ctx.CONFIG);
  if (!url) return;
  st.status = 'loading';
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
    if (ctx.mathEnabled) renderMathForCurrent(ctx);
  });
}

// ---- rendering --------------------------------------------------------

function showMathMessage(el: HTMLElement, html: string) {
  el.innerHTML = '<div class="math-empty">' + html + '</div>';
}

/** The binding the pane highlights: what the plot pane shows, else the
 *  sub-DAG root; none in module view. */
function focusedBindingName(ctx: Ctx): string | null {
  if (ctx.currentPlotBindingName) return ctx.currentPlotBindingName;
  const t = ctx.currentState && ctx.currentState.targetName;
  return t && t !== ctx.MODULE_TARGET ? t : null;
}

function bindingLine(ctx: Ctx, name: string): number | null {
  const b = ctx.currentBindings && ctx.currentBindings.get(name);
  return b && typeof b.line === 'number' && b.line >= 0 ? b.line : null;
}

function bindingDoc(ctx: Ctx, name: string): any | null {
  const b = ctx.currentBindings && ctx.currentBindings.get(name);
  return (b && b.node && b.node.doc) || null;
}

/** Memo key: the model as the Rust side sees it. */
function modelKey(ctx: Ctx): string {
  return JSON.stringify([ctx.currentSource, ctx.currentPath || null, ctx.currentBundleSources || null]);
}

function rowHtml(row: MathRow): string {
  let h = '<div class="math-row' + (row.focused ? ' focused' : '') + '" data-binding="' + esc(row.name) + '">';
  if (row.doc) {
    const doc = renderDoc(row.doc);
    if (doc) h += '<div class="math-row-doc">' + doc + '</div>';
  }
  h += '<div class="math-row-eq">' + row.mathml;
  if (row.annotation) h += '<span class="math-row-annotation">' + esc(row.annotation) + '</span>';
  h += '</div>';
  if (row.diagnostics.length) {
    h += '<ul class="math-row-diags">';
    for (const d of row.diagnostics) h += '<li>' + esc(d) + '</li>';
    h += '</ul>';
  }
  return h + '</div>';
}

function buildRows(ctx: Ctx, el: HTMLElement, res: MathResponse, notice: string | null) {
  const { rows, moduleDiagnostics } = composeMathRows(res, {
    focus: focusedBindingName(ctx),
    lineOf: function (name) { return bindingLine(ctx, name); },
    docOf: function (name) { return bindingDoc(ctx, name); },
  });
  let h = '';
  if (notice) h += '<div class="math-notice">' + notice + '</div>';
  if (moduleDiagnostics.length) {
    h += '<ul class="math-module-diags">';
    for (const d of moduleDiagnostics) h += '<li>' + esc(d) + '</li>';
    h += '</ul>';
  }
  if (rows.length === 0) h += '<div class="math-empty">No bindings to show.</div>';
  for (const r of rows) h += rowHtml(r);
  // SECURITY: the only markup not produced here is each row's `mathml`,
  // the trusted fragment of our own Rust printer (text escaped there);
  // everything else is esc()'d or renderDoc()'s sanitised output.
  el.innerHTML = h;
  ctx.mathView = { key: modelKey(ctx), response: res, lines: new Map(rows.map(function (r) { return [r.name, r.line]; })) };
  scrollFocusedIntoView(el);
}

function scrollFocusedIntoView(el: HTMLElement) {
  const f = el.querySelector('.math-row.focused') as HTMLElement | null;
  if (f && typeof f.scrollIntoView === 'function') {
    try { f.scrollIntoView({ block: 'nearest' }); } catch (_) { f.scrollIntoView(); }
  }
}

/** Move the focus highlight without rebuilding the rows. */
function updateFocus(ctx: Ctx, el: HTMLElement) {
  const focus = focusedBindingName(ctx);
  const res = ctx.mathView && ctx.mathView.response;
  const rows = el.querySelectorAll('.math-row');
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i] as HTMLElement;
    const name = r.getAttribute('data-binding');
    const b = res && res.bindings.find(function (x) { return x.name === name; });
    const names = b && b.names && b.names.length ? b.names : (name ? [name] : []);
    r.classList.toggle('focused', focus !== null && names.indexOf(focus) !== -1);
  }
  scrollFocusedIntoView(el);
}

/** Render (or re-render) the math pane for the current model; cheap
 *  when only the focus moved. Safe to call whenever the pane is enabled. */
export function renderMathForCurrent(ctx: Ctx) {
  const el = $('math-content');
  if (!el || !ctx.mathEnabled) return;
  if (!ctx.currentSource) {
    showMathMessage(el, 'Load a model to see it as mathematics.');
    return;
  }
  const st = rendererState(ctx);
  if (st.status === 'unconfigured') {
    ensureRenderer(ctx);
    if (rendererState(ctx).status === 'unconfigured') {
      showMathMessage(el, 'Math view is not available in this build.');
      return;
    }
  }
  if (st.status === 'loading') {
    showMathMessage(el, 'Loading the math renderer…');
    return;
  }
  if (st.status === 'failed') {
    showMathMessage(el, 'Math view could not load its renderer: ' + esc(st.error || 'unknown error'));
    return;
  }
  const key = modelKey(ctx);
  if (ctx.mathView && ctx.mathView.key === key && el.querySelector('.math-row')) {
    updateFocus(ctx, el);
    return;
  }
  let res: MathResponse;
  try {
    res = JSON.parse(st.render!(JSON.stringify(buildMathRequest({
      source: ctx.currentSource,
      path: ctx.currentPath,
      bundleSources: ctx.currentBundleSources,
    }))));
  } catch (err: any) {
    const msg = esc(String(err && err.message || err));
    if (ctx.mathView && ctx.mathView.response) {
      // Keep the last good rendering visible (as the DAG keeps its
      // previous bindings on a parse error) and say why it is stale.
      buildRows(ctx, el, ctx.mathView.response, 'Math view is out of date: ' + msg);
      ctx.mathView.key = key;   // don't retry until the model changes again
    } else {
      showMathMessage(el, 'Math view could not render this model: ' + msg);
    }
    return;
  }
  buildRows(ctx, el, res, null);
}

// ---- navigation ------------------------------------------------------

/** Click wiring for the pane (event delegation, installed once at mount):
 *    click on an identifier   → focus that binding (as a cursor move would)
 *    click on a row           → focus the row's binding
 *    Ctrl/Cmd+click on either → jump to the binding's source line
 *  Identifiers carry `data-flatppl-ref` on their outermost MathML element
 *  (an <mi>, or an <msub> for a subscripted symbol), so the lookup walks
 *  up from the click target to the nearest carrier. */
export function installMathPaneNavigation(ctx: Ctx) {
  const el = $('math-content');
  if (!el) return;
  el.addEventListener('click', function (ev: MouseEvent) {
    const target = ev.target as Element | null;
    if (!target) return;
    const refEl = target.closest('[data-flatppl-ref]');
    const rowEl = target.closest('.math-row') as HTMLElement | null;
    const name = refEl ? refEl.getAttribute('data-flatppl-ref') : (rowEl ? rowEl.getAttribute('data-binding') : null);
    if (!name || !ctx.currentBindings || !ctx.currentBindings.has(name)) return;
    ev.preventDefault();
    if (ev.ctrlKey || ev.metaKey) {
      const line = bindingLine(ctx, name);
      if (line !== null && ctx.host && typeof ctx.host.revealSourceLine === 'function') {
        ctx.host.revealSourceLine(line, name);
      }
      return;
    }
    focusNode(ctx, name, true);
  });
}
