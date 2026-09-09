// @ts-check
// @flatppl/viewer — math pane data layer (pure) —
//
// The viewer side of flatppl-rust's `render_math` contract
// (flatppl-dev/math-view-design.md §4, as implemented on the Rust
// `mathdoc` crate): what the request looks like, how a response becomes
// the rows the pane renders, how a row becomes markup, and the small
// rules the pane needs (which binding is focused, what identifies a
// model, where the renderer lives). No DOM, no wasm here — the wasm call
// and the DOM live in render-math.ts, so this contract logic is testable
// on its own and the fixture-backed tests pin it.
//
// Contract essentials:
//   - request  { source, path, bundle: { resolvedPath: text }, formats }
//     where `path` is the primary's resolved path and every bundle key
//     is a `load_module` directive resolved against ITS importer's
//     directory per spec §04 — exactly the map the host already hands
//     the JS engine as `bundleSources`.
//   - response { order, bindings, diagnostics }: one row per
//     module-level binding; a decomposition (`a, b ~ M`) is ONE row named
//     after its first target with every name in `names`; `order` lists
//     row names, in source order; `refs` lists the OTHER bindings a row
//     refers to (its own names excluded); `diagnostics[].binding` is "" for
//     module-level ones.
//   - `mathml` is a trusted `<math display="block"
//     data-flatppl-binding="NAME">` fragment whose identifier leaves
//     carry `data-flatppl-ref="NAME"` on their OUTERMOST element (an
//     `<mi>`, or `<msub>` for a subscripted symbol). It is optional in
//     the wire format (the Rust side only emits requested formats).
//   - doc-comments are rendered by the SAME crate (Markdown + `$…$` as
//     MathML, raw HTML escaped, unsafe link targets dropped), so doc math
//     and row math share one generator and one sanitiser: per binding
//     `doc: { html, block }` (`block` = a `%%%` multi-line comment, prose;
//     false = a one-line `%` caption), at module level `doc: { title,
//     html }` (the `flatppl_compat` doc-comment, spec §04, its first
//     heading split off as the title). Both absent without a comment. The
//     viewer's own marked + Temml pipeline stays for tooltips and hovers.

import { esc, escAttr } from './util.js';

export const MATH_FORMATS: string[] = ['mathml'];

/** The path the Rust side assumes when the host supplies none. */
export const DEFAULT_MODULE_PATH = 'model.flatppl';

export interface MathRequest {
  source: string;
  path: string;
  bundle: Record<string, string>;
  formats: string[];
}

/** A rendered doc-comment: a trusted HTML fragment of our own Rust
 *  renderer. `block` marks a `%%%` multi-line comment (prose) as opposed
 *  to a one-line `%` caption. */
export interface MathDoc { html: string; block: boolean }

/** The module documentation (the `flatppl_compat` doc-comment) with its
 *  first Markdown heading split off as the title. */
export interface MathModuleDoc { title: string | null; html: string }

export interface MathBinding {
  name: string;
  names: string[];
  kind: 'draw' | 'value' | 'measure' | 'callable' | 'likelihood' | 'module' | string;
  mathml?: string;
  refs: string[];
  loc?: { start: number; end: number };
  annotation?: string;
  doc?: MathDoc;
}

export interface MathDiagnostic { binding: string; message: string }

export interface MathResponse {
  order: string[];
  bindings: MathBinding[];
  diagnostics: MathDiagnostic[];
  doc?: MathModuleDoc;
}

/** One rendered row of the pane. */
export interface MathRow {
  name: string;
  names: string[];
  kind: string;
  mathml: string;
  refs: string[];
  annotation: string | null;
  diagnostics: string[];
  focused: boolean;
  /** 0-based source line of the binding (the engine's per-binding line),
   *  null when unknown — drives Ctrl+click → source. */
  line: number | null;
  /** The binding's rendered doc-comment, or null. */
  doc: MathDoc | null;
}

export function buildMathRequest(m: {
  source: string;
  path?: string | null;
  bundleSources?: Record<string, string> | null;
}): MathRequest {
  return {
    source: m.source,
    path: m.path || DEFAULT_MODULE_PATH,
    bundle: m.bundleSources ? Object.assign({}, m.bundleSources) : {},
    formats: MATH_FORMATS.slice(),
  };
}

/** The row that owns `name` (its own name or a further decomposition
 *  target), or null when no row claims it (a reference to something the
 *  Rust side folded, or to a builtin). */
export function rowNameFor(res: MathResponse, name: string): string | null {
  for (const b of res.bindings) {
    if (b.name === name || (b.names && b.names.indexOf(name) !== -1)) return b.name;
  }
  return null;
}

/**
 * Turn a response into rows in `order`, with the focus, per-row
 * diagnostics and source lines attached. Nothing the Rust
 * side reported is dropped: diagnostics that name no row (binding "" or
 * an unknown name) become module-level, a binding `order` forgot is
 * reported at module level, and a row without a fragment renders empty
 * with its own diagnostic instead of the text "undefined".
 */
export function composeMathRows(res: MathResponse, opts: {
  focus?: string | null;
  lineOf?: (name: string) => number | null | undefined;
}): { rows: MathRow[]; moduleDiagnostics: string[] } {
  const byName = new Map<string, MathBinding>();
  for (const b of res.bindings || []) byName.set(b.name, b);

  const rowDiagnostics = new Map<string, string[]>();
  const moduleDiagnostics: string[] = [];
  const addRowDiagnostic = (owner: string, message: string) => {
    if (!rowDiagnostics.has(owner)) rowDiagnostics.set(owner, []);
    rowDiagnostics.get(owner)!.push(message);
  };
  for (const d of res.diagnostics || []) {
    const owner = d.binding ? rowNameFor(res, d.binding) : null;
    if (owner) addRowDiagnostic(owner, d.message);
    else moduleDiagnostics.push(d.binding ? d.binding + ': ' + d.message : d.message);
  }

  const order = res.order || [];
  const listed = new Set(order);
  for (const b of res.bindings || []) {
    if (!listed.has(b.name)) moduleDiagnostics.push(b.name + ': rendered but missing from the row order (not shown)');
  }

  const focus = opts.focus || null;
  const rows: MathRow[] = [];
  for (const name of order) {
    const b = byName.get(name);
    if (!b) continue;
    const names = b.names && b.names.length ? b.names : [b.name];
    const line = opts.lineOf ? opts.lineOf(b.name) : null;
    if (typeof b.mathml !== 'string') addRowDiagnostic(b.name, 'no MathML fragment in the response');
    rows.push({
      name: b.name,
      names,
      kind: b.kind,
      mathml: typeof b.mathml === 'string' ? b.mathml : '',
      refs: b.refs || [],
      annotation: b.annotation || null,
      diagnostics: rowDiagnostics.get(b.name) || [],
      focused: focus !== null && names.indexOf(focus) !== -1,
      line: typeof line === 'number' ? line : null,
      doc: b.doc && typeof b.doc.html === 'string' ? { html: b.doc.html, block: b.doc.block === true } : null,
    });
  }
  return { rows, moduleDiagnostics };
}

/**
 * One row's markup. The ONLY markup taken verbatim is what our own Rust
 * renderer produced — `row.mathml` and the doc-comment fragment, both
 * escaped/sanitised there — and every other interpolation is escaped
 * here: the name as an attribute value, the annotation and diagnostics as
 * text.
 */
export function rowHtml(row: MathRow): string {
  let h = '<div class="math-row' + (row.focused ? ' focused' : '') + '" data-binding="' + escAttr(row.name) + '">';
  if (row.doc && row.doc.html) {
    h += '<div class="math-row-doc' + (row.doc.block ? ' block' : '') + '">' + row.doc.html + '</div>';
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

/** The module introduction above the rows: the title (escaped here) as
 *  the pane's own heading, then the rendered body (the Rust renderer's
 *  trusted fragment, its headings already shifted below h1). '' when the
 *  module carries no doc-comment. */
export function moduleDocHtml(doc: MathModuleDoc | null | undefined): string {
  if (!doc) return '';
  const title = typeof doc.title === 'string' && doc.title.trim() ? '<h1>' + esc(doc.title) + '</h1>' : '';
  const body = typeof doc.html === 'string' ? doc.html : '';
  if (!title && !body) return '';
  return '<div class="math-module-doc">' + title + body + '</div>';
}

/** The binding the pane highlights: what the plot pane shows, else the
 *  sub-DAG root; none in module view. */
export function focusedBindingName(ctx: {
  currentPlotBindingName: string | null;
  currentState: { targetName: string } | null;
  MODULE_TARGET: string;
}): string | null {
  if (ctx.currentPlotBindingName) return ctx.currentPlotBindingName;
  const t = ctx.currentState && ctx.currentState.targetName;
  return t && t !== ctx.MODULE_TARGET ? t : null;
}

/** Identity of a model as the Rust side sees it: the analysed source,
 *  its path and the dependency bundle — the pane's memo key. */
export function mathModelKey(
  source: string | null | undefined,
  path: string | null | undefined,
  bundle: Record<string, string> | null | undefined,
): string {
  return JSON.stringify([source || '', path || null, bundle || null]);
}

/** Resolve the host-configured glue URL against the page (relative URLs
 *  are relative to the document, as the gallery's `vendor/…` is), or null
 *  when the host configured none. */
export function mathWasmUrl(config: { wasmApiUrl?: string | null } | null | undefined, baseURI: string | undefined): string | null {
  const raw = config && typeof config.wasmApiUrl === 'string' ? config.wasmApiUrl.trim() : '';
  if (!raw) return null;
  try {
    return new URL(raw, baseURI).href;
  } catch (_) {
    return raw;
  }
}
