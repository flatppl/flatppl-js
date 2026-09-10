// @ts-check
// @flatppl/viewer — math pane data layer (pure) —
//
// The viewer side of flatppl-rust's `render_math` contract
// (flatppl-dev/math-view-design.md §4, as implemented on the Rust
// `mathdoc` crate): what the request looks like, what the response
// carries, and the small rules the pane needs (which binding is focused,
// what identifies a model, where the renderer lives). No DOM, no wasm
// here — the wasm call and the DOM live in render-math.ts, so this
// contract logic is testable on its own.
//
// Contract essentials:
//   - request  { source, path, bundle: { resolvedPath: text }, formats,
//     document: true } where `path` is the primary's resolved path and
//     every bundle key is a `load_module` directive resolved against ITS
//     importer's directory per spec §04 — exactly the map the host already
//     hands the JS engine as `bundleSources`.
//   - response { document, order, bindings, diagnostics, doc, notation }.
//     `document` is the whole article Rust lays out (rows, module
//     documentation, diagnostics, notation key) plus its scoped CSS; the
//     pane inserts it verbatim and only moves the focus highlight and
//     wires navigation. The per-binding fields describe the same content
//     (a decomposition `a, b ~ M` is ONE row named after its first target
//     with every name in `names`; `refs` lists the OTHER bindings a row
//     refers to) and are what the focus rule reads.
//   - Inside the article, `<mtr data-flatppl-binding="NAME">` is a row and
//     identifier leaves carry `data-flatppl-ref="NAME"` on their OUTERMOST
//     element (an `<mi>`, or `<msub>` for a subscripted symbol); that is
//     what the pane's click handling walks up to.
//   - Doc-comments are rendered by the SAME crate (Markdown + `$…$` as
//     MathML, raw HTML escaped, unsafe link targets dropped), so doc math
//     and row math share one generator and one sanitiser. The viewer's own
//     marked + Temml pipeline stays for tooltips and hovers.

export const MATH_FORMATS: string[] = ['mathml'];

/** The path the Rust side assumes when the host supplies none. */
export const DEFAULT_MODULE_PATH = 'model.flatppl';

export interface MathRequest {
  source: string;
  path: string;
  bundle: Record<string, string>;
  formats: string[];
  document: boolean;
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

/** A notation key entry: the builtin's name (a key, not for display),
 *  the symbol as MathML content without a <math> root, and the
 *  explanation as an HTML fragment (inline MathML for parameter names),
 *  the latter two produced by the same trusted renderer as the binding
 *  fragments. The math view shows no code spelling beside its notation. */
export interface MathNotation { name: string; mathml: string; note: string }

export interface MathResponse {
  /** The complete article and its scoped styles from Rust's HTML document
   *  renderer — what the pane shows. Required: the pane has no rendering of
   *  its own (flatppl-rust has returned it since 5f5ea8f, 2026-09-09). */
  document: { html: string; css: string };
  order: string[];
  bindings: MathBinding[];
  diagnostics: MathDiagnostic[];
  doc?: MathModuleDoc;
  notation?: MathNotation[];
}

/** One rendered row of the pane. */
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
    document: true,
  };
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
