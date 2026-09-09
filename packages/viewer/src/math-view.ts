// @ts-check
// @flatppl/viewer — math pane data layer (pure) —
//
// The viewer side of flatppl-rust's `render_math` contract
// (flatppl-dev/math-view-design.md §4, as implemented on the Rust
// `mathdoc` crate): what the request looks like, and how a response
// becomes the rows the pane renders. No DOM, no wasm here — the wasm
// call and the DOM live in render-math.ts, so this contract logic is
// testable on its own and the fixture-backed tests pin it.
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
//     row names; `diagnostics[].binding` is "" for module-level ones.
//   - `mathml` is a trusted `<math display="block"
//     data-flatppl-binding="NAME">` fragment whose identifier leaves
//     carry `data-flatppl-ref="NAME"` on their OUTERMOST element (an
//     `<mi>`, or `<msub>` for a subscripted symbol).

export const MATH_FORMATS: string[] = ['mathml'];

/** The path the Rust side assumes when the host supplies none. */
export const DEFAULT_MODULE_PATH = 'model.flatppl';

export interface MathRequest {
  source: string;
  path: string;
  bundle: Record<string, string>;
  formats: string[];
}

export interface MathBinding {
  name: string;
  names: string[];
  kind: 'draw' | 'value' | 'measure' | 'callable' | 'likelihood' | 'module' | string;
  mathml: string;
  refs: string[];
  loc?: { start: number; end: number };
  annotation?: string;
}

export interface MathDiagnostic { binding: string; message: string }

export interface MathResponse {
  order: string[];
  bindings: MathBinding[];
  diagnostics: MathDiagnostic[];
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
  /** The binding's doc-comment `{ markup, lines }` for renderDoc, or null. */
  doc: any | null;
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
 * diagnostics, source lines and doc-comments attached. Diagnostics that
 * name no row (binding "" or an unknown name) become module-level so
 * nothing the Rust side reported is dropped.
 */
export function composeMathRows(res: MathResponse, opts: {
  focus?: string | null;
  lineOf?: (name: string) => number | null | undefined;
  docOf?: (name: string) => any;
}): { rows: MathRow[]; moduleDiagnostics: string[] } {
  const byName = new Map<string, MathBinding>();
  for (const b of res.bindings || []) byName.set(b.name, b);

  const rowDiagnostics = new Map<string, string[]>();
  const moduleDiagnostics: string[] = [];
  for (const d of res.diagnostics || []) {
    const owner = d.binding ? rowNameFor(res, d.binding) : null;
    if (owner) {
      if (!rowDiagnostics.has(owner)) rowDiagnostics.set(owner, []);
      rowDiagnostics.get(owner)!.push(d.message);
    } else {
      moduleDiagnostics.push(d.binding ? d.binding + ': ' + d.message : d.message);
    }
  }

  const focus = opts.focus || null;
  const rows: MathRow[] = [];
  for (const name of res.order || []) {
    const b = byName.get(name);
    if (!b) continue;
    const names = b.names && b.names.length ? b.names : [b.name];
    const line = opts.lineOf ? opts.lineOf(b.name) : null;
    rows.push({
      name: b.name,
      names,
      kind: b.kind,
      mathml: b.mathml,
      refs: b.refs || [],
      annotation: b.annotation || null,
      diagnostics: rowDiagnostics.get(b.name) || [],
      focused: focus !== null && names.indexOf(focus) !== -1,
      line: typeof line === 'number' ? line : null,
      doc: (opts.docOf && opts.docOf(b.name)) || null,
    });
  }
  return { rows, moduleDiagnostics };
}
