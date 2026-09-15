'use strict';
// "FlatPPL: Export math as HTML / LaTeX / Typst" — the host-agnostic core.
//
// A model's mathematics as a whole document, written next to the source
// and opened (owner decision 2026-09-16; flatppl-dev/math-view-design.md
// §4 "Whole documents"). The conversion is flatppl-rust's
// `flatppl_wasm_api.export_math`, run IN THE EXTENSION HOST from the
// bundled artifact (lib/flatppl_wasm_api.cjs, see build-vendor.mjs) — no
// webview file plumbing. The request and the file name come from the
// viewer's contract layer (lib/math-view.cjs, bundled from
// packages/viewer/src/math-view.ts) so the gallery and VS Code derive
// them alike.
//
// No vscode import: every host piece (engine walk, workspace reads, the
// wasm entry, the overwrite prompt, the write) is injected, so the unit
// tests require this module directly with no host stub — same pattern as
// dependencyPrefetch.ts.

const { buildMathExportRequest, mathExportFileName } = require('../lib/math-view.cjs');

export type MathDocument = 'html' | 'md' | 'tex' | 'typ';

/** The saved file's name for a source path and format (`<stem>.<ext>`),
 *  the viewer's one rule, re-exported so the host derives a save-dialog
 *  default from the same place. */
export { mathExportFileName };

export interface MathExportDeps {
  document: MathDocument;
  /** The editor buffer, as the math pane renders it. */
  source: string;
  /** The engine path of the source (the URI path component, `/`-separated;
   *  the URL for a remote module) — what `load_module` deps resolve
   *  against, and the stem the document title comes from. */
  path: string;
  /** The engine's bundle walk (`resolveBundle` from the engine bundle). */
  resolveBundle: (
    primaryPath: string, source: string,
    readSource: (resolved: string) => Promise<string | null | undefined>,
  ) => Promise<{ sources: Record<string, string> }>;
  /** The host's read of one resolved dependency (null when missing). */
  readSource: (resolved: string) => Promise<string | null | undefined>;
  /** `flatppl_wasm_api.export_math`: JSON request in, document text out;
   *  throws for a malformed request or an unparsable module. */
  exportMath: (requestJson: string) => string;
  /** An explicit destination the user already chose (a save dialog, which
   *  confirms a replacement itself), same path convention as `path`;
   *  default: `<stem>.<ext>` beside the source, subject to `exists` +
   *  `confirmOverwrite`. */
  target?: string | null;
  exists: (target: string) => Promise<boolean>;
  /** Asked when the default target exists; false cancels the export. */
  confirmOverwrite: (target: string) => Promise<boolean>;
  write: (target: string, text: string) => Promise<void>;
}

export interface MathExportResult {
  status: 'written' | 'cancelled';
  target: string;
}

/** The default destination: the model's stem with the format's extension,
 *  in the source's directory (`/work/m.flatppl` + `tex` → `/work/m.tex`). */
export function mathExportTargetPath(sourcePath: string, document: MathDocument): string {
  const name = mathExportFileName(sourcePath, document);
  const i = sourcePath.lastIndexOf('/');
  return i < 0 ? name : sourcePath.slice(0, i + 1) + name;
}

/** Resolve the bundle, render the document, then write it — the render
 *  comes first so an unparsable model fails before any overwrite prompt.
 *  Errors from the renderer propagate; the caller reports them. */
export async function runMathExport(deps: MathExportDeps): Promise<MathExportResult> {
  let sources: Record<string, string> = {};
  try {
    const bundle = await deps.resolveBundle(deps.path, deps.source, deps.readSource);
    sources = (bundle && bundle.sources) || {};
  } catch (_e) {
    // Resolution failed entirely — still render the primary; the document
    // reports the unresolved dependencies itself.
  }
  const request = buildMathExportRequest({
    source: deps.source, path: deps.path, bundleSources: sources, document: deps.document,
  });
  const text = deps.exportMath(JSON.stringify(request));
  const target = deps.target || mathExportTargetPath(deps.path, deps.document);
  if (!deps.target && await deps.exists(target)) {
    if (!(await deps.confirmOverwrite(target))) return { status: 'cancelled', target };
  }
  await deps.write(target, text);
  return { status: 'written', target };
}
