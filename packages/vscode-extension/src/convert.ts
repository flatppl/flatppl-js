'use strict';
// "FlatPPL: Convert HS3 / pyhf to FlatPPL" — the host-agnostic core.
//
// The conversion is flatppl-rust's `flatppl_wasm_api.convert(input, from,
// "flatppl")` (crates/wasm-api/src/lib.rs over flatppl_hs3::read_hs3 /
// read_pyhf and the flatppl-syntax printer), run IN THE EXTENSION HOST from
// the bundled artifact — the same wasm the web gallery's "Convert to
// FlatPPL" tool calls in the browser.
//
// The importer is chosen by the file name, using the gallery's one rule
// (packages/web/src/file-types.mjs `typeForPath`), so both hosts read
// `foo.pyhf.json` the same way. A plain `.json` says nothing, and the
// caller asks the user.
//
// No vscode import: the wasm entry is injected, so the unit tests require
// this module directly with no host stub — same pattern as mathExport.ts.

/** The importers `convert` accepts as its `from` argument. */
export type SourceFormat = 'hs3' | 'pyhf';

export const SOURCE_FORMATS: SourceFormat[] = ['hs3', 'pyhf'];

/** The importer a file name names, or null when it names none. Mirrors the
 *  gallery's `typeForPath`: `.hs3.json` / `.hs3` and `.pyhf.json` /
 *  `.pyhf`. A bare `.json` is deliberately null — guessing the schema from
 *  the name would misread half the corpus. */
export function sourceFormatForPath(p: string): SourceFormat | null {
  const lower = String(p || '').toLowerCase();
  if (lower.endsWith('.hs3.json') || lower.endsWith('.hs3')) return 'hs3';
  if (lower.endsWith('.pyhf.json') || lower.endsWith('.pyhf')) return 'pyhf';
  return null;
}

/** The converted document's name: the source's basename with the import
 *  extension dropped and `.flatppl` added (`w.pyhf.json` → `w.flatppl`). */
export function convertedFlatpplName(p: string): string {
  const base = String(p || '').split('/').pop() || 'model';
  const stem = base
    .replace(/\.(hs3|pyhf)\.json$/i, '')
    .replace(/\.(hs3|pyhf|json)$/i, '');
  return (stem || 'model') + '.flatppl';
}

export interface ConvertRequest {
  /** The source document's path, for the importer guess and the name. */
  path: string;
  /** The JSON text to import. */
  source: string;
  /** An importer the user already picked; the path decides when absent. */
  from?: SourceFormat | null;
  /** `flatppl_wasm_api.convert`: throws with the importer's diagnostic. */
  convert: (input: string, from: string, to: string) => string;
}

export interface ConvertResult {
  format: SourceFormat;
  /** Suggested name for the converted document. */
  name: string;
  /** The emitted FlatPPL source. */
  source: string;
}

/** Import a workspace and render it as FlatPPL source. Errors from the
 *  importer propagate with the wasm message intact; the caller reports
 *  them. */
export function runConvert(req: ConvertRequest): ConvertResult {
  const format = req.from || sourceFormatForPath(req.path);
  if (!format) {
    throw new Error(
      'cannot tell whether ' + req.path + ' is HS3 or pyhf (name it .hs3.json or .pyhf.json, or pick a format)');
  }
  const source = req.convert(req.source, format, 'flatppl');
  return { format, name: convertedFlatpplName(req.path), source };
}
