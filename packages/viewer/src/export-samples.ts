// @flatppl/viewer — export the focused empirical measure as gzipped CSV / JSON.
//
// One column/array per scalar variate (listScalarAxes), one row/entry per draw,
// plus a `weight` column/array when the measure carries importance weights
// (exp-normalized). Full numeric precision — the export feeds downstream
// analysis. Output is always gzip-compressed (.csv.gz / .json.gz) via the
// browser-native CompressionStream (no dependency).

import { listScalarAxes } from './util.js';

declare const FlatPPLEngine: any;

/** Build CSV text for an empirical measure. Empty string if no scalar axes. */
export function measureToCsv(measure: any): string {
  const axes = listScalarAxes(measure);
  if (axes.length === 0) return '';
  const n = axes[0].samples.length;
  const logWeights = measure && measure.logWeights;
  const weighted = !!logWeights;
  const normW = weighted ? FlatPPLEngine.histogram.normaliseWeights(logWeights) : null;

  const header = axes.map(function (a: any) { return csvField(a.label); });
  if (weighted) header.push('weight');
  const lines = [header.join(',')];

  for (let i = 0; i < n; i++) {
    const row = new Array(axes.length + (weighted ? 1 : 0));
    for (let j = 0; j < axes.length; j++) row[j] = String(axes[j].samples[i]);
    if (weighted) row[axes.length] = String(normW[i]);
    lines.push(row.join(','));
  }
  return lines.join('\n') + '\n';
}

/** Build JSON text: { variates, n, samples: {name: [...]}, weight?: [...] }.
 *  Empty string if no scalar axes. */
export function measureToJson(measure: any): string {
  const axes = listScalarAxes(measure);
  if (axes.length === 0) return '';
  const n = axes[0].samples.length;
  const samples: Record<string, number[]> = {};
  for (let j = 0; j < axes.length; j++) samples[axes[j].label] = Array.from(axes[j].samples as Float64Array);
  const obj: any = { variates: axes.map(function (a: any) { return a.label; }), n: n, samples: samples };
  const logWeights = measure && measure.logWeights;
  if (logWeights) obj.weight = Array.from(FlatPPLEngine.histogram.normaliseWeights(logWeights) as Float64Array);
  return JSON.stringify(obj);
}

/** CSV-quote a header field if it contains a comma, quote, or newline. */
function csvField(s: string): string {
  if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

/** Sanitize a binding name into a filename stem. */
function safeStem(bindingName: any): string {
  return (typeof bindingName === 'string' && bindingName)
    ? bindingName.replace(/[^A-Za-z0-9._-]/g, '_')
    : 'samples';
}

/** Gzip `text` to a Blob via the native CompressionStream (no dependency). */
async function gzipBlob(text: string, mime: string): Promise<Blob> {
  const input = new Blob([text], { type: mime });
  const stream = input.stream().pipeThrough(new (globalThis as any).CompressionStream('gzip'));
  return await new Response(stream).blob();
}

/** Trigger a browser download of `blob` as `filename`. */
function triggerDownload(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Revoke on the next tick so the click's navigation has consumed the URL.
  setTimeout(function () { URL.revokeObjectURL(url); }, 0);
}

/** Serialize `measure` to `fmt` ('csv' | 'json'), gzip it, and download as
 *  `<safe>-samples.<ext>.gz`. No-op (no file) when there are no scalar axes. */
export async function downloadMeasure(measure: any, bindingName: any, fmt: 'csv' | 'json'): Promise<void> {
  try {
    const text = fmt === 'json' ? measureToJson(measure) : measureToCsv(measure);
    if (!text) return;
    const ext = fmt === 'json' ? 'json' : 'csv';
    const mime = fmt === 'json' ? 'application/json' : 'text/csv;charset=utf-8';
    const blob = await gzipBlob(text, mime);
    triggerDownload(safeStem(bindingName) + '-samples.' + ext + '.gz', blob);
  } catch (err) {
    try { console.error('[@flatppl/viewer] sample export failed:', err); } catch (_) {}
  }
}
