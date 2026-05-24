'use strict';

// =====================================================================
// dataload — CSV / WSV / JSON parsers for `load_data` (spec §07)
// =====================================================================
//
// Per spec §07: load_data(source, valueset) reads an external data
// source and returns either a vector or a table whose shape is
// governed by `valueset`. All engines must support at least:
//
//   - JSON  (.json)  — array of objects, object of arrays, or vector
//   - CSV   (.csv)   — comma-separated, column names in first row
//   - WSV   (.wsv)   — whitespace-separated, column names in first row
//   - Arrow IPC      — deferred (binary format)
//
// Architectural decisions per TODO-flatppl-js.md "Multi-file models":
//   - Lazy fetching for data values: a separate dataResolver(path)
//     callback (sampler/plot only) returns parsed values.
//   - CSV + JSON parsers ship in the engine bundle. Arrow deferred.
//
// This module exposes the PARSERS (synchronous, text → value). The
// host-side resolver layer wraps these with async fetch (web) or
// fs.readFile (vscode-extension).

import type { Value as ValueType } from './engine-types';

// Discriminate file format by extension.
function detectFormat(path: string): 'json' | 'csv' | 'wsv' | null {
  const lower = path.toLowerCase();
  if (lower.endsWith('.json'))  return 'json';
  if (lower.endsWith('.csv'))   return 'csv';
  if (lower.endsWith('.wsv'))   return 'wsv';
  return null;
}

// =====================================================================
// JSON
// =====================================================================
//
// Three accepted shapes per spec §07:
//   - array of objects     [{a: 1, b: 2}, {a: 3, b: 4}]
//     → table { a: Value, b: Value }
//   - object of arrays     { a: [1, 3], b: [2, 4] }
//     → table { a: Value, b: Value }
//   - vector               [1.0, 2.0, 3.0]
//     → Value rank-1
//
// The engine doesn't validate against `valueset` here — that's a
// caller responsibility (typeinfer + downstream classifier).

function parseJSON(text: string): any {
  const data = JSON.parse(text);
  // Vector (array of scalars)
  if (Array.isArray(data) && data.length > 0
      && (typeof data[0] === 'number' || typeof data[0] === 'boolean')) {
    const out = new Float64Array(data.length);
    for (let i = 0; i < data.length; i++) {
      out[i] = data[i] === true ? 1 : data[i] === false ? 0 : +data[i];
    }
    return { shape: [data.length], data: out };
  }
  // Array of objects (array-of-structs).
  if (Array.isArray(data) && data.length > 0
      && typeof data[0] === 'object' && data[0] !== null
      && !Array.isArray(data[0])) {
    const keys = Object.keys(data[0]);
    const table: Record<string, ValueType> = {};
    for (const k of keys) {
      const col = new Float64Array(data.length);
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] as any)[k];
        col[i] = v === true ? 1 : v === false ? 0 : +v;
      }
      table[k] = { shape: [data.length], data: col };
    }
    return table;
  }
  // Object of arrays (struct-of-arrays).
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    const table: Record<string, ValueType> = {};
    let n = -1;
    for (const k in data) {
      if (!Object.prototype.hasOwnProperty.call(data, k)) continue;
      const col = data[k];
      if (!Array.isArray(col)) {
        throw new Error('parseJSON: struct-of-arrays column "' + k
          + '" must be an array, got ' + typeof col);
      }
      if (n < 0) n = col.length;
      else if (col.length !== n) {
        throw new Error('parseJSON: struct-of-arrays columns have unequal length ('
          + n + ' vs ' + col.length + ' at "' + k + '")');
      }
      const c = new Float64Array(col.length);
      for (let i = 0; i < col.length; i++) {
        c[i] = col[i] === true ? 1 : col[i] === false ? 0 : +col[i];
      }
      table[k] = { shape: [col.length], data: c };
    }
    return table;
  }
  // Empty array — empty vector.
  if (Array.isArray(data)) return { shape: [0], data: new Float64Array(0) };
  throw new Error('parseJSON: unsupported JSON shape; expected array-of-objects, '
    + 'object-of-arrays, or vector');
}

// =====================================================================
// CSV / WSV
// =====================================================================
//
// First row: column names.  Subsequent rows: numeric values.
//
// CSV: comma-separated, optional surrounding double-quotes on cells,
//      double-quote-escaped internal quotes ("a""b" → a"b). No multi-
//      line cells.
// WSV: any run of whitespace is a separator.
//
// Empty trailing rows are dropped. Lines starting with '#' are
// treated as comments (skipped) — common convention in scientific
// data sets.

function parseCSV(text: string): any {
  return _parseDelimited(text, ',');
}

function parseWSV(text: string): any {
  return _parseDelimited(text, /\s+/);
}

function _parseDelimited(text: string, sep: string | RegExp): any {
  const lines = text.split(/\r?\n/);
  // Filter out empty + comment lines.
  const rows: string[][] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === '' || trimmed.startsWith('#')) continue;
    rows.push(_splitRow(line, sep));
  }
  if (rows.length === 0) return { shape: [0], data: new Float64Array(0) };
  const header = rows[0];
  // Single-column case → vector (per spec §07 "Tabular data with a
  // single column can be loaded as a vector instead of a table,
  // depending on valueset"). Without the valueset we don't know;
  // return a vector of the second-row-onwards if there's exactly
  // one column.
  if (header.length === 1) {
    const data = new Float64Array(rows.length - 1);
    for (let i = 1; i < rows.length; i++) {
      data[i - 1] = +rows[i][0];
    }
    return { shape: [data.length], data };
  }
  // Multi-column → table.
  const table: Record<string, ValueType> = {};
  for (let c = 0; c < header.length; c++) {
    const col = new Float64Array(rows.length - 1);
    for (let i = 1; i < rows.length; i++) {
      const cell = rows[i][c];
      col[i - 1] = cell == null ? NaN : +cell;
    }
    table[header[c]] = { shape: [rows.length - 1], data: col };
  }
  return table;
}

function _splitRow(line: string, sep: string | RegExp): string[] {
  if (typeof sep !== 'string') {
    // Whitespace: trim + split on /\s+/.
    return line.trim().split(sep);
  }
  // CSV: simple split with quoted-cell handling.
  const out: string[] = [];
  let i = 0;
  while (i < line.length) {
    if (line[i] === '"') {
      // Quoted cell.
      let j = i + 1;
      let cell = '';
      while (j < line.length) {
        if (line[j] === '"') {
          if (line[j + 1] === '"') { cell += '"'; j += 2; continue; }
          break;
        }
        cell += line[j]; j++;
      }
      out.push(cell);
      i = j + 1;
      // Skip the separator after the closing quote, if any.
      if (line[i] === sep) i++;
      continue;
    }
    // Bare cell up to next separator.
    let j = i;
    while (j < line.length && line[j] !== sep) j++;
    out.push(line.slice(i, j));
    i = j + 1;     // skip separator
  }
  return out;
}

// =====================================================================
// Format dispatch
// =====================================================================
//
// Given a path + raw text, decide which parser to use and run it.
// Used by host-side resolvers that have already fetched the text.

function parseByExtension(path: string, text: string): any {
  const fmt = detectFormat(path);
  if (fmt === 'json') return parseJSON(text);
  if (fmt === 'csv')  return parseCSV(text);
  if (fmt === 'wsv')  return parseWSV(text);
  throw new Error('load_data: unsupported file extension for "' + path
    + '"; engine supports .json, .csv, .wsv (Arrow IPC deferred)');
}

module.exports = {
  detectFormat,
  parseJSON,
  parseCSV,
  parseWSV,
  parseByExtension,
};
