'use strict';

// §07 load_data: text decoding does not infer a FlatPPL shape. The declared
// set selects and validates one value, then constructs ordinary fixed-value
// syntax shared by inference, linking and execution. Hosts own all I/O.
const AST = require('./ast.ts');
const { validateFixedVectorInput } = require('./fixed-inputs.ts');
const { inSet } = require('./density.ts')._internal;

function detectFormat(path: string): 'json' | 'csv' | 'wsv' | null {
  const lower = path.split(/[?#]/, 1)[0].toLowerCase();
  if (lower.endsWith('.json')) return 'json';
  if (lower.endsWith('.csv')) return 'csv';
  if (lower.endsWith('.wsv')) return 'wsv';
  return null;
}

function parseJSON(text: string): any { return JSON.parse(text); }
function parseCSV(text: string): any { return parseDelimited(text, ','); }
function parseWSV(text: string): any { return parseDelimited(text, /\s+/); }

// Keep cells as text until their selected scalar schema is known. In
// particular, an empty cell is missing data, not JavaScript Number('') = 0.
function parseDelimited(text: string, separator: string | RegExp): any {
  const rows = text.split(/\r?\n/).filter(line => line.trim() && !line.trim().startsWith('#'))
    .map(line => splitRow(line, separator));
  if (!rows.length) throw new Error('load_data: missing column headers');
  const header = rows[0].map(cell => cell.trim());
  if (header.some(name => !name) || new Set(header).size !== header.length) {
    throw new Error('load_data: column headers must be nonempty and distinct');
  }
  const columns: Record<string, Array<string | undefined>> = Object.create(null);
  for (let c = 0; c < header.length; c++) {
    columns[header[c]] = rows.slice(1).map(row => row[c]);
  }
  return columns;
}

function splitRow(line: string, separator: string | RegExp): string[] {
  if (typeof separator !== 'string') return line.trim().split(separator);
  const cells: string[] = [];
  let cell = '', quoted = false, closed = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quoted) {
      if (c !== '"') cell += c;
      else if (line[i + 1] === '"') { cell += '"'; i++; }
      else { quoted = false; closed = true; }
    } else if (c === separator) {
      cells.push(cell); cell = ''; closed = false;
    } else if (c === '"' && !cell.trim() && !closed) {
      cell = ''; quoted = true;
    } else if (closed && !/\s/.test(c)) {
      throw new Error('load_data: unexpected text after a quoted CSV cell');
    } else cell += c;
  }
  if (quoted) throw new Error('load_data: unterminated quoted CSV cell');
  cells.push(cell);
  return cells;
}

function parseByExtension(path: string, text: string): any {
  const format = detectFormat(path);
  if (format === 'json') return parseJSON(text);
  if (format === 'csv') return parseCSV(text);
  if (format === 'wsv') return parseWSV(text);
  throw new Error(`load_data: unsupported file extension for '${path}'; this engine supports JSON, CSV and WSV text (Arrow IPC and Safetensors are not implemented)`);
}

// Resolve schemas separately from values: a named set is not an ordinary
// evaluated value, whereas its bounds and dimensions are fixed values.
function resolveSchema(ir: any, resolveValue: (ir: any) => any,
  resolveSet: (ir: any) => any, depth = 0): any {
  if (!ir || depth > 64) throw new Error('load_data: valueset must be a fixed, supported set');
  if (ir.kind === 'ref') return resolveSchema(resolveSet(ir), resolveValue, resolveSet, depth + 1);
  if (ir.kind === 'const') return { kind: ir.name };
  if (ir.kind !== 'call') throw new Error('load_data: valueset must be a set');
  const args = ir.args || [];
  const child = (s: any) => resolveSchema(s, resolveValue, resolveSet, depth + 1);
  const dimension = (s: any): number => {
    const n = resolveValue(s);
    if (!Number.isSafeInteger(n) || n < 0) throw new Error('load_data: dimensions must be fixed nonnegative integers');
    return n;
  };
  if (ir.op === 'interval') {
    const lo = resolveValue(args[0]), hi = resolveValue(args[1]);
    if (typeof lo !== 'number' || typeof hi !== 'number' || !(lo <= hi)) {
      throw new Error('load_data: interval bounds must be fixed ordered real values');
    }
    return { kind: 'interval', lo, hi };
  }
  if (['ordered', 'zerosum', 'stdsimplex'].includes(ir.op)) {
    return { kind: ir.op, n: dimension(args[0]) };
  }
  if (ir.op === 'cartprod') {
    return ir.fields ? { kind: 'record', fields: ir.fields.map((f: any) => ({ name: f.name, schema: child(f.value) })) }
      : { kind: 'product', elements: args.map(child) };
  }
  if (ir.op === 'cartpow') {
    const size = resolveValue(args[1]);
    const shape = typeof size === 'number' ? [size]
      : size?.shape?.length === 1 ? Array.from(size.data) : null;
    if (!shape || shape.some((n: any) => !Number.isSafeInteger(n) || n < 0)) {
      throw new Error('load_data: dimensions must be a fixed integer or integer vector');
    }
    return { kind: 'array', shape, element: child(args[0]) };
  }
  throw new Error(`load_data: unsupported valueset '${ir.op}'`);
}

function containsAnything(schema: any): boolean {
  return schema.kind === 'anything' || (schema.element && containsAnything(schema.element))
    || schema.fields?.some((f: any) => containsAnything(f.schema))
    || schema.elements?.some(containsAnything) || false;
}

/** Produce a fixed RHS without changing the original source AST. */
function valueExpression(raw: any, schema: any, loc: any, textCells = false, at = 'value'): any {
  const fail = (reason: string): never => { throw new Error(`load_data: ${at} ${reason}`); };
  const call = (name: string, args: any[]) => ({ ...AST.CallExpr(AST.Identifier(name, loc), args, loc), builtin: true });
  const number = (n: number, real = false) => AST.NumberLiteral(n,
    real && Number.isInteger(n) ? `${n}.0` : String(n), loc);
  const vector = (items: any[]) => AST.ArrayLiteral(items, loc);
  const convert = (v: any, s: any, path: string) => valueExpression(v, s, loc, textCells, path);
  const field = (object: any, name: string) => {
    if (!object || typeof object !== 'object' || Array.isArray(object)
        || !Object.prototype.hasOwnProperty.call(object, name)) fail(`is missing field '${name}'`);
    return object[name];
  };
  if (schema.kind === 'record') {
    return call('record', schema.fields.map((f: any) =>
      AST.KeywordArg(f.name, convert(field(raw, f.name), f.schema, `${at}.${f.name}`), loc)));
  }
  if (schema.kind === 'product') {
    // §03 positional products concatenate component vectors, rather than
    // nesting them. Consume those contiguous components from the source.
    const length = (s: any): number => {
      if (s.kind === 'product') return s.elements.reduce((n: number, child: any) => n + length(child), 0);
      if (s.kind === 'array' && s.shape.length === 1) return s.shape[0];
      if (['ordered', 'zerosum', 'stdsimplex'].includes(s.kind)) return s.n;
      if (s.kind === 'array' || s.kind === 'record') fail('uses an unsupported positional product shape');
      return 1;
    };
    const lengths = schema.elements.map(length);
    if (!Array.isArray(raw) || raw.length !== lengths.reduce((a: number, b: number) => a + b, 0)) fail('does not match the product shape');
    let offset = 0;
    return call('cat', schema.elements.map((s: any, i: number) => {
      const array = ['array', 'product', 'ordered', 'zerosum', 'stdsimplex'].includes(s.kind);
      const value = array ? raw.slice(offset, offset + lengths[i]) : raw[offset];
      offset += lengths[i];
      return convert(value, s, `${at}[${i + 1}]`);
    }));
  }
  if (schema.kind === 'array') {
    const shape: number[] = schema.shape;
    if (shape.length === 1 && schema.element.kind === 'record') {
      const n = shape[0];
      return call('table', schema.element.fields.map((f: any) => {
        const column = Array.isArray(raw)
          ? raw.slice(0, n).map((row, i) => {
            if (!row || !Object.prototype.hasOwnProperty.call(row, f.name)) fail(`row ${i + 1} is missing field '${f.name}'`);
            return row[f.name];
          }) : field(raw, f.name);
        return AST.KeywordArg(f.name, convert(column,
          { kind: 'array', shape: [n], element: f.schema }, `${at}.${f.name}`), loc);
      }));
    }
    // A single delimited column may be selected as a vector, per §07.
    if (textCells && shape.length === 1 && raw && !Array.isArray(raw)
        && typeof raw === 'object' && Object.keys(raw).length === 1) raw = raw[Object.keys(raw)[0]];
    const leaves: any[] = [];
    function descend(value: any, axis: number, path: string): void {
      if (axis === shape.length) { leaves.push(convert(value, schema.element, path)); return; }
      if (!Array.isArray(value) || value.length < shape[axis]
          || (shape.length !== 1 && value.length !== shape[axis])) fail(`does not match shape [${shape}] at ${path}`);
      for (let i = 0; i < shape[axis]; i++) descend(value[i], axis + 1, `${path}[${i + 1}]`);
    }
    descend(raw, 0, at);
    if (shape.length === 1) return vector(leaves);
    if (['array', 'product', 'record', 'ordered', 'zerosum', 'stdsimplex', 'complexes'].includes(schema.element.kind)) {
      fail('requires a multidimensional array with unsupported element structure');
    }
    if (shape.length === 2 && shape[0] > 0) {
      const rows = Array.from({ length: shape[0] }, (_, i) => vector(leaves.slice(i * shape[1], (i + 1) * shape[1])));
      return call('rowstack', [vector(rows)]);
    }
    return call('array', [vector(leaves), vector(shape.map(n => number(n))),
      vector(shape.map((_, i) => number(i + 1)))]);
  }
  if (['ordered', 'zerosum', 'stdsimplex'].includes(schema.kind)) {
    if (!Array.isArray(raw) || raw.length !== schema.n) fail(`does not match ${schema.kind}(${schema.n})`);
    const elements = raw.map((v: any, i: number) => convert(v, { kind: 'reals' }, `${at}[${i + 1}]`));
    const values = elements.map((e: any) => e.value);
    try { validateFixedVectorInput(at, values, { vs: schema.kind, n: schema.n }); }
    catch (_) { fail(`is outside ${schema.kind}(${schema.n})`); }
    return vector(elements);
  }
  if (textCells && typeof raw === 'string') {
    const cell = raw.trim();
    if (schema.kind === 'booleans' && /^(true|false)$/i.test(cell)) raw = cell.toLowerCase() === 'true';
    else if (cell && /^[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?)$/i.test(cell)) {
      raw = /inf/i.test(cell) ? (cell.startsWith('-') ? -Infinity : Infinity) : Number(cell);
    } else fail('is missing or is not a scalar of its declared set');
  }
  if (schema.kind === 'booleans') {
    if (typeof raw !== 'boolean') fail('is not a boolean');
    return AST.BoolLiteral(raw, loc);
  }
  if (typeof raw !== 'number' || Number.isNaN(raw)) fail(`is not a number in ${schema.kind}`);
  if (schema.kind === 'complexes') return call('complex', [number(raw, true), number(0, true)]);
  if (!inSet(raw, schema)) fail(`is outside ${schema.kind}`);
  if (schema.kind.endsWith('integers') && !Number.isSafeInteger(raw)) fail('is outside the exact JavaScript integer range');
  return number(raw, !schema.kind.endsWith('integers'));
}

module.exports = { detectFormat, parseJSON, parseCSV, parseWSV, parseByExtension,
  resolveSchema, containsAnything, valueExpression };
