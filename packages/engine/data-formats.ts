'use strict';

// §07 load_data binary formats: Arrow IPC (file and stream) and Safetensors.
// Both decode to plain records and arrays whose leaves keep their source kind
// (`TypedScalar`), so the declared set checks them by §03 inclusion rather than
// by value. A column or tensor no set can select becomes `Unreadable`, which
// refuses only when it is selected. The Arrow reader covers the primitive
// column types §07 data holds (null, boolean, integer, float), so the engine
// needs no Arrow library.

/** A decoded scalar and the kind its source type declares. */
class TypedScalar {
  kind: 'real' | 'integer' | 'boolean';
  value: number | boolean;
  /** An integer outside the exact JavaScript range keeps its text. */
  inexact?: string;
  constructor(kind: 'real' | 'integer' | 'boolean', value: number | boolean, inexact?: string) {
    this.kind = kind;
    this.value = value;
    if (inexact !== undefined) this.inexact = inexact;
  }
}

/** A column or entry no value set can select, with the reason. */
class Unreadable {
  reason: string;
  constructor(reason: string) { this.reason = reason; }
}

function integer(value: bigint): TypedScalar {
  const n = Number(value);
  return Number.isSafeInteger(n) ? new TypedScalar('integer', n)
    : new TypedScalar('integer', n, value.toString());
}

function f16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function bf16(bits: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setUint32(0, bits << 16, true);
  return view.getFloat32(0, true);
}

// ---- Safetensors -----------------------------------------------------------

const TENSOR_DTYPES: Record<string, [number, (view: DataView, at: number) => TypedScalar | string]> = {
  F64: [8, (v, at) => new TypedScalar('real', v.getFloat64(at, true))],
  F32: [4, (v, at) => new TypedScalar('real', v.getFloat32(at, true))],
  F16: [2, (v, at) => new TypedScalar('real', f16(v.getUint16(at, true)))],
  BF16: [2, (v, at) => new TypedScalar('real', bf16(v.getUint16(at, true)))],
  I64: [8, (v, at) => integer(v.getBigInt64(at, true))],
  I32: [4, (v, at) => new TypedScalar('integer', v.getInt32(at, true))],
  I16: [2, (v, at) => new TypedScalar('integer', v.getInt16(at, true))],
  I8: [1, (v, at) => new TypedScalar('integer', v.getInt8(at))],
  U64: [8, (v, at) => {
    const x = v.getBigUint64(at, true);
    return x > 0x7fffffffffffffffn ? `U64 value ${x} exceeds int64` : integer(x);
  }],
  U32: [4, (v, at) => new TypedScalar('integer', v.getUint32(at, true))],
  U16: [2, (v, at) => new TypedScalar('integer', v.getUint16(at, true))],
  U8: [1, (v, at) => new TypedScalar('integer', v.getUint8(at))],
  BOOL: [1, (v, at) => {
    const x = v.getUint8(at);
    return x <= 1 ? new TypedScalar('boolean', x === 1) : `BOOL byte ${x} is neither 0 nor 1`;
  }],
};

/** Nest row-major leaves into arrays of `shape`; a scalar tensor is its leaf. */
function nest(leaves: any[], shape: number[], at = 0, depth = 0): any {
  if (depth === shape.length) return leaves[at];
  const stride = shape.slice(depth + 1).reduce((a, b) => a * b, 1);
  return Array.from({ length: shape[depth] }, (_, i) => nest(leaves, shape, at + i * stride, depth + 1));
}

function tensor(key: string, info: any, data: Uint8Array): any {
  const fail = (message: string): never => { throw new Error(`Safetensors: tensor \`${key}\`: ${message}`); };
  if (!info || typeof info.dtype !== 'string' || !Array.isArray(info.shape)
      || !info.shape.every((d: any) => Number.isSafeInteger(d) && d >= 0)
      || !Array.isArray(info.data_offsets) || info.data_offsets.length !== 2
      || !info.data_offsets.every((d: any) => Number.isSafeInteger(d) && d >= 0)) {
    fail('expects a dtype, a shape and two data offsets');
  }
  const entry = TENSOR_DTYPES[info.dtype];
  if (!entry) fail(`dtype ${info.dtype} has no FlatPPL value set (§07 maps float, integer and bool dtypes)`);
  const [size, read] = entry;
  const [begin, end] = info.data_offsets;
  if (begin > end || end > data.length) fail('data offsets lie outside the data buffer');
  const count = info.shape.reduce((a: number, b: number) => a * b, 1);
  if (!Number.isSafeInteger(count)) fail('element count overflows');
  const bytes = end - begin;
  if (bytes !== count * size) {
    fail(`${bytes} bytes hold ${Math.floor(bytes / size)} ${info.dtype} values, but the shape [${info.shape.join(', ')}] needs ${count}`);
  }
  const view = new DataView(data.buffer, data.byteOffset + begin, bytes);
  const leaves = Array.from({ length: count }, (_, i) => {
    const leaf = read(view, i * size);
    if (typeof leaf === 'string') fail(leaf);
    return leaf;
  });
  return nest(leaves, info.shape);
}

/** A Safetensors file as a nested record: a key's dot-separated segments form a
 * record path after leading and trailing dots are trimmed (§07). */
function readSafetensors(bytes: Uint8Array): any {
  const fail = (message: string): never => { throw new Error(`Safetensors: ${message}`); };
  if (bytes.length < 8) fail('missing the header length');
  const length = new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true);
  if (length > BigInt(bytes.length - 8)) fail('header length exceeds the file');
  const headerEnd = 8 + Number(length);
  let header: any;
  try { header = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(8, headerEnd))); }
  catch (e: any) { fail(`header: ${e.message}`); }
  if (!header || typeof header !== 'object' || Array.isArray(header)) fail('header: expected a JSON object');
  const data = bytes.subarray(headerEnd);
  const root: any = Object.create(null);
  const leaves = new Set<any>();
  const paths = new Set<string>();
  for (const key of Object.keys(header).sort()) {
    if (key === '__metadata__') continue;
    const path = key.replace(/^\.+|\.+$/g, '');
    if (!path) fail(`key \`${key}\` is empty after trimming its dots`);
    const segments = path.split('.');
    if (segments.some(s => !s)) fail(`key \`${key}\` has an empty path segment`);
    if (paths.has(path)) fail(`keys trimming to \`${path}\` name the same record path`);
    paths.add(path);
    const value = tensor(key, header[key], data);
    // A path that is both a tensor and a prefix of another cannot be loaded.
    let node = root;
    for (let depth = 0; depth < segments.length; depth++) {
      const segment = segments[depth];
      const last = depth === segments.length - 1;
      const existing = node[segment];
      if (last) {
        if (existing !== undefined) fail(`\`${path}\` is both a tensor and a record path prefix`);
        node[segment] = value;
        leaves.add(value);
      } else {
        if (existing !== undefined && leaves.has(existing)) {
          fail(`\`${segments.slice(0, depth + 1).join('.')}\` is both a tensor and a record path prefix`);
        }
        node = node[segment] ??= Object.create(null);
      }
    }
  }
  return root;
}

// ---- Arrow IPC ---------------------------------------------------------------

// The FlatBuffers tables of the Arrow format (Schema.fbs, Message.fbs, File.fbs),
// read by field index.
class Table {
  view: DataView;
  pos: number;
  vtable: number;
  constructor(view: DataView, pos: number) {
    this.view = view;
    this.pos = pos;
    this.vtable = pos - view.getInt32(pos, true);
  }
  static root(view: DataView, at = 0): Table { return new Table(view, at + view.getUint32(at, true)); }
  field(index: number): number {
    const slot = 4 + 2 * index;
    return slot < this.view.getUint16(this.vtable, true) ? this.view.getUint16(this.vtable + slot, true) : 0;
  }
  int(index: number, size: 1 | 2 | 4 | 8, fallback = 0): number {
    const at = this.field(index);
    if (!at) return fallback;
    const p = this.pos + at;
    if (size === 1) return this.view.getUint8(p);
    if (size === 2) return this.view.getInt16(p, true);
    if (size === 4) return this.view.getInt32(p, true);
    return Number(this.view.getBigInt64(p, true));
  }
  table(index: number): Table | null {
    const at = this.field(index);
    if (!at) return null;
    const p = this.pos + at;
    return new Table(this.view, p + this.view.getUint32(p, true));
  }
  vector(index: number): [number, number] | null {
    const at = this.field(index);
    if (!at) return null;
    const p = this.pos + at;
    const start = p + this.view.getUint32(p, true);
    return [start + 4, this.view.getUint32(start, true)];
  }
  tables(index: number): Table[] {
    const v = this.vector(index);
    if (!v) return [];
    const [start, n] = v;
    return Array.from({ length: n }, (_, i) => {
      const p = start + 4 * i;
      return new Table(this.view, p + this.view.getUint32(p, true));
    });
  }
  string(index: number): string {
    const v = this.vector(index);
    if (!v) return '';
    return new TextDecoder().decode(new Uint8Array(this.view.buffer, this.view.byteOffset + v[0], v[1]));
  }
}

const TYPE_NAMES = ['NONE', 'Null', 'Int', 'FloatingPoint', 'Binary', 'Utf8', 'Bool', 'Decimal',
  'Date', 'Time', 'Timestamp', 'Interval', 'List', 'Struct', 'Union', 'FixedSizeBinary',
  'FixedSizeList', 'Map', 'Duration', 'LargeBinary', 'LargeUtf8', 'LargeList', 'RunEndEncoded',
  'BinaryView', 'Utf8View', 'ListView', 'LargeListView'];

interface ArrowField {
  name: string;
  type: number;
  detail: Table | null;
  dictionary: boolean;
  children: ArrowField[];
}

function arrowField(t: Table): ArrowField {
  return { name: t.string(0), type: t.int(2, 1), detail: t.table(3),
    dictionary: t.table(4) !== null, children: t.tables(5).map(arrowField) };
}

/** The buffers a field occupies in a record batch, without its children
 * (Arrow columnar format, "Buffer listing for each layout"). */
function bufferCount(field: ArrowField, variadic: () => number): number {
  if (field.dictionary) return 2;
  switch (TYPE_NAMES[field.type]) {
    case 'Null': case 'RunEndEncoded': return 0;
    case 'Struct': case 'FixedSizeList': return 1;
    case 'Binary': case 'Utf8': case 'LargeBinary': case 'LargeUtf8':
    case 'ListView': case 'LargeListView': return 3;
    case 'BinaryView': case 'Utf8View': return 2 + variadic();
    case 'Union': return field.detail?.int(0, 2) === 1 ? 2 : 1;
    default: return 2;
  }
}

/** A column's reader, or why no value set reads it. */
function columnReader(field: ArrowField): ((nodes: any, buffer: (i: number) => Uint8Array) => any[]) | Unreadable {
  const name = TYPE_NAMES[field.type] || `type ${field.type}`;
  if (field.dictionary || ['Utf8', 'LargeUtf8', 'Utf8View'].includes(name)) {
    return new Unreadable(`column \`${field.name}\` holds Arrow strings, and \`load_data\` reads numbers and booleans`);
  }
  const bit = (bytes: Uint8Array, i: number) => (bytes[i >> 3] >> (i & 7)) & 1;
  const cells = (length: number, validity: Uint8Array, value: (i: number) => TypedScalar) =>
    Array.from({ length }, (_, i) => (validity.length && !bit(validity, i) ? null : value(i)));
  const view = (bytes: Uint8Array) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (name === 'Null') return (node: any) => Array.from({ length: node.length }, () => null);
  if (name === 'Bool') {
    return (node: any, buffer: any) => cells(node.length, buffer(0),
      i => new TypedScalar('boolean', bit(buffer(1), i) === 1));
  }
  if (name === 'Int') {
    const width = field.detail?.int(0, 4) ?? 0;
    const signed = field.detail?.int(1, 1) === 1;
    const get: any = {
      8: signed ? (v: DataView, i: number) => v.getInt8(i) : (v: DataView, i: number) => v.getUint8(i),
      16: signed ? (v: DataView, i: number) => v.getInt16(2 * i, true) : (v: DataView, i: number) => v.getUint16(2 * i, true),
      32: signed ? (v: DataView, i: number) => v.getInt32(4 * i, true) : (v: DataView, i: number) => v.getUint32(4 * i, true),
    }[width];
    if (get) return (node: any, buffer: any) => cells(node.length, buffer(0), i => new TypedScalar('integer', get(view(buffer(1)), i)));
    if (width === 64) {
      return (node: any, buffer: any) => cells(node.length, buffer(0), i => {
        const x = signed ? view(buffer(1)).getBigInt64(8 * i, true) : view(buffer(1)).getBigUint64(8 * i, true);
        if (x > 0x7fffffffffffffffn) throw new Error(`column \`${field.name}\` contains an integer outside int64`);
        return integer(x);
      });
    }
  }
  if (name === 'FloatingPoint') {
    const precision = field.detail?.int(0, 2) ?? 0;
    const read = [(v: DataView, i: number) => f16(v.getUint16(2 * i, true)),
      (v: DataView, i: number) => v.getFloat32(4 * i, true),
      (v: DataView, i: number) => v.getFloat64(8 * i, true)][precision];
    return (node: any, buffer: any) => cells(node.length, buffer(0), i => new TypedScalar('real', read(view(buffer(1)), i)));
  }
  return new Unreadable(`column \`${field.name}\` has unsupported Arrow type ${name}`);
}

/** The encapsulated messages of an IPC stream from `at`, up to its end marker. */
function* messages(bytes: Uint8Array, at: number): Generator<{ header: Table; body: Uint8Array }> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  while (at + 4 <= bytes.length) {
    let length = view.getInt32(at, true);
    at += 4;
    if (length === -1) {
      if (at + 4 > bytes.length) break;
      length = view.getInt32(at, true);
      at += 4;
    }
    if (length <= 0) return;
    const header = Table.root(view, at);
    at += length;
    const bodyLength = header.int(3, 8);
    yield { header, body: bytes.subarray(at, at + bodyLength) };
    at += bodyLength;
  }
}

/** An Arrow IPC file (through its footer) or stream as named columns. */
function readArrow(bytes: Uint8Array): any {
  try { return decodeArrow(bytes); }
  catch (e: any) {
    if (e instanceof RangeError) throw new Error(`Arrow IPC: malformed file (${e.message})`);
    throw e;
  }
}

function decodeArrow(bytes: Uint8Array): any {
  const fail = (message: string): never => { throw new Error(`Arrow IPC: ${message}`); };
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = (at: number) => new TextDecoder().decode(bytes.subarray(at, at + 6)) === 'ARROW1';
  let schema: Table | null = null;
  let batches: Array<{ header: Table; body: Uint8Array }> = [];
  if (magic(0)) {
    if (bytes.length < 18 || !magic(bytes.length - 6)) fail('a file must end with its footer and magic');
    const footerLength = view.getInt32(bytes.length - 10, true);
    const footer = Table.root(view, bytes.length - 10 - footerLength);
    schema = footer.table(1);
    const blocks = footer.vector(3);
    for (let i = 0; blocks && i < blocks[1]; i++) {
      const offset = Number(view.getBigInt64(blocks[0] + 24 * i, true));
      const next = messages(bytes, offset).next();
      if (!next.done) batches.push(next.value);
    }
  } else {
    for (const message of messages(bytes, 0)) {
      const kind = message.header.int(1, 1);
      if (kind === 1 && !schema) schema = message.header.table(2);
      else if (kind === 3) batches.push(message);
    }
  }
  if (!schema) fail('no schema');
  if (schema!.int(0, 2) !== 0) fail('big-endian data is not read');
  const fields = schema!.tables(1).map(arrowField);
  const readers = fields.map(columnReader);
  const columns: any[][] = fields.map(() => []);
  for (const { header, body } of batches) {
    const batch = header.table(2);
    if (!batch) continue;
    if (batch.table(3)) fail('compressed record batches are not read');
    const nodes = batch.vector(1)!;
    const buffers = batch.vector(2)!;
    const variadic = batch.vector(4);
    let node = 0, buffer = 0, variadicAt = 0;
    const nextVariadic = () => (variadic ? Number(view.getBigInt64(variadic[0] + 8 * variadicAt++, true)) : 0);
    const take = (f: ArrowField): { node: number; buffers: number[] } => {
      const own = { node: node++, buffers: Array.from({ length: bufferCount(f, nextVariadic) }, () => buffer++) };
      f.children.forEach(take);
      return own;
    };
    fields.forEach((f, i) => {
      const at = take(f);
      const reader = readers[i];
      if (reader instanceof Unreadable) return;
      const slice = (k: number) => {
        const entry = buffers[0] + 16 * at.buffers[k];
        const offset = Number(view.getBigInt64(entry, true));
        const size = Number(view.getBigInt64(entry + 8, true));
        if (offset + size > body.length) fail(`column \`${f.name}\` has a buffer outside its batch`);
        return body.subarray(offset, offset + size);
      };
      const length = Number(view.getBigInt64(nodes[0] + 16 * at.node, true));
      for (const cell of reader({ length }, slice)) columns[i].push(cell);
    });
  }
  const out: any = Object.create(null);
  fields.forEach((f, i) => { out[f.name] = readers[i] instanceof Unreadable ? readers[i] : columns[i]; });
  return out;
}

module.exports = { TypedScalar, Unreadable, readArrow, readSafetensors };
