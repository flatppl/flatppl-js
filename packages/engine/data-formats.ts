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
  /** A dictionary-encoded field's id and key type; `type` is then the value type. */
  dictionary: { id: number; index: Table | null } | null;
  children: ArrowField[];
}

function arrowField(t: Table): ArrowField {
  const encoding = t.table(4);
  return { name: t.string(0), type: t.int(2, 1), detail: t.table(3),
    dictionary: encoding ? { id: encoding.int(0, 8), index: encoding.table(1) } : null,
    children: t.tables(5).map(arrowField) };
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

type ColumnReader = (node: { length: number }, buffer: (i: number) => Uint8Array) => any[];

const bit = (bytes: Uint8Array, i: number) => (bytes[i >> 3] >> (i & 7)) & 1;
const cells = (length: number, validity: Uint8Array, value: (i: number) => TypedScalar) => {
  // A short bitmap would read as nulls past its end and never stop at the data.
  if (validity.length && validity.length * 8 < length) throw new RangeError('a validity bitmap is shorter than its column');
  return Array.from({ length }, (_, i) => (validity.length && !bit(validity, i) ? null : value(i)));
};
const dataView = (bytes: Uint8Array) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

/** A column's reader, or why no value set reads it. A reader throws an
 * `Unreadable` for a value that spoils only its own column. */
function columnReader(field: ArrowField, dictionaries: Map<number, any[] | Unreadable>): ColumnReader | Unreadable {
  const name = TYPE_NAMES[field.type] || `type ${field.type}`;
  if (['Utf8', 'LargeUtf8', 'Utf8View'].includes(name)) {
    return new Unreadable(`column \`${field.name}\` holds Arrow strings, and \`load_data\` reads numbers and booleans`);
  }
  if (field.dictionary) {
    // §07 types a column by its values, so a dictionary over numbers or
    // booleans reads as those values. Keys default to int32 (Schema.fbs).
    const values = columnReader({ ...field, dictionary: null }, dictionaries);
    if (values instanceof Unreadable) return values;
    const { id, index } = field.dictionary;
    const keys = intReader(field.name, index ? index.int(0, 4) : 32, index ? index.int(1, 1) === 1 : true)!;
    return (node, buffer) => {
      const dictionary = dictionaries.get(id);
      if (dictionary instanceof Unreadable) throw dictionary;
      return keys(node, buffer).map((key: TypedScalar | null) => {
        if (key === null) return null;
        const value = dictionary?.[key.value as number];
        if (value === undefined) throw new RangeError(`column \`${field.name}\` has a dictionary key outside its dictionary`);
        return value;
      });
    };
  }
  // No buffer bounds a Null array's length, and every value is missing, so
  // selecting it refuses anyway. It is never materialised.
  if (name === 'Null') return new Unreadable(`column \`${field.name}\` has the Arrow type Null, so every value is missing`);
  if (name === 'Bool') {
    return (node: any, buffer: any) => {
      if (buffer(1).length * 8 < node.length) throw new RangeError('a boolean buffer is shorter than its column');
      return cells(node.length, buffer(0), i => new TypedScalar('boolean', bit(buffer(1), i) === 1));
    };
  }
  if (name === 'Int') {
    const reader = intReader(field.name, field.detail?.int(0, 4) ?? 0, field.detail?.int(1, 1) === 1);
    if (reader) return reader;
  }
  if (name === 'FloatingPoint') {
    const precision = field.detail?.int(0, 2) ?? 0;
    const read = [(v: DataView, i: number) => f16(v.getUint16(2 * i, true)),
      (v: DataView, i: number) => v.getFloat32(4 * i, true),
      (v: DataView, i: number) => v.getFloat64(8 * i, true)][precision];
    return (node: any, buffer: any) => cells(node.length, buffer(0), i => new TypedScalar('real', read(dataView(buffer(1)), i)));
  }
  return new Unreadable(`column \`${field.name}\` has unsupported Arrow type ${name}`);
}

function intReader(column: string, width: number, signed: boolean): ColumnReader | null {
  const get: any = {
    8: signed ? (v: DataView, i: number) => v.getInt8(i) : (v: DataView, i: number) => v.getUint8(i),
    16: signed ? (v: DataView, i: number) => v.getInt16(2 * i, true) : (v: DataView, i: number) => v.getUint16(2 * i, true),
    32: signed ? (v: DataView, i: number) => v.getInt32(4 * i, true) : (v: DataView, i: number) => v.getUint32(4 * i, true),
  }[width];
  if (get) return (node, buffer) => cells(node.length, buffer(0), i => new TypedScalar('integer', get(dataView(buffer(1)), i)));
  if (width !== 64) return null;
  return (node, buffer) => cells(node.length, buffer(0), i => {
    const x = signed ? dataView(buffer(1)).getBigInt64(8 * i, true) : dataView(buffer(1)).getBigUint64(8 * i, true);
    if (x > 0x7fffffffffffffffn) throw new Unreadable(`column \`${column}\` contains an integer outside int64`);
    return integer(x);
  });
}

// ---- LZ4 (the Arrow IPC LZ4_FRAME buffer codec) ----------------------------

const lz4Corrupt = (what: string): never => { throw new RangeError(`LZ4 data ${what}`); };

/** A length that continues in 255-valued bytes (LZ4 block format). */
function lz4Length(src: Uint8Array, at: { ip: number }, length: number): number {
  if (length !== 15) return length;
  let byte;
  do {
    if (at.ip >= src.length) lz4Corrupt('is cut short');
    byte = src[at.ip++];
    length += byte;
  } while (byte === 255);
  return length;
}

/** Decode one LZ4 block into `out` at `op`, and return the new end. A match
 * may reach back into earlier blocks of the frame. */
function lz4Block(src: Uint8Array, out: Uint8Array, op: number): number {
  const at = { ip: 0 };
  while (at.ip < src.length) {
    const token = src[at.ip++];
    const literals = lz4Length(src, at, token >> 4);
    if (at.ip + literals > src.length || op + literals > out.length) lz4Corrupt('overruns its block');
    out.set(src.subarray(at.ip, at.ip + literals), op);
    at.ip += literals;
    op += literals;
    if (at.ip === src.length) break;  // the last sequence holds only literals
    if (at.ip + 2 > src.length) lz4Corrupt('is cut short');
    const offset = src[at.ip] | (src[at.ip + 1] << 8);
    at.ip += 2;
    if (offset === 0 || offset > op) lz4Corrupt('refers before its start');
    const match = lz4Length(src, at, token & 15) + 4;
    if (op + match > out.length) lz4Corrupt('overruns its stated size');
    for (const end = op + match; op < end; op++) out[op] = out[op - offset];
  }
  return op;
}

/** Decode LZ4 frames (the LZ4 frame format) of `size` bytes in all. */
function lz4Frames(src: Uint8Array, size: number): Uint8Array {
  const out = new Uint8Array(size);
  const view = dataView(src);
  let ip = 0, op = 0;
  while (ip < src.length) {
    if (ip + 7 > src.length || view.getUint32(ip, true) !== 0x184d2204) lz4Corrupt('has no frame magic number');
    const flags = src[ip + 4];
    if (flags >> 6 !== 1) lz4Corrupt('has an unknown frame version');
    if (flags & 1) lz4Corrupt('needs a dictionary');
    // Magic, FLG and BD, then the optional content size, and the header checksum.
    ip += 6 + (flags & 8 ? 8 : 0) + 1;
    for (;;) {
      if (ip + 4 > src.length) lz4Corrupt('is cut short');
      const word = view.getUint32(ip, true);
      ip += 4;
      if (word === 0) break;
      const length = word & 0x7fffffff;
      if (ip + length > src.length) lz4Corrupt('is cut short');
      const block = src.subarray(ip, ip + length);
      if (word & 0x80000000) {
        if (op + length > size) lz4Corrupt('overruns its stated size');
        out.set(block, op);
        op += length;
      } else {
        op = lz4Block(block, out, op);
      }
      ip += length + (flags & 0x10 ? 4 : 0);  // the optional block checksum
    }
    ip += flags & 4 ? 4 : 0;  // the optional content checksum
  }
  if (op !== size) lz4Corrupt('does not hold its stated size');
  return out;
}

/** An Arrow IPC buffer under LZ4_FRAME: an int64 decompressed length, -1 for
 * a buffer stored as is, then the frames. */
function lz4Buffer(raw: Uint8Array): Uint8Array {
  if (raw.length === 0) return raw;
  if (raw.length < 8) lz4Corrupt('has no length prefix');
  const size = dataView(raw).getBigInt64(0, true);
  const data = raw.subarray(8);
  if (size === -1n) return data;
  // LZ4 expands at most about 255 times, so a larger size is corrupt.
  if (size < 0n || size > BigInt(256 * data.length + 64)) lz4Corrupt('states an impossible size');
  return lz4Frames(data, Number(size));
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
    if (length === 0) return;
    // Every step must advance inside the file, or a crafted length loops forever.
    if (length < 0 || at + length > bytes.length) throw new RangeError('a message length leaves the file');
    const header = Table.root(view, at);
    at += length;
    const bodyLength = header.int(3, 8);
    if (bodyLength < 0 || at + bodyLength > bytes.length) throw new RangeError('a message body leaves the file');
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
  const malformed = (message: string): never => fail(`malformed file (${message})`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = (at: number) => new TextDecoder().decode(bytes.subarray(at, at + 6)) === 'ARROW1';
  let schema: Table | null = null;
  // Dictionary and record batches, in the order they apply.
  const batches: Array<{ header: Table; body: Uint8Array }> = [];
  if (magic(0)) {
    if (bytes.length < 18 || !magic(bytes.length - 6)) malformed('a file must end with its footer and magic');
    const footerLength = view.getInt32(bytes.length - 10, true);
    if (footerLength <= 0 || footerLength > bytes.length - 18) malformed('the footer length leaves the file');
    const footer = Table.root(view, bytes.length - 10 - footerLength);
    schema = footer.table(1);
    // A file has no dictionary replacements, so every dictionary precedes the batches.
    for (const index of [2, 3]) {
      const blocks = footer.vector(index);
      for (let i = 0; blocks && i < blocks[1]; i++) {
        const offset = Number(view.getBigInt64(blocks[0] + 24 * i, true));
        if (!(offset >= 0 && offset < bytes.length)) malformed('a footer block leaves the file');
        const next = messages(bytes, offset).next();
        if (next.done) continue;
        // A reader that slices by the block must find the message's own lengths there.
        const metadata = next.value.body.byteOffset - bytes.byteOffset - offset;
        if (view.getInt32(blocks[0] + 24 * i + 8, true) !== metadata
            || Number(view.getBigInt64(blocks[0] + 24 * i + 16, true)) !== next.value.body.length) {
          malformed('a footer block disagrees with its message');
        }
        batches.push(next.value);
      }
    }
  } else {
    for (const message of messages(bytes, 0)) {
      const kind = message.header.int(1, 1);
      if (kind === 1 && !schema) schema = message.header.table(2);
      else if (kind === 2 || kind === 3) batches.push(message);
    }
  }
  if (!schema) malformed('no schema');
  if (schema!.int(0, 2) !== 0) fail('big-endian data is not read');
  const fields = schema!.tables(1).map(arrowField);
  const dictionaries = new Map<number, any[] | Unreadable>();
  const readers = fields.map(f => columnReader(f, dictionaries));
  const columns: any[][] = fields.map(() => []);
  const valueFields = new Map<number, ArrowField>();
  const collect = (f: ArrowField): void => {
    if (f.dictionary) valueFields.set(f.dictionary.id, { ...f, dictionary: null });
    f.children.forEach(collect);
  };
  fields.forEach(collect);

  // Each field's cells, or the `Unreadable` a value in it threw.
  const decode = (batch: Table, body: Uint8Array, batchFields: ArrowField[],
    batchReaders: Array<ColumnReader | Unreadable>): Array<any[] | Unreadable | null> => {
    // BodyCompression: codec 0 is LZ4_FRAME, 1 is ZSTD.
    const compression = batch.table(3);
    const lz4 = compression !== null && compression.int(0, 1) === 0;
    if (compression && !lz4) {
      throw new Error(compression.int(0, 1) === 1 ? 'zstd-compressed Arrow IPC is not supported by this engine'
        : `Arrow IPC: unknown compression codec ${compression.int(0, 1)}`);
    }
    const nodes = batch.vector(1)!;
    const buffers = batch.vector(2)!;
    const variadic = batch.vector(4);
    let node = 0, buffer = 0, variadicAt = 0;
    const nextVariadic = () => (variadic ? Number(view.getBigInt64(variadic[0] + 8 * variadicAt++, true)) : 0);
    // A dictionary field's children describe its values, which a dictionary batch holds.
    const take = (f: ArrowField): { node: number; buffers: number[] } => {
      const own = { node: node++, buffers: Array.from({ length: bufferCount(f, nextVariadic) }, () => buffer++) };
      if (!f.dictionary) f.children.forEach(take);
      return own;
    };
    return batchFields.map((f, i) => {
      const at = take(f);
      const reader = batchReaders[i];
      if (reader instanceof Unreadable) return null;
      const decompressed = new Map<number, Uint8Array>();
      const slice = (k: number) => {
        const entry = buffers[0] + 16 * at.buffers[k];
        const offset = Number(view.getBigInt64(entry, true));
        const size = Number(view.getBigInt64(entry + 8, true));
        if (offset < 0 || size < 0 || offset + size > body.length) malformed(`column \`${f.name}\` has a buffer outside its batch`);
        const raw = body.subarray(offset, offset + size);
        if (!lz4) return raw;
        if (!decompressed.has(k)) decompressed.set(k, lz4Buffer(raw));
        return decompressed.get(k)!;
      };
      const length = Number(view.getBigInt64(nodes[0] + 16 * at.node, true));
      if (length < 0) malformed(`column \`${f.name}\` has a negative length`);
      try { return reader({ length }, slice); }
      catch (e) {
        if (e instanceof Unreadable) return e;
        throw e;
      }
    });
  };

  for (const { header, body } of batches) {
    const content = header.table(2);
    if (!content) continue;
    if (header.int(1, 1) === 2) {
      const id = content.int(0, 8);
      const field = valueFields.get(id);
      const batch = content.table(1);
      if (!field || !batch) malformed(`dictionary batch ${id} matches no dictionary field`);
      const [cells] = decode(batch!, body, [field!], [columnReader(field!, dictionaries)]);
      // No reader means the dictionary column itself is already unreadable.
      const previous = dictionaries.get(id);
      if (cells === null || previous instanceof Unreadable) continue;
      const delta = content.int(2, 1) === 1;
      dictionaries.set(id, cells instanceof Unreadable || !delta || !previous ? cells : previous.concat(cells));
      continue;
    }
    decode(content, body, fields, readers).forEach((cells, i) => {
      if (cells instanceof Unreadable) readers[i] = cells;
      else if (cells) for (const cell of cells) columns[i].push(cell);
    });
  }
  const out: any = Object.create(null);
  fields.forEach((f, i) => { out[f.name] = readers[i] instanceof Unreadable ? readers[i] : columns[i]; });
  return out;
}

module.exports = { TypedScalar, Unreadable, readArrow, readSafetensors };
