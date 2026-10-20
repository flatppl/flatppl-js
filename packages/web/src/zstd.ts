/**
 * One-shot Zstandard (RFC 8878) with a dictionary. Runs the libzstd 1.5.6
 * WebAssembly build of @bokuweb/zstd-wasm 0.0.27 without that package's
 * JavaScript: this module supplies the five imports and calls libzstd directly.
 * The export names are minified, so they belong to that exact binary. The
 * build checks its SHA-256.
 */

const MAGIC = 0xfd2fb528;
const PAGE_BYTES = 64 * 1024;
const MAX_BLOCK_BYTES = 128 * 1024;
const ERROR_MEMORY_ALLOCATION = 64;
const ERROR_DST_SIZE_TOO_SMALL = 70;

class HeapExhausted extends Error {
  constructor() { super('Zstandard ran out of memory.'); }
}

export interface Zstd {
  /** A single frame with its content size, or null when it would exceed `maxBytes` or the heap. */
  compress(input: Uint8Array, dictionary: Uint8Array, level: number, maxBytes: number): Uint8Array | null;
  /** The content of exactly one frame, which `frameContentSize` accepts. */
  decompress(frame: Uint8Array, dictionary: Uint8Array, maxContentBytes: number): Uint8Array;
}

/**
 * Check the frame layout that share links use and return its content size.
 * Require one single-segment frame with a content size and nothing after it.
 * libzstd itself would also decode concatenated and skippable frames.
 */
export function frameContentSize(frame: Uint8Array, maxContentBytes: number): number {
  const invalid = () => new Error('The share link contains invalid compressed data.');
  const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  if (frame.length < 5 || view.getUint32(0, true) !== MAGIC) throw invalid();
  const descriptor = frame[4];
  // Single segment, no checksum, no dictionary ID, unused and reserved bits clear.
  if ((descriptor & 0x3f) !== 0x20) throw invalid();
  const sizeBytes = [1, 2, 4, 8][descriptor >> 6];
  let offset = 5 + sizeBytes;
  if (frame.length < offset) throw invalid();
  const size = sizeBytes === 1 ? frame[5]
    : sizeBytes === 2 ? view.getUint16(5, true) + 256
    : sizeBytes === 4 ? view.getUint32(5, true)
    : view.getUint32(9, true) ? Infinity : view.getUint32(5, true);
  if (size > maxContentBytes) throw new Error('The shared bundle exceeds the size limit. Open the ZIP bundle instead.');
  for (let last = 0; !last;) {
    if (frame.length < offset + 3) throw invalid();
    const header = frame[offset] | frame[offset + 1] << 8 | frame[offset + 2] << 16;
    const type = (header >> 1) & 3, blockBytes = header >>> 3;
    last = header & 1;
    if (type === 3 || blockBytes > MAX_BLOCK_BYTES) throw invalid();
    offset += 3 + (type === 1 ? 1 : blockBytes);  // An RLE block stores one byte.
  }
  if (offset !== frame.length) throw invalid();
  return size;
}

/** Instantiate libzstd. Memory grows on demand up to `maxHeapBytes`. */
export async function loadZstd(wasm: BufferSource, maxHeapBytes: number): Promise<Zstd> {
  let memory: WebAssembly.Memory | undefined;
  const abort = () => { throw new Error('Zstandard stopped unexpectedly.'); };
  const { instance } = await WebAssembly.instantiate(wasm, { a: {
    a: abort,  // proc_exit
    b: () => {},  // runtime_keepalive_clear
    c: abort,  // abort
    d: () => 0,  // setitimer
    e: (requested: number) => {  // emscripten_resize_heap
      const current = memory!.buffer.byteLength;
      requested >>>= 0;
      if (requested > maxHeapBytes) return 0;
      try { memory!.grow(Math.ceil((requested - current) / PAGE_BYTES)); return 1; } catch (_) { return 0; }
    },
  } });
  const x = instance.exports as any;
  memory = x.f as WebAssembly.Memory;
  x.g();  // static constructors
  const isError = (code: number) => x.h(code) !== 0;
  const errorCode = (code: number) => -(code | 0);
  const heap = () => new Uint8Array(memory!.buffer);

  /** Every caller frees the result in `finally`. */
  function alloc(size: number): number {
    const pointer = x.s(Math.max(size, 1)) >>> 0;
    if (!pointer) throw new HeapExhausted();
    return pointer;
  }
  function place(bytes: Uint8Array): number {
    const pointer = alloc(bytes.length);
    heap().set(bytes, pointer);
    return pointer;
  }

  return {
    compress(input, dictionary, level, maxBytes) {
      const capacity = Math.min(x.i(input.length) >>> 0, maxBytes);
      let context = 0, source = 0, dict = 0, target = 0;
      try {
        context = x.j() >>> 0;
        if (!context) return null;
        source = place(input);
        dict = place(dictionary);
        target = alloc(capacity);
        const written = x.l(context, target, capacity, source, input.length, dict, dictionary.length, level);
        if (isError(written)) {
          if ([ERROR_DST_SIZE_TOO_SMALL, ERROR_MEMORY_ALLOCATION].includes(errorCode(written))) return null;
          throw new Error('Zstandard compression failed.');
        }
        return heap().slice(target, target + (written >>> 0));
      } catch (error) {
        if (error instanceof HeapExhausted) return null;
        throw error;
      } finally {
        x.t(target); x.t(dict); x.t(source); x.k(context);
      }
    },
    decompress(frame, dictionary, maxContentBytes) {
      const size = frameContentSize(frame, maxContentBytes);
      let context = 0, source = 0, dict = 0, target = 0;
      try {
        context = x.n() >>> 0;
        if (!context) throw new HeapExhausted();
        source = place(frame);
        dict = place(dictionary);
        target = alloc(size);
        const written = x.q(context, target, size, source, frame.length, dict, dictionary.length);
        if (isError(written) || written >>> 0 !== size) throw new Error('The share link contains invalid compressed data.');
        return heap().slice(target, target + size);
      } finally {
        x.t(target); x.t(dict); x.t(source); x.o(context);
      }
    },
  };
}
