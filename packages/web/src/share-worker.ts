/**
 * Brotli work for the browser share transport. §04 ZIP contents stay opaque
 * here. Stream through fixed buffers so declared or adversarial expansion
 * cannot make the decoder allocate an unbounded JS output buffer.
 */
import brotliReady from 'brotli-wasm';
import {
  MAX_SHARE_ARCHIVE_BYTES, MAX_SHARE_COMPRESSED_BYTES,
  type ShareWorkerRequest, type ShareWorkerReply,
} from './share-codec';

const CHUNK_BYTES = 64 * 1024;
const worker = self as unknown as {
  addEventListener: (type: 'message', listener: (event: MessageEvent<ShareWorkerRequest>) => void) => void;
  postMessage: (message: ShareWorkerReply, transfer?: Transferable[]) => void;
};

/** Release WASM results promptly. Concatenate only after the actual size passes. */
async function transform(operation: ShareWorkerRequest['operation'], input: Uint8Array): Promise<Uint8Array | null> {
  const encode = operation === 'encode';
  const inputLimit = encode ? MAX_SHARE_ARCHIVE_BYTES : MAX_SHARE_COMPRESSED_BYTES;
  const outputLimit = encode ? MAX_SHARE_COMPRESSED_BYTES : MAX_SHARE_ARCHIVE_BYTES;
  if (input.byteLength > inputLimit) return null;
  const brotli = await brotliReady;
  const stream = encode ? new brotli.CompressStream(11) : new brotli.DecompressStream();
  const chunks: Uint8Array[] = [];
  let offset = 0, total = 0;
  try {
    while (true) {
      const source = input.subarray(offset, offset + CHUNK_BYTES);
      const result = encode
        ? (stream as InstanceType<typeof brotli.CompressStream>).compress(offset === input.length ? undefined : source, CHUNK_BYTES)
        : (stream as InstanceType<typeof brotli.DecompressStream>).decompress(source, CHUNK_BYTES);
      let code: number, consumed: number, bytes: Uint8Array;
      try {
        code = result.code;
        consumed = result.input_offset;
        bytes = result.buf;
      } finally { result.free(); }
      offset += consumed;
      total += bytes.length;
      if (total > outputLimit) return null;
      if (bytes.length) chunks.push(bytes);
      if (code === brotli.BrotliStreamResultCode.ResultSuccess) {
        if (offset !== input.length) throw new Error('The share contains trailing compressed data.');
        break;
      }
      if (!encode && code === brotli.BrotliStreamResultCode.NeedsMoreInput && offset === input.length) {
        throw new Error('The share link is incomplete.');
      }
      if (!consumed && !bytes.length) {
        throw new Error('The share link contains invalid compressed data.');
      }
    }
  } finally { stream.free(); }
  const output = new Uint8Array(total);
  let position = 0;
  for (const chunk of chunks) { output.set(chunk, position); position += chunk.length; }
  return output;
}

worker.addEventListener('message', async ({ data }) => {
  try {
    if (!(data.bytes instanceof Uint8Array) || !['encode', 'decode'].includes(data.operation)) {
      throw new Error('Invalid share codec request.');
    }
    const bytes = await transform(data.operation, data.bytes);
    worker.postMessage({ id: data.id, bytes }, bytes ? [bytes.buffer as ArrayBuffer] : []);
  } catch (error) {
    worker.postMessage({ id: data.id, error: error instanceof Error ? error.message : 'Invalid compressed share data.' });
  }
});
