import { before, test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { loadZstd } from '../src/zstd.ts';

const encode = text => new TextEncoder().encode(text);
// The package exports no path to its binary. The build reads the same file.
const wasmPath = new URL('../../../node_modules/@bokuweb/zstd-wasm/dist/web/zstd.wasm', import.meta.url);
const LINE = 'y ~ Normal(mu = 0.0, sigma = 1.0)\n';
// Bytes without the dictionary magic number serve as raw dictionary content.
const dictionary = encode(LINE.repeat(4));
let zstd;

before(async () => { zstd = await loadZstd(await readFile(wasmPath), 128 * 2 ** 20); });

test('only one single-segment frame without trailing data decodes', () => {
  const content = encode('x = 1\n');
  const frame = zstd.compress(content, dictionary, 19, 1024);
  const decode = bytes => zstd.decompress(bytes, dictionary, 1024);
  const changed = (index, value) => { const copy = frame.slice(); copy[index] = value; return copy; };
  const concat = (...parts) => Uint8Array.from(parts.flatMap(part => Array.from(part)));
  assert.deepEqual(decode(frame), content);
  assert.equal(frame[4], 0x20);  // Single segment, one-byte content size.
  const rejected = {
    'trailing byte': concat(frame, [0]),
    'second frame': concat(frame, zstd.compress(new Uint8Array(), dictionary, 19, 1024)),
    'skippable frame': concat(frame, [0x50, 0x2a, 0x4d, 0x18, 0, 0, 0, 0]),
    'checksum': concat(changed(4, 0x24), [0, 0, 0, 0]),
    'dictionary ID': changed(4, 0x21),
    'no single segment': changed(4, 0x00),
    'reserved bit': changed(4, 0x28),
    'understated size': changed(5, content.length - 1),
    'overstated size': changed(5, content.length + 1),
    'truncated': frame.subarray(0, -1),
    'reserved block type': changed(6, frame[6] | 0b110),
  };
  for (const [name, bytes] of Object.entries(rejected)) assert.throws(() => decode(bytes), Error, name);
  assert.throws(() => zstd.decompress(frame, dictionary, content.length - 1), /size limit/);
  const huge = concat(frame.subarray(0, 4), [0xe0], [0, 0, 0, 0, 1, 0, 0, 0], frame.subarray(6));
  assert.throws(() => decode(huge), /size limit/);
});

test('compression stays within its heap limit', async () => {
  const small = await loadZstd(await readFile(wasmPath), 32 * 2 ** 20);
  assert.equal(small.compress(encode(LINE.repeat(250_000)), dictionary, 19, 1 << 20), null);
  assert.ok(small.compress(encode('x = 1\n'), dictionary, 19, 1024));
});
