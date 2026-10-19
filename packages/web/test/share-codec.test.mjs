import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { Worker } from 'node:worker_threads';
import { createHash } from 'node:crypto';
import { brotliCompressSync, brotliDecompressSync, constants } from 'node:zlib';
import { build } from 'esbuild';
import { ZipWriter, Uint8ArrayWriter, Uint8ArrayReader } from '@zip.js/zip.js';
import {
  createShareCodec, makeShareLinks, createShareQr,
  MAX_SHARE_ARCHIVE_BYTES, MAX_SHARE_URL_LENGTH,
} from '../src/share-codec.ts';

const require = createRequire(import.meta.url);
let temporary, workerFile, zip;
const originalWorker = globalThis.Worker;

before(async () => {
  temporary = await mkdtemp(join(tmpdir(), 'flatppl-share-codec-'));
  workerFile = join(temporary, 'bridge.cjs');
  await build({
    entryPoints: [new URL('../src/share-worker.ts', import.meta.url).pathname],
    bundle: true, platform: 'node', format: 'cjs', outfile: join(temporary, 'codec.cjs'),
    plugins: [{ name: 'native-wasm-entry', setup(build) {
      build.onResolve({ filter: /^brotli-wasm$/ }, () => ({ path: require.resolve('brotli-wasm'), external: true }));
    } }],
  });
  await writeFile(workerFile, `
    const { parentPort } = require('node:worker_threads');
    globalThis.self = {
      addEventListener(_, listener) { parentPort.on('message', data => listener({ data })); },
      postMessage(data, transfer) { parentPort.postMessage(data, transfer); },
    };
    require('./codec.cjs');
  `);
  // Bridge only the worker host API. Run production messages, WASM, and streams.
  globalThis.Worker = class {
    constructor() { this.worker = new Worker(workerFile); }
    addEventListener(name, listener) {
      this.worker.on(name, value => listener(name === 'message' ? { data: value } : value));
    }
    postMessage(data, transfer) { this.worker.postMessage(data, transfer); }
    terminate() { return this.worker.terminate(); }
  };
  const writer = new ZipWriter(new Uint8ArrayWriter(), { level: 0, useWebWorkers: false });
  await writer.add('main.flatppl', new Uint8ArrayReader(new TextEncoder().encode('x ~ Normal(0, 1)\n')));
  await writer.add('data.bin', new Uint8ArrayReader(Uint8Array.of(0, 255, 128, 7)));
  zip = await writer.close();
});

after(async () => {
  globalThis.Worker = originalWorker;
  await rm(temporary, { recursive: true, force: true });
});

test('shares preserve binary ZIP bytes and interoperate with native Brotli', async (t) => {
  const codec = createShareCodec('worker.js');
  t.after(() => codec.dispose());
  // Cross both 64 KiB stream boundaries without depending on compressibility.
  const writer = new ZipWriter(new Uint8ArrayWriter(), { level: 0, useWebWorkers: false });
  await writer.add('main.flatppl', new Uint8ArrayReader(new TextEncoder().encode('x = 1\n')));
  const binary = createHash('shake256', { outputLength: 96 * 1024 }).update('FlatPPL binary share fixture').digest();
  await writer.add('data.bin', new Uint8ArrayReader(binary));
  const archive = await writer.close();
  const original = archive.slice();
  const payload = await codec.encode(archive);
  assert.deepEqual(archive, original);
  assert.deepEqual(new Uint8Array(brotliDecompressSync(Buffer.from(payload.base64Token.slice(4), 'base64url'))), original);
  const nativeToken = 'br1.' + brotliCompressSync(original).toString('base64url');
  assert.deepEqual(await codec.decode(nativeToken), original);
});

test('both share links retain model and target solely in the fragment', async (t) => {
  const codec = createShareCodec('worker.js');
  t.after(() => codec.dispose());
  const payload = await codec.encode(zip);
  const links = makeShareLinks(payload, 'https://live.flatppl.org/sub/?old=query#old', {
    model: 'modules/café.flatppl', target: 'Pi_post',
  });
  for (const link of [links.url, links.qrUrl]) {
    const url = new URL(link);
    const state = new URLSearchParams(url.hash.slice(1));
    assert.equal(url.pathname, '/sub/');
    assert.equal(url.search, '');
    assert.equal(state.get('model'), 'modules/café.flatppl');
    assert.equal(state.get('target'), 'Pi_post');
    assert.deepEqual(await codec.decode(state.get('share')), zip);
  }
});

test('incomplete and trailing compressed streams never expose partial content', async (t) => {
  const codec = createShareCodec('worker.js');
  t.after(() => codec.dispose());
  const compressed = brotliCompressSync(zip);
  await assert.rejects(codec.decode('br1.' + compressed.subarray(0, -1).toString('base64url')));
  await assert.rejects(codec.decode('br1.' + Buffer.concat([compressed, Buffer.of(0)]).toString('base64url')));
  // A failed request does not poison the next preview.
  assert.deepEqual(await codec.decode('br1.' + compressed.toString('base64url')), zip);
});

test('actual Brotli expansion is bounded before returning content', async (t) => {
  const codec = createShareCodec('worker.js');
  t.after(() => codec.dispose());
  const compressed = brotliCompressSync(Buffer.alloc(MAX_SHARE_ARCHIVE_BYTES + 1), {
    params: { [constants.BROTLI_PARAM_QUALITY]: 0 },
  });
  await assert.rejects(codec.decode('br1.' + compressed.toString('base64url')), /size limit/);
});

test('noncanonical and unknown link encodings do not open a preview', async (t) => {
  const codec = createShareCodec('worker.js');
  t.after(() => codec.dispose());
  await assert.rejects(codec.decode('br1.AB')); // Nonzero unused base64 bits.
  await assert.rejects(codec.decode('BR1.AB')); // Nonzero unused base32 bits.
  await assert.rejects(codec.decode('br2.AA'));
});

test('large links and QR symbols retain the ZIP fallback', async (t) => {
  const codec = createShareCodec('worker.js');
  t.after(() => codec.dispose());
  assert.equal(await codec.encode(new Uint8Array(MAX_SHARE_ARCHIVE_BYTES + 1)), null);
  const payload = await codec.encode(zip);
  const links = makeShareLinks(payload, 'https://live.flatppl.org/', { target: 'x'.repeat(MAX_SHARE_URL_LENGTH) });
  assert.equal(links.tooLong, true);
  assert.equal(await createShareQr(links), null);
  const qrOverflow = makeShareLinks(payload, 'https://live.flatppl.org/', { target: 'x'.repeat(4000) });
  assert.equal(await createShareQr(qrOverflow), null);
});

test('closing the codec rejects pending work and prevents reuse', async () => {
  const codec = createShareCodec('worker.js');
  const pending = codec.encode(zip);
  codec.dispose();
  await assert.rejects(pending);
  await assert.rejects(codec.decode('br1.AA'));
});

test('the browser ESM build renders a QR for the selected complete share', async (t) => {
  const codec = createShareCodec('worker.js');
  t.after(() => codec.dispose());
  const links = makeShareLinks(await codec.encode(zip), 'https://live.flatppl.org/', { target: 'Pi_post' });
  await build({
    entryPoints: [new URL('../src/share-codec.ts', import.meta.url).pathname],
    bundle: true, platform: 'browser', format: 'esm', splitting: true,
    outdir: join(temporary, 'browser'), outExtension: { '.js': '.mjs' },
  });
  const browser = await import(pathToFileURL(join(temporary, 'browser/share-codec.mjs')).href);
  // Supply only Canvas's host boundary. The actual browser qrcode module,
  // including its CommonJS-to-ESM export shape, renders the symbol.
  const oldDocument = globalThis.document;
  t.after(() => { globalThis.document = oldDocument; });
  let image;
  globalThis.document = { createElement() { return {
    getContext() { return {
      createImageData(width, height) { return { data: new Uint8ClampedArray(width * height * 4) }; },
      clearRect() {}, putImageData(value) { image = value; },
    }; },
    toDataURL() { return 'data:image/png;base64,canvas-host'; },
  }; } };
  const qr = await browser.createShareQr(links);
  assert.ok(qr.svg.startsWith('<svg '));
  assert.ok(image.data.length > 0);
  assert.deepEqual(await codec.decode(new URLSearchParams(new URL(qr.url).hash.slice(1)).get('share')), zip);
});
