import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire, stripTypeScriptTypes } from 'node:module';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { copyFile, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { Worker } from 'node:worker_threads';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import {
  createShareCodec, makeShareLinks, createShareQr, MAX_SHARE_URL_LENGTH, SHARE_ARCHIVE_LIMITS,
} from '../src/share-codec.ts';
import {
  MAX_SHARE_CONTENT_BYTES, MAX_SHARE_DIGITS, ShareLimitError, decodeContainer, encodeContainer, framePayload,
  linkToken, qrToken,
} from '../src/share-format.ts';
import { loadZstd } from '../src/zstd.ts';

const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');
const encode = text => new TextEncoder().encode(text);
// The package exports no path to its binary. The build reads the same file.
const wasmPath = new URL('../../../node_modules/@bokuweb/zstd-wasm/dist/web/zstd.wasm', import.meta.url);
const dictionaryUrl = new URL('../src/share-dictionaries/v1.zdict', import.meta.url);
let temporary, bridge, zstd, dictionary;
const originalWorker = globalThis.Worker;

before(async () => {
  temporary = await mkdtemp(join(tmpdir(), 'flatppl-share-codec-'));
  await build({
    entryPoints: [new URL('../src/share-worker.ts', import.meta.url).pathname],
    bundle: true, platform: 'node', format: 'esm', outfile: join(temporary, 'share-worker.mjs'),
    target: 'es2020', loader: { '.zdict': 'binary' },  // As in build.mjs.
  });
  await copyFile(wasmPath, join(temporary, 'zstd.wasm'));
  bridge = join(temporary, 'bridge.mjs');
  // Bridge only the worker host API. Node's fetch has no file: scheme, so serve
  // the worker's sibling files the way a static host would.
  await writeFile(bridge, `
    import { parentPort } from 'node:worker_threads';
    import { readFile } from 'node:fs/promises';
    globalThis.self = {
      addEventListener(_, listener) { parentPort.on('message', data => listener({ data })); },
      postMessage(data, transfer) { parentPort.postMessage(data, transfer); },
    };
    globalThis.fetch = async url => new Response(await readFile(new URL(url)));
    await import('./share-worker.mjs');
  `);
  globalThis.Worker = class {
    constructor() { this.worker = new Worker(bridge); }
    addEventListener(name, listener) {
      this.worker.on(name, value => listener(name === 'message' ? { data: value } : value));
    }
    postMessage(data, transfer) { this.worker.postMessage(data, transfer); }
    terminate() { return this.worker.terminate(); }
  };
  zstd = await loadZstd(await readFile(wasmPath), 128 * 2 ** 20);
  dictionary = new Uint8Array(await readFile(dictionaryUrl));
});

after(async () => {
  globalThis.Worker = originalWorker;
  await rm(temporary, { recursive: true, force: true });
});

/** The page objects that open shares: the engine, and the bundle store and resolver scripts. */
function pageHost() {
  const window = { FlatPPLEngine: engine };
  for (const script of ['bundle-store', 'resolver']) {
    vm.runInNewContext(stripTypeScriptTypes(readFileSync(new URL(`../src/${script}.ts`, import.meta.url), 'utf8')),
      { window, console, document: { baseURI: 'https://gallery.invalid/' }, URL, TextEncoder, TextDecoder, Uint8Array });
  }
  return window;
}

function codecFor(t) {
  const codec = createShareCodec('worker.js', pageHost());
  t.after(() => codec.dispose());
  return codec;
}

/** Bundle with imports, binary data, an empty directory and a link. */
function fixtureBundle(root = '') {
  const files = {
    'main.flatppl': encode('prior = load_module("modules/prior.flatppl")\ny ~ Normal(prior.mu, 1.0)\n'),
    'modules/prior.flatppl': encode('mu ~ Normal(0.0, 10.0)\n'),
    'data/raw.bin': Uint8Array.of(0, 255, 0, 1, 2, 0),
    'data/empty/': new Uint8Array(),
    'alias.flatppl': encode('modules/prior.flatppl'),
  };
  const prefixed = Object.fromEntries(Object.entries(files).map(([name, bytes]) => [root + name, bytes]));
  return engine.writeArchive(prefixed, { symlinks: [root + 'alias.flatppl'] });
}

/** What the archive reader exposes, independent of ZIP layout. */
async function members(zip) {
  const archive = await engine.readArchive(zip, { id: 'compare.zip' });
  const result = {};
  for (const [name, entry] of Object.entries(archive.entries)) {
    result[name] = entry.directory ? 'directory'
      : entry.symlink ? 'link to ' + await archive.readLink(name)
      : Array.from(await archive.read(name)).join(',');
  }
  await archive.close();
  return result;
}

/** The share token of a link or QR URL, as the router reads it. */
function tokenOf(url) {
  const hash = new URL(url).hash.slice(1);
  return hash.startsWith('~') ? hash : new URLSearchParams(hash).get('share');
}
const archiveOf = async (codec, token) => (await codec.decode(token)).archive;

test('links and QR codes reproduce every member, including binary data, directories and links', async (t) => {
  const codec = codecFor(t);
  for (const root of ['', 'w'.repeat(1100) + '/']) {
    const zip = await fixtureBundle(root);
    const links = makeShareLinks(await codec.encode(zip), 'https://live.flatppl.org/');
    for (const link of [links.url, links.qrUrl]) {
      assert.deepEqual(await members(await archiveOf(codec, tokenOf(link))), await members(zip));
    }
  }
});

// Never regenerate these. Printed links of version 1 must keep opening.
const V1_DICTIONARY_SHA256 = '25808cf2f2261b2012104ef3677bc7d70b38fdee3205c1b5f7ff9a89fb8d216e';
const V1_LINK = 'ASDXnQIAFAIVAm11RyIobXUCDQEKAAAMcmF3LmJpbgYA_wABAgAAFwoS_MSoeRic8Ik7bnDQVdaHrLaZEx4l05m6W30MjtcTSowdLSer05TI0u6U2N9k8_c';
const V1_QR = '~94959339617056485940011024142294089516521529419130795466055769647436763978857062761354557981497567071434385292185372743579886705879453739448108635375853975365921389777779894721177798337826189023884055225123009527';

test('version-1 links keep opening', async (t) => {
  assert.equal(createHash('sha256').update(dictionary).digest('hex'), V1_DICTIONARY_SHA256);
  const codec = codecFor(t);
  const expected = await members(await fixtureBundle());
  for (const token of [V1_LINK, V1_QR]) {
    const shared = await codec.decode(token);
    assert.deepEqual(await members(shared.archive), expected);
    assert.equal(shared.model, 'modules/prior.flatppl');
    assert.equal(shared.target, 'mu');
  }
});

test('the token carries the selected member and target, and the fragment only the token', async (t) => {
  const codec = codecFor(t);
  const zip = await fixtureBundle();
  const links = makeShareLinks(await codec.encode(zip, { model: 'modules/Café.flatppl', target: 'Pi_post' }),
    'https://live.flatppl.org/sub/?old=query#old');
  assert.match(links.url, /^https:\/\/live\.flatppl\.org\/sub\/#share=A[A-Za-z0-9_-]+$/);
  assert.match(links.qrUrl, /^https:\/\/live\.flatppl\.org\/sub\/#~[1-9][0-9]*$/);
  for (const link of [links.url, links.qrUrl]) {
    const shared = await codec.decode(tokenOf(link));
    assert.equal(shared.model, 'modules/Café.flatppl');
    assert.equal(shared.target, 'Pi_post');
    assert.equal(shared.link, tokenOf(links.url));
  }
  const plain = await codec.decode(tokenOf(makeShareLinks(await codec.encode(zip, { model: 'main.flatppl' }),
    'https://live.flatppl.org/').url));
  assert.equal(plain.model, null);
  assert.equal(plain.target, null);
});

test('noncanonical, oversized and unknown tokens do not open a preview', async (t) => {
  const codec = codecFor(t);
  const payload = framePayload(1, zstd.compress(encodeContainer({ model: '', target: '' }, [
    { name: 'main.flatppl', kind: 'file', bytes: encode('x = 1\n') }]), dictionary, 19, 1024));
  assert.deepEqual(await members(await archiveOf(codec, linkToken(payload))), { 'main.flatppl': Array.from(encode('x = 1\n')).join(',') });
  const unknown = payload.slice();
  unknown[0] = 2;
  await assert.rejects(codec.decode(linkToken(unknown)), /newer version/);
  await assert.rejects(codec.decode('AB'));  // Nonzero unused base64 bits.
  await assert.rejects(codec.decode('~0' + qrToken(payload).slice(1)));  // Leading zero.
  await assert.rejects(codec.decode('~' + '1'.repeat(MAX_SHARE_DIGITS + 1)));
  await assert.rejects(codec.decode(qrToken(payload).slice(1)));  // Digits without the marker read as base64url.
  await assert.rejects(codec.decode('br1.' + linkToken(payload)));
  await assert.rejects(codec.decode(''));
  await assert.rejects(codec.decode('A'.repeat(MAX_SHARE_URL_LENGTH + 1)), /too long/);
  // A failed request does not poison the next preview.
  assert.ok(await codec.decode(qrToken(payload)));
});

test('containers parse completely or not at all', () => {
  const file = (name, text) => ({ name, kind: 'file', bytes: encode(text) });
  const bundle = [file('main.flatppl', 'x\0y'), file('b', '\0b\0'), { name: 'a', kind: 'symlink', bytes: encode('b') },
    { name: 'dir', kind: 'directory', bytes: new Uint8Array() }];
  const view = { model: 'dir/ä.flatppl', target: 'θ' };
  const decoded = decodeContainer(encodeContainer(view, bundle));
  assert.deepEqual(decoded.view, view);
  assert.deepEqual(decoded.members.map(m => [m.name, m.kind, Array.from(m.bytes)]),
    ['main.flatppl', 'a', 'b', 'dir'].map(name => bundle.find(m => m.name === name)).map(m => [m.name, m.kind, Array.from(m.bytes)]));
  assert.throws(() => encodeContainer(view, [...bundle, file('b', 'again')]), /Duplicate/);
  const record = (kind, name, data) => [kind, name.length, ...encode(name), data.length, ...data];
  const main = [0, 0, 1, 0x78];  // Empty view, then main.flatppl.
  const rejected = {
    'view names main.flatppl': [12, ...encode('main.flatppl'), 0, 1, 0x78],
    'view names a directory': [2, ...encode('a/'), 0, 1, 0x78],
    'view with invalid UTF-8': [0, 1, 0xff, 1, 0x78],
    'descending names': [...main, ...record(0, 'b', []), ...record(0, 'a', [])],
    'duplicate names': [...main, ...record(0, 'a', []), ...record(0, 'a', [])],
    'second main': [...main, ...record(0, 'main.flatppl', [])],
    'unknown kind': [...main, ...record(3, 'a', [])],
    'directory with data': [...main, ...record(1, 'a', [1])],
    'empty link': [...main, ...record(2, 'a', [])],
    'directory slash': [...main, ...record(1, 'a/', [])],
    'empty name': [...main, ...record(0, '', [])],
    'invalid UTF-8': [...main, 0, 1, 0xff, 0],
    'overlong length': [0, 0, 0x81, 0x00, 0x78],
    'truncated': [...main, ...record(0, 'a', [1, 2]).slice(0, -1)],
  };
  for (const [name, raw] of Object.entries(rejected)) assert.throws(() => decodeContainer(Uint8Array.from(raw)), Error, name);
  const limits = { maxEntries: 2, maxPathBytes: 8 };
  const exceeding = {
    'too many entries': [...main, ...record(0, 'a', []), ...record(0, 'b', [])],
    'long name': [...main, ...record(0, 'abcdefghi', [])],
    'long directory with its slash': [...main, ...record(1, 'abcdefgh', [])],
    'long link target': [...main, ...record(2, 'a', [...encode('abcdefghi')])],
    'long view': [9, ...encode('abcdefghi'), 0, 1, 0x78],
  };
  for (const [name, raw] of Object.entries(exceeding)) {
    assert.throws(() => decodeContainer(Uint8Array.from(raw), limits), ShareLimitError, name);
  }
  assert.equal(decodeContainer(Uint8Array.from([...main, ...record(1, 'abcdefg', [])]), limits).members.length, 2);
});

test('names keep a leading byte-order mark', () => {
  const view = { model: '\uFEFFm.flatppl', target: '' };
  const bundle = [{ name: 'main.flatppl', kind: 'file', bytes: encode('x = 1\n') },
    { name: '\uFEFFm.flatppl', kind: 'file', bytes: encode('y = 2\n') }];
  const decoded = decodeContainer(encodeContainer(view, bundle));
  assert.deepEqual(decoded.view, view);
  assert.deepEqual(decoded.members.map(m => m.name), ['main.flatppl', '\uFEFFm.flatppl']);
});

test('links exist only for bundles that open under the share limits', async (t) => {
  const codec = codecFor(t);
  const directory = n => engine.writeArchive({ 'main.flatppl': encode('x = 1\n'), ['d'.repeat(n) + '/']: new Uint8Array() });
  // A ZIP stores a directory name with its slash, which counts against the path limit.
  const fits = await directory(SHARE_ARCHIVE_LIMITS.maxPathBytes - 1);
  assert.deepEqual(await members(await archiveOf(codec, (await codec.encode(fits)).link)), await members(fits));
  assert.equal(await codec.encode(await directory(SHARE_ARCHIVE_LIMITS.maxPathBytes)), null);
  // The container stores main.flatppl only as a file.
  const linkedMain = await engine.writeArchive({ 'main.flatppl': encode('actual.flatppl'), 'actual.flatppl': encode('x = 1\n') },
    { symlinks: ['main.flatppl'] });
  assert.equal(await codec.encode(linkedMain), null);
  // An imported nested bundle shares the entry budget. A ZIP the model never opens is data.
  const many = Object.fromEntries(Array.from({ length: SHARE_ARCHIVE_LIMITS.maxEntries },
    (_, i) => [i ? `m${i}.flatppl` : 'main.flatppl', encode('x = 1\n')]));
  const nested = async (main, inner) => engine.writeArchive({ 'main.flatppl': encode(main), 'inner.zip': inner });
  const imported = 'm = load_module("inner.zip")\n';
  assert.equal(await codec.encode(await nested(imported, await engine.writeArchive(many))), null);
  assert.ok(await codec.encode(await nested('x = 1\n', await engine.writeArchive(many))));
  // Each identity under which the visitor opens a nested bundle counts.
  const inner = await engine.writeArchive(Object.fromEntries(Array.from({ length: 600 },
    (_, i) => [i ? `m${i}.flatppl` : 'main.flatppl', encode('x = 1\n')])));
  const twice = await engine.writeArchive({
    'main.flatppl': encode('a = load_module("sub/inner.zip")\nb = load_module("sub")\n'),
    'sub/main.flatppl': encode('c = load_module("inner.zip")\n'), 'sub/inner.zip': inner,
  });
  assert.equal(await codec.encode(twice), null);
});

test('content and payload limits hold for creating and opening links', async (t) => {
  const codec = codecFor(t);
  // Near-limit members fit even when the reconstructed ZIP's headers push it
  // above 8 MiB. No member exceeds the browser's 2 MiB input limit.
  const source = Array.from({ length: 4 }, (_, i) => `d${i} = load_data("data${i}.json", reals)\n`).join('');
  const files = { 'main.flatppl': encode(source), ...Object.fromEntries(
    Array.from({ length: 4 }, (_, i) => {
      const bytes = new Uint8Array(2 * 1024 ** 2 - 64).fill(32);
      bytes[bytes.length - 1] = 48; // JSON scalar zero, with leading whitespace.
      return [`data${i}.json`, bytes];
    })) };
  const nearLimit = await engine.writeArchive(files, { compression: 'store' });
  assert.ok(nearLimit.length > MAX_SHARE_CONTENT_BYTES);
  const tokens = await codec.encode(nearLimit);
  assert.ok(tokens);
  const host = pageHost();
  const shared = await codec.decode(tokens.link);
  const opened = await host.FlatPPLWebBundles.open(shared.archive, 'boundary', undefined,
    { limits: host.FlatPPLWebBundles.LINK_LIMITS });
  assert.equal(new TextDecoder().decode(await host.FlatPPLWebBundles.read(opened.path)), source);
  const graph = await host.FlatPPLWebResolver.resolveBundle(opened.path, undefined, { strict: true });
  assert.equal(Object.keys(graph.resolutions[graph.primaryPath].data).length, 4);
  await host.FlatPPLWebBundles.discard(opened.path);
  assert.equal(await codec.encode(await engine.writeArchive({
    'main.flatppl': encode('x = 1\n'), 'data.bin': new Uint8Array(2 * 1024 ** 2 + 1) })), null);
  const big = await engine.writeArchive({ 'main.flatppl': new Uint8Array(MAX_SHARE_CONTENT_BYTES) });
  assert.equal(await codec.encode(big), null);
  const random = createHash('shake256', { outputLength: 64 * 1024 }).update('incompressible').digest();
  assert.equal(await codec.encode(await engine.writeArchive({ 'main.flatppl': random })), null);
  // A short link that expands beyond the content limit fails before allocation.
  const bomb = framePayload(1, zstd.compress(new Uint8Array(MAX_SHARE_CONTENT_BYTES + 1), dictionary, 1, 1024));
  await assert.rejects(codec.decode(linkToken(bomb)));
});

test('dense QR symbols render while oversized links retain the ZIP fallback', async (t) => {
  const codec = codecFor(t);
  const data = n => createHash('shake256', { outputLength: n }).update('Dense FlatPPL share').digest();
  const dense = await engine.writeArchive({ 'main.flatppl': encode('x = 1\n'), 'data.bin': data(1600) });
  const links = makeShareLinks(await codec.encode(dense), 'https://live.flatppl.org/');
  const qr = await createShareQr(links);
  assert.ok(qr && qr.version > 25 && qr.version <= 40);
  assert.match(qr.url, /^HTTPS:\/\/LIVE\.FLATPPL\.ORG\/#~[1-9][0-9]*$/);
  assert.deepEqual(await members(await archiveOf(codec, tokenOf(qr.url))), await members(dense));
  const linkOnly = makeShareLinks(await codec.encode(await engine.writeArchive({
    'main.flatppl': encode('x = 1\n'), 'data.bin': data(2600) })), 'https://live.flatppl.org/');
  assert.equal(linkOnly.qrUrl, null);
  assert.equal(await createShareQr(linkOnly), null);
  const tooLong = makeShareLinks({ link: 'A'.repeat(MAX_SHARE_URL_LENGTH), qr: '~1' }, 'https://live.flatppl.org/');
  assert.equal(tooLong.tooLong, true);
  assert.equal(await createShareQr(tooLong), null);
});

test('closing the codec rejects pending work and prevents reuse', async () => {
  const codec = createShareCodec('worker.js', pageHost());
  const pending = codec.encode(await fixtureBundle());
  codec.dispose();
  await assert.rejects(pending);
  await assert.rejects(codec.decode('AA'));
});

test('the browser ESM build renders a QR for the selected complete share', async (t) => {
  const codec = codecFor(t);
  const zip = await fixtureBundle();
  const links = makeShareLinks(await codec.encode(zip, { target: 'Pi_post' }), 'https://live.flatppl.org/');
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
  const shared = await codec.decode(tokenOf(qr.url));
  assert.equal(shared.target, 'Pi_post');
  assert.deepEqual(await members(shared.archive), await members(zip));
});
