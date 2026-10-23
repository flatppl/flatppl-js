'use strict';

// §04 ZIP interoperability uses independent PKZIP headers and Node's Deflate
// codec, not the archive library under test. Scores use an analytic Normal law.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { crc32, deflateRawSync, inflateRawSync } = require('node:zlib');
const { readArchive, writeArchive, resolveBundleBytes, packBundle,
  archiveMemberId, parseArchiveMemberId } = require('../bundle.ts');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const encode = (text: string) => new TextEncoder().encode(text);
const decode = (bytes: Uint8Array) => new TextDecoder().decode(bytes);
type FixtureEntry = { name: string; bytes?: Uint8Array; text?: string;
  mode?: number; deflate?: boolean; declaredSize?: number; badCRC?: boolean };

/** Minimal independent STORE/Deflate ZIP writer for public format fixtures. */
function fixture(entries: FixtureEntry[]): Uint8Array {
  const local: Buffer[] = [], central: Buffer[] = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name);
    const bytes = Buffer.from(entry.bytes || encode(entry.text || ''));
    const payload = entry.deflate ? deflateRawSync(bytes) : bytes;
    const checksum = entry.badCRC ? 0 : crc32(bytes);
    const header = Buffer.alloc(30);
    header.writeUInt32LE(0x04034b50); header.writeUInt16LE(20, 4);
    header.writeUInt16LE(0x800, 6); header.writeUInt16LE(entry.deflate ? 8 : 0, 8);
    header.writeUInt32LE(checksum, 14); header.writeUInt32LE(payload.length, 18);
    header.writeUInt32LE(entry.declaredSize ?? bytes.length, 22); header.writeUInt16LE(name.length, 26);
    local.push(header, name, payload);
    const record = Buffer.alloc(46);
    record.writeUInt32LE(0x02014b50); record.writeUInt16LE(3 * 256 + 20, 4);
    header.copy(record, 6, 4, 28);
    record.writeUInt32LE(((entry.mode || 0o100644) << 16) >>> 0, 38);
    record.writeUInt32LE(offset, 42);
    central.push(record, name);
    offset += header.length + name.length + payload.length;
  }
  const directory = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...local, directory, end]);
}

/** Independent reader checks exported member bytes, sizes and CRCs. */
function independentRead(bytes: Uint8Array): Record<string, Uint8Array> {
  const zip = Buffer.from(bytes);
  const end = zip.length - 22;
  assert.equal(zip.readUInt32LE(end), 0x06054b50);
  let cursor = zip.readUInt32LE(end + 16);
  const result: Record<string, Uint8Array> = {};
  for (let i = 0; i < zip.readUInt16LE(end + 10); i++) {
    assert.equal(zip.readUInt32LE(cursor), 0x02014b50);
    const method = zip.readUInt16LE(cursor + 10), compressed = zip.readUInt32LE(cursor + 20);
    const nameLength = zip.readUInt16LE(cursor + 28), extraLength = zip.readUInt16LE(cursor + 30);
    const name = zip.subarray(cursor + 46, cursor + 46 + nameLength).toString('utf8');
    const local = zip.readUInt32LE(cursor + 42);
    const start = local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28);
    const payload = zip.subarray(start, start + compressed);
    const raw = method === 8 ? inflateRawSync(payload) : payload;
    assert.ok(method === 0 || method === 8);
    assert.equal(raw.length, zip.readUInt32LE(cursor + 24));
    assert.equal(crc32(raw), zip.readUInt32LE(cursor + 16));
    result[name] = raw;
    cursor += 46 + nameLength + extraLength + zip.readUInt16LE(cursor + 32);
  }
  return result;
}

const model = `helper_path = "helpers/prior.flatppl"
helpers = load_module(helper_path)
observations = load_data("data/observations.json", cartpow(reals, 2))
theta ~ helpers.prior
y ~ iid(Normal(theta, 1), 2)
prior = lawof(record(theta = theta))
K = kernelof(y, theta = theta)
L = likelihoodof(K, observations)
posterior = bayesupdate(L, prior)
score = logdensityof(posterior, record(theta = 0))
`;
const modelEntries: FixtureEntry[] = [
  { name: 'main.flatppl', text: model, deflate: true },
  { name: 'helpers/prior.flatppl', text: 'prior = Normal(0, 1)\n' },
  { name: 'data/observations.json', text: '[1,2]' },
  { name: 'opaque.bin', bytes: new Uint8Array([0, 255, 128, 12]) },
];

test('independent ZIPs retain wrapped source and binary members, and exports decode independently', async () => {
  const bytes = fixture(modelEntries.map(entry => ({ ...entry, name: `example/${entry.name}` })));
  const archive = await readArchive(bytes, { id: 'example.flatppl.zip' });
  assert.equal(archive.root, 'example');
  assert.equal(decode(await archive.read('main.flatppl')), model);
  assert.deepEqual(Array.from(await archive.read('opaque.bin')), [0, 255, 128, 12]);
  const files: Record<string, Uint8Array> = {};
  for (const [path, meta] of Object.entries(archive.entries) as Array<[string, any]>) {
    if (!meta.directory) files[path] = await archive.read(path);
  }
  for (const compression of ['store', 'deflate']) {
    const emitted = independentRead(await writeArchive(files, { compression }));
    assert.equal(decode(emitted['main.flatppl']), model);
    assert.deepEqual(Array.from(emitted['opaque.bin']), [0, 255, 128, 12]);
  }
  await archive.close();
});

test('a ZIP model scores its imported prior and JSON observations against the analytic oracle', async () => {
  const graph = await resolveBundleBytes('model.flatppl.zip', fixture(modelEntries), async () => null);
  const { ctx, built } = makeMatCtx(graph.primarySource, { path: graph.primaryPath, bundle: graph, sampleCount: 1 });
  assert.deepEqual(built.diagnostics, []);
  const score = (await ctx.getMeasure('score')).samples[0];
  const expected = -1.5 * Math.log(2 * Math.PI) - 2.5;
  assert.ok(Math.abs(score - expected) < 1e-12, `${score} != ${expected}`);
  assert.equal(expected, -5.2568155996140185);
});

test('nested ZIPs with matching member names retain distinct module identities', async () => {
  const a = fixture([{ name: 'main.flatppl', text: 'x = 1\n' }]);
  const b = fixture([{ name: 'main.flatppl', text: 'x = 2\n' }]);
  const bytes = fixture([
    { name: 'main.flatppl', text: 'a=load_module("a.zip")\nb=load_module("b.zip")\nscore=a.x+b.x\n' },
    { name: 'a.zip', bytes: a }, { name: 'b.zip', bytes: b },
  ]);
  const graph = await resolveBundleBytes('outer.zip', bytes, async () => null);
  const { ctx } = makeMatCtx(graph.primarySource, { path: graph.primaryPath, bundle: graph, sampleCount: 1 });
  assert.equal((await ctx.getMeasure('score')).samples[0], 3);
  assert.equal(new Set(Object.values(graph.resolutions[graph.primaryPath].modules)).size, 2);
  await assert.rejects(resolveBundleBytes('outer.zip', bytes, async () => null, { limits: { maxDepth: 1 } }));
});

test('directory imports narrow containment while ordinary parent references remain inside the archive', async () => {
  const good = fixture([
    { name: 'main.flatppl', text: 'm=load_module("sub")\nscore=m.x\n' },
    { name: 'sub/main.flatppl', text: 'm=load_module("deep/value.flatppl")\nx=m.x\n' },
    { name: 'sub/deep/value.flatppl', text: 'm=load_module("../value.flatppl")\nx=m.x\n' },
    { name: 'sub/value.flatppl', text: 'x=7\n' },
  ]);
  const graph = await resolveBundleBytes('outer.zip', good, async () => null);
  const { ctx } = makeMatCtx(graph.primarySource, { path: graph.primaryPath, bundle: graph, sampleCount: 1 });
  assert.equal((await ctx.getMeasure('score')).samples[0], 7);
  const escape = fixture([
    { name: 'main.flatppl', text: 'm=load_module("sub")\n' },
    { name: 'sub/main.flatppl', text: 'm=load_module("../other.flatppl")\n' },
    { name: 'other.flatppl', text: 'x=1\n' },
  ]);
  await assert.rejects(resolveBundleBytes('outer.zip', escape, async () => null));
});

test('contained symbolic links resolve canonically and survive a saved archive', async () => {
  const bytes = fixture([
    { name: 'main.flatppl', text: 'm=load_module("alias.flatppl")\nscore=m.x\n' },
    { name: 'alias.flatppl', text: 'sub/source.flatppl', mode: 0o120777 },
    { name: 'sub/source.flatppl', text: 'm=load_module("value.flatppl")\nx=m.x\n' },
    { name: 'sub/value.flatppl', text: 'x=9\n' },
    { name: 'aliasdir', text: 'sub/deep', mode: 0o120777 },
    { name: 'sub/deep/', mode: 0o040755 },
  ]);
  const archive = await readArchive(bytes, { id: 'links.zip' });
  assert.equal(decode(await archive.read('aliasdir/../value.flatppl')), 'x=9\n');
  const files: Record<string, Uint8Array> = {};
  for (const [path, meta] of Object.entries(archive.entries) as Array<[string, any]>) {
    if (!meta.directory) files[path] = meta.symlink ? encode(await archive.readLink(path)) : await archive.read(path);
  }
  files['sub/deep/'] = new Uint8Array();
  const saved = await writeArchive(files, { symlinks: ['alias.flatppl', 'aliasdir'] });
  const graph = await resolveBundleBytes('saved.zip', saved, async () => null);
  const { ctx } = makeMatCtx(graph.primarySource, { path: graph.primaryPath, bundle: graph, sampleCount: 1 });
  assert.equal((await ctx.getMeasure('score')).samples[0], 9);
  await archive.close();
});

test('ZIP path ambiguity, collisions and escaping or cyclic links never expose a bundle', async () => {
  const main = { name: 'main.flatppl', text: 'x=1\n' };
  const cases: FixtureEntry[][] = [
    [{ name: '../outside', text: '' }], [{ name: '/outside', text: '' }],
    [{ name: 'a\\b', text: '' }], [main],
    [{ name: 'a', text: '' }, { name: 'a/b', text: '' }],
    [{ name: 'link', text: '../outside', mode: 0o120777 }],
    [{ name: 'a', text: 'b', mode: 0o120777 }, { name: 'b', text: 'a', mode: 0o120777 }],
  ];
  for (const entries of cases) await assert.rejects(readArchive(fixture([main, ...entries]), { id: 'bad.zip' }));
});

test('CRC corruption and understated streamed output fail before member bytes reach callers', async () => {
  const corrupt = await readArchive(fixture([{ name: 'main.flatppl', text: 'x=1\n', badCRC: true }]), { id: 'crc.zip' });
  await assert.rejects(corrupt.read('main.flatppl'));
  await corrupt.close();
  const small = await readArchive(fixture([{ name: 'main.flatppl', text: 'x'.repeat(100), deflate: true, declaredSize: 4 }]),
    { id: 'bomb.zip', limits: { maxMemberBytes: 16, maxTotalBytes: 32 } });
  await assert.rejects(small.read('main.flatppl'));
  await small.close();
  await assert.rejects(readArchive(fixture(modelEntries), { id: 'count.zip', limits: { maxEntries: 1 } }));
  await assert.rejects(resolveBundleBytes('main.flatppl', encode('d=load_data("data.bin",anything)\n'),
    async () => new Uint8Array(128), { limits: { maxMemberBytes: 64 } }));
});

test('long paths and links stay linear in their components without materializing implicit prefixes', async () => {
  const deep = 'a/'.repeat(32_000) + 'value.flatppl';
  const unicode = 'é'.repeat(20_000) + '.flatppl';
  const archive = await readArchive(fixture([
    { name: 'main.flatppl', text: 'x=1\n' },
    { name: deep, text: 'x=2\n' },
    { name: unicode, text: 'x=3\n' },
    { name: 'link', text: './'.repeat(32_000) + 'main.flatppl', mode: 0o120777, deflate: true },
  ]), { id: 'long-paths.zip' });
  assert.equal(Object.keys(archive.entries).length, 4);
  assert.equal(await archive.resolve(deep), deep);
  assert.equal((await archive.stat('a/a'))?.directory, true);
  assert.equal(decode(await archive.read('link')), 'x=1\n');
  assert.equal(decode(await archive.read(unicode)), 'x=3\n');
  await archive.close();
  await assert.rejects(readArchive(fixture([
    { name: 'main.flatppl', text: 'x=1\n' },
    { name: 'link', text: './'.repeat(32_768) + 'main.flatppl', mode: 0o120777, deflate: true },
  ]), { id: 'oversized-link.zip' }), /path byte limit/);
});

test('direct member opens capture dirty dependencies while binary data stays opaque', async () => {
  const zip = fixture(modelEntries);
  const primary = archiveMemberId('source.zip', 'main.flatppl');
  const data = archiveMemberId('source.zip', 'data/observations.json');
  const reads: string[] = [];
  const graph = await resolveBundleBytes(primary, encode(model), async (path: string) => {
    reads.push(path);
    return path === 'source.zip' ? zip : path === data ? encode('[2,3]') : null;
  });
  assert.ok(reads.includes(data));
  assert.equal(decode(graph.files[data]), '[2,3]');
  assert.equal(graph.sources[data], undefined);
  assert.deepEqual(parseArchiveMemberId(primary), { archiveId: 'source.zip', path: 'main.flatppl' });
  const { ctx } = makeMatCtx(graph.primarySource, { path: graph.primaryPath, bundle: graph, sampleCount: 1 });
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0] - (-1.5 * Math.log(2 * Math.PI) - 6.5)) < 1e-12);
});

test('portable export rewrites only dependency arguments and preserves executable results', async () => {
  const source = `% Keep load_module("unchanged comment") and the alias declaration.
path = "https://example.org/helper.flatppl"
m = load_module(path)
data_path = "../data/bytes.bin"
raw = load_data(data_path, anything)
score = m.x
`;
  const inputs: Record<string, Uint8Array> = {
    'https://example.org/helper.flatppl': encode('x=17\n'),
    '/data/bytes.bin': new Uint8Array([255, 0, 128]),
  };
  const graph = await resolveBundleBytes('/models/model.flatppl', encode(source), async (path: string) => inputs[path]);
  const zip = await packBundle(graph);
  assert.equal(graph.primarySource, source);
  const output = independentRead(zip);
  assert.ok(decode(output['main.flatppl']).startsWith('% Keep load_module("unchanged comment") and the alias declaration.\npath = "https://example.org/helper.flatppl"'));
  assert.deepEqual(Array.from(output[Object.keys(output).find(name => name.endsWith('.bin'))!]), [255, 0, 128]);
  const unpacked = await resolveBundleBytes('portable.zip', zip, async () => null);
  const { ctx } = makeMatCtx(unpacked.primarySource, { path: unpacked.primaryPath, bundle: unpacked, sampleCount: 1 });
  assert.equal((await ctx.getMeasure('score')).samples[0], 17);
});

test('module source substitutions survive resolution, portable path rewriting, and evaluation', async () => {
  const source = 'm = load_module("leaf.flatppl", source = "input.json")\n'
    + 'score = ifelse(m.source == "input.json", 17, -1)\n';
  const reads: string[] = [];
  const graph = await resolveBundleBytes('/models/main.flatppl', encode(source), async (path: string) => {
    reads.push(path);
    return path === '/models/leaf.flatppl' ? encode('source = external(strings)\n') : null;
  });
  assert.deepEqual(reads, ['/models/leaf.flatppl']);
  const zip = await packBundle(graph);
  assert.match(decode(independentRead(zip)['main.flatppl']),
    /load_module\("modules\/leaf.flatppl", source = "input.json"\)/);
  const packed = await resolveBundleBytes('portable.zip', zip, async () => null);
  for (const bundle of [graph, packed]) {
    const { ctx, built } = makeMatCtx(bundle.primarySource, { path: bundle.primaryPath, bundle, sampleCount: 1 });
    assert.deepEqual(built.diagnostics, []);
    assert.equal((await ctx.getMeasure('score')).samples[0], 17);
  }
});

test('strict export refuses missing dependencies while tolerant loading preserves editor diagnostics', async () => {
  const source = 'm=load_module("missing.flatppl")\n';
  await assert.rejects(resolveBundleBytes('main.flatppl', encode(source), async () => null));
  const graph = await resolveBundleBytes('main.flatppl', encode(source), async () => null, { strict: false });
  assert.ok(processSource(source, { path: graph.primaryPath, bundle: graph }).diagnostics.some((d: any) => d.severity === 'error'));
  await assert.rejects(packBundle(graph));
  await assert.rejects(resolveBundleBytes('main.flatppl', encode('p=external(strings)\nm=load_module(p)\n'), async () => null));
});
