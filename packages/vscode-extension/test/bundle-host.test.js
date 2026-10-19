const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path').posix;
const { createBundleHost } = require('../src/bundleHost.js');
const { enginePathOf, BUNDLE_SCHEME } = require('../src/remoteModule.js');
const api = require('../../engine/bundle.ts');
const { makeMatCtx } = require('../../engine/test/_materialise-helpers.ts');
const bytes = text => Buffer.from(text, 'utf8');

/** Preserve URI authority like workspace.fs does on SSH and WSL hosts. */
function uri(fields) {
  return { scheme: 'vscode-remote', authority: 'ssh-remote+host', path: '/', query: '', ...fields,
    with(next) { return uri({ ...this, ...next }); },
    toString() { return `${this.scheme}://${this.authority}${this.path}?${encodeURIComponent(this.query)}`; },
  };
}
function harness() {
  const files = new Map(), directories = new Set(), reads = [], writes = [], requests = [], changes = [];
  const error = code => value => Object.assign(new Error(String(value || code)), { code });
  const vscode = {
    EventEmitter: class { event() {} dispose() {} fire(events) { changes.push(...events); } },
    Uri: { from: fields => uri(fields), joinPath: (base, ...parts) => base.with({ path: path.join(base.path, ...parts) }),
      parse: value => { const parsed = new URL(value); return uri({ scheme: parsed.protocol.slice(0, -1), authority: parsed.host,
        path: parsed.pathname, query: parsed.search.slice(1) }); } },
    FileType: { File: 1, Directory: 2 }, FilePermission: { Readonly: 1 },
    FileChangeType: { Changed: 1 },
    FileSystemError: { FileNotFound: error('FileNotFound'), FileExists: error('FileExists'),
      FileNotADirectory: error('FileNotADirectory'), NoPermissions: error('NoPermissions') },
    workspace: { textDocuments: [], fs: {
      async readFile(value) { reads.push(value.toString()); const result = files.get(value.toString());
        if (!result) throw error('FileNotFound')(value); return result; },
      async stat(value) { if (!files.has(value.toString()) && !directories.has(value.toString())) throw error('FileNotFound')(value); return {}; },
      async createDirectory(value) { directories.add(value.toString()); writes.push(value.toString()); },
      async copy(source, target, options) {
        assert.equal(options.overwrite, false);
        if (files.has(target.toString())) throw error('FileExists')(target);
        const data = await host.provider.readFile(source);
        files.set(target.toString(), data); writes.push(target.toString());
      },
    } },
  };
  const cache = { isUrl: value => /^https?:/.test(value), async fetchToCache(url, options) {
    requests.push({ url, options }); return { content: files.get(url), fromCache: true };
  } };
  const host = createBundleHost(vscode, api, cache, async () => true);
  return { host, vscode, files, directories, reads, writes, requests, changes };
}

test('export snapshots dirty imported source and data before asynchronous reads', async () => {
  const h = harness(), sourceUri = uri({ path: '/models/main.flatppl' });
  let helper = 'n = 3', data = '[1, 2, 3]';
  h.vscode.workspace.textDocuments = [
    { uri: uri({ path: '/models/helper.flatppl' }), isDirty: true, getText: () => helper },
    { uri: uri({ path: '/models/data.json' }), isDirty: true, getText: () => data },
  ];
  const source = 'h = load_module("helper.flatppl")\nd = load_data("data.json", cartpow(reals, 3))\nx = sum(d) + h.n';
  const resolving = h.host.resolve(sourceUri, source, true);
  helper = 'n = 1'; data = '[99]';
  const graph = await resolving;
  assert.deepEqual(h.reads, []);
  const packed = await h.host.pack(graph);
  const loaded = await api.resolveBundleBytes('export.zip', packed, async () => null);
  const { ctx } = makeMatCtx(loaded.primarySource, { path: loaded.primaryPath, bundle: loaded, sampleCount: 1 });
  assert.deepEqual(Array.from((await ctx.getMeasure('x')).samples), [9]);
});

test('archive preview, member navigation, and explicit extraction preserve binary bytes', async () => {
  const h = harness(), archiveUri = uri({ path: '/models/demo.zip' });
  const binary = Uint8Array.of(0, 255, 17, 0);
  h.files.set(archiveUri.toString(), await api.writeArchive({
    'wrapper/main.flatppl': bytes('m = load_module("lib/helper.flatppl")\nx = m.x'),
    'wrapper/lib/helper.flatppl': bytes('x = 7'), 'wrapper/data.bin': binary,
  }));
  const graph = await h.host.resolve(archiveUri, undefined, true);
  assert.deepEqual(h.writes, []);
  const archiveId = api.parseArchiveMemberId(graph.primaryPath).archiveId;
  const root = h.host.uriFor(api.archiveMemberId(archiveId, 'wrapper'));
  assert.ok((await h.host.provider.readDirectory(root)).some(([name, type]) => name === 'lib' && type === h.vscode.FileType.Directory));
  const directory = h.host.uriFor(api.archiveMemberId(archiveId, 'wrapper/lib'));
  assert.equal((await h.host.provider.stat(directory)).type, h.vscode.FileType.Directory);
  assert.deepEqual(await h.host.provider.readDirectory(directory), [['helper.flatppl', h.vscode.FileType.File]]);
  const outsideWrapper = api.archiveMemberId(archiveId, 'another/data.bin');
  await assert.rejects(h.host.readMember(outsideWrapper));
  const member = h.host.members(graph.primaryPath).find(entry => entry.path === 'data.bin');
  assert.deepEqual(await h.host.provider.readFile(h.host.uriFor(member.id)), binary);
  const helper = graph.resolutions[graph.primaryPath].modules['lib/helper.flatppl'];
  const helperUri = h.host.uriFor(helper);
  assert.equal(helperUri.scheme, BUNDLE_SCHEME);
  assert.equal(enginePathOf(helperUri), helper);
  assert.equal((await h.host.resolve(helperUri, undefined, true)).primarySource, 'x = 7');
  assert.throws(() => h.host.provider.writeFile(helperUri, bytes('x = 8')));
  const destination = uri({ path: '/copies/demo' });
  await h.host.extract(graph.primaryPath, destination);
  assert.deepEqual(h.files.get(uri({ path: '/copies/demo/data.bin' }).toString()), binary);
  const written = h.writes.length;
  await assert.rejects(h.host.extract(graph.primaryPath, destination));
  assert.equal(h.writes.length, written);
});

test('symlink inspection works but extraction refuses before writing a working copy', async () => {
  const h = harness(), archiveUri = uri({ path: '/models/linked.zip' });
  h.files.set(archiveUri.toString(), await api.writeArchive({
    'main.flatppl': bytes('x = 1'), 'data.bin': Uint8Array.of(0, 255), 'alias.bin': bytes('data.bin'),
  }, { symlinks: ['alias.bin'] }));
  const graph = await h.host.resolve(archiveUri, undefined, true);
  const link = h.host.members(graph.primaryPath).find(entry => entry.path === 'alias.bin');
  assert.deepEqual(await h.host.readMember(link.id), Uint8Array.of(0, 255));
  await assert.rejects(h.host.extract(graph.primaryPath, uri({ path: '/copies/linked' })));
  assert.deepEqual(h.writes, []);
});

test('fresh archive opens do not reuse stale members at the same path', async () => {
  const h = harness(), archiveUri = uri({ path: '/models/demo.zip' });
  for (const value of [1, 2]) {
    h.files.set(archiveUri.toString(), await api.writeArchive({ 'main.flatppl': bytes('x = ' + value) }));
    assert.equal((await h.host.resolve(archiveUri, undefined, true)).primarySource, 'x = ' + value);
  }
  assert.equal(h.changes.length, 1);
  assert.equal(Buffer.from(await h.host.provider.readFile(h.changes[0].uri)).toString(), 'x = 2');
});

test('remote archive reads retain cache options and never fetch inner member paths', async () => {
  const h = harness(), url = 'https://models.example/demo.zip';
  h.files.set(url, await api.writeArchive({ 'main.flatppl': bytes('x = 2') }));
  const remote = uri({ scheme: 'flatppl-remote', path: '/demo.zip', query: url });
  await h.host.resolve(remote, undefined, true, { offline: true });
  assert.deepEqual(h.requests.map(request => request.url), [url]);
  assert.equal(h.requests[0].options.offline, true);
});
