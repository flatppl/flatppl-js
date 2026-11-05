// Share links in the extension host. src/shareLinks.js and lib/zstd.wasm come from build:vendor.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createShareService, shareQrHtml, LINK_LIMITS, NOT_A_SHARE } = require('../src/shareLinks.js');
const { createBundleHost } = require('../src/bundleHost.js');
const api = require('../../engine/bundle.ts');

const wasm = async () => fs.readFileSync(path.join(__dirname, '..', 'lib', 'zstd.wasm'));
const encode = text => new TextEncoder().encode(text);
const BASE = 'https://live.flatppl.org/';
const golden = import('../../share/test/golden.mjs');

/** Archive members by name, comparable with deepEqual. */
async function listing(zip) {
  const archive = await api.readArchive(zip, { id: 'listing.zip' });
  try {
    const out = {};
    for (const [name, entry] of Object.entries(archive.entries)) {
      out[name] = entry.directory ? 'directory' : entry.symlink ? 'link to ' + await archive.readLink(name)
        : Buffer.from(await archive.read(name)).toString('hex');
    }
    return out;
  } finally { await archive.close(); }
}
const membersListing = async (_shares, members) => listing(await api.writeArchive(Object.fromEntries(members.map(m =>
  [m.kind === 'directory' ? m.name + '/' : m.name, m.bytes])), { symlinks: members.filter(m => m.kind === 'symlink').map(m => m.name) }));

/** Fail the test on any network request. */
function noNetwork(t) {
  const original = globalThis.fetch;
  globalThis.fetch = async url => { throw new Error('unexpected network request: ' + url); };
  t.after(() => { globalThis.fetch = original; });
}

test('golden tokens open in the extension and the extension makes them', async () => {
  const shares = createShareService(api, wasm);
  for (const g of (await golden).GOLDEN) {
    for (const token of [g.link, g.qr, BASE + '#share=' + g.link, BASE + '#' + g.qr]) {
      const opened = await shares.open(token);
      assert.deepEqual(await listing(opened.archive), await membersListing(shares, g.members), g.name);
      assert.deepEqual([opened.model, opened.target, opened.link], [g.view.model || null, g.view.target || null, g.link]);
    }
    const links = await shares.create(g.members, g.view, BASE);
    assert.deepEqual(links, { url: BASE + '#share=' + g.link, qrUrl: BASE + '#' + g.qr, tooLong: false });
  }
});

test('a playground route keeps its member and binding', async () => {
  const shares = createShareService(api, wasm);
  const { link } = (await golden).GOLDEN[2];
  const route = await shares.open(`${BASE}#/files/new/shared-model/main.flatppl?share=${link}`);
  assert.deepEqual([route.model, route.target], ['main.flatppl', null]);
  const target = await shares.open(`${BASE}#share=${link}&target=m`);
  assert.deepEqual([target.model, target.target], ['modules/prior.flatppl', 'm']);
});

test('links that would not open get no link, and the message names the reason', async () => {
  const shares = createShareService(api, wasm);
  const view = { model: '', target: '' };
  const main = text => ({ name: 'main.flatppl', kind: 'file', bytes: encode(text) });
  await assert.rejects(shares.create([main('x = 1\n'), { name: 'data.bin', kind: 'file', bytes: new Uint8Array(2 * 2 ** 20 + 1) }],
    view, BASE), /size or entry count limit.*2 MiB per member, 8 MiB in all, 256 entries and 4 nesting levels/);
  await assert.rejects(shares.create([main('x = 1\n')], view, 'file:///tmp/'), /HTTP or HTTPS/);
  await assert.rejects(shares.create([main('m = load_module("missing.flatppl")\n')], view, BASE), /missing\.flatppl/);
  await assert.rejects(shares.create([main('x'.repeat(8 * 2 ** 20 + 1))], view, BASE), /8 MiB of content/);
  assert.equal(LINK_LIMITS.maxEntries, 256);
});

test('example and gallery links are refused offline', async (t) => {
  noNetwork(t);
  const shares = createShareService(api, wasm);
  for (const input of [BASE, `${BASE}#/files/examples/eight-schools.flatppl?target=theta`,
    `${BASE}#model=examples/eight-schools.flatppl`, `${BASE}#archive=https%3A%2F%2Fexample.org%2Fm.zip`,
    'https://example.org/models/eight-schools.flatppl', 'examples/eight-schools.flatppl']) {
    await assert.rejects(shares.open(input), error => error.message === NOT_A_SHARE, input);
  }
  await assert.rejects(shares.open(BASE + '#share=AB'), /Invalid share link/);
});

/** A minimal VS Code surface for the bundle host, as in bundle-host.test.js. */
function host() {
  const calls = { approve: [], cache: [], reads: [] };
  const uri = fields => ({ scheme: 'file', authority: '', path: '/', query: '', ...fields,
    with(next) { return uri({ ...this, ...next }); }, toString() { return `${this.scheme}://${this.path}?${this.query}`; } });
  const vscode = {
    EventEmitter: class { event() {} dispose() {} fire() {} },
    Uri: { from: uri, parse: value => uri({ path: value }) },
    FileType: { File: 1, Directory: 2 }, FilePermission: { Readonly: 1 }, FileChangeType: { Changed: 1 },
    FileSystemError: { FileNotFound: value => new Error('FileNotFound ' + value) },
    workspace: { textDocuments: [], fs: { async readFile(value) { calls.reads.push(String(value)); throw new Error('no file'); } } },
  };
  const cache = { isUrl: value => /^https?:/.test(value),
    async fetchToCache(url, options) { calls.cache.push({ url, offline: !!options.offline }); throw new Error('not cached'); } };
  const bundles = createBundleHost(vscode, api, cache, async url => { calls.approve.push(url); return false; });
  return { bundles, calls, uri };
}

/** The shared codec without the pre-check, to make a token no host would hand out. */
async function rawCore() {
  const outfile = path.join(fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'flatppl-share-core-')), 'core.cjs');
  await require('esbuild').build({ entryPoints: [path.join(__dirname, '../../share/src/core.ts')], bundle: true,
    platform: 'node', format: 'cjs', outfile, loader: { '.zdict': 'binary' } });
  return require(outfile).createShareCore(wasm);
}

test('opening a share with a remote load_module fetches and approves nothing', async (t) => {
  noNetwork(t);
  const shares = createShareService(api, wasm);
  const main = { name: 'main.flatppl', kind: 'file', bytes: encode('r = load_module("https://example.org/r.flatppl")\nx = r.y\n') };
  // Bundle export vendors remote modules, so only a hand-made token holds one, and no host makes a link for it.
  await assert.rejects(shares.create([main], { model: '', target: '' }, BASE), /Bundle paths must be relative/);
  const { link } = await (await rawCore()).encode({ model: '', target: '' }, [main]);
  const opened = await shares.open(BASE + '#share=' + link);
  const h = host();
  const id = '/flatppl-share/remote.flatppl.zip';
  const graph = await h.bundles.mount(id, opened.archive, h.uri({ scheme: 'flatppl-share', path: '/remote' }), LINK_LIMITS);
  // Visualize resolves the member as the panel does. The engine keeps bundle paths relative (spec section 04).
  const member = h.bundles.uriFor(graph.primaryPath);
  await assert.rejects(h.bundles.resolve(member, undefined, true), /Bundle paths must be relative/);
  await h.bundles.resolve(member, undefined, false);
  assert.deepEqual(h.calls, { approve: [], cache: [], reads: [] });
});

test('the QR webview shows the symbol and the escaped link', async () => {
  const shares = createShareService(api, wasm);
  const g = (await golden).GOLDEN[1];
  const links = await shares.create(g.members, g.view, BASE);
  const qr = await shares.qr(links);
  assert.ok(qr.svg.startsWith('<svg') && qr.png.startsWith('data:image/png;base64,'));
  assert.equal(qr.url, 'HTTPS://LIVE.FLATPPL.ORG/#' + g.qr);
  const html = shareQrHtml(links.url + '"<x>', qr, 'n0nce');
  assert.ok(html.includes(qr.svg));
  assert.ok(html.includes('&#34;&#60;x&#62;') && !html.includes('"<x>'));
  assert.match(html, /script-src 'nonce-n0nce'/);
  assert.equal(await shares.qr({ ...links, qrUrl: null }), null);
});

/** The share commands over a stub VS Code with an in-memory workspace. */
function commandHarness(t, { files = {}, target = null, baseUrl } = {}) {
  noNetwork(t);
  const h = host();
  const { createShareCommands } = require('../src/shareCommands.js');
  const shown = [], clipboard = [], saved = [], panels = [], opened = [];
  const fsFiles = new Map(Object.entries(files).map(([name, text]) => [`file:///models/${name}`, encode(text)]));
  const uri = (fields) => h.uri(fields);
  const vscode = {
    Uri: { from: uri, joinPath: (base, ...parts) => base.with({ path: path.posix.join(base.path, ...parts) }) },
    ProgressLocation: { Notification: 15 }, ViewColumn: { Beside: -2 }, TextEditorRevealType: { InCenter: 2 },
    Position: class { constructor(line, character) { Object.assign(this, { line, character }); } },
    Selection: class { constructor(anchor, active) { Object.assign(this, { anchor, active }); } },
    Range: class { constructor(start, end) { Object.assign(this, { start, end }); } },
    env: { clipboard: { async writeText(text) { clipboard.push(text); } } },
    workspace: {
      workspaceFolders: [{ uri: uri({ path: '/models' }) }],
      getConfiguration: () => ({ get: key => key === 'shareBaseUrl' ? baseUrl : undefined }),
      fs: { async writeFile(target, bytes) { saved.push({ path: target.path, bytes }); } },
    },
    window: {
      activeTextEditor: undefined, input: undefined,
      async withProgress(_, task) { return task(); },
      async showInputBox() { return vscode.window.input; },
      showInformationMessage() {}, showErrorMessage(message) { throw new Error(message); },
      async showWarningMessage() { return 'Copy Link'; },
      async showSaveDialog({ defaultUri }) { return defaultUri; },
      async showTextDocument(value) {
        const bytes = await h.bundles.provider.readFile(value);
        const editor = { document: { uri: value, languageId: 'flatppl', getText: () => Buffer.from(bytes).toString('utf8') },
          revealRange(range) { editor.revealed = range.start.line; } };
        shown.push(editor);
        vscode.window.activeTextEditor = editor;
        return editor;
      },
      createWebviewPanel() {
        const panel = { webview: { html: '', onDidReceiveMessage(listener) { panel.post = listener; } } };
        panels.push(panel);
        return panel;
      },
    },
  };
  // The bundle host reads the workspace through the same stub.
  h.bundles = createBundleHost({ ...vscode, EventEmitter: class { event() {} dispose() {} fire() {} },
    FileType: { File: 1, Directory: 2 }, FilePermission: { Readonly: 1 }, FileChangeType: { Changed: 1 },
    FileSystemError: { FileNotFound: value => new Error('FileNotFound ' + value) },
    workspace: { ...vscode.workspace, textDocuments: [], fs: { async readFile(value) {
      const bytes = fsFiles.get(`${value.scheme}://${value.path}`);
      if (!bytes) throw new Error('FileNotFound ' + value.path);
      return bytes;
    } } } }, api, { isUrl: value => /^https?:/.test(value), async fetchToCache() { throw new Error('no cache'); } },
  async url => { h.calls.approve.push(url); return false; });
  const lines = (document, name) => Buffer.from(document.getText()).toString().split('\n')
    .findIndex(line => line.startsWith(name + ' '));
  const commands = createShareCommands(vscode, {
    bundleHost: h.bundles, bundleApi: api, shares: () => createShareService(api, wasm),
    shownTarget: () => target, bindingLine: (document, name) => { const i = lines(document, name); return i < 0 ? undefined : i; },
    opened: primaryPath => opened.push(primaryPath),
  });
  const edit = name => {
    const document = { uri: uri({ path: '/models/' + name }), languageId: 'flatppl',
      getText: () => Buffer.from(fsFiles.get('file:///models/' + name)).toString('utf8') };
    vscode.window.activeTextEditor = { document };
  };
  return { vscode, commands, shown, clipboard, saved, panels, opened, edit, calls: h.calls };
}

test('Open Share Link mounts the bundle read-only and shows the route member at its binding', async (t) => {
  const { GOLDEN } = await golden;
  const c = commandHarness(t, { target: 'mu' });
  c.vscode.window.input = `${BASE}#/files/new/shared-model/modules/prior.flatppl?target=mu&share=${GOLDEN[2].link}`;
  await c.commands.openShareLink();
  const [editor] = c.shown;
  assert.equal(editor.document.uri.path, '/modules/prior.flatppl');
  assert.equal(editor.document.getText(), 'mu ~ Normal(0.0, 10.0)\n');
  assert.deepEqual([editor.selection.active.line, editor.revealed, c.opened.length], [0, 0, 1]);
  assert.deepEqual(c.calls, { approve: [], cache: [], reads: [] });
  // Copying from the opened member, with the visualizer on the same binding, gives the same link back.
  await c.commands.copyShareLink();
  assert.equal(c.clipboard[0], `${BASE}#share=${GOLDEN[2].link}`);
});

test('Open Share Link refuses gallery links before any network access', async (t) => {
  const c = commandHarness(t);
  c.vscode.window.input = `${BASE}#/files/examples/eight-schools.flatppl?target=theta`;
  await assert.rejects(c.commands.openShareLink(), error => error.message === NOT_A_SHARE);
  assert.deepEqual([c.shown.length, c.opened.length], [0, 0]);
});

test('Copy Share Link packs a workspace model with the shown binding and the configured base', async (t) => {
  const c = commandHarness(t, { target: 'y', baseUrl: 'https://playground.example/sub/',
    files: { 'main.flatppl': 'h = load_module("helper.flatppl")\ny = h.n + 1\n', 'helper.flatppl': 'n = 3\n' } });
  c.edit('main.flatppl');
  await c.commands.copyShareLink();
  assert.match(c.clipboard[0], /^https:\/\/playground\.example\/sub\/#share=A[A-Za-z0-9_-]+$/);
  const opened = await createShareService(api, wasm).open(c.clipboard[0]);
  assert.deepEqual([opened.model, opened.target], [null, 'y']);
  const listed = await listing(opened.archive);
  assert.deepEqual(Object.keys(listed).sort(), ['main.flatppl', 'modules/helper.flatppl']);
});

test('Copy Share Link gives no link for a model over the limits', async (t) => {
  const c = commandHarness(t, { files: { 'main.flatppl': 'd = load_data("big.json", reals)\n',
    'big.json': ' '.repeat(2 * 2 ** 20) + '0' } });
  c.edit('main.flatppl');
  await assert.rejects(c.commands.copyShareLink(), /^Error: No share link for this model\. .*2 MiB per member/);
  assert.deepEqual(c.clipboard, []);
});

test('Show Share QR Code renders the symbol and saves SVG and PNG', async (t) => {
  const c = commandHarness(t, { files: { 'main.flatppl': 'mu ~ Normal(0.0, 1.0)\n' } });
  c.edit('main.flatppl');
  await c.commands.showShareQr();
  const [panel] = c.panels;
  assert.match(panel.webview.html, /<svg[^>]*viewBox/);
  for (const action of ['svg', 'png', 'copy']) panel.post({ action });
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(c.saved.map(s => s.path), ['/models/flatppl-model-qr.svg', '/models/flatppl-model-qr.png']);
  assert.ok(Buffer.from(c.saved[0].bytes).toString().startsWith('<svg'));
  assert.deepEqual(Array.from(c.saved[1].bytes.subarray(0, 4)), [0x89, 0x50, 0x4e, 0x47]);
  assert.equal(c.clipboard[0], `${BASE}#share=${(await golden).GOLDEN[0].link}`);
});
