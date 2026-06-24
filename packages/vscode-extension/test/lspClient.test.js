// Unit tests for the pure LSP helpers. These live in src/lspHelpers.ts, which
// imports only Node built-ins (no vscode, no vscode-languageclient), so they
// can be required directly here with no host stubbing. The vscode-coupled
// lifecycle (createClient/LspClientManager in lspClient.ts) needs the VS Code
// host and is exercised by manual/CI smoke, not here.
const test = require('node:test');
const assert = require('node:assert/strict');
const {
  hostTriple,
  bundledBinaryName,
  resolveServerBinary,
  readCatalogues,
  makeSerialQueue,
  makeDebounced,
  urlSourcesFeedSig,
} = require('../src/lspHelpers.js');

test('hostTriple maps known hosts', () => {
  assert.deepEqual(hostTriple('darwin', 'arm64'), {
    triple: 'aarch64-apple-darwin',
    exe: '',
  });
  assert.deepEqual(hostTriple('win32', 'x64'), {
    triple: 'x86_64-pc-windows-msvc',
    exe: '.exe',
  });
  assert.equal(hostTriple('sunos', 'sparc'), null);
});

test('bundledBinaryName suffixes the triple (and .exe on windows)', () => {
  assert.equal(
    bundledBinaryName('linux', 'x64'),
    'flatppl-lsp-x86_64-unknown-linux-musl',
  );
  assert.equal(
    bundledBinaryName('win32', 'x64'),
    'flatppl-lsp-x86_64-pc-windows-msvc.exe',
  );
  assert.equal(bundledBinaryName('aix', 'ppc'), null);
});

test('resolveServerBinary prefers an existing override', () => {
  const r = resolveServerBinary('/ext', '/my/flatppl-lsp', 'linux', 'x64', (p) => p === '/my/flatppl-lsp');
  assert.deepEqual(r, { path: '/my/flatppl-lsp', source: 'override' });
});

test('resolveServerBinary throws when the override path is missing', () => {
  assert.throws(
    () => resolveServerBinary('/ext', '/nope', 'linux', 'x64', () => false),
    /no file exists/,
  );
});

test('resolveServerBinary falls back to the bundled binary', () => {
  const bundled = '/ext/bin/flatppl-lsp-x86_64-unknown-linux-musl';
  const r = resolveServerBinary('/ext', '', 'linux', 'x64', (p) => p === bundled);
  assert.deepEqual(r, { path: bundled, source: 'bundled' });
});

test('resolveServerBinary throws a helpful error when bundled binary absent', () => {
  assert.throws(
    () => resolveServerBinary('/ext', '', 'linux', 'x64', () => false),
    /not found/,
  );
});

test('readCatalogues reads every *.ron, sorted, skipping non-ron', () => {
  const files = { 'b.ron': 'B', 'a.ron': 'A', 'note.txt': 'X' };
  const out = readCatalogues(
    '/cats',
    () => {},
    {
      existsSync: () => true,
      readdirSync: () => Object.keys(files),
      readFileSync: (p) => files[p.split('/').pop()],
    },
  );
  assert.deepEqual(out, ['A', 'B']);
});

test('readCatalogues returns [] for empty/missing dir', () => {
  assert.deepEqual(readCatalogues('', () => {}), []);
  assert.deepEqual(
    readCatalogues('/nope', () => {}, { existsSync: () => false }),
    [],
  );
});

test('makeSerialQueue runs overlapping ops strictly sequentially (no interleave)', async () => {
  const enqueue = makeSerialQueue();
  let active = 0;
  let maxActive = 0;
  const order = [];
  const mkOp = (id) => async () => {
    active++; maxActive = Math.max(maxActive, active);
    order.push(`start${id}`);
    await new Promise((r) => setTimeout(r, 5)); // force a tick where interleave could happen
    order.push(`end${id}`);
    active--;
  };
  // Fire 4 overlapping (do not await between enqueues).
  const ps = [enqueue(mkOp(1)), enqueue(mkOp(2)), enqueue(mkOp(3)), enqueue(mkOp(4))];
  await Promise.all(ps);
  assert.equal(maxActive, 1, 'at most one op active at a time (no overlap)');
  assert.deepEqual(order, ['start1','end1','start2','end2','start3','end3','start4','end4']);
});

test('makeSerialQueue keeps the chain alive after a rejecting op', async () => {
  const enqueue = makeSerialQueue();
  const seen = [];
  await enqueue(async () => { throw new Error('boom'); }).catch(() => seen.push('caught'));
  await enqueue(async () => { seen.push('next-ran'); });
  assert.deepEqual(seen, ['caught', 'next-ran']);
});

test('readCatalogues warns and skips an unreadable file', () => {
  const warnings = [];
  const out = readCatalogues(
    '/cats',
    (m) => warnings.push(m),
    {
      existsSync: () => true,
      readdirSync: () => ['ok.ron', 'bad.ron'],
      readFileSync: (p) => {
        if (p.endsWith('bad.ron')) throw new Error('EACCES');
        return 'OK';
      },
    },
  );
  assert.deepEqual(out, ['OK']);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /bad\.ron/);
});

test('makeDebounced collapses a burst of calls into one invocation', async () => {
  let count = 0;
  const d = makeDebounced(() => { count++; }, 20);
  d.call(); d.call(); d.call(); d.call(); d.call();   // 5 rapid calls
  assert.equal(count, 0, 'no invocation before the window elapses');
  await new Promise((r) => setTimeout(r, 40));
  assert.equal(count, 1, '5 rapid calls collapse to exactly 1 invocation');
});

test('makeDebounced cancel prevents a pending invocation', async () => {
  let count = 0;
  const d = makeDebounced(() => { count++; }, 20);
  d.call();
  d.cancel();
  await new Promise((r) => setTimeout(r, 40));
  assert.equal(count, 0);
});

// urlSourcesFeedSig — the skip/send decision behind the LSP urlSources feed.
// Empty payloads and re-feeds of identical content are skipped (re-feeding
// identical sources would needlessly re-run the server's cross-module
// inference); a changed payload yields a fresh signature to remember.

test('urlSourcesFeedSig skips an empty payload', () => {
  assert.equal(urlSourcesFeedSig([], undefined), null);
  assert.equal(urlSourcesFeedSig([], 'whatever'), null);
});

test('urlSourcesFeedSig returns a signature for the first non-empty feed', () => {
  const sig = urlSourcesFeedSig([{ uri: 'https://h/a', text: 'a = 1' }], undefined);
  assert.equal(typeof sig, 'string');
  assert.ok(sig.length > 0);
});

test('urlSourcesFeedSig skips a re-feed of identical content', () => {
  const sources = [{ uri: 'https://h/a', text: 'a = 1' }];
  const sig = urlSourcesFeedSig(sources, undefined);
  assert.equal(urlSourcesFeedSig(sources, sig), null, 'unchanged ⇒ skip');
});

test('urlSourcesFeedSig sends again when the content changes', () => {
  const sig1 = urlSourcesFeedSig([{ uri: 'https://h/a', text: 'a = 1' }], undefined);
  const sig2 = urlSourcesFeedSig([{ uri: 'https://h/a', text: 'a = 2' }], sig1);
  assert.equal(typeof sig2, 'string');
  assert.notEqual(sig2, sig1);
});
