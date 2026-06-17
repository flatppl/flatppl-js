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
