// Unit tests for the pure LSP-source decision logic in build-lsp-source.cjs.
// No I/O: chooseLspSource takes pre-probed booleans/strings and returns a
// route descriptor; build-vendor.mjs does the actual fs/spawn work.
const test = require('node:test');
const assert = require('node:assert/strict');
const { hostTripleForNode, chooseLspSource } = require('../build-lsp-source.cjs');

test('hostTripleForNode maps known hosts and rejects unknown', () => {
  assert.deepEqual(hostTripleForNode('darwin', 'arm64'), { triple: 'aarch64-apple-darwin', exe: '' });
  assert.deepEqual(hostTripleForNode('darwin', 'x64'),  { triple: 'x86_64-apple-darwin', exe: '' });
  assert.deepEqual(hostTripleForNode('linux', 'x64'), { triple: 'x86_64-unknown-linux-musl', exe: '' });
  assert.deepEqual(hostTripleForNode('linux', 'arm64'), { triple: 'aarch64-unknown-linux-musl', exe: '' });
  assert.deepEqual(hostTripleForNode('win32', 'x64'), { triple: 'x86_64-pc-windows-msvc', exe: '.exe' });
  assert.equal(hostTripleForNode('sunos', 'sparc'), null);
});

test('chooseLspSource honours precedence: bin-dir first', () => {
  assert.deepEqual(
    chooseLspSource({ binDir: '/staged', localBin: '/x', siblingExists: true, cargoAvailable: true }),
    { route: 'bin-dir', dir: '/staged' },
  );
});

test('chooseLspSource: local override beats sibling/clone/download', () => {
  assert.deepEqual(
    chooseLspSource({ localBin: '/my/flatppl-lsp', siblingExists: true, cargoAvailable: true }),
    { route: 'local', path: '/my/flatppl-lsp' },
  );
});

test('chooseLspSource: sibling beats clone/download', () => {
  assert.deepEqual(
    chooseLspSource({ siblingExists: true, cargoAvailable: true }),
    { route: 'sibling' },
  );
});

test('chooseLspSource: no sibling + cargo → clone', () => {
  assert.deepEqual(
    chooseLspSource({ siblingExists: false, cargoAvailable: true }),
    { route: 'clone' },
  );
});

test('chooseLspSource: no sibling + no cargo → download', () => {
  assert.deepEqual(
    chooseLspSource({ siblingExists: false, cargoAvailable: false }),
    { route: 'download' },
  );
});
