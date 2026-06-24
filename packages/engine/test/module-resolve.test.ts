'use strict';

// Spec §04 (Module composition, "Path resolution"): relative paths in
// `load_module(...)` resolve relative to the DIRECTORY of the file
// containing the call; `/` is the mandatory separator on all platforms;
// `..` traversal is allowed; absolute paths are permitted. These are the
// pure path-resolution rules, shared by the engine bundle-compile and the
// host resolvers (web fetch / vscode workspace.fs).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { resolveModulePath, isUrl } = require('../module-resolve.ts');

test('sibling file resolves against the importer directory', () => {
  // importer at repo root → dirname is "" → sibling stays bare.
  assert.equal(resolveModulePath('model.flatppl', 'helpers.flatppl'),
    'helpers.flatppl');
  // importer in a subdir → sibling picks up that subdir.
  assert.equal(resolveModulePath('a/model.flatppl', 'helpers.flatppl'),
    'a/helpers.flatppl');
  assert.equal(resolveModulePath('a/b/model.flatppl', 'helpers.flatppl'),
    'a/b/helpers.flatppl');
});

test('"." current-dir segments are collapsed', () => {
  assert.equal(resolveModulePath('a/model.flatppl', './helpers.flatppl'),
    'a/helpers.flatppl');
  assert.equal(resolveModulePath('a/model.flatppl', './sub/./h.flatppl'),
    'a/sub/h.flatppl');
});

test('".." traversal walks up the importer directory', () => {
  assert.equal(resolveModulePath('a/b/model.flatppl', '../shared/h.flatppl'),
    'a/shared/h.flatppl');
  assert.equal(resolveModulePath('a/b/c/model.flatppl', '../../h.flatppl'),
    'a/h.flatppl');
});

test('absolute relPath (leading /) is used as-is, normalised', () => {
  assert.equal(resolveModulePath('a/model.flatppl', '/abs/h.flatppl'),
    '/abs/h.flatppl');
  assert.equal(resolveModulePath('a/model.flatppl', '/abs/./x/../h.flatppl'),
    '/abs/h.flatppl');
});

test('an absolute importer path resolves siblings as absolute', () => {
  // The VS Code host passes an absolute URI path as the importer; a
  // sibling/`..` dep must stay absolute (not silently become relative).
  assert.equal(resolveModulePath('/proj/model.flatppl', 'helpers.flatppl'),
    '/proj/helpers.flatppl');
  assert.equal(resolveModulePath('/proj/sub/model.flatppl', '../helpers.flatppl'),
    '/proj/helpers.flatppl');
  assert.equal(resolveModulePath('/a/b/c/m.flatppl', '../../x.flatppl'),
    '/a/x.flatppl');
});

test('a null/empty importer path resolves relative to the root', () => {
  assert.equal(resolveModulePath(null, 'helpers.flatppl'), 'helpers.flatppl');
  assert.equal(resolveModulePath('', 'sub/h.flatppl'), 'sub/h.flatppl');
});

test('leading ".." that escapes the root is preserved (not silently dropped)', () => {
  // importer at root, escaping upward — unusual but spec-legal; keep the
  // segment rather than collapse it into a wrong same-dir path.
  assert.equal(resolveModulePath('model.flatppl', '../h.flatppl'),
    '../h.flatppl');
});

// --- URL sources (#sec:url-cache): http/https deps + relative deps inside a
// URL-loaded module. resolveModulePath stays the single owner, so a URL dep
// keys identically for the host bundle walk and the engine bundle compiler.

test('isUrl recognizes http/https, rejects local paths and file://', () => {
  assert.ok(isUrl('http://h/x') && isUrl('https://h/x') && isUrl('HTTPS://h/x'));
  assert.ok(!isUrl('./x.flatppl') && !isUrl('/abs/x.flatppl') && !isUrl('rel/x.flatppl') && !isUrl('file:///x'));
});

test('an absolute URL dependency is used verbatim (from a local importer)', () => {
  assert.equal(resolveModulePath('/proj/main.flatppl', 'https://h/lib/m.flatppl'),
    'https://h/lib/m.flatppl');
  assert.equal(resolveModulePath('a/model.flatppl', 'http://h/m.flatppl'),
    'http://h/m.flatppl');
});

test('a relative dep inside a URL module resolves against the importer URL', () => {
  assert.equal(resolveModulePath('https://h/a/main.flatppl', 'helper.flatppl'),
    'https://h/a/helper.flatppl');
  assert.equal(resolveModulePath('https://h/a/main.flatppl', '../b/x.flatppl'),
    'https://h/b/x.flatppl');
  assert.equal(resolveModulePath('https://h/a/main.flatppl', './sub/h.flatppl'),
    'https://h/a/sub/h.flatppl');
});

test('an absolute URL dep from a URL importer overrides the base', () => {
  assert.equal(resolveModulePath('https://h/a/main.flatppl', 'https://other/y.flatppl'),
    'https://other/y.flatppl');
});
