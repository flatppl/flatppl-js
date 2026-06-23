'use strict';

// Spec §04 (Module composition, "Path resolution"): relative paths in
// `load_module(...)` resolve relative to the DIRECTORY of the file
// containing the call; `/` is the mandatory separator on all platforms;
// `..` traversal is allowed; absolute paths are permitted. These are the
// pure path-resolution rules, shared by the engine bundle-compile and the
// host resolvers (web fetch / vscode workspace.fs).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { resolveModulePath } = require('../module-resolve.ts');

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
