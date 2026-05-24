'use strict';

// Spec §04 / TODO-flatppl-js "Multi-file models" — engine API plumbing
// for module bundles. `processSource(src, { bundle: { sources } })`
// accepts a pre-resolved map of .flatppl module sources. The engine
// pipeline stays synchronous; async I/O lives in the host layer
// (vscode-extension uses workspace.fs; web uses fetch).
//
// This file pins the BUNDLE-SHAPE plumbing only. Cross-module
// reference resolution and load_module end-to-end are separate
// follow-ups (TODO-flatppl-js.md "Multi-file models" tasks 2-5).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

test('processSource: no opts ⇒ result.bundle is an empty bundle', () => {
  const r = processSource('x = 1');
  assert.ok(r.bundle, 'result.bundle present');
  assert.deepEqual(r.bundle, { sources: {} });
});

test('processSource: opts = {} ⇒ empty bundle still returned', () => {
  const r = processSource('x = 1', {});
  assert.deepEqual(r.bundle, { sources: {} });
});

test('processSource: opts.bundle is normalised onto result.bundle', () => {
  const r = processSource('x = 1', {
    bundle: {
      sources: {
        'helpers.flatppl': 'h = 42',
        'shared/model.flatppl': 'm = 7',
      },
    },
  });
  assert.deepEqual(r.bundle.sources['helpers.flatppl'], 'h = 42');
  assert.deepEqual(r.bundle.sources['shared/model.flatppl'], 'm = 7');
});

test('processSource: malformed bundle is normalised, not thrown', () => {
  // Non-string source values are filtered out (defensive normalisation).
  const r = processSource('x = 1', { bundle: { sources: { a: 'ok', b: 42 } } });
  assert.equal(r.bundle.sources.a, 'ok');
  assert.ok(!('b' in r.bundle.sources),
    'non-string source values dropped during normalisation');
});

test('processSource: variant resolution still works alongside bundle', () => {
  // Mixing variant + bundle: both honored.
  const r = processSource('x = 1', {
    path: 'foo.flatppl',
    bundle: { sources: {} },
  });
  assert.equal(r.variant.id, 'flatppl');
  assert.deepEqual(r.bundle, { sources: {} });
});

test('processSource: bundle does not yet affect analysis (no resolution yet)', () => {
  // Until load_module end-to-end lands, a bundle is held but not
  // consumed by analyzer/lower. This test pins the explicit
  // non-behavior — once the resolution path lands, this should
  // start showing the resolved bindings.
  const r = processSource(
    'h = load_module("helpers.flatppl")',
    { bundle: { sources: { 'helpers.flatppl': 'h_inner = 99' } } });
  // The current engine reports a load_module call but doesn't yet
  // resolve through the bundle. We just assert no crash + bundle
  // carried through.
  assert.deepEqual(r.bundle.sources['helpers.flatppl'], 'h_inner = 99');
});
