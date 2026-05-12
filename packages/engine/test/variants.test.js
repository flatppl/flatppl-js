'use strict';

// Tests for the variant resolution machinery (./variants) and the
// processSource(src, opts) plumbing that threads a variant through to
// tokenize() and parse(). Per-variant grammar / lowering tests live
// alongside the feature commits that add them.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const {
  FLATPPL, FLATPPY, FLATPPJ, BY_ID, variantForPath, resolveVariant,
} = require('../variants');
const { processSource } = require('../index');

test('variants: three known ids', () => {
  assert.equal(FLATPPL.id, 'flatppl');
  assert.equal(FLATPPY.id, 'flatppy');
  assert.equal(FLATPPJ.id, 'flatppj');
  assert.deepEqual(Object.keys(BY_ID).sort(),
    ['flatppj', 'flatppl', 'flatppy']);
});

test('variants: path extension picks the variant', () => {
  assert.equal(variantForPath('foo.flatppl'),  FLATPPL);
  assert.equal(variantForPath('foo.flatppy'),  FLATPPY);
  assert.equal(variantForPath('foo.flatppj'),  FLATPPJ);
  // case-insensitive on the extension
  assert.equal(variantForPath('FOO.FLATPPL'),  FLATPPL);
  assert.equal(variantForPath('Foo.FlatPPy'),  FLATPPY);
  // path components don't matter
  assert.equal(variantForPath('a/b/c.flatppl'), FLATPPL);
  assert.equal(variantForPath('/abs/path/to/x.flatppj'), FLATPPJ);
  // unknown / missing extensions return null
  assert.equal(variantForPath('foo.txt'),       null);
  assert.equal(variantForPath('foo'),           null);
  assert.equal(variantForPath(''),              null);
  assert.equal(variantForPath(null),            null);
});

test('variants: resolveVariant precedence', () => {
  // 1. opts.variant beats opts.path
  assert.equal(resolveVariant({ variant: 'flatppy', path: 'foo.flatppl' }), FLATPPY);
  assert.equal(resolveVariant({ variant: FLATPPJ,   path: 'foo.flatppl' }), FLATPPJ);
  // 2. opts.path picks when variant absent
  assert.equal(resolveVariant({ path: 'foo.flatppy' }), FLATPPY);
  // 3. unknown extension falls through to FLATPPL default
  assert.equal(resolveVariant({ path: 'foo.txt' }), FLATPPL);
  // 4. empty opts → default FlatPPL
  assert.equal(resolveVariant({}),    FLATPPL);
  assert.equal(resolveVariant(),      FLATPPL);
  assert.equal(resolveVariant(null),  FLATPPL);
});

test('variants: unknown id throws', () => {
  assert.throws(() => resolveVariant({ variant: 'flatppz' }),
    /Unknown variant id/);
  assert.throws(() => resolveVariant({ variant: 42 }),
    /must be an id string or a known variant object/);
});

test('processSource: default variant is FlatPPL', () => {
  const r = processSource('x = 1.0');
  assert.equal(r.variant, FLATPPL);
  assert.deepEqual(r.diagnostics, []);
});

test('processSource: variant override via opts.variant', () => {
  const r1 = processSource('x = 1.0', { variant: 'flatppy' });
  assert.equal(r1.variant, FLATPPY);
  const r2 = processSource('x = 1.0', { variant: FLATPPJ });
  assert.equal(r2.variant, FLATPPJ);
});

test('processSource: variant inferred from opts.path', () => {
  const r = processSource('x = 1.0', { path: 'examples/foo.flatppy' });
  assert.equal(r.variant, FLATPPY);
});

test('processSource: parsing is unchanged at commit 1', () => {
  // The variant is plumbed but the base parser is still
  // variant-agnostic in commit 1 — the same source should produce the
  // same AST under every variant. Per-variant divergence lands in the
  // feature commits that follow.
  const src = 'x = 1.0\ny = 2.0';
  const flatppl = processSource(src, { variant: 'flatppl' });
  const flatppy = processSource(src, { variant: 'flatppy' });
  const flatppj = processSource(src, { variant: 'flatppj' });
  assert.deepEqual(flatppl.diagnostics, []);
  assert.deepEqual(flatppy.diagnostics, []);
  assert.deepEqual(flatppj.diagnostics, []);
  assert.equal(flatppl.bindings.size, 2);
  assert.equal(flatppy.bindings.size, 2);
  assert.equal(flatppj.bindings.size, 2);
});
