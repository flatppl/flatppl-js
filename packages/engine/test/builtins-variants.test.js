'use strict';

// Unit tests for builtins.ts predicate helpers + variants.ts shim.
// Both are leaf modules with small surfaces; the higher-level
// integration tests exercise their data sets transitively, but the
// predicate functions themselves and variants.variantForPath /
// resolveVariant weren't covered directly. These tests close the
// coverage gap and make regressions (e.g. accidentally narrowing
// CONSTANTS) fail at the unit level rather than buried in a chain
// of integration failures.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const builtins = require('../builtins.ts');
const variants = require('../variants.ts');

// =====================================================================
// builtins: name-classification predicates
// =====================================================================

test('builtins.isKnownName: recognises every entry in the union', () => {
  // Sample one entry from each subset (the whole union is enormous).
  for (const name of ['pi', 'reals', 'interval', 'draw', 'add', 'Normal',
                      'weighted', 'self']) {
    assert.equal(builtins.isKnownName(name), true, name);
  }
});

test('builtins.isKnownName: returns false for unknown names', () => {
  assert.equal(builtins.isKnownName('totally_not_a_real_name'), false);
  assert.equal(builtins.isKnownName(''), false);
});

test('builtins.isConstant: only true for CONSTANTS', () => {
  assert.equal(builtins.isConstant('pi'),       true);
  assert.equal(builtins.isConstant('inf'),      true);
  assert.equal(builtins.isConstant('im'),       true);
  assert.equal(builtins.isConstant('Normal'),   false);
  assert.equal(builtins.isConstant('reals'),    false);
  assert.equal(builtins.isConstant('not_a_constant'), false);
});

test('builtins.isBoolLiteral: only true for "true"/"false"', () => {
  assert.equal(builtins.isBoolLiteral('true'),  true);
  assert.equal(builtins.isBoolLiteral('false'), true);
  assert.equal(builtins.isBoolLiteral('pi'),    false);
  assert.equal(builtins.isBoolLiteral(''),      false);
});

test('builtins.isSet: true for named sets, false otherwise', () => {
  assert.equal(builtins.isSet('reals'),     true);
  assert.equal(builtins.isSet('posreals'),  true);
  assert.equal(builtins.isSet('booleans'),  true);
  // interval is a set CONSTRUCTOR not a set; SET_CONSTRUCTORS is separate.
  assert.equal(builtins.isSet('interval'),  false);
  assert.equal(builtins.isSet('Normal'),    false);
});

test('builtins.isSpecialOperation: true for ops with custom syntax rules', () => {
  for (const op of ['draw', 'lawof', 'functionof', 'kernelof', 'fn',
                    'elementof', 'external', 'load_module', 'broadcast',
                    'broadcasted', 'reduce', 'scan', 'fchain', 'bijection',
                    'checked', 'record', 'table', 'tuple', 'vector',
                    'tuple_get']) {
    assert.equal(builtins.isSpecialOperation(op), true, op);
  }
  // Ordinary builtins / distributions / measure ops are NOT special.
  assert.equal(builtins.isSpecialOperation('add'),    false);
  assert.equal(builtins.isSpecialOperation('Normal'), false);
  assert.equal(builtins.isSpecialOperation('iid'),    false);
});

test('builtins.isReserved: true for the reserved name set', () => {
  assert.equal(builtins.isReserved('self'),  true);
  assert.equal(builtins.isReserved('base'),  true);
  assert.equal(builtins.isReserved('mu'),    false);
  assert.equal(builtins.isReserved('Normal'), false);
});

test('builtins: name-sets are non-overlapping pairwise (sanity)', () => {
  // RESERVED is disjoint from every other set — accidentally bridging
  // them is a real bug class (a binding called `self` should be flagged
  // by the analyzer, not collide with a distribution name).
  const sets = {
    CONSTANTS:        builtins.CONSTANTS,
    BOOL_LITERALS:    builtins.BOOL_LITERALS,
    SETS:             builtins.SETS,
    SET_CONSTRUCTORS: builtins.SET_CONSTRUCTORS,
    RESERVED_NAMES:   builtins.RESERVED_NAMES,
  };
  const names = Object.keys(sets);
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const a = sets[names[i]], b = sets[names[j]];
      for (const x of a) {
        assert.equal(b.has(x), false,
          `name "${x}" appears in both ${names[i]} and ${names[j]}`);
      }
    }
  }
});

// =====================================================================
// variants: surface-syntax shim
// =====================================================================

test('variants.FLATPPL: shape conforms to the variants contract', () => {
  // Whatever shape FLATPPL has, downstream code (parser / tokenizer
  // / analyzer dispatch) reads at minimum `id`. The id has been
  // 'flatppl' since the surface unification (flatppl-design cc81e4b).
  assert.equal(variants.FLATPPL.id, 'flatppl');
});

test('variants.variantForPath: extension dispatch', () => {
  assert.equal(variants.variantForPath('foo.flatppl'), variants.FLATPPL);
  assert.equal(variants.variantForPath('/a/b/c.flatppl'), variants.FLATPPL);
  // Spec says ANY non-empty path returns FLATPPL (post-unification).
  assert.equal(variants.variantForPath('foo.flatppy'), variants.FLATPPL);
  assert.equal(variants.variantForPath('foo.txt'),     variants.FLATPPL);
});

test('variants.variantForPath: null / empty / non-string returns null', () => {
  assert.equal(variants.variantForPath(null),      null);
  assert.equal(variants.variantForPath(undefined), null);
  assert.equal(variants.variantForPath(''),        null);
  assert.equal(variants.variantForPath(42),        null);
});

test('variants.resolveVariant: undefined / empty opts → FLATPPL', () => {
  assert.equal(variants.resolveVariant(undefined),       variants.FLATPPL);
  assert.equal(variants.resolveVariant(null),            variants.FLATPPL);
  assert.equal(variants.resolveVariant({}),              variants.FLATPPL);
  assert.equal(variants.resolveVariant({ variant: null }),    variants.FLATPPL);
});

test('variants.resolveVariant: string id "flatppl" accepted', () => {
  assert.equal(variants.resolveVariant({ variant: 'flatppl' }), variants.FLATPPL);
});

test('variants.resolveVariant: object id "flatppl" accepted', () => {
  assert.equal(variants.resolveVariant({ variant: { id: 'flatppl' } }), variants.FLATPPL);
});

test('variants.resolveVariant: removed variant ids throw with migration hint', () => {
  assert.throws(() => variants.resolveVariant({ variant: 'flatppy' }),
                /'flatppy' was removed/);
  assert.throws(() => variants.resolveVariant({ variant: { id: 'flatppj' } }),
                /'flatppj' was removed/);
});
