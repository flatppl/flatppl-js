'use strict';

// A boolean is JS `true`/`false` as a scalar but 1/0 once stored in an array,
// so `mask[1]` reads back as the number 1. Spec §07 defines `equal`, `unequal`
// and `lxor` on `booleans`, and §03 `booleans` holds exactly `true` and
// `false`. A comparison must therefore not depend on which spelling reaches it.
// The scalar tables compared with JS identity, so `equal(mask[1], true)` was
// false and `true in booleans` was false.
//
// ORACLE: the truth tables of =, != and exclusive-or on {false, true}, with
// each operand read as its boolean value (1 and true are both true). No engine
// output is used.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const ops = require('../ops.ts');
const sampler = require('../sampler.ts');
require('../ops-declarations.ts');
const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');

const ARITH_OPS = sampler._internal.ARITH_OPS;
const SPELLINGS = [0, 1, false, true];
const truth = (x: any) => x === true || x === 1;
const TABLE: Record<string, (a: boolean, b: boolean) => boolean> = {
  equal: (a, b) => a === b,
  unequal: (a, b) => a !== b,
  lxor: (a, b) => a !== b,
};

function fixedValues(src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), []);
  return buildDerivations(r.bindings).fixedValues;
}

for (const op of Object.keys(TABLE)) {
  test(`${op} matches its truth table for every mix of 0/1 and false/true`, () => {
    for (const a of SPELLINGS) {
      for (const b of SPELLINGS) {
        const want = TABLE[op](truth(a), truth(b));
        assert.equal(!!ARITH_OPS[op](a, b), want, `ARITH_OPS.${op}(${a}, ${b})`);
        assert.equal(!!ops.dispatch(op, [a, b]), want, `ops.dispatch ${op}(${a}, ${b})`);
      }
    }
  });
}

test('a boolean read from a vector compares equal to its literal', () => {
  const fv = fixedValues(`
mask = [true, false, true]
a = equal(mask[1], true)
b = unequal(mask[1], true)
c = lxor(mask[1], true)
d = equal(mask[2], false)
e = mask[1] == true
f = mask[2] != false
g = lxor(mask[2], true)
`);
  const want: Record<string, boolean> = { a: true, b: false, c: false, d: true, e: true, f: false, g: true };
  for (const [k, v] of Object.entries(want)) assert.equal(!!fv.get(k), v, k);
});

test('integer and string equality keep their identity semantics', () => {
  const fv = fixedValues(`
a = equal(3, 3)
b = equal(3, 4)
c = unequal(2, 2)
`);
  assert.equal(fv.get('a'), true);
  assert.equal(fv.get('b'), false);
  assert.equal(fv.get('c'), false);
  for (const fn of [(a: any, b: any) => ARITH_OPS.equal(a, b), (a: any, b: any) => ops.dispatch('equal', [a, b])]) {
    assert.equal(fn('a', 'a'), true);
    assert.equal(fn('a', 'b'), false);
    assert.equal(fn('1', 1), false);
  }
});

// §03: booleans ⊂ integers, so a boolean is also in integers and nonnegintegers,
// and true (= 1) is in posintegers.
test('set membership accepts both spellings of a boolean', () => {
  const fv = fixedValues(`
mask = [true, false]
a = true in booleans
b = false in booleans
c = mask[1] in booleans
d = 2 in booleans
e = 0.5 in booleans
f = true in integers
g = true in nonnegintegers
h = true in posintegers
i = false in posintegers
j = false in nonnegintegers
`);
  const want: Record<string, boolean> = {
    a: true, b: true, c: true, d: false, e: false, f: true, g: true, h: true, i: false, j: true,
  };
  for (const [k, v] of Object.entries(want)) assert.equal(!!fv.get(k), v, k);
});
