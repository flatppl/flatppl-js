'use strict';

// §07 `in`: "The type of x must match the element type of set S". Under §03 a
// value widens along booleans ⊂ integers ⊂ reals, and a real value is not a
// member of an integer set even when integral (flatppl-design #220), so
// `2.0 in integers` and `1 in booleans` are static errors; they evaluated to
// true. `integer(x)` and `boolean(x)` refuse a value outside their set when
// fixed bindings and literal arithmetic fold it, not only when it is written.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

const errorsOf = (src: string) =>
  processSource(src).diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);

test('a value whose type is wider than the set element type is refused', () => {
  for (const [src, x, set] of [
    ['a = 2.0 in integers\n', 'real', 'integer'],
    ['a = in(2.0, integers)\n', 'real', 'integer'],
    ['a = 1.0 in posintegers\n', 'real', 'integer'],
    ['a = 1 in booleans\n', 'integer', 'boolean'],
    ['a = 0.5 in booleans\n', 'real', 'boolean'],
    ['v = [1.0, 2.0]\na = v in cartpow(integers, 2)\n', 'array of real \\(length 2\\)', 'array of integer \\(length 2\\)'],
  ]) {
    const errs = errorsOf(src);
    assert.equal(errs.length, 1, src + errs.join('; '));
    assert.match(errs[0], new RegExp('^in: x is ' + x + ", but the set's element type is " + set
      + '; spec §07: "The type of x must match the element type of set S"'), src);
  }
});

test('a value of the element type, or one that widens to it, is accepted', () => {
  for (const src of ['a = 2 in integers\n', 'a = true in booleans\n', 'a = true in integers\n',
    'a = 2 in reals\n', 'a = 2.5 in reals\n', 'a = 1 in interval(0, 2)\n',
    'a = [1, 2] in cartpow(integers, 2)\n', 'a = [1, 2] in cartpow(reals, 2)\n', 'a = integer(2.0) in integers\n']) {
    assert.deepEqual(errorsOf(src), [], src);
  }
});

test('integer and boolean refuse a folded value outside their set', () => {
  const outside = (op: string, v: string) => new RegExp('^' + op + ': ' + v + ' is not in `' + op + 's`');
  assert.match(errorsOf('y = 2.5\nz = integer(y)\n').join('\n'), outside('integer', '2\\.5'));
  assert.match(errorsOf('z = integer(2.5 + 0.0)\n').join('\n'), outside('integer', '2\\.5'));
  assert.match(errorsOf('z = integer(inf)\n').join('\n'), outside('integer', 'inf'));
  assert.match(errorsOf('z = boolean(1.0 + 1.0)\n').join('\n'), outside('boolean', '2'));
  for (const src of ['y = 3.0\nz = integer(y)\n', 'z = integer(2.0 + 1.0)\n', 'z = boolean(1.0 * 1.0)\n',
    'k = elementof(reals)\nz = integer(k)\n']) {
    assert.deepEqual(errorsOf(src), [], src);
  }
});
