'use strict';

// Spec §04 "Broadcasting" (flatppl-design #210): `broadcasted(f)(args...)` is `broadcast(f, args...)`
// and dot-call is the same, so every rule below holds in all three spellings.
//   - A table operand is traversed row-wise (§03), each row a record passed to `f`.
//   - A record operand is held constant for every call, like a scalar.
//   - A function broadcast over tuple operands iterates over the components and returns a tuple:
//     `broadcast(f, (a, b), c)` equals `(f(a, c), f(b, c))`. All tuple operands have the same
//     length (a tuple has at least two components, so no length-one operand repeats). A tuple
//     combined with an array or table operand, or a kernel broadcast over a tuple, is a
//     static error.
// ORACLES are the componentwise values written out by hand.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const F = 'f(a, b) = a + b\n';

function errors(src: string) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

async function values(src: string) {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('r');
  return Array.from(m.value ? m.value.data : m.samples);
}

// The three spellings of one call.
const spellings = (head: string, operands: string) => [
  `r = broadcast(${head}, ${operands})\n`,
  `r = ${head}.(${operands})\n`,
  `r = broadcasted(${head})(${operands})\n`,
];

test('a table operand is traversed row-wise, through broadcast, dot-call and broadcasted alike', async () => {
  const t = 'table(a = [1.0, 2.0], b = [3.0, 4.0])';
  for (const call of spellings('f', t)) assert.deepEqual(await values(F + call), [4, 6]);
  assert.deepEqual(await values(F + `bc = broadcasted(f)\nr = bc(${t})\n`), [4, 6]);
});

test('collection operands broadcast the same through both spellings', async () => {
  assert.deepEqual(await values(F + 'r = broadcasted(f)([1.0, 2.0], [3.0, 4.0])\n'), [4, 6]);
  assert.deepEqual(await values(F + 'r = broadcasted(f)(a = [1.0, 2.0], b = [3.0, 4.0])\n'), [4, 6]);
  assert.deepEqual(await values(F + 'r = broadcasted(f)([1.0, 2.0], 10.0)\n'), [11, 12]);
});

test('a record operand is held constant for every call', async () => {
  const g = 'g(a, c) = a + c.k\nrr = (k = 100.0)\n';
  for (const call of spellings('g', '[1.0, 2.0], rr')) {
    assert.deepEqual(errors(g + call), []);
    assert.deepEqual(await values(g + call), [101, 102]);
  }
  assert.deepEqual(await values(g + 'bc = broadcasted(g)\nr = bc([1.0, 2.0], rr)\n'), [101, 102]);
  // Held constant, also when every operand is a record: one call.
  assert.deepEqual(await values('h(p, q) = p.k * q.k\nr = broadcast(h, (k = 3.0), (k = 4.0))\n'), [12]);
});

test('a function broadcast over tuple operands iterates the components and returns a tuple', async () => {
  const result = (src: string) => processSource(src).loweredModule.bindings.get('r').inferredType;
  for (const call of spellings('f', '(1.0, 2.0), 10.0')) {
    assert.deepEqual(errors(F + call), []);
    assert.deepEqual(await values(F + call), [11, 12]);                // (f(1, 10), f(2, 10))
    const type = result(F + call);
    assert.equal(type.kind, 'tuple');
    assert.equal(type.elems.length, 2);
  }
  // Two tuples iterate together: (1 * 3, 2 * 4).
  for (const call of spellings('m', '(1.0, 2.0), (3.0, 4.0)')) {
    assert.deepEqual(await values('m(a, b) = a * b\n' + call), [3, 8]);
  }
  // The argument order follows the operands, not the tuple: f((a, b), c) is (f(a, c), f(b, c)).
  assert.deepEqual(await values('s(a, c) = a - c\nr = broadcast(s, 10.0, (1.0, 2.0))\n'), [9, 8]);
});

test('a tuple component is passed whole, even when it is an array', async () => {
  const src = 'f(a, c) = sum(a) + c\nr = broadcast(f, ([1.0, 2.0], [3.0, 4.0, 5.0]), 10.0)\n';
  assert.deepEqual(errors(src), []);
  // (sum([1, 2]) + 10, sum([3, 4, 5]) + 10)
  assert.deepEqual(await values(src), [13, 22]);
});

test('tuple operands must have the same length', () => {
  const errs = errors(F + 'r = broadcast(f, (1.0, 2.0), (1.0, 2.0, 3.0))\n');
  assert.ok(errs.some((d: any) => /tuple operands have different lengths 2, 3/.test(d.message) && d.loc));
});

test('a tuple combined with an array or table operand is a static error', () => {
  for (const operand of ['[1.0, 2.0]', 'table(a = [1.0, 2.0], b = [3.0, 4.0])']) {
    for (const call of spellings('f', `(1.0, 2.0), ${operand}`)) {
      assert.ok(errors(F + call).some((d: any) => /cannot be combined with an array or table/.test(d.message)), call);
    }
  }
});

test('a kernel broadcast over a tuple is a static error', () => {
  const kernel = 'K(m) = Normal(m, 1.0)\n';
  for (const call of spellings('K', '(1.0, 2.0)')) {
    assert.ok(errors(kernel + call).some((d: any) => /a kernel cannot be broadcast over a tuple/.test(d.message)), call);
  }
  assert.ok(errors('r = Normal.((1.0, 2.0), 1.0)\n').some((d: any) => /a kernel cannot be broadcast over a tuple/.test(d.message)));
});

test('a component the function cannot take is reported once, not as a tuple error', () => {
  // sum takes arrays: the first component is a scalar.
  const errs = errors('f(a, c) = sum(a) + c\nr = broadcast(f, (1.0, [1.0, 2.0]), 10.0)\n');
  assert.ok(errs.some((d: any) => /sum:/.test(d.message)));
  assert.equal(errs.filter((d: any) => /Tuple inputs/.test(d.message)).length, 0);
});
