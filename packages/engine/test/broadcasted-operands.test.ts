'use strict';

// Spec §04 "Broadcasting" (flatppl-design #210): `broadcasted(f)(args...)` is `broadcast(f, args...)`
// and dot-call is the same, so every rule below holds in all three spellings.
//   - A table operand is traversed row-wise (§03), each row a record passed to `f`.
//   - "Records and tuples are not allowed as operands of `broadcast`": neither iterated nor held
//     constant, so each is a located static error in every spelling and for every kind of callee.
//     Auto-splatting applies to the broadcast function at each per-element call (a table row reaches
//     the callable as a record), never to the operands of `broadcast`.
// ORACLES are the values written out by hand.

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

test('a record row of a table reaches the callable under its own calling convention', async () => {
  const t = 'table(a = [1.0, 2.0], b = [3.0, 4.0])';
  // One input takes the row record whole.
  for (const call of spellings('h', t)) assert.deepEqual(await values('h(row) = row.a * row.b\n' + call), [3, 8]);
  // Two or more inputs splat the row by field name.
  for (const call of spellings('f', t)) assert.deepEqual(await values(F + call), [4, 6]);
});

const OPERANDS: Array<[string, string, RegExp]> = [
  ['a record', '(k = 100.0)', /operand 2 is a record/],
  ['a record binding', 'rr', /operand 2 is a record/],
  ['a tuple', '(10.0, 20.0)', /operand 2 is a tuple/],
  ['a tuple binding', 'tt', /operand 2 is a tuple/],
];
const BINDINGS = 'rr = (k = 100.0)\ntt = (10.0, 20.0)\n';

test('a record or tuple operand is a located static error in every spelling', () => {
  const g = 'g(a, c) = a\n';
  for (const [name, operand, expected] of OPERANDS) {
    for (const call of spellings('g', `[1.0, 2.0], ${operand}`)) {
      const errs = errors(BINDINGS + g + call);
      assert.ok(errs.some((d: any) => expected.test(d.message) && /Disallowed inputs/.test(d.message) && d.loc), `${name}: ${call}`);
    }
    // Also when no collection is involved, and when the record is the only operand.
    for (const call of spellings('g', `1.0, ${operand}`)) {
      assert.ok(errors(BINDINGS + g + call).some((d: any) => expected.test(d.message)), `${name}: ${call}`);
    }
    const bc = 'bc = broadcasted(g)\nr = bc([1.0, 2.0], ' + operand + ')\n';
    assert.ok(errors(BINDINGS + g + bc).some((d: any) => expected.test(d.message)), `${name}: ${bc}`);
  }
  // The first operand too: every operand is judged, not just the later ones.
  for (const call of spellings('f', 'rr, 1.0')) {
    assert.ok(errors(BINDINGS + F + call).some((d: any) => /operand 1 is a record/.test(d.message)), call);
  }
});

test('a record or tuple operand is refused for builtin and kernel callees too', () => {
  for (const [src, expected] of [
    ['r = exp.(rr)\n', /operand 1 is a record/],
    ['r = sqrt.(tt)\n', /operand 1 is a tuple/],
    ['r = Normal.(rr, 1.0)\n', /operand 1 is a record/],
    ['r = Normal.((1.0, 2.0), 1.0)\n', /operand 1 is a tuple/],
    ['r = broadcast(Normal, [1.0, 2.0], rr)\n', /operand 2 is a record/],
    ['K(m) = Normal(m, 1.0)\nr = broadcast(K, (1.0, 2.0))\n', /operand 1 is a tuple/],
    ['r = broadcasted(exp)(rr)\n', /operand 1 is a record/],
    ['r = rr .+ 1.0\n', /operand 1 is a record/],
    ['r = [1.0, 2.0] .* tt\n', /operand 2 is a tuple/],
  ] as Array<[string, RegExp]>) {
    assert.ok(errors(BINDINGS + src).some((d: any) => expected.test(d.message) && d.loc), src);
  }
});

test('a record or tuple operand is refused at run time when static checking is bypassed', () => {
  const sampler = require('../sampler.ts');
  const body = { kind: 'call', op: 'add', args: [{ kind: 'ref', ns: '%local', name: 'a' }, { kind: 'lit', value: 1 }] };
  const fn = { kind: 'call', op: 'functionof', params: ['a'], body };
  const operand = { kind: 'call', op: 'record', fields: [{ name: 'k', value: { kind: 'lit', value: 1 } }] };
  assert.throws(() => sampler.evaluateExpr({ kind: 'call', op: 'broadcast', args: [fn, operand] }, {}),
    /records and tuples are not allowed as broadcast operands/);
});
