'use strict';

// Spec §04 (flatppl-design #210, owner-accepted): `broadcasted(f)(args...)` is
// `broadcast(f, args...)`; its arguments are operands of `broadcast`, never
// auto-splatted, and "records and tuples are not allowed as operands of broadcast".
// A table operand is traversed row-wise (§03), each row a record passed to `f`.
// ORACLE: elementwise sums of the two columns, 1+3 and 2+4.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const F = 'f(a, b) = a + b\n';
const REC = 'rr = record(a = [1.0, 2.0], b = [3.0, 4.0])\n';

function errors(src: string) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

async function values(src: string) {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('r');
  return Array.from(m.value ? m.value.data : m.samples);
}

const RECORD_OPERAND = /a record cannot be an operand of broadcast/;
const TUPLE_OPERAND = /a tuple cannot be an operand of broadcast/;

test('a record operand is a static error in every spelling', () => {
  for (const call of ['broadcast(f, rr)', 'broadcasted(f)(rr)', 'broadcast(f, record(a = [1.0, 2.0], b = [3.0, 4.0]))']) {
    const errs = errors(F + REC + `r = ${call}\n`);
    assert.ok(errs.some((d: any) => RECORD_OPERAND.test(d.message) && d.loc), `${call}: ${errs.map((d: any) => d.message)}`);
  }
  const viaBinding = errors(F + REC + 'bc = broadcasted(f)\nr = bc(rr)\n');
  assert.ok(viaBinding.some((d: any) => RECORD_OPERAND.test(d.message)));
});

test('a tuple operand is a static error', () => {
  const errs = errors(F + 'r = broadcast(f, tuple([1.0], [2.0]))\n');
  assert.ok(errs.some((d: any) => TUPLE_OPERAND.test(d.message)));
});

test('a table operand is traversed row-wise, through broadcast and broadcasted alike', async () => {
  const t = 'table(a = [1.0, 2.0], b = [3.0, 4.0])';
  assert.deepEqual(await values(F + `r = broadcast(f, ${t})\n`), [4, 6]);
  assert.deepEqual(await values(F + `r = broadcasted(f)(${t})\n`), [4, 6]);
  assert.deepEqual(await values(F + `bc = broadcasted(f)\nr = bc(${t})\n`), [4, 6]);
});

test('collection operands broadcast the same through both spellings', async () => {
  assert.deepEqual(await values(F + 'r = broadcasted(f)([1.0, 2.0], [3.0, 4.0])\n'), [4, 6]);
  assert.deepEqual(await values(F + 'r = broadcasted(f)(a = [1.0, 2.0], b = [3.0, 4.0])\n'), [4, 6]);
  assert.deepEqual(await values(F + 'r = broadcasted(f)([1.0, 2.0], 10.0)\n'), [11, 12]);
});
