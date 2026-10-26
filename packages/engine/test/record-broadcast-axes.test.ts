'use strict';

// Spec §04 "Record-returning callables": broadcast collects record results
// row-wise into a table. "This requires one broadcast axis; multiple axes are a
// static error." typeinfer checks the outer rank, but `addaxes` was untyped
// (deferred), so `broadcast(f, addaxes(v, 0, 1))` passed typeinfer and only
// failed at evaluation. §07 `addaxes(A, n_leading, n_trailing)` adds
// n_leading size-one axes before A's axes and n_trailing after them.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const BASE = 'v = [1.0, 2.0]\nM = rowstack([[1.0, 2.0], [3.0, 4.0]])\nf(x) = record(a = x, b = 2.0 * x)\n';
const MESSAGE = 'record-valued broadcast requires one axis (spec §04)';

function errors(src: string) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

for (const body of [
  't = f.(addaxes(v, 0, 1))',
  't = f.(addaxes(v, 1, 0))',
  't = broadcast(fn(record(a = _)), addaxes(v, 2, 0))',
  't = broadcast(record, a = addaxes(v, 0, 1))',
  't = f.(M)',
]) {
  test(`a record broadcast over more than one axis is a located static error: ${body}`, () => {
    const errs = errors(BASE + body + '\n');
    assert.deepEqual(errs.map((d: any) => d.message), [MESSAGE]);
    assert.equal(errs[0].loc.start.line, 3);
  });
}

test('a record broadcast over addaxes(v, 0, 0) is one axis and collects a table', async () => {
  const src = BASE + 't = f.(addaxes(v, 0, 0))\n';
  assert.deepEqual(errors(src), []);
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  const t = await ctx.getMeasure('t');
  const cols = t.fields.columns.fields;
  assert.deepEqual(Array.from(cols.a.value.data), [1, 2]);
  assert.deepEqual(Array.from(cols.b.value.data), [2, 4]);
});

// §07: addaxes(A, 2, 3) of a (3, 4, 5) array has size (1, 1, 3, 4, 5, 1, 1, 1).
test('addaxes is typed with the size-one axes around the operand shape', () => {
  const r = processSource(BASE + 'n = 1\nw = addaxes(v, 0, 1)\nq = addaxes(M, 1, 1)\np = addaxes(v, n, 0)\n'
    + 'A = fill(0.0, [3, 4, 5])\nB = addaxes(A, 2, 3)\n');
  const shape = (k: string) => r.linkedBindings.get(k).inferredType.shape;
  assert.deepEqual(shape('w'), [2, 1]);
  assert.deepEqual(shape('q'), [1, 2, 2, 1]);
  assert.deepEqual(shape('p'), [1, 2]);
  assert.deepEqual(shape('B'), [1, 1, 3, 4, 5, 1, 1, 1]);
});
