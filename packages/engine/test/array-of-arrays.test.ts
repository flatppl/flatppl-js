'use strict';

// Arrays of arrays (§03 "Vectors of vectors are not interpreted as matrices
// implicitly") and the §09 ext-linear-algebra records.
//
// `[v, w]` and `[[1, 2], [3, 4]]` evaluate to one nested Value, its leading
// `outerRank` axes indexing the inner arrays. joinblocks and blockdiagmat read
// only nested JS arrays, so a nested Value failed: blockdiagmat([A, B]) and
// joinblocks([v, w]) had no value at all, and joinblocks([[1, 2], [3, 4]])
// refused with "outer rank (0)". fill(A, size) with an array A gave NaN cells.
//
// ext-linear-algebra returned `{ shape: 'record', fields }`, while a record
// value is the plain field object, so `la.lu(A).U` could not be read. lstsq
// divided by an unpivoted R diagonal: a numerically rank-deficient A gave
// values near 1e15, an exactly rank-deficient one or a wide one threw.
//
// ORACLE: the §07 examples as stated, and numpy 2.5.1 `np.linalg.lstsq(A, b,
// rcond=None)` (the minimum-norm solution) through the python MCP.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function fixed(src: string, name: string): any {
  const errs = (processSource(src).diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [], src);
  return makeMatCtx(src).built.fixedValues.get(name);
}

function array(src: string, name: string): { shape: number[], data: number[] } {
  const v = fixed(src, name);
  assert.ok(v && v.data, `${name} has a value`);
  return { shape: v.shape, data: Array.from(v.data) };
}

// §07 `blockdiagmat`: the stated 5 × 5 result.
test('blockdiagmat reads a vector of matrix bindings (§07 example)', () => {
  const src = 'A = rowstack([[1, 2], [3, 4]])\nB = rowstack([[5, 6, 7], [8, 9, 10]])\n'
    + 'M = blockdiagmat([A, B])';
  assert.deepEqual(array(src, 'M'), { shape: [4, 5], data: [
    1, 2, 0, 0, 0,
    3, 4, 0, 0, 0,
    0, 0, 5, 6, 7,
    0, 0, 8, 9, 10,
  ] });
});

test('joinblocks reads a nested Value, from bindings or a literal', () => {
  assert.deepEqual(array('v = [1.0, 2.0]\nw = [3.0, 4.0]\nx = joinblocks([v, w])', 'x'),
    { shape: [4], data: [1, 2, 3, 4] });
  assert.deepEqual(array('x = joinblocks([[1.0, 2.0], [3.0, 4.0]])', 'x'),
    { shape: [4], data: [1, 2, 3, 4] });
  const A = 'A = rowstack([[1.0, 2.0], [3.0, 4.0]])\n';
  assert.deepEqual(array(`${A}x = joinblocks([[A, A], [A, A]])`, 'x').data,
    array(`${A}x = tile(A, [2, 2])`, 'x').data);
});

// §07 "The block operations satisfy": joinblocks(splitblocks(A, b)) = A and
// tile(A, n) = joinblocks(fill(A, n)).
test('the §07 block identities hold', () => {
  assert.deepEqual(array('x = joinblocks(splitblocks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2))', 'x'),
    { shape: [6], data: [1, 2, 3, 4, 5, 6] });
  const A = 'A = rowstack([[1.0, 2.0], [3.0, 4.0]])\n';
  assert.deepEqual(array(`${A}x = joinblocks(fill(A, [2, 3]))`, 'x'), array(`${A}x = tile(A, [2, 3])`, 'x'));
});

test('fill with an array fills an array of arrays', () => {
  const x = fixed('x = fill([1.0, 2.0], 3)', 'x');
  assert.deepEqual([x.shape, x.outerRank, Array.from(x.data)], [[3, 2], 1, [1, 2, 1, 2, 1, 2]]);
  assert.deepEqual(array('x = fill([1.0, 2.0], 3)\ny = x[2]', 'y'), { shape: [2], data: [1, 2] });
});

test('a matrix is not an array of arrays for joinblocks', () => {
  const { ctx } = makeMatCtx('M = rowstack([[1.0, 2.0], [3.0, 4.0]])\nx = joinblocks(M)');
  return assert.rejects(ctx.getMeasure('x'), /outer rank \(0\) must equal inner rank \(2\)/);
});

const LA = 'la = standard_module("ext-linear-algebra", "0.1")\n';

// §09 `lu(A)`: P·A = L·U. A = [[4, 3], [6, 3]] pivots row 2 first.
test('ext-linear-algebra record members read field by field', () => {
  const A = 'A = rowstack([[4.0, 3.0], [6.0, 3.0]])\n';
  assert.deepEqual(array(`${LA}${A}f = la.lu(A)\nP = f.P`, 'P').data, [0, 1, 1, 0]);
  assert.deepEqual(array(`${LA}${A}f = la.lu(A)\nL = f.L`, 'L').data, [1, 0, 2 / 3, 1]);
  assert.deepEqual(array(`${LA}${A}U = la.lu(A).U`, 'U').data, [6, 3, 0, 1]);
  // Singular values of [[2, 1], [1, 3]] are (5 ± √5) / 2.
  const S = array(`${LA}B = rowstack([[2.0, 1.0], [1.0, 3.0]])\nS = la.svd(B).S`, 'S').data;
  [(5 + Math.sqrt(5)) / 2, (5 - Math.sqrt(5)) / 2].forEach((s, i) =>
    assert.ok(Math.abs(S[i] - s) < 1e-12, `S[${i}] ${S[i]}`));
});

const NUMPY: Record<string, { A: number[][], b: number[], x: number[] }> = {
  full_tall: { A: [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], b: [1.0, 2.0, 2.5],
    x: [-0.14285714285714227, 0.49999999999999944] },
  // R of the unpivoted QR has a −4.6e-16 diagonal entry.
  dup_col: { A: [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], b: [1.0, 2.0, 2.0],
    x: [0.3928571428571429, 0.39285714285714285] },
  zero_col: { A: [[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]], b: [1.0, 1.0, 1.0],
    x: [0.6, 0.0] },
  // Column 3 = column 1 + column 2 up to rounding: an R diagonal of −1.9e-16.
  num_rankdef: { A: [[0.1, 0.2, 0.30000000000000004], [0.4, 0.5, 0.9], [0.7, 0.8, 1.5], [1.0, 1.1, 2.1]],
    b: [1.0, 2.0, 3.0, 4.5], x: [-2.6111111111111125, 3.8888888888888977, 1.2777777777777724] },
  wide: { A: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]], b: [1.0, 2.0],
    x: [-0.13294797687861354, 0.1156069364161849, 0.30057803468208144] },
  square: { A: [[2.0, 1.0], [1.0, 3.0]], b: [1.0, 2.0],
    x: [0.19999999999999987, 0.5999999999999996] },
};

const lit = (v: number[]) => '[' + v.map((x) => (Number.isInteger(x) ? x.toFixed(1) : String(x))).join(', ') + ']';

for (const [name, { A, b, x }] of Object.entries(NUMPY)) {
  test(`lstsq is numpy's minimum-norm solution: ${name}`, () => {
    const src = `${LA}A = rowstack([${A.map(lit).join(', ')}])\nx = la.lstsq(A, ${lit(b)})`;
    const got = array(src, 'x').data;
    assert.equal(got.length, x.length);
    got.forEach((g, i) => assert.ok(Math.abs(g - x[i]) < 1e-12 * Math.max(1, Math.abs(x[i])),
      `x[${i}] ${g} vs ${x[i]}`));
  });
}
