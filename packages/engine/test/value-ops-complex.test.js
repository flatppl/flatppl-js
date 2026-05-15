'use strict';

// Tests for value-ops complex-aware elementwise arithmetic (chunk 2 of
// the complex-values thread). Complex Values are planar (parallel re/im
// buffers). Covered here:
//
//   1. complex + / - : scalar∘scalar, scalar∘array broadcast,
//      array∘array; real⊕complex promotion (real im = implicit 0)
//   2. neg: negates both buffers, full Klein-4 tag preserved
//   3. mul: complex scalar broadcast = full complex multiply;
//      complex matmul/matvec explicitly deferred (guarded, not silent)
//   4. conjugation correctness: conj view negates logical im through
//      the elementwise algebra
//   5. transpose orientation carried on complex results
//   6. atom-batched addN/subN/negN over complex (indep⊕[N,k])

const { test } = require('node:test');
const assert = require('node:assert/strict');

const value = require('../value');
const vops = require('../value-ops');
const {
  complexValue, vector, scalar, transpose, conjugate, readComplex,
  isComplexValue,
} = value;

function reim(v) {
  const c = readComplex(v);
  return [Array.from(c.re), Array.from(c.im)];
}

// ---- add / sub --------------------------------------------------------

test('complex + complex (array∘array): re/im add independently', () => {
  const a = complexValue([1, 2], [3, 4]);
  const b = complexValue([10, 20], [30, 40]);
  const s = vops.add(a, b);
  assert.ok(isComplexValue(s));
  assert.deepEqual(reim(s), [[11, 22], [33, 44]]);
  const d = vops.sub(b, a);
  assert.deepEqual(reim(d), [[9, 18], [27, 36]]);
});

test('real + complex: real operand promotes (implicit zero imaginary)', () => {
  const r = vector([1, 2, 3]);
  const c = complexValue([10, 20, 30], [1, 1, 1]);
  const s = vops.add(r, c);
  assert.ok(isComplexValue(s));
  assert.deepEqual(reim(s), [[11, 22, 33], [1, 1, 1]]);
});

test('complex scalar ∘ array broadcast (sub is non-commutative)', () => {
  const z = complexValue([1], [1], []);
  const w = complexValue([10, 20], [5, 5]);
  assert.deepEqual(reim(vops.sub(z, w)), [[-9, -19], [-4, -4]]);
  assert.deepEqual(reim(vops.sub(w, z)), [[9, 19], [4, 4]]);
});

// ---- neg --------------------------------------------------------------

test('complex neg: negates both buffers, dtype preserved', () => {
  const a = complexValue([1, -2], [-3, 4]);
  const n = vops.neg(a);
  assert.ok(isComplexValue(n));
  assert.deepEqual(reim(n), [[-1, 2], [3, -4]]);
});

// ---- mul --------------------------------------------------------------

test('complex scalar broadcast mul = full complex multiply', () => {
  // (0 + 1i) * (a + bi) = -b + ai  (multiply by i: rotate 90°)
  const i = complexValue([0], [1], []);
  const w = complexValue([2, 3], [5, 7]);
  const p = vops.mul(i, w);
  assert.deepEqual(reim(p), [[-5, -7], [2, 3]]);
});

test('complex matmul/matvec is guarded, not a silent im-drop', () => {
  const A = complexValue(new Float64Array(4), new Float64Array(4), [2, 2]);
  const v = complexValue([1, 1], [1, 1]);
  assert.throws(() => vops.mul(A, v),
    /complex matrix\/vector products are not yet implemented/);
});

// ---- conjugation through the elementwise algebra ----------------------

test('conj view: logical im negated when consumed by add', () => {
  const z = complexValue([1, 2], [3, 4]);
  const zc = conjugate(z);                 // logical = (1-3i, 2-4i)
  const s = vops.add(zc, complexValue([0, 0], [0, 0]));
  assert.deepEqual(reim(s), [[1, 2], [-3, -4]]);
  // original storage untouched (lazy conj)
  assert.deepEqual(Array.from(z.im), [3, 4]);
});

// ---- transpose orientation carried -----------------------------------

test('complex elementwise result keeps the transpose (swapped) bit', () => {
  const a = transpose(complexValue([1, 2], [3, 4]));   // tag T (vector)
  const b = transpose(complexValue([5, 6], [7, 8]));
  const s = vops.add(a, b);
  assert.equal(s.t, 'T', 'swapped bit preserved on complex result');
  assert.deepEqual(reim(s), [[6, 8], [10, 12]]);
});

// ---- atom-batched addN / subN / negN ---------------------------------

test('addN: atom-indep complex vector + complex shape=[N,k]', () => {
  const N = 2, k = 2;
  const indep = complexValue([1, 2], [10, 20]);                 // [k]
  const batched = complexValue([0, 0, 100, 100],
                               [0, 0, 0, 0], [N, k]);            // [N,k]
  const s = vops.addN(batched, indep, N);
  assert.deepEqual(s.shape, [N, k]);
  assert.deepEqual(reim(s),
    [[1, 2, 101, 102], [10, 20, 10, 20]]);
});

test('subN swap-args order + negN over complex batched', () => {
  const N = 2, k = 1;
  const indep = complexValue([1], [1]);             // [k] (per-atom rank 1)
  const batched = complexValue([5, 9], [5, 9], [N, k]);
  // indep - batched  (indep is atom-indep ⇒ swapArgs path)
  const d = vops.subN(indep, batched, N);
  assert.deepEqual(reim(d), [[-4, -8], [-4, -8]]);
  const n = vops.negN(batched, N);
  assert.deepEqual(reim(n), [[-5, -9], [-5, -9]]);
});
