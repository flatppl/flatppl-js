'use strict';
// A TRANSPOSED MATRIX MUST BE READ AS TRANSPOSED BY EVERY LINEAR-ALGEBRA PATH.
//
// `transpose` is an O(1) view: the entries stay put and a Klein-4 tag records
// the swap. Two places read the entries without the tag.
//
//   1. The per-atom evaluator packed N per-atom results into one [N, r, c]
//      Value by copying each cell's `data`. For a transposed cell that copied
//      the UNSWAPPED entries, so `transpose(A)` of a per-atom matrix came back
//      as `A`, and `inv(transpose(A))` as `inv(A)`.
//   2. The atom-batched kernels (inv, det, logabsdet, lower_cholesky, row_gram,
//      col_gram) read `A.data` of a [N, n, n] stack without the tag.
//
// ORACLES. The fixed-value cases use inverses and solutions computed with
// numpy for a non-symmetric A (det = -11). The per-atom cases use identities
// that need no inverse routine: Aᵀ · inv(Aᵀ) = I, and (Aᵀ)(Aᵀ)ᵀ = Aᵀ A.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const valueLib = require('../value.ts');
const ops = require('../ops.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { toJS } = require('./_value-helpers.ts');

require('../ops-declarations.ts');

function fixed(source: string) {
  const parsed = processSource(source);
  const state = orchestrator.buildDerivations(parsed.bindings, {
    moduleRegistry: parsed.loweredModule.moduleRegistry,
  });
  return (name: string) => toJS(state.fixedValues.get(name));
}

function close(actual: number, expected: number, tol = 1e-12) {
  assert.ok(Math.abs(actual - expected) <= tol, `${actual} != ${expected}`);
}

const A3 = `A = rowstack([[2, 1, 3], [0, 4, 1], [5, 2, 6]])\nb = [1, 2, 3]\n`;

test('fixed: inv and linsolve of a transposed non-symmetric matrix', () => {
  const get = fixed(A3 + `
inv_t = inv(transpose(A))
solve_t = linsolve(transpose(A), b)
`);
  // Aᵀ⁻¹ = (1/11) · [[-22, -5, 20], [0, 3, -1], [11, 2, -8]]
  const want = [[-22, -5, 20], [0, 3, -1], [11, 2, -8]];
  get('inv_t').forEach((row: number[], i: number) =>
    row.forEach((x, j) => close(x, want[i][j] / 11)));
  [28, 3, -9].forEach((x, i) => close(get('solve_t')[i], x / 11));
});

const N = 64;
const SOURCE = `A = draw(iid(Normal(mu = 0.0, sigma = 1.0), [3, 3]))\n`;

async function perAtom(body: string) {
  const { ctx } = ctxFor(SOURCE + body, N);
  const a = (await ctx.getMeasure('A')).samples;
  const w = (await ctx.getMeasure('w')).samples;
  return { a, w, at: (m: any, k: number, i: number, j: number) => m[k * 9 + i * 3 + j] };
}

test('per-atom: transpose(A) is the transpose, not A', async () => {
  const { a, w, at } = await perAtom('w = transpose(A)\n');
  for (let k = 0; k < N; k++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    close(at(w, k, i, j), at(a, k, j, i));
  }
});

test('per-atom: inv(transpose(A)) satisfies Aᵀ · inv(Aᵀ) = I', async () => {
  const { a, w, at } = await perAtom('w = inv(transpose(A))\n');
  for (let k = 0; k < N; k++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let l = 0; l < 3; l++) s += at(a, k, l, i) * at(w, k, l, j);
    close(s, i === j ? 1 : 0, 1e-8);
  }
});

test('per-atom: row_gram and col_gram of transpose(A) swap roles', async () => {
  for (const [op, rowsOf] of [['row_gram', true], ['col_gram', false]] as const) {
    const { a, w, at } = await perAtom(`w = ${op}(transpose(A))\n`);
    for (let k = 0; k < N; k++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      // row_gram(Aᵀ) = Aᵀ A, col_gram(Aᵀ) = A Aᵀ.
      let s = 0;
      for (let l = 0; l < 3; l++) s += rowsOf ? at(a, k, l, i) * at(a, k, l, j) : at(a, k, i, l) * at(a, k, j, l);
      close(at(w, k, i, j), s, 1e-9);
    }
  }
});

// Two non-symmetric atoms, row-major. Inverses of the transposes by numpy.
const ATOMS = [[2, 1, 3, 0, 4, 1, 5, 2, 6], [1, 2, 0, 3, 5, 1, 0, 1, 4]];
const INV_T = [
  [-22 / 11, -5 / 11, 20 / 11, 0, 3 / 11, -1 / 11, 1, 2 / 11, -8 / 11],
  [-19 / 5, 12 / 5, -3 / 5, 8 / 5, -4 / 5, 1 / 5, -2 / 5, 1 / 5, 1 / 5],
];
const COL_GRAM = [[29, 12, 36, 12, 21, 19, 36, 19, 46], [10, 17, 3, 17, 30, 9, 3, 9, 17]];
const ROW_GRAM = [[14, 7, 30, 7, 17, 14, 30, 14, 65], [5, 13, 2, 13, 35, 9, 2, 9, 17]];

function transposedStack() {
  const data = new Float64Array(2 * 9);
  ATOMS.forEach((m, k) => data.set(m, k * 9));
  const t = valueLib.transpose({ shape: [2, 3, 3], data });
  assert.equal(t.t, 'T', 'fixture must carry the transpose tag');
  return t;
}

function expectStack(r: any, want: number[][]) {
  assert.deepEqual(r.shape, [2, 3, 3]);
  want.forEach((m, k) => m.forEach((x, e) => close(r.data[k * 9 + e], x)));
}

test('batched kernels read the transpose tag of an atom-major stack', () => {
  expectStack(ops.dispatch('inv', [transposedStack()], { atomN: 2 }), INV_T);
  // row_gram(Aᵀ) = Aᵀ A = col_gram(A), col_gram(Aᵀ) = A Aᵀ = row_gram(A).
  expectStack(ops.dispatch('row_gram', [transposedStack()], { atomN: 2 }), COL_GRAM);
  expectStack(ops.dispatch('col_gram', [transposedStack()], { atomN: 2 }), ROW_GRAM);
  // det and logabsdet are invariant under transposition.
  const det = ops.dispatch('det', [transposedStack()], { atomN: 2 });
  close(det.data[0], -11, 1e-10);
  close(det.data[1], -5, 1e-10);
});
