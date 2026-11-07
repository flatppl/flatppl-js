// A per-atom matrix plus or minus a diag-stored matrix (`diagmat(d)`).
// `_atomBroadcastBinop` took the per-atom stride from the diag operand's
// stored length (m, not m·m), so the sum read past the diagonal vector and
// left most cells unchanged. The oracle adds the dense diagonal by hand.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const valueOps = require('../value-ops.ts');
const valueLib = require('../value.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const D = [1.5, -2];
const diagAt = (k: number) => (k === 0 ? D[0] : k === 3 ? D[1] : 0);

test('addN / subN: a [N, 2, 2] batch against diagmat adds the dense diagonal per atom', () => {
  const N = 3;
  const data = Float64Array.from({ length: N * 4 }, (_, i) => 0.25 * i - 1);
  const batched = { shape: [N, 2, 2], data };
  const diag = valueLib.diagMatrix(Float64Array.from(D));
  const sum = valueOps.addN(batched, diag, N);
  const diff = valueOps.subN(diag, batched, N);
  assert.deepEqual(sum.shape, [N, 2, 2]);
  assert.deepEqual(diff.shape, [N, 2, 2]);
  for (let i = 0; i < N * 4; i++) {
    assert.equal(sum.data[i], data[i] + diagAt(i % 4), `sum cell ${i}`);
    assert.equal(diff.data[i], diagAt(i % 4) - data[i], `diff cell ${i}`);
  }
});

test('a per-atom outer product plus or minus diagmat matches the dense sum', async () => {
  for (const [expr, f] of [
    ['A + M', (a: number, k: number) => a + diagAt(k)],
    ['M - A', (a: number, k: number) => diagAt(k) - a],
  ] as Array<[string, (a: number, k: number) => number]>) {
    const src = `v ~ iid(Normal(0.0, 1.0), 2)\nA = v * transpose(v)\nM = diagmat([${D[0]}, ${D[1]}])\nY = ${expr}\n`;
    const { ctx } = makeMatCtx(src, { sampleCount: 5 });
    const a = (await ctx.getMeasure('A')).value;
    const y = (await ctx.getMeasure('Y')).value;
    assert.deepEqual(y.shape, [5, 2, 2], expr);
    for (let i = 0; i < a.data.length; i++) {
      assert.ok(Math.abs(y.data[i] - f(a.data[i], i % 4)) < 1e-12, `${expr} cell ${i}`);
    }
  }
});
