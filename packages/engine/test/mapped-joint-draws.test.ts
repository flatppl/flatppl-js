'use strict';

// Draws of a map of a positional `joint` whose components are vector laws of
// different lengths. §06 makes the positional joint's variate the concatenation
// (`cat`) of its components', so each atom's vector is the components' per-atom
// values in component order. The materialiser used to stack the components as
// scalars and read a length-3 component's 3·N samples as N atoms.
//
// The density path reads the same layout: a draw scored through `logdensityof`
// equals the closed form at that draw. Dirichlet(2, 3, 4) has density
// 3360·x1·x2²·x3³ (Γ(9) / (Γ(2)Γ(3)Γ(4)) = 40320 / 12) and Dirichlet(1, 2) has
// 2·x2, on their simplices; the map 2x + 1 has volume (5 − 2)·log 2.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const N = 6;
const PRODUCT = 'joint(Dirichlet(alpha = [2.0, 3.0, 4.0]), Dirichlet(alpha = [1.0, 2.0]))';

async function draws(measure: string, width: number): Promise<number[][]> {
  const { proc, ctx } = ctxFor(`y ~ ${measure}`, N);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [N, width], measure);
  const data = Array.from(m.samples ?? m.value.data) as number[];
  return Array.from({ length: N }, (_, i) => data.slice(i * width, (i + 1) * width));
}

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

test('draws of a mapped product of simplex laws keep each factor on its own image', async () => {
  for (const row of await draws(`pushfwd(x -> 2.0 .* x .+ 1.0, ${PRODUCT})`, 5)) {
    // 2·1 + 3·1 for the first factor, 2·1 + 2·1 for the second.
    assert.ok(Math.abs(sum(row.slice(0, 3)) - 5) < 1e-12, `${row}`);
    assert.ok(Math.abs(sum(row.slice(3)) - 4) < 1e-12, `${row}`);
  }
});

test('a drawn point of the mapped product scores the closed form', async () => {
  const [y] = await draws(`pushfwd(x -> 2.0 .* x .+ 1.0, ${PRODUCT})`, 5);
  const x = y.map((v) => (v - 1) / 2);
  const expected = Math.log(3360 * x[0] * x[1] ** 2 * x[2] ** 3) + Math.log(2 * x[4]) - 3 * Math.log(2);
  const { ctx } = ctxFor(`M = pushfwd(x -> 2.0 .* x .+ 1.0, ${PRODUCT})\n`
    + `lp = logdensityof(M, [${y.join(', ')}])`, 1);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.value.data[0] - expected) < 1e-9, `${lp.value.data[0]} vs ${expected}`);
});

test('draws of a mapped joint of iid laws of unequal lengths have the summed width', async () => {
  const rows = await draws('pushfwd(x -> 2.0 .* x, joint(iid(Normal(0.0, 1.0), 3), iid(Normal(10.0, 1.0), 2)))', 5);
  // The second component's mean 20 after the map marks its two coordinates.
  const tail = rows.map((r) => (r[3] + r[4]) / 2), head = rows.map((r) => (r[0] + r[1] + r[2]) / 3);
  assert.ok(tail.every((v) => v > 10) && head.every((v) => Math.abs(v) < 10), JSON.stringify(rows));
});
