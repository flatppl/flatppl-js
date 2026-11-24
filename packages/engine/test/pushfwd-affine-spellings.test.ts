'use strict';

// Every spelling of a uniform affine map reaches the one diagonal-affine route.
//
// §03 and §06 give `Lebesgue` on a `stdsimplex(n)` or `zerosum(n)` support as
// the measure of its n − 1 free coordinates, so `x -> a .* x .+ b` with scalar
// a has volume (n − 1)·log|a|. The same map may be written `fn(a .* _ .+ b)`,
// `x ./ (1 / a)`, undotted `a * x` (§07 `mul` admits scalar-vector), `neg` or
// `fn(neg(_))` (§07 `neg` admits arrays). A product of such laws lacks the sum
// of its factors' free coordinates.
//
// Expected values come from scipy 1.18.1, mpmath or closed forms, never from an
// engine. At x = [0.2, 0.3, 0.5], log Dirichlet(x; 2, 3, 4) = 2.0228711901914433.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const DIRICHLET = 'Dirichlet(alpha = [2.0, 3.0, 4.0])';
const BARE = 2.0228711901914433;

async function score(measure: string, point: string, prefix = ''): Promise<number> {
  const { proc, ctx } = ctxFor(`${prefix}\nM = ${measure}\n__score__ = logdensityof(M, ${point})`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const result = await ctx.getMeasure('__score__');
  return result.value.data[0];
}

async function near(measure: string, point: string, expected: number, prefix = '') {
  const got = await score(measure, point, prefix);
  assert.ok(Math.abs(got - expected) < 1e-12, `${measure}: ${got} vs ${expected}`);
}

test('the fn spelling of a uniform affine map counts the free coordinates', async () => {
  // log Dirichlet((y − 1) / 2) − 2 log 2; mpmath 0.63657682907155101.
  await near(`pushfwd(fn(2.0 .* _ .+ 1.0), ${DIRICHLET})`, '[1.4, 1.6, 2.0]', 0.63657682907155101);
  await near(`pushfwd(fn(2.0 .* _), ${DIRICHLET})`, '[0.4, 0.6, 1.0]', BARE - 2 * Math.log(2));
  await near('pushfwd(fn(2.0 .* _ .+ 3.0), Lebesgue(zerosum(3)))', '[3.5, 2.5, 3.0]', -2 * Math.log(2));
});

test('a division by a static scalar is a multiplication by its reciprocal', async () => {
  const halved = '[0.1, 0.15, 0.25]';
  await near(`pushfwd(x -> x ./ 2.0, ${DIRICHLET})`, halved, BARE + 2 * Math.log(2));
  await near(`pushfwd(fn(_ ./ 2.0), ${DIRICHLET})`, halved, BARE + 2 * Math.log(2));
  await near(`pushfwd(x -> x / 2.0, ${DIRICHLET})`, halved, BARE + 2 * Math.log(2));
  await near(`pushfwd(fn(_ ./ -2.0), ${DIRICHLET})`, '[-0.1, -0.15, -0.25]', BARE + 2 * Math.log(2));
  await near(`pushfwd(x -> x ./ s, ${DIRICHLET})`, halved, BARE + 2 * Math.log(2), 's = 2.0');
});

test('an undotted scalar scale is the uniform map', async () => {
  await near(`pushfwd(x -> 2.0 * x, ${DIRICHLET})`, '[0.4, 0.6, 1.0]', BARE - 2 * Math.log(2));
  // Σ log φ(y / 2) − 3 log 2 at y = [1, 2, 3].
  await near('pushfwd(x -> 2.0 * x, iid(Normal(0.0, 1.0), 3))', '[1.0, 2.0, 3.0]', -6.586257141293855);
});

test('every spelling of neg is volume preserving', async () => {
  for (const map of ['neg', 'fn(neg(_))', 'x -> .-x', 'x -> -x']) {
    await near(`pushfwd(${map}, ${DIRICHLET})`, '[-0.2, -0.3, -0.5]', BARE);
  }
});

test('nested uniform maps compose their volumes', async () => {
  // log Dirichlet − 2 log 2 − 2 log 3.
  const expected = BARE - 2 * Math.log(2) - 2 * Math.log(3);
  await near(`pushfwd(fn(3.0 .* _), pushfwd(fn(2.0 .* _), ${DIRICHLET}))`, '[1.2, 1.8, 3.0]', expected);
  await near(`pushfwd(x -> 3.0 .* x, pushfwd(fn(_ ./ 0.5), ${DIRICHLET}))`, '[1.2, 1.8, 3.0]', expected);
});

test('a uniform map of a product of simplex laws counts their summed free coordinates', async () => {
  // log Dir(0.2, 0.3, 0.5; 2, 3, 4) + log Dir(0.4, 0.6; 1, 2) − (5 − 2) log 2 (scipy).
  await near('pushfwd(x -> 2.0 .* x .+ 1.0, joint(Dirichlet(alpha = [2.0, 3.0, 4.0]), Dirichlet(alpha = [1.0, 2.0])))',
    '[1.4, 1.6, 2.0, 1.8, 2.2]', 0.12575120530556205);
});

test('a division by zero or by a computed scalar is not read as a uniform map', async () => {
  const reason = /pushfwd of a law on a lower-dimensional support/;
  for (const map of ['x -> x ./ 0.0', 'x -> x ./ (1.0 + 1.0)']) {
    const { ctx } = ctxFor(`M = pushfwd(${map}, ${DIRICHLET})\n__score__ = logdensityof(M, [0.1, 0.15, 0.25])`, 1);
    await assert.rejects(() => ctx.getMeasure('__score__'), reason, map);
  }
});

// Draws of a mapped simplex law: a [N, n] batch whose rows lie on the image of
// the simplex, so x -> a .* x .+ b sends each row sum 1 to a + n·b.
async function draws(measure: string, n: number, N = 4): Promise<number[][]> {
  const { proc, ctx } = ctxFor(`y ~ ${measure}`, N);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [N, n], measure);
  const data = Array.from(m.samples ?? m.value.data) as number[];
  return Array.from({ length: N }, (_, i) => data.slice(i * n, (i + 1) * n));
}

test('draws of every newly read spelling over a simplex law lie on its image', async () => {
  for (const [measure, sum] of [
    [`pushfwd(fn(2.0 .* _ .+ 1.0), ${DIRICHLET})`, 5],
    [`pushfwd(fn(_ ./ 2.0), ${DIRICHLET})`, 0.5],
    [`pushfwd(x -> x / 2.0, ${DIRICHLET})`, 0.5],
    [`pushfwd(x -> 2.0 * x, ${DIRICHLET})`, 2],
    [`pushfwd(neg, ${DIRICHLET})`, -1],
    [`pushfwd(fn(neg(_)), ${DIRICHLET})`, -1],
  ] as Array<[string, number]>) {
    for (const row of await draws(measure, 3)) {
      const s = row.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(s - sum) < 1e-12, `${measure}: row sum ${s}, want ${sum}`);
    }
  }
});

test('draws of the newly read spellings over an iid law keep its shape', async () => {
  for (const map of ['fn(2.0 .* _ .+ 1.0)', 'fn(_ ./ 2.0)', 'neg', 'fn(neg(_))']) {
    await draws(`pushfwd(${map}, iid(Normal(0.0, 1.0), 3))`, 3);
  }
});

test('a map whose body holds a nested reification keeps its own hole', async () => {
  // fn(_ + 0.0) owns its hole; the outer fn has one, so this is 2 .* x.
  await near('pushfwd(fn(2.0 .* broadcast(fn(_ + 0.0), _)), iid(Normal(0.0, 1.0), 3))',
    '[1.0, 2.0, 3.0]', -6.586257141293855);
});
