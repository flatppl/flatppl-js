'use strict';

// A pushforward of a vector law consumes the whole vector point. The density
// walk took the footprint from iid and wrapper shapes only, so over a Dirichlet,
// a Multinomial or a `Lebesgue` on a §03 vector set it consumed one scalar, and
// the map's inverse then threw. Expected values are closed forms.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function score(measure: string, point: string): Promise<number> {
  const { proc, ctx } = ctxFor(`M = ${measure}\n__score__ = logdensityof(M, ${point})`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const result = await ctx.getMeasure('__score__');
  return result.value.data[0];
}

const DOUBLE = 'bijection(fn(2.0 .* _), fn(_ ./ 2.0), x -> 2.0 * log(2.0))';

test('an explicit bijection over a simplex law scores with its stated volume', async () => {
  // Dirichlet(2, 3, 4) has density 3360·x1·x2²·x3³ (Γ(9) / (Γ(2)Γ(3)Γ(4))); at
  // x = (0.2, 0.3, 0.5) less the stated volume 2·log 2.
  const expected = Math.log(3360 * 0.2 * 0.3 ** 2 * 0.5 ** 3) - 2 * Math.log(2);
  const got = await score(`pushfwd(${DOUBLE}, Dirichlet(alpha = [2.0, 3.0, 4.0]))`, '[0.4, 0.6, 1.0]');
  assert.ok(Math.abs(got - expected) < 1e-12, `${got} vs ${expected}`);
});

test('an explicit bijection over Lebesgue on a vector set reads the whole point', async () => {
  // Lebesgue on zerosum(3) and on stdsimplex(3) has density 1 on its set.
  const expected = -2 * Math.log(2);
  const zerosum = await score(`pushfwd(${DOUBLE}, Lebesgue(zerosum(3)))`, '[1.0, -2.0, 1.0]');
  assert.ok(Math.abs(zerosum - expected) < 1e-12, `${zerosum}`);
  const simplex = await score(`pushfwd(${DOUBLE}, Lebesgue(stdsimplex(3)))`, '[0.4, 0.6, 1.0]');
  assert.ok(Math.abs(simplex - expected) < 1e-12, `${simplex}`);
});

test('a non-affine map of a counting vector law scores its pmf at the preimage', async () => {
  // Multinomial(5, (0.2, 0.3, 0.5)) at (1, 2, 2): 30·0.2·0.3²·0.5² = 0.135; a
  // counting reference takes no volume term.
  const got = await score('pushfwd(fn(exp.(_)), Multinomial(n = 5, p = [0.2, 0.3, 0.5]))',
    '[2.718281828459045, 7.38905609893065, 7.38905609893065]');
  assert.ok(Math.abs(got - Math.log(0.135)) < 1e-12, `${got}`);
});
