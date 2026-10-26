'use strict';

// §06 "Known-bijection registry": `pow` with a finite nonzero literal exponent,
// "of which `sqrt` = `pow(_, 1/2)` is a case". A quotient of numeric literals,
// also under `neg`, is therefore a literal exponent. It folds to its value as a
// real (§03), so it is never an odd integer: the non-negative support rule holds.
//
// ORACLE: closed forms from the change of variables, evaluated in 30-digit
// mpmath, not by the engine. X ~ Exponential(1):
//   Y = X^(1/2):  f(y) = 2y exp(-y^2)        log f(0.5) = -0.25, log f(2) = -2.6137056388801094
//   Y = X^(1/3):  f(y) = 3y^2 exp(-y^3)      log f(0.5) = -0.41268207245178093, log f(1.2) = -0.26474459774398106
//   Y = X^(-1/2): f(y) = 2y^-3 exp(-y^-2)   log f(2) = -1.6362943611198906, log f(0.8) = -0.19992216549742542
// mpmath quad gives mass 1 for all three densities.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { processSource } = require('../index.ts');

async function scores(measure: string, pts: number[]): Promise<number[]> {
  const src = `M = ${measure}\nlp = broadcast(logdensityof, M, [${pts.join(', ')}])\n`;
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('lp');
  return Array.from((m.value?.data ?? m.samples) as ArrayLike<number>);
}

const close = (got: number, want: number, what: string) =>
  assert.ok(Math.abs(got - want) < 1e-12 * Math.max(1, Math.abs(want)), `${what}: got ${got}, want ${want}`);

const E = 'Exponential(1)';
const CASES: [string, number[], number[]][] = [
  [`pushfwd(fn(_^(1/2)), ${E})`, [0.5, 2, -1], [-0.25, -2.6137056388801094, -Infinity]],
  [`pushfwd(fn(pow(_, 1/3)), ${E})`, [0.5, 1.2, -1], [-0.41268207245178093, -0.26474459774398106, -Infinity]],
  [`pushfwd(fn(_^(-1/2)), ${E})`, [2, 0.8, -1], [-1.6362943611198906, -0.19992216549742542, -Infinity]],
  [`pushfwd(fn(_^(-(1/2))), ${E})`, [2], [-1.6362943611198906]],
  [`pushfwd(fn(_^(1.0/(-2))), ${E})`, [2], [-1.6362943611198906]],
];

for (const [measure, ys, want] of CASES) {
  test(`${measure} scores the closed form`, async () => {
    const got = await scores(measure, ys);
    got.forEach((g, i) => {
      if (want[i] === -Infinity) assert.equal(g, -Infinity, `${measure} at ${ys[i]}`);
      else close(g, want[i], `${measure} at ${ys[i]}`);
    });
  });
}

// The quotient folds to the same double as the decimal, so it must take the same
// inverse and give the same bits.
test('a quotient exponent scores bit-identically to its decimal spelling', async () => {
  const ys = [0.1, 0.5, 2, 7.5, -1];
  for (const [q, d] of [['(1/2)', '0.5'], ['(1/3)', '0.3333333333333333'], ['(-1/2)', '-0.5'], ['(1/4)', '0.25']]) {
    assert.deepEqual(await scores(`pushfwd(fn(_^${q}), ${E})`, ys),
      await scores(`pushfwd(fn(_^${d}), ${E})`, ys), `_^${q} vs _^${d}`);
  }
});

// Substituting y = t^k places the nodes. The density integrated is the engine's.
for (const [q, k, lo, hi] of [['(1/2)', 0.5, 0, 8], ['(1/3)', 1 / 3, 0, 4]] as const) {
  test(`_^${q} of an Exponential(1) keeps mass 1 on y >= 0`, async () => {
    const h = 0.0005;
    const ys: number[] = [];
    for (let i = 0; i < (hi - lo) / h; i++) ys.push(lo + h / 2 + i * h);
    const ld = await scores(`pushfwd(fn(_^${q}), ${E})`, ys);
    const mass = ld.reduce((acc, l) => acc + Math.exp(l) * h, 0);
    assert.ok(Math.abs(mass - 1) < 1e-6, `mass: ${mass} (k = ${k})`);
  });
}

// y = 1/t maps (0, inf) to (0, inf) and removes the heavy y^-3 tail.
test('_^(-1/2) of an Exponential(1) keeps mass 1 on y > 0', async () => {
  const h = 0.0005;
  const ts: number[] = [];
  for (let i = 0; i < 8000; i++) ts.push(h / 2 + i * h);
  const ld = await scores(`pushfwd(fn(_^(-1/2)), ${E})`, ts.map((t) => 1 / t));
  const mass = ld.reduce((acc, l, i) => acc + Math.exp(l) / (ts[i] * ts[i]) * h, 0);
  assert.ok(Math.abs(mass - 1) < 1e-6, `mass: ${mass}`);
});

test('a quotient exponent is never an odd integer, so a real support is refused', () => {
  const errs = processSource('M = pushfwd(fn(_^(1/3)), Normal(0, 1))\nlp = logdensityof(M, 0.5)')
    .diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /requires the base measure's support to be non-negative/);
});

// Only literals and quotients of literals fold; a named exponent or another op
// still needs a bijection annotation, and a zero or infinite quotient is no bijection.
for (const exponent of ['k', '(1/0)', '(0/2)', '(k/2)', '(1 + 1)']) {
  test(`_^${exponent} is not a literal exponent`, async () => {
    const { ctx } = ctxFor(`k = 0.5\nM = pushfwd(fn(_^${exponent}), ${E})\nlp = logdensityof(M, 0.5)\n`, 1);
    await assert.rejects(ctx.getMeasure('lp'), /requires a bijection annotation/);
  });
}
