'use strict';

// A function whose result is a callable, applied. FlatPPL has no closures
// (§04 "Lambda notation"), so `f = a -> (a -> a + 1.0)` returns the inner
// lambda whatever its argument: `f(2.0)(3.0)` is 4. Two defects stopped this:
// - Inlining `f(2.0)` substituted the outer argument inside the inner lambda,
//   which rebinds it (§04 "Placeholders and holes": a placeholder's scope is
//   the nearest enclosing reification), giving `functionof(2 + 1, a = 2)`.
// - A chained call `f(x)(y)` was never reduced, so it got no derivation.
// Expected values are closed forms.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const lnN = (x: number, m: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - m) ** 2;

async function value(src: string, name: string): Promise<number> {
  const { ctx, proc } = ctxFor(src, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  const m = await ctx.getMeasure(name);
  return m.samples ? m.samples[0] : m.value.data[0];
}

test('a returned lambda ignores the outer argument it shadows', async () => {
  assert.equal(await value('f = a -> (a -> a + 1.0)\nz = f(2.0)(3.0)\n', 'z'), 4);
  assert.equal(await value('f = a -> (a -> a + 1.0)\nh = f(2.0)\nz = h(3.0)\n', 'z'), 4);
});

test('a returned functionof with a renamed input applies by that name', async () => {
  // f(x) = functionof(2.0 * x, y = x) is functionof(functionof(2 * _x_, y = _x_), x = _x_).
  assert.equal(await value('f(x) = functionof(2.0 * x, y = x)\nz = f(1.0)(y = 4.0)\n', 'z'), 8);
});

const KERNEL = 'g(mu) = functionof(Normal(mu, 1.0), mu = mu)\nK = g(0.0)\n';

test('a returned kernel scores as a measure and as a likelihood', async () => {
  const atMu = (mu: number) => KERNEL + `M = K(mu = ${mu})\nlp = logdensityof(M, 0.5)\n`;
  assert.ok(Math.abs(await value(atMu(2.0), 'lp') - lnN(0.5, 2.0)) < 1e-12);
  const lik = KERNEL + 'lp = logdensityof(likelihoodof(K, 0.5), record(mu = 2.0))\n';
  assert.ok(Math.abs(await value(lik, 'lp') - lnN(0.5, 2.0)) < 1e-12);
  // d/dmu log N(0.5; mu, 1) = 0.5 - mu, against a central difference.
  const h = 1e-5;
  const slope = (await value(atMu(2.0 + h), 'lp') - await value(atMu(2.0 - h), 'lp')) / (2 * h);
  assert.ok(Math.abs(slope - (0.5 - 2.0)) < 1e-7, `${slope}`);
});

test('a returned kernel samples its measure', async () => {
  const N = 20000;
  const { ctx } = ctxFor(KERNEL + 'M = K(mu = 2.0)\n', N);
  const s = (await ctx.getMeasure('M')).samples;
  let m = 0, v = 0;
  for (const x of s) m += x;
  m /= s.length;
  for (const x of s) v += (x - m) ** 2;
  v /= s.length;
  assert.ok(Math.abs(m - 2) < 0.03, `mean ${m}`);
  assert.ok(Math.abs(v - 1) < 0.05, `variance ${v}`);
});
