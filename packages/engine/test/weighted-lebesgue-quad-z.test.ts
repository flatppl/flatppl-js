'use strict';
// #322: normalize(weighted(fn, Lebesgue(interval(a,b)))) — the density-by-formula
// idiom (§06) with a θ-independent weight — resolves its normalizer
// Z = ∫_a^b fn dx by DETERMINISTIC 1-D quadrature (composite midpoint, reusing
// the truncate-Z path), not the seeded Monte-Carlo massFrom fallback. So the
// density is tight and independent of the sample count N.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function score(src: string, x: number, N: number): Promise<number> {
  const { ctx } = ctxFor(src + `\n__score__ = logdensityof(m, ${x})\n`, N);
  const mm = await ctx.getMeasure('__score__');
  return mm.value ? mm.value.data[0] : mm.samples[0];
}
const H = 'flatppl_compat = "0.1"\n';
const TRI = H + 'm = normalize(weighted(x -> x, Lebesgue(support = interval(0.0, 1.0))))';
const CHEB = H + 'b0 = 1.1\nb1 = 0.2\nb2 = 0.2\n'
  + 'm = normalize(weighted(x -> sum([b0, b1, b2] .* [1.0, (2.0 * x - 3.0) / 2.0, 2.0 * ((2.0 * x - 3.0) / 2.0)^2 - 1.0]), Lebesgue(support = interval(0.5, 2.5))))';

test('#322: triangular density is exact (deterministic quadrature Z)', async () => {
  for (const [x, want] of [[0.25, 0.5], [0.5, 1.0], [0.75, 1.5]] as [number, number][]) {
    const v = Math.exp(await score(TRI, x, 1));
    assert.ok(Math.abs(v - want) < 1e-6, `triangular pdf(${x})=${v} want ${want}`);
  }
});

test('#322: Z is independent of the sample count N (deterministic, not MC)', async () => {
  const a = await score(TRI, 0.75, 1);
  const b = await score(TRI, 0.75, 20000);
  assert.ok(Math.abs(a - b) < 1e-12, `pdf must not depend on N: N=1 ${a} vs N=20000 ${b}`);
});

test('#322: Chebyshev density matches the closed-form oracle tightly', async () => {
  // Z = 2.0666666666666673; pdf(x) = cheb(x)/Z (scipy, #307 oracle)
  for (const [x, want] of [[0.6, 0.5051612903225805], [1.0, 0.4354838709677418], [2.4, 0.6793548387096772]] as [number, number][]) {
    const v = Math.exp(await score(CHEB, x, 1));
    assert.ok(Math.abs(v - want) < 1e-5, `cheb pdf(${x})=${v} want ${want}`);
  }
});
