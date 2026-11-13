'use strict';
// lift inlines a user call by substituting each argument into its parameter's
// reads. An argument that is read two or more times, and that is not a leaf,
// is evaluated once: a synthetic module binding when it reads only module
// bindings, or, when it reads an input of an enclosing function, a call left
// to the IR inliner's `let`. Copying it into each read multiplied the
// expression: the amplitude models' rate_fn body grew 35×.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const first = (m: any) => (m.value ? m.value.data : m.samples)[0];
function moments(xs: ArrayLike<number>) {
  let mean = 0, variance = 0;
  for (let i = 0; i < xs.length; i++) mean += xs[i] / xs.length;
  for (let i = 0; i < xs.length; i++) variance += (xs[i] - mean) ** 2 / xs.length;
  return { mean, variance };
}

test('an indexed call in a draw parameter samples from its closed form', async () => {
  // y | a ~ Normal(2a, 1), so d = y − 2a ~ Normal(0, 1), with total mass 1.
  const N = 4000;
  const src = 'a ~ Normal(0.0, 1.0)\nf(x) = [2.0 * x, x]\ny ~ Normal(f(a)[1], 1.0)\nd = y - 2.0 * a\n'
    + 'mass = totalmass(lawof(d))\n';
  const { ctx } = makeMatCtx(src, { sampleCount: N, rootSeed: 7 });
  const { mean, variance } = moments((await ctx.getMeasure('d')).samples);
  assert.ok(Math.abs(mean) < 4 / Math.sqrt(N), `mean ${mean}`);
  assert.ok(Math.abs(variance - 1) < 4 * Math.sqrt(2 / N), `variance ${variance}`);
  assert.ok(Math.abs(first(await ctx.getMeasure('mass')) - 1) < 1e-12);
});

test('a call nested in a function body that reads its input twice', async () => {
  // g(p) = p₁ p₂ + p₁, and h(x) = g([x, 2x]) = 2x² + x. The argument [x, 2x]
  // reads h's input, so it stays a call there. ∫₀¹ h = 2/3 + 1/2 = 7/6.
  const src = 'g(p) = p[1] * p[2] + p[1]\nh(x) = g([x, 2.0 * x])\nv = h(3.0)\n'
    + 'M = normalize(weighted(h, Lebesgue(interval(0.0, 1.0))))\nlp = logdensityof(M, 0.5)\n'
    + 'a ~ Normal(0.0, 1.0)\ny ~ Normal(h(a), 1.0)\nd = y - (2.0 * a^2 + a)\n';
  const N = 4000;
  const { ctx } = makeMatCtx(src, { sampleCount: N, rootSeed: 11 });
  assert.equal(first(await ctx.getMeasure('v')), 21);
  assert.ok(Math.abs(first(await ctx.getMeasure('lp')) - Math.log(1 / (7 / 6))) < 1e-9);
  const { mean, variance } = moments((await ctx.getMeasure('d')).samples);
  assert.ok(Math.abs(mean) < 4 / Math.sqrt(N), `mean ${mean}`);
  assert.ok(Math.abs(variance - 1) < 4 * Math.sqrt(2 / N), `variance ${variance}`);
});

test('nested rotations of an indexed parameter keep the body linear in its calls', async () => {
  // The amplitude models' frame transforms: each rotation reads its vector
  // argument at every component. Six nested rotations copied into each read
  // would be 4⁶ copies of the innermost vector.
  let src = 'rz(p, a) = [p[1] * cos(a) - p[2] * sin(a), p[1] * sin(a) + p[2] * cos(a), p[3], p[4]]\n'
    + 'q = elementof(cartpow(reals, 4))\n';
  let expr = 'q';
  for (let k = 1; k <= 6; k++) expr = `rz(${expr}, ${0.1 * k})`;
  src += `turn = functionof(${expr}[1], q)\nv = turn([1.0, 2.0, 3.0, 4.0])\n`;
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  let size = 0;
  const seen = new Set<any>();
  (function go(n: any) {
    if (!n || typeof n !== 'object') return;
    size++;
    if (seen.has(n)) return;
    seen.add(n);
    for (const k in n) if (k !== 'loc') go(n[k]);
  })(ctx.bindings.get('turn').ir.body);
  assert.ok(size < 2000, `turn body has ${size} positions`);
  // Six turns by 0.1, …, 0.6 rotate (1, 2) by 2.1 rad.
  const want = Math.cos(2.1) * 1 - Math.sin(2.1) * 2;
  assert.ok(Math.abs(first(await ctx.getMeasure('v')) - want) < 1e-12);
});
