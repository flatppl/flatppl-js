'use strict';

// Spec §04 "Calling conventions": a callable that does not accept a lone positional
// argument (two or more required inputs, or keyword-only inputs) auto-splats. A sole
// tuple argument of an ordered callable splats into positional arguments, and a tuple
// with fewer elements than the required inputs, or more than all of them, is a static
// error. A callable that accepts a lone positional argument receives the tuple whole.
// Splatting is shallow. §11: each splatted element is a projection of the original.
// Every score below is a closed form.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

function errors(src: string) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}
async function score(src: string) {
  assert.deepEqual(errors(src), []);
  const { ctx } = ctxFor(src, 4);
  const m = await ctx.getMeasure('s');
  return (m.samples || m.value.data)[0];
}
const close = (got: number, want: number) =>
  assert.ok(Math.abs(got - want) < 1e-12, `${got} vs ${want}`);
const lnNormal = (x: number, mu: number, sigma: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - ((x - mu) / sigma) ** 2 / 2;

const KERNEL = `
a = elementof(reals)
b = elementof(posreals)
K = functionof(Normal(mu = a, sigma = b), a = a, b = b)
`;

test('a written tuple splats into a user function\'s positional inputs', async () => {
  close(await score('f(a, b) = a * 2.0 + b\ns = f((1.5, 0.5))\n'), 3.5);
});

test('a tuple value splats through projections of the original', async () => {
  close(await score('f(a, b) = a * 2.0 + b\nt = (1.5, 0.5)\ns = f(t)\n'), 3.5);
});

test('the elements keep their order: a non-commutative function distinguishes a swap', async () => {
  close(await score('f(a, b, c) = a - b * c\ns = f((10.0, 2.0, 3.0))\n'), 4.0);
});

test('a tuple splats into a builtin with ordered inputs', async () => {
  close(await score('s = add((1.5, 0.5))\n'), 2.0);
  close(await score('t = (2.0, 3.0)\ns = pow(t)\n'), 8.0);
});

test('a tuple binds a kernel\'s inputs in order', async () => {
  close(await score(KERNEL + 's = logdensityof(K((0.5, 2.0)), 0.3)\n'), lnNormal(0.3, 0.5, 2.0));
  close(await score(KERNEL + 't = (0.5, 2.0)\ns = logdensityof(K(t), 0.3)\n'),
    lnNormal(0.3, 0.5, 2.0));
});

test('a tuple point splats over a likelihood\'s ordered inputs', async () => {
  const src = KERNEL + 'L = likelihoodof(K, 0.3)\n';
  close(await score(src + 's = logdensityof(L, (0.5, 2.0))\n'), lnNormal(0.3, 0.5, 2.0));
  close(await score(src + 't = (0.5, 2.0)\ns = logdensityof(L, t)\n'), lnNormal(0.3, 0.5, 2.0));
});

test('splatting is shallow: an element that is a tuple binds whole', async () => {
  close(await score('f(a, b) = get(a, 2) + b\ns = f(((1.0, 2.0), 3.0))\n'), 5.0);
});

test('a callable accepting a lone positional argument receives the tuple whole', async () => {
  close(await score('h(a) = get(a, 2)\ns = h((1.0, 2.0))\n'), 2.0);
});

test('too few elements for the required inputs is a static error', () => {
  const found = errors('g(a, b, c) = a + b + c\ns = g((1.0, 2.0))\n');
  assert.equal(found.length, 1, String(found));
  assert.match(found[0], /call to "g" splats a sole positional tuple of 2 elements/);
  assert.match(found[0], /g requires 3 of its 3 inputs/);
});

test('too many elements for the inputs is a static error', () => {
  const found = errors('f(a, b) = a + b\ns = f((1.0, 2.0, 3.0))\n');
  assert.equal(found.length, 1, String(found));
  assert.match(found[0], /tuple of 3 elements/);
  assert.match(found[0], /f requires 2 of its 2 inputs/);
});

test('a builtin with the wrong tuple length is a static error', () => {
  const found = errors('s = add((1.0, 2.0, 3.0))\n');
  assert.equal(found.length, 1, String(found));
  assert.match(found[0], /^add: a sole positional tuple of 3 elements splats into positional/);
});

test('a likelihood tuple point with the wrong length is a static error', () => {
  const found = errors(KERNEL + 'L = likelihoodof(K, 0.3)\ns = logdensityof(L, (0.5, 2.0, 1.0))\n');
  assert.equal(found.length, 1, String(found));
  assert.match(found[0], /the point is a tuple of 3 elements splatted over the likelihood inputs/);
});

test('keyword-only inputs do not splat a tuple', () => {
  const found = errors(
    'x = elementof(reals)\ny = elementof(reals)\nk = functionof(x + y)\ns = k((1.0, 2.0))\n');
  assert.equal(found.length, 1, String(found));
  assert.match(found[0], /call to "k" does not accept 1 positional arguments/);
});
