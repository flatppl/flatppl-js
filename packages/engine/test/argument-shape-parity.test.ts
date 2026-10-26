'use strict';

// Argument shapes the spec declares, refused statically and located:
//   * §08 multivariate parameters (MvNormal `mu` a vector and `cov` a matrix,
//     Dirichlet `alpha` and Multinomial `p` vectors, Wishart `nu` a scalar, …);
//   * a distribution's positional arguments, which bind its parameters in
//     declared order (§04 "Calling conventions") and were not checked at all;
//   * `eye(n)`, whose `n` §07 gives the domain positive integer;
//   * `boolean`, `integer` and `real`, whose domain §07 gives as "any scalar
//     numeric" — the elementwise spelling is dotted.
// A binding refused as a non-scalar operand of a scalar-only function is also
// refused by the materialiser, which would otherwise apply the elementwise
// primitive and return a number.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const PRELUDE = 'v = [1.0, 2.0]\nM = rowstack([[1.0, 0.0], [0.0, 1.0]])\n';
const errorsOf = (src: string) =>
  processSource(PRELUDE + src).diagnostics.filter((d: any) => d.severity === 'error');

function refuses(src: string, message: RegExp, col: number) {
  const errs = errorsOf(src);
  assert.equal(errs.length, 1, src + ': ' + errs.map((d: any) => d.message).join('; '));
  assert.match(errs[0].message, message, src);
  assert.deepEqual([errs[0].loc.start.line, errs[0].loc.start.col], [2, col], src);
}

test('§08 multivariate parameters carry their declared rank', () => {
  refuses('x = MvNormal(mu = 1.0, cov = M)\n', /^MvNormal: kwarg "mu" expects array of real, got real$/, 18);
  refuses('x = MvNormal(mu = M, cov = M)\n', /^MvNormal: kwarg "mu" expects array of real, got 2d array/, 18);
  refuses('x = MvNormal(mu = v, cov = v)\n', /^MvNormal: kwarg "cov" expects 2d array of real, got array of real \(length 2\)$/, 27);
  refuses('x = Dirichlet(alpha = 2.0)\n', /^Dirichlet: kwarg "alpha" expects array of real, got real$/, 22);
  refuses('x = Multinomial(n = 3, p = 0.5)\n', /^Multinomial: kwarg "p" expects array of real, got real$/, 27);
  refuses('x = Multinomial(n = v, p = v)\n', /^Multinomial: kwarg "n" expects integer, got array of real/, 20);
  refuses('x = Wishart(nu = v, scale = M)\n', /^Wishart: kwarg "nu" expects real, got array of real \(length 2\)$/, 17);
  refuses('x = InverseWishart(nu = 3.0, scale = v)\n', /^InverseWishart: kwarg "scale" expects 2d array of real/, 37);
  refuses('x = LKJ(n = v, eta = 1.0)\n', /^LKJ: kwarg "n" expects integer, got array of real/, 12);
  refuses('x = LKJCholesky(n = 2, eta = v)\n', /^LKJCholesky: kwarg "eta" expects real, got array of real/, 29);
  for (const ok of [
    'x = MvNormal(mu = v, cov = M)\n', 'x = MvNormal(mu = [0, 0], cov = rowstack([[1, 0], [0, 1]]))\n',
    'x = Dirichlet(alpha = [1, 2, 3])\n', 'x = Multinomial(n = 3, p = v)\n',
    'x = Wishart(nu = 3, scale = M)\n', 'x = InverseWishart(3.0, M)\n', 'x = LKJ(3, 2)\n',
  ]) assert.deepEqual(errorsOf(ok).map((d: any) => d.message), [], ok);
});

test('positional distribution arguments are checked as the parameters they bind', () => {
  refuses('x = Normal(0.0, v)\n', /^Normal: arg 2 \("sigma"\) expects real, got array of real \(length 2\)$/, 16);
  refuses('x = Poisson(v)\n', /^Poisson: arg 1 \("rate"\) expects real, got array of real/, 12);
  refuses('x = Binomial(2.5, 0.5)\n', /^Binomial: arg 1 \("n"\) expects integer, got real$/, 13);
  refuses('x = MvNormal(1.0, M)\n', /^MvNormal: arg 1 \("mu"\) expects array of real, got real$/, 13);
  refuses('x = Dirichlet(2.0)\n', /^Dirichlet: arg 1 \("alpha"\) expects array of real, got real$/, 14);
  refuses('x = MvNormal(v, [v, v])\n', /^`MvNormal`'s `cov` must be a matrix/, 16);
  // An inline constructor is inferred more than once but reported once.
  refuses('lp = logdensityof(Dirichlet(alpha = 2.0), [0.5])\n', /^Dirichlet: kwarg "alpha" expects array of real, got real$/, 36);
  refuses('lp = logdensityof(Dirichlet(2.0), [0.5])\n', /^Dirichlet: arg 1 \("alpha"\) expects array of real, got real$/, 28);
  // A surplus argument binds no parameter; it was dropped and the call scored.
  refuses('x = Normal(1.0, 2.0, 3.0)\n', /^Normal expects at most 2 positional argument\(s\), got 3$/, 4);
  // A lone record auto-splats over a multi-parameter distribution.
  for (const ok of ['x = Normal(0.0, 1.0)\n', 'x = Normal((mu = 0.0, sigma = 1.0))\n',
    'x = Normal(record(mu = 0.0, sigma = 1.0))\n', 'x = Binomial(integer(1e3), 0.5)\n']) {
    assert.deepEqual(errorsOf(ok).map((d: any) => d.message), [], ok);
  }
});

test('a vector step parameter written positionally is a static error', () => {
  const errs = processSource('f = (x, d) -> Normal(x, [1.0, 2.0])\ntraj = kscan(f, 0.0, [0.5, 1.0])\n')
    .diagnostics.filter((d: any) => d.severity === 'error');
  assert.ok(errs.some((d: any) => /^Normal: arg 2 \("sigma"\) expects real, got array of real \(length 2\)$/
    .test(d.message)), errs.map((d: any) => d.message).join('; '));
});

test('§07 eye takes a positive integer', () => {
  refuses('x = eye(v)\n', /^eye: arg 1 expects integer, got array of real \(length 2\)$/, 8);
  refuses('x = eye(2.0)\n', /^eye: arg 1 expects integer, got real$/, 8);
  refuses('x = eye(0)\n', /^eye: spec §07 "Array and table generation" gives `n` the domain positive integer, but it is written as 0$/, 8);
  refuses('x = eye(-1)\n', /written as -1$/, 8);
  for (const ok of ['x = eye(2)\n', 'x = eye(n = 3)\n', 'x = eye(lengthof(v))\n']) {
    assert.deepEqual(errorsOf(ok).map((d: any) => d.message), [], ok);
  }
});

test('§07 scalar constructors refuse an array; the dotted form applies per cell', () => {
  const cite = (op: string) => new RegExp('^' + op + ': arg 1 expects a scalar, got array of \\w+ \\(length \\d\\) — '
    + 'spec §07 "Scalar restrictions and constructors" gives `' + op + '` the domain "any scalar numeric"; '
    + 'the elementwise form is the dotted spelling, `' + op + '\\.\\(…\\)`$');
  refuses('x = real(v)\n', cite('real'), 9);
  refuses('x = integer([1, 2])\n', cite('integer'), 12);
  refuses('x = boolean([true])\n', cite('boolean'), 12);
  for (const ok of ['x = real.(v)\n', 'x = integer.([1.0])\n', 'x = real(2)\n', 'x = real.(round.(v))\n']) {
    assert.deepEqual(errorsOf(ok).map((d: any) => d.message), [], ok);
  }
});

test('the materialiser refuses a binding with a non-scalar operand of a scalar-only function', async () => {
  const src = 'v = [1.0, 4.0]\nx = sqrt(v)\ny = x[1] + 1.0\nr = real(v)\nz = sqrt.(v)\n';
  for (const [name, op] of [['x', 'sqrt'], ['y', 'sqrt'], ['r', 'real']]) {
    const { ctx } = ctxFor(src, 1);
    await assert.rejects(ctx.getMeasure(name), (err: any) => {
      assert.match(err.message, new RegExp("^'" + name + "' is statically invalid at \\d+:\\d+: " + op
        + ': arg 1 expects a scalar'));
      assert.equal(err.modelRefusal, true);
      return true;
    });
  }
  const { ctx } = ctxFor(src, 1);
  const z = await ctx.getMeasure('z');
  assert.deepEqual(Array.from(z.value ? z.value.data : z.samples), [1, 2]);
});

test('a self-recursive function reports its cycle instead of overflowing the stack', () => {
  for (const src of ['f(x) = f(x)\ny = f(1.0)\n', 'f(x) = g(x)\ng(x) = f(x)\ny = exp(f(1.0))\n']) {
    const errs = processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
    assert.ok(errs.some((d: any) => /^cyclic binding dependency at "[fg]"$/.test(d.message)),
      errs.map((d: any) => d.message).join('; '));
  }
});
