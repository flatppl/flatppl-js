'use strict';

// §06 "Known-bijection registry": a domain-restricted forward requires the base
// measure's support to lie within its domain, and density is refused where it
// does not. The guard refused `log` over `pushfwd(exp, Normal(0, 1))` too: the
// base had no support rule, and the type's extent, reals, stood in for it. An
// upper bound cannot prove that a support leaves a domain. Only an exact
// support (a distribution's, carried through iid, truncate, normalize and
// weighted, or an affine map of the real line) refuses now. A fixed-image op
// (exp, sqrt, abs, invlogit, ...) gives an upper bound that can admit.
// Expected densities are closed-form sums of log N(x; 0, 1).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const lnN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;
const errors = (src: string) => processSource(src).diagnostics
  .filter((d: any) => d.severity === 'error').map((d: any) => d.message);
const GUARD = /pushfwd: forward function uses '(log|logit)'/;

async function score(src: string) {
  assert.deepEqual(errors(src), []);
  const { ctx } = ctxFor(src, 1);
  return (await ctx.getMeasure('lp')).samples[0];
}

function close(got: number, want: number) {
  assert.ok(Math.abs(got - want) < 1e-12, `${got} != ${want}`);
}

test('log of the image of exp round-trips a Normal', async () => {
  close(await score('X = pushfwd(x -> exp(x), Normal(0.0, 1.0))\nY = pushfwd(x -> log(x), X)\n'
    + 'lp = logdensityof(Y, 0.3)\n'), lnN(0.3));
  close(await score('X = pushfwd(exp, Normal(0.0, 1.0))\nY = pushfwd(log, X)\n'
    + 'lp = logdensityof(Y, 0.3)\n'), lnN(0.3));
});

test('dotted log of the dotted image of exp round-trips an iid Normal', async () => {
  const pre = 'Z = iid(Normal(0.0, 1.0), 3)\nX = pushfwd(x -> exp.(x), Z)\nY = pushfwd(x -> log.(x), X)\n';
  close(await score(pre + 'lp = logdensityof(Y, [0.3, -1.0, 2.0])\n'), lnN(0.3) + lnN(-1) + lnN(2));
  const { ctx } = ctxFor(pre, 16);
  const Z = await ctx.getMeasure('Z');
  const Y = await ctx.getMeasure('Y');
  for (let i = 0; i < Z.value.data.length; i++) {
    assert.ok(Math.abs(Y.value.data[i] - Z.value.data[i]) < 1e-12, `cell ${i}`);
  }
});

test('nested log of the nested image of exp is admitted over an iid of vectors', () => {
  assert.deepEqual(errors('Z = iid(iid(Normal(0.0, 1.0), 2), 3)\n'
    + 'X = pushfwd(x -> (y -> exp.(y)).(x), Z)\nY = pushfwd(x -> (y -> log.(y)).(x), X)\n'), []);
});

test('an exact support outside the domain still refuses', () => {
  for (const src of [
    'Y = pushfwd(x -> log(x), Normal(0.0, 1.0))\n',
    'Y = pushfwd(x -> log.(x), iid(Normal(0.0, 1.0), 3))\n',
    'Y = pushfwd(x -> log(x), truncate(Normal(0.0, 1.0), interval(-1.0, 1.0)))\n',
    'X = pushfwd(x -> 2.0 * x + 1.0, Normal(0.0, 1.0))\nY = pushfwd(log, X)\n',
    'X = pushfwd(x -> 2.0 .* x .+ 1.0, iid(Normal(0.0, 1.0), 3))\nY = pushfwd(x -> log.(x), X)\n',
  ]) {
    const found = errors(src);
    assert.ok(found.length === 1 && GUARD.test(found[0]), `${src}: ${JSON.stringify(found)}`);
  }
});

test('a support inside the domain, or only bounded from above, is admitted', () => {
  for (const src of [
    'Y = pushfwd(x -> log(x), truncate(Normal(0.0, 1.0), interval(0.5, 2.0)))\n',
    // The support is [0, 5], the intersection with Exponential's.
    'Y = pushfwd(x -> log(x), truncate(Exponential(1.0), interval(-1.0, 5.0)))\n',
    // exp of (-5, -1) lies in (0, 1), inside logit's domain, though posreals is not.
    'X = pushfwd(exp, Uniform(interval(-5.0, -1.0)))\nY = pushfwd(logit, X)\n',
  ]) {
    assert.deepEqual(errors(src), [], src);
  }
});
