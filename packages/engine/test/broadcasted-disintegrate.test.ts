'use strict';

// §04 `broadcasted(f)(args...)` is `broadcast(f, args...)`. A curried
// broadcast feeding a model that `disintegrate` splits must classify and
// score like the direct spelling. An older engine gave the downstream
// bindings no derivation and no diagnostic.
//
// ORACLE: closed form. a ~ Normal(0, 1), b ~ Normal.(mu, 1) observed at
// [0.5, 1.5]; the posterior at a = 0.3 is a sum of Normal log densities.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const logN = (x: number, m = 0) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - m) ** 2;

async function score(model: string): Promise<number> {
  const src = 'a ~ Normal(0, 1)\n' + model
    + 'jm = lawof(record(a = a, b = b))\n'
    + 'K, prior = disintegrate(["b"], jm)\n'
    + 'L = likelihoodof(K, record(b = [0.5, 1.5]))\n'
    + 'post = bayesupdate(L, prior)\n'
    + 'lp = logdensityof(post, record(a = 0.3))\n';
  assert.deepEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error'), []);
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('lp');
  return (m.value ? m.value.data : m.samples)[0];
}

function near(got: number, want: number) {
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
}

const ADD = logN(0.3) + logN(0.5, 1.3) + logN(1.5, 2.3);

test('a curried broadcasted(add) feeding disintegrate scores like broadcast(add, ...)', async () => {
  near(await score('f = broadcasted(add)\nmu = f(fill(a, 2), [1.0, 2.0])\nb ~ Normal.(mu, 1)\n'), ADD);
  near(await score('mu = broadcasted(add)(fill(a, 2), [1.0, 2.0])\nb ~ Normal.(mu, 1)\n'), ADD);
  near(await score('mu = broadcast(add, fill(a, 2), [1.0, 2.0])\nb ~ Normal.(mu, 1)\n'), ADD);
});

test('a curried broadcasted user function and kernel feeding disintegrate', async () => {
  near(await score('h(x, y) = x + 2 * y\nf = broadcasted(h)\nmu = f(fill(a, 2), [1.0, 2.0])\nb ~ Normal.(mu, 1)\n'),
    logN(0.3) + logN(0.5, 2.3) + logN(1.5, 4.3));
  near(await score('g = broadcasted(Normal)\nb ~ g(fill(a, 2), 1)\n'),
    logN(0.3) + logN(0.5, 0.3) + logN(1.5, 0.3));
});
