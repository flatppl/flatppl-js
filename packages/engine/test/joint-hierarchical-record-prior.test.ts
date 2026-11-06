'use strict';

// §06 joint merges record components into one record. In
// joint(lawof(record(lambda, theta)), logweighted(bw, Lebesgue(...))), theta
// depends on lambda, and both are observed fields of the merged record, so
// nothing is marginalised: the density is p(lambda) p(theta | lambda) e^bw(b).
//
// ORACLE: closed form with StudentT(3) density 2 / (pi sqrt(3)) (1 + x^2/3)^-2.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const logT3 = (x: number) => Math.log(2 / (Math.PI * Math.sqrt(3))) - 2 * Math.log1p(x * x / 3);
const logN = (x: number, m = 0, s = 1) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
const M = [[1, 0.5], [0, 2], [1, 1]];
const LAMBDA = [0.2, -0.1];
const THETA = [0.3, 0.0, -0.2];
const MU = M.map((row) => row[0] * LAMBDA[0] + row[1] * LAMBDA[1]);
const PRIOR = LAMBDA.reduce((s, l) => s + logT3(l), 0) + THETA.reduce((s, t, i) => s + logN(t, MU[i]), 0);

const HEAD = [
  'M = rowstack([[1.0, 0.5], [0.0, 2.0], [1.0, 1.0]])',
  'lambda ~ iid(StudentT(3), 2)',
  'theta ~ Normal.(M * lambda, 1)',
].join('\n') + '\n';

async function values(src: string, N: number): Promise<number[]> {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('lp');
  return Array.from(m.value ? m.value.data : m.samples);
}

function near(got: number[], want: number) {
  assert.ok(got.length > 0);
  got.forEach((g) => assert.ok(Math.abs(g - want) < 1e-12, `got ${g}, want ${want}`));
}

test('a hierarchical record law joined with a log-weighted Lebesgue factor', async () => {
  const src = HEAD + [
    'b = elementof(reals)',
    'y ~ Normal.(theta .+ b, 1)',
    'bw(p) = logdensityof(Normal(0, 2), p.b)',
    'Pi_prior = joint(lawof(record(lambda = lambda, theta = theta)), logweighted(bw, Lebesgue(cartprod(b = reals))))',
    'K = kernelof(record(y = y), lambda = lambda, theta = theta, b = b)',
    'L = likelihoodof(K, record(y = [0.5, -0.5, 1.0]))',
    'P = bayesupdate(L, Pi_prior)',
    'lp = logdensityof(P, record(lambda = [0.2, -0.1], theta = [0.3, 0.0, -0.2], b = 0.4))',
  ].join('\n') + '\n';
  const y = [0.5, -0.5, 1.0];
  const want = PRIOR + logN(0.4, 0, 2) + y.reduce((s, v, i) => s + logN(v, THETA[i] + 0.4), 0);
  near(await values(src, 1), want);
  near(await values(src, 3), want);
});

test('a shared matrix whose row count equals the atom count is not atom-batched', async () => {
  const src = HEAD + 'lp = logdensityof(lawof(record(lambda = lambda, theta = theta)), '
    + 'record(lambda = [0.2, -0.1], theta = [0.3, 0.0, -0.2]))\n';
  near(await values(src, 3), PRIOR);
});
