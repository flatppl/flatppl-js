'use strict';

// §04 broadcasting iterates outer array axes, and §03 keeps a vector of
// vectors distinct from a matrix. §08 declares Categorical's p as a vector, so
// `Categorical.(rows)` over a vector of probability vectors scores observation
// i with rows[i]. §07 rowstack turns a vector of vectors into a matrix.
//
// ORACLES are closed forms. With a = 0.3, rows = [[a, 1 - a], [1 - a, a]] and
// rows[[1, 2, 2]] at y = [1, 1, 2], the log density is 2 log a + log(1 - a).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const A = 0.3;
const WANT = 2 * Math.log(A) + Math.log(1 - A);
const logN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;

async function value(src: string, name: string, N = 1): Promise<number[]> {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure(name);
  return Array.from(m.value ? m.value.data : m.samples);
}

function near(got: number[], want: number, tol = 1e-12) {
  assert.ok(got.length > 0);
  got.forEach((g) => assert.ok(Math.abs(g - want) < tol, `got ${g}, want ${want}`));
}

test('Categorical.(rows) scores each observation with its own probability vector', async () => {
  near(await value(`lp = logdensityof(Categorical.([[${A}, ${1 - A}], [${1 - A}, ${A}], [${1 - A}, ${A}]]), [1, 1, 2])\n`, 'lp'),
    WANT);
});

test('Categorical.(rows[idx]) gathers rows of a fixed vector of vectors', async () => {
  const src = `a = ${A}\nrows = [[a, 1 - a], [1 - a, a]]\nlp = logdensityof(Categorical.(rows[[1, 2, 2]]), [1, 1, 2])\n`;
  near(await value(src, 'lp'), WANT);
});

test('Categorical0.(rows[idx]) uses zero-based outcomes', async () => {
  const src = `a = ${A}\nrows = [[a, 1 - a], [1 - a, a]]\nlp = logdensityof(Categorical0.(rows[[1, 2, 2]]), [0, 0, 1])\n`;
  near(await value(src, 'lp'), WANT);
});

test('Categorical.(rows[idx]) with stochastic rows scores the posterior at every atom', async () => {
  const src = [
    'a ~ Uniform(unitinterval)',
    'rows = [[a, 1 - a], [1 - a, a]]',
    'y ~ Categorical.(rows[[1, 2, 2]])',
    'L = likelihoodof(kernelof(record(y = y), a = a), record(y = [1, 1, 2]))',
    'P = bayesupdate(L, lawof(record(a = a)))',
    `lp = logdensityof(P, record(a = ${A}))`,
  ].join('\n') + '\n';
  near(await value(src, 'lp'), WANT);
  near(await value(src, 'lp', 3), WANT);
});

test('rowstack of a bound nested-iid variate is its matrix in a likelihood', async () => {
  // x ~ iid(iid(Normal(0, 1), 2), 3), y ~ Normal.(rowstack(x) * [1, 1], 1).
  const x = [[0.1, 0.2], [0, -0.3], [0.4, 0.4]];
  const y = [0.5, 0, -0.5];
  const want = x.flat().reduce((s, v) => s + logN(v), 0)
    + y.reduce((s, yi, i) => s + logN(yi - x[i][0] - x[i][1]), 0);
  const src = [
    'x ~ iid(iid(Normal(0, 1), 2), 3)',
    'y ~ Normal.(rowstack(x) * [1.0, 1.0], 1)',
    'L = likelihoodof(kernelof(record(y = y), x = x), record(y = [0.5, 0.0, -0.5]))',
    'P = bayesupdate(L, lawof(record(x = x)))',
    `lp = logdensityof(P, record(x = ${JSON.stringify(x)}))`,
  ].join('\n') + '\n';
  near(await value(src, 'lp'), want);
  near(await value(src, 'lp', 3), want);
});

test('colstack of a bound nested-iid variate is its transpose in a likelihood', async () => {
  const x = [[0.1, 0.2, -0.1], [0, -0.3, 0.2]];
  const y = [0.5, 0, -0.5];
  const want = x.flat().reduce((s, v) => s + logN(v), 0)
    + y.reduce((s, yi, i) => s + logN(yi - x[0][i] - x[1][i]), 0);
  const src = [
    'x ~ iid(iid(Normal(0, 1), 3), 2)',
    'y ~ Normal.(colstack(x) * [1.0, 1.0], 1)',
    'L = likelihoodof(kernelof(record(y = y), x = x), record(y = [0.5, 0.0, -0.5]))',
    'P = bayesupdate(L, lawof(record(x = x)))',
    `lp = logdensityof(P, record(x = ${JSON.stringify(x)}))`,
  ].join('\n') + '\n';
  near(await value(src, 'lp'), want);
});
