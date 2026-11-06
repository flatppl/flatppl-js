'use strict';

// §06 joint: "all records with distinct field names (yielding a merged
// record)". A positional joint of record-valued priors is a record prior, so
// its fields bind a kernel's inputs by name, like the keyword form. A joint of
// scalars yields a vector and still cannot feed a kernel with several inputs.
//
// ORACLE: the closed-form sum of Normal log densities. Lebesgue factors add 0.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const logN = (x: number, mu = 0) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - mu) ** 2;
const [M, S, A, B] = [0.1, 0.2, 0.3, 0.4];
const LIK = logN(0.5, M + S * A + B);

const model = (prior: string) => [
  'a ~ Normal(0, 1)',
  'b ~ Normal(0, 1)',
  'm = elementof(reals)',
  's = elementof(reals)',
  'y ~ Normal(m + s * a + b, 1)',
  `Pi_prior = ${prior}`,
  'K = kernelof(record(y = y), m = m, s = s, a = a, b = b)',
  'L = likelihoodof(K, record(y = 0.5))',
  'Pi_post = bayesupdate(L, Pi_prior)',
].join('\n') + '\n';

async function score(src: string, point: string, N = 1): Promise<number[]> {
  const full = src + `lp = logdensityof(Pi_post, ${point})\n`;
  const errors = processSource(full).diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors.map((d: any) => d.message), []);
  const { ctx } = ctxFor(full, N);
  const v = await ctx.getMeasure('lp');
  return Array.from(v.value ? v.value.data : v.samples);
}

function near(got: number[], want: number) {
  assert.ok(got.length > 0);
  got.forEach((g) => assert.ok(Math.abs(g - want) < 1e-12, `got ${g}, want ${want}`));
}

test('a multi-field Lebesgue factor joined with a lawof record feeds the kernel by name', async () => {
  const src = model('joint(Lebesgue(cartprod(m = reals, s = reals)), lawof(record(a = a, b = b)))');
  const want = logN(A) + logN(B) + LIK;
  near(await score(src, `record(m = ${M}, s = ${S}, a = ${A}, b = ${B})`), want);
  near(await score(src, `record(m = ${M}, s = ${S}, a = ${A}, b = ${B})`, 3), want);
});

test('the merged record keeps the factor order', async () => {
  const src = model('joint(lawof(record(a = a, b = b)), Lebesgue(cartprod(m = reals, s = reals)))');
  near(await score(src, `record(a = ${A}, b = ${B}, m = ${M}, s = ${S})`), logN(A) + logN(B) + LIK);
});

test('a keyword joint factor merges the same way', async () => {
  const src = model('joint(lawof(record(a = a, b = b)), joint(m = Normal(0, 1), s = Normal(0, 1)))');
  near(await score(src, `record(a = ${A}, b = ${B}, m = ${M}, s = ${S})`),
    logN(A) + logN(B) + logN(M) + logN(S) + LIK);
});

test('a joint of scalars is a vector and cannot feed a kernel with several inputs', () => {
  const src = [
    'a ~ Normal(0, 1)',
    'b ~ Normal(0, 1)',
    'y ~ Normal(a + b, 1)',
    'Pi_prior = joint(Normal(0, 1), Normal(0, 1))',
    'K = kernelof(record(y = y), a = a, b = b)',
    'L = likelihoodof(K, record(y = 0.5))',
    'Pi_post = bayesupdate(L, Pi_prior)',
  ].join('\n') + '\n';
  const errors = processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
  assert.ok(errors.some((d: any) => d.message.includes("the prior's variate is array of real")),
    JSON.stringify(errors.map((d: any) => d.message)));
});
