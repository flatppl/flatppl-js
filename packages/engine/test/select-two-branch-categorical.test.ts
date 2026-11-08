'use strict';
// A TWO-COMPONENT CATEGORICAL MIXTURE `xs[i]` MUST SAMPLE EACH BRANCH IN
// PROPORTION TO ITS WEIGHT.
//
// matSelect's gather picked the branch with `sel ? 0 : 1` whenever there were
// two branches. That is the Bernoulli / ifelse rule. A Categorical selector
// takes the values {1, 2}, so `Categorical` sampled branch 0 for every atom,
// while the density, which indexes by `selector - base`, stayed right.
//
// `xs[i]` is the 1-based `get`. A Categorical0 selector takes the values
// {0, 1}, so `xs[i]` with it refuses rather than scoring `get0` semantics.
//
// ORACLE: branch k is chosen with probability p_k, so the fraction of atoms
// near each branch's mean is p_k. The branches sit 10 apart with sigma 0.1, so
// `x > 5` identifies branch 1 without ambiguity. The tolerance is six binomial
// standard deviations at N = 40000, about 0.0137 at p = 0.3.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const N = 40000;
const BRANCHES = `
a = draw(Normal(mu = 0.0, sigma = 0.1))
b = draw(Normal(mu = 10.0, sigma = 0.1))
`;

async function fractionOnBranchB(src: string): Promise<number> {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('x');
  let hits = 0;
  for (let i = 0; i < N; i++) if (m.samples[i] > 5) hits++;
  return hits / N;
}

function assertFraction(got: number, p: number, label: string) {
  const tol = 6 * Math.sqrt(p * (1 - p) / N);
  assert.ok(Math.abs(got - p) < tol, `${label}: fraction on branch b ${got}, expected ${p} ± ${tol}`);
}

test('two-branch Categorical xs[i] samples each branch with its weight', async () => {
  const got = await fractionOnBranchB(BRANCHES + `
i = draw(Categorical(p = [0.3, 0.7]))
xs = [a, b]
x = xs[i]
`);
  assertFraction(got, 0.7, 'Categorical');
});

test('a Categorical0 selector refuses the 1-based xs[i]', async () => {
  await assert.rejects(() => fractionOnBranchB(BRANCHES + `
i = draw(Categorical0(p = [0.3, 0.7]))
xs = [a, b]
x = xs[i]
`), /a Categorical0 selector is 0-based; index the tuple with get0/);
});

test('two-branch Categorical xs[i] with a computed weight vector', async () => {
  const got = await fractionOnBranchB(BRANCHES + `
q = 0.9
i = draw(Categorical(p = [q, 1 - q]))
xs = [a, b]
x = xs[i]
`);
  assertFraction(got, 0.1, 'Categorical, computed p');
});

test('ifelse on a Bernoulli keeps its truthy-first rule', async () => {
  // Control: c = 1 (probability 0.25) selects the first branch, so branch b is
  // chosen with probability 0.75.
  const got = await fractionOnBranchB(BRANCHES + `
c = draw(Bernoulli(p = 0.25))
x = ifelse(c, a, b)
`);
  assertFraction(got, 0.75, 'ifelse');
});

test('a constant-weight two-branch superpose keeps its weights', async () => {
  // Control: no external selector, so matSelect synthesises a Bernoulli one.
  const got = await fractionOnBranchB(`
m = normalize(superpose(weighted(0.3, Normal(mu = 0.0, sigma = 0.1)), weighted(0.7, Normal(mu = 10.0, sigma = 0.1))))
x = draw(m)
`);
  assertFraction(got, 0.7, 'superpose');
});

test('three-branch Categorical xs[i] keeps its weights', async () => {
  const { ctx } = ctxFor(`
a = draw(Normal(mu = 0.0, sigma = 0.1))
b = draw(Normal(mu = 10.0, sigma = 0.1))
c = draw(Normal(mu = 20.0, sigma = 0.1))
i = draw(Categorical(p = [0.2, 0.3, 0.5]))
xs = [a, b, c]
x = xs[i]
`, N);
  const m = await ctx.getMeasure('x');
  const count = [0, 0, 0];
  for (let k = 0; k < N; k++) count[Math.round(m.samples[k] / 10)]++;
  [0.2, 0.3, 0.5].forEach((p, k) => assertFraction(count[k] / N, p, `K=3 branch ${k}`));
});

test('a computed weight vector pairs branch k with its own weight in the density', async () => {
  // Oracle: log sum_k w_k N(4; mu_k, 1) with w = (0.2, 0.3, 0.5) and
  // mu = (0, 5, 10). The weights differ, so an off-by-one weight index
  // shifts the mixture or reads past the vector.
  const { makeMatCtx } = require('./_materialise-helpers.ts');
  const { ctx } = makeMatCtx(`
w = softmax([log(0.2), log(0.3), log(0.5)])
i ~ Categorical(w)
a ~ Normal(0, 1)
b ~ Normal(5, 1)
c ~ Normal(10, 1)
xs = [a, b, c]
x = xs[i]
lp = logdensityof(lawof(x), 4.0)
`, { sampleCount: 1 });
  const m = await ctx.getMeasure('lp');
  const got = (m.value ? Array.from(m.value.data) : m.samples)[0] as number;
  const phi = (x: number, mu: number) => Math.exp(-0.5 * (x - mu) ** 2) / Math.sqrt(2 * Math.PI);
  const want = Math.log(0.2 * phi(4, 0) + 0.3 * phi(4, 5) + 0.5 * phi(4, 10));
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
});

test('a Categorical0 selector refuses the 1-based xs[i] in the density', async () => {
  const { makeMatCtx } = require('./_materialise-helpers.ts');
  await assert.rejects(async () => {
    const { ctx } = makeMatCtx(`
w = softmax([log(0.2), log(0.8)])
i ~ Categorical0(w)
a ~ Normal(0, 1)
b ~ Normal(5, 1)
xs = [a, b]
x = xs[i]
lp = logdensityof(lawof(x), 1.0)
`, { sampleCount: 1 });
    await ctx.getMeasure('lp');
  }, /a Categorical0 selector is 0-based; index the tuple with get0/);
});
