'use strict';
// A TWO-COMPONENT CATEGORICAL MIXTURE `xs[i]` MUST SAMPLE EACH BRANCH IN
// PROPORTION TO ITS WEIGHT.
//
// matSelect's gather picked the branch with `sel ? 0 : 1` whenever there were
// two branches. That is the Bernoulli / ifelse rule. A Categorical selector
// takes the values {1, 2} (Categorical0: {0, 1}), so `Categorical` sampled
// branch 0 for every atom and `Categorical0` swapped the two branches, while
// the density, which indexes by `selector - base`, stayed right.
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

test('two-branch Categorical0 xs[i] samples each branch with its weight', async () => {
  const got = await fractionOnBranchB(BRANCHES + `
i = draw(Categorical0(p = [0.3, 0.7]))
xs = [a, b]
x = xs[i]
`);
  assertFraction(got, 0.7, 'Categorical0');
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
