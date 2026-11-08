'use strict';
// §06: bayesupdate(L, prior) is the unnormalized posterior dν = L(θ) dπ(θ),
// lowering to logweighted((x, p) -> logdensityof(p, x), prior, L). Below the
// top level of the density walker it failed "unsupported measure op
// 'bayesupdate'" inside truncate, iid and joint.
//
// Oracle: prior mu ~ Normal(0, 2), observation 2.5 ~ Normal(mu, 1), so the
// posterior log-density at mu is log N(2.5; mu, 1) + log N(mu; 0, 2).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const lnN = (x: number, m: number, s: number) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
const post = (mu: number) => lnN(2.5, mu, 1) + lnN(mu, 0, 2);
const lse = (a: number, b: number) => { const m = Math.max(a, b); return m + Math.log(Math.exp(a - m) + Math.exp(b - m)); };

const HEAD = `mu = elementof(reals)
L = likelihoodof(functionof(Normal(mu = mu, sigma = 1.0)), 2.5)
prior = joint(mu = Normal(mu = 0.0, sigma = 2.0))
post = bayesupdate(L, prior)
`;

async function score(body: string, head = HEAD): Promise<number> {
  const { ctx } = makeMatCtx(head + body, { sampleCount: 1 });
  const m = await ctx.getMeasure('lp');
  return (m.value ? Array.from(m.value.data) : m.samples)[0] as number;
}
function close(got: number, want: number, label: string) {
  assert.ok(Math.abs(got - want) < 1e-12, `${label}: got ${got}, want ${want}`);
}

test('a posterior scores inside truncate', async () => {
  close(await score('lp = logdensityof(truncate(post, cartprod(mu = interval(0.0, 5.0))), record(mu = 0.7))\n'), post(0.7), 'inside');
  assert.equal(await score('lp = logdensityof(truncate(post, cartprod(mu = interval(0.0, 5.0))), record(mu = -0.7))\n'), -Infinity);
});

test('a posterior scores inside iid', async () => {
  close(await score('lp = logdensityof(iid(post, 2), table(mu = [0.7, 1.1]))\n'), post(0.7) + post(1.1), 'iid');
});

test('a posterior scores as a joint component', async () => {
  close(await score('lp = logdensityof(joint(a = post, b = Normal(0.0, 1.0)), record(a = record(mu = 0.7), b = 0.3))\n'),
    post(0.7) + lnN(0.3, 0, 1), 'joint');
});

test('a named kernel and a multi-observation likelihood score below the top level', async () => {
  const named = `mu = elementof(reals)
K = functionof(Normal(mu = mu, sigma = 1.0))
L = likelihoodof(K, 2.5)
prior = joint(mu = Normal(mu = 0.0, sigma = 2.0))
post = bayesupdate(L, prior)
`;
  close(await score('lp = logdensityof(truncate(post, cartprod(mu = interval(0.0, 5.0))), record(mu = 0.7))\n', named), post(0.7), 'named K');
  const iidObs = `mu = elementof(reals)
L = likelihoodof(functionof(iid(Normal(mu = mu, sigma = 1.0), 3)), [2.5, 1.0, 3.0])
prior = joint(mu = Normal(mu = 0.0, sigma = 2.0))
post = bayesupdate(L, prior)
`;
  close(await score('lp = logdensityof(joint(a = post, b = Normal(0.0, 1.0)), record(a = record(mu = 0.7), b = 0.3))\n', iidObs),
    lnN(2.5, 0.7, 1) + lnN(1.0, 0.7, 1) + lnN(3.0, 0.7, 1) + lnN(0.7, 0, 2) + lnN(0.3, 0, 1), 'iid obs');
});

test('the top-level, weighted and superpose paths keep their values', async () => {
  close(await score('lp = logdensityof(post, record(mu = 0.7))\n'), post(0.7), 'top level');
  close(await score('lp = logdensityof(weighted(2.0, post), record(mu = 0.7))\n'), Math.log(2) + post(0.7), 'weighted');
  close(await score('lp = logdensityof(superpose(post, joint(mu = Normal(0.0, 1.0))), record(mu = 0.7))\n'),
    lse(post(0.7), lnN(0.7, 0, 1)), 'superpose');
});
