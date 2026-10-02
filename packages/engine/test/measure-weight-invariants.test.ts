'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

function context(source: string, count = 64) {
  const { ctx } = ctxFor(source, count);
  ctx.rootKey = [4711, 0x9e3779b9];
  return ctx;
}

test('pushforward carries a captured tilted law once, including a shared base parameter', async () => {
  // §04 captured draws and §06 pushfwd: the proposal draws theta from N(0,1),
  // while its target is N(1,1). The centered log ratio is theta_i-theta_0.
  for (const base of ['Normal(0,1)', 'Normal(theta,1)']) {
    const ctx = context(`theta ~ normalize(weighted(fn(exp(_)), Normal(0,1)))
x ~ pushfwd(fn(_ + theta), ${base})`);
    const theta = await ctx.getMeasure('theta');
    const x = await ctx.getMeasure('x');
    for (let i = 0; i < x.samples.length; i++) {
      const actual = (x.logWeights?.[i] ?? 0) - (x.logWeights?.[0] ?? 0);
      assert.ok(Math.abs(actual - theta.samples[i] + theta.samples[0]) < 1e-12);
    }
  }
});

test('callable reweighting counts the base mass once for each variate shape', async () => {
  // §06 dν=f dM: every shape is one atom of mass 3 with weight 2.
  for (const [base, weight] of [
    ['Dirac(2)', 'fn(_)'],
    ['joint(a=Dirac(2))', 'fn(_.a)'],
    ['iid(Dirac(2),2)', 'fn(_[1])'],
  ]) {
    const ctx = context(`M=weighted(${weight}, weighted(3,${base}))
z=totalmass(M)`, 8);
    assert.ok(Math.abs((await ctx.getMeasure('z')).samples[0] - 6) < 1e-12);
  }
  const zero = context('M=weighted(fn(_),weighted(3,Dirac(0)))\nz=totalmass(M)', 8);
  assert.equal((await zero.getMeasure('z')).samples[0], 0);
});

test('normalizing a unit vector atom preserves its subsequent variate weight', async () => {
  // §06 normalization is the identity on this unit atom [2,2].
  const ctx = context(`M=weighted(fn(_[1]),normalize(iid(Dirac(2),2)))
z=totalmass(M)`, 8);
  assert.ok(Math.abs((await ctx.getMeasure('z')).samples[0] - 2) < 1e-12);
});

test('a callable weight carries its captured parameter law', async () => {
  // §06 conditional normalization gives x|theta ~ N(theta,1). Against the
  // proposal phi(theta)phi(x), the joint log ratio is theta+theta*x-theta²/2.
  const ctx = context(`theta~normalize(weighted(fn(exp(_)),Normal(0,1)))
x~normalize(logweighted(fn(theta*_),Normal(0,1)))`);
  const theta = await ctx.getMeasure('theta');
  const x = await ctx.getMeasure('x');
  const logRatio = (i: number) => theta.samples[i]
    + theta.samples[i] * x.samples[i] - theta.samples[i] ** 2 / 2;
  for (let i = 0; i < x.samples.length; i++) {
    const actual = x.logWeights[i] - x.logWeights[0];
    // The existing deterministic conditional normalizer has 3.25e-7 error.
    assert.ok(Math.abs(actual - logRatio(i) + logRatio(0)) < 1e-6);
  }
});

test('a truncation indicator preserves shared parameter weight provenance', async () => {
  // GeneralizedNormal uses the filter fallback. Joining the same theta again
  // must retain its exp(theta) importance event exactly once on accepted atoms.
  const ctx = context(`theta~normalize(weighted(fn(exp(_)),Normal(0,1)))
M=truncate(GeneralizedNormal(theta,1,2),interval(-1,1))
J=joint(t=lawof(theta),y=M)`);
  const theta = await ctx.getMeasure('theta');
  const joint = await ctx.getMeasure('J');
  const accepted = Array.from(joint.logWeights, (_w: number, i: number) => i)
    .filter((i: number) => Number.isFinite(joint.logWeights[i]));
  assert.ok(accepted.length > 1);
  const first = accepted[0];
  for (const i of accepted) {
    const actual = joint.logWeights[i] - joint.logWeights[first];
    assert.ok(Math.abs(actual - theta.samples[i] + theta.samples[first]) < 1e-12);
  }
});
