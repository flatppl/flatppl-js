'use strict';

// §06 each conditional law divides by its own mass at the current theta.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const unit = 'cartprod(interval(0.0,1.0),interval(0.0,1.0))';
function source(weight: string, box = unit, point = '[0.2,0.4]') {
  return 'theta ~ Uniform(interval(0.5,2.0))\n'
    + `m=normalize(truncate(weighted((v,t)->${weight},Lebesgue(${box}),theta),${box}))\n`
    + `y~m\nK=kernelof(record(y=y))\nL=likelihoodof(K,record(y=${point}))\nposterior=bayesupdate(L,lawof(theta))\n`;
}
async function scorer(src: string) {
  const { ctx } = ctxFor(src, 16);
  const d = Object.values(ctx.derivations).find((x: any) => x.kind === 'bayesupdate');
  return buildLogPi(ctx, d);
}
function close(got: number, want: number) { assert.ok(Math.abs(got - want) < 2e-12, `${got} vs ${want}`); }

test('MCMC scalar and batch scores retain each live polynomial normalizer', async () => {
  const p = await scorer(source('t+v[1]'));
  const points = [{ theta: 0.6 }, { theta: 1.4 }];
  const batch = p.priorLikBatch(points);
  points.forEach((pt, i) => {
    const want = Math.log(pt.theta + 0.2) - Math.log(pt.theta + 0.5);
    close(p.likOf(pt), want);
    close(batch.lik[i], want);
    close(p.logPi(pt), want - Math.log(1.5));
    p.probeLik(pt);
  });
  const outside = await scorer(source('t+v[1]', unit, '[1.2,0.4]'));
  assert.equal(outside.likOf({ theta: 1 }), -Infinity);
});

test('MCMC uses centered exponential box scores and skips out-of-prior normalizers', async () => {
  const wide = 'cartprod(interval(0.0,2.0),interval(0.0,1.0))';
  const p = await scorer(source('exp(710.0-1.0e308*t*v[1])', wide, '[0.0,0.5]'));
  for (const theta of [0.6, 1.4]) close(p.likOf({ theta }), Math.log(1e308 * theta));
  const batch = p.priorLikBatch([{ theta: -1 }, { theta: 1 }]);
  assert.equal(batch.prior[0], -Infinity);
  assert.equal(batch.lik[0], -Infinity);
  close(batch.lik[1], Math.log(1e308));
});

test('unresolved MCMC truncation refuses through scalar, probe and batch APIs', async () => {
  const p = await scorer(source('t+100.0*exp(-0.5*((v[1]-0.3)/0.001)^2)'));
  for (const run of [() => p.likOf({ theta: 1 }), () => p.probeLik({ theta: 1 }),
    () => p.priorLikBatch([{ theta: 1 }]), () => p.logPi({ theta: 1 })]) {
    assert.throws(run, { code: 'ENGINE_LIMITATION' });
  }
});

test('the Lebesgue box proof does not drop a shifted Uniform reference density', async () => {
  const src = source('t+v', 'interval(0.0,1.0)', '0.2')
    .replace('Lebesgue(interval(0.0,1.0))', 'logweighted(3.0,Uniform(interval(0.0,2.0)))');
  await assert.rejects(async () => (await scorer(src)).likOf({ theta: 0.6 }), { code: 'ENGINE_LIMITATION' });
});
