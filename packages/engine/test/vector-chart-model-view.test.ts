'use strict';
// §03 ordered/zero-sum coordinate measures and §06 tilt, through the posterior
// ModelView used by MCMC, AMIS and SMC. Closed-form scores include normalization.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');

const LN_2PI = Math.log(2 * Math.PI);
function close(actual: number, expected: number) {
  assert.ok(Math.abs(actual - expected) < 1e-11, `${actual} vs ${expected}`);
}
async function view(prior: string, mean = 'x[1]') {
  const { ctx } = ctxFor(`${prior}
y ~ Normal(${mean}, 1)
P = lawof(record(x = x))
L = likelihoodof(pars -> kernelof(y, x = x)(x = pars.x), 0)
posterior = bayesupdate(L, P)
`, 32);
  return buildModelViewFromCtx(ctx, ctx.derivations.posterior);
}

for (const domain of ['ordered', 'zerosum']) {
  test(`${domain} posterior uses its intrinsic chart and reference-measure Jacobian`, async () => {
    const mv = await view(`x ~ tilt(iid(Normal(0, 1), 3), Lebesgue(${domain}(3)))`);
    const ordered = domain === 'ordered';
    const x = ordered ? [1, 3, 6] : [1, 2, -3];
    const y = Float64Array.from(ordered ? [1, Math.log(2), Math.log(3)] : [1, 2]);
    // Ordered density: 3! prod Normal(x_i); zero-sum density:
    // sqrt(3)/(2pi) exp(-sum(x_i²)/2) in its two free coordinates.
    const constrained = ordered ? Math.log(6) - 2 * LN_2PI - 23.5
      : 0.5 * Math.log(3) - 1.5 * LN_2PI - 7.5;
    const jacobian = ordered ? Math.log(6) : 0;
    assert.equal(mv.dim, y.length);
    assert.deepEqual(mv.names, ['x[0]', 'x[1]', 'x[2]']);
    const flat = mv.constrainAll(y);
    x.forEach((v, i) => close(flat[`x[${i}]`], v));
    const back = mv.unconstrainAll({ x: Float64Array.from(x) });
    y.forEach((v, i) => close(back[i], v));
    close(mv.logPosteriorConstrained({ x }), constrained);
    close(mv.logPosterior(y), constrained + jacobian);
    close(mv.logPosteriorBatch([y])[0], constrained + jacobian);
    const { prior, lik } = mv.logPriorLikBatch([y]);
    close(prior[0], constrained + jacobian + 0.5 * LN_2PI + 0.5);
    close(lik[0], -0.5 * LN_2PI - 0.5);

    // The Gaussian proposal at equal free coordinates used to leave both sets.
    const proposed = mv.constrainAll(new Float64Array(mv.dim).fill(1));
    if (ordered) assert.ok(proposed['x[0]'] < proposed['x[1]'] && proposed['x[1]'] < proposed['x[2]']);
    else close(proposed['x[0]'] + proposed['x[1]'] + proposed['x[2]'], 0);
  });
}

test('mixed vector charts retain block offsets and an inferred prior scale', async () => {
  const { ctx } = ctxFor(`
scale ~ Exponential(1)
effects ~ tilt(iid(Normal(0, scale), 3), Lebesgue(zerosum(3)))
cuts ~ tilt(iid(Normal(0, 1), 3), Lebesgue(ordered(3)))
tail ~ Normal(0, 1)
y ~ Normal(sum(cuts) + effects[1] + tail, 1)
P = lawof(record(scale = scale, effects = effects, cuts = cuts, tail = tail))
L = likelihoodof(kernelof(y, scale = scale, effects = effects, cuts = cuts, tail = tail), 0)
posterior = bayesupdate(L, P)
`, 32);
  const mv = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  const point = { scale: 2, effects: [1, 2, -3], cuts: [-1, 0, 2], tail: 0.5 };
  const y = mv.unconstrainAll(point);
  assert.equal(mv.dim, 7);
  const want = [Math.log(2), 1, 2, -1, 0, Math.log(2), 0.5];
  want.forEach((v, i) => close(y[i], v));
  const constrained = Math.log(6) + 0.5 * Math.log(3) - 2 * Math.log(2) - 3.5 * LN_2PI - 9.5;
  close(mv.logPosteriorConstrained(point), constrained);
  close(mv.logPosterior(y), constrained + 2 * Math.log(2));
  close(mv.logPosteriorBatch([y])[0], constrained + 2 * Math.log(2));
  for (const initial of mv.initFromPrior(4, () => 0.5)) {
    const flat = mv.constrainAll(initial);
    close(flat['effects[0]'] + flat['effects[1]'] + flat['effects[2]'], 0);
    assert.ok(flat['cuts[0]'] < flat['cuts[1]'] && flat['cuts[1]'] < flat['cuts[2]']);
    assert.ok(Number.isFinite(mv.logPosterior(initial)));
  }
});

for (const domain of ['ordered', 'zerosum']) {
  test(`${domain}(1) retains a vector value with the right free dimension`, async () => {
    const mv = await view(`n = 1
S = ${domain}(n)
B = Lebesgue(S)
x ~ tilt(iid(Normal(0, 1), n), B)`);
    const ordered = domain === 'ordered';
    assert.equal(mv.dim, ordered ? 1 : 0);
    assert.deepEqual(mv.names, ['x[0]']);
    const y = new Float64Array(mv.dim);
    close(mv.constrainAll(y)['x[0]'], 0);
    assert.equal(mv.unconstrainAll({ x: [0] }).length, mv.dim);
    close(mv.logPosterior(y), ordered ? -LN_2PI : -0.5 * LN_2PI);
  });
}
