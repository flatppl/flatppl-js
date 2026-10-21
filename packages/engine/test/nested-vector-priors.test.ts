'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildPriorTransform } = require('../prior-transform.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const { materialiser } = require('..');

// Phi(1), Phi(-1), Phi(1/2): fixed standard Normal CDF values.
const P1 = 0.8413447460685429, PM1 = 0.15865525393145707, PHALF = 0.6914624612740131;
const LN_2PI = Math.log(2 * Math.PI);
function close(actual: number, expected: number, tol = 1e-11) {
  assert.ok(Math.abs(actual - expected) < tol, `${actual} vs ${expected}`);
}
function setup(source: string, fields = 'x = x') {
  const { ctx } = ctxFor(`${source}
prior = lawof(record(${fields}))
y ~ Normal(x[1], 1)
L = likelihoodof(pars -> kernelof(y, x = x)(x = pars.x), 0)
posterior = bayesupdate(L, prior)
`, 32);
  const d = ctx.derivations.posterior;
  return { ctx, d, pt: buildPriorTransform(ctx, d) };
}

// Numerical derivative of the first three output coordinates with respect to
// the three cube coordinates; independent of the production map's formula.
function logVolume3(transform: any, u: Float64Array): number {
  const J: number[][] = [[], [], []], h = 1e-6;
  for (let j = 0; j < 3; j++) {
    const lo = u.slice(), hi = u.slice(); lo[j] -= h; hi[j] += h;
    const a = transform(lo).x, b = transform(hi).x;
    for (let i = 0; i < 3; i++) J[i][j] = (b[i] - a[i]) / (2 * h);
  }
  const [a, b, c] = J;
  return Math.log(Math.abs(a[0] * (b[1] * c[2] - b[2] * c[1])
    - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])));
}

test('ordered Normal cube prior has the sorted law and its factorial density', async () => {
  const { ctx, d, pt } = setup('x ~ tilt(iid(Normal(1, 2), 3), Lebesgue(ordered(3)))');
  assert.equal(pt.dim, 3);
  const u = Float64Array.from([P1, 0.5, PM1]);
  const rec = pt.transform(u);
  [-1, 1, 3].forEach((x, i) => close(rec.x[i], x));
  const logPrior = Math.log(6) - 3 * Math.log(2) - 1.5 * LN_2PI - 1;
  close((await buildLogPi(ctx, d)).priorOf(rec), logPrior);
  // Six permutation chambers map onto ordered(3), each with this local volume.
  close(logPrior + logVolume3(pt.transform, u), Math.log(6), 1e-8);
});

test('zero-sum Normal cube prior has intrinsic covariance and coordinate density', async () => {
  const { ctx, d, pt } = setup('x ~ tilt(iid(Normal(7, 2), 4), Lebesgue(zerosum(4)))');
  assert.equal(pt.dim, 3);
  const columns = Array.from({ length: 3 }, (_, j) => {
    const u = new Float64Array(3).fill(0.5); u[j] = P1;
    return pt.transform(u).x;
  });
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const covariance = columns.reduce((sum, col) => sum + col[i] * col[j], 0);
      close(covariance, 4 * ((i === j ? 1 : 0) - 0.25));
    }
  }
  const u = Float64Array.from([P1, PM1, PHALF]);
  const rec = pt.transform(u);
  close(rec.x.reduce((a: number, b: number) => a + b, 0), 0);
  const logPrior = 0.5 * Math.log(4) - 3 * Math.log(2) - 1.5 * LN_2PI
    - rec.x.reduce((sum: number, x: number) => sum + x * x, 0) / 8;
  close((await buildLogPi(ctx, d)).priorOf(rec), logPrior);
  close(logPrior + logVolume3(pt.transform, u), 0, 1e-8);
});

test('vector prior transforms condition on current parents and derived scale', () => {
  const { pt } = setup(`
mu ~ Normal(1, 2)
raw_scale ~ Exponential(1)
scale = 2 * raw_scale
x ~ tilt(iid(Normal(mu, scale), 3), Lebesgue(ordered(3)))
z ~ tilt(iid(Normal(mu, scale), 3), Lebesgue(zerosum(3)))
`, 'mu = mu, raw_scale = raw_scale, x = x, z = z');
  assert.equal(pt.dim, 7);
  const u = Float64Array.from([0.5, 0.5, PM1, 0.5, P1, P1, PM1]);
  const a = pt.transform(u);
  close(a.mu, 1); close(a.raw_scale, Math.log(2));
  [-1, 0, 1].forEach((z, i) => close(a.x[i], 1 + 2 * Math.log(2) * z));
  const shifted = u.slice(); shifted[0] = P1;
  const b = pt.transform(shifted);
  for (let i = 0; i < 3; i++) { close(b.x[i] - a.x[i], 2); close(b.z[i], a.z[i]); }
  const scaled = u.slice(); scaled[1] = 0.75;
  const c = pt.transform(scaled);
  for (let i = 0; i < 3; i++) { close(c.x[i] - 1, 2 * (a.x[i] - 1)); close(c.z[i], 2 * a.z[i]); }
});

test('singleton vector prior transforms retain one or zero cube axes', () => {
  for (const domain of ['ordered', 'zerosum']) {
    const { pt } = setup(`x ~ tilt(iid(Normal(7, 2), 1), Lebesgue(${domain}(1)))`);
    assert.equal(pt.dim, domain === 'ordered' ? 1 : 0);
    const rec = pt.transform(new Float64Array(pt.dim).fill(0.5));
    assert.equal(rec.x.length, 1);
    close(rec.x[0], domain === 'ordered' ? 7 : 0);
  }
});

test('nested posterior retains singleton vector rank at the likelihood boundary', async () => {
  const { ctx } = setup('x ~ tilt(iid(Normal(7, 2), 1), Lebesgue(zerosum(1)))');
  const posterior = await materialiser.materialiseMeasure('posterior', ctx,
    { backend: 'nested', nLive: 16, dlogz: 0.5, seed: 7 });
  // The normalized prior is a point at [0], so evidence is exactly Normal(0,1)[0].
  close(posterior.logTotalmass, -0.5 * LN_2PI);
  assert.equal(posterior.fields.x.value.shape[1], 1);
  assert.ok(posterior.fields.x.value.data.every((x: number) => x === 0));
});

test('nested tilt support does not reinterpret other exact laws as iid Normal', () => {
  assert.throws(() => setup('x ~ tilt(iid(Exponential(1), 3), Lebesgue(ordered(3)))'),
    /nested backend.*tilt/);
});
