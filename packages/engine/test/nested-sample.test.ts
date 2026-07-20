'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { runNested } = require('../nested-sample.ts');
const { quantile } = require('../inverse-cdf.ts');

test('nested: 1-D Gaussian×Gaussian evidence matches closed form', () => {
  const sigmaL = 2.0;
  const transform = (u: Float64Array) => ({ theta: quantile('Normal', u[0], { mu: 0, sigma: 1 }) });  // N(0,1) prior
  const LN2PI = Math.log(2 * Math.PI);
  const logLik = (rec: any) => -0.5 * (rec.theta / sigmaL) ** 2 - 0.5 * (LN2PI + 2 * Math.log(sigmaL));
  // deterministic PRNG (seeded LCG) so the test is reproducible.
  let s = 12345 >>> 0;
  const prng = () => { s = (1103515245 * s + 12345) >>> 0; return s / 4294967296; };
  const res = runNested(transform, 1, logLik, { nLive: 400, dlogz: 0.01, prng });
  const logZexact = -0.5 * (LN2PI + Math.log(1 + sigmaL ** 2));
  assert.ok(Math.abs(res.logZ - logZexact) < 5 * res.logZerr + 0.05, `logZ ${res.logZ} vs ${logZexact} (err ${res.logZerr})`);
});

test('nested: nLive < 2 throws (constrained-draw seed needs another live point)', () => {
  const transform = (u: Float64Array) => ({ theta: quantile('Normal', u[0], { mu: 0, sigma: 1 }) });
  const logLik = (rec: any) => -0.5 * rec.theta ** 2;
  let s = 12345 >>> 0;
  const prng = () => { s = (1103515245 * s + 12345) >>> 0; return s / 4294967296; };
  assert.throws(() => runNested(transform, 1, logLik, { nLive: 1, prng }));
});
