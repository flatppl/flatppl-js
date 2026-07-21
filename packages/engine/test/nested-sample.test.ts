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

test('nested: region rejection keeps evidence correct and lifts efficiency', () => {
  const sigmaL = 2.0;
  const transform = (u: Float64Array) => ({ theta: quantile('Normal', u[0], { mu: 0, sigma: 1 }) });
  const LN2PI = Math.log(2 * Math.PI);
  const logLik = (rec: any) => -0.5 * (rec.theta / sigmaL) ** 2 - 0.5 * (LN2PI + 2 * Math.log(sigmaL));
  let s = 999 >>> 0; const prng = () => { s = (1103515245 * s + 12345) >>> 0; return s / 4294967296; };
  const res = runNested(transform, 1, logLik, { nLive: 300, dlogz: 0.01, prng, useRegion: true });
  const logZexact = -0.5 * (LN2PI + Math.log(1 + sigmaL ** 2));
  assert.ok(Math.abs(res.logZ - logZexact) < 0.1, `logZ ${res.logZ} vs ${logZexact}`);
  assert.ok(res.efficiency > 0.10, `region efficiency ${res.efficiency} should beat region-free ~0.02`);
});

test('nested: closed-form multi-D Gaussian evidence is unbiased across seeds (region-free)', () => {
  // Independent oracle: prior theta_i ~ N(0,1) iid, likelihood L(theta) =
  // prod_i N(theta_i; 0, sigmaL) => Z = prod_i N(0; 0, sqrt(1+sigmaL^2)),
  // so logZexact = -(d/2)*(log(2*pi) + log(1+sigmaL^2)). Region-free static NS
  // is the shipped default (mat-density.ts backend:'nested'); this checks it
  // has no dimensional bias and is seed-stable at d=3 and d=5.
  const LN2PI = Math.log(2 * Math.PI);
  const sigmaL = 2.0;
  const seeds = [111, 222, 333];
  for (const d of [3, 5]) {
    const transform = (u: Float64Array) => {
      const theta = new Array(d);
      for (let i = 0; i < d; i++) theta[i] = quantile('Normal', u[i], { mu: 0, sigma: 1 });
      return { theta };
    };
    const logLik = (rec: any) => {
      let s = 0;
      for (let i = 0; i < d; i++) s += -0.5 * (rec.theta[i] / sigmaL) ** 2 - 0.5 * (LN2PI + 2 * Math.log(sigmaL));
      return s;
    };
    const logZexact = -(d / 2) * (LN2PI + Math.log(1 + sigmaL ** 2));
    const logZs: number[] = [];
    for (const seed0 of seeds) {
      let s = seed0 >>> 0;
      const prng = () => { s = (1103515245 * s + 12345) >>> 0; return s / 4294967296; };
      const res = runNested(transform, d, logLik, { nLive: 400, dlogz: 0.05, prng, useRegion: false });
      const tol = Math.max(0.15, 4 * res.logZerr);
      assert.ok(Math.abs(res.logZ - logZexact) < tol,
        `d=${d} seed=${seed0}: logZ ${res.logZ} vs exact ${logZexact} (tol ${tol}, logZerr ${res.logZerr})`);
      logZs.push(res.logZ);
    }
    const mean = logZs.reduce((a, b) => a + b, 0) / logZs.length;
    assert.ok(Math.abs(mean - logZexact) < 0.1,
      `d=${d}: 3-seed mean logZ ${mean} vs exact ${logZexact} (seeds ${logZs.join(', ')})`);
  }
});

test('nested: nLive < 2 throws (constrained-draw seed needs another live point)', () => {
  const transform = (u: Float64Array) => ({ theta: quantile('Normal', u[0], { mu: 0, sigma: 1 }) });
  const logLik = (rec: any) => -0.5 * rec.theta ** 2;
  let s = 12345 >>> 0;
  const prng = () => { s = (1103515245 * s + 12345) >>> 0; return s / 4294967296; };
  assert.throws(() => runNested(transform, 1, logLik, { nLive: 1, prng }));
});
