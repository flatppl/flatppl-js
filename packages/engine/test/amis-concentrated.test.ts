'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { amisSample } = require('../amis-sample.ts');
const fs = require('node:fs');
const path = require('node:path');
const { resolveBundle } = require('..');
const { createWorkerHandler } = require('../worker.ts');

test('AMIS retains mass and uncertainty when the posterior is much narrower than its initial proposal', () => {
  const dim = 6, sigma = 0.001;
  const mv = { dim, logPosterior: (x: Float64Array) =>
    -dim * Math.log(sigma * Math.sqrt(2 * Math.PI))
    -x.reduce((s, a) => s + ((a - 1) / sigma) ** 2, 0) / 2 };
  const result = amisSample(mv, { seed: 2, amisIters: 30, amisSamples: 300 });
  let mass = 0;
  const first = new Float64Array(dim), second = new Float64Array(dim);
  for (let i = 0; i < result.samples.length; i++) {
    const w = Math.exp(result.logW[i]) / result.samples.length;
    mass += w;
    for (let j = 0; j < dim; j++) {
      const z = (result.samples[i][j] - 1) / sigma;
      first[j] += w * z;
      second[j] += w * z * z;
    }
  }
  assert.ok(Math.abs(mass - 1) < 0.15, `mass ${mass}`);
  for (let j = 0; j < dim; j++) {
    assert.ok(Math.abs(first[j] / mass) < 0.15, `coordinate ${j}: standardized mean ${first[j] / mass}`);
    assert.ok(Math.abs(second[j] / mass - 1) < 0.2, `coordinate ${j}: standardized second moment ${second[j] / mass}`);
  }
});

for (const seed of [1, 2]) test(`AMIS samples posteriorDB BLR with narrow coefficients and inferred noise (seed ${seed})`, async () => {
  // Fixture from flatppl-examples bb31fa7. Complete the square in beta:
  // V=(X'X/sigma²+I/100)^-1, m=V X'y/sigma². Integrating beta analytically
  // and log(sigma) by midpoint quadrature (2000 and 4000 panels on [-8,8])
  // gives these moments and logZ, agreeing within 1e-10 at both resolutions.
  const betaMean = [0.9994620027, 1.0002321420, 1.0004349482, 1.0011426136, 1.0015483619];
  const betaSd = [0.0009644104, 0.0011491356, 0.0009518520, 0.0010617039, 0.0010462710];
  const filename = path.join(__dirname, 'fixtures/posteriordb/blr.flatppl');
  const source = fs.readFileSync(filename, 'utf8');
  const bundle = await resolveBundle(filename, source, (p: string) => fs.promises.readFile(p, 'utf8'));
  const result = await createWorkerHandler().handle({
    type: 'mcmcRun', id: 1, name: 'Pi_post', source,
    processOpts: { path: filename, bundle }, sampleCount: 2000, seed,
    inferenceOpts: { backend: 'amis', seed, amisIters: 30, amisSamples: 300 },
  });
  assert.equal(result.type, 'mcmcResult', result.message);
  const m = result.measure;
  assert.ok(Math.abs(m.diagnostics.logZ + 184.7731723118) < 0.15, `logZ ${m.diagnostics.logZ}`);
  const check = (xs: number[], mean: number, sd: number) => {
    const first = xs.reduce((s, x) => s + (x - mean) / sd, 0) / xs.length;
    const second = xs.reduce((s, x) => s + ((x - mean) / sd) ** 2, 0) / xs.length;
    assert.ok(Math.abs(first) < 0.15, `standardized mean ${first}`);
    assert.ok(Math.abs(second - 1) < 0.2, `standardized second moment ${second}`);
  };
  for (let j = 0; j < 5; j++) {
    check(Array.from(m.fields.beta.samples as Float64Array).filter((_, i) => i % 5 === j), betaMean[j], betaSd[j]);
  }
  check(Array.from(m.fields.sigma.samples), 0.9642112369, 0.0711710886);
});

test('AMIS crosses hard support and preserves target mass scaling', () => {
  const logPrior = (x: Float64Array) => -Math.log(2 * Math.PI) / 2 - x[0] ** 2 / 2;
  const run = (shift: number) => amisSample({ dim: 1,
    logPriorLikBatch: (xs: Float64Array[]) => ({
      prior: Float64Array.from(xs, logPrior),
      lik: Float64Array.from(xs, x => x[0] > 1.9 && x[0] < 2.1 ? shift : -Infinity),
    }),
  }, { seed: 4, amisIters: 30, amisSamples: 300, amisEpsK: 0 });
  const a = run(0), b = run(100);
  let mass = 0, mean = 0, second = 0;
  for (let i = 0; i < a.samples.length; i++) {
    assert.ok(Math.abs(a.samples[i][0] - b.samples[i][0]) < 1e-9);
    if (Number.isFinite(a.logW[i])) assert.ok(Math.abs(b.logW[i] - a.logW[i] - 100) < 1e-8);
    const x = a.samples[i][0], w = Math.exp(a.logW[i]) / a.samples.length;
    mass += w;
    mean += w * x;
    second += w * (x - 2) ** 2;
  }
  // Integral phi(x) dx from 1.9 to 2.1 and truncated-Normal moments.
  assert.ok(Math.abs(mass / 0.0108521393 - 1) < 0.08, `mass ${mass}`);
  assert.ok(Math.abs(mean / mass - 1.99336) < 0.01, `mean ${mean / mass}`);
  assert.ok(Math.abs(second / mass - 0.00334658) < 0.0005, `second moment ${second / mass}`);
});
