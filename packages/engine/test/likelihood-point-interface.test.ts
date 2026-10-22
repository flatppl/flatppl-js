'use strict';

// Spec §04 "Calling conventions" + §06 `likelihoodof`:
// `densityof(likelihoodof(K, obs), theta)` is `pdf(kappa(theta), x)`, so
// `theta` is the argument of a call to `K`. A kernel with explicit boundary
// inputs admits one positional input, so a sole record point is bound whole
// to it and rejected by its domain; it is not splatted. The engine used to
// accept the source and score NaN with no diagnostic.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const MODEL = `
theta ~ Uniform(unitinterval)
y ~ Binomial(10, theta)
K = kernelof(record(y = y), theta = theta)
`;
// pmf of Binomial(10, 0.3) at 3: 120 * 0.3^3 * 0.7^7.
const CLOSED_FORM = Math.log(120 * 0.3 ** 3 * 0.7 ** 7);

function errors(src: string) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

async function score(src: string) {
  const { ctx } = ctxFor(src, 4);
  return (await ctx.getMeasure('s')).samples[0];
}

for (const op of ['logdensityof', 'densityof']) {
  test(`${op}: a record point for a one-boundary kernel is refused, naming the fix`, () => {
    const src = MODEL + `L = likelihoodof(K, record(y = 3))\ns = ${op}(L, record(theta = 0.3))\n`;
    const errs = errors(src);
    assert.equal(errs.length, 1);
    const [d] = errs;
    assert.match(d.message, new RegExp(
      `^${op}: the point is bound whole to likelihood input "theta", which expects real, `
      + 'got record with fields theta: real\\.'));
    assert.match(d.message, /`likelihoodof\(pars -> K\(theta = pars\.theta\), obs\)`/);
    // The diagnostic is located on the point argument, 0-based line 5 of this source.
    assert.equal(d.loc.start.line, 5);
  });
}

test('the scalar point scores the closed form', async () => {
  const src = MODEL + 'L = likelihoodof(K, record(y = 3))\ns = logdensityof(L, 0.3)\n';
  assert.deepEqual(errors(src), []);
  assert.ok(Math.abs(await score(src) - CLOSED_FORM) < 1e-12);
});

test('the adapter form scores the closed form from a record point', async () => {
  const src = MODEL
    + 'L = likelihoodof(pars -> K(theta = pars.theta), record(y = 3))\n'
    + 's = logdensityof(L, record(theta = 0.3))\n';
  assert.deepEqual(errors(src), []);
  assert.ok(Math.abs(await score(src) - CLOSED_FORM) < 1e-12);
});

test('a kernel without boundary inputs still splats a record point', async () => {
  const src = `
theta = elementof(unitinterval)
y ~ Binomial(10, theta)
K = kernelof(record(y = y))
L = likelihoodof(K, record(y = 3))
s = logdensityof(L, record(theta = 0.3))
`;
  assert.deepEqual(errors(src), []);
  assert.ok(Math.abs(await score(src) - CLOSED_FORM) < 1e-12);
});

test('a multi-input boundary kernel still splats a record point', async () => {
  const src = `
a ~ Normal(0, 1)
b ~ Normal(0, 1)
y ~ Normal(a + b, 1)
K = kernelof(record(y = y), a = a, b = b)
L = likelihoodof(K, record(y = 0.0))
s = logdensityof(L, record(a = 0.5, b = 0.25))
`;
  assert.deepEqual(errors(src), []);
  const z = 0.75;
  assert.ok(Math.abs(await score(src) - (-0.5 * Math.log(2 * Math.PI) - z * z / 2)) < 1e-12);
});
