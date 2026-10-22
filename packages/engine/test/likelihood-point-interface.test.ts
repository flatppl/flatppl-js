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

// Guards: the check stays silent where the point's type, or the likelihood's
// interface, is not known statically, and states the mismatch without the
// record hint when the point is not a record.
const WHOLE = 'bound whole to likelihood input';
function pointErrors(src: string) {
  return errors(src).filter((d: any) => d.message.includes(WHOLE));
}

test('a point of deferred, any or failed type is not checked', () => {
  const base = MODEL + 'L = likelihoodof(K, record(y = 3))\n';
  for (const point of ['external(reals)', 'elementof(anything)', '0.3 + nosuch']) {
    assert.deepEqual(pointErrors(base + `pt = ${point}\ns = logdensityof(L, pt)\n`), [], point);
  }
});

// §06 `joint_likelihood(L1, L2)` is `L(θ) = L1(θ) + L2(θ)` and is equivalent to
// `likelihoodof(joint(K1, K2), obs)`. A `joint` of kernels takes "the union of the
// component kernels' inputs by name", so when every term splats a record point,
// the point splats over the union and every field must name an input of a term.
const JOINT = `
mu = elementof(reals)
a ~ Normal(mu, 1)
b ~ Normal(mu, 2)
K1 = kernelof(record(a = a))
K2 = kernelof(record(b = b))
L = joint_likelihood(likelihoodof(K1, record(a = 0.1)), likelihoodof(K2, record(b = 0.2)))
`;

test('a joint_likelihood splats its point over the union of the terms\' inputs', () => {
  assert.deepEqual(errors(JOINT + 's = logdensityof(L, record(mu = 0.3))\n'), []);
});

for (const op of ['logdensityof', 'densityof']) {
  test(`${op}: a joint_likelihood point with a field naming no input is refused`, () => {
    const errs = errors(JOINT + `s = ${op}(L, record(mu = 0.3, nu = 1.0))\n`);
    assert.equal(errs.length, 1);
    assert.match(errs[0].message, new RegExp(
      `^${op}: the point is splatted over the likelihood inputs, but has no input named "nu" `
      + 'for its field \\(inputs: "mu"; '));
    assert.equal(errs[0].loc.start.line, 7);
  });
}

test('a joint_likelihood over terms with different inputs takes the union by name', () => {
  const src = `
mu = elementof(reals)
sd = elementof(posreals)
a ~ Normal(mu, 1)
b ~ Normal(mu, sd)
K1 = kernelof(record(a = a))
K2 = kernelof(record(b = b))
L = joint_likelihood(likelihoodof(K1, record(a = 0.1)), likelihoodof(K2, record(b = 0.2)))
ok = logdensityof(L, record(mu = 0.3, sd = 1.5))
bad = logdensityof(L, record(mu = 0.3, sd = 1.5, extra = 1.0))
`;
  const errs = errors(src);
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /no input named "extra"/);
  assert.match(errs[0].message, /inputs: "mu", "sd"/);
});

test('a joint_likelihood term taking one positional input receives the point whole', () => {
  // Each term applies the shared point through its own interface, so a record
  // point is not checked against the union of names (`pars` here is an
  // adapter's argument, not a field of the point).
  const src = `
mu = elementof(reals)
a ~ Normal(mu, 1)
b ~ Normal(mu, 2)
K1 = kernelof(record(a = a), mu = mu)
K2 = kernelof(record(b = b), mu = mu)
L = joint_likelihood(likelihoodof(pars -> K1(mu = pars.mu), record(a = 0.1)),
                     likelihoodof(pars -> K2(mu = pars.mu), record(b = 0.2)))
s = logdensityof(L, record(mu = 0.3, nu = 1.0))
`;
  assert.deepEqual(errors(src), []);
});

test('a joint_likelihood with a term of unknown interface is not checked', () => {
  // `hepphys.ContinuedPoisson.(...)` types as a function, so its likelihood
  // carries no inputs and the union of the terms' inputs would be incomplete.
  const src = `
hepphys = standard_module("particle-physics", "0.1")
mu = elementof(reals)
sb = elementof(cartpow(posreals, 2))
tau = [1.0, 2.0]
c1 = functionof(Normal(mu = mu, sigma = 1.0))
c2 = functionof(hepphys.ContinuedPoisson.(sb .* tau))
L = joint_likelihood(likelihoodof(c1, 0.1), likelihoodof(c2, [1, 2]))
s = logdensityof(L, record(mu = 1.0, sb = [1.0, 1.0]))
`;
  assert.deepEqual(errors(src), []);
});

test('a joint_likelihood of terms without inputs is not checked', () => {
  const src = `
M = Normal(0, 1)
L = joint_likelihood(likelihoodof(M, 0.1), likelihoodof(M, 0.2))
s = logdensityof(L, record(x = 1.0))
`;
  assert.deepEqual(pointErrors(src), []);
});

test('a bayesupdate prior over a joint_likelihood must name only the union of inputs', () => {
  const errs = errors(JOINT + 'prior = joint(mu = Normal(0, 1), nu = Normal(0, 1))\npost = bayesupdate(L, prior)\n');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /^bayesupdate: the prior's variate is splatted/);
});

test('a likelihood of a kernel with unknown inputs is not checked', () => {
  const src = MODEL + 'M = external(reals)\nL = likelihoodof(M, record(y = 3))\n'
    + 's = logdensityof(L, record(theta = 0.3))\n';
  assert.deepEqual(pointErrors(src), []);
});

test('a non-record point of the wrong type is refused without the record hint', () => {
  const src = MODEL + 'L = likelihoodof(K, record(y = 3))\ns = logdensityof(L, [0.3])\n';
  const errs = pointErrors(src);
  assert.equal(errs.length, 1);
  assert.match(errs[0].message,
    /^logdensityof: the point is bound whole to likelihood input "theta", which expects real, got array of real \(length 1\)\.$/);
});

// Spec §04 "Calling conventions": when auto-splatting applies, "field or
// column names that do not match the callable's argument names are a static
// error".
const SPLAT_MODEL = `
theta = elementof(unitinterval)
y ~ Binomial(10, theta)
K = kernelof(record(y = y))
L = likelihoodof(K, record(y = 3))
`;

for (const op of ['logdensityof', 'densityof']) {
  test(`${op}: an extra record field for a splatting kernel is refused`, () => {
    const errs = errors(SPLAT_MODEL + `s = ${op}(L, record(theta = 0.3, unused = 7))\n`);
    assert.equal(errs.length, 1);
    assert.match(errs[0].message, new RegExp(
      `^${op}: the point is splatted over the likelihood inputs, but has no input named `
      + '"unused" for its field \\(inputs: "theta"'));
    assert.equal(errs[0].loc.start.line, 5);
  });
}

test('bayesupdate: an extra prior field for a splatting kernel is refused', () => {
  const errs = errors(SPLAT_MODEL
    + 'prior = lawof(record(theta = theta, other = theta))\npost = bayesupdate(L, prior)\n');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /^bayesupdate: the prior's variate is splatted .* no input named "other"/);
});

test('a record point naming exactly the inputs still scores the closed form', async () => {
  const src = SPLAT_MODEL + 's = logdensityof(L, record(theta = 0.3))\n';
  assert.deepEqual(errors(src), []);
  assert.ok(Math.abs(await score(src) - CLOSED_FORM) < 1e-12);
});

test('several unknown fields, and unknown table columns, are all named', () => {
  const fields = errors(SPLAT_MODEL + 's = logdensityof(L, record(theta = 0.3, a = 1, b = 2))\n');
  assert.equal(fields.length, 1);
  assert.match(fields[0].message, /has no input named "a", "b" for its fields \(inputs: "theta"/);
  const columns = errors(SPLAT_MODEL + 's = logdensityof(L, table(theta = [0.3, 0.4], extra = [1, 2]))\n');
  assert.equal(columns.length, 1);
  assert.match(columns[0].message, /has no input named "extra" for its column \(inputs: "theta"/);
});

// Spec §06 `bayesupdate`: the posterior weights the prior by
// `densityof(L, theta)` at the prior's variate, so a record-variate prior is
// the same whole-binding mismatch as a record point.
test('bayesupdate: a record prior for a one-boundary kernel is refused, naming both fixes', () => {
  const src = MODEL + 'L = likelihoodof(K, record(y = 3))\n'
    + 'post = bayesupdate(L, lawof(record(theta = theta)))\n';
  const errs = errors(src);
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, new RegExp(
    '^bayesupdate: the prior\'s variate is bound whole to likelihood input "theta", which '
    + 'expects real, got record with fields theta: real\\.'));
  assert.match(errs[0].message, /`lawof\(theta\)`/);
  assert.match(errs[0].message, /`likelihoodof\(pars -> K\(theta = pars\.theta\), obs\)`/);
  assert.equal(errs[0].loc.start.line, 5);
});

test('bayesupdate: a scalar prior and the adapter form are legal and score the closed form', async () => {
  for (const [L, prior] of [
    ['likelihoodof(K, record(y = 3))', 'lawof(theta)'],
    ['likelihoodof(pars -> K(theta = pars.theta), record(y = 3))', 'lawof(record(theta = theta))'],
  ]) {
    const src = MODEL + `L = ${L}\nprior = ${prior}\npost = bayesupdate(L, prior)\n`;
    assert.deepEqual(errors(src), []);
    const { ctx } = ctxFor(src, 4);
    const scorer = await require('../mcmc-density.ts').buildLogPi(ctx, ctx.derivations.post);
    // The prior is Uniform(unitinterval), so the log posterior at 0.3 is the log likelihood.
    assert.ok(Math.abs(scorer.logPi({ theta: 0.3 }) - CLOSED_FORM) < 1e-12);
  }
});
