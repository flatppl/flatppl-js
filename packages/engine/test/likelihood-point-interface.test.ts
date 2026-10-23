'use strict';

// Spec §06 `likelihoodof`: "The parameter point `theta` is a record with one field per input of
// `K`, bound to those inputs by name for any number of inputs, so a kernel with the single input
// `mu` is evaluated at `record(mu = 1.0)`, not at `1.0`." The same holds for the variate of the
// prior in `bayesupdate(L, prior)`. A point that is not a record, or a field that names no input,
// is a located static error. This is a rule of `likelihoodof`; the general §04 call convention
// (a single-input callable receives a sole record whole) is unchanged.

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
  test(`${op}: a scalar point for a one-input kernel is refused, naming the record`, () => {
    const src = MODEL + `L = likelihoodof(K, record(y = 3))\ns = ${op}(L, 0.3)\n`;
    const errs = errors(src);
    assert.equal(errs.length, 1);
    const [d] = errs;
    assert.match(d.message, new RegExp(
      `^${op}: the point is real, but the parameter point of a likelihood is a record with one field `
      + 'per input \\("theta"\\), bound by name for any number of inputs \\(spec §06 `likelihoodof`\\); '
      + 'even a kernel with one input is evaluated at record\\(theta = \\.\\.\\)\\.'));
    assert.match(d.message, /`record\(theta = \.\.\.\)`/);
    // The diagnostic is located on the point argument, 0-based line 5 of this source.
    assert.equal(d.loc.start.line, 5);
  });

}

test('a one-field record point scores the closed form', async () => {
  const src = MODEL + 'L = likelihoodof(K, record(y = 3))\ns = logdensityof(L, record(theta = 0.3))\n';
  assert.deepEqual(errors(src), []);
  assert.ok(Math.abs(await score(src) - CLOSED_FORM) < 1e-12);
});

test('a kernel without boundary inputs binds a record point by name', async () => {
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

test('a multi-input boundary kernel binds a record point by name', async () => {
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
const NOT_RECORD = 'but the parameter point of a likelihood is a record with one field per input';
function pointErrors(src: string) {
  return errors(src).filter((d: any) => d.message.includes(NOT_RECORD));
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
      `^${op}: the point has no input named "nu" for its field \\(inputs: "mu"; `));
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
  assert.match(errs[0].message, /^bayesupdate: the prior's variate has no input named "nu"/);
});

test('a likelihood of a kernel with unknown inputs is not checked', () => {
  const src = MODEL + 'M = external(reals)\nL = likelihoodof(M, record(y = 3))\n'
    + 's = logdensityof(L, record(theta = 0.3))\n';
  assert.deepEqual(pointErrors(src), []);
});

test('a point that is not a record is refused whatever its type', () => {
  const base = MODEL + 'L = likelihoodof(K, record(y = 3))\n';
  for (const [point, type] of [['[0.3]', 'array of real \\(length 1\\)'], ['(0.3, 1.0)', 'tuple'],
    ['table(theta = [0.3, 0.4])', 'table']]) {
    const errs = pointErrors(base + `s = logdensityof(L, ${point})\n`);
    assert.equal(errs.length, 1, point);
    assert.match(errs[0].message, new RegExp(`the point is ${type}`), point);
  }
});

// Spec §04 "Calling conventions": field or column names that do not match the callable's argument
// names are a static error; for a likelihood point the same holds by name., "field or
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
      `^${op}: the point has no input named "unused" for its field \\(inputs: "theta"`));
    assert.equal(errs[0].loc.start.line, 5);
  });
}

test('bayesupdate: an extra prior field for a splatting kernel is refused', () => {
  const errs = errors(SPLAT_MODEL
    + 'prior = lawof(record(theta = theta, other = theta))\npost = bayesupdate(L, prior)\n');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /^bayesupdate: the prior's variate has no input named "other"/);
});

test('a record point naming exactly the inputs still scores the closed form', async () => {
  const src = SPLAT_MODEL + 's = logdensityof(L, record(theta = 0.3))\n';
  assert.deepEqual(errors(src), []);
  assert.ok(Math.abs(await score(src) - CLOSED_FORM) < 1e-12);
});

test('several unknown fields are all named', () => {
  const fields = errors(SPLAT_MODEL + 's = logdensityof(L, record(theta = 0.3, a = 1, b = 2))\n');
  assert.equal(fields.length, 1);
  assert.match(fields[0].message, /has no input named "a", "b" for its fields \(inputs: "theta"/);
});

// Spec §06 `bayesupdate`: the posterior weights the prior by `densityof(L, theta)` at the prior's
// variate, so that variate is the likelihood's point: a record, for any number of inputs.
test('bayesupdate: a scalar prior for a one-input kernel is refused, naming the record prior', () => {
  const src = MODEL + 'L = likelihoodof(K, record(y = 3))\n'
    + 'post = bayesupdate(L, lawof(theta))\n';
  const errs = errors(src);
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, new RegExp(
    '^bayesupdate: the prior\'s variate is real, but the parameter point of a likelihood is a record '
    + 'with one field per input \\("theta"\\), bound by name for any number of inputs '
    + '\\(spec §06 `likelihoodof`\\); even a kernel with one input is evaluated at record\\(theta = \\.\\.\\)\\.'));
  assert.match(errs[0].message, /`joint\(theta = \.\.\.\)`/);
  assert.match(errs[0].message, /`lawof\(record\(theta = theta\)\)`/);
  assert.equal(errs[0].loc.start.line, 5);
});

test('bayesupdate: a one-field record prior is legal and scores the closed form', async () => {
  const src = MODEL + 'L = likelihoodof(K, record(y = 3))\n'
    + 'prior = lawof(record(theta = theta))\npost = bayesupdate(L, prior)\n';
  assert.deepEqual(errors(src), []);
  const { ctx } = ctxFor(src, 4);
  const scorer = await require('../mcmc-density.ts').buildLogPi(ctx, ctx.derivations.post);
  // The prior is Uniform(unitinterval), so the log posterior at 0.3 is the log likelihood.
  assert.ok(Math.abs(scorer.logPi({ theta: 0.3 }) - CLOSED_FORM) < 1e-12);
});

test('a record point missing a likelihood input is a static error', () => {
  const src = `
a ~ Normal(0, 1)
b ~ Normal(0, 1)
y ~ Normal(a + b, 1)
K = kernelof(record(y = y), a = a, b = b)
L = likelihoodof(K, record(y = 0.0))
s = logdensityof(L, record(a = 0.5))
`;
  const errs = errors(src);
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /^logdensityof: the point has no field for the likelihood input "b" \(inputs: "a", "b"/);
  assert.equal(errs[0].loc.start.line, 6);
});
