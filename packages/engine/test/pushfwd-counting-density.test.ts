'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');

async function score(measure: string, point: string, prefix = ''): Promise<number> {
  const { proc, ctx } = ctxFor(`${prefix}\nM = ${measure}\n__score__ = logdensityof(M, ${point})`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const result = await ctx.getMeasure('__score__');
  return result.value.data[0];
}

// §06 counting reference: an injective map relabels atoms without rescaling mass.
// Nonbinary coefficients deliberately make inverse arithmetic round off the atom.
test('scalar affine pushforwards preserve discrete and real atomic masses and support', async () => {
  const bernoulli = 'locscale(normalize(weighted(2, Bernoulli(0.3))), 0.1, 0.3)';
  assert.ok(Math.abs(await score(bernoulli, '0.4') - Math.log(0.3)) < 1e-14);
  assert.ok(Math.abs(await score(bernoulli, '0.1') - Math.log(0.7)) < 1e-14);
  assert.equal(await score(bernoulli, '0.2'), -Infinity);
  assert.equal(await score(bernoulli, '0.7'), -Infinity);
  assert.ok(Math.abs(await score('locscale(Bernoulli(0.3), mu, 0.3)', '0.4', 'mu ~ Dirac(0.1)') - Math.log(0.3)) < 1e-14);
  assert.ok(Math.abs(await score('locscale(truncate(Bernoulli(0.3), interval(1, 1)), 0.1, 0.3)', '0.4') - Math.log(0.3)) < 1e-14);
  assert.equal(await score('locscale(truncate(Bernoulli(0.3), interval(1, 1)), 0.1, 0.3)', '0.1'), -Infinity);
  const atom = 0.3 + 1.7 * 1.2;
  const dirac = 'pushfwd(x -> 1.7 * x + 0.3, lawof(z))';
  assert.equal(await score(dirac, String(atom), 'z ~ Dirac(1.2)'), 0);
  assert.equal(await score(dirac, String(atom + 0.01), 'z ~ Dirac(1.2)'), -Infinity);
});

test('constructor-broadcast affine density retains each counting factor', async () => {
  const measure = 'pushfwd(x -> (x .* 0.3) .+ 0.1, Bernoulli.([0.2, 0.3]))';
  assert.ok(Math.abs(await score(measure, '[0.1, 0.4]') - Math.log(0.8 * 0.3)) < 1e-14);
  assert.equal(await score(measure, '[0.1, 0.2]'), -Infinity);
  const atoms = `[${0.1 + 0.3 * 1.2}, ${0.1 + 0.3 * 2.3}]`;
  assert.equal(await score('pushfwd(x -> 0.3 .* x .+ 0.1, Dirac.([1.2, 2.3]))', atoms), 0);
  assert.equal(await score('pushfwd(x -> 0.3 .* x .+ 0.1, Dirac.(value = atoms))',
    `[${0.1 + 0.3 * 1.2}]`, 'atoms = [1.2]'), 0);
});

test('composed and mixed-atom counting laws recover generated affine atoms', async () => {
  const nested = 'locscale(locscale(Bernoulli(0.3), 0.1, 0.3), 0.2, 0.7)';
  assert.ok(Math.abs(await score(nested, String(0.2 + 0.7 * 0.4)) - Math.log(0.3)) < 1e-14);
  const vector = 'pushfwd(x -> 0.7 .* x .+ 0.2, pushfwd(x -> 0.3 .* x .+ 0.1, iid(Bernoulli(0.3), 2)))';
  assert.ok(Math.abs(await score(vector, '[0.48, 0.48]') - 2 * Math.log(0.3)) < 1e-14);
  const mixture = 'locscale(superpose(weighted(0.5, pushfwd(x -> x + 0.0, Bernoulli(0.3))), weighted(0.5, Dirac(1.2))), 0.1, 0.3)';
  assert.ok(Math.abs(await score(mixture, '0.4') - Math.log(0.15)) < 1e-14);
  assert.ok(Math.abs(await score(mixture, String(0.1 + 0.3 * 1.2)) - Math.log(0.5)) < 1e-14);
  const vectorMixture = 'pushfwd(x -> 0.3 .* x .+ 0.1, superpose(weighted(0.5, iid(Dirac(1.2), 2)), weighted(0.5, iid(Dirac(2.3), 2))))';
  assert.ok(Math.abs(await score(vectorMixture, `[${0.1 + 0.3 * 1.2}, ${0.1 + 0.3 * 1.2}]`) - Math.log(0.5)) < 1e-14);
});

test('diagonal affine volumes use the free dimension of simplex and zero-sum laws', async () => {
  assert.ok(Math.abs(await score('pushfwd(x -> 2 .* x .+ 3, Dirichlet(alpha = [1, 1]))', '[3.5, 4.5]') + Math.log(2)) < 1e-14);
  assert.ok(Math.abs(await score('pushfwd(x -> 2 .* x .+ 3, Lebesgue(zerosum(2)))', '[3.5, 2.5]') + Math.log(2)) < 1e-14);
  const simplex = 'pushfwd(x -> 0.5 .* x .+ 1, Dirichlet(alpha = [1, 1]))';
  const mixture = `normalize(superpose(weighted(2, ${simplex}), weighted(1, ${simplex})))`;
  assert.ok(Math.abs(await score(`pushfwd(x -> 2 .* x .+ 3, ${mixture})`, '[5.25, 5.75]')) < 1e-14);
  const tilted = 'pushfwd(x -> 2 .* x .+ 3, tilt(iid(Normal(0, 1), 2), Lebesgue(zerosum(2))))';
  assert.ok(Math.abs(await score(tilted, '[3, 3]') + 0.5 * Math.log(Math.PI) + Math.log(2)) < 1e-14);
  const restricted = 'pushfwd(x -> 2 .* x .+ 3, truncate(Dirichlet(alpha = [1, 1]), stdsimplex(2)))';
  assert.ok(Math.abs(await score(restricted, '[3.5, 4.5]') + Math.log(2)) < 1e-14);
});

test('mixtures preserve the density of nested continuous and counting product laws', async () => {
  const component = (p: number) => `joint(inner = joint(x = Normal(0, 1), c = Bernoulli(${p})), z = Normal(0, 1))`;
  const mixture = `superpose(weighted(0.3, ${component(0.2)}), weighted(0.7, ${component(0.8)}))`;
  const actual = await score(mixture, 'record(inner = record(x = 0, c = true), z = 0)');
  assert.ok(Math.abs(actual - Math.log((0.3 * 0.2 + 0.7 * 0.8) / (2 * Math.PI))) < 1e-14);
});

test('vector affine counting density preserves distinct product factors', async () => {
  const measure = 'pushfwd(x -> 0.3 .* x .+ 0.1, lawof(z))';
  const prefix = 'a ~ Bernoulli(0.2)\nb ~ Bernoulli(0.3)\nz = [a, b]';
  assert.ok(Math.abs(await score(measure, '[0.1, 0.4]', prefix) - Math.log(0.8 * 0.3)) < 1e-14);
});

test('vector affine counting density preserves iid mass and rejects off-support coordinates', async () => {
  const measure = 'pushfwd(x -> (x .* -0.3) .+ 0.1, iid(Bernoulli(0.3), 2))';
  const atom = 0.1 - 0.3;
  assert.ok(Math.abs(await score(measure, `[0.1, ${atom}]`) - Math.log(0.7 * 0.3)) < 1e-14);
  assert.equal(await score(measure, '[0.1, 0.0]'), -Infinity);
  assert.equal(await score(measure, '[0.1, -0.5]'), -Infinity);
  const multinomial = 'pushfwd(x -> 0.3 .* x .+ 0.1, Multinomial(n = 3, p = [0.2, 0.8]))';
  assert.ok(Math.abs(await score(multinomial, '[0.4, 0.7]') - Math.log(3 * 0.2 * 0.8 ** 2)) < 1e-14);
});

test('affine likelihood batches resolve live real-valued atom support per row', async () => {
  const { proc, ctx } = ctxFor(`
atom ~ Normal(0, 1)
y ~ locscale(Dirac(atom + 0.0), 0.3, 1.7)
K = kernelof(record(y = y), atom = atom)
L = likelihoodof(K, record(y = ${0.3 + 1.7 * 1.2}))
posterior = bayesupdate(L, lawof(record(atom = atom)))`, 3);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  const points = [{ atom: 1.2 }, { atom: 2.3 }, { atom: 1.2 }];
  const expected = [0, -Infinity, 0];
  assert.deepEqual(Array.from(scorer.priorLikBatch(points).lik), expected);
  assert.deepEqual(points.map(point => scorer.likOf(point)), expected);
});

test('nested counting likelihood batches retain live shifts and probabilities', async () => {
  const { proc, ctx } = ctxFor(`
atom ~ Normal(0, 1)
p ~ Uniform(interval(0, 1))
y ~ locscale(locscale(Bernoulli(p), atom, 0.3), 0.2, 0.7)
K = kernelof(record(y = y), atom = atom, p = p)
L = likelihoodof(K, record(y = 0.48))
posterior = bayesupdate(L, lawof(record(atom = atom, p = p)))`, 3);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  const points = [{ atom: 0.1, p: 0.3 }, { atom: 0.4, p: 0.8 }, { atom: 0.2, p: 0.3 }];
  const batch = scorer.priorLikBatch(points).lik;
  assert.ok(Math.abs(batch[0] - Math.log(0.3)) < 1e-14);
  assert.ok(Math.abs(batch[1] - Math.log(0.2)) < 1e-14);
  assert.equal(batch[2], -Infinity);
});
