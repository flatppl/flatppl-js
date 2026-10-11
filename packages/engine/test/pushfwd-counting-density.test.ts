'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

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
});

test('composed and mixed-atom counting laws recover generated affine atoms', async () => {
  const nested = 'locscale(locscale(Bernoulli(0.3), 0.1, 0.3), 0.2, 0.7)';
  assert.ok(Math.abs(await score(nested, String(0.2 + 0.7 * 0.4)) - Math.log(0.3)) < 1e-14);
  const vector = 'pushfwd(x -> 0.7 .* x .+ 0.2, pushfwd(x -> 0.3 .* x .+ 0.1, iid(Bernoulli(0.3), 2)))';
  assert.ok(Math.abs(await score(vector, '[0.48, 0.48]') - 2 * Math.log(0.3)) < 1e-14);
  const mixture = 'locscale(superpose(weighted(0.5, pushfwd(x -> x + 0.0, Bernoulli(0.3))), weighted(0.5, Dirac(1.2))), 0.1, 0.3)';
  assert.ok(Math.abs(await score(mixture, '0.4') - Math.log(0.15)) < 1e-14);
  assert.ok(Math.abs(await score(mixture, String(0.1 + 0.3 * 1.2)) - Math.log(0.5)) < 1e-14);
});

test('diagonal affine volumes use the free dimension of simplex and zero-sum laws', async () => {
  assert.ok(Math.abs(await score('pushfwd(x -> 2 .* x .+ 3, Dirichlet(alpha = [1, 1]))', '[3.5, 4.5]') + Math.log(2)) < 1e-14);
  assert.ok(Math.abs(await score('pushfwd(x -> 2 .* x .+ 3, Lebesgue(zerosum(2)))', '[3.5, 2.5]') + Math.log(2)) < 1e-14);
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

test('affine counting density resolves live source atoms with fixed transform parameters', async () => {
  const measure = 'locscale(Dirac(atom), 0.3, 1.7)';
  assert.equal(await score(measure, String(0.3 + 1.7 * 1.2), 'atom ~ Dirac(1.2)'), 0);
  assert.equal(await score(measure, '2.35', 'atom ~ Dirac(1.2)'), -Infinity);
});
