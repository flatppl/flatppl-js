'use strict';

// Independent oracles use sums of ordinary Normal densities, avoiding the
// implementation's logarithmic mixture calculation. Existing density and
// multivariate suites retain their precomputed distribution-library oracles.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { processSource } = require('..');

async function value(src: string, expr: string): Promise<number> {
  const { ctx } = ctxFor(src + '\n__value__ = ' + expr, 1);
  const m = await ctx.getMeasure('__value__');
  return (m.samples || m.value.data)[0];
}
function pdf(x: number, mu: number, sigma: number): number {
  return Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (Math.sqrt(2 * Math.PI) * sigma);
}
function close(actual: number, expected: number) {
  assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
}

test('singleton and scalar weights repeat; normalization sums expanded weights', async () => {
  for (const w of ['[1.0]', '1.0']) {
    const src = `m = ksuperpose(Normal, ${w}, [-1.0, 2.0], sigma = [1.0, 0.5])`;
    const density = pdf(0.5, -1, 1) + pdf(0.5, 2, 0.5);
    close(await value(src, 'totalmass(m)'), 2);
    close(await value(src, 'logdensityof(m, 0.5)'), Math.log(density));
    close(await value(src, 'logdensityof(normalize(m), 0.5)'), Math.log(density / 2));
    assert.notEqual(processSource(src).bindings.get('m').inferredType.mass, 'normalized');
  }
});

test('multiple broadcast axes expand both weights and parameter singleton axes', async () => {
  const src = `
w = rowstack([[0.25], [0.75]])
mu = rowstack([[-2.0, 1.0, 3.0]])
s = rowstack([[0.5], [2.0]])
m = ksuperpose(Normal, w, mu, sigma = s)
`;
  const density = [-2, 1, 3].reduce((sum, mu) =>
    sum + 0.25 * pdf(0.4, mu, 0.5) + 0.75 * pdf(0.4, mu, 2), 0);
  close(await value(src, 'totalmass(m)'), 3);
  close(await value(src, 'logdensityof(normalize(m), 0.4)'), Math.log(density / 3));
});

test('tables supply rows with ordinary unary and auto-splat interfaces', async () => {
  const density = (pdf(0.5, -1, 1) + pdf(0.5, 2, 0.5)) / 2;
  for (const [kernel, definition] of [
    ['Normal', ''],
    ['K', 'K = r -> Normal(r.mu, r.sigma)\n'],
    ['K', 'K(a, b) = Normal(a, b)\n'],
  ]) {
    const table = kernel === 'K' && definition.includes('K(a')
      ? 'table(a = [-1.0, 2.0], b = [1.0, 0.5])'
      : 'table(mu = [-1.0, 2.0], sigma = [1.0, 0.5])';
    const src = `${definition}t = ${table}\nm = ksuperpose(${kernel}, [1.0], t)`;
    close(await value(src, 'logdensityof(normalize(m), 0.5)'), Math.log(density));
  }
});

test('runtime broadcast repeats weights and table rows through existing mixture paths', async () => {
  const src = `
w = reverse([1.0])
t = table(mu = [-1.0, 2.0], sigma = [1.0, 0.5])
m = ksuperpose(Normal, w, t)
`;
  close(await value(src, 'totalmass(m)'), 2);
  close(await value(src, 'logdensityof(normalize(m), 0.5)'),
    Math.log((pdf(0.5, -1, 1) + pdf(0.5, 2, 0.5)) / 2));
});

test('simplex recognition is retained only when weights do not expand', () => {
  const base = 'p = elementof(stdsimplex(1))\n';
  assert.equal(processSource(base + 'm = ksuperpose(Normal, p, mu = [0.0], sigma = 1.0)')
    .bindings.get('m').inferredType.mass, 'normalized');
  assert.notEqual(processSource(base + 'm = ksuperpose(Normal, p, mu = [0.0, 2.0], sigma = 1.0)')
    .bindings.get('m').inferredType.mass, 'normalized');
});

test('broadcast accepts positional kernel parameters before named parameters', async () => {
  const src = `f(a, b) = a + 2.0 * b
x = broadcast(f, [1.0, 2.0], b = [3.0])
y = broadcast(add, [1.0, 2.0], b = [3.0])
`;
  close(await value(src, 'sum(x)'), 15);
  close(await value(src, 'sum(y)'), 9);
});

test('an inline kernel binds mixed component parameters by its ordinary interface', async () => {
  const src = 'm = ksuperpose((a, b) -> Normal(a, b), [1.0], [-1.0, 2.0], b = 0.5)';
  close(await value(src, 'logdensityof(normalize(m), 0.5)'),
    Math.log((pdf(0.5, -1, 0.5) + pdf(0.5, 2, 0.5)) / 2));
});

test('mixed distribution broadcasts keep both parameter sources when scoring', async () => {
  const src = 'm = broadcast(Normal, [1.0, 2.0], sigma = [0.5])';
  close(await value(src, 'logdensityof(m, [0.0, 0.0])'),
    Math.log(pdf(0, 1, 0.5)) + Math.log(pdf(0, 2, 0.5)));
});

test('expanded weights select components in proportion to their repeated masses', async () => {
  const { ctx } = ctxFor('m = ksuperpose(Dirac, [1.0], value = [0.0, 1.0, 2.0])\ny ~ normalize(m)', 10000);
  const samples = (await ctx.getMeasure('y')).samples;
  const counts = [0, 0, 0];
  for (const sample of samples) counts[sample]++;
  // Each count has standard deviation sqrt(N*(1/3)*(2/3)) = 47.2.
  for (const count of counts) assert.ok(Math.abs(count / 10000 - 1 / 3) < 0.025);
});

test('runtime shapes retain multiple outer axes and repeat weights per axis', async () => {
  const src = `
w = addaxes(reverse([0.25, 0.75]), 0, 1)
mu = rowstack([[-2.0, 1.0, 3.0]])
m = ksuperpose(Normal, w, mu = mu, sigma = 1.0)
`;
  close(await value(src, 'totalmass(m)'), 3);
  close(await value(src, 'logdensityof(normalize(m), 0.4)'),
    Math.log([-2, 1, 3].reduce((sum, mu) => sum + pdf(0.4, mu, 1), 0) / 3));
});

test('empty broadcast mixtures have zero mass and minus-infinite log-density', async () => {
  const empty = 'empty = filter(x -> false, [1.0])\nwgrid = addaxes(empty, 0, 1)\nmeansgrid = rowstack([[0.0, 1.0]])\n';
  for (const mixture of [
    'ksuperpose(Normal, [1.0], mu = empty, sigma = 1.0)',
    'ksuperpose(x -> Normal(x, 1.0), [1.0], empty)',
    'ksuperpose(x -> Normal(x, 1.0), [1.0], x = empty)',
    'ksuperpose(Normal, empty, mu = 0.0, sigma = 1.0)',
    'ksuperpose(Normal, wgrid, mu = meansgrid, sigma = 1.0)',
  ]) {
    const src = empty + 'm = ' + mixture;
    close(await value(src, 'totalmass(m)'), 0);
    assert.equal(await value(src, 'logdensityof(m, 0.4)'), -Infinity);
    assert.equal(await value(src, 'logdensityof(joint(m, Normal(0.0, 1.0)), [0.4, 0.2])'), -Infinity);
    await assert.rejects(() => value(src, 'logdensityof(normalize(m), 0.4)'), /ZERO|zero.*mass/i);
    const { ctx } = ctxFor(src, 4);
    await assert.rejects(async () => ctx.getMeasure('m'), /zero measure.*sampling.*undefined/i);
  }
});
