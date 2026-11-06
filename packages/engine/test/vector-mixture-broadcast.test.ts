'use strict';

// §06 normalize divides by the total mass. A §04 kernel broadcast of a
// built-in distribution is a product of probability measures, so its mass is
// 1, and normalize(M) has mass 1. Broadcasting iterates only the collection
// axes, so a cell of a vector of vectors, or a joint or iid variate, reaches
// the kernel whole. §08 Bernoulli variates are booleans.
//
// ORACLES are closed-form mixture probabilities.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function value(src: string, N = 1): Promise<number[]> {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('lp');
  return Array.from(m.value ? m.value.data : m.samples);
}

function near(got: number[], want: number, tol = 1e-12) {
  assert.ok(got.length > 0);
  got.forEach((g) => assert.ok(Math.abs(g - want) < tol, `got ${g}, want ${want}`));
}

test('a zero-inflated Bernoulli vector has a parameter-dependent mixture of unit masses', async () => {
  // y = [[true, false], [false, false]] at omega = 0.6, p = [0.3, 0.7].
  const src = [
    'omega ~ Uniform(unitinterval)',
    'p ~ iid(Uniform(unitinterval), 2)',
    'ch = normalize(superpose(weighted(omega, Bernoulli.(p)), weighted(1 - omega, Dirac(fill(false, 2)))))',
    'y ~ iid(ch, 2)',
    'L = likelihoodof(kernelof(record(y = y), omega = omega, p = p), record(y = [[true, false], [false, false]]))',
    'P = bayesupdate(L, lawof(record(omega = omega, p = p)))',
    'lp = logdensityof(P, record(omega = 0.6, p = [0.3, 0.7]))',
  ].join('\n') + '\n';
  const want = Math.log(0.6 * 0.3 * 0.3) + Math.log(0.6 * 0.7 * 0.3 + 0.4);
  near(await value(src), want);
  near(await value(src, 3), want);
});

test('a fixed Bernoulli broadcast scores boolean outcomes', async () => {
  near(await value('lp = logdensityof(Bernoulli.([0.2, 0.3]), [true, false])\n'), Math.log(0.2) + Math.log(0.7));
});

test('a kernel broadcast over a vector of vectors passes each inner vector whole', async () => {
  const src = [
    'cap(p) = normalize(superpose(weighted(0.5, Bernoulli.(p)), weighted(0.5, Dirac(fill(false, 2)))))',
    'lp = logdensityof(cap.([[0.2, 0.3], [0.4, 0.5]]), [[true, false], [false, false]])',
  ].join('\n') + '\n';
  near(await value(src), Math.log(0.5 * 0.2 * 0.7) + Math.log(0.5 * 0.6 * 0.5 + 0.5));
});

test('a kernel broadcast scores a joint-valued cell whole', async () => {
  const src = [
    'em(w) = normalize(superpose(weighted(w, joint(Exponential(1), Exponential(2))),',
    '    weighted(1 - w, joint(Exponential(3), Exponential(4)))))',
    'lp = logdensityof(em.([0.3, 0.6]), [[0.5, 0.2], [0.1, 0.4]])',
  ].join('\n') + '\n';
  const e = (r: number, x: number) => r * Math.exp(-r * x);
  const mix = (w: number, x: number, y: number) => w * e(1, x) * e(2, y) + (1 - w) * e(3, x) * e(4, y);
  near(await value(src), Math.log(mix(0.3, 0.5, 0.2)) + Math.log(mix(0.6, 0.1, 0.4)));
});

test('a nested zero-inflated kernel resolves the inner normalizer of each cell', async () => {
  // multi_occupancy's shape: species(psi) = normalize(superpose(omega iid(site(psi), 2), Dirac zeros)).
  const src = [
    'site(psi) = normalize(superpose(weighted(psi, Binomial(3, 0.4)), weighted(1 - psi, Dirac(0))))',
    'species(psi, omega) = normalize(superpose(weighted(omega, iid(site(psi), 2)), weighted(1 - omega, Dirac(fill(0, 2)))))',
    'lp = logdensityof(species.([0.3, 0.6], 0.5), [[0, 2], [0, 0]])',
  ].join('\n') + '\n';
  const b = (k: number) => [1, 3, 3, 1][k] * 0.4 ** k * 0.6 ** (3 - k);
  const site = (psi: number, k: number) => psi * b(k) + (1 - psi) * (k === 0 ? 1 : 0);
  const want = Math.log(0.5 * site(0.3, 0) * site(0.3, 2)) + Math.log(0.5 * site(0.6, 0) ** 2 + 0.5);
  near(await value(src), want);
});
