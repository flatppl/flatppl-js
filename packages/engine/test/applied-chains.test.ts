'use strict';

// Spec §04 "Additional arguments", §06 kchain / jointchain: an applied kernel-first chain
// `kchain(K1, K2, ...)(x, ...)` passes the additional arguments unchanged to every kernel, each
// later kernel also receiving the fed value whole as its first argument. Every expected value
// is the closed-form log density of the Gaussian chain.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const logN = (x: number, mean: number, sd: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(sd) - 0.5 * ((x - mean) / sd) ** 2;

async function value(src: string, name: string, n = 1): Promise<number[]> {
  const { ctx } = ctxFor(src, n);
  const m = await ctx.getMeasure(name);
  return Array.from(m.value ? m.value.data : m.samples);
}

function errors(src: string): string[] {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

function near(got: number[], want: number[], tol = 1e-12) {
  assert.equal(got.length, want.length);
  want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) < tol, `got ${got[i]}, want ${w}`));
}

// K1(x, p) ~ Normal(x + p, 1); K2(a, p) ~ Normal(a p, 1); K3(ab, p) ~ Normal(sum(ab) p, 1).
const KERNELS = 'K1(x, p) = Normal(x + p, 1.0)\nK2(a, p) = Normal(a * p, 1.0)\n'
  + 'K3(ab, p) = Normal(sum(ab) * p, 1.0)\n';

test('kchain applied with additional arguments marginalizes the intermediate variate', async () => {
  // x1 ~ N(0 + 2, 1), y ~ N(2 x1, 1) = N(4, sqrt 5).
  const want = [logN(0.9, 4, Math.sqrt(5))];
  near(await value(`${KERNELS}C = kchain(K1, K2)\nM = C(0.0, 2.0)\nlp = logdensityof(M, 0.9)\n`, 'lp'), want);
  near(await value(`${KERNELS}C = kchain(K1, K2)\nM = C(x = 0.0, p = 2.0)\nlp = logdensityof(M, 0.9)\n`, 'lp'), want);
  near(await value(`${KERNELS}M = kchain(K1, K2)(0.0, 2.0)\nlp = logdensityof(M, 0.9)\n`, 'lp'), want);
});

test('jointchain applied with additional arguments retains every variate', async () => {
  const want = [logN(1.5, 2, 1) + logN(0.9, 3, 1)];
  near(await value(`${KERNELS}C = jointchain(K1, K2)\nM = C(0.0, 2.0)\nlp = logdensityof(M, [1.5, 0.9])\n`, 'lp'), want);
  near(await value(`${KERNELS}C = jointchain(K1, K2)\nM = C(p = 2.0, x = 0.0)\nlp = logdensityof(M, [1.5, 0.9])\n`, 'lp'), want);
});

test('a third kernel receives the accumulated variates and the same additional arguments', async () => {
  // jointchain: x1 ~ N(2, 1), x2 ~ N(2 x1, 1), y ~ N(2 (x1 + x2), 1).
  near(await value(`${KERNELS}C = jointchain(K1, K2, K3)\nM = C(0.0, 2.0)\nlp = logdensityof(M, [1.5, 0.9, 3.0])\n`, 'lp'),
    [logN(1.5, 2, 1) + logN(0.9, 3, 1) + logN(3.0, (1.5 + 0.9) * 2, 1)]);
  // kchain marginalizes: y = 6 x1 + 2 e2 + e3 with x1 ~ N(2, 1), so y ~ N(12, sqrt 41).
  near(await value(`${KERNELS}C = kchain(K1, K2, K3)\nM = C(0.0, 2.0)\nlp = logdensityof(M, 3.0)\n`, 'lp'),
    [logN(3.0, 12, Math.sqrt(41))]);
});

test('an applied chain without additional arguments keeps its meaning', async () => {
  const src = 'K1(x) = Normal(x + 1.0, 1.0)\nK2(a) = Normal(a * 2.0, 1.0)\n';
  // x1 ~ N(1, 1), y ~ N(2 x1, 1) = N(2, sqrt 5).
  near(await value(`${src}C = kchain(K1, K2)\nM = C(x = 0.0)\nlp = logdensityof(M, 0.9)\n`, 'lp'),
    [logN(0.9, 2, Math.sqrt(5))]);
  near(await value(`${src}C = jointchain(K1, K2)\nM = C(0.0)\nlp = logdensityof(M, [1.5, 0.9])\n`, 'lp'),
    [logN(1.5, 1, 1) + logN(0.9, 3, 1)]);
});

test('an applied chain samples from the chain law', async () => {
  const draws = await value(`${KERNELS}C = kchain(K1, K2)\nM = C(0.0, 2.0)\ny ~ M\n`, 'y', 4000);
  const mean = draws.reduce((a, b) => a + b, 0) / draws.length;
  const variance = draws.reduce((a, b) => a + (b - mean) ** 2, 0) / draws.length;
  // N(4, 5): the standard error of the mean over 4000 draws is 0.035.
  assert.ok(Math.abs(mean - 4) < 0.2, `mean ${mean}`);
  assert.ok(Math.abs(variance - 5) < 0.5, `variance ${variance}`);
});

test('a later step with a different number of inputs than the first is a static error', () => {
  // Every step receives the same additional arguments, so K2 would need the input `p` too.
  const kernels = 'K1(x, p) = Normal(x + p, 1.0)\nK2(a) = Normal(a, 1.0)\n';
  for (const op of ['kchain', 'jointchain']) {
    const found = errors(`${kernels}C = ${op}(K1, K2)\nM = C(0.0, 2.0)\n`);
    assert.ok(found.some(m => /the step takes 1 inputs, but the first step takes 2/.test(m)), op + ': ' + found);
  }
  // The same holds for fchain, where the extra argument was silently dropped.
  const f = errors('a1(x, p) = x + p\na2(y) = y * 2.0\nF = fchain(a1, a2)\nr = F(1.0, 2.0)\n');
  assert.ok(f.some(m => /the step takes 1 inputs, but the first step takes 2/.test(m)), String(f));
});
