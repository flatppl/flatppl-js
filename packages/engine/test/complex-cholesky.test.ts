'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

for (const { name, binding, samples } of [
  { name: 'fixed', binding: 't = 1.0', samples: 1 },
  { name: 'sampled', binding: 't ~ Bernoulli(0.5)', samples: 8 },
]) {
  test(`lower_cholesky reconstructs a ${name} Hermitian matrix`, async () => {
    const { ctx } = makeMatCtx(binding + `
A = rowstack([[4.0, complex(2.0, -2*t), complex(4.0, 2.0)],
              [complex(2.0, 2*t), 10.0+t*t, complex(-1.0-t, -5.0+2*t)],
              [complex(4.0, -2.0), complex(-1.0-t, 5.0-2*t), 26.0]])
L = lower_cholesky(A)
reconstructed = L * adjoint(L)
`, { sampleCount: samples, rootSeed: 117 });
    const [t, A, L, reconstructed] = await Promise.all(
      ['t', 'A', 'L', 'reconstructed'].map(key => ctx.getMeasure(key)));
    if (samples > 1) assert.ok(t.samples.includes(0) && t.samples.includes(1));
    assert.equal(L.samples.length, samples * 9);
    for (let i = 0; i < samples; i++) {
      const start = i * 9, end = start + 9;
      assert.deepEqual(Array.from(L.samples.slice(start, end)), [2, 0, 0, 1, 3, 0, 2, -1, 4]);
      assert.deepEqual(Array.from(L.imag.slice(start, end)), [0, 0, 0, t.samples[i], 0, 0, -1, 2, 0]);
    }
    assert.deepEqual(reconstructed.samples, A.samples);
    assert.ok(reconstructed.imag.every((x: number, i: number) => x === A.imag[i]));
  });
}

test('lower_cholesky rejects an indefinite Hermitian matrix', async () => {
  const { ctx } = makeMatCtx(`
A = rowstack([[1.0, im], [-im, 0.0]])
L = lower_cholesky(A)
`, { sampleCount: 1 });
  await assert.rejects(ctx.getMeasure('L'), /matrix is not positive definite/);
});
