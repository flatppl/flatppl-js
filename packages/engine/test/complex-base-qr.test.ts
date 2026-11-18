'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function close(values: Float64Array, expected: number[], scale = 1) {
  assert.equal(values.length, expected.length);
  expected.forEach((value, i) => assert.ok(
    Math.abs(values[i] / scale - value) <= 32 * Number.EPSILON * Math.max(1, Math.abs(value))));
}

test('thin complex qr reconstructs a tall matrix with orthonormal columns', async () => {
  const { ctx } = makeMatCtx(`
A = rowstack([[complex(1, 1), im], [2.0, complex(1, -1)], [im, 2.0]])
f = qr(A)
Q = f.Q
R = f.R
reconstructed = Q * R
orthogonal = adjoint(Q) * Q
`, { sampleCount: 1 });
  const [q, r, reconstructed, orthogonal] = await Promise.all(
    ['Q', 'R', 'reconstructed', 'orthogonal'].map(key => ctx.getMeasure(key)));
  assert.deepEqual(q.intrinsicShape, [3, 2]);
  assert.deepEqual(r.intrinsicShape, [2, 2]);
  assert.equal(r.samples[2], 0);
  assert.equal(r.imag[2], 0);
  close(reconstructed.samples, [1, 0, 2, 1, 0, 2]);
  close(reconstructed.imag, [1, 1, 0, -1, 1, 0]);
  close(orthogonal.samples, [1, 0, 0, 1]);
  close(orthogonal.imag, [0, 0, 0, 0]);
});

test('complex qr keeps the basis with a zero column and a scaled column', async () => {
  const { ctx } = makeMatCtx(`
A = 1e200 * rowstack([[0.0, complex(1, 1)], [0.0, im], [0.0, complex(2, -1)]])
f = qr(A)
Q = f.Q
R = f.R
reconstructed = Q * R
orthogonal = adjoint(Q) * Q
`, { sampleCount: 1 });
  const [reconstructed, orthogonal] = await Promise.all(
    ['reconstructed', 'orthogonal'].map(key => ctx.getMeasure(key)));
  close(reconstructed.samples, [0, 1, 0, 0, 0, 2], 1e200);
  close(reconstructed.imag, [0, 1, 0, 1, 0, -1], 1e200);
  close(orthogonal.samples, [1, 0, 0, 1]);
  close(orthogonal.imag, [0, 0, 0, 0]);
});
