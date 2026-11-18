'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

for (const { name, binding, sampleCount } of [
  { name: 'fixed', binding: 't = 1.0', sampleCount: 1 },
  { name: 'sampled', binding: 't ~ Bernoulli(0.5)', sampleCount: 8 },
]) {
  test(`aggregate preserves complex values in ${name} reductions`, async () => {
    const { ctx } = makeMatCtx(binding + `
v = [complex(1.0, t + 1.0), complex(3.0, 4.0)]
m = aggregate(mean, [], v[.i])
s = aggregate(sum, [], conj(v[.i]) * complex(t, 2.0))
p = aggregate(prod, [], v[.i])
A = rowstack([v, v])
columns = aggregate(mean, [.j], A[.i, .j])
swapped = aggregate(sum, [.i], complex(imag(v[.i]), real(v[.i])))
`, { sampleCount, rootSeed: 117 });
    const [draws, m, s, p, columns, swapped] = await Promise.all(
      ['t', 'm', 's', 'p', 'columns', 'swapped'].map(key => ctx.getMeasure(key)));
    if (sampleCount > 1) assert.ok(draws.samples.includes(0) && draws.samples.includes(1));
    for (let i = 0; i < sampleCount; i++) {
      const t = draws.samples[i], y = t + 1;
      assert.equal(m.samples[i], 2);
      assert.equal(m.imag[i], (y + 4) / 2);
      assert.equal(s.samples[i], 4 * t + 2 * (y + 4));
      assert.equal(s.imag[i], 8 - t * (y + 4));
      assert.equal(p.samples[i], 3 - 4 * y);
      assert.equal(p.imag[i], 4 + 3 * y);
      assert.deepEqual(Array.from(columns.samples.subarray(i * 2, i * 2 + 2)), [1, 3]);
      assert.deepEqual(Array.from(columns.imag.subarray(i * 2, i * 2 + 2)), [y, 4]);
      assert.deepEqual(Array.from(swapped.samples.subarray(i * 2, i * 2 + 2)), [y, 4]);
      assert.deepEqual(Array.from(swapped.imag.subarray(i * 2, i * 2 + 2)), [1, 3]);
    }
  });
}
