'use strict';

// §07 splitblocks: "the outer-array shape is the elementwise quotient", and
// joinblocks: "The outer and inner arrays must have the same number of
// dimensions". So splitblocks of a rank-r array is a rank-r outer array of
// rank-r blocks. A 1-D array still gives a vector of vectors. §04 broadcasting
// iterates every outer axis and passes each block whole.
//
// ORACLE: closed form for the capture mixture, per block omega times the
// Bernoulli product plus (1 - omega) times a Dirac at all-false.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

function fixed(src: string, name: string): any {
  const r = processSource(src);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  return orchestrator.buildDerivations(r.bindings).fixedValues.get(name);
}

test('a 1-D splitblocks is a vector of vectors', () => {
  const v = fixed('b = splitblocks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)\n', 'b');
  assert.deepEqual(v.shape, [3, 2]);
  assert.equal(v.outerRank, 1);
  assert.deepEqual(Array.from(v.data), [1, 2, 3, 4, 5, 6]);
});

test('a 2-D splitblocks is a 2-D outer array of 2-D blocks', () => {
  const v = fixed('b = splitblocks(rowstack([[1.0, 2.0], [3.0, 4.0]]), [1, 2])\n', 'b');
  assert.deepEqual(v.shape, [2, 1, 1, 2]);
  assert.equal(v.outerRank, 2);
  assert.deepEqual(Array.from(v.data), [1, 2, 3, 4]);
});

test('joinblocks inverts a 2-D splitblocks', () => {
  const A = 'rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])';
  for (const b of ['[1, 3]', '[2, 1]', '[2, 3]', '[4, 1]']) {
    const v = fixed(`j = joinblocks(splitblocks(${A}, ${b}))\n`, 'j');
    assert.deepEqual(v.shape, [4, 3], b);
    assert.deepEqual(Array.from(v.data), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], b);
  }
});

test('a kernel broadcast over splitblocks passes each block whole', async () => {
  // testsuite's capture mixture over matrix cells, at omega = 0.2, 0.6, 0.9.
  const lp = [[0.2, -0.4], [1.0, 0.5]];
  const y = [[true, false], [false, false]];
  const sig = (x: number) => 1 / (1 + Math.exp(-x));
  for (const omega of [0.2, 0.6, 0.9]) {
    const want = lp.reduce((s, row, i) => {
      const product = row.reduce((p, x, j) => p * (y[i][j] ? sig(x) : 1 - sig(x)), 1);
      const dirac = y[i].every((v) => !v) ? 1 : 0;
      return s + Math.log(omega * product + (1 - omega) * dirac);
    }, 0);
    const src = 'capture(row, w) = normalize(superpose(weighted(w, Bernoulli.(invlogit.(row))), '
      + 'weighted(1 - w, Dirac(fill(false, [1, 2])))))\n'
      + `lp = logdensityof(capture.(splitblocks(rowstack([[0.2, -0.4], [1.0, 0.5]]), [1, 2]), ${omega}), `
      + 'splitblocks(rowstack([[true, false], [false, false]]), [1, 2]))\n';
    const { ctx } = ctxFor(src, 1);
    const m = await ctx.getMeasure('lp');
    const got = (m.value ? m.value.data : m.samples)[0];
    assert.ok(Math.abs(got - want) < 1e-12, `omega ${omega}: got ${got}, want ${want}`);
  }
});

test('a latent-dependent splitblocks keeps its blocks at every atom', async () => {
  // The F17 repro: omega ~ Uniform, observed at omega = 0.6.
  const src = [
    'omega ~ Uniform(unitinterval)',
    'lp = rowstack([[0.2, -0.4], [1.0, 0.5]])',
    'capture(row, omega) = normalize(superpose(',
    '    weighted(omega, Bernoulli.(invlogit.(row))), weighted(1 - omega, Dirac(fill(false, [1, 2])))))',
    'y ~ capture.(splitblocks(lp, [1, 2]), omega)',
    'L = likelihoodof(kernelof(record(y = y), omega = omega), record(y = splitblocks(rowstack([[1, 0], [0, 0]]), [1, 2])))',
    'P = bayesupdate(L, lawof(record(omega = omega)))',
    'score = logdensityof(P, record(omega = 0.6))',
  ].join('\n') + '\n';
  for (const N of [1, 3]) {
    const { ctx } = ctxFor(src, N);
    const m = await ctx.getMeasure('score');
    for (const g of Array.from(m.value ? m.value.data : m.samples) as number[]) {
      assert.ok(Math.abs(g - -2.3965066391888614) < 1e-12, `N ${N}: got ${g}`);
    }
  }
});
