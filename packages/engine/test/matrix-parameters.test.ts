'use strict';

// A vector of vectors is not a matrix argument.
//
// §08 declares `MvNormal`'s `cov` and the `scale` of `Wishart` and
// `InverseWishart` as n × n matrices. §03 "Arrays": "Vectors of vectors are not
// interpreted as matrices implicitly, but can be turned into matrices explicitly
// using `rowstack` or `colstack`". These calls used to type cleanly and fail only
// when a fixed value was evaluated, or never, on paths that do not evaluate it.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

function matrixErrors(src: string): any[] {
  return (processSource(src).diagnostics || []).filter((d: any) =>
    d.severity === 'error' && /must be a matrix \(spec §08\)/.test(d.message));
}

function assertRefuses(src: string, dist: string, param: string) {
  const errors = matrixErrors(src);
  assert.equal(errors.length, 1, src + '\n' + JSON.stringify(errors));
  assert.ok(errors[0].message.startsWith('`' + dist + "`'s `" + param + '` must be a matrix'),
    errors[0].message);
  assert.match(errors[0].message, /rowstack\(\.\.\.\)/);
  assert.ok(errors[0].loc, 'the diagnostic is located');
}

const POINT = 'lp = logdensityof(m, [0.7, -1.2])\n';

test('a nested covariance literal is a static error', () => {
  assertRefuses('m = MvNormal(mu = [0.2, -0.1], cov = [[1.0, 0.3], [0.3, 2.0]])\n' + POINT,
    'MvNormal', 'cov');
  assertRefuses('m = MvNormal([0.2, -0.1], [[1.0, 0.3], [0.3, 2.0]])\n' + POINT,
    'MvNormal', 'cov');
  assertRefuses('c = [[1.0, 0.3], [0.3, 2.0]]\nm = MvNormal(mu = [0.2, -0.1], cov = c)\n' + POINT,
    'MvNormal', 'cov');
  // An inline constructor is reported once, though inference reaches it twice.
  assertRefuses('lp = logdensityof(MvNormal(mu = [0.2, -0.1], cov = [[1.0, 0.3], [0.3, 2.0]]), '
    + '[0.7, -1.2])\n', 'MvNormal', 'cov');
});

test('a nested scale literal is a static error', () => {
  for (const dist of ['Wishart', 'InverseWishart']) {
    assertRefuses(`w = ${dist}(nu = 4.0, scale = [[1.0, 0.0], [0.0, 1.0]])\n`
      + 'lp = logdensityof(w, rowstack([[1.0, 0.2], [0.2, 1.0]]))\n', dist, 'scale');
  }
});

test('a rowstack or colstack matrix types', () => {
  for (const build of ['rowstack', 'colstack']) {
    assert.deepEqual(matrixErrors(
      `m = MvNormal(mu = [0.2, -0.1], cov = ${build}([[1.0, 0.3], [0.3, 2.0]]))\n` + POINT), []);
  }
  assert.deepEqual(matrixErrors('w = Wishart(nu = 4.0, scale = rowstack([[1.0, 0.0], [0.0, 1.0]]))\n'
    + 'lp = logdensityof(w, rowstack([[1.0, 0.2], [0.2, 1.0]]))\n'), []);
});
