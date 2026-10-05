'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const { toJS } = require('./_value-helpers.ts');

function errors(src: any) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

for (const { name, source, expected } of [
  {
    name: 'nested arrays',
    source: 'A = [[[0.0,0.0],[0.0,0.0]], [[0.0,1.0],[1.0,0.0]]]\ny = sum(exp.(sum.(A[i])))',
    expected: [2, 2 * Math.E],
  },
  {
    name: 'complex arrays',
    source: 'A = complex.(rowstack([[1,3],[5,7]]), rowstack([[2,4],[6,8]]))\ny = sum(imag.(A[i])) + imag(A[i,1])',
    expected: [8, 20],
  },
  {
    name: 'complex views',
    source: 'A = complex.(rowstack([[1,3,5],[7,9,11]]), rowstack([[2,4,6],[8,10,12]]))\n'
      + 'y = sum(imag.(transpose(A)[i])) + imag(conj.(A)[i,1]) + imag(adjoint(A[1])[i])',
    expected: [6, 2],
  },
]) {
  test(`indexing: dynamic element access preserves ${name}`, () => {
    const proc = processSource(`${source}\ni = elementof(integers)\nf = functionof(y, i = i)\nz = f.(i = [1,2])\n`);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
    const value = buildDerivations(proc.bindings).fixedValues.get('z');
    assert.deepEqual(toJS(value), expected);
  });
}

// --- Valid 1-based indices ---

test('indexing: x[1] is valid', () => {
  assert.equal(errors('a = [1, 2, 3]\nb = a[1]\n').length, 0);
});

test('indexing: matrix M[i, j] with positive literals is valid', () => {
  assert.equal(errors('M = rowstack([[1,2],[3,4]])\nx = M[1, 2]\n').length, 0);
});

test('indexing: slice A[:, 1] is valid', () => {
  assert.equal(errors('A = rowstack([[1,2,3],[4,5,6]])\nx = A[:, 1]\n').length, 0);
});

test('indexing: runtime expression x[i] is allowed (not a literal)', () => {
  assert.equal(errors('a = [1, 2, 3]\ni = elementof(posintegers)\nb = a[i]\n').length, 0);
});

// --- Invalid: zero index ---

test('indexing: x[0] is an error', () => {
  const errs = errors('a = [1, 2, 3]\nb = a[0]\n');
  assert.ok(errs.some((d: any) => /1-based|index/.test(d.message)));
});

test('indexing: matrix M[0, 1] is an error', () => {
  const errs = errors('M = rowstack([[1,2],[3,4]])\nx = M[0, 1]\n');
  assert.ok(errs.some((d: any) => /1-based|index/.test(d.message)));
});

test('indexing: slice with zero index A[:, 0] is an error', () => {
  const errs = errors('A = rowstack([[1,2],[3,4]])\nx = A[:, 0]\n');
  assert.ok(errs.some((d: any) => /1-based|index/.test(d.message)));
});

// --- Invalid: negative index ---

test('indexing: x[-1] is an error', () => {
  const errs = errors('a = [1, 2, 3]\nb = a[-1]\n');
  assert.ok(errs.some((d: any) => /1-based|index/.test(d.message)));
});

test('indexing: x[-3] is an error', () => {
  const errs = errors('a = [1, 2, 3]\nb = a[-3]\n');
  assert.ok(errs.some((d: any) => /1-based|index/.test(d.message)));
});

// --- Edge cases ---

test('indexing: nested IndexExpr — both checked', () => {
  // a[b[0]] — outer index `b[0]` is runtime; inner `0` is the literal violator
  const errs = errors('b = [1, 2]\na = [3, 4]\nx = a[b[0]]\n');
  assert.ok(errs.some((d: any) => /1-based|index/.test(d.message)));
});

test('indexing: integer-valued real (e.g. 0.0) is also flagged', () => {
  // 0.0 is parsed as a NumberLiteral(0); Number.isInteger(0) is true.
  const errs = errors('a = [1, 2, 3]\nb = a[0.0]\n');
  assert.ok(errs.some((d: any) => /1-based|index/.test(d.message)));
});

test('indexing: existing fixture files still parse cleanly', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const fixtures = [
    'bayesian_inference_1.flatppl',
    'bayesian_inference_2.flatppl',
    'flatppl-uncorrelated_background-ma-auxm.flatppl',
    'flatppl-uncorrelated_background-ma-priors.flatppl',
    'flatppl-uncorrelated_background-draws-auxm.flatppl',
    'flatppl-uncorrelated_background-draws-priors.flatppl',
    'disintegrate-complex.flatppl',
  ];
  for (const f of fixtures) {
    const src = fs.readFileSync(path.join(__dirname, 'fixtures', f), 'utf8');
    const errs = errors(src);
    assert.equal(errs.length, 0, `${f} produced errors: ${JSON.stringify(errs.map((d: any) => d.message))}`);
  }
});
