'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index');

function errors(src) {
  return processSource(src).diagnostics.filter(d => d.severity === 'error');
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
  assert.ok(errs.some(d => /1-based|index/.test(d.message)));
});

test('indexing: matrix M[0, 1] is an error', () => {
  const errs = errors('M = rowstack([[1,2],[3,4]])\nx = M[0, 1]\n');
  assert.ok(errs.some(d => /1-based|index/.test(d.message)));
});

test('indexing: slice with zero index A[:, 0] is an error', () => {
  const errs = errors('A = rowstack([[1,2],[3,4]])\nx = A[:, 0]\n');
  assert.ok(errs.some(d => /1-based|index/.test(d.message)));
});

// --- Invalid: negative index ---

test('indexing: x[-1] is an error', () => {
  const errs = errors('a = [1, 2, 3]\nb = a[-1]\n');
  assert.ok(errs.some(d => /1-based|index/.test(d.message)));
});

test('indexing: x[-3] is an error', () => {
  const errs = errors('a = [1, 2, 3]\nb = a[-3]\n');
  assert.ok(errs.some(d => /1-based|index/.test(d.message)));
});

// --- Edge cases ---

test('indexing: nested IndexExpr — both checked', () => {
  // a[b[0]] — outer index `b[0]` is runtime; inner `0` is the literal violator
  const errs = errors('b = [1, 2]\na = [3, 4]\nx = a[b[0]]\n');
  assert.ok(errs.some(d => /1-based|index/.test(d.message)));
});

test('indexing: integer-valued real (e.g. 0.0) is also flagged', () => {
  // 0.0 is parsed as a NumberLiteral(0); Number.isInteger(0) is true.
  const errs = errors('a = [1, 2, 3]\nb = a[0.0]\n');
  assert.ok(errs.some(d => /1-based|index/.test(d.message)));
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
    assert.equal(errs.length, 0, `${f} produced errors: ${JSON.stringify(errs.map(d => d.message))}`);
  }
});
