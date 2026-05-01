'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index');

function errors(src) {
  return processSource(src).diagnostics.filter(d => d.severity === 'error');
}

// --- Hole (_) ---

test('hole: valid inside fn(...)', () => {
  assert.equal(errors('f = fn(_ * 2)\n').length, 0);
});

test('hole: multiple holes inside fn(...) all valid', () => {
  assert.equal(errors('f = fn(_ + _ * _)\n').length, 0);
});

test('hole: error when used outside fn', () => {
  const errs = errors('x = _ + 1\n');
  assert.ok(errs.some(d => /Hole.*fn/.test(d.message)));
});

test('hole: error when used inside functionof', () => {
  // _ is hole-only-in-fn; placeholder _x_ is for functionof
  const errs = errors('f = functionof(_ * 2, x = _x_)\n');
  assert.ok(errs.some(d => /Hole.*fn/.test(d.message)));
});

test('hole: error when used inside kernelof', () => {
  const errs = errors('m = kernelof(_, x = _x_)\n');
  assert.ok(errs.some(d => /Hole.*fn/.test(d.message)));
});

test('hole: nested fn — inner fn redefines the hole scope', () => {
  // fn(...) inside fn(...) — both _ are valid
  assert.equal(errors('f = fn(fn(_ + 1)(_))\n').length, 0);
});

// --- Placeholder (_name_) ---

test('placeholder: valid inside functionof', () => {
  assert.equal(errors('f = functionof(_par_ * 2, par = _par_)\n').length, 0);
});

test('placeholder: valid inside kernelof', () => {
  assert.equal(errors('m = kernelof(_x_, x = _x_)\n').length, 0);
});

test('placeholder: error when used outside reification', () => {
  const errs = errors('x = _par_ * 2\n');
  assert.ok(errs.some(d => /Placeholder.*functionof.*kernelof/.test(d.message)));
});

test('placeholder: error when used inside fn', () => {
  // fn allows holes only, not placeholders.
  const errs = errors('f = fn(_par_ + _)\n');
  assert.ok(errs.some(d => /Placeholder.*functionof.*kernelof/.test(d.message)));
});

// --- LHS underscore is fine (not a hole or placeholder per parser) ---

test('decomposition with bare _ on LHS is not flagged', () => {
  // The LHS _ is a Name (discard binding), not a hole reference
  assert.equal(errors('value, _ = (1, 2)\n').length, 0);
});

// --- Real-world examples should still pass ---

test('bayesian_inference_2 fixture parses cleanly', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const file = path.join(__dirname, 'fixtures', 'bayesian_inference_2.flatppl');
  const src = fs.readFileSync(file, 'utf8');
  assert.equal(errors(src).length, 0);
});
